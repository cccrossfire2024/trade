# -*- coding: utf-8 -*-
"""
train_gate_lgbm_wfv.py (FIXED)

Walk-Forward (dev 区)：
- train: 9 months
- embargo: 2 days
- test: 1 month
- step: 1 month

目标：
- 二分类：y = 1[label != NONE]
- 生成严格 OOS 的 p_hat 与 gate_pos0_hat（0..1）

修复点：
- 事件总数只有 ~5k，单窗训练集远小于 5000，原脚本全部跳过
- 新增/降低最小样本阈值：min_train_rows=800, min_test_rows=80（可改）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz=ts.tz)


def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=n)


def build_windows(start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                  train_months=9, test_months=1,
                  embargo_days=2, step_months=1):
    """
    以月为锚点滚动：
      train: [t_train_start, t_train_end)
      test_start = t_train_end + embargo_days
      test_end   = 月边界推进 test_months
    """
    windows = []

    # 用 start_ts 所在月的月初作为基准
    base = month_start(start_ts)

    # 第一个训练结束锚点：base + train_months
    anchor = add_months(base, train_months)

    while True:
        t_train_end = anchor
        t_train_start = add_months(t_train_end, -train_months)

        t_test_start = t_train_end + pd.Timedelta(days=embargo_days)
        t_test_end = add_months(month_start(t_test_start), test_months)

        if t_test_end > end_ts:
            break

        if t_train_start < start_ts:
            anchor = add_months(anchor, step_months)
            continue

        windows.append((t_train_start, t_train_end, t_test_start, t_test_end))
        anchor = add_months(anchor, step_months)

    return windows


def pick_feature_cols(df: pd.DataFrame):
    exclude = {
        "symbol", "event_id", "twarn_ts", "t0_ts", "t1_ts",
        "label", "mfe_atr", "y"
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def pos_from_p(p: np.ndarray, p_enter=0.55):
    p = np.asarray(p, dtype=float)
    pos = (p - p_enter) / (1.0 - p_enter)
    return np.clip(pos, 0.0, 1.0)


def auc_mw(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Mann–Whitney U 等价 AUC，避免 sklearn 依赖
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    ranks = pd.Series(y_score).rank().to_numpy()
    r_pos = ranks[y_true == 1].sum()
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n0 + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="datasets_v2/gate_samples_v1.parquet")
    ap.add_argument("--out_dir", type=str, default="models_gate_v1")

    ap.add_argument("--train_months", type=int, default=9)
    ap.add_argument("--test_months", type=int, default=1)
    ap.add_argument("--embargo_days", type=int, default=2)
    ap.add_argument("--step_months", type=int, default=1)

    ap.add_argument("--p_enter", type=float, default=0.55)

    # ✅ 修复：降低最小样本门槛（你总事件 5244，不可能每窗 train>=5000）
    ap.add_argument("--min_train_rows", type=int, default=800)
    ap.add_argument("--min_test_rows", type=int, default=80)

    # LGB params（先用稳健默认）
    ap.add_argument("--num_leaves", type=int, default=64)
    ap.add_argument("--min_data_in_leaf", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--n_estimators", type=int, default=4000)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.7)
    ap.add_argument("--reg_lambda", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.in_path)
    df["t0_ts"] = pd.to_datetime(df["t0_ts"], utc=True)
    df = df.sort_values("t0_ts").reset_index(drop=True)

    # target
    df["y"] = (df["label"].astype(str) != "NONE").astype(int)

    feat_cols = pick_feature_cols(df)
    if len(feat_cols) < 10:
        raise ValueError(f"Too few feature cols: {len(feat_cols)}")

    start_ts = df["t0_ts"].min()
    end_ts = df["t0_ts"].max()
    windows = build_windows(
        start_ts, end_ts,
        train_months=args.train_months,
        test_months=args.test_months,
        embargo_days=args.embargo_days,
        step_months=args.step_months
    )
    if not windows:
        raise ValueError("No windows generated. Check your t0_ts range.")

    print(f"[INFO] total_events={len(df)} start={start_ts} end={end_ts} windows={len(windows)}")

    oos_parts = []
    window_summaries = []

    for wi, (t_tr0, t_tr1, t_te0, t_te1) in enumerate(windows):
        tr = df[(df["t0_ts"] >= t_tr0) & (df["t0_ts"] < t_tr1)].copy()
        te = df[(df["t0_ts"] >= t_te0) & (df["t0_ts"] < t_te1)].copy()

        if len(tr) < args.min_train_rows or len(te) < args.min_test_rows:
            window_summaries.append({
                "window": wi,
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "train_start": str(t_tr0),
                "train_end": str(t_tr1),
                "test_start": str(t_te0),
                "test_end": str(t_te1),
                "skipped": True,
                "reason": "too_few_rows"
            })
            continue

        X_tr = tr[feat_cols]
        y_tr = tr["y"].astype(int)
        X_te = te[feat_cols]
        y_te = te["y"].astype(int)

        # class imbalance handling
        pos = int(y_tr.sum())
        neg = int(len(y_tr) - pos)
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0

        clf = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=-1,
            scale_pos_weight=spw
        )

        # time-consistent validation split inside train (last 15% as valid)
        split = int(len(tr) * 0.85)
        # 防止 valid 太小
        if len(tr) - split < 200:
            split = max(int(len(tr) * 0.8), len(tr) - 200)

        X_fit, y_fit = X_tr.iloc[:split], y_tr.iloc[:split]
        X_val, y_val = X_tr.iloc[split:], y_tr.iloc[split:]

        clf.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )

        p_hat = clf.predict_proba(X_te)[:, 1]
        pos_hat = pos_from_p(p_hat, p_enter=args.p_enter)

        out = te[["symbol", "event_id", "t0_ts", "t1_ts", "label", "mfe_atr", "y"]].copy()
        out["p_hat"] = p_hat
        out["gate_pos0_hat"] = pos_hat
        out["window_id"] = wi
        out["train_start"] = t_tr0
        out["train_end"] = t_tr1
        out["test_start"] = t_te0
        out["test_end"] = t_te1
        oos_parts.append(out)

        auc = auc_mw(y_te.to_numpy(), p_hat)

        model_path = out_dir / "models" / f"lgbm_gate_window_{wi:03d}.txt"
        clf.booster_.save_model(str(model_path))

        imp = pd.DataFrame({
            "feature": feat_cols,
            "gain": clf.booster_.feature_importance(importance_type="gain"),
            "split": clf.booster_.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        imp_path = out_dir / "models" / f"importance_window_{wi:03d}.csv"
        imp.to_csv(imp_path, index=False)

        window_summaries.append({
            "window": wi,
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "train_start": str(t_tr0),
            "train_end": str(t_tr1),
            "test_start": str(t_te0),
            "test_end": str(t_te1),
            "auc": auc,
            "best_iteration": int(getattr(clf, "best_iteration_", 0) or 0),
            "scale_pos_weight": float(spw),
            "model_path": str(model_path),
            "importance_path": str(imp_path),
            "skipped": False
        })

        print(f"[WIN {wi:03d}] train={len(tr)} test={len(te)} pos_rate={pos/len(tr):.3f} auc={auc:.4f} best_iter={window_summaries[-1]['best_iteration']}")

    if not oos_parts:
        # 额外输出一个可诊断的窗口统计
        dbg_path = out_dir / "debug_windows.json"
        dbg_path.write_text(json.dumps(window_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ValueError("No OOS predictions produced (all windows skipped). "
                         f"Check min_train_rows/min_test_rows. Debug saved: {dbg_path}")

    oos = pd.concat(oos_parts, ignore_index=True)
    oos_path = out_dir / "gate_oos_dev.parquet"
    oos.to_parquet(oos_path, index=False)

    auc_all = auc_mw(oos["y"].to_numpy(), oos["p_hat"].to_numpy())

    summary = {
        "in_path": args.in_path,
        "out_dir": str(out_dir),
        "oos_path": str(oos_path),
        "n_total_events": int(len(df)),
        "n_oos_events": int(len(oos)),
        "coverage": float(len(oos) / max(len(df), 1)),
        "wf_params": {
            "train_months": args.train_months,
            "test_months": args.test_months,
            "embargo_days": args.embargo_days,
            "step_months": args.step_months
        },
        "min_rows": {"min_train_rows": args.min_train_rows, "min_test_rows": args.min_test_rows},
        "target": "y = 1[label != NONE]",
        "p_enter": args.p_enter,
        "auc_all_oos": auc_all,
        "n_features": int(len(feat_cols)),
        "feature_cols_preview": feat_cols[:25],
        "windows": window_summaries
    }

    (out_dir / "summary_gate_wfv.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("[DONE]", json.dumps({
        "oos_path": str(oos_path),
        "coverage": summary["coverage"],
        "auc_all_oos": summary["auc_all_oos"],
        "n_windows": len(windows)
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
