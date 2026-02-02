# -*- coding: utf-8 -*-
"""
train_gate_lgbm_wfv_v2.py

读取 datasets_v2plus/gate_samples_v2.parquet
对 Gate 做二分类：y = 1 if label != 'NONE' else 0
Walk-forward：按 t0_ts 时间滚动训练/测试，带 embargo（默认2天）
输出：
- out_dir/gate_oos_dev.parquet：包含每个 event 的 OOS 概率预测
- out_dir/summary_gate_train_v2.json：训练统计
- out_dir/models/lgbm_gate_winXXX.txt：每个窗口模型（可选保存）

用法：
python train_gate_lgbm_wfv_v2.py --in_path datasets_v2plus\\gate_samples_v2.parquet --out_dir models_gate_v2
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import roc_auc_score


EPS = 1e-12


def to_utc_ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def pick_feature_cols(df: pd.DataFrame):
    # 排除标识列、标签列、时间列
    drop = {
        "symbol", "event_id", "t0_ts", "t1_ts", "twarn_ts",
        "label", "mfe_atr"
    }
    cols = [c for c in df.columns if c not in drop]
    # 只保留数值列
    num_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def build_windows(times: pd.Series,
                  train_days: int = 540,
                  test_days: int = 30,
                  step_days: int = 30,
                  embargo_days: int = 2):
    """
    生成滚动窗口（时间切分），返回 list of (train_start, train_end, test_start, test_end)
    注意：embargo 会从 train_end 往后空出 embargo_days，再开始 test
    """
    tmin = times.min()
    tmax = times.max()
    if pd.isna(tmin) or pd.isna(tmax):
        return []

    windows = []
    cur_train_end = tmin + pd.Timedelta(days=train_days)

    # 让第一段 test 能落在数据范围内
    while True:
        train_start = cur_train_end - pd.Timedelta(days=train_days)
        train_end = cur_train_end

        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > tmax:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cur_train_end = cur_train_end + pd.Timedelta(days=step_days)

        if cur_train_end > tmax:
            break

    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--train_days", type=int, default=540)   # 18个月
    ap.add_argument("--test_days", type=int, default=30)     # 1个月
    ap.add_argument("--step_days", type=int, default=30)     # 每月滚动
    ap.add_argument("--embargo_days", type=int, default=2)   # 2天禁区

    ap.add_argument("--min_train", type=int, default=400)    # 最少训练样本
    ap.add_argument("--min_test", type=int, default=50)      # 最少测试样本

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_boost_round", type=int, default=5000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)

    ap.add_argument("--min_data_in_leaf", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--feature_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_freq", type=int, default=1)

    ap.add_argument("--save_models", action="store_true", help="save each window model to out_dir/models/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.in_path)
    if "t0_ts" not in df.columns:
        raise ValueError("gate samples missing t0_ts")
    if "label" not in df.columns:
        raise ValueError("gate samples missing label")

    df = df.copy()
    df["t0_ts"] = to_utc_ts(df["t0_ts"])
    df = df.dropna(subset=["t0_ts"]).sort_values("t0_ts").reset_index(drop=True)

    # 二分类：label != NONE
    y = (df["label"].astype(str) != "NONE").astype(int).values

    feat_cols = pick_feature_cols(df)
    if len(feat_cols) < 5:
        raise ValueError(f"too few numeric feature cols: {len(feat_cols)}")

    X = df[feat_cols]

    # LightGBM 允许 NaN，保持即可
    times = df["t0_ts"]
    windows = build_windows(times,
                            train_days=args.train_days,
                            test_days=args.test_days,
                            step_days=args.step_days,
                            embargo_days=args.embargo_days)
    if not windows:
        raise ValueError("No windows produced. Check date coverage or window params.")

    oos_pred = np.full(len(df), np.nan, dtype=float)
    win_stats = []

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "seed": args.seed,
        "verbosity": -1,
        "force_col_wise": True,
    }

    win_id = 0
    for (tr_s, tr_e, te_s, te_e) in windows:
        tr_mask = (times >= tr_s) & (times < tr_e)
        te_mask = (times >= te_s) & (times < te_e)

        n_tr = int(tr_mask.sum())
        n_te = int(te_mask.sum())
        if n_tr < args.min_train or n_te < args.min_test:
            win_stats.append({
                "win": win_id, "skipped": True,
                "train_n": n_tr, "test_n": n_te,
                "train_start": str(tr_s), "train_end": str(tr_e),
                "test_start": str(te_s), "test_end": str(te_e),
            })
            win_id += 1
            continue

        X_tr = X.loc[tr_mask]
        y_tr = y[tr_mask]
        X_te = X.loc[te_mask]
        y_te = y[te_mask]

        # class imbalance：用 scale_pos_weight 让分裂更稳
        pos = float(y_tr.sum())
        neg = float(len(y_tr) - y_tr.sum())
        spw = (neg / (pos + EPS)) if pos > 0 else 1.0
        params_win = dict(params)
        params_win["scale_pos_weight"] = spw

        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=True)
        dte = lgb.Dataset(X_te, label=y_te, feature_name=feat_cols, free_raw_data=True)

        booster = lgb.train(
            params_win,
            dtr,
            num_boost_round=args.num_boost_round,
            valid_sets=[dte],
            valid_names=["test"],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )

        pred = booster.predict(X_te, num_iteration=booster.best_iteration)
        oos_pred[te_mask.values] = pred

        auc = roc_auc_score(y_te, pred) if len(np.unique(y_te)) > 1 else np.nan
        pos_rate = float(y_te.mean()) if len(y_te) > 0 else np.nan

        win_stats.append({
            "win": win_id,
            "skipped": False,
            "train_n": n_tr,
            "test_n": n_te,
            "pos_rate": pos_rate,
            "auc": float(auc) if pd.notna(auc) else None,
            "best_iter": int(booster.best_iteration),
            "scale_pos_weight": float(spw),
            "train_start": str(tr_s), "train_end": str(tr_e),
            "test_start": str(te_s), "test_end": str(te_e),
        })

        print(f"[WIN {win_id:03d}] train={n_tr} test={n_te} pos_rate={pos_rate:.3f} auc={auc:.4f} best_iter={booster.best_iteration}")

        if args.save_models:
            model_path = out_dir / "models" / f"lgbm_gate_win{win_id:03d}.txt"
            booster.save_model(str(model_path))

        win_id += 1

    # 输出 OOS
    out = df[["symbol", "event_id", "t0_ts", "t1_ts", "twarn_ts", "label", "mfe_atr"]].copy()
    out["y"] = y.astype(int)
    out["gate_prob_oos"] = oos_pred

    # 覆盖率（有预测的占比）
    coverage = float(np.isfinite(out["gate_prob_oos"]).mean())

    # 全部 OOS AUC
    mask = np.isfinite(out["gate_prob_oos"].values)
    if mask.sum() == 0:
        raise ValueError("No OOS predictions produced (all windows skipped).")
    auc_all = roc_auc_score(out.loc[mask, "y"].values, out.loc[mask, "gate_prob_oos"].values) if len(np.unique(out.loc[mask, "y"])) > 1 else np.nan

    oos_path = out_dir / "gate_oos_dev.parquet"
    out.to_parquet(oos_path, index=False)

    summary = {
        "oos_path": str(oos_path),
        "in_path": args.in_path,
        "n_rows": int(len(df)),
        "n_features": int(len(feat_cols)),
        "coverage": coverage,
        "auc_all_oos": float(auc_all) if pd.notna(auc_all) else None,
        "n_windows": int(len(win_stats)),
        "windows": win_stats,
        "params_base": params,
        "feature_cols": feat_cols,
    }
    (out_dir / "summary_gate_train_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE]", json.dumps({
        "oos_path": str(oos_path),
        "coverage": coverage,
        "auc_all_oos": float(auc_all) if pd.notna(auc_all) else None,
        "n_windows": int(sum(1 for w in win_stats if not w.get("skipped"))),
        "n_features": int(len(feat_cols)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
