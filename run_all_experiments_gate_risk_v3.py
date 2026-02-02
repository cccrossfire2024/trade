# -*- coding: utf-8 -*-
"""
run_all_experiments_gate_risk_v3.py

一键实验总控（v3 修复版）：
- 训练 Gate（WFV）：bin / reg / multi
- 保存每个窗口 (train_start, train_end, test_start, test_end) 到 windows.json
- blind 推断：按 event 的 t0_ts 落在哪个 test 区间选择对应 win 模型（严格对应）
- 输出 dev/blind gate_oos（v1兼容：p_hat, gate_pos0_hat）
- 调用执行器 build_position_curve_gate_oos_only_v1logic.py 回测
- 汇总 compare_table.csv / compare_summary.json

关键修复：
1) reg：回归输出 yhat=mfe_atr_hat，用映射A：pos0=clip((yhat-1)/5,0,1)
2) multi：score=P(BIG)+0.6P(MID)+0.3P(SMALL)，用 dev_oos score 的 q20/q80 定标：
         pos0=clip((score-q20)/(q80-q20),0,1)
3) bin：p_hat -> (thr_low/thr_high) -> 0/0.5/1
4) 修复 blind_oos_path 未定义、multi_scale None 的报错

依赖：pandas, numpy, lightgbm, sklearn
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

EPS = 1e-12


# ---------------------------
# Utils
# ---------------------------

def to_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_feature_cols(df: pd.DataFrame) -> List[str]:
    drop = {
        "symbol","event_id","t0_ts","t1_ts","twarn_ts",
        "label","mfe_atr",
        "mfe","mae","mfe_pct","mae_pct"
    }
    cols = [c for c in df.columns if c not in drop]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def gate_pos_from_bin_p(p: np.ndarray, thr_low=0.50, thr_high=0.65):
    return np.where(p >= thr_high, 1.0, np.where(p >= thr_low, 0.5, 0.0)).astype("float32")

def gate_pos_from_reg_yhat(yhat: np.ndarray):
    # 映射A：pos0 = clip((ŷ-1)/5, 0, 1)
    pos0 = (np.asarray(yhat, dtype=float) - 1.0) / 5.0
    pos0 = np.clip(pos0, 0.0, 1.0)
    return pos0.astype("float32")

def multi_score_from_proba(pred4: np.ndarray):
    pred4 = np.asarray(pred4, dtype=float)
    # classes: [NONE, SMALL, MID, BIG]
    return (pred4[:,3] + 0.6*pred4[:,2] + 0.3*pred4[:,1]).astype(float)

def gate_pos_from_multi_score(score: np.ndarray, q20: float, q80: float):
    score = np.asarray(score, dtype=float)
    den = float(q80 - q20)
    if not np.isfinite(den) or abs(den) < 1e-9:
        den = 1.0
    pos0 = (score - float(q20)) / den
    pos0 = np.clip(pos0, 0.0, 1.0)
    return pos0.astype("float32")


# ---------------------------
# Walk-forward splitter
# ---------------------------

@dataclass
class WFVConfig:
    train_days: int = 540
    test_days: int = 30
    embargo_days: int = 2
    step_days: int = 30

def make_windows(t0: pd.Series, cfg: WFVConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    windows:
      (train_start, train_end, test_start, test_end)  UTC
    """
    t0 = pd.to_datetime(t0, utc=True)
    start = t0.min().floor("D")
    end = t0.max().ceil("D")

    windows = []
    cur_test_start = start + pd.Timedelta(days=cfg.train_days + cfg.embargo_days)
    while True:
        test_start = cur_test_start
        test_end = test_start + pd.Timedelta(days=cfg.test_days)
        train_end = test_start - pd.Timedelta(days=cfg.embargo_days)
        train_start = train_end - pd.Timedelta(days=cfg.train_days)

        if test_start > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cur_test_start = cur_test_start + pd.Timedelta(days=cfg.step_days)

    return windows


# ---------------------------
# Targets
# ---------------------------

def make_targets(df: pd.DataFrame, mode: str):
    if mode == "bin":
        if "label" not in df.columns:
            raise ValueError("bin mode requires 'label'")
        return (df["label"].astype(str) != "NONE").astype(int).values
    if mode == "reg":
        if "mfe_atr" not in df.columns:
            raise ValueError("reg mode requires 'mfe_atr'")
        y = pd.to_numeric(df["mfe_atr"], errors="coerce").fillna(0.0).astype(float).values
        return np.clip(y, 0.0, 10.0)
    if mode == "multi":
        if "label" not in df.columns:
            raise ValueError("multi mode requires 'label'")
        mp = {"NONE":0, "SMALL":1, "MID":2, "BIG":3}
        return df["label"].astype(str).map(mp).fillna(0).astype(int).values
    raise ValueError(f"unknown mode: {mode}")


# ---------------------------
# LightGBM params
# ---------------------------

def lgb_params(mode: str):
    base = dict(
        learning_rate=0.03,
        num_leaves=64,
        min_data_in_leaf=80,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        max_depth=-1
    )
    if mode == "bin":
        base.update(dict(objective="binary", metric="auc"))
    elif mode == "reg":
        base.update(dict(objective="huber", metric="l2"))
    elif mode == "multi":
        base.update(dict(objective="multiclass", num_class=4, metric="multi_logloss"))
    return base


# ---------------------------
# Train WFV
# ---------------------------

def train_wfv(dev: pd.DataFrame, feats: List[str], mode: str, out_dir: Path, cfg: WFVConfig,
              thr_low: float, thr_high: float,
              multi_q20: float, multi_q80: float):
    ensure_dir(out_dir)
    models_dir = out_dir / "models"
    ensure_dir(models_dir)

    dev = dev.copy()
    dev["t0_ts"] = to_utc(dev["t0_ts"])
    dev = dev.dropna(subset=["t0_ts"]).sort_values("t0_ts").reset_index(drop=True)

    X = dev[feats]
    y = make_targets(dev, mode)
    windows = make_windows(dev["t0_ts"], cfg)

    # windows meta
    win_meta = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        win_meta.append({
            "win": wi,
            "train_start": str(tr_s),
            "train_end": str(tr_e),
            "test_start": str(te_s),
            "test_end": str(te_e),
        })
    (out_dir / "windows.json").write_text(json.dumps(win_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    oos_pred = np.full(len(dev), np.nan, dtype=float)
    oos_win = np.full(len(dev), -1, dtype=int)

    params = lgb_params(mode)

    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        tr_mask = (dev["t0_ts"] >= tr_s) & (dev["t0_ts"] < tr_e)
        te_mask = (dev["t0_ts"] >= te_s) & (dev["t0_ts"] < te_e)

        n_tr = int(tr_mask.sum())
        n_te = int(te_mask.sum())
        if n_tr < 300 or n_te < 30:
            continue

        X_tr, y_tr = X.loc[tr_mask], y[tr_mask.values]
        X_te, y_te = X.loc[te_mask], y[te_mask.values]

        dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        dte = lgb.Dataset(X_te, label=y_te, free_raw_data=False)

        booster = lgb.train(
            params,
            dtr,
            num_boost_round=5000,
            valid_sets=[dte],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )

        model_path = models_dir / f"lgbm_gate_win{wi:03d}.txt"
        booster.save_model(str(model_path))

        pred = booster.predict(X_te, num_iteration=booster.best_iteration)

        if mode == "multi":
            pred1 = multi_score_from_proba(np.asarray(pred))
        else:
            pred1 = np.asarray(pred).reshape(-1)

        oos_pred[te_mask.values] = pred1
        oos_win[te_mask.values] = wi

        if mode == "bin":
            try:
                auc = float(roc_auc_score(y_te, pred1))
            except Exception:
                auc = float("nan")
            pos_rate = float(np.mean(y_te))
            print(f"[WIN {wi:03d}] train={n_tr} test={n_te} pos_rate={pos_rate:.3f} auc={auc:.4f} best_iter={booster.best_iteration}")
        else:
            print(f"[WIN {wi:03d}] train={n_tr} test={n_te} best_iter={booster.best_iteration}")

    # dev gate_oos (v1 compatible)
    out = dev[["symbol","event_id","t0_ts"]].copy()
    out["p_hat"] = pd.Series(oos_pred).fillna(0.0).astype(float).values

    if mode == "bin":
        out["gate_pos0_hat"] = gate_pos_from_bin_p(out["p_hat"].values, thr_low, thr_high)
        ms = None
        rs = None
    elif mode == "reg":
        out["gate_pos0_hat"] = gate_pos_from_reg_yhat(out["p_hat"].values)
        ms = None
        rs = {"pos0_map": "clip((yhat-1)/5,0,1)"}
    else:
        out["gate_pos0_hat"] = gate_pos_from_multi_score(out["p_hat"].values, float(multi_q20), float(multi_q80))
        ms = {"q20": float(multi_q20), "q80": float(multi_q80)}
        rs = None

    oos_path = out_dir / "gate_oos_dev_for_v1.parquet"
    out.to_parquet(oos_path, index=False)

    coverage = float(np.mean(~np.isnan(oos_pred)))
    res = {
        "mode": mode,
        "oos_path": str(oos_path),
        "coverage": coverage,
        "n_windows_total": len(windows),
        "n_windows_with_pred": int(np.max(oos_win) + 1) if np.max(oos_win) >= 0 else 0,
        "n_features": len(feats),
        "models_dir": str(models_dir),
        "bin_thresholds": {"thr_low": thr_low, "thr_high": thr_high} if mode == "bin" else None,
        "multi_scale": ms,
        "reg_scale": rs
    }
    (out_dir / "summary_train.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE TRAIN]", res)
    return res


# ---------------------------
# Blind inference strict by windows
# ---------------------------

def load_windows_meta(windows_json: Path):
    meta = json.loads(windows_json.read_text(encoding="utf-8"))
    for w in meta:
        w["test_start"] = to_utc(w["test_start"])
        w["test_end"] = to_utc(w["test_end"])
        w["win"] = int(w["win"])
    return meta

def infer_blind_strict(blind: pd.DataFrame, feats: List[str], mode: str,
                       models_dir: Path, windows_json: Path,
                       out_path: Path,
                       thr_low: float, thr_high: float,
                       multi_q20: float, multi_q80: float):
    blind = blind.copy()
    blind["t0_ts"] = to_utc(blind["t0_ts"])
    blind = blind.dropna(subset=["t0_ts"]).sort_values("t0_ts").reset_index(drop=True)

    X = blind[feats]
    win_meta = load_windows_meta(windows_json)

    win_id = np.full(len(blind), -1, dtype=int)
    for w in win_meta:
        m = (blind["t0_ts"] >= w["test_start"]) & (blind["t0_ts"] < w["test_end"])
        win_id[m.values] = w["win"]

    # 对覆盖之外的：向前填充最近窗口；开头仍无则用第一个窗口
    last = -1
    for i in range(len(win_id)):
        if win_id[i] >= 0:
            last = win_id[i]
        else:
            win_id[i] = last
    if win_id[0] < 0:
        win_id[:] = win_meta[0]["win"]

    p_hat = np.zeros(len(blind), dtype=float)
    unique_wins = sorted(set(win_id.tolist()))
    for w in unique_wins:
        idx = np.where(win_id == w)[0]
        if len(idx) == 0:
            continue

        mf = models_dir / f"lgbm_gate_win{w:03d}.txt"
        if not mf.exists():
            # 向下回退到最近存在的模型
            ww = w
            while ww >= 0 and not (models_dir / f"lgbm_gate_win{ww:03d}.txt").exists():
                ww -= 1
            if ww < 0:
                raise FileNotFoundError(f"no model found for win {w} in {models_dir}")
            mf = models_dir / f"lgbm_gate_win{ww:03d}.txt"

        booster = lgb.Booster(model_file=str(mf))
        pred = booster.predict(X.iloc[idx], num_iteration=booster.best_iteration)

        if mode == "multi":
            pred1 = multi_score_from_proba(np.asarray(pred))
        else:
            pred1 = np.asarray(pred).reshape(-1)

        p_hat[idx] = pred1

    out = blind[["symbol","event_id","t0_ts"]].copy()
    out["p_hat"] = p_hat.astype(float)

    if mode == "bin":
        out["gate_pos0_hat"] = gate_pos_from_bin_p(out["p_hat"].values, thr_low, thr_high)
    elif mode == "reg":
        out["gate_pos0_hat"] = gate_pos_from_reg_yhat(out["p_hat"].values)
    else:
        out["gate_pos0_hat"] = gate_pos_from_multi_score(out["p_hat"].values, float(multi_q20), float(multi_q80))

    ensure_dir(out_path.parent)
    out.to_parquet(out_path, index=False)
    print("[DONE INFER] saved:", out_path, "rows:", len(out))
    return str(out_path)


# ---------------------------
# Run executor
# ---------------------------

def run_executor(executor_py: Path, out_dir: Path,
                 dev_gate_oos: Path, blind_gate_oos: Path,
                 dev_universe_long: Path, blind_universe_long: Path,
                 dev_risk_states: Path, blind_risk_states: Path):
    ensure_dir(out_dir)
    cmd = (
        f'python "{executor_py}" '
        f'--out_dir "{out_dir}" '
        f'--dev_gate_oos "{dev_gate_oos}" '
        f'--blind_gate_oos "{blind_gate_oos}" '
        f'--dev_universe_long "{dev_universe_long}" '
        f'--blind_universe_long "{blind_universe_long}" '
        f'--dev_risk_states "{dev_risk_states}" '
        f'--blind_risk_states "{blind_risk_states}" '
    )
    print("[EXEC]", cmd)
    r = os.system(cmd)
    if r != 0:
        raise RuntimeError(f"executor failed with code {r}")
    summary_path = out_dir / "summary_position_curve.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return summary_path


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dev_gate_samples", type=str, required=True)
    ap.add_argument("--blind_gate_samples", type=str, required=True)

    ap.add_argument("--dev_universe_long", type=str, default="data_clean/universe_long_dev.parquet")
    ap.add_argument("--blind_universe_long", type=str, default="data_clean/universe_long_blind.parquet")
    ap.add_argument("--dev_risk_states", type=str, default="datasets_v3/risk_states_v1.parquet")
    ap.add_argument("--blind_risk_states", type=str, default="datasets_blind_v1/step3_risk_states/risk_states_v1.parquet")

    ap.add_argument("--executor_py", type=str, default="build_position_curve_gate_oos_only_v1logic.py")

    ap.add_argument("--out_root", type=str, default="experiments_gate_v3_suite")
    ap.add_argument("--modes", nargs="+", default=["bin","reg","multi"])

    ap.add_argument("--train_days", type=int, default=540)
    ap.add_argument("--test_days", type=int, default=30)
    ap.add_argument("--embargo_days", type=int, default=2)
    ap.add_argument("--step_days", type=int, default=30)

    ap.add_argument("--thr_low", type=float, default=0.50)
    ap.add_argument("--thr_high", type=float, default=0.65)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    dev = pd.read_parquet(args.dev_gate_samples)
    blind = pd.read_parquet(args.blind_gate_samples)

    if "t0_ts" not in dev.columns or "t0_ts" not in blind.columns:
        raise ValueError("gate_samples must contain column: t0_ts")

    dev["t0_ts"] = to_utc(dev["t0_ts"])
    blind["t0_ts"] = to_utc(blind["t0_ts"])

    feats_all = pick_feature_cols(dev)
    print("[INFO] features:", len(feats_all))

    cfg = WFVConfig(args.train_days, args.test_days, args.embargo_days, args.step_days)

    rows = []
    summaries = {}

    for mode in list(args.modes):
        exp_dir = out_root / f"gate_{mode}"
        ensure_dir(exp_dir)

        # 注意：blind_oos_path 必须先定义（修复 NameError）
        blind_oos_path = exp_dir / "gate_oos_blind_for_v1.parquet"

        # multi 的 q20/q80：如果用户没给，先占位，训练后从 dev_oos score 估计并回写
        multi_q20, multi_q80 = 0.0, 1.0

        train_res = train_wfv(
            dev, feats_all, mode, exp_dir, cfg,
            thr_low=args.thr_low, thr_high=args.thr_high,
            multi_q20=multi_q20, multi_q80=multi_q80
        )

        # 若 mode==multi：用 dev_oos 的 score 估计 q20/q80，然后重写 dev_oos gate_pos0_hat
        if mode == "multi":
            dev_oos = pd.read_parquet(train_res["oos_path"])
            score = dev_oos["p_hat"].astype(float).values
            q20 = float(np.nanquantile(score, 0.20))
            q80 = float(np.nanquantile(score, 0.80))
            dev_oos["gate_pos0_hat"] = gate_pos_from_multi_score(score, q20, q80)
            dev_oos.to_parquet(train_res["oos_path"], index=False)

            train_res["multi_scale"] = {"q20": q20, "q80": q80}
            (exp_dir / "summary_train.json").write_text(json.dumps(train_res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[MULTI SCALE] q20={q20:.6f} q80={q80:.6f}")

        # 取 multi_scale（修复 None.get 报错）
        ms = train_res.get("multi_scale") or {}
        q20_use = float(ms.get("q20", 0.0))
        q80_use = float(ms.get("q80", 1.0))

        # blind 推断（严格按 windows）
        infer_blind_strict(
            blind, feats_all, mode,
            models_dir=Path(train_res["models_dir"]),
            windows_json=exp_dir / "windows.json",
            out_path=blind_oos_path,
            thr_low=args.thr_low, thr_high=args.thr_high,
            multi_q20=q20_use, multi_q80=q80_use
        )

        # 执行器回测
        exec_out = exp_dir / "exec"
        summary_path = run_executor(
            Path(args.executor_py),
            exec_out,
            Path(train_res["oos_path"]),
            Path(blind_oos_path),
            Path(args.dev_universe_long),
            Path(args.blind_universe_long),
            Path(args.dev_risk_states),
            Path(args.blind_risk_states),
        )

        summ = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        summaries[mode] = summ

        dev_over = summ["dev"]["overall_equal_weight"]
        blind_over = summ["blind"]["overall_equal_weight"]

        rows.append({
            "mode": mode,
            "dev_sharpe_like": dev_over["sharpe_like"],
            "dev_mdd": dev_over["max_drawdown"],
            "dev_total_return": dev_over["total_return"],
            "blind_sharpe_like": blind_over["sharpe_like"],
            "blind_mdd": blind_over["max_drawdown"],
            "blind_total_return": blind_over["total_return"],
            "coverage": train_res["coverage"],
            "n_features": train_res["n_features"],
            "models_dir": train_res["models_dir"],
            "dev_gate_oos": train_res["oos_path"],
            "blind_gate_oos": str(blind_oos_path),
            "exec_summary": str(summary_path),
            "notes": train_res.get("reg_scale") or train_res.get("multi_scale") or train_res.get("bin_thresholds")
        })

    compare = pd.DataFrame(rows).sort_values("blind_sharpe_like", ascending=False)
    compare_path = out_root / "compare_table.csv"
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")

    sum_path = out_root / "compare_summary.json"
    (out_root / "compare_summary.json").write_text(json.dumps({"experiments": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] saved:", compare_path)
    print("[DONE] saved:", sum_path)


if __name__ == "__main__":
    main()
