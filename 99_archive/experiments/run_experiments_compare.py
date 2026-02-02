# -*- coding: utf-8 -*-
"""
run_experiments_compare.py

一键跑多组实验（Gate x Risk），并生成 compare summary（dev+blind）

依赖：
- pandas, numpy, lightgbm

假设你已有这些文件（按你当前工程进度）：
DEV:
- data_clean/universe_long_dev.parquet
- datasets_v2/gate_samples_v1.parquet
- datasets_v2/risk_bars_features_v1.parquet
- datasets_v3/risk_states_v1.parquet  (可选；若不存在会自动从 risk_bars 生成 v1 简化版 or 直接跳过 R1-dev)

BLIND:
- data_clean/universe_long_blind.parquet
- datasets_blind_v1/step2_features/gate_samples_v1.parquet
- datasets_blind_v1/step2_features/risk_bars_features_v1.parquet
- datasets_blind_v1/step3_risk_states/risk_states_v1.parquet (可选；若不存在会自动用 risk_bars 生成 v1 简化版)

已训练二分类 Gate（可选，但建议有）：
- models_gate_v1/gate_oos_dev.parquet
- models_gate_v1/models/lgbm_gate_window_*.txt

输出：
- out_dir/compare_summary.json
- out_dir/compare_table.csv
- out_dir/<exp_name>/... 中间产物与 summary

运行：
  python run_experiments_compare.py --out_dir experiments_out
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb


# ----------------------------
# Metrics
# ----------------------------
def ann_factor_15m():
    return np.sqrt(96 * 365)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())

def calmar_like(equity: pd.Series) -> float:
    if len(equity) < 10:
        return float("nan")
    total_ret = float(equity.iloc[-1] / (equity.iloc[0] + 1e-12) - 1.0)
    years = len(equity) / (96 * 365)
    if years <= 0:
        return float("nan")
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-12:
        return float("nan")
    return float(cagr / mdd)

def sanity_backtest(curve: pd.DataFrame, fee: float):
    """long-only, pnl_t = pos_{t-1}*ret - fee*|dpos|"""
    curve = curve.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = curve.groupby("symbol", group_keys=False)
    curve["ret1"] = g["close"].pct_change().fillna(0.0)
    curve["pos_prev"] = g["pos"].shift(1).fillna(0.0)
    curve["dpos"] = (curve["pos"] - curve["pos_prev"]).abs()
    curve["pnl"] = curve["pos_prev"] * curve["ret1"] - fee * curve["dpos"]
    curve["equity"] = g["pnl"].cumsum() + 1.0

    per_symbol = {}
    for sym, sub in curve.groupby("symbol"):
        pnl = sub["pnl"]
        mu, sd = float(pnl.mean()), float(pnl.std())
        sharpe = (mu / (sd + 1e-12)) * ann_factor_15m()
        eq = sub["equity"]
        per_symbol[sym] = {
            "rows": int(len(sub)),
            "sharpe_like": sharpe,
            "max_drawdown": max_drawdown(eq),
            "calmar_like": calmar_like(eq),
            "turnover_per_bar": float(sub["dpos"].mean()),
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        }

    pivot = curve.pivot(index="ts", columns="symbol", values="pnl").fillna(0.0)
    pnl_all = pivot.mean(axis=1)
    eq_all = pnl_all.cumsum() + 1.0
    mu, sd = float(pnl_all.mean()), float(pnl_all.std())
    overall = {
        "sharpe_like": (mu / (sd + 1e-12)) * ann_factor_15m(),
        "max_drawdown": max_drawdown(eq_all),
        "calmar_like": calmar_like(eq_all),
        "total_return": float(eq_all.iloc[-1] / eq_all.iloc[0] - 1.0),
    }
    return curve, per_symbol, overall


# ----------------------------
# Gate mapping
# ----------------------------
def gate_oracle_from_label(gate_samples: pd.DataFrame, pos_map=None):
    if pos_map is None:
        pos_map = {"NONE": 0.0, "SMALL": 0.4, "MID": 0.7, "BIG": 1.0}
    out = gate_samples[["symbol", "event_id"]].copy()
    out["p_hat"] = np.nan
    out["gate_pos0_hat"] = gate_samples["label"].map(pos_map).astype(float).fillna(0.0)
    return out

def gate_pos_from_prob(p_hat: np.ndarray, p_enter=0.55):
    pos = (p_hat - p_enter) / (1.0 - p_enter)
    return np.clip(pos, 0.0, 1.0)

def gate_pos_from_mfe_hat(mfe_hat: np.ndarray):
    # 将 mfe_hat (ATR倍数) 映射到 pos0：>1 开始给仓位，>7 近满仓
    mfe_hat = np.asarray(mfe_hat, dtype=float)
    pos = (mfe_hat - 1.0) / 6.0
    return np.clip(pos, 0.0, 1.0)


# ----------------------------
# Risk state machine v2 (WARN/HARD + adaptive cooldown)
# ----------------------------
def rolling_q(s: pd.Series, win: int, q: float, minp: int):
    return s.rolling(win, min_periods=minp).quantile(q)

def safe_z(s: pd.Series, win: int, minp: int):
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - mu) / (sd + 1e-12)

def build_risk_states_v2(risk_bars_feat: pd.DataFrame,
                         win_short=24, win_mid=48, win_long=96,
                         pos_base=0.6, pos_add=0.25, pos_reduce=0.25,
                         cooldown_low=8, cooldown_mid=16, cooldown_high=32):
    """
    输出 columns: symbol,event_id,ts,state,target_pos
    state: ADVANCE, EXPANSION, DIGEST, EXHAUST, WARN_BREAK, HARD_BREAK
    """
    df = risk_bars_feat.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "event_id", "ts"]).reset_index(drop=True)
    g = df.groupby(["symbol", "event_id"], group_keys=False)

    minS = max(16, win_short // 3)
    minM = max(16, win_mid // 3)
    minL = max(16, win_long // 3)

    df["gap_chg_1"] = g["gmma_gap"].diff(1)
    df["gap_chg_z"] = g["gap_chg_1"].apply(lambda s: safe_z(s, win_short, minS))

    df["slopeS_q70"] = g["gmma_slopeS"].apply(lambda s: rolling_q(s, win_mid, 0.70, minM))
    df["slopeS_q30"] = g["gmma_slopeS"].apply(lambda s: rolling_q(s, win_mid, 0.30, minM))

    df["acc_q10"] = g["gmma_slopeS_acc"].apply(lambda s: rolling_q(s, win_mid, 0.10, minM))
    df["er_q70"] = g["er_16"].apply(lambda s: rolling_q(s, win_mid, 0.70, minM))
    df["er_q40"] = g["er_16"].apply(lambda s: rolling_q(s, win_mid, 0.40, minM))
    df["er_q20"] = g["er_16"].apply(lambda s: rolling_q(s, win_mid, 0.20, minM))

    df["rs_q70"] = g["rs_vol_rel"].apply(lambda s: rolling_q(s, win_mid, 0.70, minM))
    df["rs_q90"] = g["rs_vol_rel"].apply(lambda s: rolling_q(s, win_mid, 0.90, minM))
    df["vov_q70"] = g["vov_96"].apply(lambda s: rolling_q(s, win_long, 0.70, minL))
    df["vov_q90"] = g["vov_96"].apply(lambda s: rolling_q(s, win_long, 0.90, minL))
    df["lr16_q10"] = g["lr_16"].apply(lambda s: rolling_q(s, win_mid, 0.10, minM))
    df["lr16_q30"] = g["lr_16"].apply(lambda s: rolling_q(s, win_mid, 0.30, minM))

    df["ready"] = df[["slopeS_q70", "er_q70", "rs_q70", "vov_q70", "lr16_q30"]].notna().all(axis=1)

    # ---- state rules ----
    # HARD_BREAK：结构快速收敛 + 风险尖刺到极端 + 动能明显转负
    hard_break = (
        (df["gap_chg_z"] < -1.2) &
        ((df["vov_96"] > df["vov_q90"]) | (df["rs_vol_rel"] > df["rs_q90"])) &
        (df["lr_16"] < df["lr16_q10"])
    )

    # WARN_BREAK：结构收敛但没到极端（洗盘/回撤）
    warn_break = (
        (df["gap_chg_z"] < -0.8) &
        (~hard_break) &
        ((df["vov_96"] > df["vov_q70"]) | (df["rs_vol_rel"] > df["rs_q70"]) | (df["lr_16"] < df["lr16_q30"]))
    )

    exhaust = (
        (df["gmma_slopeS_acc"] < df["acc_q10"]) &
        (df["er_16"] < df["er_q20"]) &
        (df["lr_16"] < df["lr16_q30"])
    )

    advance = (
        (df["gmma_slopeS"] > df["slopeS_q70"]) &
        (df["er_16"] > df["er_q70"]) &
        (df["gap_chg_1"] > 0) &
        (df["gmma_slopeS"] > 0)
    )

    expansion = (
        (df["gmma_slopeS"] > df["slopeS_q30"]) &
        (df["er_16"] > df["er_q40"]) &
        (df["gmma_slopeS"] > 0) &
        ((df["rs_vol_rel"] > df["rs_q70"]) | (df["vov_96"] > df["vov_q70"]))
    )

    df["state"] = "DIGEST"
    df.loc[expansion, "state"] = "EXPANSION"
    df.loc[advance, "state"] = "ADVANCE"
    df.loc[exhaust, "state"] = "EXHAUST"
    df.loc[warn_break, "state"] = "WARN_BREAK"
    df.loc[hard_break, "state"] = "HARD_BREAK"

    # not-ready fallback（短事件也别全 DIGEST）
    not_ready = ~df["ready"]
    fallback_adv = (df["gmma_slopeS"] > 0) & (df["gap_chg_1"] > 0) & (df["er_16"] > 0.35)
    fallback_brk = (df["gap_chg_1"] < 0) & (df["lr_16"] < 0) & (df["rs_vol_rel"] > 1.0)
    df.loc[not_ready & fallback_adv, "state"] = "ADVANCE"
    df.loc[not_ready & fallback_brk, "state"] = "WARN_BREAK"

    # ---- target_pos mapping ----
    pos_map = {
        "ADVANCE":   min(1.0, pos_base + pos_add),
        "EXPANSION": min(1.0, pos_base + 0.5 * pos_add),
        "DIGEST":    pos_base,
        "EXHAUST":   max(0.0, pos_base - pos_reduce),
        "WARN_BREAK": max(0.0, pos_base - 0.15),  # 不清仓，只减仓
        "HARD_BREAK": 0.0
    }
    df["target_pos_raw"] = df["state"].map(pos_map).astype(float)

    # adaptive cooldown only for HARD_BREAK
    def apply_cd(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values("ts").copy()
        cd = 0
        out = []
        for _, r in sub.iterrows():
            if r["state"] == "HARD_BREAK":
                # 根据 vov 强度决定冷却长度
                if pd.notna(r.get("vov_q90")) and r["vov_96"] > r["vov_q90"]:
                    cd = cooldown_high
                elif pd.notna(r.get("vov_q70")) and r["vov_96"] > r["vov_q70"]:
                    cd = cooldown_mid
                else:
                    cd = cooldown_low
            if cd > 0:
                out.append(0.0)
                cd -= 1
            else:
                out.append(float(r["target_pos_raw"]))
        sub["target_pos"] = out
        return sub

    df = df.groupby(["symbol", "event_id"], group_keys=False).apply(apply_cd)
    out = df[["symbol", "event_id", "ts", "state", "target_pos"]].copy()
    return out


# ----------------------------
# Curve builder: Gate x Risk
# ----------------------------
def build_position_curve(universe_long_path: str,
                         risk_states_df: pd.DataFrame,
                         gate_event_df: pd.DataFrame,
                         fee: float,
                         risk_pos_base: float,
                         pos_cap: float):
    uni = pd.read_parquet(universe_long_path)
    uni["ts"] = pd.to_datetime(uni["ts"], utc=True)
    uni = uni.sort_values(["symbol", "ts"]).reset_index(drop=True)

    risk = risk_states_df.copy()
    risk["ts"] = pd.to_datetime(risk["ts"], utc=True)

    # risk multiplier
    risk["risk_mult"] = (risk["target_pos"] / (risk_pos_base + 1e-12)).clip(0.0, pos_cap/(risk_pos_base+1e-12))

    # join gate per event
    m = risk.merge(gate_event_df[["symbol", "event_id", "gate_pos0_hat", "p_hat"]],
                   on=["symbol", "event_id"], how="left")
    m["gate_pos0_hat"] = m["gate_pos0_hat"].fillna(0.0)  # 严谨：没 gate 就不进
    m["p_hat"] = m["p_hat"].fillna(0.0)
    m["pos"] = (m["gate_pos0_hat"] * m["risk_mult"]).clip(0.0, pos_cap)

    base = uni[["ts", "symbol", "close"]].copy()
    base = base.merge(m[["symbol", "ts", "event_id", "state", "target_pos", "risk_mult", "gate_pos0_hat", "p_hat", "pos"]],
                      on=["symbol", "ts"], how="left")
    base["pos"] = base["pos"].fillna(0.0)
    base["event_id"] = base["event_id"].fillna(-1).astype(int)
    base["state"] = base["state"].fillna("OUT")
    base["target_pos"] = base["target_pos"].fillna(0.0)
    base["risk_mult"] = base["risk_mult"].fillna(0.0)
    base["gate_pos0_hat"] = base["gate_pos0_hat"].fillna(0.0)
    base["p_hat"] = base["p_hat"].fillna(0.0)

    base, per_symbol, overall = sanity_backtest(base, fee=fee)
    return base, per_symbol, overall


# ----------------------------
# Gate training (reg mfe) WFV
# ----------------------------
def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz=ts.tz)

def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=n)

def build_windows(start_ts, end_ts, train_months=9, test_months=1, embargo_days=2, step_months=1):
    windows = []
    base = month_start(start_ts)
    anchor = add_months(base, train_months)
    while True:
        t_tr1 = anchor
        t_tr0 = add_months(t_tr1, -train_months)
        t_te0 = t_tr1 + pd.Timedelta(days=embargo_days)
        t_te1 = add_months(month_start(t_te0), test_months)
        if t_te1 > end_ts:
            break
        if t_tr0 < start_ts:
            anchor = add_months(anchor, step_months)
            continue
        windows.append((t_tr0, t_tr1, t_te0, t_te1))
        anchor = add_months(anchor, step_months)
    return windows

def pick_gate_feature_cols(df: pd.DataFrame):
    exclude = {"symbol", "event_id", "twarn_ts", "t0_ts", "t1_ts", "label", "mfe_atr", "y"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def train_gate_reg_wfv(gate_dev_samples_path: str, out_dir: Path,
                       train_months=9, test_months=1, embargo_days=2, step_months=1,
                       min_train=600, min_test=60,
                       mfe_clip=8.0, seed=42):
    """
    回归预测 mfe_atr（clip），输出 dev 严格 OOS 的 mfe_hat 与 gate_pos0_hat
    同时保存每个窗口的模型
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models_reg").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gate_dev_samples_path)
    df["t0_ts"] = pd.to_datetime(df["t0_ts"], utc=True)
    df = df.sort_values("t0_ts").reset_index(drop=True)

    feat_cols = pick_gate_feature_cols(df)
    y_all = df["mfe_atr"].astype(float).clip(lower=0.0, upper=mfe_clip)

    start_ts, end_ts = df["t0_ts"].min(), df["t0_ts"].max()
    wins = build_windows(start_ts, end_ts, train_months, test_months, embargo_days, step_months)
    oos_parts = []
    ws = []

    for wi, (t_tr0, t_tr1, t_te0, t_te1) in enumerate(wins):
        tr = df[(df["t0_ts"] >= t_tr0) & (df["t0_ts"] < t_tr1)].copy()
        te = df[(df["t0_ts"] >= t_te0) & (df["t0_ts"] < t_te1)].copy()
        if len(tr) < min_train or len(te) < min_test:
            ws.append({"window": wi, "skipped": True, "train_rows": int(len(tr)), "test_rows": int(len(te))})
            continue

        X_tr = tr[feat_cols]
        y_tr = tr["mfe_atr"].astype(float).clip(0.0, mfe_clip)
        X_te = te[feat_cols]
        y_te = te["mfe_atr"].astype(float).clip(0.0, mfe_clip)

        # time-consistent valid split
        split = int(len(tr) * 0.85)
        if len(tr) - split < 200:
            split = max(int(len(tr) * 0.8), len(tr) - 200)

        X_fit, y_fit = X_tr.iloc[:split], y_tr.iloc[:split]
        X_val, y_val = X_tr.iloc[split:], y_tr.iloc[split:]

        reg = lgb.LGBMRegressor(
            objective="regression_l1",  # 对尾部更稳
            num_leaves=64,
            min_data_in_leaf=100,
            learning_rate=0.03,
            n_estimators=5000,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1
        )

        reg.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )

        mfe_hat = reg.predict(X_te, num_iteration=getattr(reg, "best_iteration_", None))
        mfe_hat = np.clip(np.asarray(mfe_hat, dtype=float), 0.0, mfe_clip)
        pos0_hat = gate_pos_from_mfe_hat(mfe_hat)

        out = te[["symbol", "event_id", "t0_ts", "t1_ts", "label", "mfe_atr"]].copy()
        out["mfe_hat"] = mfe_hat
        out["p_hat"] = np.nan
        out["gate_pos0_hat"] = pos0_hat
        out["window_id"] = wi
        oos_parts.append(out)

        model_path = out_dir / "models_reg" / f"lgbm_gate_reg_window_{wi:03d}.txt"
        reg.booster_.save_model(str(model_path))
        ws.append({"window": wi, "skipped": False, "train_rows": int(len(tr)), "test_rows": int(len(te)),
                   "best_iter": int(getattr(reg, "best_iteration_", 0) or 0), "model_path": str(model_path)})

        print(f"[REG WIN {wi:03d}] train={len(tr)} test={len(te)} best_iter={ws[-1]['best_iter']}")

    if not oos_parts:
        raise ValueError("reg_wfv: no oos produced (all windows skipped)")

    oos = pd.concat(oos_parts, ignore_index=True)
    oos_path = out_dir / "gate_reg_oos_dev.parquet"
    oos.to_parquet(oos_path, index=False)

    summary = {"oos_path": str(oos_path), "windows": ws, "n_oos": int(len(oos)), "n_total": int(len(df))}
    (out_dir / "summary_gate_reg_wfv.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # last model file
    model_files = sorted((out_dir / "models_reg").glob("lgbm_gate_reg_window_*.txt"))
    last_model = model_files[-1] if model_files else None
    return oos_path, last_model, feat_cols


def infer_gate_bin_blind(gate_blind_samples: pd.DataFrame, models_dir: Path, p_enter=0.55):
    model_files = sorted(models_dir.glob("lgbm_gate_window_*.txt"))
    if not model_files:
        raise FileNotFoundError(f"No binary gate models in {models_dir}")
    booster = lgb.Booster(model_file=str(model_files[-1]))
    feat_cols = pick_gate_feature_cols(gate_blind_samples)
    p_hat = booster.predict(gate_blind_samples[feat_cols], num_iteration=booster.best_iteration)
    p_hat = np.asarray(p_hat, dtype=float)
    out = gate_blind_samples[["symbol", "event_id"]].copy()
    out["p_hat"] = p_hat
    out["gate_pos0_hat"] = gate_pos_from_prob(p_hat, p_enter=p_enter)
    return out, model_files[-1].name

def infer_gate_reg_blind(gate_blind_samples: pd.DataFrame, model_path: Path, mfe_clip=8.0):
    booster = lgb.Booster(model_file=str(model_path))
    feat_cols = pick_gate_feature_cols(gate_blind_samples)
    mfe_hat = booster.predict(gate_blind_samples[feat_cols], num_iteration=booster.best_iteration)
    mfe_hat = np.clip(np.asarray(mfe_hat, dtype=float), 0.0, mfe_clip)
    out = gate_blind_samples[["symbol", "event_id"]].copy()
    out["p_hat"] = np.nan
    out["gate_pos0_hat"] = gate_pos_from_mfe_hat(mfe_hat)
    return out, model_path.name


# ----------------------------
# Main Orchestrator
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="experiments_out")

    # paths
    ap.add_argument("--dev_universe", type=str, default="data_clean/universe_long_dev.parquet")
    ap.add_argument("--dev_gate_samples", type=str, default="datasets_v2/gate_samples_v1.parquet")
    ap.add_argument("--dev_risk_bars", type=str, default="datasets_v2/risk_bars_features_v1.parquet")
    ap.add_argument("--dev_risk_states_v1", type=str, default="datasets_v3/risk_states_v1.parquet")
    ap.add_argument("--dev_gate_oos_bin", type=str, default="models_gate_v1/gate_oos_dev.parquet")
    ap.add_argument("--bin_models_dir", type=str, default="models_gate_v1/models")

    ap.add_argument("--blind_universe", type=str, default="data_clean/universe_long_blind.parquet")
    ap.add_argument("--blind_gate_samples", type=str, default="datasets_blind_v1/step2_features/gate_samples_v1.parquet")
    ap.add_argument("--blind_risk_bars", type=str, default="datasets_blind_v1/step2_features/risk_bars_features_v1.parquet")
    ap.add_argument("--blind_risk_states_v1", type=str, default="datasets_blind_v1/step3_risk_states/risk_states_v1.parquet")

    # common params
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--risk_pos_base", type=float, default=0.6)
    ap.add_argument("--pos_cap", type=float, default=1.0)
    ap.add_argument("--p_enter", type=float, default=0.55)

    # risk v2 params
    ap.add_argument("--risk2_win_short", type=int, default=24)
    ap.add_argument("--risk2_win_mid", type=int, default=48)
    ap.add_argument("--risk2_win_long", type=int, default=96)

    # gate reg params
    ap.add_argument("--run_gate_reg", action="store_true", help="训练 gate 回归 (mfe_atr) 并加入实验")
    ap.add_argument("--gate_reg_dir", type=str, default="models_gate_reg_v1")
    ap.add_argument("--mfe_clip", type=float, default=8.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load core data
    dev_gate_samples = pd.read_parquet(args.dev_gate_samples)
    blind_gate_samples = pd.read_parquet(args.blind_gate_samples)

    dev_risk_bars = pd.read_parquet(args.dev_risk_bars)
    blind_risk_bars = pd.read_parquet(args.blind_risk_bars)

    # risk v1 (直接读现成的)
    dev_risk_v1 = pd.read_parquet(args.dev_risk_states_v1)
    blind_risk_v1 = pd.read_parquet(args.blind_risk_states_v1)

    # risk v2 (现场生成)
    dev_risk_v2 = build_risk_states_v2(dev_risk_bars,
                                       win_short=args.risk2_win_short,
                                       win_mid=args.risk2_win_mid,
                                       win_long=args.risk2_win_long)
    blind_risk_v2 = build_risk_states_v2(blind_risk_bars,
                                         win_short=args.risk2_win_short,
                                         win_mid=args.risk2_win_mid,
                                         win_long=args.risk2_win_long)

    # save risk v2
    (out_dir / "risk_v2").mkdir(exist_ok=True)
    dev_risk_v2.to_parquet(out_dir / "risk_v2" / "dev_risk_states_v2.parquet", index=False)
    blind_risk_v2.to_parquet(out_dir / "risk_v2" / "blind_risk_states_v2.parquet", index=False)

    # gate variants
    gate_dev_oracle = gate_oracle_from_label(dev_gate_samples)
    gate_blind_oracle = gate_oracle_from_label(blind_gate_samples)

    gate_dev_bin = pd.read_parquet(args.dev_gate_oos_bin)[["symbol","event_id","p_hat","gate_pos0_hat"]].copy()
    gate_blind_bin, bin_model_used = infer_gate_bin_blind(
        blind_gate_samples, Path(args.bin_models_dir), p_enter=args.p_enter
    )

    # optional: gate reg mfe
    gate_dev_reg = None
    gate_blind_reg = None
    reg_model_used = None
    if args.run_gate_reg:
        reg_dir = Path(args.gate_reg_dir)
        oos_path, last_model, _feat_cols = train_gate_reg_wfv(
            args.dev_gate_samples, reg_dir,
            mfe_clip=args.mfe_clip
        )
        gate_dev_reg = pd.read_parquet(oos_path)[["symbol","event_id","p_hat","gate_pos0_hat"]].copy()
        gate_blind_reg, reg_model_used = infer_gate_reg_blind(
            blind_gate_samples, last_model, mfe_clip=args.mfe_clip
        )

    experiments = []

    # E0 oracle + R1
    experiments.append(("E0_oracle_R1", gate_dev_oracle, dev_risk_v1, gate_blind_oracle, blind_risk_v1))
    # E1 bin + R1
    experiments.append(("E1_bin_R1", gate_dev_bin, dev_risk_v1, gate_blind_bin, blind_risk_v1))
    # E3 bin + R2
    experiments.append(("E3_bin_R2", gate_dev_bin, dev_risk_v2, gate_blind_bin, blind_risk_v2))

    if args.run_gate_reg:
        # E2 reg + R1
        experiments.append(("E2_reg_R1", gate_dev_reg, dev_risk_v1, gate_blind_reg, blind_risk_v1))
        # E4 reg + R2
        experiments.append(("E4_reg_R2", gate_dev_reg, dev_risk_v2, gate_blind_reg, blind_risk_v2))

    compare_rows = []
    compare_json = {"experiments": {}, "notes": {
        "fee": args.fee,
        "risk_pos_base": args.risk_pos_base,
        "pos_cap": args.pos_cap,
        "bin_model_used_blind": bin_model_used,
        "reg_model_used_blind": reg_model_used
    }}

    for name, gate_dev, risk_dev, gate_blind, risk_blind in experiments:
        exp_dir = out_dir / name
        exp_dir.mkdir(exist_ok=True)

        # DEV
        dev_curve, dev_ps, dev_ov = build_position_curve(
            args.dev_universe, risk_dev, gate_dev,
            fee=args.fee, risk_pos_base=args.risk_pos_base, pos_cap=args.pos_cap
        )
        dev_curve.to_parquet(exp_dir / "dev_position_curve.parquet", index=False)
        (exp_dir / "summary_dev.json").write_text(json.dumps({
            "overall_equal_weight": dev_ov,
            "per_symbol": dev_ps
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # BLIND
        blind_curve, blind_ps, blind_ov = build_position_curve(
            args.blind_universe, risk_blind, gate_blind,
            fee=args.fee, risk_pos_base=args.risk_pos_base, pos_cap=args.pos_cap
        )
        blind_curve.to_parquet(exp_dir / "blind_position_curve.parquet", index=False)
        (exp_dir / "summary_blind.json").write_text(json.dumps({
            "overall_equal_weight": blind_ov,
            "per_symbol": blind_ps
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        compare_json["experiments"][name] = {
            "dev": {"overall_equal_weight": dev_ov},
            "blind": {"overall_equal_weight": blind_ov}
        }

        compare_rows.append({
            "exp": name,
            "dev_sharpe": dev_ov["sharpe_like"],
            "dev_calmar": dev_ov["calmar_like"],
            "dev_mdd": dev_ov["max_drawdown"],
            "dev_ret": dev_ov["total_return"],
            "blind_sharpe": blind_ov["sharpe_like"],
            "blind_calmar": blind_ov["calmar_like"],
            "blind_mdd": blind_ov["max_drawdown"],
            "blind_ret": blind_ov["total_return"],
        })

        print(f"[OK] {name} DEV sharpe={dev_ov['sharpe_like']:.3f} blind sharpe={blind_ov['sharpe_like']:.3f}")

    # save compare
    compare_path = out_dir / "compare_summary.json"
    compare_path.write_text(json.dumps(compare_json, indent=2, ensure_ascii=False), encoding="utf-8")

    tab = pd.DataFrame(compare_rows).sort_values(["blind_sharpe", "blind_calmar"], ascending=False)
    tab.to_csv(out_dir / "compare_table.csv", index=False, encoding="utf-8-sig")

    print("\n[DONE] compare saved:")
    print(" -", compare_path)
    print(" -", out_dir / "compare_table.csv")
    print("\nTop by blind_sharpe:")
    print(tab.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
