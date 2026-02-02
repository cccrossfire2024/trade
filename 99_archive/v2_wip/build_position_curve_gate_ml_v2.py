# -*- coding: utf-8 -*-
"""
build_position_curve_gate_ml_v2.py

用 Gate ML（dev OOS + dev full-fit -> blind predict）结合 RiskStates(v1) 生成仓位曲线。
输出 dev/blind summary，并保存 position_curve_{dev,blind}.parquet

用法：
python build_position_curve_gate_ml_v2.py --out_dir position_ml_v2

依赖：
- data_clean/universe_long_dev.parquet
- data_clean/universe_long_blind.parquet
- datasets_v1/gmma_events_dev.parquet
- datasets_v1/gmma_events_blind.parquet (如果没有，会从 dev events 用 symbol+t0_ts 方式 fallback)
- datasets_v3/risk_states_v1.parquet
- models_gate_v2/gate_oos_dev.parquet
- datasets_v2plus/gate_samples_v2.parquet (用于训练 full-fit 模型给 blind)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

EPS = 1e-12


def to_utc_ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_universe_long(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.name in ("ts", "timestamp"):
        df = df.reset_index()
    if "ts" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    # normalize lower
    rename_map = {}
    for c in df.columns:
        lo = c.lower()
        if lo in ("open", "high", "low", "close", "volume", "atr", "symbol", "ts"):
            rename_map[c] = lo
    if rename_map:
        df = df.rename(columns=rename_map)
    df["ts"] = to_utc_ts(df["ts"])
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df


def pick_feature_cols(df: pd.DataFrame):
    drop = {
        "symbol", "event_id", "t0_ts", "t1_ts", "twarn_ts",
        "label", "mfe_atr"
    }
    cols = [c for c in df.columns if c not in drop]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def calc_metrics_from_curve(curve: pd.DataFrame, ret_col="pnl"):
    # sharpe_like: mean/std * sqrt(365*24*4)  (15m bars)
    pnl = curve[ret_col].astype(float).fillna(0.0).values
    mu = pnl.mean()
    sd = pnl.std() + EPS
    ann = np.sqrt(365.0 * 24.0 * 4.0)
    sharpe_like = float(mu / sd * ann)

    # equity curve
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min())  # negative

    total_return = float(eq[-1] - eq[0]) if len(eq) > 1 else float(eq[-1]) if len(eq) else 0.0
    calmar_like = float((total_return / (abs(max_dd) + EPS))) if max_dd < 0 else float("inf")

    return sharpe_like, max_dd, calmar_like, total_return


def build_bar_level_positions(universe: pd.DataFrame,
                              risk_states: pd.DataFrame,
                              gate_events: pd.DataFrame,
                              gate_prob_by_event: pd.DataFrame,
                              fee: float = 0.0004,
                              thr_low: float = 0.50,
                              thr_high: float = 0.65):
    """
    生成每根bar的仓位 pos_t（等权组合的单币仓位，最后等权合成）
    规则（可解释且稳）：
    - 只在 ADVANCE/EXPANSION 才允许加大仓位
    - DIGEST 只保留小仓位（避免 churn）
    - BREAKDOWN/EXHAUST 清仓并触发 cooldown（risk_states 已经内置 cooldown）
    - gate_prob 控制初始/最大仓位：weak=0.5, strong=1.0（与你之前一致）
    """
    u = universe.copy()
    u["ts"] = to_utc_ts(u["ts"])
    u = u.sort_values(["symbol", "ts"]).reset_index(drop=True)

    rs = risk_states.copy()
    rs["ts"] = to_utc_ts(rs["ts"])
    rs = rs.sort_values(["symbol", "ts"]).reset_index(drop=True)

    ge = gate_events.copy()
    ge["t0_ts"] = to_utc_ts(ge["t0_ts"])
    ge = ge[["symbol", "event_id", "t0_ts"]].dropna().copy()

    gp = gate_prob_by_event.copy()
    # expect columns: symbol,event_id,gate_prob
    gp = gp[["symbol", "event_id", "gate_prob"]].copy()

    # map event_id->gate_prob
    ge = ge.merge(gp, on=["symbol", "event_id"], how="left")
    # if missing gate_prob: treat as 0 (no trade)
    ge["gate_prob"] = ge["gate_prob"].fillna(0.0)

    # merge risk state to universe
    u = u.merge(rs[["symbol", "ts", "event_id", "state"]], on=["symbol", "ts"], how="left")

    # attach gate_prob to each bar via event_id
    u = u.merge(ge[["symbol", "event_id", "gate_prob", "t0_ts"]], on=["symbol", "event_id"], how="left")

    # compute bar return
    u["ret_1"] = np.log((u["close"].astype(float) + EPS) / (u["close"].astype(float).shift(1) + EPS))
    u.loc[u.groupby("symbol").head(1).index, "ret_1"] = 0.0

    # position logic per symbol
    all_curves = []
    for sym, sub in u.groupby("symbol", sort=False):
        sub = sub.sort_values("ts").reset_index(drop=True)

        pos = np.zeros(len(sub), dtype=float)
        gatep = sub["gate_prob"].fillna(0.0).values
        state = sub["state"].fillna("NONE").values

        for i in range(len(sub)):
            gp_i = gatep[i]
            st = state[i]

            # base target size from gate probability
            if gp_i >= thr_high:
                base = 1.0
            elif gp_i >= thr_low:
                base = 0.5
            else:
                base = 0.0

            # state modulation
            if st in ("BREAKDOWN", "EXHAUST"):
                tgt = 0.0
            elif st == "EXPANSION":
                tgt = base * 1.0
            elif st == "ADVANCE":
                tgt = base * 0.8
            elif st == "DIGEST":
                tgt = base * 0.3
            else:
                # NONE / unknown
                tgt = 0.0

            # simple smooth (avoid micro-churn): only move 50% toward target each bar
            if i == 0:
                pos[i] = tgt
            else:
                pos[i] = pos[i - 1] + 0.5 * (tgt - pos[i - 1])

        sub["pos"] = pos

        # pnl with fee
        dpos = np.abs(np.diff(np.r_[0.0, pos]))
        fee_cost = fee * dpos
        pnl = pos[:-1] * sub["ret_1"].values[1:]  # pos_{t-1} * ret_t
        pnl = np.r_[0.0, pnl] - fee_cost

        sub["fee_cost"] = fee_cost
        sub["pnl"] = pnl
        sub["turnover"] = dpos

        # diagnostics
        sub["exposure"] = np.abs(sub["pos"])
        all_curves.append(sub[["ts", "symbol", "event_id", "state", "gate_prob", "pos", "pnl", "turnover", "fee_cost", "exposure"]])

    curve = pd.concat(all_curves, ignore_index=True)
    return curve


def train_fullfit_gate_model(gate_samples_path: str, seed: int = 42):
    df = pd.read_parquet(gate_samples_path)
    df["t0_ts"] = to_utc_ts(df["t0_ts"])
    df = df.dropna(subset=["t0_ts"]).sort_values("t0_ts").reset_index(drop=True)

    y = (df["label"].astype(str) != "NONE").astype(int).values
    feat_cols = pick_feature_cols(df)
    X = df[feat_cols]

    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    spw = (neg / (pos + EPS)) if pos > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1,
        "force_col_wise": True,
        "scale_pos_weight": spw,
    }

    dtr = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=True)
    booster = lgb.train(params, dtr, num_boost_round=800)

    return booster, feat_cols


def predict_gate_prob_for_events(booster, feat_cols, gate_samples_path: str, events_path: str, split_name: str):
    """
    用 full-fit 模型对指定 split（dev 或 blind）的 gate events 生成 gate_prob
    输出 columns: symbol,event_id,gate_prob
    """
    gs = pd.read_parquet(gate_samples_path)
    gs["t0_ts"] = to_utc_ts(gs["t0_ts"])
    X = gs[feat_cols]
    prob = booster.predict(X)
    pred = gs[["symbol", "event_id", "t0_ts"]].copy()
    pred["gate_prob"] = prob.astype(float)

    # 对齐到 events（确保 event_id 集合一致）
    ev = pd.read_parquet(events_path)
    # normalize minimal cols
    if "t0_ts" not in ev.columns:
        # fallback detection
        for c in ("t0", "start_ts", "start_time"):
            if c in ev.columns:
                ev = ev.rename(columns={c: "t0_ts"})
                break
    ev["t0_ts"] = to_utc_ts(ev["t0_ts"])
    ev = ev[["symbol", "event_id", "t0_ts"]].copy()

    # merge by (symbol,event_id)
    out = ev.merge(pred[["symbol", "event_id", "gate_prob"]], on=["symbol", "event_id"], how="left")

    # 若缺失（极少数），用 (symbol,t0_ts) 再兜底
    miss = out["gate_prob"].isna()
    if miss.any():
        out2 = ev.merge(pred[["symbol", "t0_ts", "gate_prob"]], on=["symbol", "t0_ts"], how="left", suffixes=("", "_ts"))
        out.loc[miss, "gate_prob"] = out2.loc[miss, "gate_prob"]

    out["gate_prob"] = out["gate_prob"].fillna(0.0)
    out["split"] = split_name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--universe_dev", type=str, default="data_clean/universe_long_dev.parquet")
    ap.add_argument("--universe_blind", type=str, default="data_clean/universe_long_blind.parquet")

    ap.add_argument("--events_dev", type=str, default="datasets_v1/gmma_events_dev.parquet")
    ap.add_argument("--events_blind", type=str, default="datasets_blind_v1/step1_events/gmma_events_blind.parquet")
    ap.add_argument("--risk_states", type=str, default="datasets_v3/risk_states_v1.parquet")

    ap.add_argument("--gate_oos_dev", type=str, default="models_gate_v2/gate_oos_dev.parquet")
    ap.add_argument("--gate_samples_v2_dev", type=str, default="datasets_v2plus/gate_samples_v2.parquet")
    ap.add_argument("--gate_samples_v2_blind", type=str, default="datasets_blind_v1/step2_features/gate_samples_v2.parquet")

    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--thr_low", type=float, default=0.50)
    ap.add_argument("--thr_high", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load universe
    dev_u = load_universe_long(args.universe_dev)
    blind_u = load_universe_long(args.universe_blind)

    # Load risk states
    rs = pd.read_parquet(args.risk_states)
    rs["ts"] = to_utc_ts(rs["ts"])

    # Load events (dev always)
    dev_ev = pd.read_parquet(args.events_dev)
    if "t0_ts" not in dev_ev.columns:
        for c in ("t0", "start_ts", "start_time"):
            if c in dev_ev.columns:
                dev_ev = dev_ev.rename(columns={c: "t0_ts"})
                break
    dev_ev["t0_ts"] = to_utc_ts(dev_ev["t0_ts"])

    # Blind events：若缺失就用 dev events 做 fallback（不推荐，但能跑）
    try:
        blind_ev = pd.read_parquet(args.events_blind)
    except Exception:
        blind_ev = None

    if blind_ev is None or len(blind_ev) == 0:
        # fallback: use dev events as placeholder (won't match blind event_id, so almost all gate_prob=0)
        blind_ev = dev_ev.copy()
        blind_ev = blind_ev.iloc[0:0].copy()

    if "t0_ts" not in blind_ev.columns:
        for c in ("t0", "start_ts", "start_time"):
            if c in blind_ev.columns:
                blind_ev = blind_ev.rename(columns={c: "t0_ts"})
                break
    blind_ev["t0_ts"] = to_utc_ts(blind_ev["t0_ts"])

    # --- DEV gate prob: use OOS first ---
    gate_oos = pd.read_parquet(args.gate_oos_dev)
    gate_oos["t0_ts"] = to_utc_ts(gate_oos["t0_ts"])
    gate_prob_dev = gate_oos[["symbol", "event_id", "gate_prob_oos"]].rename(columns={"gate_prob_oos": "gate_prob"})
    gate_prob_dev["gate_prob"] = gate_prob_dev["gate_prob"].fillna(0.0)

    # --- BLIND gate prob: train full-fit on dev gate_samples_v2, then predict blind gate_samples_v2 ---
    booster, feat_cols = train_fullfit_gate_model(args.gate_samples_v2_dev, seed=args.seed)

    # blind gate samples v2：如果你还没跑 blind v2 features，这里会报错
    blind_gate_prob = None
    try:
        blind_gate_prob = predict_gate_prob_for_events(
            booster, feat_cols,
            gate_samples_path=args.gate_samples_v2_blind,
            events_path=args.events_blind,
            split_name="blind"
        )
        blind_gate_prob = blind_gate_prob[["symbol", "event_id", "gate_prob"]].copy()
    except Exception as e:
        print(f"[WARN] blind gate samples v2 not found or failed: {e}")
        print("[WARN] fallback: set blind gate_prob=0 (no trades).")
        blind_gate_prob = blind_ev[["symbol", "event_id"]].copy()
        blind_gate_prob["gate_prob"] = 0.0

    # --- build curves ---
    # risk states should contain both dev+blind rows; if it's dev-only, blind will have many NaN states -> no trades
    curve_dev = build_bar_level_positions(
        universe=dev_u,
        risk_states=rs,
        gate_events=dev_ev.rename(columns={"t0_ts": "t0_ts"})[["symbol", "event_id", "t0_ts"]],
        gate_prob_by_event=gate_prob_dev,
        fee=args.fee,
        thr_low=args.thr_low,
        thr_high=args.thr_high
    )

    curve_blind = build_bar_level_positions(
        universe=blind_u,
        risk_states=rs,
        gate_events=blind_ev.rename(columns={"t0_ts": "t0_ts"})[["symbol", "event_id", "t0_ts"]] if len(blind_ev) else blind_ev,
        gate_prob_by_event=blind_gate_prob,
        fee=args.fee,
        thr_low=args.thr_low,
        thr_high=args.thr_high
    )

    # --- aggregate to equal-weight portfolio ---
    def summarize(curve: pd.DataFrame):
        # equal weight across symbols each bar
        curve = curve.sort_values(["ts", "symbol"]).reset_index(drop=True)
        port = curve.groupby("ts", as_index=False).agg(
            pnl=("pnl", "mean"),
            turnover=("turnover", "mean"),
            fee_cost=("fee_cost", "mean"),
            exposure=("exposure", "mean"),
        )
        sharpe_like, max_dd, calmar_like, total_return = calc_metrics_from_curve(port, ret_col="pnl")
        fee_share = float(port["fee_cost"].sum() / (port["pnl"].abs().sum() + EPS))
        return {
            "sharpe_like": sharpe_like,
            "max_drawdown": max_dd,
            "calmar_like": calmar_like,
            "total_return": total_return,
            "turnover_per_bar": float(port["turnover"].mean()),
            "avg_exposure": float(port["exposure"].mean()),
            "fee_share": fee_share,
            "rows": int(len(port)),
        }, port

    dev_sum, dev_port = summarize(curve_dev)
    blind_sum, blind_port = summarize(curve_blind)

    # save
    dev_curve_path = out_dir / "position_curve_dev_ml_v2.parquet"
    blind_curve_path = out_dir / "position_curve_blind_ml_v2.parquet"
    dev_port.to_parquet(dev_curve_path, index=False)
    blind_port.to_parquet(blind_curve_path, index=False)

    summary = {
        "dev": dev_sum,
        "blind": blind_sum,
        "paths": {
            "dev_port_curve": str(dev_curve_path),
            "blind_port_curve": str(blind_curve_path),
        },
        "params": {
            "fee": args.fee,
            "thr_low": args.thr_low,
            "thr_high": args.thr_high,
        }
    }
    (out_dir / "summary_position_ml_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DEV DONE]", json.dumps(dev_sum, indent=2, ensure_ascii=False))
    print("[BLIND DONE]", json.dumps(blind_sum, indent=2, ensure_ascii=False))
    print("[DONE] saved:", json.dumps(summary["paths"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
