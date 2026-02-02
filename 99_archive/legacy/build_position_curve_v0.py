# -*- coding: utf-8 -*-
"""
build_position_curve_v0.py

融合 Gate(事件级) + Risk(事件内状态机 target_pos) -> 最终仓位曲线 position
并做一个轻量回测 sanity check（含手续费、换手、粗略 Sharpe/Calmar）

输入：
- data_clean/universe_long_dev.parquet
- datasets_v2/gate_samples_v1.parquet
- datasets_v3/risk_states_v1.parquet

输出：
- datasets_v4/position_curve_dev.parquet
- datasets_v4/summary_position_curve_v0.json

运行：
  python build_position_curve_v0.py --out_dir datasets_v4
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def annualization_factor_15m():
    # 15m bar per day = 96
    # crypto 365 days
    return np.sqrt(96 * 365)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())


def calmar_ratio(equity: pd.Series, bars_per_year=96*365):
    if len(equity) < 10:
        return np.nan
    total_ret = float(equity.iloc[-1] / (equity.iloc[0] + 1e-12) - 1.0)
    years = len(equity) / bars_per_year
    if years <= 0:
        return np.nan
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-12:
        return np.nan
    return float(cagr / mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_long_dev", type=str, default="data_clean/universe_long_dev.parquet")
    ap.add_argument("--gate_samples", type=str, default="datasets_v2/gate_samples_v1.parquet")
    ap.add_argument("--risk_states", type=str, default="datasets_v3/risk_states_v1.parquet")
    ap.add_argument("--out_dir", type=str, default="datasets_v4")

    # fee model (simple)
    ap.add_argument("--fee", type=float, default=0.0004)

    # Gate -> base position mapping (heuristic v0; later replace by model output)
    ap.add_argument("--pos_none", type=float, default=0.0)
    ap.add_argument("--pos_small", type=float, default=0.4)
    ap.add_argument("--pos_mid", type=float, default=0.7)
    ap.add_argument("--pos_big", type=float, default=1.0)

    # Risk multiplier: use target_pos relative to pos_base
    ap.add_argument("--risk_pos_base", type=float, default=0.6)
    ap.add_argument("--pos_cap", type=float, default=1.0)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    uni = pd.read_parquet(args.universe_long_dev)
    uni["ts"] = pd.to_datetime(uni["ts"], utc=True)
    uni = uni.sort_values(["symbol","ts"]).reset_index(drop=True)

    gate = pd.read_parquet(args.gate_samples)
    gate["t0_ts"] = pd.to_datetime(gate["t0_ts"], utc=True)
    gate["t1_ts"] = pd.to_datetime(gate["t1_ts"], utc=True)

    risk = pd.read_parquet(args.risk_states)
    risk["ts"] = pd.to_datetime(risk["ts"], utc=True)

    # -------------------------
    # Gate base position
    # -------------------------
    pos_map = {
        "NONE": args.pos_none,
        "SMALL": args.pos_small,
        "MID": args.pos_mid,
        "BIG": args.pos_big
    }
    gate["gate_pos0"] = gate["label"].map(pos_map).astype(float)

    # Keep only required gate columns for join
    gate_key = gate[["symbol","event_id","gate_pos0","label","mfe_atr","t0_ts","t1_ts"]].copy()

    # -------------------------
    # Risk multiplier from risk target_pos
    # -------------------------
    # risk_states_v1 has: symbol,event_id,ts,state,target_pos,(and other cols)
    needed = ["symbol","event_id","ts","state","target_pos"]
    missing = [c for c in needed if c not in risk.columns]
    if missing:
        raise ValueError(f"risk_states missing columns: {missing}")

    risk_small = risk[needed].copy()

    # relative multiplier: target_pos / pos_base
    risk_small["risk_mult"] = risk_small["target_pos"] / (args.risk_pos_base + 1e-12)
    # cap multiplier to avoid explosive leverage
    risk_small["risk_mult"] = risk_small["risk_mult"].clip(0.0, args.pos_cap / (args.risk_pos_base + 1e-12))

    # -------------------------
    # Merge Gate + Risk to get final position inside events
    # -------------------------
    merged = risk_small.merge(gate_key, on=["symbol","event_id"], how="left")

    # If for some reason gate missing, default to 0 (safer)
    merged["gate_pos0"] = merged["gate_pos0"].fillna(0.0)

    merged["pos"] = (merged["gate_pos0"] * merged["risk_mult"]).clip(0.0, args.pos_cap)

    # -------------------------
    # Build full position curve aligned with universe bars
    # Outside events: pos=0
    # -------------------------
    # Create base frame: all bars
    base = uni[["ts","symbol","close"]].copy()

    # join positions by (symbol, ts)
    base = base.merge(
        merged[["symbol","ts","event_id","state","target_pos","gate_pos0","risk_mult","pos"]],
        on=["symbol","ts"],
        how="left"
    )

    base["pos"] = base["pos"].fillna(0.0)
    base["event_id"] = base["event_id"].fillna(-1).astype(int)
    base["state"] = base["state"].fillna("OUT")
    base["gate_pos0"] = base["gate_pos0"].fillna(0.0)
    base["risk_mult"] = base["risk_mult"].fillna(0.0)
    base["target_pos"] = base["target_pos"].fillna(0.0)

    # -------------------------
    # Sanity backtest (long-only)
    # pnl_t = pos_{t-1} * ret_t - fee * |pos_t - pos_{t-1}|
    # -------------------------
    base = base.sort_values(["symbol","ts"]).reset_index(drop=True)
    g = base.groupby("symbol", group_keys=False)

    base["ret1"] = g["close"].pct_change().fillna(0.0)
    base["pos_prev"] = g["pos"].shift(1).fillna(0.0)
    base["dpos"] = (base["pos"] - base["pos_prev"]).abs()

    base["pnl"] = base["pos_prev"] * base["ret1"] - args.fee * base["dpos"]
    # equity per symbol
    base["equity"] = g["pnl"].cumsum() + 1.0

    # -------------------------
    # Summary metrics per symbol and overall (equal-weight)
    # -------------------------
    summaries = {}
    for sym, sub in base.groupby("symbol"):
        pnl = sub["pnl"]
        mu = float(pnl.mean())
        sd = float(pnl.std())
        sharpe = (mu / (sd + 1e-12)) * annualization_factor_15m()
        eq = sub["equity"]
        mdd = max_drawdown(eq)
        calmar = calmar_ratio(eq)
        turnover = float(sub["dpos"].mean())  # average abs position change per bar

        summaries[sym] = {
            "rows": int(len(sub)),
            "sharpe_like": sharpe,
            "max_drawdown": mdd,
            "calmar_like": calmar,
            "turnover_per_bar": turnover,
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        }

    # overall equal-weight pnl
    pivot = base.pivot(index="ts", columns="symbol", values="pnl").fillna(0.0)
    pnl_all = pivot.mean(axis=1)
    eq_all = pnl_all.cumsum() + 1.0
    mu = float(pnl_all.mean())
    sd = float(pnl_all.std())
    sharpe_all = (mu / (sd + 1e-12)) * annualization_factor_15m()
    mdd_all = max_drawdown(eq_all)
    calmar_all = calmar_ratio(eq_all)

    # save
    out_path = out_dir / "position_curve_dev.parquet"
    base.to_parquet(out_path, index=False)

    summary = {
        "out_path": str(out_path),
        "rows": int(len(base)),
        "fee": args.fee,
        "gate_pos_map": {
            "NONE": args.pos_none,
            "SMALL": args.pos_small,
            "MID": args.pos_mid,
            "BIG": args.pos_big
        },
        "risk_pos_base": args.risk_pos_base,
        "pos_cap": args.pos_cap,
        "per_symbol": summaries,
        "overall_equal_weight": {
            "sharpe_like": sharpe_all,
            "max_drawdown": mdd_all,
            "calmar_like": calmar_all,
            "total_return": float(eq_all.iloc[-1] / eq_all.iloc[0] - 1.0),
        }
    }

    (out_dir / "summary_position_curve_v0.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("[DONE]", json.dumps({
        "out_path": summary["out_path"],
        "overall_equal_weight": summary["overall_equal_weight"],
        "rows": summary["rows"]
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
