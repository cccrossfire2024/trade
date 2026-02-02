# -*- coding: utf-8 -*-
"""
backtest_oos.py

OOS backtest from bt_start:
- Gate: infer p_hat + gate_pos0_hat using trained model + feature list
- Risk: read risk_states_full (precomputed)
- Position curve with v1 state-machine executor
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


EXCLUDE_COLS = {
    "symbol", "event_id", "twarn_ts", "t0_ts", "t1_ts",
    "label", "mfe_atr", "y",
}
EPS = 1e-12


def to_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_feature_list(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def pick_feature_cols(df: pd.DataFrame, feature_list_path: Path | None) -> list[str]:
    if feature_list_path and feature_list_path.exists():
        cols = load_feature_list(feature_list_path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature list has missing columns: {missing}")
        return cols

    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def gate_pos_from_threshold(p: np.ndarray, threshold: float, floor: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    pos = (p - threshold) / (1.0 - threshold + EPS)
    pos = np.clip(pos, 0.0, 1.0)
    if floor > 0:
        pos = np.where(p >= threshold, np.maximum(pos, floor), 0.0)
    return pos


def sharpe_like(lr_1):
    lr = np.asarray(lr_1, dtype=float)
    m = np.nanmean(lr)
    s = np.nanstd(lr)
    ann = np.sqrt(365 * 24 * 4)  # 15m -> 96 bars/day
    return float(m / (s + EPS) * ann)


def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + EPS) - 1.0
    return float(np.nanmin(dd))


def calmar_like(total_ret, mdd):
    return float((total_ret + EPS) / (abs(mdd) + EPS))


def read_universe_long(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if df.index.name in ("ts", "timestamp"):
            df = df.reset_index().rename(columns={df.index.name: "ts"})
        else:
            raise ValueError("universe_long missing ts column")
    for c in ["symbol", "close"]:
        if c not in df.columns:
            raise ValueError(f"universe_long missing required col: {c}")
    df["ts"] = to_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df[["symbol", "ts", "close"]]


def read_risk_states(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if df.index.name in ("ts", "timestamp"):
            df = df.reset_index().rename(columns={df.index.name: "ts"})
        else:
            raise ValueError("risk_states missing ts column")
    need = ["symbol", "ts", "event_id", "state"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"risk_states missing required col: {c}")
    df["ts"] = to_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df[need]


def target_pos_from_state(base_pos: float, state: str, mult_expansion: float, mult_digest: float):
    if base_pos <= 0:
        return 0.0
    if state == "ADVANCE":
        return float(base_pos)
    if state == "EXPANSION":
        return float(min(1.0, base_pos * mult_expansion))
    if state == "DIGEST":
        return float(max(0.0, base_pos * mult_digest))
    if state in ("BREAKDOWN", "EXHAUST"):
        return 0.0
    return 0.0


def build_curve_one_symbol(px, rs, gate, fee, cooldown_bars, k_smooth, mult_expansion, mult_digest):
    df = px.copy().sort_values("ts").reset_index(drop=True)
    df["lr_1"] = np.log(df["close"] / df["close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rs2 = rs.copy()
    rs2 = rs2[["ts", "event_id", "state"]]
    df = df.merge(rs2, on="ts", how="left")

    if len(gate) > 0:
        df = df.merge(gate[["event_id", "p_hat", "gate_pos0_hat"]], on="event_id", how="left")
    else:
        df["p_hat"] = np.nan
        df["gate_pos0_hat"] = np.nan

    df["p_hat"] = df["p_hat"].fillna(0.0).astype(float)
    df["gate_pos0_hat"] = df["gate_pos0_hat"].fillna(0.0).astype(float)
    df["state"] = df["state"].fillna("NONE").astype(str)

    n = len(df)
    pos = np.zeros(n, dtype=float)
    turnover = np.zeros(n, dtype=float)
    fee_paid = np.zeros(n, dtype=float)

    cd = 0
    last_event = None

    for i in range(1, n):
        state = df.at[i, "state"]
        eid = df.at[i, "event_id"]
        base = float(df.at[i, "gate_pos0_hat"])

        in_event = pd.notna(eid) and (state != "NONE")

        if cd > 0:
            cd -= 1

        if not in_event:
            tgt = 0.0
        else:
            if (last_event is None) or (eid != last_event):
                last_event = eid
                if cd > 0:
                    base = 0.0

            tgt = target_pos_from_state(base, state, mult_expansion, mult_digest)

            if state in ("BREAKDOWN", "EXHAUST"):
                cd = max(cd, cooldown_bars)
                tgt = 0.0

            if base <= 0.0:
                tgt = 0.0

        pos[i] = pos[i - 1] + k_smooth * (tgt - pos[i - 1])
        turnover[i] = abs(pos[i] - pos[i - 1])
        fee_paid[i] = turnover[i] * fee

    strat_lr = pos * df["lr_1"].values - fee_paid
    equity = np.exp(np.nancumsum(strat_lr))

    out = pd.DataFrame({
        "ts": df["ts"].values,
        "pos": pos.astype("float32"),
        "turnover": turnover.astype("float32"),
        "fee": fee_paid.astype("float32"),
        "lr_1": df["lr_1"].astype("float32"),
        "strat_lr": strat_lr.astype("float32"),
        "equity": equity.astype("float64"),
        "state": df["state"].values,
        "event_id": df["event_id"].values,
        "p_hat": df["p_hat"].astype("float32"),
        "gate_pos0_hat": df["gate_pos0_hat"].astype("float32"),
    })
    return out


def summarize_symbol(curve: pd.DataFrame):
    lr = curve["strat_lr"].astype(float).values
    eq = curve["equity"].astype(float).values
    mdd = max_drawdown(eq)
    tot = float(eq[-1] - 1.0)
    turnover = curve["turnover"].astype(float).values
    fee = curve["fee"].astype(float).values
    gross = curve["pos"].astype(float).values * curve["lr_1"].astype(float).values
    gross_sum = float(np.nansum(gross))
    fee_sum = float(np.nansum(fee))
    fee_share = float(fee_sum / (abs(gross_sum) + fee_sum + EPS))
    return {
        "rows": int(len(curve)),
        "sharpe_like": sharpe_like(lr),
        "max_drawdown": mdd,
        "calmar_like": calmar_like(tot, mdd),
        "turnover_per_bar": float(np.nanmean(turnover)),
        "avg_exposure": float(np.nanmean(curve["pos"].astype(float).values)),
        "fee_share": fee_share,
        "total_return": tot,
    }


def summarize_portfolio(curves_by_symbol: dict):
    series = []
    for sym, c in curves_by_symbol.items():
        s = pd.Series(c["strat_lr"].astype(float).values, index=c["ts"].values, name=sym)
        series.append(s)
    lr_df = pd.concat(series, axis=1).fillna(0.0)
    lr_eq = lr_df.mean(axis=1).values
    eq = np.exp(np.nancumsum(lr_eq))
    mdd = max_drawdown(eq)
    tot = float(eq[-1] - 1.0)
    return {
        "sharpe_like": sharpe_like(lr_eq),
        "max_drawdown": mdd,
        "calmar_like": calmar_like(tot, mdd),
        "total_return": tot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_long", type=str, default="01_data/universe_long_all.parquet")
    ap.add_argument("--risk_states", type=str, default="artifacts/v1/models/risk/risk_states_full.parquet")
    ap.add_argument("--gate_samples", type=str, default="artifacts/v1/features/gate_samples.parquet")
    ap.add_argument("--gate_model", type=str, default="artifacts/v1/models/gate/models/lgbm_gate_dev.txt")
    ap.add_argument("--gate_feature_list", type=str, default="artifacts/v1/models/gate/gate_features.txt")
    ap.add_argument("--gate_thresholds", type=str, default="artifacts/v1/models/gate/summary_gate_dev.json")
    ap.add_argument("--out_dir", type=str, default="artifacts/v1/backtest")

    ap.add_argument("--bt_start", type=str, default="2025-01-01")
    ap.add_argument("--pass_tag", type=str, default="pass20")
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--floor", type=float, default=0.2)
    ap.add_argument("--min_bars", type=int, default=16)
    ap.add_argument("--cooldown_bars", type=int, default=32)
    ap.add_argument("--k_smooth", type=float, default=0.35)
    ap.add_argument("--mult_expansion", type=float, default=1.5)
    ap.add_argument("--mult_digest", type=float, default=0.5)
    ap.add_argument("--tf", type=int, default=15)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bt_start = pd.to_datetime(args.bt_start, utc=True)

    gate_samples = pd.read_parquet(args.gate_samples)
    gate_samples["t0_ts"] = to_utc(gate_samples["t0_ts"])
    gate_samples = gate_samples.dropna(subset=["t0_ts"]).sort_values("t0_ts").reset_index(drop=True)
    gate_oos = gate_samples[gate_samples["t0_ts"] >= bt_start].copy()

    feature_list_path = Path(args.gate_feature_list) if args.gate_feature_list else None
    feat_cols = pick_feature_cols(gate_oos, feature_list_path)
    if len(feat_cols) < 10:
        raise ValueError(f"Too few feature cols: {len(feat_cols)}")

    model = lgb.Booster(model_file=args.gate_model)
    p_hat = model.predict(gate_oos[feat_cols], num_iteration=model.best_iteration)

    thresholds = json.loads(Path(args.gate_thresholds).read_text(encoding="utf-8"))
    pass_thresholds = thresholds.get("thresholds", {})
    if args.pass_tag not in pass_thresholds:
        raise ValueError(f"pass_tag {args.pass_tag} not found in thresholds")
    thr = float(pass_thresholds[args.pass_tag])

    gate_oos["p_hat"] = p_hat
    gate_oos["gate_pos0_hat"] = gate_pos_from_threshold(p_hat, thr, args.floor)

    gate_oos_path = out_dir / "gate_oos.parquet"
    gate_oos[["symbol", "event_id", "p_hat", "gate_pos0_hat"]].to_parquet(gate_oos_path, index=False)

    px = read_universe_long(args.universe_long)
    px = px[px["ts"] >= bt_start].copy()
    rs = read_risk_states(args.risk_states)
    rs = rs[rs["ts"] >= bt_start].copy()

    syms = sorted(px["symbol"].unique().tolist())
    curves = {}
    per_sym = {}

    for sym in syms:
        px_s = px[px["symbol"] == sym][["ts", "close"]].copy()
        rs_s = rs[rs["symbol"] == sym][["ts", "event_id", "state"]].copy()
        gate_s = gate_oos[gate_oos["symbol"] == sym][["event_id", "p_hat", "gate_pos0_hat"]].copy()

        c = build_curve_one_symbol(
            px_s, rs_s, gate_s,
            fee=args.fee,
            cooldown_bars=args.cooldown_bars,
            k_smooth=args.k_smooth,
            mult_expansion=args.mult_expansion,
            mult_digest=args.mult_digest,
        )
        c.insert(0, "symbol", sym)
        curves[sym] = c
        per_sym[sym] = summarize_symbol(c)

    overall = summarize_portfolio(curves)
    curve_all = pd.concat(curves.values(), ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)
    out_path = out_dir / "position_curve_oos.parquet"
    curve_all.to_parquet(out_path, index=False)

    summary = {
        "bt_start": args.bt_start,
        "tf": args.tf,
        "min_bars": args.min_bars,
        "gate_oos": str(gate_oos_path),
        "position_curve": str(out_path),
        "overall_equal_weight": overall,
        "per_symbol": per_sym,
        "exec_params": {
            "fee": args.fee,
            "cooldown_bars": args.cooldown_bars,
            "k_smooth": args.k_smooth,
            "mult_expansion": args.mult_expansion,
            "mult_digest": args.mult_digest,
            "floor": args.floor,
            "pass_tag": args.pass_tag,
        },
    }

    summary_path = out_dir / "summary_backtest_oos.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE]", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
