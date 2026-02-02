# -*- coding: utf-8 -*-
"""
risk_state_machine_v1.py

相比 v0：
- rolling quantile 不再要求满窗口：min_periods = max(16, win//3)
- ~ready 不再强制 DIGEST，而是走简化规则
- 让短事件（MID/SMALL/NONE）也能产生合理状态分布
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def rolling_q(s: pd.Series, win: int, q: float, minp: int) -> pd.Series:
    return s.rolling(win, min_periods=minp).quantile(q)


def safe_z(s: pd.Series, win: int, minp: int) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - mu) / (sd + 1e-12)


@dataclass
class RiskStateConfig:
    win_short: int = 24
    win_mid: int = 48
    win_long: int = 96
    cooldown_bars: int = 32
    pos_base: float = 0.6
    pos_add: float = 0.25
    pos_reduce: float = 0.25
    pos_max: float = 1.0
    pos_min: float = 0.0


def build_risk_states(df: pd.DataFrame, cfg: RiskStateConfig) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "event_id", "ts"]).reset_index(drop=True)

    required = [
        "symbol", "event_id", "ts", "progress",
        "gmma_gap", "gmma_slopeS", "gmma_slopeS_acc",
        "er_16", "vr_16_64",
        "rs_vol_rel", "vov_96",
        "lr_16", "wick_skew", "body_eff",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    winS, winM, winL = cfg.win_short, cfg.win_mid, cfg.win_long
    minS = max(16, winS // 3)
    minM = max(16, winM // 3)
    minL = max(16, winL // 3)

    g = df.groupby(["symbol", "event_id"], group_keys=False)

    # gap change
    df["gap_chg_1"] = g["gmma_gap"].diff(1)
    df["gap_chg_z"] = g["gap_chg_1"].apply(lambda s: safe_z(s, winS, minS))

    # quantile thresholds (with relaxed min_periods)
    df["slopeS_q70"] = g["gmma_slopeS"].apply(lambda s: rolling_q(s, winM, 0.70, minM))
    df["slopeS_q30"] = g["gmma_slopeS"].apply(lambda s: rolling_q(s, winM, 0.30, minM))

    df["acc_q30"] = g["gmma_slopeS_acc"].apply(lambda s: rolling_q(s, winM, 0.30, minM))
    df["acc_q10"] = g["gmma_slopeS_acc"].apply(lambda s: rolling_q(s, winM, 0.10, minM))

    df["er_q70"] = g["er_16"].apply(lambda s: rolling_q(s, winM, 0.70, minM))
    df["er_q40"] = g["er_16"].apply(lambda s: rolling_q(s, winM, 0.40, minM))
    df["er_q20"] = g["er_16"].apply(lambda s: rolling_q(s, winM, 0.20, minM))

    df["rs_q70"] = g["rs_vol_rel"].apply(lambda s: rolling_q(s, winM, 0.70, minM))
    df["rs_q90"] = g["rs_vol_rel"].apply(lambda s: rolling_q(s, winM, 0.90, minM))

    df["vov_q70"] = g["vov_96"].apply(lambda s: rolling_q(s, winL, 0.70, minL))
    df["vov_q90"] = g["vov_96"].apply(lambda s: rolling_q(s, winL, 0.90, minL))

    df["lr16_q30"] = g["lr_16"].apply(lambda s: rolling_q(s, winM, 0.30, minM))
    df["lr16_q10"] = g["lr_16"].apply(lambda s: rolling_q(s, winM, 0.10, minM))

    df["body_q30"] = g["body_eff"].apply(lambda s: rolling_q(s, winM, 0.30, minM))
    df["wick_q70"] = g["wick_skew"].apply(lambda s: rolling_q(s, winM, 0.70, minM))

    # readiness：只要 mid 的阈值大部分存在就算 ready
    need_some = ["slopeS_q70", "er_q70", "rs_q70", "lr16_q30", "acc_q30"]
    df["ready"] = df[need_some].notna().all(axis=1)

    # --------------------------
    # Main rules (same spirit as v0)
    # --------------------------
    breakdown = (
        (df["gap_chg_z"] < -0.8)
        & (
            (df["vov_96"] > df["vov_q90"])
            | (df["rs_vol_rel"] > df["rs_q90"])
            | (df["lr_16"] < df["lr16_q10"])
        )
    )

    exhaust = (
        (df["gmma_slopeS_acc"] < df["acc_q10"])
        & (df["er_16"] < df["er_q20"])
        & (
            (df["body_eff"] < df["body_q30"])
            | (df["lr_16"] < df["lr16_q30"])
            | (df["wick_skew"] > df["wick_q70"])
        )
    )

    advance = (
        (df["gmma_slopeS"] > df["slopeS_q70"])
        & (df["er_16"] > df["er_q70"])
        & (df["gap_chg_1"] > 0)
        & (df["gmma_slopeS"] > 0)
    )

    expansion = (
        (df["gmma_slopeS"] > df["slopeS_q30"])
        & (df["er_16"] > df["er_q40"])
        & (df["gmma_slopeS"] > 0)
        & ((df["rs_vol_rel"] > df["rs_q70"]) | (df["vov_96"] > df["vov_q70"]))
    )

    df["state"] = "DIGEST"
    df.loc[expansion, "state"] = "EXPANSION"
    df.loc[advance, "state"] = "ADVANCE"
    df.loc[exhaust, "state"] = "EXHAUST"
    df.loc[breakdown, "state"] = "BREAKDOWN"

    # --------------------------
    # Fallback rules for ~ready
    # --------------------------
    not_ready = ~df["ready"]

    fallback_breakdown = (
        (df["gap_chg_1"] < 0)
        & (df["lr_16"] < 0)
        & (df["rs_vol_rel"] > 1.0)
    )

    fallback_advance = (
        (df["gmma_slopeS"] > 0)
        & (df["gap_chg_1"] > 0)
        & (df["er_16"] > 0.35)
    )

    df.loc[not_ready & fallback_advance, "state"] = "ADVANCE"
    df.loc[not_ready & fallback_breakdown, "state"] = "BREAKDOWN"

    # --------------------------
    # Target position + cooldown
    # --------------------------
    base = cfg.pos_base
    pos_map = {
        "ADVANCE": min(cfg.pos_max, base + cfg.pos_add),
        "EXPANSION": min(cfg.pos_max, base + 0.5 * cfg.pos_add),
        "DIGEST": base,
        "EXHAUST": max(cfg.pos_min, base - cfg.pos_reduce),
        "BREAKDOWN": cfg.pos_min,
    }
    df["target_pos_raw"] = df["state"].map(pos_map).astype(float)

    cooldown = cfg.cooldown_bars

    def apply_cooldown(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values("ts").copy()
        cd = 0
        out = []
        for _, r in sub.iterrows():
            if r["state"] == "BREAKDOWN":
                cd = cooldown
            if cd > 0:
                out.append(0.0)
                cd -= 1
            else:
                out.append(float(r["target_pos_raw"]))
        sub["target_pos"] = out
        return sub

    df = df.groupby(["symbol", "event_id"], group_keys=False).apply(apply_cooldown)
    df["target_pos"] = df["target_pos"].clip(cfg.pos_min, cfg.pos_max)

    summary = {
        "rows": int(len(df)),
        "state_counts": df["state"].value_counts().to_dict(),
        "windows": {"win_short": winS, "win_mid": winM, "win_long": winL},
        "min_periods": {"minS": minS, "minM": minM, "minL": minL},
        "cooldown_bars": cooldown,
        "pos_params": {
            "pos_base": cfg.pos_base,
            "pos_add": cfg.pos_add,
            "pos_reduce": cfg.pos_reduce,
            "pos_min": cfg.pos_min,
            "pos_max": cfg.pos_max,
        },
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="artifacts/v1/features/risk_bars_features.parquet")
    ap.add_argument("--out_dir", type=str, default="artifacts/v1/models/risk")

    ap.add_argument("--win_short", type=int, default=24)
    ap.add_argument("--win_mid", type=int, default=48)
    ap.add_argument("--win_long", type=int, default=96)

    ap.add_argument("--cooldown_bars", type=int, default=32)

    ap.add_argument("--pos_base", type=float, default=0.6)
    ap.add_argument("--pos_add", type=float, default=0.25)
    ap.add_argument("--pos_reduce", type=float, default=0.25)
    ap.add_argument("--pos_max", type=float, default=1.0)
    ap.add_argument("--pos_min", type=float, default=0.0)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RiskStateConfig(
        win_short=args.win_short,
        win_mid=args.win_mid,
        win_long=args.win_long,
        cooldown_bars=args.cooldown_bars,
        pos_base=args.pos_base,
        pos_add=args.pos_add,
        pos_reduce=args.pos_reduce,
        pos_max=args.pos_max,
        pos_min=args.pos_min,
    )

    df = pd.read_parquet(args.in_path)
    df, summary = build_risk_states(df, cfg)

    out_path = out_dir / "risk_states_full.parquet"
    df.to_parquet(out_path, index=False)

    summary.update({
        "in_path": args.in_path,
        "out_path": str(out_path),
    })
    (out_dir / "summary_risk_states_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("[DONE]", json.dumps({
        "rows": summary["rows"],
        "out_path": summary["out_path"],
        "state_counts": summary["state_counts"],
        "cooldown_bars": summary["cooldown_bars"],
        "windows": summary["windows"],
        "min_periods": summary["min_periods"]
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
