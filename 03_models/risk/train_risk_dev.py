# -*- coding: utf-8 -*-
"""
train_risk_dev.py

Build rule-based risk states on full data, and export dev subset by train_end.
This keeps the v1 pipeline reproducible while matching the dev/OOS split.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from risk_state_machine_v1 import RiskStateConfig, build_risk_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk_bars_features", type=str, default="artifacts/v1/features/risk_bars_features.parquet")
    ap.add_argument("--out_dir", type=str, default="artifacts/v1/models/risk")
    ap.add_argument("--train_end", type=str, default="2025-01-01")
    ap.add_argument("--tf", type=int, default=15)

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

    df = pd.read_parquet(args.risk_bars_features)
    states_full, summary = build_risk_states(df, cfg)

    full_path = out_dir / "risk_states_full.parquet"
    states_full.to_parquet(full_path, index=False)

    train_end = pd.to_datetime(args.train_end, utc=True)
    states_dev = states_full[states_full["ts"] < train_end].copy()
    dev_path = out_dir / "risk_states_dev.parquet"
    states_dev.to_parquet(dev_path, index=False)

    summary.update({
        "risk_bars_features": args.risk_bars_features,
        "train_end": args.train_end,
        "tf": args.tf,
        "out_paths": {
            "full": str(full_path),
            "dev": str(dev_path),
        },
        "rows_full": int(len(states_full)),
        "rows_dev": int(len(states_dev)),
    })

    summary_path = out_dir / "summary_risk_dev.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE]", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
