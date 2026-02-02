# -*- coding: utf-8 -*-
"""
one_click_backtest_v4_15m_bar_gate_pipeline.py (v1 entry)

Wrapper entry for the v1 OOS backtest pipeline.
Delegates to the canonical backtest script.
"""

import argparse
import subprocess
from pathlib import Path


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
    ap.add_argument("--tf", type=int, default=15)
    args, rest = ap.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "04_backtest" / "backtest_oos.py"

    cmd = [
        "python", str(script),
        "--universe_long", args.universe_long,
        "--risk_states", args.risk_states,
        "--gate_samples", args.gate_samples,
        "--gate_model", args.gate_model,
        "--gate_feature_list", args.gate_feature_list,
        "--gate_thresholds", args.gate_thresholds,
        "--out_dir", args.out_dir,
        "--bt_start", args.bt_start,
        "--tf", str(args.tf),
    ] + rest

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
