# -*- coding: utf-8 -*-
"""
train_risk_lgbm_dd16_pipeline.py (v1 entry)

Wrapper entry for the v1 risk pipeline.
Currently delegates to the rule-based risk state builder used in v1.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk_bars_features", type=str, default="artifacts/v1/features/risk_bars_features.parquet")
    ap.add_argument("--out_dir", type=str, default="artifacts/v1/models/risk")
    ap.add_argument("--train_end", type=str, default="2025-01-01")
    ap.add_argument("--tf", type=int, default=15)
    args, rest = ap.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "03_models" / "risk" / "train_risk_dev.py"

    cmd = [
        "python", str(script),
        "--risk_bars_features", args.risk_bars_features,
        "--out_dir", args.out_dir,
        "--train_end", args.train_end,
        "--tf", str(args.tf),
    ] + rest

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
