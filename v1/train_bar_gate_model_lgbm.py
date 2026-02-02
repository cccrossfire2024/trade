# -*- coding: utf-8 -*-
"""
train_bar_gate_model_lgbm.py (v1 entry)

Wrapper entry for the v1 gate training pipeline.
Delegates to the canonical dev training script.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_samples", type=str, default="artifacts/v1/features/gate_samples.parquet")
    ap.add_argument("--out_dir", type=str, default="artifacts/v1/models/gate")
    ap.add_argument("--train_end", type=str, default="2025-01-01")
    ap.add_argument("--tf", type=int, default=15)
    args, rest = ap.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "03_models" / "gate" / "train_gate_dev.py"

    cmd = [
        "python", str(script),
        "--gate_samples", args.gate_samples,
        "--out_dir", args.out_dir,
        "--train_end", args.train_end,
        "--tf", str(args.tf),
    ] + rest

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
