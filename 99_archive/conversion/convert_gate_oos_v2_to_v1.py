# -*- coding: utf-8 -*-
"""
convert_gate_oos_v2_to_v1.py

把 v2 gate_oos_dev.parquet 转成 v1 build_position_curve_gate_ml_v1.py 期望的列：
- p_hat: 事件为正类的概率（用 v2 的 gate_prob_oos）
- gate_pos0_hat: 初始仓位建议（默认：weak=0.5, strong=1.0）
同时保留 symbol/event_id/t0_ts 等字段，便于 merge。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--thr_low", type=float, default=0.50)
    ap.add_argument("--thr_high", type=float, default=0.65)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)

    # v2 standard column
    if "gate_prob_oos" in df.columns:
        prob = df["gate_prob_oos"].astype(float)
    elif "gate_prob" in df.columns:
        prob = df["gate_prob"].astype(float)
    else:
        raise ValueError("Input does not contain gate_prob_oos or gate_prob.")

    out = df.copy()
    out["p_hat"] = prob

    # map to gate_pos0_hat (match your v1 sizing convention)
    # strong -> 1.0, weak -> 0.5, else 0.0
    out["gate_pos0_hat"] = np.where(
        out["p_hat"] >= args.thr_high, 1.0,
        np.where(out["p_hat"] >= args.thr_low, 0.5, 0.0)
    ).astype("float32")

    # ensure required keys exist
    for c in ["symbol", "event_id"]:
        if c not in out.columns:
            raise ValueError(f"Missing required key column: {c}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_path, index=False)

    print("[DONE] saved:", args.out_path)
    print("columns:", [c for c in ["symbol","event_id","p_hat","gate_pos0_hat"] if c in out.columns])


if __name__ == "__main__":
    main()
