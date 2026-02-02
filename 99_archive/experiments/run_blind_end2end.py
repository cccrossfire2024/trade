# -*- coding: utf-8 -*-
"""
run_blind_end2end.py

Blind end-to-end pipeline (2025-01-02 ~ 2026-01-30)

Inputs:
- data_clean/universe_long_blind.parquet
- (must already have your scripts in the same folder):
    extract_gmma_events.py
    build_gate_risk_features_v1.py
    risk_state_machine_v1.py
    build_position_curve_v0.py

Outputs under --out_root (default: datasets_blind_v1):
- step1_events/...
- step2_features/...
- step3_risk_states/...
- step4_position_curve/...
And prints final summary from step4_position_curve/summary_position_curve_v0.json

Run:
  python run_blind_end2end.py --out_root datasets_blind_v1
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None):
    print("\n[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd)
    if p.returncode != 0:
        raise SystemExit(f"[ERROR] command failed with return code {p.returncode}: {' '.join(cmd)}")


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="datasets_blind_v1")
    ap.add_argument("--blind_long", type=str, default="data_clean/universe_long_blind.parquet")

    # params consistent with dev run
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--gate_thr_big", type=float, default=6.0)
    ap.add_argument("--gate_thr_mid", type=float, default=3.0)
    ap.add_argument("--gate_thr_small", type=float, default=1.0)
    ap.add_argument("--twarn_lookback", type=int, default=256)

    ap.add_argument("--w_pre", type=int, default=16)
    ap.add_argument("--w_post", type=int, default=8)

    # risk state v1 params
    ap.add_argument("--cooldown_bars", type=int, default=32)

    # position curve params
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--pos_none", type=float, default=0.0)
    ap.add_argument("--pos_small", type=float, default=0.4)
    ap.add_argument("--pos_mid", type=float, default=0.7)
    ap.add_argument("--pos_big", type=float, default=1.0)
    ap.add_argument("--risk_pos_base", type=float, default=0.6)
    ap.add_argument("--pos_cap", type=float, default=1.0)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # step dirs
    d1 = out_root / "step1_events"
    d2 = out_root / "step2_features"
    d3 = out_root / "step3_risk_states"
    d4 = out_root / "step4_position_curve"
    for d in [d1, d2, d3, d4]:
        d.mkdir(parents=True, exist_ok=True)

    # ---------- Step 1: extract events ----------
    run_cmd([
        "python", "extract_gmma_events.py",
        "--input_long_dev", args.blind_long,    # script uses this arg name; fine for blind too
        "--out_dir", str(d1),
        "--atr_window", str(args.atr_window),
        "--gate_thr_big", str(args.gate_thr_big),
        "--gate_thr_mid", str(args.gate_thr_mid),
        "--gate_thr_small", str(args.gate_thr_small),
        "--twarn_lookback", str(args.twarn_lookback),
    ])

    # ---------- Step 2: build gate & risk features ----------
    run_cmd([
        "python", "build_gate_risk_features_v1.py",
        "--universe_long_dev", args.blind_long,      # same arg name, reuse
        "--events_dev", str(d1 / "gmma_events_dev.parquet"),
        "--risk_bars_dev", str(d1 / "gmma_risk_bars_dev.parquet"),
        "--out_dir", str(d2),
        "--w_pre", str(args.w_pre),
        "--w_post", str(args.w_post),
    ])

    # ---------- Step 3: risk state machine v1 ----------
    run_cmd([
        "python", "risk_state_machine_v1.py",
        "--in_path", str(d2 / "risk_bars_features_v1.parquet"),
        "--out_dir", str(d3),
        "--cooldown_bars", str(args.cooldown_bars),
    ])

    # ---------- Step 4: build position curve ----------
    run_cmd([
        "python", "build_position_curve_v0.py",
        "--universe_long_dev", args.blind_long,
        "--gate_samples", str(d2 / "gate_samples_v1.parquet"),
        "--risk_states", str(d3 / "risk_states_v1.parquet"),
        "--out_dir", str(d4),
        "--fee", str(args.fee),
        "--pos_none", str(args.pos_none),
        "--pos_small", str(args.pos_small),
        "--pos_mid", str(args.pos_mid),
        "--pos_big", str(args.pos_big),
        "--risk_pos_base", str(args.risk_pos_base),
        "--pos_cap", str(args.pos_cap),
    ])

    # ---------- Print final summary ----------
    summary_path = d4 / "summary_position_curve_v0.json"
    if summary_path.exists():
        s = read_json(summary_path)
        print("\n[BLIND DONE] Final summary:")
        print(json.dumps({
            "out_path": s.get("out_path"),
            "overall_equal_weight": s.get("overall_equal_weight"),
            "per_symbol": s.get("per_symbol")
        }, indent=2, ensure_ascii=False))
    else:
        print(f"[WARN] summary not found: {summary_path}")

    # quick pointer paths
    print("\n[PATHS]")
    print("events:", d1)
    print("features:", d2)
    print("risk_states:", d3)
    print("position_curve:", d4)


if __name__ == "__main__":
    main()
