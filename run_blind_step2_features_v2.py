# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path

def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, shell=False)

def main():
    # === 你只需要确认这几个路径 ===
    universe_blind = r"data_clean\universe_long_blind.parquet"

    # 如果你已经有 step1_events 的结果，把这两个填对就行
    events_blind = r"datasets_blind_v1\step1_events\gmma_events_blind.parquet"
    risk_bars_blind = r"datasets_blind_v1\step1_events\gmma_risk_bars_blind.parquet"

    out_dir = r"datasets_blind_v1\step2_features_v2"

    # === 下面自动检查 & 运行 ===
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not Path(universe_blind).exists():
        raise FileNotFoundError(universe_blind)

    if not Path(events_blind).exists() or not Path(risk_bars_blind).exists():
        print("[WARN] step1_events not found, try to generate via extract_gmma_events.py ...")
        Path(r"datasets_blind_v1\step1_events").mkdir(parents=True, exist_ok=True)
        run([
            "python", "extract_gmma_events.py",
            "--input_long_dev", universe_blind,
            "--out_dir", r"datasets_blind_v1\step1_events"
        ])

    # Build v2 features for blind
    run([
        "python", "build_gate_risk_features_v2.py",
        "--universe_long_dev", universe_blind,
        "--events_dev", events_blind,
        "--risk_bars_dev", risk_bars_blind,
        "--out_dir", out_dir
    ])

    print("\n[DONE] blind v2 features saved to:", out_dir)

if __name__ == "__main__":
    main()
