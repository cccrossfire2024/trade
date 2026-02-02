# -*- coding: utf-8 -*-
"""
sanity_suite_gate_pipeline.py

综合泄露/非因果/错位验尸脚本：
- Baseline: 调用 run_all_experiments_gate_risk_v3.py 训练+推断+回测
- Shuffle: dev 内按 symbol 打乱 label（reg 则打乱 mfe_atr），再跑一次
- Zero-signal: gate_oos 全置零，直接跑执行器
- Shift-signal: gate_oos 按 t0_ts 错位 1 格（每 symbol 内 shift），再跑执行器
- 生成 sanity_summary.json + sanity_table.csv 方便你贴给我审计

依赖：pandas, numpy
（训练/推断/回测由你现有 v3 总控和执行器脚本完成）
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

EPS = 1e-12


def to_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str], log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
        p.wait()
        f.write(f"\nRET={p.returncode}\n")
    return p.returncode


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any):
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_gate_samples(gate_samples_path: Path) -> Dict[str, Any]:
    # 只读必要列，避免太慢
    cols_try = ["symbol", "event_id", "t0_ts", "label", "mfe_atr"]
    df = pd.read_parquet(gate_samples_path)
    for c in ["symbol", "event_id", "t0_ts"]:
        if c not in df.columns:
            raise ValueError(f"{gate_samples_path} missing required column: {c}")

    df["t0_ts"] = to_utc(df["t0_ts"])
    key = df[["symbol", "event_id"]]
    uniq = key.drop_duplicates().shape[0]
    dup_rate = 1.0 - (uniq / max(len(df), 1))

    res = {
        "path": str(gate_samples_path),
        "rows": int(len(df)),
        "unique_key": int(uniq),
        "dup_key_rate": float(dup_rate),
        "t0_min": str(df["t0_ts"].min()),
        "t0_max": str(df["t0_ts"].max()),
        "has_label": bool("label" in df.columns),
        "has_mfe_atr": bool("mfe_atr" in df.columns),
        "label_counts": df["label"].astype(str).value_counts().to_dict() if "label" in df.columns else None,
        "mfe_atr_desc": df["mfe_atr"].describe().to_dict() if "mfe_atr" in df.columns else None,
    }
    return res


def audit_gate_oos_merge(gate_oos_path: Path, gate_samples_path: Path) -> Dict[str, Any]:
    oos = pd.read_parquet(gate_oos_path)
    gs = pd.read_parquet(gate_samples_path, columns=["symbol", "event_id", "t0_ts"])

    for c in ["symbol", "event_id", "t0_ts", "p_hat", "gate_pos0_hat"]:
        if c not in oos.columns:
            raise ValueError(f"{gate_oos_path} missing required column: {c}")

    oos["t0_ts"] = to_utc(oos["t0_ts"])
    gs["t0_ts"] = to_utc(gs["t0_ts"])

    uniq = oos[["symbol", "event_id"]].drop_duplicates().shape[0]
    dup_rate = 1.0 - (uniq / max(len(oos), 1))

    m = oos.merge(gs, on=["symbol", "event_id"], how="left", suffixes=("", "_gs"))
    miss = float(m["t0_ts_gs"].isna().mean())
    # t0 是否一致（防错位）：绝对差 > 0 表示疑点（但注意时区/对齐）
    dt = (m["t0_ts"] - m["t0_ts_gs"]).abs()
    bad_dt = float((dt > pd.Timedelta(minutes=0)).mean()) if "t0_ts_gs" in m.columns else float("nan")

    res = {
        "gate_oos_path": str(gate_oos_path),
        "rows": int(len(oos)),
        "dup_key_rate": float(dup_rate),
        "merge_missing_rate": miss,
        "t0_mismatch_rate": bad_dt,
        "p_hat_nonzero_rate": float((oos["p_hat"].astype(float) != 0).mean()),
        "gate_pos_nonzero_rate": float((oos["gate_pos0_hat"].astype(float) != 0).mean()),
        "t0_min": str(oos["t0_ts"].min()),
        "t0_max": str(oos["t0_ts"].max()),
    }
    return res


def parse_exec_summary(summary_json: Path) -> Dict[str, Any]:
    s = read_json(summary_json)
    # 兼容：{"dev":{"overall_equal_weight":...},"blind":{...}}
    def pick_side(side: str):
        if side not in s:
            return None
        over = s[side].get("overall_equal_weight") or s[side].get("overall") or s[side]
        out = {}
        for k in ["sharpe_like", "max_drawdown", "calmar_like", "total_return", "turnover_per_bar", "avg_exposure", "fee_share"]:
            if isinstance(over, dict) and k in over:
                out[k] = over[k]
        return out

    return {
        "path": str(summary_json),
        "dev": pick_side("dev"),
        "blind": pick_side("blind"),
    }


def make_shuffled_dev(dev_path: Path, out_path: Path, modes: List[str], seed: int = 42) -> str:
    df = pd.read_parquet(dev_path)
    if "symbol" not in df.columns:
        raise ValueError("dev gate_samples missing symbol")

    rng = np.random.default_rng(seed)
    df2 = df.copy()

    # bin/multi: shuffle label；reg: shuffle mfe_atr
    if ("bin" in modes or "multi" in modes):
        if "label" not in df2.columns:
            raise ValueError("shuffle requested for bin/multi but dev gate_samples missing label")
        def shuf(s):
            a = s.to_numpy().copy()
            rng.shuffle(a)
            return a
        df2["label"] = df2.groupby("symbol")["label"].transform(shuf)

    if "reg" in modes:
        if "mfe_atr" not in df2.columns:
            raise ValueError("shuffle requested for reg but dev gate_samples missing mfe_atr")
        def shuf_num(s):
            a = s.to_numpy().copy()
            rng.shuffle(a)
            return a
        df2["mfe_atr"] = df2.groupby("symbol")["mfe_atr"].transform(shuf_num)

    ensure_dir(out_path.parent)
    df2.to_parquet(out_path, index=False)
    return str(out_path)


def make_gate_oos_zero(in_path: Path, out_path: Path):
    df = pd.read_parquet(in_path)
    for c in ["p_hat", "gate_pos0_hat"]:
        if c not in df.columns:
            raise ValueError(f"{in_path} missing {c}")
    df["p_hat"] = 0.0
    df["gate_pos0_hat"] = 0.0
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)


def make_gate_oos_shift(in_path: Path, out_path: Path):
    """
    在每个 symbol 内按 t0_ts 排序，将 (p_hat, gate_pos0_hat) 往后 shift 1 格。
    这是“错位”测试：如果 shift 后表现仍然很强，强烈怀疑执行器/merge/非因果。
    """
    df = pd.read_parquet(in_path)
    for c in ["symbol", "t0_ts", "p_hat", "gate_pos0_hat"]:
        if c not in df.columns:
            raise ValueError(f"{in_path} missing {c}")
    df["t0_ts"] = to_utc(df["t0_ts"])
    df = df.sort_values(["symbol", "t0_ts"]).reset_index(drop=True)

    df2 = df.copy()
    df2[["p_hat", "gate_pos0_hat"]] = (
        df2.groupby("symbol")[["p_hat", "gate_pos0_hat"]].shift(1).fillna(0.0)
    )
    ensure_dir(out_path.parent)
    df2.to_parquet(out_path, index=False)


def run_executor(executor_py: Path, out_dir: Path,
                 dev_gate_oos: Path, blind_gate_oos: Path,
                 dev_universe_long: Path, blind_universe_long: Path,
                 dev_risk_states: Path, blind_risk_states: Path,
                 log_path: Path) -> Path:
    ensure_dir(out_dir)
    cmd = [
        "python", str(executor_py),
        "--out_dir", str(out_dir),
        "--dev_gate_oos", str(dev_gate_oos),
        "--blind_gate_oos", str(blind_gate_oos),
        "--dev_universe_long", str(dev_universe_long),
        "--blind_universe_long", str(blind_universe_long),
        "--dev_risk_states", str(dev_risk_states),
        "--blind_risk_states", str(blind_risk_states),
    ]
    rc = run_cmd(cmd, log_path)
    if rc != 0:
        raise RuntimeError(f"executor failed (rc={rc}). see log: {log_path}")
    summary_json = out_dir / "summary_position_curve.json"
    if not summary_json.exists():
        raise FileNotFoundError(summary_json)
    return summary_json


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--v3_script", type=str, default="run_all_experiments_gate_risk_v3.py")
    ap.add_argument("--executor_py", type=str, default="build_position_curve_gate_oos_only_v1logic.py")

    ap.add_argument("--dev_gate_samples", type=str, default=r"datasets_v2plus\gate_samples_v2.parquet")
    ap.add_argument("--blind_gate_samples", type=str, default=r"datasets_blind_v1\step2_features_v2\gate_samples_v2.parquet")

    ap.add_argument("--dev_universe_long", type=str, default=r"data_clean\universe_long_dev.parquet")
    ap.add_argument("--blind_universe_long", type=str, default=r"data_clean\universe_long_blind.parquet")
    ap.add_argument("--dev_risk_states", type=str, default=r"datasets_v3\risk_states_v1.parquet")
    ap.add_argument("--blind_risk_states", type=str, default=r"datasets_blind_v1\step3_risk_states\risk_states_v1.parquet")

    ap.add_argument("--out_root", type=str, default="sanity_suite_out")
    ap.add_argument("--modes", nargs="+", default=["bin"])  # 默认先跑 bin，最快
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    v3_script = Path(args.v3_script)
    executor_py = Path(args.executor_py)

    dev_path = Path(args.dev_gate_samples)
    blind_path = Path(args.blind_gate_samples)

    # 0) 审计 gate_samples
    audit = {
        "dev_gate_samples": audit_gate_samples(dev_path),
        "blind_gate_samples": audit_gate_samples(blind_path),
    }
    write_json(out_root / "audit_gate_samples.json", audit)

    results_rows = []
    full = {"audit": audit, "runs": {}}

    # 1) BASELINE（调用 v3 总控）
    baseline_root = out_root / "baseline"
    ensure_dir(baseline_root)
    cmd = [
        "python", str(v3_script),
        "--dev_gate_samples", str(dev_path),
        "--blind_gate_samples", str(blind_path),
        "--out_root", str(baseline_root),
        "--modes",
    ] + list(args.modes)
    rc = run_cmd(cmd, out_root / "logs" / "baseline_v3.log")
    if rc != 0:
        raise RuntimeError(f"baseline v3 failed (rc={rc}). see log.")

    # baseline 读取每个 mode 的执行器 summary
    base_mode_summaries = {}
    for mode in args.modes:
        summ_path = baseline_root / f"gate_{mode}" / "exec" / "summary_position_curve.json"
        if not summ_path.exists():
            raise FileNotFoundError(summ_path)
        base_mode_summaries[mode] = parse_exec_summary(summ_path)

        # 同时做 merge 审计
        dev_oos = baseline_root / f"gate_{mode}" / "gate_oos_dev_for_v1.parquet"
        audit_merge = audit_gate_oos_merge(dev_oos, dev_path)
        base_mode_summaries[mode]["audit_merge_dev_oos"] = audit_merge

        # blind oos merge 审计（用 blind gate_samples）
        blind_oos = baseline_root / f"gate_{mode}" / "gate_oos_blind_for_v1.parquet"
        audit_merge_b = audit_gate_oos_merge(blind_oos, blind_path)
        base_mode_summaries[mode]["audit_merge_blind_oos"] = audit_merge_b

        # 记录表格行
        r = base_mode_summaries[mode]
        results_rows.append({
            "test": "baseline",
            "mode": mode,
            **{f"dev_{k}": v for k, v in (r["dev"] or {}).items()},
            **{f"blind_{k}": v for k, v in (r["blind"] or {}).items()},
            "dev_dup_key_rate": audit_merge["dup_key_rate"],
            "dev_merge_missing_rate": audit_merge["merge_missing_rate"],
            "dev_t0_mismatch_rate": audit_merge["t0_mismatch_rate"],
        })

    full["runs"]["baseline"] = base_mode_summaries

    # 2) SHUFFLE（dev 打乱标签/目标）
    shuffle_root = out_root / "shuffle"
    ensure_dir(shuffle_root)

    dev_shuf_path = out_root / "dev_gate_samples_SHUFFLE.parquet"
    make_shuffled_dev(dev_path, dev_shuf_path, modes=list(args.modes), seed=args.seed)

    cmd = [
        "python", str(v3_script),
        "--dev_gate_samples", str(dev_shuf_path),
        "--blind_gate_samples", str(blind_path),
        "--out_root", str(shuffle_root),
        "--modes",
    ] + list(args.modes)
    rc = run_cmd(cmd, out_root / "logs" / "shuffle_v3.log")
    if rc != 0:
        raise RuntimeError(f"shuffle v3 failed (rc={rc}). see log.")

    shuf_mode_summaries = {}
    for mode in args.modes:
        summ_path = shuffle_root / f"gate_{mode}" / "exec" / "summary_position_curve.json"
        shuf_mode_summaries[mode] = parse_exec_summary(summ_path)
        r = shuf_mode_summaries[mode]
        results_rows.append({
            "test": "shuffle",
            "mode": mode,
            **{f"dev_{k}": v for k, v in (r["dev"] or {}).items()},
            **{f"blind_{k}": v for k, v in (r["blind"] or {}).items()},
        })
    full["runs"]["shuffle"] = shuf_mode_summaries

    # 3) ZERO-SIGNAL（直接用 baseline 产物 gate_oos，置零跑执行器）
    zero_mode_summaries = {}
    for mode in args.modes:
        zroot = out_root / "zero" / mode
        ensure_dir(zroot)

        dev_oos_in = baseline_root / f"gate_{mode}" / "gate_oos_dev_for_v1.parquet"
        blind_oos_in = baseline_root / f"gate_{mode}" / "gate_oos_blind_for_v1.parquet"
        dev_oos_zero = zroot / "gate_oos_dev_ZERO.parquet"
        blind_oos_zero = zroot / "gate_oos_blind_ZERO.parquet"

        make_gate_oos_zero(dev_oos_in, dev_oos_zero)
        make_gate_oos_zero(blind_oos_in, blind_oos_zero)

        summ_json = run_executor(
            executor_py=executor_py,
            out_dir=zroot / "exec",
            dev_gate_oos=dev_oos_zero,
            blind_gate_oos=blind_oos_zero,
            dev_universe_long=Path(args.dev_universe_long),
            blind_universe_long=Path(args.blind_universe_long),
            dev_risk_states=Path(args.dev_risk_states),
            blind_risk_states=Path(args.blind_risk_states),
            log_path=out_root / "logs" / f"zero_exec_{mode}.log"
        )
        zero_mode_summaries[mode] = parse_exec_summary(summ_json)
        r = zero_mode_summaries[mode]
        results_rows.append({
            "test": "zero_signal",
            "mode": mode,
            **{f"dev_{k}": v for k, v in (r["dev"] or {}).items()},
            **{f"blind_{k}": v for k, v in (r["blind"] or {}).items()},
        })
    full["runs"]["zero_signal"] = zero_mode_summaries

    # 4) SHIFT-SIGNAL（错位1格再跑执行器）
    shift_mode_summaries = {}
    for mode in args.modes:
        sroot = out_root / "shift" / mode
        ensure_dir(sroot)

        dev_oos_in = baseline_root / f"gate_{mode}" / "gate_oos_dev_for_v1.parquet"
        blind_oos_in = baseline_root / f"gate_{mode}" / "gate_oos_blind_for_v1.parquet"
        dev_oos_shift = sroot / "gate_oos_dev_SHIFT.parquet"
        blind_oos_shift = sroot / "gate_oos_blind_SHIFT.parquet"

        make_gate_oos_shift(dev_oos_in, dev_oos_shift)
        make_gate_oos_shift(blind_oos_in, blind_oos_shift)

        summ_json = run_executor(
            executor_py=executor_py,
            out_dir=sroot / "exec",
            dev_gate_oos=dev_oos_shift,
            blind_gate_oos=blind_oos_shift,
            dev_universe_long=Path(args.dev_universe_long),
            blind_universe_long=Path(args.blind_universe_long),
            dev_risk_states=Path(args.dev_risk_states),
            blind_risk_states=Path(args.blind_risk_states),
            log_path=out_root / "logs" / f"shift_exec_{mode}.log"
        )
        shift_mode_summaries[mode] = parse_exec_summary(summ_json)
        r = shift_mode_summaries[mode]
        results_rows.append({
            "test": "shift_signal",
            "mode": mode,
            **{f"dev_{k}": v for k, v in (r["dev"] or {}).items()},
            **{f"blind_{k}": v for k, v in (r["blind"] or {}).items()},
        })
    full["runs"]["shift_signal"] = shift_mode_summaries

    # 输出总表
    table = pd.DataFrame(results_rows)
    table_path = out_root / "sanity_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    # 计算几个关键“判决信号”
    verdict = {}
    for mode in args.modes:
        # baseline vs shuffle: 如果 shuffle 仍然很强 -> 高度怀疑泄露/错位/非因果
        b = full["runs"]["baseline"][mode]["dev"] or {}
        sh = full["runs"]["shuffle"][mode]["dev"] or {}
        z = full["runs"]["zero_signal"][mode]["dev"] or {}
        sf = full["runs"]["shift_signal"][mode]["dev"] or {}
        verdict[mode] = {
            "baseline_dev_total_return": b.get("total_return", None),
            "shuffle_dev_total_return": sh.get("total_return", None),
            "zero_dev_total_return": z.get("total_return", None),
            "shift_dev_total_return": sf.get("total_return", None),
            "red_flag_shuffle_still_good": (sh.get("total_return", 0) is not None and sh.get("total_return", 0) > 1.0),
            "red_flag_zero_still_profit": (z.get("total_return", 0) is not None and z.get("total_return", 0) > 0.05),
            "red_flag_shift_still_good": (sf.get("total_return", 0) is not None and sf.get("total_return", 0) > 1.0),
        }

    full["verdict"] = verdict
    full["outputs"] = {
        "sanity_table": str(table_path),
        "sanity_summary": str(out_root / "sanity_summary.json"),
        "audit_gate_samples": str(out_root / "audit_gate_samples.json"),
        "logs_dir": str(out_root / "logs"),
    }

    write_json(out_root / "sanity_summary.json", full)
    print("[DONE] saved:", table_path)
    print("[DONE] saved:", out_root / "sanity_summary.json")


if __name__ == "__main__":
    main()
