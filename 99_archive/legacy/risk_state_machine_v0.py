# -*- coding: utf-8 -*-
"""
risk_state_machine_v0.py

目的：
- 基于 datasets_v2/risk_bars_features_v1.parquet
- 为事件内每个 bar 打“生命周期状态 state”
- 输出一个目标仓位 target_pos（0~1），并带 BREAKDOWN 冷却机制

状态（5类）：
- ADVANCE     推进：趋势扩张、效率高 -> 加仓
- EXPANSION   扩张：趋势仍强但波动放大 -> 持仓/小加
- DIGEST      消化：趋势还在但效率下降 -> 减仓到中性
- EXHAUST     衰竭：动能衰减 -> 持续减仓
- BREAKDOWN   崩坏：结构收敛/回撤风险尖刺 -> 清仓 + 冷却

完全 Anti-Drift：
- 阈值全部来自“事件内滚动分位数 / 过去窗口统计”，不使用固定绝对数值（除了 pos 映射参数）

运行：
  python risk_state_machine_v0.py --in_path datasets_v2\\risk_bars_features_v1.parquet --out_dir datasets_v3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rolling_q(s: pd.Series, win: int, q: float) -> pd.Series:
    return s.rolling(win, min_periods=win).quantile(q)


def safe_z(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return (s - mu) / (sd + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="datasets_v2/risk_bars_features_v1.parquet")
    ap.add_argument("--out_dir", type=str, default="datasets_v3")

    # rolling windows in bars (15m bars)
    ap.add_argument("--win_short", type=int, default=32)   # ~8 hours
    ap.add_argument("--win_mid", type=int, default=96)     # ~1 day
    ap.add_argument("--win_long", type=int, default=192)   # ~2 days

    # cooldown
    ap.add_argument("--cooldown_bars", type=int, default=32)

    # position mapping
    ap.add_argument("--pos_base", type=float, default=0.6)     # 默认中性仓位
    ap.add_argument("--pos_add", type=float, default=0.2)      # 每次“加仓状态”上调
    ap.add_argument("--pos_reduce", type=float, default=0.2)   # 每次“减仓状态”下调
    ap.add_argument("--pos_max", type=float, default=1.0)
    ap.add_argument("--pos_min", type=float, default=0.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.in_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 必需列检查
    required = [
        "symbol","event_id","ts","progress",
        "gmma_gap","gmma_slopeS","gmma_slopeS_acc",
        "er_16","vr_16_64",
        "rs_vol_rel","vov_96",
        "lr_16","wick_skew","body_eff"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 排序（按事件内时间）
    df = df.sort_values(["symbol","event_id","ts"]).reset_index(drop=True)

    # 事件内滚动统计（关键：只用过去数据，避免泄露）
    winS, winM, winL = args.win_short, args.win_mid, args.win_long

    # groupby event
    g = df.groupby(["symbol","event_id"], group_keys=False)

    # 结构：gap 的变化（收敛/扩张）
    df["gap_chg_1"] = g["gmma_gap"].diff(1)
    df["gap_chg_z"] = g["gap_chg_1"].apply(lambda s: safe_z(s, winS))

    # 推进：斜率 / 加速度 的事件内分位数
    df["slopeS_q70"] = g["gmma_slopeS"].apply(lambda s: rolling_q(s, winM, 0.70))
    df["slopeS_q30"] = g["gmma_slopeS"].apply(lambda s: rolling_q(s, winM, 0.30))

    df["acc_q30"] = g["gmma_slopeS_acc"].apply(lambda s: rolling_q(s, winM, 0.30))
    df["acc_q10"] = g["gmma_slopeS_acc"].apply(lambda s: rolling_q(s, winM, 0.10))

    # 效率：ER 分位数
    df["er_q70"] = g["er_16"].apply(lambda s: rolling_q(s, winM, 0.70))
    df["er_q40"] = g["er_16"].apply(lambda s: rolling_q(s, winM, 0.40))
    df["er_q20"] = g["er_16"].apply(lambda s: rolling_q(s, winM, 0.20))

    # 波动与风险尖刺：rs_vol_rel / vov_96 分位数
    df["rs_q70"] = g["rs_vol_rel"].apply(lambda s: rolling_q(s, winM, 0.70))
    df["rs_q90"] = g["rs_vol_rel"].apply(lambda s: rolling_q(s, winM, 0.90))

    df["vov_q70"] = g["vov_96"].apply(lambda s: rolling_q(s, winL, 0.70))
    df["vov_q90"] = g["vov_96"].apply(lambda s: rolling_q(s, winL, 0.90))

    # 回撤/动能恶化：lr_16 的事件内分位数 + 实体/影线质量
    df["lr16_q30"] = g["lr_16"].apply(lambda s: rolling_q(s, winM, 0.30))
    df["lr16_q10"] = g["lr_16"].apply(lambda s: rolling_q(s, winM, 0.10))

    df["body_q30"] = g["body_eff"].apply(lambda s: rolling_q(s, winM, 0.30))
    df["wick_q70"] = g["wick_skew"].apply(lambda s: rolling_q(s, winM, 0.70))  # 上影偏多偏正

    # 过滤掉滚动窗口还不够的行（否则阈值 NaN）
    need_cols = [
        "slopeS_q70","slopeS_q30","acc_q30","acc_q10",
        "er_q70","er_q40","er_q20",
        "rs_q70","rs_q90","vov_q70","vov_q90",
        "lr16_q30","lr16_q10","body_q30","wick_q70"
    ]
    df["ready"] = df[need_cols].notna().all(axis=1)

    # --------------------------
    # State rules (v0)
    # --------------------------
    # BREAKDOWN：结构快速收敛 + 风险尖刺 or 动能转负
    breakdown = (
        (df["gap_chg_z"] < -1.0) &
        (
            (df["vov_96"] > df["vov_q90"]) |
            (df["rs_vol_rel"] > df["rs_q90"]) |
            (df["lr_16"] < df["lr16_q10"])
        )
    )

    # EXHAUST：加速度显著为负 + 效率低 + candle 质量恶化 或 lr_16 走弱
    exhaust = (
        (df["gmma_slopeS_acc"] < df["acc_q10"]) &
        (df["er_16"] < df["er_q20"]) &
        (
            (df["body_eff"] < df["body_q30"]) |
            (df["lr_16"] < df["lr16_q30"]) |
            (df["wick_skew"] > df["wick_q70"])
        )
    )

    # ADVANCE：推进期（趋势扩张、效率高）
    advance = (
        (df["gmma_slopeS"] > df["slopeS_q70"]) &
        (df["er_16"] > df["er_q70"]) &
        (df["gap_chg_1"] > 0)
    )

    # EXPANSION：趋势仍强，但波动放大（主升段常见）
    expansion = (
        (df["gmma_slopeS"] > df["slopeS_q30"]) &
        (df["er_16"] > df["er_q40"]) &
        (
            (df["rs_vol_rel"] > df["rs_q70"]) |
            (df["vov_96"] > df["vov_q70"])
        )
    )

    # DIGEST：不满足推进/扩张，且没有进入衰竭/崩坏 => 消化
    # 我们后面用优先级覆盖即可

    # 初始化 state
    df["state"] = "DIGEST"

    # 优先级：BREAKDOWN > EXHAUST > ADVANCE > EXPANSION > DIGEST
    df.loc[expansion, "state"] = "EXPANSION"
    df.loc[advance, "state"] = "ADVANCE"
    df.loc[exhaust, "state"] = "EXHAUST"
    df.loc[breakdown, "state"] = "BREAKDOWN"

    # 只对 ready 行可信；不 ready 统一设为 DIGEST（保守）
    df.loc[~df["ready"], "state"] = "DIGEST"

    # --------------------------
    # Target position with cooldown
    # --------------------------
    # 目标：将 state 映射到目标仓位（0..1）
    # 先用静态映射，再加 BREAKDOWN 冷却。
    base = args.pos_base
    pos_map = {
        "ADVANCE":   min(args.pos_max, base + args.pos_add),      # 加仓
        "EXPANSION": min(args.pos_max, base + 0.5 * args.pos_add),# 小加/保持
        "DIGEST":    base,                                        # 中性
        "EXHAUST":   max(args.pos_min, base - args.pos_reduce),   # 减仓
        "BREAKDOWN": args.pos_min,                                # 清仓
    }
    df["target_pos_raw"] = df["state"].map(pos_map).astype(float)

    # 冷却：事件内一旦进入 BREAKDOWN，未来 cooldown_bars 内 target_pos 强制为 0
    cooldown = args.cooldown_bars

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

    df = df.groupby(["symbol","event_id"], group_keys=False).apply(apply_cooldown)

    # clamp
    df["target_pos"] = df["target_pos"].clip(args.pos_min, args.pos_max)

    # --------------------------
    # Save + summary
    # --------------------------
    out_path = out_dir / "risk_states_v0.parquet"
    df.to_parquet(out_path, index=False)

    # summary stats
    state_counts = df["state"].value_counts().to_dict()
    state_by_label = (
        df.groupby(["event_label","state"]).size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
        if "event_label" in df.columns else {}
    )

    summary = {
        "in_path": args.in_path,
        "out_path": str(out_path),
        "rows": int(len(df)),
        "state_counts": state_counts,
        "state_by_event_label": state_by_label,
        "windows": {"win_short": winS, "win_mid": winM, "win_long": winL},
        "cooldown_bars": cooldown,
        "pos_params": {
            "pos_base": args.pos_base,
            "pos_add": args.pos_add,
            "pos_reduce": args.pos_reduce,
            "pos_min": args.pos_min,
            "pos_max": args.pos_max
        }
    }
    (out_dir / "summary_risk_states_v0.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[DONE]", json.dumps({
        "rows": summary["rows"],
        "out_path": summary["out_path"],
        "state_counts": summary["state_counts"],
        "cooldown_bars": summary["cooldown_bars"]
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
