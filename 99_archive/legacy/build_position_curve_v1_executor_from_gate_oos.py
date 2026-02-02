# -*- coding: utf-8 -*-
"""
build_position_curve_v1_executor_from_gate_oos.py

目标：
- 不依赖 models_dir、不加载逐窗模型
- 直接用 gate_oos（v1兼容：symbol,event_id,p_hat,gate_pos0_hat）来驱动 v1 执行器逻辑
- 读取 risk_states_v1（每条bar带 state）
- 输出 dev / blind 的 position_curve parquet + summary json

注意：
这是“执行器复刻版”，逻辑尽量贴近你现在 pipeline 的结构：
- 只做多头
- 以 risk_state 作为持仓状态机调仓（ADVANCE/EXPANSION/DIGEST/BREAKDOWN/EXHAUST）
- gate_pos0_hat 作为起始仓位强度（0/0.5/1.0）
- 费用用每次仓位变化的 turnover * fee_rate（默认 4bp）
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FEE_DEFAULT = 0.0004  # 4bp
EPS = 1e-12

# 你现在 risk_states 的状态名（来自你 summary）
RISK_STATES = ["DIGEST", "ADVANCE", "EXPANSION", "BREAKDOWN", "EXHAUST"]

def to_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def sharpe_like(ret):
    ret = np.asarray(ret, dtype=float)
    m = np.nanmean(ret)
    s = np.nanstd(ret)
    return float(m / (s + EPS) * np.sqrt(365 * 24 * 4))  # 15m -> 96 bars/day

def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + EPS) - 1.0
    return float(np.nanmin(dd))

def calmar_like(total_ret, mdd):
    # total_ret as simple return, mdd negative
    return float((total_ret + EPS) / (abs(mdd) + EPS))

def read_long(universe_long_path: str):
    df = pd.read_parquet(universe_long_path)
    # 兼容列名：有的用 ts，有的 index
    if "ts" not in df.columns:
        if df.index.name in ("ts", "timestamp"):
            df = df.reset_index().rename(columns={df.index.name: "ts"})
        else:
            raise ValueError("universe_long missing ts column")
    df["ts"] = to_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df

def read_risk_states(path: str):
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if df.index.name in ("ts","timestamp"):
            df = df.reset_index().rename(columns={df.index.name:"ts"})
        else:
            raise ValueError("risk_states missing ts column")
    df["ts"] = to_utc(df["ts"])
    # 需要的 key 列
    need = ["symbol", "event_id", "ts", "state"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"risk_states missing required col: {c}")
    df = df.dropna(subset=["ts"]).sort_values(["symbol","ts"]).reset_index(drop=True)
    return df[need]

def read_gate_oos(path: str):
    df = pd.read_parquet(path)
    need = ["symbol","event_id","p_hat","gate_pos0_hat"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"gate_oos missing required col: {c}")
    df = df[need].copy()
    df["p_hat"] = pd.to_numeric(df["p_hat"], errors="coerce").fillna(0.0).astype(float)
    df["gate_pos0_hat"] = pd.to_numeric(df["gate_pos0_hat"], errors="coerce").fillna(0.0).astype(float)
    return df

def build_positions_one_symbol(panel_sym: pd.DataFrame, risk_sym: pd.DataFrame, gate_sym: pd.DataFrame,
                               fee_rate: float,
                               k_smooth: float = 0.5):
    """
    输入：
    - panel_sym: [symbol,ts,close]（用于算收益）
    - risk_sym:  [symbol,event_id,ts,state]（每bar状态）
    - gate_sym:  [symbol,event_id,p_hat,gate_pos0_hat]（事件级起仓）
    输出：
    - df curve per symbol: ts, symbol, ret, pos, turnover, fee
    """
    # 合并 risk_states -> gate 起仓强度（按 event_id）
    rs = risk_sym.merge(gate_sym, on=["symbol","event_id"], how="left")
    rs["gate_pos0_hat"] = rs["gate_pos0_hat"].fillna(0.0)
    rs["p_hat"] = rs["p_hat"].fillna(0.0)

    # 与价格对齐
    # panel 只取 close
    px = panel_sym[["ts","close"]].copy().sort_values("ts")
    rs = rs.sort_values("ts")

    # inner join：只在 risk_states 有定义的bar上持仓（与你现有 risk_bars 对齐）
    df = rs.merge(px, on="ts", how="inner")
    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) < 10:
        return None

    # 计算 15m log return
    df["lr_1"] = np.log(df["close"] / df["close"].shift(1)).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # 状态 -> 目标仓位映射（你现在 v1 执行器的近似版本）
    # - ADVANCE: 跟随上升，起仓=gate_pos0_hat
    # - EXPANSION: 加仓（放大 gate_pos0_hat）
    # - DIGEST: 持有/降杠杆
    # - EXHAUST/BREAKDOWN: 退出
    base = df["gate_pos0_hat"].astype(float).values
    st = df["state"].astype(str).values

    tgt = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        if st[i] == "ADVANCE":
            tgt[i] = base[i]
        elif st[i] == "EXPANSION":
            tgt[i] = min(1.0, base[i] * 1.5)  # 扩张期加仓
        elif st[i] == "DIGEST":
            tgt[i] = base[i] * 0.5            # 消化期降杠杆
        elif st[i] in ("BREAKDOWN","EXHAUST"):
            tgt[i] = 0.0
        else:
            tgt[i] = 0.0

    # 平滑调仓（避免抖动）
    pos = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        pos[i] = pos[i-1] + k_smooth * (tgt[i] - pos[i-1])

    # turnover & fee
    turnover = np.abs(np.diff(pos, prepend=0.0))
    fee = turnover * fee_rate

    # 策略 bar 收益（近似：pos * lr - fee）
    strat_lr = pos * df["lr_1"].values - fee
    equity = np.exp(np.nancumsum(strat_lr))

    out = pd.DataFrame({
        "ts": df["ts"].values,
        "symbol": df["symbol"].values,
        "pos": pos.astype("float32"),
        "turnover": turnover.astype("float32"),
        "fee": fee.astype("float32"),
        "strat_lr": strat_lr.astype("float32"),
        "equity": equity.astype("float64"),
    })
    return out

def summarize_curve(curve: pd.DataFrame):
    # curve: concat all symbols with strat_lr & equity per symbol
    # overall equal-weight: 平均各symbol的 bar lr
    syms = sorted(curve["symbol"].unique().tolist())
    per = {}
    # 构建每个 symbol 的等权曲线
    all_lr = []
    all_ts = None
    for s in syms:
        sub = curve[curve["symbol"] == s].sort_values("ts")
        lr = sub["strat_lr"].astype(float).values
        eq = np.exp(np.nancumsum(lr))
        mdd = max_drawdown(eq)
        tot = float(eq[-1] - 1.0)
        per[s] = {
            "rows": int(len(sub)),
            "sharpe_like": sharpe_like(lr),
            "max_drawdown": mdd,
            "calmar_like": calmar_like(tot, mdd),
            "turnover_per_bar": float(np.nanmean(sub["turnover"].astype(float).values)),
            "total_return": tot,
        }
        all_lr.append(pd.Series(lr, index=sub["ts"].values))

    # 对齐 ts 后等权平均
    lr_df = pd.concat(all_lr, axis=1).fillna(0.0)
    lr_eq = lr_df.mean(axis=1).values
    eq = np.exp(np.nancumsum(lr_eq))
    mdd = max_drawdown(eq)
    tot = float(eq[-1] - 1.0)
    overall = {
        "sharpe_like": sharpe_like(lr_eq),
        "max_drawdown": mdd,
        "calmar_like": calmar_like(tot, mdd),
        "total_return": tot,
    }
    return per, overall

def run_split(name: str, universe_long: str, risk_states: str, gate_oos: str, out_dir: Path, fee: float):
    panel = read_long(universe_long)
    risk = read_risk_states(risk_states)
    gate = read_gate_oos(gate_oos)

    syms = sorted(panel["symbol"].unique().tolist())
    curves = []
    for s in syms:
        p = panel[panel["symbol"] == s].copy()
        r = risk[risk["symbol"] == s].copy()
        g = gate[gate["symbol"] == s].copy()
        c = build_positions_one_symbol(p, r, g, fee_rate=fee)
        if c is not None and len(c) > 0:
            curves.append(c)

    if not curves:
        raise RuntimeError(f"No curves produced for {name}")

    curve = pd.concat(curves, ignore_index=True).sort_values(["ts","symbol"]).reset_index(drop=True)
    per, overall = summarize_curve(curve)

    out_path = out_dir / f"position_curve_{name}.parquet"
    curve.to_parquet(out_path, index=False)

    return out_path, per, overall, int(len(curve))

def main():
    ap = argparse.ArgumentParser()

    # DEV
    ap.add_argument("--dev_universe_long", type=str, default="data_clean/universe_long_dev.parquet")
    ap.add_argument("--dev_risk_states", type=str, default="datasets_v3/risk_states_v1.parquet")
    ap.add_argument("--dev_gate_oos", type=str, required=True)

    # BLIND
    ap.add_argument("--blind_universe_long", type=str, default="data_clean/universe_long_blind.parquet")
    ap.add_argument("--blind_risk_states", type=str, default="datasets_blind_v1/step3_risk_states/risk_states_v1.parquet")
    ap.add_argument("--blind_gate_oos", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fee", type=float, default=FEE_DEFAULT)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_path, dev_per, dev_overall, dev_rows = run_split(
        "dev", args.dev_universe_long, args.dev_risk_states, args.dev_gate_oos, out_dir, args.fee
    )
    print("[DEV DONE]", dev_overall)

    blind_path, blind_per, blind_overall, blind_rows = run_split(
        "blind", args.blind_universe_long, args.blind_risk_states, args.blind_gate_oos, out_dir, args.fee
    )
    print("[BLIND DONE]", blind_overall)

    summary = {
        "dev": {
            "out_path": str(dev_path),
            "overall_equal_weight": dev_overall,
            "per_symbol": dev_per,
            "rows": dev_rows
        },
        "blind": {
            "out_path": str(blind_path),
            "overall_equal_weight": blind_overall,
            "per_symbol": blind_per,
            "rows": blind_rows
        },
        "fee": args.fee
    }

    summary_path = out_dir / "summary_position_curve.json"
    pd.Series(summary).to_json(summary_path, force_ascii=False, indent=2)
    print("[DONE] saved:", summary_path)

if __name__ == "__main__":
    main()
