# -*- coding: utf-8 -*-
"""
build_position_curve_gate_oos_only_v1logic.py

可信的 v1 执行器（OOS-only）：
- 使用完整 universe_long 时间轴（每根 15m bar）
- risk_states 左连接到 bar（没有 state/event_id 的 bar 默认不持仓）
- gate_oos（v1兼容）提供每个 event_id 的 gate_pos0_hat / p_hat
- 状态机控制加减仓 + cooldown
- 费用按 turnover 计费

输出：
- position_curve_dev.parquet / position_curve_blind.parquet
- summary_position_curve.json（含 overall/per_symbol、turnover、avg_exposure、fee_share）
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12

def to_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def sharpe_like(lr_1):
    lr = np.asarray(lr_1, dtype=float)
    m = np.nanmean(lr)
    s = np.nanstd(lr)
    ann = np.sqrt(365 * 24 * 4)  # 15m -> 96 bars/day
    return float(m / (s + EPS) * ann)

def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + EPS) - 1.0
    return float(np.nanmin(dd))

def calmar_like(total_ret, mdd):
    return float((total_ret + EPS) / (abs(mdd) + EPS))

def read_universe_long(path: str):
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if df.index.name in ("ts", "timestamp"):
            df = df.reset_index().rename(columns={df.index.name: "ts"})
        else:
            raise ValueError("universe_long missing ts column")
    for c in ["symbol", "close"]:
        if c not in df.columns:
            raise ValueError(f"universe_long missing required col: {c}")
    df["ts"] = to_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values(["symbol","ts"]).reset_index(drop=True)
    return df[["symbol","ts","close"]]

def read_risk_states(path: str):
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if df.index.name in ("ts","timestamp"):
            df = df.reset_index().rename(columns={df.index.name:"ts"})
        else:
            raise ValueError("risk_states missing ts column")
    need = ["symbol","ts","event_id","state"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"risk_states missing required col: {c}")
    df["ts"] = to_utc(df["ts"])
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

def target_pos_from_state(base_pos: float, state: str,
                          mult_expansion: float,
                          mult_digest: float):
    if base_pos <= 0:
        return 0.0
    if state == "ADVANCE":
        return float(base_pos)
    if state == "EXPANSION":
        return float(min(1.0, base_pos * mult_expansion))
    if state == "DIGEST":
        return float(max(0.0, base_pos * mult_digest))
    # 风险/衰竭 -> 清仓
    if state in ("BREAKDOWN","EXHAUST"):
        return 0.0
    # 其它/未知 -> 不交易
    return 0.0

def build_curve_one_symbol(px: pd.DataFrame,
                           rs: pd.DataFrame,
                           gate: pd.DataFrame,
                           fee: float,
                           cooldown_bars: int,
                           k_smooth: float,
                           mult_expansion: float,
                           mult_digest: float):
    """
    px:  [ts, close] 全bar
    rs:  [ts, event_id, state] 子集（事件bar）
    gate:[event_id, p_hat, gate_pos0_hat] 事件级
    """
    df = px.copy().sort_values("ts").reset_index(drop=True)

    # 计算 bar log-return
    df["lr_1"] = np.log(df["close"] / df["close"].shift(1)).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # 把 risk_states 左连接到全bar
    rs2 = rs.copy()
    rs2 = rs2[["ts","event_id","state"]]
    df = df.merge(rs2, on="ts", how="left")

    # 把 gate_oos 左连接到 event_id
    if len(gate) > 0:
        df = df.merge(gate[["event_id","p_hat","gate_pos0_hat"]], on="event_id", how="left")
    else:
        df["p_hat"] = np.nan
        df["gate_pos0_hat"] = np.nan

    df["p_hat"] = df["p_hat"].fillna(0.0).astype(float)
    df["gate_pos0_hat"] = df["gate_pos0_hat"].fillna(0.0).astype(float)
    df["state"] = df["state"].fillna("NONE").astype(str)

    n = len(df)
    pos = np.zeros(n, dtype=float)
    turnover = np.zeros(n, dtype=float)
    fee_paid = np.zeros(n, dtype=float)

    # cooldown 计数器（>0 表示禁止入场）
    cd = 0
    last_event = None

    for i in range(1, n):
        state = df.at[i, "state"]
        eid = df.at[i, "event_id"]
        base = float(df.at[i, "gate_pos0_hat"])

        in_event = pd.notna(eid) and (state != "NONE")

        # 更新 cooldown
        if cd > 0:
            cd -= 1

        # 如果不在事件中：目标仓位=0
        if not in_event:
            tgt = 0.0
        else:
            # 切换到新 event：如果 cooldown 中则禁止入场
            if (last_event is None) or (eid != last_event):
                last_event = eid
                if cd > 0:
                    base = 0.0  # cooldown 禁止入场

            # 状态机给目标仓位
            tgt = target_pos_from_state(base, state, mult_expansion, mult_digest)

            # 如果处在 BREAKDOWN/EXHAUST，触发 cooldown
            if state in ("BREAKDOWN","EXHAUST"):
                cd = max(cd, cooldown_bars)
                tgt = 0.0

            # 再加一道 gate 过滤：base==0 就不交易
            if base <= 0.0:
                tgt = 0.0

        # 平滑调仓（降低抖动与费损）；k_smooth=1 则为直接到目标
        pos[i] = pos[i-1] + k_smooth * (tgt - pos[i-1])

        # 费用：按仓位变化
        turnover[i] = abs(pos[i] - pos[i-1])
        fee_paid[i] = turnover[i] * fee

    strat_lr = pos * df["lr_1"].values - fee_paid
    equity = np.exp(np.nancumsum(strat_lr))

    out = pd.DataFrame({
        "ts": df["ts"].values,
        "pos": pos.astype("float32"),
        "turnover": turnover.astype("float32"),
        "fee": fee_paid.astype("float32"),
        "lr_1": df["lr_1"].astype("float32"),
        "strat_lr": strat_lr.astype("float32"),
        "equity": equity.astype("float64"),
        "state": df["state"].values,
        "event_id": df["event_id"].values,
        "p_hat": df["p_hat"].astype("float32"),
        "gate_pos0_hat": df["gate_pos0_hat"].astype("float32"),
    })
    return out

def summarize_symbol(curve: pd.DataFrame):
    lr = curve["strat_lr"].astype(float).values
    eq = curve["equity"].astype(float).values
    mdd = max_drawdown(eq)
    tot = float(eq[-1] - 1.0)
    turnover = curve["turnover"].astype(float).values
    fee = curve["fee"].astype(float).values
    gross = curve["pos"].astype(float).values * curve["lr_1"].astype(float).values
    gross_sum = float(np.nansum(gross))
    fee_sum = float(np.nansum(fee))
    fee_share = float(fee_sum / (abs(gross_sum) + fee_sum + EPS))
    return {
        "rows": int(len(curve)),
        "sharpe_like": sharpe_like(lr),
        "max_drawdown": mdd,
        "calmar_like": calmar_like(tot, mdd),
        "turnover_per_bar": float(np.nanmean(turnover)),
        "avg_exposure": float(np.nanmean(curve["pos"].astype(float).values)),
        "fee_share": fee_share,
        "total_return": tot,
    }

def summarize_portfolio(curves_by_symbol: dict):
    # 对齐时间戳做等权（缺失用0收益）
    series = []
    for sym, c in curves_by_symbol.items():
        s = pd.Series(c["strat_lr"].astype(float).values, index=c["ts"].values, name=sym)
        series.append(s)
    lr_df = pd.concat(series, axis=1).fillna(0.0)
    lr_eq = lr_df.mean(axis=1).values
    eq = np.exp(np.nancumsum(lr_eq))
    mdd = max_drawdown(eq)
    tot = float(eq[-1] - 1.0)
    return {
        "sharpe_like": sharpe_like(lr_eq),
        "max_drawdown": mdd,
        "calmar_like": calmar_like(tot, mdd),
        "total_return": tot,
    }

def run_split(name: str,
              universe_long: str,
              risk_states: str,
              gate_oos: str,
              out_dir: Path,
              fee: float,
              cooldown_bars: int,
              k_smooth: float,
              mult_expansion: float,
              mult_digest: float):
    px = read_universe_long(universe_long)
    rs = read_risk_states(risk_states)
    gate = read_gate_oos(gate_oos)

    syms = sorted(px["symbol"].unique().tolist())
    curves = {}
    per_sym = {}

    for sym in syms:
        px_s = px[px["symbol"] == sym][["ts","close"]].copy()
        rs_s = rs[rs["symbol"] == sym][["ts","event_id","state"]].copy()
        gate_s = gate[gate["symbol"] == sym][["event_id","p_hat","gate_pos0_hat"]].copy()

        c = build_curve_one_symbol(
            px_s, rs_s, gate_s,
            fee=fee,
            cooldown_bars=cooldown_bars,
            k_smooth=k_smooth,
            mult_expansion=mult_expansion,
            mult_digest=mult_digest
        )
        c.insert(0, "symbol", sym)
        curves[sym] = c
        per_sym[sym] = summarize_symbol(c)

    overall = summarize_portfolio(curves)
    curve_all = pd.concat(curves.values(), ignore_index=True).sort_values(["ts","symbol"]).reset_index(drop=True)

    out_path = out_dir / f"position_curve_{name}.parquet"
    curve_all.to_parquet(out_path, index=False)

    return out_path, per_sym, overall, int(len(curve_all))

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

    # EXEC PARAMS
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--cooldown_bars", type=int, default=32)
    ap.add_argument("--k_smooth", type=float, default=0.35)  # 0~1, 越大越快调仓
    ap.add_argument("--mult_expansion", type=float, default=1.5)
    ap.add_argument("--mult_digest", type=float, default=0.5)

    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_path, dev_per, dev_overall, dev_rows = run_split(
        "dev", args.dev_universe_long, args.dev_risk_states, args.dev_gate_oos,
        out_dir, args.fee, args.cooldown_bars, args.k_smooth, args.mult_expansion, args.mult_digest
    )
    print("[DEV DONE]", dev_overall)

    blind_path, blind_per, blind_overall, blind_rows = run_split(
        "blind", args.blind_universe_long, args.blind_risk_states, args.blind_gate_oos,
        out_dir, args.fee, args.cooldown_bars, args.k_smooth, args.mult_expansion, args.mult_digest
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
        "exec_params": {
            "fee": args.fee,
            "cooldown_bars": args.cooldown_bars,
            "k_smooth": args.k_smooth,
            "mult_expansion": args.mult_expansion,
            "mult_digest": args.mult_digest
        }
    }

    summary_path = out_dir / "summary_position_curve.json"
    pd.Series(summary).to_json(summary_path, force_ascii=False, indent=2)
    print("[DONE] saved:", summary_path)

if __name__ == "__main__":
    main()
