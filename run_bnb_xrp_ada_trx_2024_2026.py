# -*- coding: utf-8 -*-
"""
run_bnb_xrp_ada_trx_2024_2026.py

一键测试新币种（BNB/XRP/ADA/TRX）在 2024-01-01 ~ 2026-01-30 的表现：
- 下载 Binance Futures 15m OHLCV（带 warmup）
- 构建 universe_long
- 跑 GMMA event / features / risk states v1
- 用已训练 Gate（二分类）最后一个窗口模型做推断
- 生成 position curve + summary（仅统计评估区间）

依赖：
  pip install ccxt pyarrow pandas numpy lightgbm

要求：同目录存在你的脚本
  - extract_gmma_events.py
  - build_gate_risk_features_v1.py
  - risk_state_machine_v1.py
以及已训练模型目录：
  - models_gate_v1/models/lgbm_gate_window_*.txt

运行：
  python run_bnb_xrp_ada_trx_2024_2026.py --out_root test_bnb_xrp_ada_trx_24_26
"""

import argparse
import json
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
import lightgbm as lgb


SYMBOLS = ["BNB/USDT", "XRP/USDT", "ADA/USDT", "TRX/USDT"]
TIMEFRAME = "15m"
FREQ = "15min"


def get_exchange():
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    return ex


def fetch_ohlcv_all(exchange, symbol, timeframe, start_iso):
    since = exchange.parse8601(start_iso)
    out = []
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            last = batch[-1][0]
            out.extend(batch)
            if last <= since:
                break
            since = last + 1
            # stop when near now
            if last >= exchange.milliseconds() - 15 * 60 * 1000:
                break
        except Exception as e:
            print(f"[WARN] fetch error {symbol}: {e} -> sleep 5s")
            time.sleep(5)
    return out


def calc_atr(df, window=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    return atr


def to_15m_panel(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample(FREQ).ffill()
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype("float32")
    df["atr"] = calc_atr(df, window=14).astype("float32")
    df = df.dropna()
    df = df.reset_index()
    return df


def run_cmd(cmd):
    print("\n[CMD]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(f"[ERROR] command failed: {' '.join(cmd)}")


def ann_factor_15m():
    return np.sqrt(96 * 365)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())


def calmar_like(equity: pd.Series) -> float:
    if len(equity) < 10:
        return float("nan")
    total_ret = float(equity.iloc[-1] / (equity.iloc[0] + 1e-12) - 1.0)
    years = len(equity) / (96 * 365)
    if years <= 0:
        return float("nan")
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-12:
        return float("nan")
    return float(cagr / mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="test_bnb_xrp_ada_trx_24_26")
    ap.add_argument("--data_raw_dir", type=str, default="data_raw_ext")
    ap.add_argument("--models_dir", type=str, default="models_gate_v1/models")

    # warmup + eval range
    ap.add_argument("--fetch_start", type=str, default="2023-09-01 00:00:00")
    ap.add_argument("--eval_start", type=str, default="2024-01-01 00:00:00")
    ap.add_argument("--eval_end", type=str, default="2026-01-30 23:45:00")

    # backtest fee
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--risk_pos_base", type=float, default=0.6)
    ap.add_argument("--pos_cap", type=float, default=1.0)
    ap.add_argument("--p_enter", type=float, default=0.55)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_raw_dir = Path(args.data_raw_dir)
    data_raw_dir.mkdir(parents=True, exist_ok=True)

    # -------- Step 0: fetch & save parquet per symbol --------
    ex = get_exchange()
    saved = []
    for sym in SYMBOLS:
        safe = sym.replace("/","")
        p = data_raw_dir / f"{safe}_{TIMEFRAME}.parquet"
        if p.exists():
            print(f"[SKIP] exists: {p}")
            saved.append((sym, p))
            continue
        print(f"\n=== Fetch {sym} {TIMEFRAME} (futures) ===")
        raw = fetch_ohlcv_all(ex, sym, TIMEFRAME, args.fetch_start.replace(" ", "T") + "Z")
        df = to_15m_panel(raw)
        df.to_parquet(p, index=False)
        print(f"[OK] saved: {p} rows={len(df)}")
        saved.append((sym, p))

    # -------- Step 1: build universe_long --------
    frames = []
    for sym, p in saved:
        safe = sym.replace("/","")
        df = pd.read_parquet(p)
        df["symbol"] = safe
        frames.append(df)

    uni = pd.concat(frames, ignore_index=True)
    uni = uni.rename(columns={"ts": "ts"})
    # ensure types
    uni["ts"] = pd.to_datetime(uni["ts"], utc=True)
    uni = uni.sort_values(["symbol","ts"]).reset_index(drop=True)

    universe_long_path = out_root / "universe_long.parquet"
    uni.to_parquet(universe_long_path, index=False)
    print(f"[OK] universe_long: {universe_long_path} rows={len(uni)}")

    # -------- Step 2: run your existing pipeline scripts --------
    step1 = out_root / "step1_events"
    step2 = out_root / "step2_features"
    step3 = out_root / "step3_risk_states"
    step4 = out_root / "step4_position"
    for d in [step1, step2, step3, step4]:
        d.mkdir(parents=True, exist_ok=True)

    run_cmd([
        "python", "extract_gmma_events.py",
        "--input_long_dev", str(universe_long_path),
        "--out_dir", str(step1)
    ])

    run_cmd([
        "python", "build_gate_risk_features_v1.py",
        "--universe_long_dev", str(universe_long_path),
        "--events_dev", str(step1 / "gmma_events_dev.parquet"),
        "--risk_bars_dev", str(step1 / "gmma_risk_bars_dev.parquet"),
        "--out_dir", str(step2)
    ])

    run_cmd([
        "python", "risk_state_machine_v1.py",
        "--in_path", str(step2 / "risk_bars_features_v1.parquet"),
        "--out_dir", str(step3)
    ])

    # -------- Step 3: Gate inference with last binary model --------
    model_files = sorted(Path(args.models_dir).glob("lgbm_gate_window_*.txt"))
    if not model_files:
        raise FileNotFoundError(f"No models found in {args.models_dir}")
    last_model = model_files[-1]
    booster = lgb.Booster(model_file=str(last_model))
    feat_need = booster.feature_name()

    gate_samples = pd.read_parquet(step2 / "gate_samples_v1.parquet")
    # align features: missing -> 0, extra -> drop, order -> model feature order
    X = gate_samples.copy()
    for c in feat_need:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_need]
    p_hat = booster.predict(X, num_iteration=booster.best_iteration)
    p_hat = np.asarray(p_hat, dtype=float)

    gate_pos0_hat = (p_hat - args.p_enter) / (1.0 - args.p_enter)
    gate_pos0_hat = np.clip(gate_pos0_hat, 0.0, 1.0)

    gate_event = gate_samples[["symbol","event_id"]].copy()
    gate_event["p_hat"] = p_hat
    gate_event["gate_pos0_hat"] = gate_pos0_hat

    # -------- Step 4: Build position curve (like build_position_curve_gate_ml_v1) --------
    risk_states = pd.read_parquet(step3 / "risk_states_v1.parquet")
    risk_states["ts"] = pd.to_datetime(risk_states["ts"], utc=True)

    # risk mult
    risk_states["risk_mult"] = (risk_states["target_pos"] / (args.risk_pos_base + 1e-12)).clip(
        0.0, args.pos_cap/(args.risk_pos_base+1e-12)
    )

    m = risk_states.merge(gate_event, on=["symbol","event_id"], how="left")
    m["gate_pos0_hat"] = m["gate_pos0_hat"].fillna(0.0)
    m["p_hat"] = m["p_hat"].fillna(0.0)
    m["pos"] = (m["gate_pos0_hat"] * m["risk_mult"]).clip(0.0, args.pos_cap)

    base = uni[["ts","symbol","close"]].copy()
    base = base.merge(
        m[["symbol","ts","event_id","state","target_pos","risk_mult","gate_pos0_hat","p_hat","pos"]],
        on=["symbol","ts"], how="left"
    )
    base["pos"] = base["pos"].fillna(0.0)
    base = base.sort_values(["symbol","ts"]).reset_index(drop=True)

    # backtest
    g = base.groupby("symbol", group_keys=False)
    base["ret1"] = g["close"].pct_change().fillna(0.0)
    base["pos_prev"] = g["pos"].shift(1).fillna(0.0)
    base["dpos"] = (base["pos"] - base["pos_prev"]).abs()
    base["pnl"] = base["pos_prev"] * base["ret1"] - args.fee * base["dpos"]
    base["equity"] = g["pnl"].cumsum() + 1.0

    # -------- Step 5: evaluate only 2024-01-01 ~ 2026-01-30 --------
    eval_start = pd.to_datetime(args.eval_start, utc=True)
    eval_end = pd.to_datetime(args.eval_end, utc=True)
    ev = base[(base["ts"] >= eval_start) & (base["ts"] <= eval_end)].copy()

    per_symbol = {}
    for sym, sub in ev.groupby("symbol"):
        pnl = sub["pnl"]
        mu, sd = float(pnl.mean()), float(pnl.std())
        sharpe = (mu / (sd + 1e-12)) * ann_factor_15m()
        mdd = max_drawdown(sub["equity"])
        cal = calmar_like(sub["equity"])
        per_symbol[sym] = {
            "rows": int(len(sub)),
            "sharpe_like": sharpe,
            "max_drawdown": mdd,
            "calmar_like": cal,
            "turnover_per_bar": float(sub["dpos"].mean()),
            "total_return": float(sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0),
        }

    pivot = ev.pivot(index="ts", columns="symbol", values="pnl").fillna(0.0)
    pnl_all = pivot.mean(axis=1)
    eq_all = pnl_all.cumsum() + 1.0
    mu, sd = float(pnl_all.mean()), float(pnl_all.std())
    overall = {
        "sharpe_like": (mu / (sd + 1e-12)) * ann_factor_15m(),
        "max_drawdown": max_drawdown(eq_all),
        "calmar_like": calmar_like(eq_all),
        "total_return": float(eq_all.iloc[-1] / eq_all.iloc[0] - 1.0),
    }

    out_curve = step4 / "position_curve_eval.parquet"
    ev.to_parquet(out_curve, index=False)

    summary = {
        "symbols": [s.replace("/","") for s in SYMBOLS],
        "timeframe": TIMEFRAME,
        "eval_range": {"start": str(eval_start), "end": str(eval_end)},
        "fee": args.fee,
        "model_used": last_model.name,
        "overall_equal_weight": overall,
        "per_symbol": per_symbol,
        "paths": {
            "universe_long": str(universe_long_path),
            "gate_samples": str(step2 / "gate_samples_v1.parquet"),
            "risk_states": str(step3 / "risk_states_v1.parquet"),
            "position_curve_eval": str(out_curve),
        }
    }
    (step4 / "summary_eval_2024_2026.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n[DONE] 2024-2026 summary:")
    print(json.dumps({"overall_equal_weight": overall, "per_symbol": per_symbol}, indent=2, ensure_ascii=False))
    print("\n[OUT]", step4 / "summary_eval_2024_2026.json")


if __name__ == "__main__":
    main()
