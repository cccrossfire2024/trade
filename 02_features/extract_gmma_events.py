# -*- coding: utf-8 -*-
"""
extract_gmma_events.py

从 data_clean/universe_long_dev.parquet（长表：ts,symbol,open,high,low,close,volume）
抽取 GMMA 生命周期事件（A 金叉确认 -> 完全死叉），并记录 B 预警时刻。

事件定义：
- 预警 twarn（B）：median(short) > median(long) 的最近一次上升沿
- 确认 t0（A）：min(short) > max(long) 的上升沿（强金叉）
- 结束 t1：max(short) < min(long) 的首次成立（强死叉）

Gate label（事件级）：
- 以 t0 为锚，计算 event 内 MFE（最大有利涨幅），再用 ATR(t0) 归一：
    mfe_atr = max((close/close0 - 1)) / (atr0/close0)
- 按阈值分类：BIG/MID/SMALL/NONE

输出：
- datasets_v1/gmma_events_dev.parquet
- datasets_v1/gmma_risk_bars_dev.parquet
- datasets_v1/summary_events.json

运行：
  python extract_gmma_events.py --input_long_dev 01_data\\universe_long_all.parquet --out_dir artifacts\\v1\\events
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------
# GMMA params (classic)
# ----------------------------
SHORT_SPANS = [3, 5, 8, 10, 12, 15]
LONG_SPANS  = [30, 35, 40, 45, 50, 60]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    # Wilder-style EMA: alpha = 1/window
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def compute_gmma_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - twarn_flag: median(short) > median(long)       [B]
      - t0_flag:    min(short) > max(long)            [A]
      - t1_flag:    max(short) < min(long)            [dead-cross end]
      - gmma_gap, gmma_widthS, gmma_widthL            (dimensionless)
    """
    close = df["close"]

    short_emas = np.vstack([ema(close, s).to_numpy() for s in SHORT_SPANS])
    long_emas  = np.vstack([ema(close, s).to_numpy() for s in LONG_SPANS])

    # These are safe because we will warmup-trim before calling this in datasets;
    # still, keep nan-aware ops.
    min_short = np.nanmin(short_emas, axis=0)
    max_short = np.nanmax(short_emas, axis=0)
    med_short = np.nanmedian(short_emas, axis=0)

    min_long = np.nanmin(long_emas, axis=0)
    max_long = np.nanmax(long_emas, axis=0)
    med_long = np.nanmedian(long_emas, axis=0)

    out = df.copy()
    out["twarn_flag"] = (med_short > med_long)
    out["t0_flag"]    = (min_short > max_long)
    out["t1_flag"]    = (max_short < min_long)

    out["gmma_gap"]    = (med_short - med_long) / (np.abs(med_long) + 1e-12)
    out["gmma_widthS"] = (max_short - min_short) / (np.abs(med_short) + 1e-12)
    out["gmma_widthL"] = (max_long - min_long) / (np.abs(med_long) + 1e-12)

    return out


def extract_events_for_symbol(
    df_sym: pd.DataFrame,
    atr_window: int,
    gate_mfe_atr_thresholds=(6.0, 3.0, 1.0),
    twarn_lookback_bars: int = 256,
) -> pd.DataFrame:
    """
    Extract events for ONE symbol. df_sym must have columns:
      ts, open, high, low, close, volume, symbol
    """
    df = df_sym.copy()
    df["atr"] = atr(df, window=atr_window)

    # Drop ATR warmup, then also drop GMMA warmup (need at least max(LONG_SPANS))
    df = df.dropna(subset=["atr"]).copy()
    warmup = max(LONG_SPANS)
    if len(df) <= warmup + 10:
        return pd.DataFrame()

    df = df.iloc[warmup:].copy().reset_index(drop=True)

    df = compute_gmma_flags(df)

    ts = df["ts"].to_numpy()
    t0 = df["t0_flag"].to_numpy(dtype=bool)
    tw = df["twarn_flag"].to_numpy(dtype=bool)
    t1 = df["t1_flag"].to_numpy(dtype=bool)

    close = df["close"].to_numpy()
    atrv  = df["atr"].to_numpy()

    thr_big, thr_mid, thr_small = gate_mfe_atr_thresholds

    events = []
    i = 0
    n = len(df)

    while i < n:
        if not t0[i]:
            i += 1
            continue

        # ensure rising edge of t0_flag (strong cross confirmation)
        if i > 0 and t0[i - 1]:
            i += 1
            continue

        i0 = i  # t0 index

        # find first t1 after t0
        j = i0 + 1
        while j < n and not t1[j]:
            j += 1
        if j >= n:
            break
        i1 = j

        # find twarn: most recent rising edge of twarn_flag before i0 within lookback
        twarn_idx = None
        start = max(1, i0 - twarn_lookback_bars)
        for u in range(i0 - 1, start - 1, -1):
            if tw[u] and (not tw[u - 1]):
                twarn_idx = u
                break
        # if none, but twarn already true at i0, allow twarn_idx = i0
        if twarn_idx is None and tw[i0]:
            twarn_idx = i0

        lead_bars = int(i0 - twarn_idx) if twarn_idx is not None else None

        close0 = close[i0]
        atr0 = atrv[i0]
        if not np.isfinite(close0) or not np.isfinite(atr0) or atr0 <= 0:
            i = i1 + 1
            continue

        segment_close = close[i0:i1 + 1]
        ret = (segment_close / close0) - 1.0
        mfe = float(np.nanmax(ret))
        mfe_atr = float(mfe / (atr0 / close0 + 1e-12))

        if mfe_atr >= thr_big:
            label = "BIG"
        elif mfe_atr >= thr_mid:
            label = "MID"
        elif mfe_atr >= thr_small:
            label = "SMALL"
        else:
            label = "NONE"

        events.append({
            "twarn_ts": str(ts[twarn_idx]) if twarn_idx is not None else None,
            "t0_ts": str(ts[i0]),
            "t1_ts": str(ts[i1]),
            "lead_bars": lead_bars,
            "duration_bars": int(i1 - i0),
            "mfe_atr": mfe_atr,
            "label": label,
            # quick structural readings at t0 (useful for later filtering / sanity)
            "gmma_gap_t0": float(df["gmma_gap"].iloc[i0]),
            "gmma_widthS_t0": float(df["gmma_widthS"].iloc[i0]),
            "gmma_widthL_t0": float(df["gmma_widthL"].iloc[i0]),
        })

        i = i1 + 1

    return pd.DataFrame(events)


def build_risk_bars(df_sym: pd.DataFrame, events: pd.DataFrame, atr_window: int) -> pd.DataFrame:
    """
    Build bar-level samples inside each event for ONE symbol.
    Minimal skeleton now; we'll add full features/state labels next.

    Output columns include:
      ts, open, high, low, close, volume, atr,
      twarn_flag, t0_flag, t1_flag, gmma_gap, gmma_widthS, gmma_widthL,
      event_id, t0_ts, t1_ts, bar_idx, duration_bars, progress
    """
    if events.empty:
        return pd.DataFrame()

    df = df_sym.copy()
    df["atr"] = atr(df, window=atr_window)
    df = df.dropna(subset=["atr"]).copy()

    warmup = max(LONG_SPANS)
    if len(df) <= warmup + 10:
        return pd.DataFrame()
    df = df.iloc[warmup:].copy().reset_index(drop=True)

    df = compute_gmma_flags(df)
    df = df.set_index("ts")

    rows = []
    for eid, ev in events.reset_index(drop=True).iterrows():
        t0 = pd.to_datetime(ev["t0_ts"], utc=True)
        t1 = pd.to_datetime(ev["t1_ts"], utc=True)
        seg = df.loc[t0:t1].copy()
        if seg.empty:
            continue

        seg["symbol"] = df_sym["symbol"].iloc[0]
        seg["event_id"] = int(eid)
        seg["t0_ts"] = t0
        seg["t1_ts"] = t1

        seg["bar_idx"] = np.arange(len(seg), dtype=int)
        seg["duration_bars"] = int(len(seg) - 1)
        seg["progress"] = seg["bar_idx"] / np.maximum(seg["duration_bars"], 1)

        # carry event label too (optional)
        seg["event_label"] = ev["label"]
        seg["event_mfe_atr"] = float(ev["mfe_atr"])

        rows.append(seg.reset_index())

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_long_dev", type=str, default="01_data/universe_long_all.parquet")
    ap.add_argument("--out_dir", type=str, default="artifacts/v1/events")
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--gate_thr_big", type=float, default=6.0)
    ap.add_argument("--gate_thr_mid", type=float, default=3.0)
    ap.add_argument("--gate_thr_small", type=float, default=1.0)
    ap.add_argument("--twarn_lookback", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_long_dev)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    thr = (args.gate_thr_big, args.gate_thr_mid, args.gate_thr_small)

    all_events = []
    all_risk = []

    for sym in sorted(df["symbol"].unique()):
        d = df[df["symbol"] == sym].copy().sort_values("ts").reset_index(drop=True)

        print(f"[EVENTS] extracting {sym} ...")
        ev = extract_events_for_symbol(
            d,
            atr_window=args.atr_window,
            gate_mfe_atr_thresholds=thr,
            twarn_lookback_bars=args.twarn_lookback
        )
        if ev.empty:
            print(f"  - no events for {sym}")
            continue

        ev.insert(0, "symbol", sym)
        ev.insert(1, "event_id", np.arange(len(ev), dtype=int))
        all_events.append(ev)
        print(f"  - events: {len(ev)}")

        rb = build_risk_bars(d, ev, atr_window=args.atr_window)
        if not rb.empty:
            # rb already contains symbol
            all_risk.append(rb)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    risk_df = pd.concat(all_risk, ignore_index=True) if all_risk else pd.DataFrame()

    events_path = out_dir / "gmma_events_dev.parquet"
    risk_path = out_dir / "gmma_risk_bars_dev.parquet"

    events_df.to_parquet(events_path, index=False)
    risk_df.to_parquet(risk_path, index=False)

    summary = {
        "events_rows": int(len(events_df)),
        "risk_rows": int(len(risk_df)),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "label_counts": events_df["label"].value_counts().to_dict() if not events_df.empty else {},
        "events_path": str(events_path),
        "risk_path": str(risk_path),
        "atr_window": args.atr_window,
        "twarn_lookback_bars": args.twarn_lookback,
        "gate_thresholds_mfe_atr": {
            "BIG": args.gate_thr_big,
            "MID": args.gate_thr_mid,
            "SMALL": args.gate_thr_small
        }
    }
    (out_dir / "summary_events.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("[DONE]", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
