# -*- coding: utf-8 -*-
"""
build_gate_risk_features_v1.py

输入：
- data_clean/universe_long_dev.parquet
- datasets_v1/gmma_events_dev.parquet
- datasets_v1/gmma_risk_bars_dev.parquet

输出（out_dir 下）：
- gate_samples_v1.parquet            # 每事件一行（Gate 训练集 v1）
- risk_bars_features_v1.parquet      # 事件内 bar 级特征（Risk 数据集 v1）
- summary_features_v1.json

Gate 快照窗口（方案1）：
- pre:  t0 前 16 根 + t0 本身
- post: t0 后 8 根（含 t0）

特征：全部相对化/无量纲（logret、ATR归一、比例、rolling z 等）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHORT_SPANS = [3, 5, 8, 10, 12, 15]
LONG_SPANS  = [30, 35, 40, 45, 50, 60]
WARMUP = max(LONG_SPANS)  # 60


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(h, l, c):
    pc = c.shift(1)
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window=14) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def logret(c: pd.Series, k: int) -> pd.Series:
    return np.log(c / c.shift(k))


def rvol(v: pd.Series, window=96) -> pd.Series:
    ma = v.rolling(window, min_periods=window).mean()
    return v / (ma + 1e-12)


def clv(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    return ((c - l) - (h - c)) / ((h - l) + 1e-12)


def wick_skew(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    return (upper - lower) / ((h - l) + 1e-12)


def body_eff(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    return (c - o) / ((h - l) + 1e-12)


def compute_gmma_struct(close: pd.Series) -> pd.DataFrame:
    short_emas = [ema(close, s) for s in SHORT_SPANS]
    long_emas  = [ema(close, s) for s in LONG_SPANS]

    short_df = pd.concat(short_emas, axis=1)
    long_df  = pd.concat(long_emas, axis=1)

    med_short = short_df.median(axis=1)
    med_long  = long_df.median(axis=1)

    min_short = short_df.min(axis=1)
    max_short = short_df.max(axis=1)

    min_long = long_df.min(axis=1)
    max_long = long_df.max(axis=1)

    out = pd.DataFrame(index=close.index)
    out["gmma_gap"] = (med_short - med_long) / (med_long.abs() + 1e-12)
    out["gmma_widthS"] = (max_short - min_short) / (med_short.abs() + 1e-12)
    out["gmma_widthL"] = (max_long - min_long) / (med_long.abs() + 1e-12)

    # slope (1 bar diff, dimensionless-ish)
    out["gmma_slopeS"] = (med_short.diff(1)) / (med_long.abs() + 1e-12)
    out["gmma_slopeL"] = (med_long.diff(1)) / (med_long.abs() + 1e-12)
    out["gmma_slopeS_acc"] = out["gmma_slopeS"].diff(1)
    return out


def add_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单币 df（index=ts，含 open/high/low/close/volume）
    输出：添加因果 bar 级特征（无量纲/相对化）
    """
    x = df.copy()
    x["atr14"] = atr(x, 14)

    c = x["close"]
    v = x["volume"]

    # returns
    for k in [1, 4, 16, 64]:
        x[f"lr_{k}"] = logret(c, k)

    # Rogers-Satchell (dimensionless)
    o, h, l = x["open"], x["high"], x["low"]
    rs_var = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))
    rs_var = rs_var.replace([np.inf, -np.inf], np.nan)
    x["rs_var"] = rs_var
    x["rs_vol"] = np.sqrt(x["rs_var"].clip(lower=0))

    # RS vol relative to rolling median (causal)
    x["rs_vol_rel"] = x["rs_vol"] / (x["rs_vol"].rolling(96, min_periods=96).median() + 1e-12)

    # volume features
    x["rvol_96"] = rvol(v, 96)
    x["dlogv_1"] = np.log(v + 1e-12).diff(1)
    x["dlogv_z"] = (x["dlogv_1"] - x["dlogv_1"].rolling(96, min_periods=96).mean()) / (
        x["dlogv_1"].rolling(96, min_periods=96).std() + 1e-12
    )

    # candle shape
    x["clv"] = clv(x)
    x["wick_skew"] = wick_skew(x)
    x["body_eff"] = body_eff(x)

    # ATR-normalized range/body
    x["range_atr"] = (x["high"] - x["low"]) / (x["atr14"] + 1e-12)
    x["body_atr"] = (x["close"] - x["open"]) / (x["atr14"] + 1e-12)

    # Efficiency Ratio over 16
    disp = (c - c.shift(16)).abs()
    path = (c.diff(1).abs()).rolling(16, min_periods=16).sum()
    x["er_16"] = disp / (path + 1e-12)

    # Variance ratio VR(16/64)
    r1 = c.diff(1)
    var1 = r1.rolling(64, min_periods=64).var()
    r16 = c.diff(16)
    var16 = r16.rolling(64, min_periods=64).var()
    x["vr_16_64"] = var16 / (16.0 * var1 + 1e-12)

    # GMMA structural
    g = compute_gmma_struct(c)
    x = pd.concat([x, g], axis=1)

    # VoV: volatility-of-volatility (ATR pct change std)
    d_atr = x["atr14"].pct_change()
    x["vov_96"] = d_atr.rolling(96, min_periods=96).std()

    return x


def summarize_window(df_feat: pd.DataFrame, t_center: pd.Timestamp, w_pre: int, w_post: int) -> dict:
    """
    从 df_feat（index=ts）取 t0 前后窗口，并聚合成一行特征。
    pre:  [t0-w_pre, t0]  含 t0
    post: [t0, t0+w_post] 含 t0
    """
    idx = df_feat.index
    if t_center not in idx:
        return {}

    loc = idx.get_loc(t_center)
    pre_slice = df_feat.iloc[max(0, loc - w_pre): loc + 1]
    post_slice = df_feat.iloc[loc: min(len(df_feat), loc + w_post + 1)]

    cols = [
        "lr_1", "lr_4", "lr_16", "lr_64",
        "rs_vol_rel", "rvol_96", "dlogv_z",
        "clv", "wick_skew", "body_eff",
        "range_atr", "body_atr",
        "er_16", "vr_16_64",
        "gmma_gap", "gmma_widthS", "gmma_widthL",
        "gmma_slopeS", "gmma_slopeL", "gmma_slopeS_acc",
        "vov_96"
    ]
    cols = [c for c in cols if c in df_feat.columns]

    feat = {}

    for c in cols:
        s = pre_slice[c]
        feat[f"pre_{c}_mean"] = float(s.mean())
        feat[f"pre_{c}_std"] = float(s.std())
        feat[f"pre_{c}_last"] = float(s.iloc[-1])

    for c in cols:
        s = post_slice[c]
        feat[f"post_{c}_mean"] = float(s.mean())
        feat[f"post_{c}_std"] = float(s.std())
        feat[f"post_{c}_last"] = float(s.iloc[-1])

    if "atr14" in df_feat.columns:
        feat["atr14_t0"] = float(df_feat.loc[t_center, "atr14"])

    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_long_dev", type=str, default="data_clean/universe_long_dev.parquet")
    ap.add_argument("--events_dev", type=str, default="datasets_v1/gmma_events_dev.parquet")
    ap.add_argument("--risk_bars_dev", type=str, default="datasets_v1/gmma_risk_bars_dev.parquet")
    ap.add_argument("--out_dir", type=str, default="datasets_v2")
    ap.add_argument("--w_pre", type=int, default=16)
    ap.add_argument("--w_post", type=int, default=8)
    ap.add_argument("--min_event_bars", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load universe ----
    uni = pd.read_parquet(args.universe_long_dev)
    uni["ts"] = pd.to_datetime(uni["ts"], utc=True)
    uni = uni.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # ---- load events ----
    events = pd.read_parquet(args.events_dev)
    events["t0_ts"] = pd.to_datetime(events["t0_ts"], utc=True)
    events["t1_ts"] = pd.to_datetime(events["t1_ts"], utc=True)
    events["twarn_ts"] = pd.to_datetime(events["twarn_ts"], utc=True, errors="coerce")

    if args.min_event_bars > 0:
        events = events.loc[events["duration_bars"] >= args.min_event_bars].copy()

    # ---- build per-symbol feature tables ----
    feat_tables = {}
    for sym in sorted(uni["symbol"].unique()):
        d = uni[uni["symbol"] == sym].copy()
        d = d.sort_values("ts").reset_index(drop=True)
        d = d.set_index("ts")

        print(f"[FEAT] computing bar features for {sym} ...")
        f = add_bar_features(d)

        # warmup trim to reduce NaNs in EMA-derived features
        f = f.iloc[WARMUP:].copy()
        feat_tables[sym] = f

    # ---- Gate samples ----
    print("[GATE] building event snapshots ...")
    gate_rows = []

    for _, ev in events.iterrows():
        sym = ev["symbol"]
        t0 = ev["t0_ts"]

        df_feat = feat_tables.get(sym)
        if df_feat is None or t0 not in df_feat.index:
            continue

        row = {
            "symbol": sym,
            "event_id": int(ev["event_id"]),
            "twarn_ts": ev["twarn_ts"],
            "t0_ts": t0,
            "t1_ts": ev["t1_ts"],
            "lead_bars": ev["lead_bars"],
            "duration_bars": int(ev["duration_bars"]),
            "label": ev["label"],
            "mfe_atr": float(ev["mfe_atr"]),
        }

        snap = summarize_window(df_feat, t0, w_pre=args.w_pre, w_post=args.w_post)
        if not snap:
            continue

        row.update(snap)
        gate_rows.append(row)

    gate_df = pd.DataFrame(gate_rows)
    gate_path = out_dir / "gate_samples_v1.parquet"
    gate_df.to_parquet(gate_path, index=False)

    # ---- Risk bar features ----
    print("[RISK] building risk bars features ...")
    risk_in = pd.read_parquet(args.risk_bars_dev)
    risk_in["ts"] = pd.to_datetime(risk_in["ts"], utc=True)

    keep_risk_cols = [
        "symbol", "event_id", "ts", "t0_ts", "t1_ts",
        "bar_idx", "duration_bars", "progress",
        "event_label", "event_mfe_atr"
    ]
    keep_risk_cols = [c for c in keep_risk_cols if c in risk_in.columns]
    risk = risk_in[keep_risk_cols].copy()

    risk_cols = [
        "lr_1", "lr_4", "lr_16",
        "rs_vol_rel", "rvol_96", "dlogv_z",
        "clv", "wick_skew", "body_eff",
        "range_atr", "body_atr",
        "er_16", "vr_16_64",
        "gmma_gap", "gmma_widthS", "gmma_widthL",
        "gmma_slopeS", "gmma_slopeL", "gmma_slopeS_acc",
        "vov_96",
        "atr14"
    ]

    risk_parts = []
    for sym in sorted(risk["symbol"].unique()):
        part = risk[risk["symbol"] == sym].copy()
        f = feat_tables.get(sym)
        if f is None:
            continue

        fx = f[risk_cols].copy().reset_index()  # reset_index -> has "ts" col
        merged = part.merge(fx, on="ts", how="left")
        risk_parts.append(merged)

    risk_feat = pd.concat(risk_parts, ignore_index=True) if risk_parts else pd.DataFrame()
    risk_feat_path = out_dir / "risk_bars_features_v1.parquet"
    risk_feat.to_parquet(risk_feat_path, index=False)

    # ---- Summary ----
    summary = {
        "gate_rows": int(len(gate_df)),
        "risk_rows": int(len(risk_feat)),
        "gate_path": str(gate_path),
        "risk_path": str(risk_feat_path),
        "w_pre": args.w_pre,
        "w_post": args.w_post,
        "min_event_bars": args.min_event_bars,
        "gate_label_counts": gate_df["label"].value_counts().to_dict() if not gate_df.empty else {},
    }

    summary_path = out_dir / "summary_features_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[DONE]", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
