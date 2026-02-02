# -*- coding: utf-8 -*-
"""
build_gate_risk_features_v2.py (fixed v2.1)

修复：
- 添加 lr_48（供 nr_48 使用）
- universe_long_dev 若缺 atr 自动计算
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def _to_utc_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def rolling_zscore(x: pd.Series, win: int, minp: int) -> pd.Series:
    mu = x.rolling(win, min_periods=minp).mean()
    sd = x.rolling(win, min_periods=minp).std()
    return (x - mu) / (sd + EPS)


def calc_atr_wilder(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr


def efficiency_ratio(close: pd.Series, k: int) -> pd.Series:
    disp = (close - close.shift(k)).abs()
    path = close.diff().abs().rolling(k, min_periods=max(2, k // 3)).sum()
    return disp / (path + EPS)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()

    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / (atr + EPS)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / (atr + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    lo = np.log((high + EPS) / (open_ + EPS))
    lc = np.log((high + EPS) / (close + EPS))
    so = np.log((low + EPS) / (open_ + EPS))
    sc = np.log((low + EPS) / (close + EPS))
    return lo * lc + so * sc


def candle_parts(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, atr: pd.Series):
    body = (close - open_).abs() / (atr + EPS)
    upwick = (high - np.maximum(open_, close)) / (atr + EPS)
    lowwick = (np.minimum(open_, close) - low) / (atr + EPS)
    rng = (high - low) / (atr + EPS)
    clv = (2.0 * close - high - low) / ((high - low) + EPS)
    return body, upwick, lowwick, rng, clv


def nanmin_safe(arr: np.ndarray, axis: int):
    all_nan = np.all(np.isnan(arr), axis=axis)
    out = np.nanmin(arr, axis=axis)
    if np.any(all_nan):
        out = out.astype(float, copy=False)
        out[all_nan] = np.nan
    return out


def nanmax_safe(arr: np.ndarray, axis: int):
    all_nan = np.all(np.isnan(arr), axis=axis)
    out = np.nanmax(arr, axis=axis)
    if np.any(all_nan):
        out = out.astype(float, copy=False)
        out[all_nan] = np.nan
    return out


def nanmedian_safe(arr: np.ndarray, axis: int):
    all_nan = np.all(np.isnan(arr), axis=axis)
    out = np.nanmedian(arr, axis=axis)
    if np.any(all_nan):
        out = out.astype(float, copy=False)
        out[all_nan] = np.nan
    return out


def compute_gmma_features(
    df: pd.DataFrame,
    short_spans=(3, 5, 8, 10, 12, 15),
    long_spans=(30, 35, 40, 45, 50, 60),
):
    close = df["close"].astype(float)
    atr = df["atr"].astype(float)

    short_emas = []
    for s in short_spans:
        short_emas.append(close.ewm(span=s, adjust=False, min_periods=s).mean())
    long_emas = []
    for s in long_spans:
        long_emas.append(close.ewm(span=s, adjust=False, min_periods=s).mean())

    short_emas = np.vstack([x.to_numpy(dtype=float) for x in short_emas])
    long_emas = np.vstack([x.to_numpy(dtype=float) for x in long_emas])

    short_min = nanmin_safe(short_emas, axis=0)
    short_max = nanmax_safe(short_emas, axis=0)
    long_min = nanmin_safe(long_emas, axis=0)
    long_max = nanmax_safe(long_emas, axis=0)
    short_med = nanmedian_safe(short_emas, axis=0)
    long_med = nanmedian_safe(long_emas, axis=0)

    df["gmma_widthS"] = (short_max - short_min) / (atr + EPS)
    df["gmma_widthL"] = (long_max - long_min) / (atr + EPS)
    df["gmma_gap"] = (pd.Series(short_med, index=df.index) - pd.Series(long_med, index=df.index)) / (atr + EPS)
    df["gmma_div"] = (pd.Series(short_min, index=df.index) - pd.Series(long_max, index=df.index)) / (atr + EPS)

    df["gmma_slopeS"] = pd.Series(short_med, index=df.index).diff(1) / (atr + EPS)
    df["gmma_slopeL"] = pd.Series(long_med, index=df.index).diff(1) / (atr + EPS)
    df["gmma_attack"] = df["gmma_slopeS"]

    long_slopes = []
    for i in range(long_emas.shape[0]):
        s = pd.Series(long_emas[i, :], index=df.index).diff(1) / (atr + EPS)
        long_slopes.append(s.to_numpy(dtype=float))
    long_slopes = np.vstack(long_slopes)
    df["gmma_parallelL"] = np.nanstd(long_slopes, axis=0)

    df["gmma_slopeS_acc"] = df["gmma_slopeS"].diff(1)
    df["gmma_gap_chg"] = df["gmma_gap"].diff(1)
    df["gmma_gap_acc"] = df["gmma_gap_chg"].diff(1)
    return df


def compute_bar_features_one_symbol(
    df: pd.DataFrame,
    rvol_ma: int = 96,
    rvol_z_win: int = 192,
    bb_win: int = 20,
    z_win_short: int = 24,
    z_win_long: int = 192,
):
    df = df.sort_values("ts").reset_index(drop=True)

    close = df["close"].astype(float)
    df["lr_1"] = np.log((close + EPS) / (close.shift(1) + EPS))

    # ✅ FIX: include lr_48 because nr_48 depends on it
    for k in (4, 16, 48, 64):
        df[f"lr_{k}"] = np.log((close + EPS) / (close.shift(k) + EPS))

    df["atr_pct"] = df["atr"].astype(float) / (close + EPS)

    body, upwick, lowwick, rng, clv = candle_parts(
        df["open"].astype(float),
        df["high"].astype(float),
        df["low"].astype(float),
        close,
        df["atr"].astype(float),
    )
    df["body_atr"] = body
    df["upwick_atr"] = upwick
    df["lowwick_atr"] = lowwick
    df["range_atr"] = rng
    df["clv"] = clv

    df["upwick_z24"] = rolling_zscore(df["upwick_atr"], z_win_short, minp=max(16, z_win_short // 3))
    df["body_z24"] = rolling_zscore(df["body_atr"], z_win_short, minp=max(16, z_win_short // 3))
    df["clv_mean_16"] = df["clv"].rolling(16, min_periods=8).mean()
    df["clv_z48"] = rolling_zscore(df["clv"], 48, minp=16)

    vol = df["volume"].astype(float)
    vol_ma = vol.rolling(rvol_ma, min_periods=max(16, rvol_ma // 3)).mean()
    df["rvol_96"] = vol / (vol_ma + EPS)
    df["rvol_log_96"] = np.log1p(df["rvol_96"])
    df["rvol_z_192"] = rolling_zscore(df["rvol_log_96"], rvol_z_win, minp=max(48, rvol_z_win // 3))
    df["rs_vol_rel"] = df["rvol_96"]

    rs = rogers_satchell(df["open"].astype(float), df["high"].astype(float), df["low"].astype(float), close)
    df["rs"] = rs
    df["rsvol_mean_48"] = rs.rolling(48, min_periods=16).mean()
    df["rsvol_z192"] = rolling_zscore(df["rsvol_mean_48"], z_win_long, minp=max(48, z_win_long // 3))

    ma = close.rolling(bb_win, min_periods=max(10, bb_win // 2)).mean()
    sd = close.rolling(bb_win, min_periods=max(10, bb_win // 2)).std()
    df["bbw"] = (4.0 * sd) / (ma.abs() + EPS)

    df["adx_14"] = adx(df["high"].astype(float), df["low"].astype(float), close, n=14)

    df["er_16"] = efficiency_ratio(close, 16)
    df["er_48"] = efficiency_ratio(close, 48)
    df["er_96"] = efficiency_ratio(close, 96)

    df["vol_8"] = df["lr_1"].rolling(8, min_periods=3).std()
    df["vol_16"] = df["lr_1"].rolling(16, min_periods=6).std()
    df["vol_24"] = df["lr_1"].rolling(24, min_periods=8).std()
    df["vol_48"] = df["lr_1"].rolling(48, min_periods=16).std()
    df["vol_96"] = df["lr_1"].rolling(96, min_periods=32).std()

    df["vov_24"] = df["vol_8"] / (df["vol_24"] + EPS)
    df["vov_48"] = df["vol_16"] / (df["vol_48"] + EPS)
    df["vov_96"] = df["vol_24"] / (df["vol_96"] + EPS)

    # Noise ratio
    k = 48
    if "lr_48" not in df.columns:
        df["lr_48"] = np.log((close + EPS) / (close.shift(48) + EPS))
    df["nr_48"] = df["lr_1"].abs().rolling(k, min_periods=max(16, k // 3)).sum() / (df["lr_48"].abs() + EPS)

    # Breakout strength
    for kk in (16, 48):
        prior_max = df["high"].astype(float).rolling(kk, min_periods=max(8, kk // 3)).max().shift(1)
        df[f"brs_{kk}"] = (close - prior_max) / (df["atr"].astype(float) + EPS)

    df = compute_gmma_features(df)

    df["gap_chg_z24"] = rolling_zscore(df["gmma_gap_chg"], 24, minp=16)
    df["gap_acc_z24"] = rolling_zscore(df["gmma_gap_acc"], 24, minp=16)
    df["gap_chg_z96"] = rolling_zscore(df["gmma_gap_chg"], 96, minp=32)

    return df


def infer_event_cols(events: pd.DataFrame):
    def pick(cands):
        for c in cands:
            if c in events.columns:
                return c
        return None

    c_sym = pick(["symbol"])
    c_eid = pick(["event_id", "eid"])
    c_t0 = pick(["t0_ts", "t0", "start_ts", "start_time"])
    c_t1 = pick(["t1_ts", "t1", "end_ts", "end_time"])
    c_tw = pick(["twarn_ts", "twarn", "warn_ts"])
    c_lbl = pick(["label", "gate_label"])
    c_mfe = pick(["mfe_atr", "mfe", "mfe_norm"])

    if c_sym is None or c_eid is None or c_t0 is None:
        raise ValueError("events missing required columns among: symbol, event_id, t0_ts")

    return c_sym, c_eid, c_t0, c_t1, c_tw, c_lbl, c_mfe


def build_gate_samples(events: pd.DataFrame, panel: pd.DataFrame, w_pre: int = 16):
    c_sym, c_eid, c_t0, c_t1, c_tw, c_lbl, c_mfe = infer_event_cols(events)

    events = events.copy()
    events["t0_ts"] = _to_utc_ts(events[c_t0])
    events["t1_ts"] = _to_utc_ts(events[c_t1]) if c_t1 else pd.NaT
    events["twarn_ts"] = _to_utc_ts(events[c_tw]) if c_tw else pd.NaT

    base = pd.DataFrame(
        {
            "symbol": events[c_sym].astype(str),
            "event_id": events[c_eid].astype(int),
            "t0_ts": events["t0_ts"],
            "t1_ts": events["t1_ts"],
            "twarn_ts": events["twarn_ts"],
            "label": events[c_lbl].astype(str) if c_lbl else "NA",
            "mfe_atr": pd.to_numeric(events[c_mfe], errors="coerce") if c_mfe else np.nan,
        }
    )

    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    sym_groups = {sym: sub.reset_index(drop=True) for sym, sub in panel.groupby("symbol", sort=False)}

    snap_cols = [
        "gmma_gap", "gmma_div", "gmma_widthS", "gmma_widthL",
        "gmma_slopeS", "gmma_slopeL", "gmma_slopeS_acc",
        "er_16", "er_48", "er_96",
        "vov_96", "rs_vol_rel", "rvol_log_96", "rvol_z_192",
        "rsvol_mean_48", "rsvol_z192",
        "bbw", "adx_14",
        "nr_48",
        "brs_16", "brs_48",
        "body_atr", "upwick_atr", "lowwick_atr", "range_atr", "clv",
        "gap_chg_z24", "gap_acc_z24",
        "lr_16",
    ]
    snap_cols = [c for c in snap_cols if c in panel.columns]

    agg_cols = [
        "lr_1", "lr_4", "lr_16",
        "range_atr", "body_atr", "upwick_atr", "clv",
        "rvol_log_96", "rvol_z_192", "rsvol_mean_48",
        "gmma_gap", "gmma_slopeS", "er_16", "vov_96", "nr_48",
        "brs_48",
    ]
    agg_cols = [c for c in agg_cols if c in panel.columns]

    rows = []
    skipped = 0

    for r in base.itertuples(index=False):
        sym = r.symbol
        t0 = r.t0_ts
        if pd.isna(t0) or sym not in sym_groups:
            skipped += 1
            continue

        sub = sym_groups[sym]
        t0_ns = t0.to_datetime64()
        loc = np.where(sub["ts"].values == t0_ns)[0]
        if len(loc) == 0:
            skipped += 1
            continue
        i0 = int(loc[0])

        i_start = max(0, i0 - w_pre)
        win = sub.iloc[i_start : i0 + 1]

        out = {
            "symbol": sym,
            "event_id": int(r.event_id),
            "t0_ts": t0,
            "t1_ts": r.t1_ts,
            "twarn_ts": r.twarn_ts,
            "label": r.label,
            "mfe_atr": float(r.mfe_atr) if pd.notna(r.mfe_atr) else np.nan,
        }

        cur = sub.iloc[i0]
        for c in snap_cols:
            out[f"t0_{c}"] = float(cur[c]) if pd.notna(cur[c]) else np.nan

        for c in agg_cols:
            s = win[c].astype(float)
            out[f"pre_mean_{c}"] = float(s.mean()) if s.notna().any() else np.nan
            out[f"pre_std_{c}"] = float(s.std()) if s.notna().any() else np.nan
            out[f"pre_p90_{c}"] = float(s.quantile(0.9)) if s.notna().any() else np.nan
            out[f"pre_p10_{c}"] = float(s.quantile(0.1)) if s.notna().any() else np.nan

        if "lr_1" in win.columns and win["lr_1"].notna().any():
            lr_sum = float(win["lr_1"].sum())
            lr_abs_sum = float(win["lr_1"].abs().sum())
            out["pre_lr_sum_1"] = lr_sum
            out["pre_lr_abs_sum_1"] = lr_abs_sum
            out["pre_noise_ratio_16"] = float(lr_abs_sum / (abs(lr_sum) + EPS))
        else:
            out["pre_lr_sum_1"] = np.nan
            out["pre_lr_abs_sum_1"] = np.nan
            out["pre_noise_ratio_16"] = np.nan

        rows.append(out)

    gate = pd.DataFrame(rows)
    meta = {"skipped_events": skipped, "built_events": len(gate), "total_events": len(base)}
    return gate, meta


def build_risk_bars_features(risk_bars: pd.DataFrame, panel: pd.DataFrame):
    rb = risk_bars.copy()

    if "ts" not in rb.columns:
        if "timestamp" in rb.columns:
            rb = rb.rename(columns={"timestamp": "ts"})
        else:
            raise ValueError("risk_bars missing ts/timestamp column")
    if "symbol" not in rb.columns:
        raise ValueError("risk_bars missing symbol column")
    if "event_id" not in rb.columns:
        raise ValueError("risk_bars missing event_id column")

    rb["ts"] = _to_utc_ts(rb["ts"])
    panel = panel.copy()
    panel["ts"] = _to_utc_ts(panel["ts"])

    drop_raw = {"open", "high", "low", "close", "volume", "atr"}
    feat_cols = [c for c in panel.columns if c not in drop_raw]
    panel_small = panel[["symbol", "ts"] + [c for c in feat_cols if c not in ("symbol", "ts")]]

    out = rb.merge(panel_small, on=["symbol", "ts"], how="left")

    must = ["gmma_gap", "gmma_slopeS", "gmma_slopeS_acc", "er_16", "vov_96", "rs_vol_rel", "lr_16"]
    for c in must:
        if c not in out.columns:
            out[c] = np.nan

    return out


def load_universe_long(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex) and df.index.name in ("ts", "timestamp"):
        df = df.reset_index()

    if "ts" not in df.columns and "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    if "ts" not in df.columns:
        raise ValueError("universe_long_dev missing ts/timestamp column")

    # normalize common upper-case names
    rename_map = {}
    for c in df.columns:
        lo = c.lower()
        if lo in ("open", "high", "low", "close", "volume", "atr", "symbol", "ts"):
            rename_map[c] = lo
    if rename_map:
        df = df.rename(columns=rename_map)

    if "symbol" not in df.columns:
        raise ValueError("universe_long_dev missing symbol column")
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            raise ValueError(f"universe_long_dev missing required column: {c}")

    df["ts"] = _to_utc_ts(df["ts"])
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)

    # 输入路径：建议显式传入（dev / blind 都用同一套参数名，避免串数据）
    # 若不传，则回退到 *_dev 兼容旧命令行。
    ap.add_argument("--universe_long", type=str, default=None, help="Universe long parquet (e.g. data_clean/universe_long_dev.parquet or ..._blind.parquet)")
    ap.add_argument("--events", type=str, default=None, help="GMMA events parquet (e.g. datasets_v1/gmma_events_dev.parquet or datasets_blind_v1/step1_events/gmma_events_dev.parquet)")
    ap.add_argument("--risk_bars", type=str, default=None, help="GMMA risk bars parquet (e.g. datasets_v1/gmma_risk_bars_dev.parquet or datasets_blind_v1/step1_events/gmma_risk_bars_dev.parquet)")

    # 兼容旧参数名（不建议继续使用）
    ap.add_argument("--universe_long_dev", type=str, default="data_clean/universe_long_dev.parquet")
    ap.add_argument("--events_dev", type=str, default="datasets_v1/gmma_events_dev.parquet")
    ap.add_argument("--risk_bars_dev", type=str, default="datasets_v1/gmma_risk_bars_dev.parquet")

    ap.add_argument("--w_pre", type=int, default=16)
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--rvol_ma", type=int, default=96)
    ap.add_argument("--rvol_z_win", type=int, default=192)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析最终输入路径（显式参数优先，避免 dev/blind 串线）
    in_universe = args.universe_long if args.universe_long is not None else args.universe_long_dev
    in_events = args.events if args.events is not None else args.events_dev
    in_risk_bars = args.risk_bars if args.risk_bars is not None else args.risk_bars_dev
    print(f"[INPUT] universe_long={in_universe}")
    print(f"[INPUT] events={in_events}")
    print(f"[INPUT] risk_bars={in_risk_bars}")


    uni = load_universe_long(in_universe)

    if "atr" not in uni.columns:
        print(f"[INFO] 'atr' not found in universe_long_dev; computing ATR({args.atr_window}) per symbol ...")
        uni["atr"] = np.nan
        for sym, idx in uni.groupby("symbol").groups.items():
            sub = uni.loc[idx].sort_values("ts").copy()
            atr = calc_atr_wilder(sub, window=args.atr_window)
            uni.loc[idx, "atr"] = atr.values
        uni["atr"] = uni["atr"].astype("float32")
        uni = uni.dropna(subset=["atr"]).reset_index(drop=True)

    feats = []
    for sym, sub in uni.groupby("symbol", sort=False):
        print(f"[FEAT v2] computing bar features for {sym} ...")
        sub = sub.copy()
        sub = compute_bar_features_one_symbol(
            sub,
            rvol_ma=args.rvol_ma,
            rvol_z_win=args.rvol_z_win,
        )
        feats.append(sub)

    panel = pd.concat(feats, ignore_index=True)
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)

    events = pd.read_parquet(in_events)
    risk_bars = pd.read_parquet(in_risk_bars)

    # 输入范围审计：防止把 dev 事件写进 blind 目录之类的问题
    if 't0_ts' in events.columns:
        _t0 = _to_utc_ts(events['t0_ts'])
        print(f"[AUDIT] events rows={len(events)} t0_min={_t0.min()} t0_max={_t0.max()}")
    if 'ts' in uni.columns:
        _ts = _to_utc_ts(uni['ts'])
        print(f"[AUDIT] universe rows={len(uni)} ts_min={_ts.min()} ts_max={_ts.max()}")


    print("[GATE v2] building event snapshots (history-only) ...")
    gate, gate_meta = build_gate_samples(events, panel, w_pre=args.w_pre)

    print("[RISK v2] building risk bars features ...")
    risk_feat = build_risk_bars_features(risk_bars, panel)

    gate_path = out_dir / "gate_samples_v2.parquet"
    risk_path = out_dir / "risk_bars_features_v2.parquet"
    gate.to_parquet(gate_path, index=False)
    risk_feat.to_parquet(risk_path, index=False)

    summary = {
        "gate_rows": int(len(gate)),
        "risk_rows": int(len(risk_feat)),
        "gate_path": str(gate_path),
        "risk_path": str(risk_path),
        "w_pre": int(args.w_pre),
        "atr_window": int(args.atr_window),
        "rvol_ma": int(args.rvol_ma),
        "rvol_z_win": int(args.rvol_z_win),
        "gate_meta": gate_meta,
        "gate_label_counts": gate["label"].value_counts(dropna=False).to_dict() if "label" in gate.columns else {},
        "feature_cols_gate": [c for c in gate.columns if c not in ("symbol", "event_id", "t0_ts", "t1_ts", "twarn_ts", "label", "mfe_atr")],
        "feature_cols_risk": [c for c in risk_feat.columns if c not in ("symbol", "event_id", "ts")],
    }

    (out_dir / "summary_features_v2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[DONE]", json.dumps(
        {
            "gate_rows": summary["gate_rows"],
            "risk_rows": summary["risk_rows"],
            "gate_path": summary["gate_path"],
            "risk_path": summary["risk_path"],
            "skipped_events": summary["gate_meta"]["skipped_events"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()