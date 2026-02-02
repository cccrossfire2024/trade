import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

def is_15m_grid(idx: pd.DatetimeIndex) -> np.ndarray:
    return ((idx.minute % 15) == 0) & (idx.second == 0) & (idx.microsecond == 0)

def _ensure_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一：确保存在 ts 列（UTC datetime），并且 index 不叫 ts（避免歧义）
    """
    x = df.copy()

    # 1) 如果 index 是 DatetimeIndex，先变成列
    if isinstance(x.index, pd.DatetimeIndex):
        idx_name = x.index.name
        # 如果 index 名叫 ts，先重命名 index，避免与列冲突
        if idx_name == "ts":
            x.index = x.index.rename("ts_index")
        x = x.reset_index()

    # 2) 现在 ts 可能在列里：ts / ts_index / timestamp / open_time
    if "ts" in x.columns:
        x["ts"] = pd.to_datetime(x["ts"], utc=True)
    elif "ts_index" in x.columns:
        x["ts"] = pd.to_datetime(x["ts_index"], utc=True)
        x = x.drop(columns=["ts_index"])
    elif "timestamp" in x.columns:
        x["ts"] = pd.to_datetime(x["timestamp"], utc=True)
        x = x.drop(columns=["timestamp"])
    elif "open_time" in x.columns:
        # 兼容 ms epoch
        x["ts"] = pd.to_datetime(x["open_time"], unit="ms", utc=True)
        x = x.drop(columns=["open_time"])
    else:
        raise ValueError(f"Cannot find timestamp in columns: {list(x.columns)[:30]}")

    return x

def load_one(parquet_path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = _ensure_ts_column(df)

    # 强制字段存在
    keep = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{parquet_path} missing columns: {missing}. columns={list(df.columns)[:30]}")

    df = df[keep].copy()

    # 强制 UTC、排序、去重
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")
    df = df.drop_duplicates(subset=["ts"], keep="last")

    # 过滤非15m网格（你的 audit=0，这里保险）
    idx = pd.DatetimeIndex(df["ts"])
    df = df.loc[is_15m_grid(idx)].copy()

    # 数值清理（nan/inf）
    arr = df[["open","high","low","close","volume"]].to_numpy()
    mask = np.isfinite(arr).all(axis=1)
    df = df.loc[mask].copy()

    # 再做一次 OHLC 合法性检查（保险）
    bad = (
        (df["high"] < df[["open","close"]].max(axis=1)) |
        (df["low"]  > df[["open","close"]].min(axis=1)) |
        (df["high"] < df["low"])
    )
    df = df.loc[~bad].copy()

    df["symbol"] = symbol
    return df

def align_intersection(frames):
    sets = [set(pd.DatetimeIndex(f["ts"])) for f in frames]
    common = set.intersection(*sets)
    common_idx = pd.DatetimeIndex(sorted(common))

    out = []
    for f in frames:
        g = f.set_index("ts").reindex(common_idx).reset_index()
        g = g.rename(columns={"index": "ts"})
        out.append(g)

    long_df = pd.concat(out, ignore_index=True).sort_values(["ts","symbol"]).reset_index(drop=True)
    return long_df

def make_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in long_df.columns if c not in ["ts","symbol"]]
    wide = long_df.pivot(index="ts", columns="symbol", values=value_cols)
    wide.columns = [f"{sym}_{val}".upper() for val, sym in wide.columns]
    wide = wide.sort_index()
    return wide

def split_by_date(long_df: pd.DataFrame, dev_start: str, dev_end: str, blind_start: str, blind_end: str):
    ts = pd.to_datetime(long_df["ts"], utc=True)

    dev_mask   = (ts >= pd.to_datetime(dev_start, utc=True)) & (ts <= pd.to_datetime(dev_end, utc=True))
    blind_mask = (ts >= pd.to_datetime(blind_start, utc=True)) & (ts <= pd.to_datetime(blind_end, utc=True))

    dev = long_df.loc[dev_mask].copy()
    blind = long_df.loc[blind_mask].copy()
    return dev, blind

def summarize(long_df: pd.DataFrame) -> dict:
    if long_df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(long_df)),
        "n_ts": int(long_df["ts"].nunique()),
        "symbols": sorted(long_df["symbol"].unique().tolist()),
        "start_ts": str(long_df["ts"].min()),
        "end_ts": str(long_df["ts"].max()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="01_data/raw")
    ap.add_argument("--out_dir", type=str, default="01_data")
    ap.add_argument("--symbols", type=str, nargs="*", default=SYMBOLS_DEFAULT)
    ap.add_argument("--dev_start", type=str, default="2021-01-01 00:00:00+00:00")
    ap.add_argument("--dev_end", type=str, default="2025-01-01 00:00:00+00:00")
    ap.add_argument("--blind_start", type=str, default="2025-01-02 00:00:00+00:00")
    ap.add_argument("--blind_end", type=str, default="2026-01-30 23:59:59+00:00")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sym in args.symbols:
        p = raw_dir / f"{sym}_15m.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        print(f"[LOAD] {p}")
        frames.append(load_one(p, sym))

    long_all = align_intersection(frames)
    wide_all = make_wide(long_all)

    dev_long, blind_long = split_by_date(long_all, args.dev_start, args.dev_end, args.blind_start, args.blind_end)
    dev_wide = make_wide(dev_long) if not dev_long.empty else pd.DataFrame()
    blind_wide = make_wide(blind_long) if not blind_long.empty else pd.DataFrame()

    # save
    (out_dir / "universe_long_all.parquet").write_bytes(long_all.to_parquet(index=False))
    wide_all.to_parquet(out_dir / "universe_wide_all.parquet")

    (out_dir / "universe_long_dev.parquet").write_bytes(dev_long.to_parquet(index=False))
    (out_dir / "universe_long_blind.parquet").write_bytes(blind_long.to_parquet(index=False))

    dev_wide.to_parquet(out_dir / "universe_wide_dev.parquet")
    blind_wide.to_parquet(out_dir / "universe_wide_blind.parquet")

    summary = {
        "all": summarize(long_all),
        "dev": summarize(dev_long),
        "blind": summarize(blind_long),
        "paths": {
            "universe_long_all": str(out_dir / "universe_long_all.parquet"),
            "universe_wide_all": str(out_dir / "universe_wide_all.parquet"),
            "universe_long_dev": str(out_dir / "universe_long_dev.parquet"),
            "universe_long_blind": str(out_dir / "universe_long_blind.parquet"),
            "universe_wide_dev": str(out_dir / "universe_wide_dev.parquet"),
            "universe_wide_blind": str(out_dir / "universe_wide_blind.parquet"),
        }
    }

    summary_path = out_dir / "summary_universe.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[DONE] summary saved:", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
