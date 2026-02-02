# -*- coding: utf-8 -*-
"""
download_ohlcv_binance_vision.py

Download 15m OHLCV from data.binance.vision (spot) without using api.binance.com.
Supports monthly klines with daily fallback when monthly files are missing.

Output:
  01_data/raw/{SYMBOL}_15m.parquet
  01_data/audits/audit_{SYMBOL}_15m.json
"""

import argparse
import calendar
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
import zipfile

import numpy as np
import pandas as pd
import urllib.error
import urllib.request


BASE_URL = "https://data.binance.vision/data/spot"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-12-31"
DEFAULT_SYMBOLS = ["XRPUSDT", "TRXUSDT", "UNIUSDT"]


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime

    @classmethod
    def from_strings(cls, start: str, end: str) -> "DateRange":
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        if end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")
        return cls(start=start_dt, end=end_dt)


def iter_months(start: datetime, end: datetime) -> Iterable[tuple[int, int]]:
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    last = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while cur <= last:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)


def iter_days(year: int, month: int) -> Iterable[datetime]:
    days = calendar.monthrange(year, month)[1]
    for day in range(1, days + 1):
        yield datetime(year, month, day, tzinfo=timezone.utc)


def is_15m_grid(ts: pd.DatetimeIndex) -> pd.Series:
    return (ts.minute % 15 == 0) & (ts.second == 0) & (ts.microsecond == 0)


def audit_df(df: pd.DataFrame, symbol: str) -> dict:
    a = {"symbol": symbol}
    if df.empty:
        a["rows"] = 0
        return a

    a["rows"] = int(len(df))
    a["start_ts"] = str(df.index.min())
    a["end_ts"] = str(df.index.max())

    grid_ok = is_15m_grid(df.index)
    a["n_off_grid"] = int((~grid_ok).sum())
    a["pct_off_grid"] = float((~grid_ok).mean())

    a["n_duplicates"] = int(df.index.duplicated().sum())

    diffs = df.index.to_series().diff().dropna()
    expected = pd.Timedelta(minutes=15)
    gaps = diffs[diffs != expected]
    a["n_gaps"] = int(len(gaps))
    a["max_gap_minutes"] = float(gaps.max().total_seconds() / 60) if len(gaps) else 0.0

    bad = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    a["n_bad_ohlc"] = int(bad.sum())
    a["pct_bad_ohlc"] = float(bad.mean())

    vol = df["volume"].to_numpy()
    zv = (vol <= 0) | (~np.isfinite(vol))
    a["pct_zero_or_nan_volume"] = float(zv.mean())

    return a


def download_zip(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return True


def read_zip_csv(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        with zf.open(names[0]) as fh:
            data = fh.read()
    return pd.read_csv(io.BytesIO(data), header=None, names=KLINE_COLUMNS)


def load_month(symbol: str, timeframe: str, year: int, month: int, cache_dir: str) -> pd.DataFrame:
    ym = f"{year:04d}-{month:02d}"
    monthly_name = f"{symbol}-{timeframe}-{ym}.zip"
    monthly_url = f"{BASE_URL}/monthly/klines/{symbol}/{timeframe}/{monthly_name}"
    monthly_path = os.path.join(cache_dir, "monthly", symbol, timeframe, monthly_name)
    has_month = download_zip(monthly_url, monthly_path)
    if has_month:
        return read_zip_csv(monthly_path)

    daily_frames = []
    for day in iter_days(year, month):
        dstr = day.strftime("%Y-%m-%d")
        daily_name = f"{symbol}-{timeframe}-{dstr}.zip"
        daily_url = f"{BASE_URL}/daily/klines/{symbol}/{timeframe}/{daily_name}"
        daily_path = os.path.join(cache_dir, "daily", symbol, timeframe, daily_name)
        has_day = download_zip(daily_url, daily_path)
        if not has_day:
            continue
        df = read_zip_csv(daily_path)
        if not df.empty:
            daily_frames.append(df)
    if not daily_frames:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    return pd.concat(daily_frames, ignore_index=True)


def process_to_df(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    df = df.loc[(df["ts"] >= start) & (df["ts"] <= end)]
    df = df.set_index("ts")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grid_ok = is_15m_grid(df.index)
    df = df.loc[grid_ok]

    bad = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    df = df.loc[~bad]

    return df[["open", "high", "low", "close", "volume"]]


def fetch_symbol(symbol: str, timeframe: str, date_range: DateRange, cache_dir: str) -> pd.DataFrame:
    frames = []
    for year, month in iter_months(date_range.start, date_range.end):
        df = load_month(symbol, timeframe, year, month, cache_dir)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    return process_to_df(merged, date_range.start, date_range.end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME)
    ap.add_argument("--start_date", type=str, default=DEFAULT_START)
    ap.add_argument("--end_date", type=str, default=DEFAULT_END)
    ap.add_argument("--data_dir", type=str, default="01_data/raw")
    ap.add_argument("--audit_dir", type=str, default="01_data/audits")
    ap.add_argument("--cache_dir", type=str, default="01_data/binance_vision_cache")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.audit_dir, exist_ok=True)

    date_range = DateRange.from_strings(args.start_date, args.end_date)
    for symbol in args.symbols:
        print(f"\n=== Fetch {symbol} {args.timeframe} via data.binance.vision ===")
        df = fetch_symbol(symbol, args.timeframe, date_range, args.cache_dir)
        if df.empty:
            print(f"[WARN] {symbol}: no data found")
            continue

        out_path = os.path.join(args.data_dir, f"{symbol}_{args.timeframe}.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")

        aud = audit_df(df, symbol)
        aud_path = os.path.join(args.audit_dir, f"audit_{symbol}_{args.timeframe}.json")
        with open(aud_path, "w", encoding="utf-8") as f:
            json.dump(aud, f, ensure_ascii=False, indent=2)

        print(f"[OK] saved: {out_path}")
        print(f"[OK] audit: {aud_path}")
        print(f"[INFO] rows={len(df)}, gaps={aud['n_gaps']}, off_grid={aud['pct_off_grid']:.4f}, bad_ohlc={aud['pct_bad_ohlc']:.6f}")


if __name__ == "__main__":
    main()
