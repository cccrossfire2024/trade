import argparse
import ccxt
import pandas as pd
import time
import json
import numpy as np
import os
from datetime import datetime, timezone

# ================= 配置区域 =================
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']  # CCXT 通常用这个写法
TIMEFRAME = '15m'
START_DATE = '2021-01-01 00:00:00'  # UTC
END_DATE = None  # None 表示拉到最新
DATA_DIR = '01_data/raw'
AUDIT_DIR = '01_data/audits'
EXCHANGE_ID = "binance"
MARKET_TYPE = "future"

# 代理（可选）
PROXIES = None
# ===========================================

TF_MS = 15 * 60 * 1000

def get_exchange(exchange_id: str, market_type: str):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange_id: {exchange_id}")
    ex = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,
        "proxies": PROXIES,
        "options": {
            "defaultType": market_type,
        },
    })
    ex.load_markets()
    return ex

def utc_ms(dt_str: str) -> int:
    dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def is_15m_grid(ts: pd.DatetimeIndex) -> pd.Series:
    return (ts.minute % 15 == 0) & (ts.second == 0) & (ts.microsecond == 0)

def fetch_ohlcv_all(ex, symbol, timeframe, since_ms, end_ms=None, limit=1000, max_retries=8):
    """
    分页拉取：每次最多 limit 根，直到 end_ms 或没有新数据
    """
    out = []
    cur = since_ms
    while True:
        for k in range(max_retries):
            try:
                chunk = ex.fetch_ohlcv(symbol, timeframe, since=cur, limit=limit)
                break
            except Exception as e:
                if k == max_retries - 1:
                    raise
                time.sleep(min(2 ** k, 30))
        if not chunk:
            break

        out.extend(chunk)
        last_ts = chunk[-1][0]

        # 终止条件：到达 end_ms 或者没有推进
        if end_ms is not None and last_ts >= end_ms:
            break
        if last_ts <= cur:
            break

        # 下一页：直接跳到下一根 bar 起点（更稳）
        cur = last_ts + TF_MS

        # CCXT 限速
        time.sleep(ex.rateLimit / 1000)

        # 控制台进度
        print(f"[{symbol}] downloaded until {pd.to_datetime(last_ts, unit='ms', utc=True)}", end="\r")

        # 如果已经接近最新（避免死循环）
        now_ms = ex.milliseconds()
        if end_ms is None and last_ts >= now_ms - TF_MS:
            break

    print()
    return out

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
        (df["high"] < df[["open","close"]].max(axis=1)) |
        (df["low"]  > df[["open","close"]].min(axis=1)) |
        (df["high"] < df["low"])
    )
    a["n_bad_ohlc"] = int(bad.sum())
    a["pct_bad_ohlc"] = float(bad.mean())

    vol = df["volume"].to_numpy()
    zv = (vol <= 0) | (~np.isfinite(vol))
    a["pct_zero_or_nan_volume"] = float(zv.mean())

    return a

def process_to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts_ms","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"]).set_index("ts").sort_index()

    # 去重 + 类型
    df = df[~df.index.duplicated(keep="last")]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 丢掉非15m网格（不补）
    grid_ok = is_15m_grid(df.index)
    df = df.loc[grid_ok]

    # 丢掉坏OHLC
    bad = (
        (df["high"] < df[["open","close"]].max(axis=1)) |
        (df["low"]  > df[["open","close"]].min(axis=1)) |
        (df["high"] < df["low"])
    )
    df = df.loc[~bad]

    return df

def safe_name(symbol: str) -> str:
    return symbol.replace("/", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--timeframe", type=str, default=TIMEFRAME)
    ap.add_argument("--start_date", type=str, default=START_DATE)
    ap.add_argument("--end_date", type=str, default=END_DATE)
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--audit_dir", type=str, default=AUDIT_DIR)
    ap.add_argument("--exchange", type=str, default=EXCHANGE_ID)
    ap.add_argument("--market_type", type=str, default=MARKET_TYPE)
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.audit_dir, exist_ok=True)

    ex = get_exchange(args.exchange, args.market_type)

    since_ms = utc_ms(args.start_date)
    end_ms = utc_ms(args.end_date) if args.end_date else None

    for sym in args.symbols:
        print(f"\n=== Fetch {sym} {args.timeframe} ({args.market_type}) ===")
        # 简单验证：确保 market 存在
        if sym not in ex.markets:
            print(f"[WARN] {sym} not in markets. Available example: {list(ex.markets.keys())[:5]}")
            continue

        data = fetch_ohlcv_all(ex, sym, args.timeframe, since_ms, end_ms=end_ms, limit=1000)
        if not data or len(data) < 1000:
            print(f"[WARN] {sym}: too few rows ({0 if not data else len(data)})")
            continue

        df = process_to_df(data)
        aud = audit_df(df, sym)

        out_path = os.path.join(args.data_dir, f"{safe_name(sym)}_{args.timeframe}.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")

        aud_path = os.path.join(args.audit_dir, f"audit_{safe_name(sym)}_{args.timeframe}.json")
        with open(aud_path, "w", encoding="utf-8") as f:
            json.dump(aud, f, ensure_ascii=False, indent=2)

        print(f"[OK] saved: {out_path}")
        print(f"[OK] audit: {aud_path}")
        print(f"[INFO] rows={len(df)}, gaps={aud['n_gaps']}, off_grid={aud['pct_off_grid']:.4f}, bad_ohlc={aud['pct_bad_ohlc']:.6f}")

if __name__ == "__main__":
    main()
