# -*- coding: utf-8 -*-
"""
summary_yearly_performance.py

Summarize per-symbol and overall yearly performance from position_curve_oos.parquet.
Outputs a CSV (and optional Markdown) report with metrics aligned to v1 backtest stats.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def sharpe_like(lr_1: np.ndarray) -> float:
    lr = np.asarray(lr_1, dtype=float)
    m = np.nanmean(lr)
    s = np.nanstd(lr)
    ann = np.sqrt(365 * 24 * 4)  # 15m -> 96 bars/day
    return float(m / (s + EPS) * ann)


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + EPS) - 1.0
    return float(np.nanmin(dd))


def calmar_like(total_ret: float, mdd: float) -> float:
    return float((total_ret + EPS) / (abs(mdd) + EPS))


def summarize_slice(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    lr = df["strat_lr"].astype(float).to_numpy()
    eq = np.exp(np.nancumsum(lr))
    total_ret = float(eq[-1] - 1.0)
    mdd = max_drawdown(eq)

    turnover = df["turnover"].astype(float).to_numpy()
    fee = df["fee"].astype(float).to_numpy()
    gross = df["pos"].astype(float).to_numpy() * df["lr_1"].astype(float).to_numpy()
    gross_sum = float(np.nansum(gross))
    fee_sum = float(np.nansum(fee))
    fee_share = float(fee_sum / (abs(gross_sum) + fee_sum + EPS))

    return {
        "rows": int(len(df)),
        "total_return": total_ret,
        "sharpe_like": sharpe_like(lr),
        "max_drawdown": mdd,
        "calmar_like": calmar_like(total_ret, mdd),
        "turnover_per_bar": float(np.nanmean(turnover)),
        "avg_exposure": float(np.nanmean(df["pos"].astype(float).to_numpy())),
        "fee_share": fee_share,
    }


def summarize_overall(lr_df: pd.DataFrame) -> dict:
    if lr_df.empty:
        return {}
    lr_eq = lr_df.mean(axis=1).to_numpy(dtype=float)
    eq = np.exp(np.nancumsum(lr_eq))
    total_ret = float(eq[-1] - 1.0)
    mdd = max_drawdown(eq)
    return {
        "rows": int(len(lr_df)),
        "total_return": total_ret,
        "sharpe_like": sharpe_like(lr_eq),
        "max_drawdown": mdd,
        "calmar_like": calmar_like(total_ret, mdd),
    }


def build_yearly_report(curve: pd.DataFrame) -> pd.DataFrame:
    curve["year"] = curve["ts"].dt.year
    rows = []

    for (sym, year), df in curve.groupby(["symbol", "year"]):
        metrics = summarize_slice(df)
        if metrics:
            rows.append({"scope": sym, "year": int(year), **metrics})

    # Overall per year (equal-weighted across symbols)
    for year, df in curve.groupby("year"):
        lr_df = df.pivot(index="ts", columns="symbol", values="strat_lr").fillna(0.0)
        metrics = summarize_overall(lr_df)
        if metrics:
            rows.append({"scope": "ALL", "year": int(year), **metrics})

    # Overall across all years
    lr_df_all = curve.pivot(index="ts", columns="symbol", values="strat_lr").fillna(0.0)
    metrics_all = summarize_overall(lr_df_all)
    if metrics_all:
        rows.append({"scope": "ALL", "year": "ALL", **metrics_all})

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position_curve", type=str, default="artifacts/v1/backtest/position_curve_oos.parquet")
    ap.add_argument("--out_csv", type=str, default="04_backtest/yearly_performance.csv")
    ap.add_argument("--out_md", type=str, default="04_backtest/yearly_performance.md")
    args = ap.parse_args()

    curve = pd.read_parquet(args.position_curve)
    curve["ts"] = pd.to_datetime(curve["ts"], utc=True)
    curve = curve.sort_values(["symbol", "ts"]).reset_index(drop=True)

    report = build_yearly_report(curve)
    report = report.sort_values(["scope", "year"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Yearly Performance Report (v1)\n\n")
        headers = list(report.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in report.iterrows():
            vals = [str(row[h]) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

    print(f"[DONE] wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
