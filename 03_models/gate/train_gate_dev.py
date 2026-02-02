# -*- coding: utf-8 -*-
"""
train_gate_dev.py

Train Gate classifier on dev split (t0_ts < train_end) and save:
- model (LightGBM txt)
- feature list (txt)
- thresholds (json) for pass10/20/30/40/50
- dev OOF-style predictions (in-sample for now)
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


EXCLUDE_COLS = {
    "symbol", "event_id", "twarn_ts", "t0_ts", "t1_ts",
    "label", "mfe_atr", "y"
}


def load_feature_list(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def pick_feature_cols(df: pd.DataFrame, feature_list_path: Path | None) -> list[str]:
    if feature_list_path and feature_list_path.exists():
        cols = load_feature_list(feature_list_path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature list has missing columns: {missing}")
        return cols

    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def gate_pos_from_threshold(p: np.ndarray, threshold: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    pos = (p - threshold) / (1.0 - threshold + 1e-12)
    return np.clip(pos, 0.0, 1.0)


def train_model(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> lgb.LGBMClassifier:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    clf = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    # time-consistent split (last 15% as valid)
    split = int(len(X) * 0.85)
    if len(X) - split < 200:
        split = max(int(len(X) * 0.8), len(X) - 200)

    X_fit, y_fit = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    clf.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_samples", type=str, default="artifacts/v1/features/gate_samples.parquet")
    ap.add_argument("--out_dir", type=str, default="artifacts/v1/models/gate")
    ap.add_argument("--feature_list", type=str, default="")
    ap.add_argument("--train_end", type=str, default="2025-01-01")
    ap.add_argument("--tf", type=int, default=15)

    ap.add_argument("--min_train_rows", type=int, default=800)

    # LGB params
    ap.add_argument("--num_leaves", type=int, default=64)
    ap.add_argument("--min_data_in_leaf", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--n_estimators", type=int, default=4000)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.7)
    ap.add_argument("--reg_lambda", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.gate_samples)
    df["t0_ts"] = pd.to_datetime(df["t0_ts"], utc=True)
    df = df.sort_values("t0_ts").reset_index(drop=True)

    train_end = pd.to_datetime(args.train_end, utc=True)
    dev = df[df["t0_ts"] < train_end].copy()
    if len(dev) < args.min_train_rows:
        raise ValueError(f"Too few rows for training: {len(dev)} < {args.min_train_rows}")

    dev["y"] = (dev["label"].astype(str) != "NONE").astype(int)

    feature_list_path = Path(args.feature_list) if args.feature_list else None
    feat_cols = pick_feature_cols(dev, feature_list_path)
    if len(feat_cols) < 10:
        raise ValueError(f"Too few feature cols: {len(feat_cols)}")

    X = dev[feat_cols]
    y = dev["y"].astype(int)

    clf = train_model(X, y, args)

    model_path = out_dir / "models" / "lgbm_gate_dev.txt"
    clf.booster_.save_model(str(model_path))

    # store feature list for reproducibility
    feature_path = out_dir / "gate_features.txt"
    feature_path.write_text("\n".join(feat_cols) + "\n", encoding="utf-8")

    # thresholds from dev predictions
    p_hat = clf.predict_proba(X)[:, 1]
    quantiles = {}
    for q in (10, 20, 30, 40, 50):
        quantiles[f"pass{q}"] = float(np.nanquantile(p_hat, q / 100))

    pass20 = quantiles["pass20"]
    gate_pos0_hat = gate_pos_from_threshold(p_hat, pass20)

    oof = dev[["symbol", "event_id", "t0_ts", "t1_ts", "label", "mfe_atr"]].copy()
    oof["p_hat"] = p_hat
    oof["gate_pos0_hat"] = gate_pos0_hat
    oof_path = out_dir / "gate_oof_dev.parquet"
    oof.to_parquet(oof_path, index=False)

    summary = {
        "gate_samples": args.gate_samples,
        "train_end": args.train_end,
        "tf": args.tf,
        "rows_train": int(len(dev)),
        "model_path": str(model_path),
        "feature_list": str(feature_path),
        "thresholds": quantiles,
        "oof_path": str(oof_path),
        "n_features": int(len(feat_cols)),
        "feature_preview": feat_cols[:25],
    }
    summary_path = out_dir / "summary_gate_dev.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE]", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
