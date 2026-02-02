# -*- coding: utf-8 -*-
"""
debug_gate_v1_v2_blind_collapse.py

目的：
- 解释 v2 在 blind 上 AUC ~ 0.5、decile 不单调的原因
- 重点诊断：
  1) v2 预测是否“塌缩成常数”（全局&分月）
  2) v2 特征是否大量 NaN / 近似常数（导致走同一叶子）
  3) 评估 y 口径是否合理（label 分布）
  4) 不同 window 模型在 blind 上的输出分布差异（抽样检查）

输出：
- out_dir/summary_debug.json
- out_dir/v2_feature_missing_std.csv
- out_dir/v2_p_by_month.csv
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import lightgbm as lgb

EPS = 1e-12


def to_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def find_first(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_gate_samples(path):
    df = pd.read_parquet(path)
    sym = find_first(df, ["symbol", "sym"])
    eid = find_first(df, ["event_id", "eid", "event"])
    t0  = find_first(df, ["t0_ts", "t0", "t0_time", "t0_datetime", "t0_dt"])
    lab = find_first(df, ["label", "gate_label"])
    mfe = find_first(df, ["mfe_atr", "mfe_atr_32", "mfe_atr_gate"])

    if sym is None or eid is None or t0 is None:
        raise ValueError(f"gate_samples missing meta cols: got {list(df.columns)[:50]}")

    out = df.copy()
    out = out.rename(columns={sym:"symbol", eid:"event_id", t0:"t0_ts"})
    out["t0_ts"] = to_utc(out["t0_ts"])
    if out["t0_ts"].isna().any():
        raise ValueError("t0_ts has NaT")

    if lab is None:
        out["label"] = "NONE"
    else:
        out = out.rename(columns={lab:"label"})

    if mfe is None:
        out["mfe_atr"] = np.nan
    else:
        out = out.rename(columns={mfe:"mfe_atr"})
        out["mfe_atr"] = pd.to_numeric(out["mfe_atr"], errors="coerce")

    return out


def load_gate_oos(path):
    df = pd.read_parquet(path)
    sym = find_first(df, ["symbol", "sym"])
    eid = find_first(df, ["event_id", "eid", "event"])
    p   = find_first(df, ["p_hat", "p", "score", "pred"])
    if sym is None or eid is None:
        raise ValueError(f"gate_oos missing meta cols: got {list(df.columns)[:50]}")
    out = df[[sym, eid] + ([p] if p else [])].copy()
    out = out.rename(columns={sym:"symbol", eid:"event_id"})
    if p:
        out = out.rename(columns={p:"p_hat"})
        out["p_hat"] = pd.to_numeric(out["p_hat"], errors="coerce").fillna(0.0)
    else:
        out["p_hat"] = 0.0
    return out


def auc_fast(y_true, y_score):
    m = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[m].astype(np.int32)
    y_score = y_score[m].astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    yt = y_true[order]
    ys = y_score[order]

    ranks = np.empty_like(ys, dtype=np.float64)
    i = 0
    r = 1.0
    while i < len(ys):
        j = i
        while j+1 < len(ys) and ys[j+1] == ys[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[i:j+1] = avg_rank
        r += (j - i + 1)
        i = j + 1

    sum_ranks_pos = ranks[yt == 1].sum()
    auc = (sum_ranks_pos - n_pos*(n_pos+1)/2.0) / (n_pos*n_neg + EPS)
    return float(auc)


def parse_win_id(p: Path):
    m = re.search(r"(win|window)(\d+)", p.name)
    return int(m.group(2)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_samples_blind_v1", required=True)
    ap.add_argument("--gate_oos_blind_v1", required=True)
    ap.add_argument("--gate_samples_blind_v2", required=True)
    ap.add_argument("--gate_oos_blind_v2", required=True)
    ap.add_argument("--models_dir_v2", required=True, help="models_gate_v2/models")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--check_models", type=int, default=3, help="how many v2 windows to spot-check")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gs1 = load_gate_samples(args.gate_samples_blind_v1)
    go1 = load_gate_oos(args.gate_oos_blind_v1)
    gs2 = load_gate_samples(args.gate_samples_blind_v2)
    go2 = load_gate_oos(args.gate_oos_blind_v2)

    # merge
    m1 = gs1.merge(go1, on=["symbol","event_id"], how="left")
    m2 = gs2.merge(go2, on=["symbol","event_id"], how="left")
    m1["p_hat"] = pd.to_numeric(m1["p_hat"], errors="coerce").fillna(0.0)
    m2["p_hat"] = pd.to_numeric(m2["p_hat"], errors="coerce").fillna(0.0)

    # y def (current evaluation): label != NONE
    m1["y"] = (m1["label"].astype(str) != "NONE").astype(np.int32)
    m2["y"] = (m2["label"].astype(str) != "NONE").astype(np.int32)

    # global stats
    summary = {
        "v1": {
            "rows": int(len(m1)),
            "pos_rate(label!=NONE)": float(m1["y"].mean()),
            "p_hat_mean": float(m1["p_hat"].mean()),
            "p_hat_std": float(m1["p_hat"].std()),
            "auc(label!=NONE)": auc_fast(m1["y"].values, m1["p_hat"].values),
            "label_counts": m1["label"].value_counts().to_dict(),
        },
        "v2": {
            "rows": int(len(m2)),
            "pos_rate(label!=NONE)": float(m2["y"].mean()),
            "p_hat_mean": float(m2["p_hat"].mean()),
            "p_hat_std": float(m2["p_hat"].std()),
            "auc(label!=NONE)": auc_fast(m2["y"].values, m2["p_hat"].values),
            "label_counts": m2["label"].value_counts().to_dict(),
        }
    }

    # by month AUC & p_hat dispersion (v2)
    m2["ym"] = m2["t0_ts"].dt.to_period("M").astype(str)
    rows = []
    for ym, g in m2.groupby("ym"):
        rows.append({
            "ym": ym,
            "n": int(len(g)),
            "pos_rate": float(g["y"].mean()),
            "p_mean": float(g["p_hat"].mean()),
            "p_std": float(g["p_hat"].std()),
            "auc": auc_fast(g["y"].values, g["p_hat"].values) if g["y"].nunique()>1 else float("nan"),
        })
    bym = pd.DataFrame(rows).sort_values("ym")
    bym.to_csv(out_dir / "v2_p_by_month.csv", index=False, encoding="utf-8-sig")

    summary["v2_by_month"] = {
        "path": str(out_dir / "v2_p_by_month.csv"),
        "note": "如果很多月份 p_std 极小 / auc ~0.5，说明模型在 blind 上普遍塌缩"
    }

    # feature missing/std for v2 (use v2 model feature names)
    models_dir = Path(args.models_dir_v2)
    model_files = sorted(models_dir.glob("lgbm_gate_win*.txt"))
    if not model_files:
        model_files = sorted(models_dir.glob("lgbm_gate_window_*.txt"))
    if not model_files:
        raise FileNotFoundError(f"no v2 models in {models_dir}")

    # pick one representative model to get feature list
    b0 = lgb.Booster(model_file=str(model_files[0]))
    feats = b0.feature_name()

    # compute missing/std in v2 samples for feats
    fdf = gs2[feats].copy()
    miss_rate = fdf.isna().mean()
    std = fdf.astype("float32").std(skipna=True)
    out_feat = pd.DataFrame({"feature": feats, "missing_rate": miss_rate.values, "std": std.values})
    out_feat = out_feat.sort_values(["missing_rate","std"], ascending=[False, True])
    out_feat.to_csv(out_dir / "v2_feature_missing_std.csv", index=False, encoding="utf-8-sig")

    summary["v2_feature_health"] = {
        "path": str(out_dir / "v2_feature_missing_std.csv"),
        "top10_high_missing": out_feat.head(10).to_dict(orient="records"),
        "top10_low_std": out_feat.sort_values("std").head(10).to_dict(orient="records"),
        "note": "如果很多特征 missing_rate 很高 或 std 接近 0，会导致预测塌缩"
    }

    # spot-check a few windows on blind to see if ANY model has dispersion
    spot = []
    pick = model_files[:args.check_models] + model_files[-args.check_models:]
    # unique
    seen = set()
    pick2 = []
    for p in pick:
        if p.name not in seen:
            seen.add(p.name)
            pick2.append(p)
    pick = pick2

    X = gs2[feats].astype("float32")
    for mf in pick:
        b = lgb.Booster(model_file=str(mf))
        p = b.predict(X, num_iteration=b.best_iteration or -1)
        spot.append({
            "model": mf.name,
            "win_id": parse_win_id(mf),
            "p_mean": float(np.mean(p)),
            "p_std": float(np.std(p)),
            "p_min": float(np.min(p)),
            "p_max": float(np.max(p)),
        })
    summary["v2_model_spotcheck"] = spot

    # save summary
    (out_dir / "summary_debug.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[DONE] saved:")
    print(" -", out_dir / "summary_debug.json")
    print(" -", out_dir / "v2_feature_missing_std.csv")
    print(" -", out_dir / "v2_p_by_month.csv")
    print("\n[KEY]")
    print("v1 auc:", summary["v1"]["auc(label!=NONE)"], "p_std:", summary["v1"]["p_hat_std"])
    print("v2 auc:", summary["v2"]["auc(label!=NONE)"], "p_std:", summary["v2"]["p_hat_std"])
    print("If v2 p_std is tiny and features have high missing/low std -> feature collapse on blind.")


if __name__ == "__main__":
    main()
