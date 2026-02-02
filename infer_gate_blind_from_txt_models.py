# -*- coding: utf-8 -*-
"""
analyze_gate_v1_vs_v2_blind.py

比较 v1 vs v2 在 blind 上 Gate 预测质量：
- AUC(label!=NONE)
- 分位桶单调性（p_hat越高，win_rate/mfe_atr越高）
- top/bottom lift

修复：
- decile_table 的 bin 是 pandas Interval，写 JSON 前转为 str，避免 TypeError
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def to_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def auc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC without sklearn (handles ties)."""
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask].astype(np.int32)
    y_score = y_score[mask].astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # rank with ties
    ranks = np.empty_like(y_score_sorted, dtype=np.float64)
    i = 0
    r = 1.0
    while i < len(y_score_sorted):
        j = i
        while j + 1 < len(y_score_sorted) and y_score_sorted[j + 1] == y_score_sorted[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[i:j + 1] = avg_rank
        r += (j - i + 1)
        i = j + 1

    sum_ranks_pos = ranks[y_true_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg + EPS)
    return float(auc)


def find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_gate_samples(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    sym_col = find_first_col(df, ["symbol", "sym"])
    eid_col = find_first_col(df, ["event_id", "eid", "event"])
    t0_col = find_first_col(df, ["t0_ts", "t0", "t0_time", "t0_datetime", "t0_dt"])
    if sym_col is None or eid_col is None or t0_col is None:
        raise ValueError(
            f"gate_samples missing required columns. need symbol/event_id/t0. got: {list(df.columns)[:30]}..."
        )

    label_col = find_first_col(df, ["label", "gate_label"])
    mfe_col = find_first_col(df, ["mfe_atr", "mfe_atr_32", "mfe_atr_gate", "mfe_atr_future"])

    out_cols = [sym_col, eid_col, t0_col]
    if label_col: out_cols.append(label_col)
    if mfe_col: out_cols.append(mfe_col)

    out = df[out_cols].copy()
    out = out.rename(columns={sym_col: "symbol", eid_col: "event_id", t0_col: "t0_ts"})
    out["t0_ts"] = to_utc(out["t0_ts"])

    if label_col:
        out = out.rename(columns={label_col: "label"})
    else:
        out["label"] = "NONE"

    if mfe_col:
        out = out.rename(columns={mfe_col: "mfe_atr"})
        out["mfe_atr"] = pd.to_numeric(out["mfe_atr"], errors="coerce")
    else:
        out["mfe_atr"] = np.nan

    return out


def load_gate_oos(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    sym_col = find_first_col(df, ["symbol", "sym"])
    eid_col = find_first_col(df, ["event_id", "eid", "event"])
    p_col = find_first_col(df, ["p_hat", "p", "score", "pred"])

    if sym_col is None or eid_col is None:
        raise ValueError(f"gate_oos missing symbol/event_id. got: {list(df.columns)[:30]}...")

    out_cols = [sym_col, eid_col]
    if p_col: out_cols.append(p_col)

    out = df[out_cols].copy()
    out = out.rename(columns={sym_col: "symbol", eid_col: "event_id"})
    if p_col:
        out = out.rename(columns={p_col: "p_hat"})
        out["p_hat"] = pd.to_numeric(out["p_hat"], errors="coerce")
    else:
        out["p_hat"] = 0.0

    return out


def decile_table(df: pd.DataFrame, score_col="p_hat") -> pd.DataFrame:
    x = df.copy()
    x = x[np.isfinite(x[score_col].values)]
    if len(x) < 200:
        return pd.DataFrame()

    edges = np.nanquantile(x[score_col].values, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame()

    x["bin"] = pd.cut(x[score_col], bins=edges, include_lowest=True, duplicates="drop")
    g = x.groupby("bin", observed=True)

    tab = pd.DataFrame({
        "count": g.size(),
        "p_hat_mean": g[score_col].mean(),
        "win_rate(label!=NONE)": g["y"].mean(),
        "mfe_atr_mean": g["mfe_atr"].mean(),
        "mfe_atr_median": g["mfe_atr"].median(),
    }).reset_index()

    # ✅ 修复点：Interval -> str，避免 json dumps 报错
    tab["bin"] = tab["bin"].astype(str)

    if len(tab) >= 2:
        top = tab.iloc[-1]
        bot = tab.iloc[0]
        tab["lift_vs_bottom_winrate"] = float(top["win_rate(label!=NONE)"] / (bot["win_rate(label!=NONE)"] + EPS))
        if np.isfinite(bot["mfe_atr_mean"]) and (abs(bot["mfe_atr_mean"]) > EPS):
            tab["lift_vs_bottom_mfe_mean"] = float(top["mfe_atr_mean"] / (bot["mfe_atr_mean"] + EPS))
        else:
            tab["lift_vs_bottom_mfe_mean"] = np.nan

    return tab


def summarize_one(name: str, gate_samples: pd.DataFrame, gate_oos: pd.DataFrame) -> dict:
    m = gate_samples.merge(gate_oos, on=["symbol", "event_id"], how="left")
    m["p_hat"] = pd.to_numeric(m["p_hat"], errors="coerce").fillna(0.0)

    m["y"] = (m["label"].astype(str) != "NONE").astype(np.int32)

    coverage = float(np.isfinite(m["p_hat"]).mean())
    nonzero = float((m["p_hat"].values != 0).mean())
    auc = auc_fast(m["y"].values, m["p_hat"].values) if m["y"].nunique() > 1 else float("nan")

    tab = decile_table(m, "p_hat")
    top_bin = tab.iloc[-1].to_dict() if not tab.empty else {}
    bot_bin = tab.iloc[0].to_dict() if not tab.empty else {}

    mono = None
    if not tab.empty and len(tab) >= 3:
        wr = tab["win_rate(label!=NONE)"].values.astype(float)
        rk = np.argsort(np.argsort(wr))
        mono = float(np.corrcoef(np.arange(len(wr)), rk)[0, 1])

    return {
        "name": name,
        "rows": int(len(m)),
        "coverage_p_hat": coverage,
        "p_hat_nonzero_rate": nonzero,
        "auc(label!=NONE)": auc,
        "monotonicity_proxy": mono,
        "deciles": tab.to_dict(orient="records") if not tab.empty else [],
        "top_decile": top_bin,
        "bottom_decile": bot_bin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_samples_blind_v1", required=True)
    ap.add_argument("--gate_oos_blind_v1", required=True)
    ap.add_argument("--gate_samples_blind_v2", required=True)
    ap.add_argument("--gate_oos_blind_v2", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gs1 = load_gate_samples(args.gate_samples_blind_v1)
    go1 = load_gate_oos(args.gate_oos_blind_v1)
    gs2 = load_gate_samples(args.gate_samples_blind_v2)
    go2 = load_gate_oos(args.gate_oos_blind_v2)

    res1 = summarize_one("v1", gs1, go1)
    res2 = summarize_one("v2", gs2, go2)

    comp = {
        "blind_gate_quality": {"v1": res1, "v2": res2},
        "how_to_read": [
            "看 v2 是否在 blind 上提升 AUC、分位桶单调性、top/bottom lift",
            "若 AUC/单调性没提升但收益提升，通常是映射/执行器更激进导致，而非特征贡献",
        ],
    }

    (out_dir / "compare_gate_blind_summary.json").write_text(
        json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if res1.get("deciles"):
        pd.DataFrame(res1["deciles"]).to_csv(out_dir / "v1_deciles.csv", index=False, encoding="utf-8-sig")
    if res2.get("deciles"):
        pd.DataFrame(res2["deciles"]).to_csv(out_dir / "v2_deciles.csv", index=False, encoding="utf-8-sig")

    print("[DONE] saved:")
    print(" -", out_dir / "compare_gate_blind_summary.json")
    if res1.get("deciles"): print(" -", out_dir / "v1_deciles.csv")
    if res2.get("deciles"): print(" -", out_dir / "v2_deciles.csv")

    print("\n[BLIND GATE QUALITY]")
    print("v1: auc=%.4f coverage=%.3f nonzero=%.3f mono=%s" %
          (res1["auc(label!=NONE)"], res1["coverage_p_hat"], res1["p_hat_nonzero_rate"], str(res1["monotonicity_proxy"])))
    print("v2: auc=%.4f coverage=%.3f nonzero=%.3f mono=%s" %
          (res2["auc(label!=NONE)"], res2["coverage_p_hat"], res2["p_hat_nonzero_rate"], str(res2["monotonicity_proxy"])))


if __name__ == "__main__":
    main()
