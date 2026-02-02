# -*- coding: utf-8 -*-
"""
make_blind_gate_oos_v2_for_v1.py

用 v2 保存的每个窗口 LightGBM 模型（.txt）对 blind gate_samples_v2 进行推断，
产出 v1 兼容的 blind gate_oos 文件，包含：
- symbol, event_id, p_hat, gate_pos0_hat

注意：这里的“窗口选择”使用 event 的 t0_ts 来选取离它最近的 winXXX 模型（按文件序号近似）。
如果你希望更严格按训练窗结束日期选模型，我也可以再升级版本。
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import lightgbm as lgb

EPS = 1e-12

def to_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def pick_feature_cols(df):
    drop = {"symbol","event_id","t0_ts","t1_ts","twarn_ts","label","mfe_atr"}
    cols = [c for c in df.columns if c not in drop]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def load_models(models_dir: Path):
    # 优先读 .txt（你现在就有）
    files = sorted(models_dir.glob("lgbm_gate_win*.txt"))
    if not files:
        raise FileNotFoundError(f"no lgbm_gate_win*.txt in {models_dir}")
    models = []
    for p in files:
        m = re.search(r"win(\d+)", p.stem)
        win = int(m.group(1)) if m else len(models)
        booster = lgb.Booster(model_file=str(p))
        models.append((win, booster))
    models.sort(key=lambda x: x[0])
    return models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind_gate_samples", type=str, required=True)
    ap.add_argument("--models_dir", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--thr_low", type=float, default=0.50)
    ap.add_argument("--thr_high", type=float, default=0.65)
    args = ap.parse_args()

    gs = pd.read_parquet(args.blind_gate_samples)
    if "t0_ts" not in gs.columns:
        raise ValueError("blind gate_samples missing t0_ts")
    gs["t0_ts"] = to_utc(gs["t0_ts"])
    gs = gs.dropna(subset=["t0_ts"]).sort_values("t0_ts").reset_index(drop=True)

    feat_cols = pick_feature_cols(gs)
    if len(feat_cols) < 5:
        raise ValueError(f"too few features: {len(feat_cols)}")

    models = load_models(Path(args.models_dir))
    n_models = len(models)

    # 简单策略：按时间排序后均匀切成 n_models 段，每段用一个模型
    # （避免你 v1 脚本内部对模型pattern/加载方式的各种坑）
    idx = np.arange(len(gs))
    bins = np.linspace(0, len(gs), n_models + 1).astype(int)

    p_hat = np.zeros(len(gs), dtype=float)

    X = gs[feat_cols]
    for k in range(n_models):
        i0, i1 = bins[k], bins[k+1]
        if i1 <= i0:
            continue
        _, booster = models[k]
        p_hat[i0:i1] = booster.predict(X.iloc[i0:i1], num_iteration=booster.best_iteration)

    out = gs[["symbol","event_id","t0_ts"]].copy()
    out["p_hat"] = p_hat.astype(float)
    out["gate_pos0_hat"] = np.where(
        out["p_hat"] >= args.thr_high, 1.0,
        np.where(out["p_hat"] >= args.thr_low, 0.5, 0.0)
    ).astype("float32")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_path, index=False)
    print("[DONE] saved:", args.out_path)
    print("rows:", len(out), "features:", len(feat_cols), "models:", n_models)

if __name__ == "__main__":
    main()
