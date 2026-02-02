# -*- coding: utf-8 -*-
"""
convert_lgbm_txt_models_to_pkl.py

把 LightGBM 的 .txt 模型批量转存成 .pkl（joblib），放在同一目录下。
这样老脚本（通常只找 .pkl/.joblib）就能识别。

用法：
python convert_lgbm_txt_models_to_pkl.py --models_dir models_gate_v2\\models
"""

import argparse
from pathlib import Path
import re

import joblib
import lightgbm as lgb


def parse_win_id(stem: str):
    m = re.search(r"win(\d+)", stem)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, required=True)
    args = ap.parse_args()

    d = Path(args.models_dir)
    if not d.exists():
        raise FileNotFoundError(d)

    txts = sorted(d.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No .txt models found in {d}")

    n = 0
    for p in txts:
        booster = lgb.Booster(model_file=str(p))

        stem = p.stem
        win = parse_win_id(stem)

        # 1) 同名 .pkl：lgbm_gate_win000.pkl（最可能被 glob 到）
        out1 = d / f"{stem}.pkl"
        joblib.dump(booster, out1)

        # 2) 兼容另一种老命名：model_win000.pkl（以防 v1 写死前缀）
        if win is not None:
            out2 = d / f"model_win{win}.pkl"
            if out2 != out1:
                joblib.dump(booster, out2)

        n += 1

    print(f"[DONE] converted {n} txt models into pkl in: {d}")
    print("Example files:")
    for ex in list(d.glob("*.pkl"))[:5]:
        print(" -", ex.name)


if __name__ == "__main__":
    main()
