@echo off
set ROOT=%~dp0..
set PY=python

%PY% "%ROOT%\v1\train_risk_lgbm_dd16_pipeline.py" ^
  --risk_bars_features "%ROOT%\artifacts\v1\features\risk_bars_features.parquet" ^
  --out_dir "%ROOT%\artifacts\v1\models\risk" ^
  --train_end 2025-01-01 ^
  --tf 15
