@echo off
set ROOT=%~dp0..
set PY=python

%PY% "%ROOT%\v1\train_bar_gate_model_lgbm.py" ^
  --gate_samples "%ROOT%\artifacts\v1\features\gate_samples.parquet" ^
  --out_dir "%ROOT%\artifacts\v1\models\gate" ^
  --train_end 2025-01-01 ^
  --tf 15
