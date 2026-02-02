@echo off
set ROOT=%~dp0..
set PY=python

%PY% "%ROOT%\04_backtest\backtest_oos.py" ^
  --universe_long "%ROOT%\01_data\universe_long_all.parquet" ^
  --risk_states "%ROOT%\artifacts\v1\models\risk\risk_states_full.parquet" ^
  --gate_samples "%ROOT%\artifacts\v1\features\gate_samples.parquet" ^
  --gate_model "%ROOT%\artifacts\v1\models\gate\models\lgbm_gate_dev.txt" ^
  --gate_feature_list "%ROOT%\artifacts\v1\models\gate\gate_features.txt" ^
  --gate_thresholds "%ROOT%\artifacts\v1\models\gate\summary_gate_dev.json" ^
  --bt_start 2025-01-01 ^
  --tf 15 ^
  --fee 0.0004 ^
  --floor 0.2 ^
  --min_bars 16 ^
  --pass_tag pass20
