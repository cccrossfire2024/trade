# Cleanup Report

## Moved (re-homed)

### Data
- `data.py` → `01_data/download_ohlcv_ccxt.py` (normalize location/name)
- `build_universe_panel.py` → `01_data/build_universe_panel.py`

### Features
- `extract_gmma_events.py` → `02_features/extract_gmma_events.py`
- `build_gate_risk_features_v1.py` → `02_features/build_gate_risk_features_v1.py`

### Models / Backtest
- `risk_state_machine_v1.py` → `03_models/risk/risk_state_machine_v1.py`
- `build_position_curve_gate_oos_only_v1logic.py` → `04_backtest/backtest_oos.py`

### Diagnose
- `analyze_gate_v1_vs_v2_blind.py` → `06_diagnose/analyze_gate_v1_vs_v2_blind.py`
- `debug_gate_v1_v2_blind_collapse.py` → `06_diagnose/debug_gate_v1_v2_blind_collapse.py`
- `sanity_suite_gate_pipeline.py` → `06_diagnose/sanity_suite_gate_pipeline.py`

### Archive
- `build_gate_risk_features_v1plus4.py` → `99_archive/legacy/build_gate_risk_features_v1plus4.py`
- `build_gate_risk_features_v2.py` → `99_archive/legacy/build_gate_risk_features_v2.py`
- `build_position_curve_gate_ml_v1.py` → `99_archive/legacy/build_position_curve_gate_ml_v1.py`
- `build_position_curve_gate_ml_v2.py` → `99_archive/legacy/build_position_curve_gate_ml_v2.py`
- `build_position_curve_v0.py` → `99_archive/legacy/build_position_curve_v0.py`
- `build_position_curve_v1_executor_from_gate_oos.py` → `99_archive/legacy/build_position_curve_v1_executor_from_gate_oos.py`
- `risk_state_machine_v0.py` → `99_archive/legacy/risk_state_machine_v0.py`
- `train_gate_lgbm_wfv.py` → `99_archive/legacy/train_gate_lgbm_wfv.py`
- `train_gate_lgbm_wfv_v2.py` → `99_archive/legacy/train_gate_lgbm_wfv_v2.py`
- `infer_gate_blind_from_txt_models.py` → `99_archive/legacy/infer_gate_blind_from_txt_models.py`
- `check.py` → `99_archive/legacy/check.py`
- `run_all_experiments_gate_risk_v3.py` → `99_archive/experiments/run_all_experiments_gate_risk_v3.py`
- `run_experiments_compare.py` → `99_archive/experiments/run_experiments_compare.py`
- `run_blind_end2end.py` → `99_archive/experiments/run_blind_end2end.py`
- `run_blind_step2_features_v2.py` → `99_archive/experiments/run_blind_step2_features_v2.py`
- `run_bnb_xrp_ada_trx_2024_2026.py` → `99_archive/experiments/run_bnb_xrp_ada_trx_2024_2026.py`
- `convert_gate_oos_v2_to_v1.py` → `99_archive/conversion/convert_gate_oos_v2_to_v1.py`
- `convert_lgbm_txt_models_to_pkl.py` → `99_archive/conversion/convert_lgbm_txt_models_to_pkl.py`
- `make_blind_gate_oos_v2_for_v1.py` → `99_archive/conversion/make_blind_gate_oos_v2_for_v1.py`
- `infer_v1_blind.log` → `99_archive/logs/infer_v1_blind.log`

## Deleted
- `dir` (empty placeholder)
- `python` (empty placeholder)
