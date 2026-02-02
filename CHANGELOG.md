# Changelog

## 2025-02-02
### Changed
- `05_runs/*` entry scripts now call the required v1 golden entrypoints under `v1/`
  to avoid v1/v2 confusion and preserve the mandated script names.

### Why
- Users require `train_risk_lgbm_dd16_pipeline.py`, `train_bar_gate_model_lgbm.py`,
  and `one_click_backtest_v4_15m_bar_gate_pipeline.py` as the v1 “golden” entrypoints.

### How to verify
- Run `05_runs/run_train_risk_dev.(bat|sh)` to verify `--train_end` and `--tf` are passed through.
- Run `05_runs/run_train_gate_dev.(bat|sh)` to verify `--train_end` and `--tf` are passed through.
- Run `05_runs/run_backtest_oos.(bat|sh)` to verify `--bt_start` and `--tf` are passed through.
