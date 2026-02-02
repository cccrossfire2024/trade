#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT}/01_data/download_ohlcv_binance_vision.py" \
  --symbols XRPUSDT TRXUSDT UNIUSDT \
  --timeframe 15m \
  --start_date "2024-01-01" \
  --end_date "2026-12-31" \
  --data_dir "${ROOT}/01_data/raw" \
  --audit_dir "${ROOT}/01_data/audits"

python "${ROOT}/01_data/build_universe_panel.py" \
  --raw_dir "${ROOT}/01_data/raw" \
  --out_dir "${ROOT}/01_data" \
  --symbols XRPUSDT TRXUSDT UNIUSDT \
  --dev_start "2024-01-01 00:00:00+00:00" \
  --dev_end "2025-01-01 00:00:00+00:00" \
  --blind_start "2025-01-02 00:00:00+00:00" \
  --blind_end "2026-12-31 23:59:59+00:00"

python "${ROOT}/02_features/extract_gmma_events.py" \
  --input_long_dev "${ROOT}/01_data/universe_long_all.parquet" \
  --out_dir "${ROOT}/artifacts/v1/events"

python "${ROOT}/02_features/build_gate_risk_features_v1.py" \
  --universe_long_dev "${ROOT}/01_data/universe_long_all.parquet" \
  --events_dev "${ROOT}/artifacts/v1/events/gmma_events_dev.parquet" \
  --risk_bars_dev "${ROOT}/artifacts/v1/events/gmma_risk_bars_dev.parquet" \
  --out_dir "${ROOT}/artifacts/v1/features"

python "${ROOT}/v1/train_risk_lgbm_dd16_pipeline.py" \
  --risk_bars_features "${ROOT}/artifacts/v1/features/risk_bars_features.parquet" \
  --out_dir "${ROOT}/artifacts/v1/models/risk" \
  --train_end "2025-01-01" \
  --tf 15

python "${ROOT}/v1/train_bar_gate_model_lgbm.py" \
  --gate_samples "${ROOT}/artifacts/v1/features/gate_samples.parquet" \
  --out_dir "${ROOT}/artifacts/v1/models/gate" \
  --train_end "2025-01-01" \
  --tf 15

python "${ROOT}/v1/one_click_backtest_v4_15m_bar_gate_pipeline.py" \
  --universe_long "${ROOT}/01_data/universe_long_all.parquet" \
  --risk_states "${ROOT}/artifacts/v1/models/risk/risk_states_full.parquet" \
  --gate_samples "${ROOT}/artifacts/v1/features/gate_samples.parquet" \
  --gate_model "${ROOT}/artifacts/v1/models/gate/models/lgbm_gate_dev.txt" \
  --gate_feature_list "${ROOT}/artifacts/v1/models/gate/gate_features.txt" \
  --gate_thresholds "${ROOT}/artifacts/v1/models/gate/summary_gate_dev.json" \
  --bt_start 2025-01-01 \
  --tf 15 \
  --pass_tag pass20

python "${ROOT}/04_backtest/summary_yearly_performance.py" \
  --position_curve "${ROOT}/artifacts/v1/backtest/position_curve_oos.parquet" \
  --out_csv "${ROOT}/artifacts/v1/backtest/yearly_performance.csv" \
  --out_md "${ROOT}/artifacts/v1/backtest/yearly_performance.md"
