# Repository Inventory (v1 baseline)

This inventory covers all `*.py`, `*.bat`, `*.sh`, and `*.md` files.

## KEEP (core + docs)

| Path | Purpose | Inputs | Outputs | Status | Reason |
| --- | --- | --- | --- | --- | --- |
| README.md | Project overview + quickstart | — | — | KEEP | Primary onboarding doc |
| AUDIT_REPORT.md | v1/v2 audit report | — | — | KEEP | Required audit output |
| PROJECT_STRUCTURE.md | Directory structure guide | — | — | KEEP | Required deliverable |
| CHANGELOG.md | v1 entrypoint change log | — | — | KEEP | Required deliverable |
| 00_docs/feature_policy.md | Feature list rules | — | — | KEEP | Consistency policy |
| 00_docs/inventory.md | Script inventory | — | — | KEEP | Required audit |
| 00_docs/cleanup_report.md | Move/delete report | — | — | KEEP | Required audit |
| 01_data/download_ohlcv_ccxt.py | Download 15m OHLCV via CCXT | Exchange API | `01_data/raw/*.parquet`, audits | KEEP | Data ingestion |
| 01_data/build_universe_panel.py | Normalize/align universe panel | `01_data/raw/*.parquet` | `01_data/universe_*` | KEEP | Data preparation |
| 02_features/extract_gmma_events.py | Build GMMA events + risk bars | `01_data/universe_long_all.parquet` | `artifacts/v1/events/*` | KEEP | Feature pipeline step |
| 02_features/build_gate_risk_features_v1.py | Build gate/risk features | Universe + events + risk bars | `artifacts/v1/features/*` | KEEP | Feature pipeline step |
| 03_models/gate/train_gate_dev.py | Train Gate model (dev) | `artifacts/v1/features/gate_samples.parquet` | Gate model + thresholds + OOF | KEEP | Official entry |
| 03_models/risk/risk_state_machine_v1.py | Risk state machine rules | `artifacts/v1/features/risk_bars_features.parquet` | `artifacts/v1/models/risk/risk_states_full.parquet` | KEEP | Rule-based risk core |
| 03_models/risk/train_risk_dev.py | Build risk states (full + dev) | `artifacts/v1/features/risk_bars_features.parquet` | `artifacts/v1/models/risk/*` | KEEP | Official entry |
| 04_backtest/backtest_oos.py | OOS backtest | Universe + risk_states + gate model | `artifacts/v1/backtest/*` | KEEP | Official entry |
| v1/train_risk_lgbm_dd16_pipeline.py | v1 golden entry: risk training | See script | See script | KEEP | Required v1 entry |
| v1/train_bar_gate_model_lgbm.py | v1 golden entry: gate training | See script | See script | KEEP | Required v1 entry |
| v1/one_click_backtest_v4_15m_bar_gate_pipeline.py | v1 golden entry: OOS backtest | See script | See script | KEEP | Required v1 entry |
| 05_runs/run_train_gate_dev.bat | Windows entry: train Gate | See script | See script | KEEP | Official entry |
| 05_runs/run_train_risk_dev.bat | Windows entry: train Risk | See script | See script | KEEP | Official entry |
| 05_runs/run_backtest_oos.bat | Windows entry: OOS backtest | See script | See script | KEEP | Official entry |
| 05_runs/run_train_gate_dev.sh | Linux entry: train Gate | See script | See script | KEEP | Optional |
| 05_runs/run_train_risk_dev.sh | Linux entry: train Risk | See script | See script | KEEP | Optional |
| 05_runs/run_backtest_oos.sh | Linux entry: OOS backtest | See script | See script | KEEP | Optional |
| 05_runs/run_train_gate_dev.bat | Windows entry: train Gate | See script | See script | KEEP | Official entry |
| 05_runs/run_train_risk_dev.bat | Windows entry: train Risk | See script | See script | KEEP | Official entry |
| 05_runs/run_backtest_oos.bat | Windows entry: OOS backtest | See script | See script | KEEP | Official entry |
| 06_diagnose/analyze_gate_v1_vs_v2_blind.py | Gate eval comparison (blind) | Gate samples + gate oos | JSON report | KEEP | Diagnostic tool |
| 06_diagnose/debug_gate_v1_v2_blind_collapse.py | Gate debug utilities | Gate samples + oos | JSON/prints | KEEP | Diagnostic tool |
| 06_diagnose/sanity_suite_gate_pipeline.py | Pipeline sanity runner | Multiple datasets/models | Reports | KEEP | Diagnostic tool |

## ARCHIVE (legacy/experiments/conversions)

| Path | Purpose | Inputs | Outputs | Status | Reason |
| --- | --- | --- | --- | --- | --- |
| 99_archive/README_DEPRECATED.md | Archive policy | — | — | ARCHIVE | Not an entry point |
| 99_archive/legacy/build_gate_risk_features_v1plus4.py | Legacy features | Various | Various | ARCHIVE | Superseded by v1 features |
| 99_archive/legacy/build_position_curve_gate_ml_v1.py | Legacy/duplicate | Various | Various | ARCHIVE | Duplicate of gate training |
| 99_archive/legacy/build_gate_risk_features_v2.py | Legacy features v2 | Various | Various | ARCHIVE | Superseded by v1 features |
| 99_archive/legacy/build_position_curve_gate_ml_v1.py | Legacy/duplicate | Various | Various | ARCHIVE | Duplicate of gate training |
| 99_archive/legacy/build_position_curve_gate_ml_v2.py | Legacy backtest | Various | Various | ARCHIVE | Replaced by `backtest_oos.py` |
| 99_archive/legacy/build_position_curve_v0.py | Legacy backtest | Gate labels + risk | Curve outputs | ARCHIVE | Replaced by `backtest_oos.py` |
| 99_archive/legacy/build_position_curve_v1_executor_from_gate_oos.py | Legacy executor | Gate OOS + risk | Curve outputs | ARCHIVE | Replaced by `backtest_oos.py` |
| 99_archive/legacy/risk_state_machine_v0.py | Legacy risk state machine | Risk bars | Risk states | ARCHIVE | Superseded by v1 |
| 99_archive/legacy/train_gate_lgbm_wfv.py | Legacy gate WFV | Gate samples | Models + OOS | ARCHIVE | Replaced by `train_gate_dev.py` |
| 99_archive/legacy/train_gate_lgbm_wfv_v2.py | Legacy gate WFV v2 | Gate samples | Models + OOS | ARCHIVE | Replaced by `train_gate_dev.py` |
| 99_archive/legacy/infer_gate_blind_from_txt_models.py | Legacy analysis | Gate samples + models | Reports | ARCHIVE | Not baseline |
| 99_archive/legacy/check.py | Scratch checks | Various | Various | ARCHIVE | Not baseline |
| 99_archive/experiments/run_all_experiments_gate_risk_v3.py | Full experiment runner | Many datasets | Compare tables | ARCHIVE | Too broad for v1 |
| 99_archive/experiments/run_experiments_compare.py | Experiment compare | Many datasets | Compare tables | ARCHIVE | Experimental |
| 99_archive/experiments/run_blind_end2end.py | Legacy end-to-end | Various | Various | ARCHIVE | Superseded by v1 entrypoints |
| 99_archive/experiments/run_bnb_xrp_ada_trx_2024_2026.py | One-off run | Various | Various | ARCHIVE | One-off |
| 99_archive/conversion/convert_lgbm_txt_models_to_pkl.py | Model conversion | LGBM txt | pkl | ARCHIVE | Not baseline |
| 99_archive/v2_wip/build_gate_risk_features_v2.py | v2 WIP features | Various | Various | ARCHIVE | v2 WIP isolated |
| 99_archive/v2_wip/build_position_curve_gate_ml_v2.py | v2 WIP backtest | Various | Various | ARCHIVE | v2 WIP isolated |
| 99_archive/v2_wip/train_gate_lgbm_wfv_v2.py | v2 WIP gate training | Gate samples | Models + OOS | ARCHIVE | v2 WIP isolated |
| 99_archive/v2_wip/run_blind_step2_features_v2.py | v2 WIP blind features | Various | Various | ARCHIVE | v2 WIP isolated |
| 99_archive/v2_wip/convert_gate_oos_v2_to_v1.py | v2 conversion | v2 outputs | v1 outputs | ARCHIVE | v2 WIP isolated |
| 99_archive/v2_wip/make_blind_gate_oos_v2_for_v1.py | v2 conversion | v2 outputs | v1 outputs | ARCHIVE | v2 WIP isolated |
| 99_archive/experiments/run_blind_step2_features_v2.py | Legacy blind feature build | Various | Various | ARCHIVE | Superseded by v1 features |
| 99_archive/experiments/run_bnb_xrp_ada_trx_2024_2026.py | One-off run | Various | Various | ARCHIVE | One-off |
| 99_archive/conversion/convert_gate_oos_v2_to_v1.py | Legacy conversion | v2 outputs | v1 outputs | ARCHIVE | Not baseline |
| 99_archive/conversion/convert_lgbm_txt_models_to_pkl.py | Model conversion | LGBM txt | pkl | ARCHIVE | Not baseline |
| 99_archive/conversion/make_blind_gate_oos_v2_for_v1.py | Legacy conversion | v2 outputs | v1 outputs | ARCHIVE | Not baseline |
| 99_archive/logs/infer_v1_blind.log | Old log | — | — | ARCHIVE | Historical log |

## DELETE (removed)

| Path | Purpose | Status | Reason |
| --- | --- | --- | --- |
| dir | Empty placeholder | DELETE | Unused |
| python | Empty placeholder | DELETE | Unused |
