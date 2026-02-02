# Project Structure (v1 Baseline)

```
.
├── AUDIT_REPORT.md                # v1/v2 separation audit
├── CHANGELOG.md                   # entrypoint/path changes
├── PROJECT_STRUCTURE.md           # this document
├── README.md
├── 00_docs/                       # policies and inventory
├── 01_data/                       # download + universe panel
├── 02_features/                   # GMMA events + features
├── 03_models/
│   ├── gate/                      # gate training
│   └── risk/                      # risk state machine + training
├── 04_backtest/                   # OOS backtest
├── 05_runs/                       # official run scripts (bat/sh)
├── 06_diagnose/                   # diagnostics (non-entry)
├── v1/                            # v1 golden entry scripts
└── 99_archive/
    ├── v2_wip/                    # v2 WIP (isolated, not runnable)
    ├── legacy/                    # legacy scripts
    ├── experiments/               # old experiments
    └── conversion/                # conversion helpers
```

## v1 entrypoints
The canonical v1 entry scripts live under `v1/`:
- `train_risk_lgbm_dd16_pipeline.py`
- `train_bar_gate_model_lgbm.py`
- `one_click_backtest_v4_15m_bar_gate_pipeline.py`

`05_runs/` wraps these with OS-friendly scripts.

## v2 boundary
All v2-related scripts are isolated under `99_archive/v2_wip/` and must not
be imported by v1.
