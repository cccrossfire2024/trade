# Crypto 15m Multi-Symbol Gate + Risk Backtest (v1 baseline)

This repository is a **15‑minute, multi‑symbol crypto trading pipeline** with:
- **Gate model** (event-level classifier) to filter low-quality entries.
- **Risk model** (rule-based risk state machine) to reduce exposure during drawdown‑risk regimes.
- **OOS backtest** that combines Gate + Risk into a position curve.

Target usage: **train on dev (t < 2025‑01‑01)** and **OOS backtest from 2025‑01‑01 onward**.

---

## Directory layout (v1)

```
00_docs/        docs, inventory, feature policy, cleanup report
01_data/        data download + universe construction
02_features/    event/feature builders
03_models/      gate/ and risk/ training outputs
04_backtest/    unified OOS backtest
05_runs/        official entry scripts (Windows)
06_diagnose/    diagnostics / sanity checks
99_archive/     legacy scripts (not baseline)
artifacts/v1/   canonical outputs (features / models / backtests)
```

---

## OOS split rule

- **train_end**: `2025-01-01` (train/dev is **strictly earlier** than this timestamp)
- **bt_start**: `2025-01-01` (OOS backtest starts here)

---

## Quickstart (Windows)

> Run these from the repository root.

1. **(Optional) Build the universe panel**
   ```bat
   python 01_data\download_ohlcv_ccxt.py
   python 01_data\build_universe_panel.py
   ```

2. **Build events + features**
   ```bat
   python 02_features\extract_gmma_events.py
   python 02_features\build_gate_risk_features_v1.py
   ```

3. **Official entry points (v1)**
   ```bat
   05_runs\run_train_risk_dev.bat
   05_runs\run_train_gate_dev.bat
   05_runs\run_backtest_oos.bat
   ```

---

## One-click pipeline (XRP/TRX/UNI 2024-2026, Linux/macOS)

This pipeline downloads 15m spot OHLCV from **data.binance.vision** (no Binance API),
builds v1 features/models, runs the OOS backtest, and writes yearly performance reports:

```bash
05_runs/run_pipeline_xrp_trx_uni_2024_2026.sh
```

---

## Official entry scripts

| Entry | What it does | Outputs |
| --- | --- | --- |
| `run_train_risk_dev.bat` | Build risk states (full + dev) | `artifacts/v1/models/risk/*` |
| `run_train_gate_dev.bat` | Train Gate model on dev | `artifacts/v1/models/gate/*` |
| `run_backtest_oos.bat` | OOS backtest from `bt_start` | `artifacts/v1/backtest/*` |

---

## Outputs to check

- **Gate**: `artifacts/v1/models/gate/summary_gate_dev.json`
- **Risk**: `artifacts/v1/models/risk/summary_risk_dev.json`
- **OOS backtest**:
  - `artifacts/v1/backtest/summary_backtest_oos.json`
  - `artifacts/v1/backtest/position_curve_oos.parquet`

---

## Feature list policy (important)

See: `00_docs/feature_policy.md`

**Summary**:
1. If `gate_features.txt` exists, use it for both training and inference.
2. Otherwise, auto‑select numeric columns and exclude ID/label fields.

---

## Common pitfalls

1. **Windows `.bat` line endings**  
   Ensure the `05_runs/*.bat` files stay in CRLF to avoid `^M` errors.

2. **Path mismatches**  
   This baseline uses **relative paths** under `artifacts/v1/`.

3. **Feature mismatch**  
   Always keep the same `gate_features.txt` between training and backtest.

4. **PSI / drift**  
   Monitor feature drift when moving from pre‑2025 to OOS. PSI spikes can
   cause sharp degradation if absolute‑price features leak in.

---

## Known issues (not fixed in this refactor)

- **Risk model is rule‑based** (state machine), not the dd16 classifier described in the strategy spec.
  The dd16 ML classifier + symbol thresholds are not present in the current codebase.
- **Gate thresholds** are computed from in‑sample dev predictions (not true OOF).
  This is acceptable for baseline reproducibility but should be upgraded later.

---

## Docs

See `00_docs/` for:
- Inventory
- Feature policy
- Cleanup report
