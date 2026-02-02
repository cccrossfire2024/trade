# Feature List Policy (v1 Baseline)

## Rule
1. **If an explicit feature list exists, always use it.**
   - Example: `artifacts/v1/models/gate/gate_features.txt`
2. **If no explicit list exists, fall back to automatic selection**:
   - Keep **numeric columns** only.
   - Exclude ID/label/leakage columns:
     - `symbol`, `event_id`, `twarn_ts`, `t0_ts`, `t1_ts`
     - `label`, `mfe_atr`, `y`

## Why this matters
- Prevents training/inference mismatches.
- Guards against accidental leakage or ID columns creeping into models.

## Where implemented
- Gate training: `03_models/gate/train_gate_dev.py`
- OOS backtest (gate inference): `04_backtest/backtest_oos.py`

## Expected layout
- `artifacts/v1/models/gate/gate_features.txt` is the **single source of truth** once created.
