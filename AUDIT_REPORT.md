# Audit Report: v1 vs v2 Separation

## 1) v1 “黄金入口”盘点

### 期望的 v1 入口脚本（用户指定）
**已补齐并放置在 `v1/` 目录：**
- `v1/train_risk_lgbm_dd16_pipeline.py`【F:v1/train_risk_lgbm_dd16_pipeline.py†L1-L33】
- `v1/train_bar_gate_model_lgbm.py`【F:v1/train_bar_gate_model_lgbm.py†L1-L33】
- `v1/one_click_backtest_v4_15m_bar_gate_pipeline.py`【F:v1/one_click_backtest_v4_15m_bar_gate_pipeline.py†L1-L37】

### 实际存在的入口脚本（当前仓库）
**`05_runs/` 下现有入口：**
- `05_runs/run_train_risk_dev.bat`（调用 `v1/train_risk_lgbm_dd16_pipeline.py`）【F:05_runs/run_train_risk_dev.bat†L1-L9】
- `05_runs/run_train_gate_dev.bat`（调用 `v1/train_bar_gate_model_lgbm.py`）【F:05_runs/run_train_gate_dev.bat†L1-L9】
- `05_runs/run_backtest_oos.bat`（调用 `v1/one_click_backtest_v4_15m_bar_gate_pipeline.py`）【F:05_runs/run_backtest_oos.bat†L1-L17】

## 2) v1 入口脚本是否引用 v2

### 搜索命令
`rg -n "import v2|from v2|sys.path.append\\(.*v2" -g '*.py' -g '*.bat' -g '*.sh'`

### 结果
**无命中。**  
结论：目前 v1 入口脚本未发现显式引入 v2 模块/路径。

## 3) v1 输出 artifacts 自洽性检查

### Gate 训练输出 vs 回测读取
- **训练输出**：`train_gate_dev.py` 写出模型与阈值到：
  - `models/lgbm_gate_dev.txt`
  - `gate_features.txt`
  - `summary_gate_dev.json`（含 `thresholds`）【F:03_models/gate/train_gate_dev.py†L138-L173】
- **回测读取**：`backtest_oos.py` 读取：
  - `gate_model`（LGBM txt）
  - `gate_feature_list`
  - `gate_thresholds`（JSON 中 `thresholds`）【F:04_backtest/backtest_oos.py†L245-L287】

**结论**：Gate 训练输出与回测读取路径与字段 **一致**。

### Risk 输出 vs 回测读取
- **训练输出**：`train_risk_dev.py` 写出：
  - `risk_states_full.parquet`
  - `risk_states_dev.parquet`【F:03_models/risk/train_risk_dev.py†L54-L61】
- **回测读取**：`backtest_oos.py` 使用 `risk_states_full.parquet`【F:04_backtest/backtest_oos.py†L245-L298】

**结论**：Risk 输出与回测读取 **一致**。

## 4) v2 的位置与边界（当前）

### 发现的 v2 相关文件（位置+示例）
目前 v2 相关文件已集中到 `99_archive/v2_wip/`（WIP / not runnable）：
- `99_archive/v2_wip/build_gate_risk_features_v2.py`【F:99_archive/v2_wip/build_gate_risk_features_v2.py†L1-L6】
- `99_archive/v2_wip/build_position_curve_gate_ml_v2.py`【F:99_archive/v2_wip/build_position_curve_gate_ml_v2.py†L1-L9】
- `99_archive/v2_wip/train_gate_lgbm_wfv_v2.py`【F:99_archive/v2_wip/train_gate_lgbm_wfv_v2.py†L1-L14】
- `99_archive/v2_wip/run_blind_step2_features_v2.py`【F:99_archive/v2_wip/run_blind_step2_features_v2.py†L1-L44】
- `99_archive/v2_wip/convert_gate_oos_v2_to_v1.py`【F:99_archive/v2_wip/convert_gate_oos_v2_to_v1.py†L1-L8】
- `99_archive/v2_wip/make_blind_gate_oos_v2_for_v1.py`【F:99_archive/v2_wip/make_blind_gate_oos_v2_for_v1.py†L1-L8】

诊断脚本中存在 v1 vs v2 对比（用于分析，不作为入口）：  
`06_diagnose/analyze_gate_v1_vs_v2_blind.py`【F:06_diagnose/analyze_gate_v1_vs_v2_blind.py†L1-L6】

---

## 5) 运行链路静态自检（v1 入口脚本依赖）

### `train_risk_lgbm_dd16_pipeline.py`（v1 风险入口）
**输入参数：**
- `--risk_bars_features`
- `--train_end` / `--tf`  
（其余参数透传给 `train_risk_dev.py`）【F:v1/train_risk_lgbm_dd16_pipeline.py†L12-L30】

**关键列依赖（由 risk_state_machine）**：
- `symbol,event_id,ts,progress,gmma_gap,gmma_slopeS,gmma_slopeS_acc,er_16,vr_16_64,rs_vol_rel,vov_96,lr_16,wick_skew,body_eff`  
【F:03_models/risk/risk_state_machine_v1.py†L41-L73】

### `train_bar_gate_model_lgbm.py`（v1 Gate 入口）
**输入参数：**
- `--gate_samples`
- `--train_end` / `--tf`  
（其余参数透传给 `train_gate_dev.py`）【F:v1/train_bar_gate_model_lgbm.py†L12-L30】

**关键列依赖：**
- `t0_ts` / `label` / `mfe_atr`（用于目标与输出）【F:03_models/gate/train_gate_dev.py†L117-L158】

### `one_click_backtest_v4_15m_bar_gate_pipeline.py`（v1 OOS 入口）
**输入参数：**
- `--universe_long` / `--risk_states` / `--gate_samples`
- `--gate_model` / `--gate_feature_list` / `--gate_thresholds`
- `--bt_start` / `--tf`  
（其余参数透传给 `backtest_oos.py`）【F:v1/one_click_backtest_v4_15m_bar_gate_pipeline.py†L12-L36】

**关键列依赖：**
- `universe_long`: `symbol, ts, close`【F:04_backtest/backtest_oos.py†L81-L93】
- `risk_states`: `symbol, ts, event_id, state`【F:04_backtest/backtest_oos.py†L96-L109】
- `gate_samples`: `t0_ts` + 数值特征列（用于推断）【F:04_backtest/backtest_oos.py†L270-L281】

---

## 6) 结论与最小改动建议

1. **v1 黄金入口脚本已补齐**  
   `v1/` 下三个脚本已落地，并由 `05_runs` 调用。  

2. **v2 已集中隔离**  
   v2 脚本已统一移动到 `99_archive/v2_wip/`。  

3. **未改变 v1 输出格式**  
   所有调整为入口与目录整理，输出列名与路径协议保持不变。  
