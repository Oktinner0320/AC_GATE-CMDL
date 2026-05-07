# Informal 预处理产物与 AC-GATE 输入说明

本文档说明 `data/Informal/prepare.py` 当前已经生成的预处理产物、哪些文件可以直接作为 AC-GATE 输入、以及后续针对当前 RQ 的实验组织方式。

## 1. 当前状态

- 原始来源工作簿：`black_economy_structure - Final 20250819.xlsx`
- 预处理入口脚本：`data/Informal/prepare.py`
- 当前直接面向 AC-GATE 的输出目录：`data/Informal/processed/acgate_inputs/`
- 当前推荐的主实验输入：`informal_acgate_multiseq_overlap_ready.csv`
- 当前不进入 curated multiseq 的特征：`TD`

当前已经确认：

- `RCW` 可稳定提供 informal target 与 proxy/static 支持。
- `RPCYD` 可稳定提供长期 formal sequence。
- `RYDGDP` 可稳定提供 2019-2023 的 region-level formal sequence。
- `TD` 仍然是 audit-only sparse candidate，不应直接放入主训练 bundle。

## 2. 预处理产物分层

### 2.1 审计层

这些文件用于回答“工作簿里到底有哪些块、哪些列、哪些 codebook 项被真正观察到了”。

| 文件 | 作用 | 备注 |
| --- | --- | --- |
| `processed/informal_workbook_audit.json` | 总审计清单与导出索引 | 包含所有关键输出路径 |
| `processed/informal_codebook_registry.csv` | codebook 注册表 | 修正过 NaN 行混入问题 |
| `processed/informal_column_inventory.csv` | 扁平化后的列清单 | 用于追踪 block / 年份列来源 |
| `processed/informal_block_summary.csv` | block 级概览 | 用于快速判断哪些块值得继续解析 |
| `processed/blocks/*_block_raw.csv` | 原始 block 导出 | 保留 workbook 的局部结构 |
| `processed/blocks/*_block_year_long.csv` | 年份列展开后的长表 | 只对可识别 year map 的 block 导出 |

### 2.2 领域面板层

这些文件已经从 block 级别提升到 entity-year panel，可以作为 AC-GATE 输入构造的中间层。

| 文件 | 作用 | 当前状态 |
| --- | --- | --- |
| `processed/informal_rcw_candidate_panel.csv` | RCW 初始候选面板 | 仍保留重复与冲突计数 |
| `processed/informal_rcw_cleaned_panel.csv` | RCW 清洗后 region-year 面板 | 当前 informal target 的主来源 |
| `processed/informal_rcw_model_ready_panel.csv` | RCW 简化建模视图 | 便于快速检查 target/proxy 质量 |
| `processed/informal_rpcyd_panel.csv` | RPCYD 正式 income panel | 长时间跨度、覆盖稳定 |
| `processed/informal_rydgdp_candidate_panel.csv` | RYDGDP region-level panel | 已稳定到 21 个 region、2019-2023 |
| `processed/informal_td_candidate_panel.csv` | TD 稀疏候选 panel | 仅审计用途，不进入主 multiseq |
| `processed/informal_formal_merged_long.csv` | RCW + RPCYD + RYDGDP + TD 合并长表 | AC-GATE ready 的直接前一步 |

### 2.3 AC-GATE 输入层

这些文件是当前最接近模型契约的输入产物。

| 文件 | 用途 | 当前覆盖 | 是否推荐主用 |
| --- | --- | --- | --- |
| `processed/informal_acgate_ready_panel.csv` | 根目录保留的 multiseq ready 总表 | 378 行，21 个实体，2006-2023 | 用于总览，不是首选训练入口 |
| `processed/acgate_inputs/informal_acgate_single_feature_ready.csv` | 单 formal feature 兼容版 | 继承 legacy `x_t` 工作流 | 可作为 baseline |
| `processed/acgate_inputs/informal_acgate_multiseq_ready.csv` | 多 formal sequence 全时段版 | 2006-2023；RYDGDP 只在 2019-2023 有值 | 用于 sensitivity，不建议直接当主实验 |
| `processed/acgate_inputs/informal_acgate_multiseq_overlap_ready.csv` | 多 formal sequence dense overlap 版 | 105 行，21 个实体，2019-2023 | 当前主实验推荐 |
| `processed/acgate_inputs/informal_acgate_manifest.json` | 输入元数据与字段清单 | 记录 sequence/proxy/static 列、覆盖范围与 KNN 口径 | 必读 |

## 3. 当前 AC-GATE 输入列约定

### 3.1 目标列

- `y_t`：当前 informal target，来自 `RCW` 的 `y_rcv_total_raw`

### 3.2 formal sequence 列

- `x_t`：legacy primary sequence alias，等于 `RPCYD total income`
- `x_seq_rpcyd_total_income`：显式保留的 RPCYD 序列列
- `x_seq_rydgdp_disposable_income`
- `x_seq_rydgdp_disposable_income_per_capita`
- `x_seq_rydgdp_grdp`
- `x_seq_rydgdp_grdp_per_capita`
- `x_seq_rydgdp_ratio`

说明：

- `x_t` 的存在是为了兼容旧的单序列接口。
- 在 multiseq 设定里，真正建议显式传入的是整组 `x_seq_*` 列。
- 当前 `RYDGDP` 的稳定覆盖只在 2019-2023，因此 dense multiseq 主实验应优先用 overlap 版。

### 3.3 proxy 列

- `proxy_atm_count`
- `proxy_terminal_count`
- `proxy_credit_card_depth`

这些列当前承担 AC / adoption / transaction infrastructure 的 proxy 角色。

### 3.4 static 列

- `static_noncash_value`
- `static_withdrawal_value`

这些列更适合作为静态背景量或 normalization / diagnostics 辅助特征。

### 3.5 optional 审计列

- `optional_x_td_direct_taxes_on_labour`
- `optional_x_td_indirect_taxes_on_labour`
- `optional_x_td_taxes_on_capital`
- `optional_x_td_total_tax_revenues`

这些列只用于审计或未来补充实验，不应并入当前 curated multiseq 主 bundle。

## 4. 三套 AC-GATE 输入如何使用

### 4.1 单变量 baseline

使用：`processed/acgate_inputs/informal_acgate_single_feature_ready.csv`

适用场景：

- 复用现有单序列 CMDL / AC-GATE 训练代码
- 与后续 multiseq 版本做增量比较
- 验证“多 formal sequence 是否真的带来机制提升”

### 4.2 多变量 full-span sensitivity

使用：`processed/acgate_inputs/informal_acgate_multiseq_ready.csv`

适用场景：

- 保留 2006-2023 的长时间跨度
- 做缺失策略或 loader 设计的敏感性分析
- 不适合作为第一版主结果，因为 2019 前 `RYDGDP` 是结构性缺口，不是普通随机缺失

### 4.3 多变量 overlap 主实验

使用：`processed/acgate_inputs/informal_acgate_multiseq_overlap_ready.csv`

适用场景：

- 当前最干净的多变量 formal sequence 设定
- 2019-2023 内所有 curated `x_seq_*` 均完整
- 最适合先回答“formal-informal delay 是否存在可解释的异质滞后结构”

## 5. train-window-only KNN 方案

### 5.1 结论先行

- KNN 不应写进 `prepare.py` 的导出阶段。
- KNN 只能作为 loader 内部的可选 sensitivity 方案。
- KNN 不能用来填补当前 `2006-2018` 的 `RYDGDP` 结构性空白。
- KNN 只能用于“训练窗口内可支持年份”的零散缺失，而不是伪造不存在的历史子表。

### 5.2 为什么不能直接填 2006-2018

当前 `RYDGDP` 在 2019-2023 之前不是“观测到但有缺口”，而是“没有稳定 region-level key 可以恢复”。这意味着：

- 这些空值是 structural missing，不是 incidental missing。
- 一旦用全样本 KNN 把它们补满，会把“解析不到的数据”伪装成“真实存在但缺测的数据”。
- 如果 KNN 在全时段上拟合，还会把未来年份的信息带回过去，造成 temporal leakage。

因此，当前主分析不应对这段结构性空白做插补。

### 5.3 允许 KNN 的安全边界

KNN 只允许在以下条件同时满足时启用：

1. 数据已经先按时间切成 train / validation / test。
2. imputer 只在 train window 上 `fit`。
3. validation 和 test 只做 `transform`，不能参与 `fit`。
4. 只对 train window 内有真实支持范围的列做插补。
5. `y_t` 绝不能参与 KNN 特征构造。
6. 对于 feature-year 组合若在训练窗内没有真实支持，必须继续保持 NaN。

其中第 6 条尤其关键。一个安全实现必须先构造“可插补支持表”。例如：

- `x_seq_rydgdp_*` 当前只在 2019-2023 有真实支持。
- 因此即使以后跑 full-span sensitivity，`2006-2018` 的这些列仍应保留为 NaN。
- KNN 最多只处理 2019-2023 内部的零散缺失，不能越过支持边界补值。

### 5.4 推荐实现位置

推荐把 KNN 放到未来的 Informal loader 中，而不是放到 `prepare.py`：

- `prepare.py` 负责结构恢复、可审计导出、保留原始缺失形态。
- loader 负责依据 train split 做标准化、缺失处理、窗口裁剪。
- 这样才能保证插补严格遵守训练窗口边界。

### 5.5 推荐 workflow

### 方案 A：当前默认方案

- 主实验直接使用 `multiseq_overlap_ready.csv`
- 不启用 KNN
- 先得到一个完全无缺失、最容易解释的 multiseq 机制结果

### 方案 B：full-span sensitivity 方案

1. 使用 `multiseq_ready.csv`
2. 先按年份切分 train / validation / test
3. 构造 `eligible_impute_cols`
4. 对每个 `col` 建立 `support_years[col]`
5. 仅对 `(year in support_years[col])` 的缺失单元开放 KNN
6. `fit` 只用 train rows
7. validation / test 只做 `transform`
8. 输出额外的插补审计表：每列、每年、每个 split 被补了多少格

### 5.6 推荐特征集

KNN 的输入特征可以从以下列中选择：

- `x_t`
- `x_seq_rpcyd_total_income`
- `x_seq_rydgdp_*` 中当前有支持的列
- `proxy_atm_count`
- `proxy_terminal_count`
- `proxy_credit_card_depth`
- `static_noncash_value`
- `static_withdrawal_value`

不应纳入：

- `y_t`
- 任何来自 validation / test 的目标统计量
- 结构性无支持年份的 feature-year 组合

### 5.7 推荐报告内容

如果后续启用 KNN，结果表里应额外报告：

- `imputed_cell_count`
- `imputed_cell_share`
- 每列每年的插补比例
- KNN 相对 no-imputation 的 `delta test_r2`
- KNN 相对 no-imputation 的 `delta anchor_adjusted_rho`

如果 KNN 只能改善预测却破坏机制指标，则不应把它作为主结果。

### 5.8 最小伪代码

```python
feature_cols = [
    "x_t",
    "x_seq_rpcyd_total_income",
    "x_seq_rydgdp_disposable_income",
    "x_seq_rydgdp_disposable_income_per_capita",
    "x_seq_rydgdp_grdp",
    "x_seq_rydgdp_grdp_per_capita",
    "x_seq_rydgdp_ratio",
    "proxy_atm_count",
    "proxy_terminal_count",
    "proxy_credit_card_depth",
    "static_noncash_value",
    "static_withdrawal_value",
]

train_frame = panel.loc[train_mask].copy()
valid_frame = panel.loc[valid_mask].copy()
test_frame = panel.loc[test_mask].copy()

eligible_cols = []
for col in feature_cols:
    support_years = train_frame.loc[train_frame[col].notna(), "year"].unique()
    if len(support_years) > 0:
        eligible_cols.append(col)

scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5, weights="distance")

X_train = scaler.fit_transform(train_frame[eligible_cols])
imputer.fit(X_train)

train_frame[eligible_cols] = scaler.inverse_transform(imputer.transform(X_train))
valid_frame[eligible_cols] = scaler.inverse_transform(
    imputer.transform(scaler.transform(valid_frame[eligible_cols]))
)
test_frame[eligible_cols] = scaler.inverse_transform(
    imputer.transform(scaler.transform(test_frame[eligible_cols]))
)

# 对 support_years 外的 structural missing 单元，重新强制写回 NaN
```

## 6. 当前 RQ 的操作化定义

当前 RQ 可以具体化为：

> 在 region-level panel 上，formal economic processes 对 informal outcome 的影响是否体现为可学习、可解释、并且受 AC/proxy 条件调制的异质滞后结构？

在当前 Informal 数据里，可以按以下方式操作化：

- informal outcome：`y_t`
- formal processes：`x_t` 与整组 `x_seq_*`
- AC / adoption / infrastructure proxies：`proxy_*`
- 静态背景控制：`static_*`

## 7. RQ 实验计划

### 7.1 Phase 1：数据冻结与主数据选择

目标：先固定一个最可信的数据面板，避免训练阶段一边改 loader 一边改数据定义。

执行建议：

1. 把 `multiseq_overlap_ready.csv` 作为主实验面板。
2. 把 `single_feature_ready.csv` 作为 baseline 面板。
3. 把 `multiseq_ready.csv` 只留给 sensitivity 与 KNN 研究。
4. 暂不将 `TD` 纳入主实验。

### 7.2 Phase 2：模型接入

目标：让 Informal 数据真正进入 CMDL / AC-GATE 训练链路。

执行建议：

1. 新增 Informal loader，显式读取 `x_seq_*`、`proxy_*`、`static_*`。
2. 把 config 中 `seq_features` 从单序列改成多序列配置。
3. 保留 single-feature 模式，便于与 multiseq 做一一对应比较。

### 7.3 Phase 3：基线与主实验

建议至少跑以下组别：

1. single-feature AC-GATE
2. multiseq overlap AC-GATE
3. Plain LSTM baseline
4. Grouped ARDL baseline
5. 必要的 ablations：No AC Encoder、Uniform Lag、No Recon

### 7.4 Phase 4：KNN sensitivity

仅在主实验完成后再做：

1. no-imputation 的 `multiseq_overlap`
2. no-imputation 的 `multiseq_ready`
3. train-window-only KNN 的 `multiseq_ready`

这样才能回答：

- RQ 的机制证据是否必须依赖插补？
- KNN 是真正提高信息利用率，还是只是制造表面平滑？

### 7.5 Phase 5：多 seed 与论文化整理

建议：

1. 固定主实验后做多 seed 聚合
2. 输出 compact comparison 和 stratified k* 结果
3. 报告正向 seed 占比，而不是只报单次均值

## 8. 用哪些指标衡量这条 RQ

下面这些指标应被明确区分为“主机制指标”“辅助机制指标”“预测校准指标”和“synthetic 校准指标”。

### 8.1 主机制指标

| 指标 | 代码中的字段 | 作用 |
| --- | --- | --- |
| anchor-adjusted lag alignment | `test_effective_kstar_proxy_spearman_adjusted_rho` | 主指标。判断 learned effective lag 是否与 AC/proxy 排序方向一致 |
| 多 proxy 平均对齐 | `test_effective_kstar_proxy_mean_spearman_adjusted_rho` | 当 proxy 不止一个时，判断整体机制是否稳健 |
| lag gate 敏感度 | `test_lag_gate_sensitivity_range` | 判断 lag gate 是否真的随着 proxy 条件变化而变化 |
| 异质性强度 | `test_effective_kstar_std` | 判断模型是否学出了 entity-level lag heterogeneity，而不是退化成统一滞后 |

这些指标最直接对应当前 RQ，因为 RQ 关心的不是单纯预测，而是“formal-informal delay 是否存在、是否异质、是否受 AC 条件调制”。

### 8.2 辅助机制指标

| 指标 | 代码中的字段 | 作用 |
| --- | --- | --- |
| lag entropy | `test_effective_lag_entropy_mean` | 判断 `omega` 是否过于塌缩或过于平坦 |
| lag top-1 concentration | `test_effective_lag_top1_share` | 观察 lag profile 是否被单一滞后完全主导 |
| proxy reconstruction | `test_proxy_signal_r2` | 判断 latent proxy signal 是否保留了 proxy 信息 |
| z-anchor 方向检查 | `test_z_anchor_adjusted_rho` | 若后续显式引入 anchor，可作为辅助方向诊断 |

### 8.3 预测校准指标

| 指标 | 代码中的字段或函数 | 作用 |
| --- | --- | --- |
| `R^2` | `test_r2` / `compute_r2` | 预测校准与 sanity check，不是主 claim |
| MAE | `compute_mae` | 辅助衡量绝对误差 |
| MSE | `compute_mse` | 辅助衡量平方误差 |

这些指标只用于检查模型有没有因为追求机制解释而完全失去预测能力，不应用来替代机制指标。

### 8.4 synthetic 校准指标

| 指标 | 代码中的字段 | 作用 |
| --- | --- | --- |
| k* recovery rho | `kstar_spearman_rho` | synthetic 下的 ground-truth lag recovery |
| k* recovery MAE | `kstar_mae` | synthetic 下的 lag recovery 误差 |
| lag peak accuracy | `omega_peak_accuracy` | learned lag profile 是否抓住真实峰值 |
| proxy reconstruction | `proxy_recon_r2` | latent channel 是否保留 proxy signal |

如果 synthetic 指标不成立，真实数据上的机制主张就不应被强化。

### 8.5 多 seed 稳健性指标

多 seed 聚合时，建议额外看：

- `anchor_positive_seed_share`
- `mean_proxy_positive_seed_share`
- `kstar_positive_seed_share`

原因是这条 RQ 不能只靠单次运行的点估计回答，尤其是方向性指标更需要看 seed 稳定性。

## 9. RQ 的判定口径

当前建议的判定逻辑如下：

### 9.1 支持 RQ 的最低条件

至少同时满足：

1. `multiseq_overlap` 相比 `single_feature` 在主机制指标上更好，尤其是 `anchor_adjusted_rho` 或 `mean_proxy_adjusted_rho`。
2. `test_lag_gate_sensitivity_range > 0`，说明 lag gate 不是静止的。
3. `test_effective_kstar_std > 0`，说明存在实体间异质滞后。
4. `test_effective_lag_entropy_mean` 没有显示明显退化。

### 9.2 不足以支持 RQ 的情形

以下任一情况都说明当前只能声称“有预测结果”，不能声称“RQ 已回答”：

1. `R^2` 提升但 `anchor_adjusted_rho` 为负或接近 0。
2. `lag_gate_sensitivity_range` 接近 0。
3. `kstar_std` 接近 0，说明模型实际上学成了统一滞后。
4. 只有启用 KNN 后机制指标才成立，而 no-imputation 不成立。

## 10. 当前最推荐的执行顺序

1. 先用 `multiseq_overlap_ready.csv` 接入 Informal loader，跑 single-feature 与 multiseq 的无插补对照。
2. 先看主机制指标，再看 `R^2`、MAE、MSE 等预测校准指标。
3. 主结果稳定后，再把 `multiseq_ready.csv` 加入 train-window-only KNN sensitivity。
4. 最后才考虑是否继续扩展 `TD` 或更激进的缺失处理。