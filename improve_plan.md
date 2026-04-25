# AC-GATE 后续改进计划

更新时间：2026-04-25

## 一、目标与边界

本文件用于记录 AC-GATE 改进计划、已执行结果和下一轮最小改进路径。核心判断保持不变：AC-GATE 的原创贡献不是把 CMDL 改造成通用 forecasting SOTA，而是提出一个可诊断的 **AC-conditioned heterogeneous lag discovery** 机制。后续所有改进都应服务于以下目标：

1. 让实体级 proxy 通过 `z_i -> omega -> k*` 的路径更稳定地表达滞后异质性。
2. 提升 `omega/k*` 的可解释性、非退化性和跨 seed 稳定性。
3. 保持 forecast 作为 calibration 和 sanity check，而不是唯一成功标准。
4. 保护 synthetic 上已经成立的机制恢复能力。
5. 外部方法只能作为参考和对照，不能替代 AC-GATE 的核心机制。

原创性边界：

1. 保留 AC-GATE 的主路径：entity proxy -> scalar AC latent `z_i` -> lag distribution `omega` -> effective lag `k*`。
2. 不把模型主体替换为 TFT、Transformer、Neural Granger 或 ARDL。
3. 可以吸收外部方法中的通用思想，例如门控、稀疏性、结构化滞后选择、强基线校准。
4. 新机制必须以可选参数形式引入，默认配置尽量保持当前行为，便于最小 review 和回归测试。

## 二、外部参考与可借鉴点

已查询的公开参考方向如下。它们用于启发 AC-GATE 的改进，不作为直接复制对象。

| 参考方向 | 可借鉴点 | AC-GATE 中的使用方式 | 原创性约束 |
|---|---|---|---|
| Temporal Fusion Transformer | 使用 gating 和变量选择增强可解释预测；用门控抑制不必要组件 | 参考其“门控用于选择有效信息”的思想，增强 proxy reliability 或 lag gate diagnostics | 不复制 TFT 主体，不引入完整 attention forecasting 架构 |
| Neural Granger Causality | 用结构化稀疏惩罚和自动 lag selection 识别非线性滞后关系 | 参考 lag 选择的稀疏/组稀疏思想，作为 `omega` 正则或 grouped baseline 的理论依据 | 不把 AC-GATE 改成 Granger 网络；仍以 AC 条件化滞后为核心 |
| Sparsemax / Entmax | 将 dense softmax attention 改为更稀疏、更紧凑的概率分布 | 后续可作为可选 `omega_transform`，或先用 entropy band penalty 达到类似目的 | 先做正则，不急于改概率映射，避免破坏 synthetic recovery |
| ARDL / Panel lag models | 传统模型显式指定 endogenous/exogenous lags，可做公平 lag baseline | 构建 grouped ARDL，检验人工分组异质滞后是否能解释 AC-GATE 的增量价值 | ARDL 是对照，不是主模型；不改变 AC-GATE 机制 claim |

参考网址：

1. TFT: https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/
2. Neural Granger Causality: https://arxiv.org/abs/1802.05842
3. Sparsemax: https://arxiv.org/abs/1602.02068
4. Sparse sequence-to-sequence / Entmax: https://arxiv.org/abs/1905.05702
5. Statsmodels ARDL: https://www.statsmodels.org/stable/generated/statsmodels.tsa.ardl.ARDL.html

## 三、现有问题归纳

### 问题 1：真实域机制证据不均衡

Energy 的 WGI proxy alignment 已经较强，`k*` 与治理 proxy 的 adjusted rho 为正，且 lag gate 非退化。Economics 的 effective-labor anchor 仍为负方向，说明该域不能直接作为强机制支持证据。

处理原则：

1. Energy 作为真实域机制展示主案例。
2. Economics 作为 anchor audit 与边界案例处理。
3. 不为了把 economics rho 调成正而强行加入过强监督。

### 问题 2：forecast 不能支撑全面性能优越性

CMDL 在 focused run 中能超过 matched Plain LSTM，但不稳定超过 persistence 和 Panel OLS。预测指标应继续保留，但写作和实验口径应保持 calibration，而不是 universal SOTA。

处理原则：

1. 保留 `r2_delta_vs_persistence` 和 `r2_delta_vs_panel_ols`。
2. 后续增加 grouped ARDL，作为更贴近滞后异质性的传统基线。
3. 若强基线继续领先，不调整主 claim，只报告边界。

### 问题 3：`omega` 仍主要依赖 dense softmax

当前 `ScaleInvariantLagGate` 使用 temperature softmax。它简单稳定，但容易出现两个解释性问题：

1. 分布过密时，`k*` 可解释性下降。
2. 分布过尖时，可能退化为固定 lag，削弱异质性解释。

处理原则：

1. 优先增加轻量的 entropy band diagnostics / penalty。
2. 暂不默认切换 sparsemax 或 entmax。
3. 所有稀疏化改动必须通过 synthetic recovery 回归。

### 问题 4：`z_i` 与 proxy anchor 的语义约定还不够稳定

当前 `AdaptiveACEncoder` 输出标量 `z_i`，但 `z_i` 的方向本身存在符号不识别问题。真实域中如果只看 `k*` 与 proxy 的相关性，可能混合了 proxy 语义、lag gate 方向和训练随机性的影响。

处理原则：

1. 先把 `z_i` 与 anchor proxy 的方向作为诊断项报告。
2. 再考虑可选的 `z_anchor_alignment_loss`，使较强 AC proxy 对应较高 `z_i`。
3. 最后才考虑 `k*_z` 单调约束，且只能作为 ablation，而不是默认主模型。

### 问题 5：多 seed 与论文级汇总仍需补齐

当前 formal notebook 主要是 focused single-seed evidence。机制论文最终需要多 seed mean ± std、per-proxy audit、ablation guard 和 forecast calibration 同时闭合。

处理原则：

1. 最低论文级配置为 seeds `[0, 1, 2]`。
2. 最终稳健性配置建议为 seeds `[0, 1, 2, 3, 4]`。
3. 所有表格优先从 comparison/reporting API 生成，减少 notebook 手写逻辑。

## 四、改进路线图

### Phase 1：机制 reporting 固化

目标：先保证已有 evidence 能稳定、可复现、可聚合。

涉及文件：

1. `evaluation/economics_comparison.py`
2. `evaluation/energy_comparison.py`
3. `notebooks/02_economics_results.ipynb`
4. `notebooks/03_energy_results.ipynb`

计划：

1. 用公共 helper 统一生成 per-proxy alignment table。
2. 用公共 helper 统一生成 mechanism result log。
3. notebook 不再手写列名解析，避免 reporting 口径分叉。
4. 在输出目录固定保存：`mechanism_summary.csv`、`per_proxy_alignment.csv`、`mechanism_result_log.csv`。

验收标准：

1. comparison 相关测试全部通过。
2. Economics / energy notebook 读取同一批 summary 时，输出与 helper 一致。
3. 真实域 result log 能直接回答 forecast、simple baseline、mechanism、per-proxy、heterogeneity、ablation guard 六类问题。

### Phase 2：`omega` 可解释性正则

目标：增强 lag distribution 的可解释性，同时避免 dense 和 one-hot 两类退化。

涉及文件：

1. `config/cmdl_config.py`
2. `model/loss.py`
3. `model/cmdl_model.py`
4. `experiments/run_synthetic.py`
5. `experiments/run_economics.py`
6. `experiments/run_energy.py`
7. `tests/test_step3_model.py`
8. `tests/test_step2_modules.py`

最小实现方案：

1. 增加可选配置，默认全关闭：
   - `lambda_omega_entropy = 0.0`
   - `omega_entropy_min = None`
   - `omega_entropy_max = None`
2. 在 loss 中增加 entropy band penalty：
   - 若 entropy 低于下界，惩罚过尖分布；
   - 若 entropy 高于上界，惩罚过密分布；
   - 默认不开启，不影响现有结果。
3. 在 summary 中记录：
   - `omega_entropy_penalty`
   - `omega_entropy_mean`
   - `omega_entropy_band_violation_share`
4. 先不改 softmax 为 sparsemax/entmax，除非 Phase 2 仍不能改善解释性。

验收标准：

1. Synthetic linear / nonlinear 的 `kstar_spearman_rho` 不明显下降。
2. `no_ac_encoder` 和 `uniform_lag` 仍能暴露退化边界。
3. Energy 的 per-proxy adjusted rho 不下降到 0 以下。
4. Economics 不以强制变正为目标，只观察稳定性变化。

### Phase 3：`z_i` anchor 语义稳定化

目标：让 `z_i` 的方向更符合“较强吸收能力或治理质量对应较高 latent AC”的语义，减少跨 seed 符号不稳定。

涉及文件：

1. `model/loss.py`
2. `experiments/run_economics.py`
3. `experiments/run_energy.py`
4. `evaluation/realdata_diagnostics.py`
5. `tests/test_step3_model.py`

最小实现方案：

1. 先增加诊断，不先加约束：
   - `z_anchor_spearman_rho`
   - `z_anchor_expected_sign`
   - `z_anchor_adjusted_rho`
2. 若诊断表明 `z_i` 方向跨 seed 不稳定，再增加可选损失：
   - `lambda_z_anchor = 0.0`
   - `z_anchor_target_sign = +1`
   - batch Pearson 近似相关损失或 rank-free covariance loss。
3. 对 `k*` 与 anchor 的方向仍只作为结果诊断，不直接硬约束。

验收标准：

1. 开启 `lambda_z_anchor` 后，`z_anchor_adjusted_rho` 更稳定。
2. Energy 的 `kstar_proxy_adjusted_rho` 保持为正。
3. Economics 若仍为负，记录为 anchor mismatch，而不是继续加大约束。
4. Synthetic recovery 不被破坏。

### Phase 4：Grouped ARDL 机制基线

目标：增加一个“人工分组版异质滞后”传统基线，检验 AC-GATE 的连续 AC-conditioned lag gate 是否有额外价值。

涉及文件：

1. `baselines/panel_ols.py` 或新增 `baselines/grouped_ardl.py`
2. `experiments/run_economics.py` 或新增 `experiments/run_economics_grouped_ardl.py`
3. `experiments/run_energy.py` 或新增 `experiments/run_energy_grouped_ardl.py`
4. `evaluation/economics_comparison.py`
5. `evaluation/energy_comparison.py`

最小实现方案：

1. 按 anchor proxy 的 train-window 分位数把实体分为 low / mid / high 三组。
2. 每组拟合相同 lag window 的 ARDL 或 distributed-lag OLS。
3. 统一使用相同 train / validation / test split。
4. 输出每组最优 lag、R2、MSE 和 lag summary。
5. comparison 表中把它标为 `Grouped ARDL`，用于 calibration 和机制对照。

验收标准：

1. grouped baseline 能在 economics / energy 上稳定运行。
2. 如果 Grouped ARDL 预测更强，AC-GATE 仍可保留机制优势 claim，但必须说明预测边界。
3. 如果 AC-GATE 的 `k*` 分组趋势与 Grouped ARDL 的 lag order 趋势一致，则是强机制支持。

### Phase 5：Sparse lag transform 备选实验

目标：在 Phase 2 正则不足时，再评估 sparsemax / entmax 是否能提升 `omega` 可解释性。

涉及文件：

1. `model/lag_gate.py`
2. `config/cmdl_config.py`
3. `tests/test_step2_modules.py`
4. `evaluation/kstar_eval.py`

最小实现方案：

1. 增加 `omega_transform`，默认 `softmax`。
2. 可选实现 `sparsemax`，不新增外部依赖。
3. `entmax` 暂作为研究备选，除非有成熟轻量实现或明确收益，否则不引入依赖。
4. 对每种 transform 固定报告：
   - `kstar_spearman_rho`
   - `omega_entropy_mean`
   - `omega_top1_share`
   - `lag_gate_sensitivity_range`
   - `forecast_delta_vs_plain_lstm`

验收标准：

1. sparse transform 只有在 synthetic recovery 不下降、real-data mechanism 更清晰时才保留。
2. 如果 sparse transform 只是让 `omega` 变尖但不改善 proxy alignment，则不作为主模型。
3. 最终论文可把 sparse transform 写作 robustness，而不是替换 AC-GATE 主设定。

### Phase 6：多 seed formal validation

目标：把 focused single-seed evidence 升级为可写入论文表格的多 seed 证据。

涉及文件：

1. `experiments/run_economics.py`
2. `experiments/run_energy.py`
3. `experiments/run_economics_ablation.py`
4. `experiments/run_energy_ablation.py`
5. `notebooks/02_economics_results.ipynb`
6. `notebooks/03_energy_results.ipynb`
7. `evaluation/economics_comparison.py`
8. `evaluation/energy_comparison.py`

计划：

1. 第一轮 seeds `[0, 1, 2]`。
2. 第二轮如资源允许扩展到 `[0, 1, 2, 3, 4]`。
3. 每个域固定跑：
   - Full CMDL
   - Plain LSTM
   - No AC Encoder
   - Uniform Lag
   - No Recon Regularization
   - 若 Phase 2/3 引入新机制，再加对应 ablation
4. 输出统一表：
   - task calibration table
   - mechanism summary table
   - per-proxy alignment table
   - ablation guard table
   - result log table

验收标准：

1. Synthetic：full CMDL 继续显著优于 no_ac_encoder 和 uniform_lag 的 k* recovery。
2. Energy：多 seed 平均 adjusted rho 为正，且 per-proxy adjusted rho 大部分为正。
3. Economics：若多 seed 仍为负，则将其记录为 boundary case。
4. Forecast：至少保持不显著弱于 matched Plain LSTM；若弱于 simple baseline，保留 calibration 边界说明。

## 五、推荐执行顺序

### Sprint 1：先做 reporting 清理

优先级最高。它不改变模型行为，却能立刻减少 notebook 手写逻辑，并让后续实验的判断口径稳定。

执行项：

1. notebook 使用公共 comparison helper。
2. 确认 economics / energy 的 result log 由同一套代码生成。
3. 补充或更新 comparison tests。

### Sprint 2：实现 `omega` entropy band penalty

这是对 AC-GATE 核心机制最直接、风险较低的改进。

执行项：

1. 在配置中加入默认关闭的 entropy penalty 参数。
2. 在 loss 或训练循环中计算 `omega` entropy penalty。
3. 在 summary 中记录 penalty 与 violation share。
4. 先跑 synthetic，再跑 energy，最后跑 economics。

### Sprint 3：补 `z_i` anchor 诊断，再决定是否加 loss

不要一开始就强制 `k*` 与 proxy 对齐。先判断问题是 `z_i` 方向不稳，还是 lag gate 学到的机制确实与理论 anchor 不一致。

执行项：

1. 增加 `z_anchor_adjusted_rho` diagnostics。
2. 多 seed 检查 `z_i` 方向。
3. 只有当 `z_i` 方向明显随机时，才引入 `lambda_z_anchor`。

### Sprint 4：实现 Grouped ARDL baseline

这是最适合回应“性能和传统滞后模型比较”的新增基线。它比 TFT 更贴近本文问题，也不会抢走 AC-GATE 的机制主线。

执行项：

1. 先实现 distributed-lag OLS 版本，保证无额外复杂依赖。
2. 再评估是否调用 statsmodels ARDL。
3. 把 grouped lag trend 与 AC-GATE `k*` trend 放在同一张机制表中。

### Sprint 5：多 seed formal validation

最后统一跑多 seed，而不是在每个小改动后都做昂贵实验。

执行项：

1. synthetic quick regression。
2. energy three-seed formal validation。
3. economics three-seed anchor audit。
4. 若结果稳定，再扩展到 five-seed。

## 六、代码修改原则

1. 默认参数必须保持当前行为，避免无意改变已有实验结果。
2. 每次只打开一个新机制，避免 entropy、z-anchor、sparse transform 同时混入导致归因困难。
3. 所有新增机制都必须有 ablation 或关闭选项。
4. 先改 shared module，再改 runner，再改 notebook。
5. 先写 focused unit tests，再跑相关 regression tests。
6. 不因 economics 的负 rho 结果而定向调参；它可能是理论 anchor 不成立的证据。

## 七、验收矩阵

| 层级 | 必须观察的指标 | 通过条件 | 失败后的处理 |
|---|---|---|---|
| Synthetic recovery | `kstar_mae`, `kstar_spearman_rho`, `z_spearman_rho` | 不低于当前主结果的可接受范围 | 回退新机制或默认关闭 |
| Ablation necessity | no_ac_encoder, uniform_lag | 退化边界仍清晰 | 检查新正则是否绕过 AC-GATE |
| Lag interpretability | `omega_entropy_mean`, `omega_top1_share`, `lag_gate_sensitivity_range` | 非全密、非全尖、非零敏感性 | 调整 entropy band 或关闭 sparse transform |
| Proxy alignment | anchor-adjusted rho, per-proxy adjusted rho | Energy 稳定为正；economics 如不成立则明确标注 | 不强行优化 economics 结论 |
| Forecast calibration | R2 / MSE vs Plain LSTM, persistence, Panel OLS, Grouped ARDL | 至少不明显劣于 matched LSTM；强基线作为边界 | 降低 forecasting claim |
| Reporting stability | mechanism result log | 多 seed 表能自动生成 | 修 reporting，不手写 notebook logic |

## 八、当前建议的下一步代码任务

下一轮代码应按如下顺序推进：

1. 把 notebooks 中的 Phase4/Phase6 手写 reporting 替换为 `build_per_proxy_alignment_table`、`build_mechanism_summary_table` 和 `build_mechanism_result_log`。
2. 为 `omega` entropy band 增加默认关闭的配置和测试。
3. 在真实域 diagnostics 中加入 `z_anchor_adjusted_rho`。
4. 跑 synthetic quick regression，确认 AC-GATE 机制恢复未受影响。
5. 跑 energy focused validation，检查机制证据是否更清晰。
6. 再跑 economics anchor audit，判断是否仍应作为 boundary case。

## 九、预期论文口径

若以上计划完成，论文主线建议保持为：

1. AC-GATE 提出一种实体级 AC 条件化的滞后分布学习机制。
2. Synthetic 证明该机制能恢复 ground-truth heterogeneous lags。
3. Ablation 证明 AC encoder 和 adaptive lag gate 是机制必要组件。
4. Real data 展示机制在真实面板上非退化，并在 energy 中与治理 proxy 稳定对齐。
5. Forecast calibration 说明模型具备合理预测能力，但不承诺全面超过 persistence、Panel OLS 或 grouped ARDL。
6. Economics 若仍不对齐，应诚实作为 anchor mismatch 或 boundary case，而不是改写为成功案例。

最终判断标准不是“AC-GATE 是否在所有数据集上预测第一”，而是：

> AC-GATE 是否提供了传统 forecasting baseline 难以直接给出的、可检验的实体级异质滞后结构。

## 十、2026-04-25 执行后状态更新

本轮已经完成原计划中的 6 个 Phase 的代码任务，并完成 focused validation、notebook 汇总和回归测试。当前状态不再是“待实现计划”，而是“已实现机制后的证据评估与下一轮改进路线”。

### 已完成实现

1. `omega_transform` 已支持 `softmax` 与可选 `sparsemax`，默认仍为 `softmax`。
2. `DomainAgnosticLoss` 已加入默认关闭的 `omega` entropy band penalty。
3. `DomainAgnosticLoss` 已加入默认关闭的 `z_anchor` alignment loss。
4. real-data diagnostics 已加入 `z_anchor_*` 显式字段。
5. economics / energy 已加入 Grouped ARDL-style distributed-lag baseline。
6. comparison API 已统一支持 task table、interpretability table、per-proxy alignment、mechanism summary 和 mechanism result log。
7. economics / energy notebook 的 Phase4/Phase6 汇总已改为公共 helper，并纳入 Grouped ARDL。
8. 全量测试通过：`59 passed, 0 failed`。

### 当前核心结果

| 域 | 当前结论 | 关键证据 | 论文用途 |
|---|---|---|---|
| Synthetic | 机制成立 | softmax run: `kstar_spearman_rho=0.9808`, `proxy_recon_r2=0.9508`; sparse/entropy/anchor run: `kstar_spearman_rho=0.9588`, `proxy_recon_r2=0.9534` | 作为 AC-GATE 可恢复 ground-truth heterogeneous lag 的主证明 |
| Energy | 真实域机制支持 | anchor-adjusted rho `0.6442`；WGI per-proxy adjusted rho 全为正；lag gate 非退化 | 作为真实域机制展示主案例 |
| Economics | 混合证据 / boundary case | CMDL 高于 Plain LSTM 和 Grouped ARDL，但 anchor-adjusted rho `-0.1590` | 作为 anchor mismatch、边界条件和后续改进对象 |

Forecast 结论必须克制：CMDL 在 focused validation 中能超过 matched Plain LSTM，但 energy 明显低于 Grouped ARDL，economics 低于 persistence。因此，forecast 应写作 calibration evidence，不应写作 universal forecasting superiority。

### outputs 清理建议

已删除所有 `__pycache__` 目录内的缓存文件。当前 outputs 中建议分三类处理：

1. 建议保留：
   - `outputs/notebook_economics/phase46_formal_validation/`
   - `outputs/notebook_energy/phase46_formal_validation/`
   - `outputs/notebook_step4/formal_target/`
   - `outputs/notebook_step45/formal_target/`
   - `outputs/notebook_economics/formal_mechanism_*`
2. 可以删除或归档：
   - `outputs/improve_plan_validation/`：本轮代码 smoke / quick validation 产物，关键结论已经写入本文档和 comparison CSV。
   - `outputs/phase_execution_smoke/`：阶段执行 smoke 产物，非论文证据。
   - `outputs/step5/economics_cleaned_smoke/`：早期 smoke run，若不再用于 debug 可删除。
3. 暂不建议删除：
   - `outputs/notebook_economics/phase46_formal_validation/grouped_ardl/`
   - `outputs/notebook_energy/phase46_formal_validation/grouped_ardl/`
   这些目录虽小，但已被 notebook comparison 读取，用于 Grouped ARDL 对照。

### 机制是否支持论文撰写

当前结果已经支持撰写一篇机制导向论文的初稿，但论文主张必须限定为：

1. AC-GATE 能在 synthetic 中恢复已知异质滞后结构。
2. AC encoder 与 adaptive lag gate 是必要组件；退化 ablation 会暴露边界。
3. 真实域中，energy 支持 proxy-aligned heterogeneous lag discovery。
4. economics 暂不能作为强机制正例，应作为 boundary case 或 anchor-audit case。
5. 预测指标用于校准与 sanity check，不作为全面 SOTA 主张。

### 下一轮最小改进路径

下一轮不应继续盲目加复杂模型，而应按以下顺序推进：

1. **多 seed 正式复核**：把 economics / energy 的 Phase46 从当前 focused seed=0 扩展到 seeds `[0, 1, 2]` 的完整 CMDL、Plain LSTM、Ablation、Grouped ARDL 汇总。
2. **Grouped ARDL 机制趋势对照**：不仅比较 R2，还比较 low / mid / high proxy 组的 best lag / effective lag 与 AC-GATE `k*` 分位趋势是否一致。
3. **Energy 机制图表**：固定输出 WGI proxy quartile 下的 `k*` summary、omega heatmap 和 per-proxy adjusted rho 表，作为论文真实域主图。
4. **Economics anchor audit**：分开报告 effective labor anchor、employment、human capital level、human capital trend 的 adjusted rho，判断负方向是理论错位还是训练不稳。
5. **可选机制实验**：只在多 seed 后再开启 `lambda_z_anchor` 或 entropy band，不要同时开启多个新机制。
6. **写作策略**：先写 Method + Synthetic + Energy mechanism，最后再写 Economics limitation。
