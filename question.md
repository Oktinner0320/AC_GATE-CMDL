# AC-GATE 当前研究问题与回答

更新时间：2026-04-25

## 一、核心研究问题

### Q1：本研究是否必须证明 AC-GATE 在预测性能上全面优于其他方法？

不必须。当前最稳妥的定位是：AC-GATE 是一个机制导向的 neural panel model，用于发现 **AC-conditioned heterogeneous lag structures**。预测性能需要作为 calibration 和 sanity check 报告，但不是唯一成败标准。

论文不能写成“AC-GATE universally outperforms all forecasting baselines”。更合理的主张是：AC-GATE 提供传统 forecasting baseline 难以直接给出的实体级 `omega/k*` 滞后结构，并能用 proxy alignment、ablation guard 和 forecast calibration 来检验该结构是否有意义。

### Q2：当前结果是否支持 AC-GATE 机制成立？

支持，但需要限定证据范围。

1. **Synthetic 支持机制成立**：默认 softmax run 的 `kstar_spearman_rho=0.9808`、`proxy_recon_r2=0.9508`；sparse/entropy/anchor 控制 run 的 `kstar_spearman_rho=0.9588`、`proxy_recon_r2=0.9534`。这说明新增机制没有破坏 ground-truth heterogeneous lag recovery。
2. **Energy 支持真实域机制方向**：CMDL 的 anchor-adjusted rho 为 `0.6442`，WGI per-proxy adjusted rho 全为正，lag gate sensitivity 非零，说明 learned heterogeneity 与治理 proxy 对齐。
3. **Economics 不能作为强机制正例**：CMDL 高于 Plain LSTM 与 Grouped ARDL，但 anchor-adjusted rho 为 `-0.1590`，说明当前 effective-labor anchor 下的机制方向没有被支持。

因此，当前结论应写作：AC-GATE 机制在 synthetic 中成立，在 energy 中获得真实域支持，在 economics 中暴露 anchor mismatch / boundary case。

### Q3：当前结果是否足以开始写论文？

可以开始写机制导向论文初稿，但需要克制 claim。

可以现在写的部分：

1. Method：AC-GATE 的 `p_i -> z_i -> omega_i -> k_i^*` 方程、entropy / z-anchor 可选正则、Grouped ARDL 对照。
2. Synthetic：ground-truth recovery 与 ablation necessity。
3. Energy：真实域 proxy-aligned lag heterogeneity。
4. Discussion：forecast calibration 边界与 economics anchor mismatch。

暂时不应写死的部分：

1. 全面预测优越性。
2. economics 已经验证预期机制。
3. 所有真实域多 seed 结论已经稳定。

### Q4：Grouped ARDL 对当前结论意味着什么？

Grouped ARDL 是强 baseline，不是主模型替代品。它的作用是检验“人工分组异质滞后”能否解释掉 AC-GATE 的增量价值。

当前 focused validation 显示：

1. Economics：CMDL `test_r2=0.0445`，Grouped ARDL `test_r2=-0.0891`，CMDL delta 为 `+0.1336`。
2. Energy：CMDL `test_r2=-0.0265`，Grouped ARDL `test_r2=0.6071`，CMDL delta 为 `-0.6336`。

解释：

1. Economics 中，AC-GATE 在预测上优于人工分组滞后基线，但机制方向不支持。
2. Energy 中，AC-GATE 机制方向支持，但预测明显弱于 Grouped ARDL。
3. 因此论文必须区分 “mechanism alignment” 与 “forecast superiority”。两者不能混为一个结论。

### Q5：哪些 outputs 可以删除？

已清理：所有 `__pycache__` 目录内的 `.pyc` 文件已删除。

建议可以删除或归档的输出：

1. `outputs/improve_plan_validation/`：本轮 quick validation / smoke 产物，关键结论已经转写到文档和 notebook comparison。
2. `outputs/phase_execution_smoke/`：阶段 smoke 产物，非论文证据。
3. `outputs/step5/economics_cleaned_smoke/`：早期真实数据 smoke 产物，如不再 debug 可删除。

暂时建议保留的输出：

1. `outputs/notebook_economics/phase46_formal_validation/`
2. `outputs/notebook_energy/phase46_formal_validation/`
3. `outputs/notebook_step4/formal_target/`
4. `outputs/notebook_step45/formal_target/`
5. `outputs/notebook_economics/formal_mechanism_*`

原因：这些目录包含 synthetic formal evidence、Phase46 comparison、Grouped ARDL 对照、formal mechanism audit 或 notebook 可复现结果。

## 二、下一步改进问题

### Q6：如果要继续增强机制证据，下一步最小工作是什么？

优先级如下：

1. **多 seed 正式复核**：把 economics / energy Phase46 扩展为完整 seeds `[0, 1, 2]`，包括 CMDL、Plain LSTM、No AC Encoder、Uniform Lag、No Recon、Grouped ARDL。
2. **Grouped ARDL lag trend 对照**：比较 low / mid / high proxy 组的 `best_lag`、`effective_lag` 与 AC-GATE `k*` 分位趋势是否一致。
3. **Energy 主图固化**：输出 WGI proxy quartile 的 `k*` summary、omega heatmap、per-proxy adjusted rho。
4. **Economics anchor audit**：分别审查 effective labor anchor、employment、human capital level、human capital trend 的方向，不再只看聚合指标。
5. **可选正则实验**：在多 seed 后再分别打开 `lambda_z_anchor` 或 entropy band，避免多个机制同时变化导致无法归因。

### Q7：如果要改进 economics，应该先改模型还是先改解释口径？

先改解释口径和审计，不要先改复杂模型。

原因：当前 economics 的负 rho 可能是真实的 anchor mismatch，也可能是理论符号设定过强。若直接加大 `lambda_z_anchor` 或改网络结构，容易变成结果导向调参。

建议顺序：

1. 先做 per-proxy audit。
2. 再做多 seed 稳定性检查。
3. 若只有 `z_i` 方向不稳，才尝试弱 `z_anchor`。
4. 若 proxy 本身方向冲突，则 economics 应作为 boundary case 写入论文。

### Q8：当前论文标题或主张应该怎么收束？

主张建议：

> AC-GATE: Absorptive-Capacity Conditioned Lag Discovery for Heterogeneous Panel Dynamics

核心贡献写法：

1. 提出一个实体级 proxy conditioned lag gate。
2. 学习每个实体的 lag distribution，而不只输出点预测。
3. 用 synthetic ground truth 验证可恢复性。
4. 用 ablation 证明 AC encoder 和 adaptive lag gate 的必要性。
5. 用 real data 展示机制非退化，并在 energy 中与治理 proxy 对齐。
6. 诚实报告 forecast calibration 边界和 economics anchor mismatch。
