# 当前问题解决方案与后续改进计划

更新时间：2026-04-25

## 一、总策略：围绕 AC-GATE 机制，而不是单纯追逐预测 SOTA

`plan.md` 在论文撰写部分已经给出关键定位：模型价值在于发现异质滞后结构 `omega/k*`，而不是只压低 MSE。结合最新 Phase4/Phase6 验证，后续改进应围绕以下原则推进：

1. 主贡献聚焦于 **AC-conditioned heterogeneous lag discovery**。
2. Forecast 指标作为公平性、实用性和 sanity check，而不是唯一成功标准。
3. 真实域必须与 Plain LSTM、persistence、Panel OLS 等基线比较，但不必承诺在所有数据集上全面优于它们。
4. 若强简单基线持续领先，应降低预测性能 claim，转而强调机制诊断、异质性发现和适用边界。
5. 所有真实域修改都必须回查 synthetic，确保不破坏已验证的机制恢复能力。

一句话策略：

> 先证明 AC-GATE 能稳定识别和解释滞后异质性，再用 forecast calibration 证明它没有为解释性付出不可接受的预测代价。

## 二、已完成的关键改进

相比 2026-04-19 版本，当前已经完成以下改进：

1. 增加真实域统一机制诊断：anchor-aware、sign-adjusted rho、per-proxy rho、z_std、lag gate sensitivity。
2. 增加 Phase6 forecast calibration：train mean、entity mean、persistence、Panel OLS、best simple baseline。
3. economics / energy loader 已输出 anchor proxy metadata 和 expected sign。
4. CMDL objective 已支持 `anchor_weighted`、`anchor_only`、`reconstruction_detach` 控制。
5. economics / energy CMDL、Plain LSTM、ablation runners 已接入新诊断与新基线。
6. comparison table 已包含 Phase4/Phase6 关键字段。
7. 单元测试已通过：`56 passed, 0 failed`。
8. economics 和 energy notebook 已新增 Phase4/Phase6 upgraded formal validation section，并完成单 seed focused formal validation。

这些改进说明：当前问题已经不再是“缺少诊断工具”，而是“如何根据诊断结果收束论文 claim 和后续实验”。

## 三、当前问题与解决方案

### 问题 1：真实域 forecast 不能支撑全面性能优越性

现象：

1. Economics：CMDL `test_r2=0.0445`，高于 Plain LSTM `0.0084`，也略高于 Panel OLS，但低于 persistence。
2. Energy：CMDL `test_r2=-0.0265`，略高于 Plain LSTM `-0.0325`，但低于 persistence 和 Panel OLS。

判断：

1. AC-GATE 可以说在 focused run 中超过 matched Plain LSTM。
2. 不能说 AC-GATE 在真实域预测上全面优于简单强基线。
3. persistence 在真实面板中非常强，若强行以预测 SOTA 为主目标，会削弱论文说服力。

解决方案：

1. 论文中把 forecast 作为 calibration evidence，而不是主贡献。
2. 保留 `delta_vs_persistence`、`delta_vs_panel_ols`，明确说明模型是否超过强简单基线。
3. 若未来补 TFT / grouped ARDL，需将其作为强 baseline 或 appendix robustness，而不是把所有工作都改成追逐预测榜单。
4. 真实域 claim 改为“competitive with matched LSTM and diagnostically informative”，而不是“dominates forecasting baselines”。

### 问题 2：economics 机制方向仍未被支持

现象：

1. Economics anchor-adjusted rho 为 `-0.1590`。
2. 这意味着当前 effective-labor anchor 下，k* 与预期方向没有对齐。
3. 但 `lag_gate_sensitivity_range=0.1046`、`z_std=0.1704`，说明模型不是简单退化。

判断：

1. economics 的问题不是模型完全学不到结构，而是结构方向与当前理论 anchor 不一致。
2. 不应为了让 rho 变正而盲目加复杂结构，否则可能变成结果导向调参。
3. 需要把 economics 写成限制性证据或机制失败案例，除非多 seed 和 anchor audit 后能稳定修复。

解决方案：

1. 先做多 seed 复核，判断负方向是否稳定。
2. 审查 economics anchor 的理论符号：effective labor、人力资本、TFP 之间是否真的应对应更短滞后。
3. 分开报告 anchor proxy 与 auxiliary proxies，不再用单一聚合 rho 掩盖方向差异。
4. 若多 seed 后仍为负，将 economics 降级为 limitation / boundary case，而不是继续强推为机制支持域。
5. 只有当 anchor audit 证明目标函数错位时，才继续尝试 anchor-only 或 two-head reconstruction。

### 问题 3：energy 机制证据强，但预测校准不足

现象：

1. Energy anchor-adjusted rho 为 `0.6442`。
2. WGI 三个 proxy 的 adjusted rho 都为正：government effectiveness `0.6442`，regulatory quality `0.6444`，rule of law `0.6820`。
3. `lag_gate_sensitivity_range=0.1830`，`z_std=0.0956`，说明 learned heterogeneity 非退化。
4. 但 CMDL 预测低于 persistence 和 Panel OLS。

判断：

1. energy 是当前更好的真实域机制诊断证据。
2. 但它不能支持“预测性能优越”主张。
3. 它适合作为“AC-GATE 在真实域能输出与治理 proxy 对齐的滞后异质性”的案例。

解决方案：

1. 将 energy 的主叙事从 forecast superiority 改为 mechanism alignment。
2. 保留 Phase6 calibration，诚实说明简单强基线预测更强。
3. 后续可尝试目标变换、异常值处理、按国家类型分组报告，以检查 forecast 劣势是否由极端国家 / 极端年份驱动。
4. 若多 seed 机制方向仍稳定，则 energy 可作为真实域主机制展示图表。

### 问题 4：单 seed notebook validation 还不够论文级

现象：

1. 当前 notebook 正式验证是 seed `[0]` 的 focused run。
2. 它足以证明改进 pipeline 可运行，并给出当前诊断方向。
3. 但不足以支撑最终论文表格。

解决方案：

1. 将 Phase4/Phase6 notebook 或 runner 扩展到 seeds `[0, 1, 2]`。
2. 对 task metrics、adjusted rho、z_std、lag sensitivity 做 mean ± std。
3. 对 CMDL vs Plain LSTM、CMDL vs ablation 做 paired comparison。
4. 若真实域多 seed 仍不稳，在论文中明确标注为 exploratory real-data evidence。

### 问题 5：需要把 ablation 从“附属实验”提升为核心证据

现象：

1. `no_ac_encoder` 在真实域会让 kstar_std 退化为 0。
2. `uniform_lag` 会让 lag 分布退化为固定峰值。
3. 这些结果直接证明 AC-GATE 的异质滞后结构不是自然出现的，而是由核心模块产生的。

解决方案：

1. 在论文表格中固定报告 full CMDL、Plain LSTM、no_ac_encoder、uniform_lag、no_recon。
2. 将 ablation 作为机制必要性的主证据，而不是仅作为 robustness。
3. 对真实域也报告 `z_std`、`kstar_std`、`lag_gate_sensitivity_range`、`omega_top1_share`，避免只看 R2。

## 四、下一阶段推荐推进顺序

### Phase A：先收束研究问题与 claim

1. 明确主问题：AC-GATE 是否能学习实体级异质滞后结构？
2. 明确副问题：这种结构是否在真实域保持合理预测能力？
3. 明确不主张：不承诺真实域全面预测 SOTA。
4. 修改论文叙事：把 forecast baseline 写成 calibration，不写成唯一 scoreboard。

### Phase B：补多 seed formal validation

1. economics Phase4/Phase6 seeds `[0, 1, 2]`。
2. energy Phase4/Phase6 seeds `[0, 1, 2]`。
3. 输出 mean ± std result logs。
4. 若资源允许，再扩到 5 seeds 用于最终统计检验。

### Phase C：补强 Plan.md 中 Step 6 的强 baseline

优先级建议：

1. persistence 和 Panel OLS 已完成，应保留为最小强校准。
2. grouped ARDL 是最重要的下一 baseline，因为它是“人工分组版 AC-GATE”，最能检验异质滞后发现是否有增量价值。
3. TFT baseline 可作为 appendix 或后续扩展；若时间紧，不应让 TFT 抢走机制论文主线。

### Phase D：针对 economics 做 anchor audit，而不是先改复杂模型

1. 检查 effective labor anchor 与 TFP lag 的理论方向。
2. 分开看 human capital、employment、hours worked 等 proxy 的 adjusted rho。
3. 若不同 proxy 方向冲突，则说明 economics 不适合作为强机制支持域。
4. 若只有当前 anchor 错位，再考虑重定义 anchor 或切换 target。

### Phase E：把 energy 做成机制展示域

1. 多 seed 验证 WGI per-proxy adjusted rho 是否稳定。
2. 画 k* vs WGI proxy 分位数组图。
3. 报告 lag distribution heatmap 和 proxy quartile k* summary。
4. 同时在表中说明 forecast 没有超过 persistence / Panel OLS。

### Phase F：保持 synthetic 为机制回归护栏

所有真实域改动必须回查 synthetic：

1. full CMDL 的 k* recovery 不能被破坏。
2. no_ac_encoder 和 uniform_lag 必须继续明显退化。
3. no_recon 与 full 的关系应继续用于说明 reconstruction 不是核心增益来源。

## 五、论文最终建议口径

建议主线：

1. Method：提出 AC-GATE，将实体级 proxy 映射为滞后权重分布。
2. Synthetic：在有 ground truth 的环境下证明能恢复异质滞后，并通过消融证明模块必要。
3. Real data：展示该机制在真实面板上可运行、非退化，并在部分域与理论 proxy 对齐。
4. Forecast：报告 matched LSTM 与强简单 baseline 的校准结果，说明模型预测能力的边界。
5. Limitation：真实域预测不一定超过 persistence / Panel OLS，economics anchor alignment 仍是未解决问题。

## 六、当前最重要的结论

当前改进方向不应再是“继续加模型复杂度以追求 R2”，而应是：

1. 明确 AC-GATE 的研究目标是滞后异质性机制，而不是纯预测榜单；
2. 用 synthetic 和 ablation 证明机制有效；
3. 用 real data 证明机制能在真实域产生非退化、可诊断的结构；
4. 用 forecast calibration 诚实报告适用边界；
5. 对 economics 保持克制，对 energy 强化机制展示。

也就是说，性能优于其他方法是有价值的加分项，但不是本文唯一的成败标准。本文真正需要守住的是：AC-GATE 是否提供了传统 forecasting baseline 无法直接给出的、可解释的实体级滞后异质性证据。