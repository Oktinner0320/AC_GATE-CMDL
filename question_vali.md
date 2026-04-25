# 当前问题、证据与研究定位记录

更新时间：2026-04-25

## 一、先回答核心定位问题

结合 `plan.md` 的研究设计，本项目的核心目的不是提出一个在所有真实数据集上都必须压过所有 forecasting baseline 的通用预测器，而是提出并验证 **AC-GATE 机制**：用吸收能力 / 治理质量 / 人力资本等实体级 proxy 去条件化滞后权重，从而研究不同实体之间的 **滞后异质性**。

因此，性能比较的地位应当这样界定：

1. **预测性能不是唯一核心贡献**：AC-GATE 的主要贡献是让模型输出实体级 lag distribution、effective k* 和与 proxy 对齐的机制诊断。
2. **预测性能仍是必要约束**：如果模型预测明显差于 matched LSTM 或简单强基线，就很难说明学到的滞后结构具有实用意义。
3. **不需要宣称全面 SOTA**：真实面板数据中 persistence、Panel OLS 或 ARDL 可能天然很强，AC-GATE 不必在所有域都超过它们，尤其不应把论文主张写成“预测性能全面优于传统方法”。
4. **需要证明机制不是装饰**：AC-GATE 至少要在 synthetic 与消融实验中证明 AC conditioning 和 adaptive lag gate 对滞后恢复是必要的；在真实域中则要展示非退化的 lag heterogeneity 和可解释的 proxy alignment。

最稳妥的研究口径是：

> AC-GATE is a mechanism-oriented neural panel model for discovering AC-conditioned heterogeneous lag structures. Forecasting performance is used as a calibration and sanity check, not as the sole contribution.

## 二、当前最新证据概览

### 1. Synthetic：机制主张已经成立

synthetic 仍是最强的机制证明域：

1. full CMDL 在 linear / nonlinear synthetic 场景下明显优于 matched plain LSTM 的 k* 恢复。
2. `no_ac_encoder` 和 `uniform_lag` 会系统性破坏 k* 恢复，说明 AC conditioning 与 adaptive lag gating 是必要模块。
3. `no_recon_regularization` 与 full CMDL 接近，说明主要增益来自 AC-GATE 的条件化滞后结构，而不是单纯 reconstruction regularization。

这说明当前问题不是 AC-GATE 机制本身无效，而是真实域中的证据强度、评估 anchor、简单基线校准和论文口径需要进一步收束。

### 2. Economics：forecast 有局部改善，机制方向仍不支持

最新 Phase4/Phase6 notebook focused formal validation 使用 seed `[0]`、`smoke=False`、30 epochs。结果显示：

1. Forecast 层面：CMDL `test_r2=0.0445`，高于 Plain LSTM `test_r2=0.0084`。
2. Phase6 calibration：CMDL 高于 Panel OLS，`delta_vs_panel_ols=+0.0235`，但明显低于 persistence，`delta_vs_persistence=-0.8923`，因此只能判为 partial。
3. Phase4 mechanism：anchor-adjusted rho 为 `-0.1590`，没有满足预期机制方向。
4. Heterogeneity：`lag_gate_sensitivity_range=0.1046`，`z_std=0.1704`，说明 lag gate 和 latent 表征不是常数退化。
5. Ablation guard：`no_ac_encoder` 的 `kstar_std=0.0`，`uniform_lag` 的 `top1_share=1.0`，退化对照能暴露异质性边界。

由此可得：economics 可以支持“AC-GATE 在真实经济面板上具备一定 forecast signal 和非退化滞后结构”，但不能支持“economics 已经提供强机制方向证据”。当前 economics 的主要问题仍是机制 anchor 与学到的 k* 方向不一致。

### 3. Energy：机制证据明显增强，但预测仍输给简单强基线

最新 Phase4/Phase6 notebook focused formal validation 同样使用 seed `[0]`、`smoke=False`、30 epochs。结果显示：

1. Forecast 层面：CMDL `test_r2=-0.0265`，略高于 Plain LSTM `test_r2=-0.0325`。
2. Phase6 calibration：CMDL 低于 persistence 和 Panel OLS，`delta_vs_persistence=-0.7854`，`delta_vs_panel_ols=-0.6594`，因此不能宣称预测优于简单强基线。
3. Phase4 mechanism：anchor-adjusted rho 为 `0.6442`，满足预期机制方向。
4. Per-proxy WGI：三组 proxy 均为正，`government_effectiveness=0.6442`，`regulatory_quality=0.6444`，`rule_of_law=0.6820`。
5. Heterogeneity：`lag_gate_sensitivity_range=0.1830`，`z_std=0.0956`，说明 learned lag gate 非退化。

由此可得：energy 当前更适合作为真实域机制可解释性和跨域可运行性的支持证据，但它不是强 forecasting superiority 证据。

## 三、当前真正的问题

### 问题 1：论文主张需要从“预测优越性”转向“机制识别 + 合理预测”

如果把项目写成预测模型论文，就会被 persistence、Panel OLS、TFT、ARDL 等强基线直接挑战。更合理的主张是：AC-GATE 提供传统方法难以直接给出的实体级异质滞后分布，并通过机制诊断解释不同 proxy 条件下的传导速度差异。

### 问题 2：真实域 forecast 只能作为校准证据，不能单独支撑主贡献

economics 和 energy 都能在 focused run 中超过 matched Plain LSTM，但都没有稳定压过最强简单 baseline。尤其 persistence 在真实面板中非常强，这说明当前模型不能被包装成通用预测 SOTA。

### 问题 3：economics 机制方向仍是主要短板

economics 的 adjusted rho 仍为负，说明当前 effective-labor anchor 下，学到的滞后结构没有按预期方向排列。可能原因包括：

1. 经济机制假设本身过于简化；
2. anchor proxy 的符号或含义与 k* 预期关系不完全匹配；
3. 训练目标虽然已 anchor-weighted，但真实数据中的 proxy 关系仍不支持该方向；
4. 单 seed focused run 仍不足以判断稳定性。

### 问题 4：energy 的机制结果好，但 forecast 口径必须克制

energy 的 WGI proxy alignment 很强，但 test R2 不及 persistence / Panel OLS。该域应写成“机制诊断支持 + 可运行性证据”，而不是“预测性能优势证据”。

### 问题 5：当前 notebook 正式验证仍是单 seed

这轮 notebook 验证是必要且有价值的 focused formal validation，但还不是论文级 3-seed 或 5-seed 聚合。所有结论都应标注为 focused run evidence，最终表格需要多 seed 复核。

## 四、当前最稳妥的研究结论

1. AC-GATE 的核心机制在 synthetic 上已经被清楚验证：它能恢复 AC-conditioned heterogeneous lags，且关键模块消融会退化。
2. 在真实域中，AC-GATE 已经能输出非退化的 lag heterogeneity，并在 economics / energy 上都略优于 matched Plain LSTM。
3. 但真实域中的 forecast superiority 不能作为主结论，因为简单强基线仍然很难被稳定超过。
4. economics 的机制方向仍不成立，应作为限制、反例或待修复域处理。
5. energy 的机制方向较好，可以作为真实域机制诊断的主要支持案例，但仍需承认预测校准不足。

## 五、建议论文口径

当前论文不应写成：

1. AC-GATE universally outperforms all baselines in forecasting.
2. Real-data domains already prove stable predictive superiority.
3. Economics already validates the expected mechanism.

当前论文更适合写成：

1. AC-GATE proposes a learnable mechanism for entity-level heterogeneous lag discovery.
2. Synthetic experiments identify and validate the mechanism under known ground truth.
3. Real-data experiments test whether the mechanism remains non-degenerate and diagnostically meaningful.
4. Forecasting metrics are reported as calibration evidence: the model should remain competitive, but full forecasting dominance is not the central claim.
5. The strongest current real-data mechanism evidence appears in energy; economics reveals important limitations of proxy-anchor alignment.

## 六、当前判断

当前项目仍然成立，但需要把主轴从“我要证明预测性能优于其他方法”调整为“我要证明 AC-GATE 能研究滞后异质性，并在预测上保持合理竞争力”。

性能优于其他方法是加分项，不是唯一成败标准；但若模型长期显著弱于 matched LSTM 或简单强基线，则必须降低真实域 claim 强度。最合理的评价层级应为：

1. synthetic 证明机制可恢复；
2. ablation 证明模块必要；
3. real data 证明 lag gate 非退化且部分域 proxy alignment 成立；
4. forecast baseline 证明模型没有为了解释性付出不可接受的预测代价。