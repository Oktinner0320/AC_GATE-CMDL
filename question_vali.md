# 当前结论与问题记录

更新时间：2026-04-19

## 一、总体结论


1. 在 synthetic 合成数据上，AC-GATE 不仅验证了机制可行，而且在结构恢复和任务指标上都明显优于 plain LSTM。
2. 在 economics 和 energy 两个真实数据域上，AC-GATE 的模块必要性和异质滞后学习迹象是存在的，但这种结构优势尚未稳定转化为更好的预测性能。
3. 因此，当前真正的问题不是“AC-GATE 机制无效”，而是“真实域中的机制优势尚未稳定兑现为 forecast 优势，且机制解释证据仍不够稳”。

## 二、分域发现

### 1. Synthetic：机制已验证，不是当前瓶颈

当前 synthetic 结果已经说明：

1. full CMDL 相比 plain LSTM，在 linear 和 nonlinear 场景下的 k* 恢复明显更好。
2. no_ac_encoder 与 uniform_lag 会系统性破坏 k* 排序恢复，说明 AC conditioning 和 adaptive lag gating 是必要模块。
3. no_recon_regularization 与 full CMDL 基本重合，说明 reconstruction regularization 不是 synthetic 主增益来源。

由此可得：

1. synthetic 已经承担了主要机制证明任务。
2. 当前不应继续把 synthetic 当作主调参战场，而应把它作为后续真实域修改的回归护栏。

### 2. Economics：预测增益很弱，机制符号不稳甚至反向

当前 economics 域暴露出两个层面的问题：

#### 2.1 Forecast 层面

1. formal_target 下，CMDL 的平均 test R2 只比 plain LSTM 略好一点，但两者整体仍处于负 R2 区间。
2. 这说明 AC-GATE 在 economics 上并非完全无效，但现阶段预测收益非常有限，尚不足以构成强有力优势。

#### 2.2 Mechanism 层面

1. formal_target 下，CMDL 的 lag-proxy Spearman rho 明显为负，而 plain LSTM 反而接近零。
2. growth_aware 与 effective_labor_aware 虽然改善了部分 forecast 表现，但并没有稳定修复机制符号问题。
3. effective_labor_aware 相比 growth_aware 更有利于 forecast，但 CMDL 的机制相关指标仍以负号或符号不一致为主。
4. 当前 CMDL 的 effective k* 标准差普遍偏小，说明模型虽未完全塌缩，但异质滞后展开仍不充分。

由此可得：

1. economics 的核心问题不是“模型学不到任何结构”，而是“学到的结构与当前 proxy anchor 的评估方向不一致”。
2. 当前还不能把 economics 写成 AC-GATE 的强机制验证域。

### 3. Energy：只有边际预测优势，机制证据不足

当前 energy 域的主要现象是：

1. CMDL 对 plain LSTM 只有很小的平均 test R2 优势。
2. lag-proxy Spearman rho 的均值为负，且 seed 间波动非常大，说明机制符号不稳定。
3. effective k* 标准差明显小于 plain LSTM 的 post-hoc k* spread，说明 learned heterogeneity 仍偏弱。
4. 现有 energy ablation 证据强度不足，仍偏接近单 seed 结论，无法支持强机制主张。

由此可得：

1. energy 当前最多能支持“AC-GATE 在该域可运行且有轻微 forecast 改善”。
2. 还不能支持“AC-GATE 在 energy 上稳定优于 LSTM”这一强表述。

## 三、当前已确认的问题

### 问题 1：真实域里 forecast 优势不稳定

1. synthetic 上 CMDL 明确优于 LSTM。
2. 但在 economics 和 energy 上，forecast 优势要么很弱，要么只在少量配置下成立。
3. 这说明当前 AC-GATE 的结构收益跨域迁移能力仍不足。

### 问题 2：真实域 mechanism 指标与 forecast 指标脱节

1. 在 economics 和 energy 上，模型并非完全没有预测能力。
2. 但 lag-proxy rho、k* spread、sign consistency 等 mechanism 证据不稳。
3. 因此当前出现了“预测还能跑，但机制解释不成立或不稳”的割裂状态。

### 问题 3：当前真实域的评估口径可能与训练目标不完全对齐

1. 经济域当前训练时重构多个 proxies，但 notebook 机制判断往往只盯一个 anchor proxy。
2. 这可能导致模型优化目标和最终机制评价目标不一致。
3. 现有负 rho 结果不能直接解释为 AC-GATE 机制失败，也可能是 objective 与 evaluation anchor 错位。

### 问题 4：reconstruction 分支的解释需要谨慎

1. 当前 AC encoder 中 proxy reconstruction 使用了 detached z 表示。
2. economics 和 energy 的 proxy 指标又依赖训练后 OLS refit。
3. 因此真实域里的 proxy_recon_r2 和 lag-proxy rho，更接近“冻结后 latent 的可读性诊断”，而不完全等同于“端到端联合学习是否成功”。

### 问题 5：energy 目前证据强度不够

1. 当前 energy 的多 seed 机制结论仍然很弱。
2. 若不先补强证据，就继续做复杂建模修改，风险很高。

## 四、当前最稳妥的研究口径

现阶段最稳妥的总结应为：

1. AC-GATE 已在 synthetic 上强力验证了异质滞后机制本身是有效且优于 plain LSTM 的。
2. 在 economics 和 energy 上，AC-GATE 已显示出一定的结构学习迹象和局部 forecast 收益。
3. 但真实域中的机制证据仍不稳定，尚不足以支持“AC-GATE 在真实数据上稳定优于 LSTM”的强结论。
4. 因而当前论文叙事应把 synthetic 作为主要机制证明域，把 economics 和 energy 暂时视为 generalization 或 feasibility 域，而不是同强度的机制验证域。
