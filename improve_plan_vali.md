# 潜在改进方法记录

更新时间：2026-04-19

## 一、总体策略

后续改进不应再围绕 synthetic 做大规模调参，而应集中处理真实域中的两类核心问题：

1. forecast 收益弱或不稳定；
2. mechanism diagnostics 与当前训练目标、评价 anchor 之间可能存在错位。

总体策略应遵循以下原则：

1. 先做低成本诊断和公平性控制，再做目标函数修改。
2. 先解决 economics，因为该域已有更清晰的 feature bundle、假设链和诊断线索。
3. energy 在证据不足前，不应过早投入复杂模型升级。
4. 所有真实域修改都必须回查 synthetic，确保不破坏已确认的机制结论。

## 二、优先级最高的改进方向

### 方向 1：先补齐 economics 的公平性与因果对比控制

目标：先确认当前 no_recon 与 full CMDL 之间的差距，究竟来自 reconstruction objective、initialization，还是优化副作用。

建议动作：

1. 继续使用 matched-init no_recon 对照，避免把初始化差异误判成 reconstruction 正负效应。
2. 补做 split_clip 与 global_clip 对照，检验 global gradient clipping 是否放大了 no_recon 与 full CMDL 的差异。
3. 将这些控制实验纳入 economics formal suite 的固定比较项。

预期收益：

1. 将“重构项到底有没有帮助”从混杂状态里分离出来。
2. 避免基于脏对照得出错误的机制结论。

### 方向 2：增加 anchor-aligned diagnostics，而不是立刻改模型

目标：先判断当前负 rho 是否来自真实反向关系，还是来自评价口径与训练目标不一致。

建议动作：

1. 在 notebook 和 comparison 导出中显式记录 anchor_proxy_name。
2. 显式记录 anchor_sign_convention，避免不同 proxy 定义下的符号误判。
3. 同时汇报 anchor-only readout quality 和 all-proxy readout quality。
4. 将 anchor proxy 与 auxiliary proxies 分开输出对应的 mechanism diagnostics，而不是压缩成单一 rho。

预期收益：

1. 可以区分“模型确实学反了”和“训练-评价口径错位”两种情况。
2. 为后续 objective 修改提供更明确的依据。

### 方向 3：只有在诊断仍失败时，才升级 economics 的 reconstruction objective

目标：在确认问题来自 objective mismatch 后，再最小幅度修改训练目标。

建议动作按顺序推进：

1. anchor-weighted reconstruction：保留当前多 proxy 合同，但给 anchor proxy 更高重构权重。
2. anchor-only reconstruction：只对机制 anchor 计算 reconstruction loss，辅助 proxies 只作为输入或诊断项。
3. two-head reconstruction：将 anchor 与 auxiliary proxies 分成两个重构头，只在前两步都说明存在真实目标冲突时再上。

预期收益：

1. 更直接地把训练目标对齐到机制评价目标。
2. 有机会在不牺牲 forecast 的情况下，提高 sign consistency 与 anchor-aligned mechanism quality。

## 三、Economics 的具体判断标准

后续任何 economics 改进版本，至少需要同时满足以下三条，才可视为真正改善：

1. mechanism sign consistency 提升，不再以负号主导。
2. effective k* 标准差不再接近塌缩水平，说明异质性确实被展开。
3. forecast 指标不低于 plain LSTM，至少不能用机制改善换来明显更差的预测表现。

如果只能改善其中一条，则不能宣称 economics 已成为强机制验证域。

## 四、Energy 的建议路线

### 方向 4：先补证据，不先改大模型

目标：先确认 energy 当前的微弱优势和负 rho 结论是否稳健。

建议动作：

1. 将 current energy ablation 扩展到至少 3 seeds。
2. 导出按 proxy 分解的 diagnostics，而不是只给聚合 rho。
3. 检查不同 seeds 下 sign 是否一致、k* spread 是否稳定，以及 CMDL 是否持续优于 plain LSTM。

预期收益：

1. 判断当前结论到底是稳定事实，还是单次随机结果。
2. 决定 energy 是否值得继续作为主验证域投入。

### 方向 5：若 energy 仍然弱，则优先收缩 claim，而不是继续加复杂度

目标：避免在弱证据域里不断堆模型复杂度。

建议动作：

1. 若多 seed 后仍只有边际 forecast 优势且 mechanism sign 不稳，则将 energy 下调为 generalization appendix。
2. 只保留最小修改尝试，例如更合适的 proxy anchor、目标变换或更合理的 treatment-target 配对。
3. 在没有明确证据表明 domain contract 有问题前，不做大规模结构改造。

预期收益：

1. 控制研究范围，避免 energy 吸走过多时间。
2. 让主线聚焦于更有希望做出强结论的 economics。

## 五、Synthetic 的使用方式

synthetic 后续的作用不是继续主攻，而是作为所有真实域修改的回归护栏。

具体要求：

1. 任何对 AC encoder、lag gate、reconstruction objective 的改动，都要回查 synthetic formal_target。
2. 必须继续保持以下结论不被破坏：
	1. full CMDL 明显优于 plain LSTM 的 k* 恢复；
	2. no_ac_encoder 与 uniform_lag 会显著退化；
	3. no_recon 与 full CMDL 基本接近。
3. 如果真实域修复破坏了 synthetic 上的这些已知结论，则该改动不应直接进入主线。

## 六、当前最推荐的推进顺序

### Phase 1：先补 diagnostics

1. economics 做 matched-init no_recon 与 clip control。
2. economics 增加 anchor-aligned diagnostics。
3. energy 扩到多 seed ablation，并增加分 proxy diagnostics。

### Phase 2：再做最小 objective 调整

1. economics 先试 anchor-weighted reconstruction。
2. 若仍无改善，再试 anchor-only reconstruction。
3. two-head reconstruction 只作为最后升级项。

### Phase 3：最后决定论文口径

1. 若 economics 成功提升 sign consistency 且 forecast 不输 baseline，则可把 economics 升格为机制支持域。
2. 若 energy 仍然只有弱收益，则将其降级为 generalization 或 feasibility 域。
3. synthetic 持续作为主要机制证明域。

## 七、当前最推荐的论文叙事

如果短期内不继续改代码，只依据现有结果，最稳妥的写法应为：

1. synthetic 负责强机制证明；
2. economics 负责展示真实域中的潜力和当前瓶颈；
3. energy 负责展示跨域可运行性，而不是强机制支持；
4. 论文中明确区分 forecast evidence 与 mechanism evidence，不将两者混为一谈。

## 八、结论

当前最值得投入的方法方向不是“继续盲目提升模型复杂度”，而是：

1. 先让真实域中的训练目标、诊断指标和机制评价 anchor 对齐；
2. 先用低成本控制实验把混杂因素拆掉；
3. 只有在确认 objective mismatch 后，才做有针对性的 loss 修改。

也就是说，下一阶段的重点不是证明 AC-GATE 在 synthetic 上还能更强，而是让 economics 与 energy 中“已经可见的结构优势”真正变成稳定、可解释、可报告的真实域收益。
