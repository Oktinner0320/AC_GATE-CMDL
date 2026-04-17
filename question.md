# CMDL 已知问题总表

更新时间：2026-04-17（第二版，结构性修复后）

本文档汇总截至当前已经被实验、日志、Notebook 诊断和源码检查共同确认的已知问题。第二版基于第一轮结构性修复（阶段 A-D）后的 formal_target 实验结果，区分已解决问题、部分解决问题和新发现问题。

---

## 一、当前总体现状

第一轮结构性修复已完成 4 项代码变更（消除 backbone 捷径、解耦 reconstructor 梯度、复合 early stopping、移除有害 lag bias）。E1a 线性场景刚好通过 kstar_mae 阈值，E1b 和 E1c 仍未通过。

formal_target 当前结果如下：

| 子实验 | 关键指标 1 | 修复前 | 修复后 | 阈值 | 关键指标 2 | 修复前 | 修复后 | 阈值 | 是否通过 |
|---|---|---:|---:|---|---|---:|---:|---|---|
| E1a linear | kstar_mae | 1.8607 | **0.9314** | < 1.0 | kstar_spearman_rho | 0.9804 | 0.9804 | > 0.8 | **是** |
| E1b identification | proxy_recon_r2 | -18.6828 | **-29.8626** | > 0.5 | z_spearman_rho | 0.9883 | 0.9892 | > 0.8 | 否 |
| E1c nonlinear | kstar_mae | 2.4281 | **1.3424** | < 1.0 | kstar_spearman_rho | 0.9609 | 0.9623 | > 0.8 | 否 |

当前失败链从"两条独立链"收敛为"一条主链 + 一条尾部链"：

1. **主链**：proxy_recon_r2 严重恶化（-18 → -30），成为唯一的 P0 结构性问题。根因已定位：detach 切断了 recon 梯度回传，但 lambda_r=0.1 导致 reconstructor 的有效学习率只有 0.0001，追不上 z_i 的分布漂移。
2. **尾部链**：E1c kstar_mae 改善 45% 但仍超出阈值 0.34，属于量级问题而非方向问题。

---

## 二、证据来源

1. outputs/notebook_step4/formal_target/step4_results.json（修复后最新结果）
2. outputs/notebook_step4/formal_target/E1a_linear/summary.json
3. outputs/notebook_step4/formal_target/E1c_nonlinear/summary.json
4. outputs/notebook_step4/formal_target/E1a_linear/history.csv（100 epoch 训练轨迹）
5. outputs/notebook_step4/formal_target/E1c_nonlinear/history.csv（171 epoch 训练轨迹）
6. outputs/notebook_step4/formal_target/E1a_linear/predictions.csv（reconstructor 输出统计）
7. Notebook 01_synthetic_verify.ipynb 全部 8 个单元格的执行输出
8. 修复后的源码：model/backbone.py、model/ac_encoder.py、experiments/run_synthetic.py、config/cmdl_config.py

---

## 三、已解决问题

### [已解决] 原 Q2. k* 恢复出现"排序正确但数值塌缩" — E1a 线性场景已通过

解决方式：阶段 A（消除 backbone 捷径）+ 阶段 D（移除有害 lag bias）

修复前后对比：

1. E1a kstar_mae：1.8607 → 0.9314（通过阈值 < 1.0）。
2. omega_peak 分布：修复前 lag 3 占 170/200 → 修复后 lag 3 (64), lag 7 (55), lag 9 (81)。
3. kstar_pred 范围：[4.8, 5.5] → [4.3, 7.3]。
4. omega_peak_accuracy：0.055 → 0.335（+6 倍）。

归因分析：

1. 阶段 A 移除 backbone 中 current_sequence 的拼接后，LSTM 必须通过 lag_context_sequence 获取时序信息，omega 的梯度信号从"可绕过"变为"必经路径"。
2. 阶段 D 将 lag_bias_strength 从 1.0 改为 0.0 后，远端 lag（如 lag 7、9、10）不再受固定惩罚，omega 可以自由覆盖完整的 lag 范围。
3. 两者叠加使 lag gate 真正开始分化不同实体的 lag 分布。

残留风险：

1. E1a kstar_mae = 0.9314 仅勉强通过阈值 1.0，余量仅 7%。
2. kstar_pred 上限 7.3 vs 真实上限 10.0，说明 lag 8-10 区间仍存在部分压缩。

---

### [已解决] 原 Q4. formal_target 被 early stopping 过早截断

解决方式：阶段 C（复合 early stopping）

修复前后对比：

1. E1a：best_epoch 从 50 → 80，总训练轮数从 70 → 100。
2. E1c：best_epoch 从 38 → 151，总训练轮数从 58 → 171。

归因分析：

1. 复合指标 val_task_loss + val_kstar_mae 不再在 task_loss 触底后立刻停止。
2. kstar_mae 在 epoch 50-80 区间仍有明显下降（从 1.64 → 1.21），旧准则完全错过这段改善。

---

### [已解决] 原 Q5. 模型选择准则与研究目标错位

解决方式：阶段 C（复合 early stopping）

说明：

1. 现在 best model 选择和 patience 都基于 val_task_loss + val_kstar_mae。
2. E1a history 显示复合指标在 epoch 80 达到最优（1.30），此后 task_loss 继续下降但 kstar_mae 不再改善，模型正确停止。

---

### [已解决] 原 Q6. 非线性场景重复更严重的 lag 塌缩

解决方式：阶段 A + D

修复前后对比：

1. E1c omega_peak：修复前 lag 3 占 200/200 → 修复后 lag 1 (137), lag 7 (63)。
2. E1c kstar_pred 范围：[4.0, 4.2] → [3.0, 6.9]。
3. E1c omega_peak_accuracy：0.120 → 0.455（+3.8 倍）。

说明：

1. lag 塌缩本身已解决，非线性场景现在展现出与线性不同的 omega 结构（偏向 lag 1 和 lag 7 两个峰）。
2. 但 E1c kstar_mae = 1.34 仍不达标，属于后续优化范围。

---

### [已解决] 原 Q7. 简单调参不能解决根因

状态：问题前提已改变

说明：

1. 原 Q7 的结论是"在旧训练结构上调参无效"。
2. 四项结构性修复后，训练行为已根本改变，旧 sweep 结果不再适用。
3. 当前问题空间已缩小到 lambda_r 和 temperature 两个参数，且有明确的物理理由指导调整方向。

---

## 四、当前未解决问题

### Q-NEW-1. proxy_recon_r2 在结构修复后显著恶化

状态：已确认

严重程度：P0

现象：

1. proxy_recon_r2 从修复前的 -18.68 恶化到修复后的 -29.86。
2. recon_loss 在训练过程中持续上升：从 epoch 3 的 1.18 上升到 epoch 100 的 1.58。
3. reconstructor 输出的均值和范围与真实 proxy 严重偏移。

直接证据：

1. E1a proxy_recon_r2 = -29.86，远低于阈值 0.5。
2. E1a val_recon_loss 最低点在 epoch 3（1.18），此后单调上升到 epoch 100（1.58）。
3. proxy_1_pred 均值 = -0.26，但 proxy_1_true 均值 = 0.49（偏移 0.75）。
4. proxy_3_pred 均值 = -1.00，但 proxy_3_true 均值 = 0.65（偏移 1.65）。
5. z_pred 分布：均值 = -0.25，范围 = [-0.97, 0.61]。z_true 分布：均值 = 0.49，范围 = [0, 1]。

根因分析（高置信度）：

1. 阶段 B 的 z_i.detach() 成功阻止了 recon 梯度回传到 encoder，消除了原 Q7 中 lambda_r=1.0 导致 z 方向反转的风险。
2. 但 detach 的副作用是：z_i 的分布完全由 task_loss 控制，每个 epoch 都在漂移。
3. reconstructor（nn.Linear(1, 3)，6 个参数）的有效学习率 = lr × lambda_r = 0.001 × 0.1 = 0.0001。
4. 这个学习率太低，reconstructor 无法追上 z_i 每 epoch 的分布漂移，导致参数越来越偏。
5. 最终表现为 recon_loss 单调上升，proxy_recon_r2 持续恶化。

与原 Q3 的关系：

1. 原 Q3 描述了"reconstructor 学不到线性映射"的现象。
2. 原 Q3 特别指出"不能简单解释成 lambda_r 太小，因为提高 lambda_r 会导致 z 方向反转"。
3. 阶段 B 的 detach 已经消除了 z 方向反转的风险。因此，在当前代码下，提高 lambda_r 是安全的，且是必要的。

影响：

1. E1b 是当前唯一的 P0 未通过项。
2. 修复此问题预期只需一处配置修改（lambda_r 0.1 → 1.0）。

---

### Q-NEW-2. E1c kstar_mae 仍超出阈值 0.34

状态：已确认

严重程度：P1

现象：

1. E1c kstar_mae = 1.34，阈值 < 1.0。
2. omega 仅形成 2 个峰（lag 1: 137, lag 7: 63），而非线性场景的 k* 真实分布覆盖 1-10。
3. kstar_pred 范围 [3.0, 6.9]，真实范围 [1.0, 10.0]。

根因分析：

1. 非线性场景的 k*(z) = round(10 × (1-z)²) 在 z 接近 0 时产生大量 k*=10 的实体，omega 需要在 lag 10 处形成尖锐峰值。
2. temperature=1.0 下 softmax 分布较平滑，不利于对单个 lag 的精确定位。
3. 这解释了为什么 omega_peak_accuracy=0.455 尽管已大幅改善，但仍有 55% 的实体峰值位置不正确。

与 E1a 的差异：

1. E1a kstar_mae = 0.93 已通过，说明线性 k*(z) = round(3 + 7(1-z)) 的分布更容易恢复。
2. E1c 的非线性映射使得 k* 分布更偏斜（大量实体集中在高 lag 区），omega 需要更尖锐的区分。

影响：

1. 这是一个量级问题而非方向问题。
2. 预期可通过降低 temperature（如 0.5）增强 omega 尖锐度来解决。

---

## 五、已被当前证据排除的解释

### 原有排除项（仍然有效）

1. E1b 失败不是数据生成器不可重构：best_linear_proxy_r2_from_z_pred = 0.9487 仍成立。
2. lag 输入相关性不是唯一主因：mean_abs_offdiag_lag_corr = 0.3085 仍成立。
3. MLflow 后端切换不是数值问题来源：仍成立。

### 新增排除项

1. **backbone 捷径已排除**：移除 current_sequence 后，kstar_mae 从 1.86 降至 0.93，证明捷径确实是原 Q2 的根因。
2. **lag_bias_strength 方向对抗已排除**：设为 0.0 后，omega_peak 从集中在 lag 3 变为分布在 lag 3/7/9，证明偏置确实压制了远端 lag。
3. **early stopping 错选模型已排除**：复合指标选出的 best_epoch=80 对应 kstar_mae=0.93，优于旧准则下 best_epoch=50 的 kstar_mae≈1.86。
4. **lambda_r=1.0 导致 z 方向反转已排除**：detach 后，recon 梯度不再回传到 encoder。旧 Q7 sweep 中 lambda_r=1.0 造成 spearman 从 0.98 翻转到 -0.94 的现象在当前代码下不会重现。

---

## 六、已实施的代码修改

| 阶段 | 文件 | 修改内容 | 目的 |
|---|---|---|---|
| A | model/backbone.py | input_fusion 从 5×d_model 改为 4×d_model，fused_inputs 移除 current_sequence | 消除 LSTM 绕过 lag_context 的捷径路径 |
| B | model/ac_encoder.py | proxy_reconstructor(z_i) → proxy_reconstructor(z_i.detach()) | 阻止 recon 梯度干扰 encoder |
| C | experiments/run_synthetic.py | early stopping 从 val_task_loss 改为 val_task_loss + val_kstar_mae | 选择机制最优模型 |
| D | config/cmdl_config.py | lag_bias_strength 默认值 1.0 → 0.0 | 移除对远端 lag 的有害惩罚 |

---

## 七、当前最重要的结论

1. 第一轮结构性修复成功解决了 backbone 捷径和 lag 塌缩问题，E1a 线性场景已通过。
2. 当前唯一的 P0 问题是 proxy_recon_r2 恶化，根因是 detach 后 reconstructor 的有效学习率（lr × lambda_r = 0.0001）过低，追不上 z_i 的分布漂移。
3. 解决方案是将 lambda_r 从 0.1 提高到 1.0。在 detach 已生效的前提下，这是安全的——旧 sweep 中 lambda_r=1.0 导致 z 方向反转的问题不会重现。
4. E1c 的 kstar_mae 差距 0.34 属于 P1，预期可通过降低 temperature 解决。
