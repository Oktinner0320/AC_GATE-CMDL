# CMDL Step 4 改进计划

更新时间：2026-04-17（第二版）

本文档记录 Step 4 合成实验的结构性修复方案。第二版标注了已完成阶段的实际效果，并新增了针对遗留问题的后续阶段。

---

## 总体策略

第一轮修复（阶段 A-D）聚焦于 4 个已确认的训练结构缺陷。第二轮修复（阶段 E-F）聚焦于第一轮修复暴露出的参数不适配问题。

当前修复链：

- 阶段 A-D（已完成）：消除根因级训练结构缺陷 → E1a 通过，E1b/E1c 暴露新瓶颈
- 阶段 E（P0）：修复 detach 后 reconstructor 学习率不足 → 预期 E1b 通过
- 阶段 F（P1）：增强非线性场景 omega 尖锐度 → 预期 E1c 通过

---

## 阶段 A：消除 backbone 捷径路径 — ✅ 已完成

问题：backbone LSTM 的 input_fusion 拼接了 current_sequence，模型可以绕过 lag_context_sequence 直接获取时序信息，使 omega 的梯度信号极弱。

修改文件：model/backbone.py

修改内容：

1. input_fusion 维度从 5×d_model 改为 4×d_model。
2. fused_inputs 的 torch.cat 中移除 current_sequence。
3. current_sequence 参数保留但仅用于形状推断。

实际效果：

1. E1a kstar_mae：1.8607 → 0.9314（与阶段 D 联合效果）。
2. omega_peak 分布：从 lag 3 占 170/200 变为 lag 3 (64), lag 7 (55), lag 9 (81)。
3. omega_peak_accuracy：0.055 → 0.335。

---

## 阶段 B：解耦 reconstructor 梯度 — ✅ 已完成

问题：proxy_reconstructor(z_i) 的梯度会回传到 encoder，在 lambda_r 较大时导致 z 的排序方向反转（Q7 sweep 已证实）。

修改文件：model/ac_encoder.py

修改内容：

1. proxy_reconstructor(z_i) → proxy_reconstructor(z_i.detach())。

实际效果：

1. z_spearman_rho 保持 0.9892（未受影响）。
2. 成功消除了 lambda_r 增大时 z 方向反转的风险。
3. 副作用：reconstructor 有效学习率降至 lr × lambda_r = 0.0001，追不上 z_i 漂移，proxy_recon_r2 从 -18.68 恶化到 -29.86。此副作用将在阶段 E 中通过提高 lambda_r 解决。

---

## 阶段 C：复合 early stopping — ✅ 已完成

问题：early stopping 仅监控 val_task_loss，在 task_loss 触底后立即停止，但此时 kstar_mae 仍在改善中。

修改文件：experiments/run_synthetic.py

修改内容：

1. 新增 best_val_score 变量。
2. val_score = val_task_loss + val_kstar_mae。
3. best model 选择和 patience 都改为基于 val_score。

实际效果：

1. E1a best_epoch：50 → 80，总训练轮数：70 → 100。
2. E1c best_epoch：38 → 151，总训练轮数：58 → 171。
3. 模型现在能够在 task_loss 触底后继续等待 kstar_mae 改善。

---

## 阶段 D：移除有害 lag bias — ✅ 已完成

问题：lag_bias_strength=1.0 在 lag gate logits 上施加递增的负偏置，压制远端 lag（如 lag 7-10），导致 omega 集中在前几个 lag。

修改文件：config/cmdl_config.py

修改内容：

1. lag_bias_strength 默认值从 1.0 改为 0.0。
2. 合成数据域配置中 lag_bias_strength 也改为 0.0。

实际效果：

1. E1a omega_peak：从 lag 3 独占变为 lag 3/7/9 分布（与阶段 A 联合效果）。
2. E1c omega_peak：从 lag 3 占 200/200 变为 lag 1 (137), lag 7 (63)。
3. kstar_pred 范围扩展：E1a [4.8, 5.5] → [4.3, 7.3]，E1c [4.0, 4.2] → [3.0, 6.9]。

---

## 阶段 E：提高 lambda_r 修复 reconstructor 学习率 — ⏳ 待实施

对应问题：Q-NEW-1（P0）

问题诊断：

1. 阶段 B 的 detach 使 z_i 的分布完全由 task_loss 控制，每个 epoch 漂移。
2. reconstructor 的有效学习率 = lr × lambda_r = 0.001 × 0.1 = 0.0001，追不上漂移。
3. 表现为 recon_loss 单调上升（1.18 → 1.58），proxy_recon_r2 持续恶化。

安全性论证：

1. 在 detach 之前，lambda_r=1.0 会导致 z 方向反转（Q7 sweep 实证）。
2. 在 detach 之后，recon 梯度被完全截断，不可能影响 encoder。
3. 因此 lambda_r=1.0 现在只会加速 reconstructor 收敛，不会干扰 z。

修改文件：config/cmdl_config.py + notebooks/01_synthetic_verify.ipynb cell 3

修改内容：

1. lambda_r 默认值从 0.1 改为 1.0。
2. Notebook cell 3 配置中 lambda_r 也改为 1.0。

预期效果：

1. reconstructor 有效学习率提升 10 倍（0.0001 → 0.001），与主模型相同。
2. recon_loss 应转为下降趋势。
3. proxy_recon_r2 预期从 -29.86 提升到正值区间（阈值 > 0.5）。

验证方法：重跑 Notebook 01_synthetic_verify.ipynb，检查 E1b proxy_recon_r2 是否 > 0.5。

---

## 阶段 F：降低 temperature 增强 omega 尖锐度 — ⏳ 待实施（阶段 E 之后）

对应问题：Q-NEW-2（P1）

问题诊断：

1. E1c 的 k*(z) = round(10 × (1-z)²) 产生偏斜分布，大量实体的 k* 在 8-10 范围。
2. temperature=1.0 下 softmax 分布较平滑，omega 难以在单个 lag 上形成尖锐峰值。
3. omega_peak_accuracy=0.455 说明仍有 55% 的实体峰值位置不正确。

修改文件：config/cmdl_config.py + notebooks/01_synthetic_verify.ipynb cell 3

修改内容：

1. temperature 默认值从 1.0 改为 0.5。
2. Notebook cell 3 配置中 temperature 也改为 0.5。

预期效果：

1. softmax 输出更尖锐，omega 更集中在正确的 lag 位置。
2. E1c kstar_mae 预期从 1.34 降到 < 1.0。

注意事项：

1. 温度过低可能导致 softmax 输出接近 one-hot，梯度消失。
2. 0.5 是第一个测试点，如果效果不足或过度，可再调整为 0.3 或 0.7。
3. 阶段 F 应在阶段 E 验证通过后再实施，避免同时修改多个变量。

验证方法：重跑 Notebook 01_synthetic_verify.ipynb，检查 E1c kstar_mae 是否 < 1.0。

---

## 实施时间线

| 阶段 | 状态 | 对应问题 | 优先级 |
|---|---|---|---|
| A 消除 backbone 捷径 | ✅ 已完成 | 原 Q2, Q6 | P0 |
| B 解耦 reconstructor 梯度 | ✅ 已完成 | 原 Q3, Q7 | P0 |
| C 复合 early stopping | ✅ 已完成 | 原 Q4, Q5 | P0 |
| D 移除有害 lag bias | ✅ 已完成 | 原 Q2, Q6 | P0 |
| E 提高 lambda_r | ⏳ 待实施 | Q-NEW-1 | P0 |
| F 降低 temperature | ⏳ 待实施 | Q-NEW-2 | P1 |

---

## 风险与回退

1. 阶段 E 失败回退：如果 lambda_r=1.0 仍无法修复 proxy_recon_r2，可能需要为 reconstructor 使用独立优化器（独立学习率），而非通过总损失的权重隐式控制。
2. 阶段 F 失败回退：如果 temperature=0.5 导致梯度消失，可使用 temperature annealing（从 1.0 逐渐降到 0.3）替代固定值。
3. 如果 E1a kstar_mae 在后续修改中回退到 > 1.0，需要回滚到阶段 D 完成时的代码快照重新评估。
# CMDL Step 4 结构性修复计划

更新时间：2026-04-17

## 根因总结

当前 Step 4 失败源于三个耦合的结构缺陷：

1. **Backbone 捷径路径**：LSTM 输入拼接了 `current_sequence`，模型可绕过 `lag_context_sequence` 降低 task_loss，omega 收不到有效梯度。
2. **Reconstructor 梯度冲突**：z_i 被 task_loss 和 recon_loss 两条路径拉扯，lambda_r=0.1 时 reconstructor 追不上 z_i 变化，lambda_r=1.0 时 z_i 方向反转。
3. **Early stopping 选错模型**：只看 val_task_loss，选出的是捷径最优模型而非机制最优模型。
4. **Lag bias 方向对抗**：lag_bias_strength=1.0 惩罚远端 lag，但真实 k* 分布在 3-10 区间。

## 修改计划（4 阶段，约 9 行代码）

### 阶段 A：消除 Backbone 捷径路径

文件：model/backbone.py

修改 1：`input_fusion` 定义，5 * d_model → 4 * d_model
修改 2：`fused_inputs` 拼接列表中移除 `current_sequence`

### 阶段 B：解耦 Reconstructor 梯度

文件：model/ac_encoder.py

修改：`p_hat_i = self.proxy_reconstructor(z_i)` → `p_hat_i = self.proxy_reconstructor(z_i.detach())`

### 阶段 C：修正 Early Stopping 准则

文件：experiments/run_synthetic.py

修改：监控指标从 `val_task_loss` 改为 `val_task_loss + val_kstar_mae`

### 阶段 D：移除有害 Lag Bias

文件：config/cmdl_config.py

修改：`lag_bias_strength` 默认值从 1.0 → 0.0，synthetic preset 同步

## 执行顺序

A 和 B 互相独立可并行 → C 依赖 A → D 独立

## 验证

1. python -m pytest tests/
2. python experiments/run_synthetic.py --scenario all --epochs 200 --patience 20 --output-dir outputs/step4_fix
3. 检查：E1a kstar_mae < 1.0、E1b proxy_recon_r2 > 0.5、E1c kstar_mae < 1.0