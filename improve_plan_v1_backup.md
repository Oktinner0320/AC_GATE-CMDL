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