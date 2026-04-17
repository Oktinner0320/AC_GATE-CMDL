# Step 4 — 合成数据实验 E1 实验计划书

> 基于已完成的 Step 1–3（合成数据生成器、AC 编码器 + Lag Gate、LSTM Backbone + 端到端模型），
> 在有 ground truth 的合成面板数据上验证 AC-Gate 机制的正确性与可解释性。

---

## 一、实验设计总览

| 子实验 | 目标 | 输入场景 | 评估量 | 合格线 |
|---|---|---|---|---|
| **E1a 机制验证** | 模型能否恢复已知的 $k^*(z)$ 映射 | linear: $k^*=\text{round}(3+7(1-z))$ | k\* MAE, Spearman($k^*_{\text{pred}}$, $k^*_{\text{true}}$) | MAE < 1.0, $\rho_s > 0.8$ |
| **E1b $z_i$ 识别性** | 学到的 $z_i$ 是否保留 proxy 语义 | 同 E1a | proxy 重构 R², Spearman($z_i$, $z_{\text{true}}$) | R² > 0.5, $\rho_s > 0.8$ |
| **E1c 非线性场景** | MLP 能否拟合非线性映射 | nonlinear: $k^*=\text{round}(10(1-z)^2)$ | 同 E1a | 同 E1a |

**训练配置**：

| 参数 | 值 | 说明 |
|---|---|---|
| 优化器 | Adam | — |
| 学习率 | 1e-3 | loss 不降时可降至 3e-4 |
| 最大 epoch | 200 | — |
| Early stopping | patience=20, monitor=val_task_loss | 仅看 task_loss，不看 total_loss |
| Batch size | 全量（N=200） | 数据量小，无需 DataLoader 分批 |
| 梯度裁剪 | max_norm=1.0 | 预防 LSTM 梯度爆炸 |

---

## 二、文件级实现方案

### 2.1 `evaluation/metrics.py` — 通用评估指标

| 函数 | 签名 | 职责 |
|---|---|---|
| `compute_mse` | `(pred, true) → float` | 预测 MSE |
| `compute_mae` | `(pred, true) → float` | 预测 MAE |
| `compute_r2` | `(pred, true) → float` | R²，包装 `sklearn.metrics.r2_score` |
| `compute_spearman` | `(pred, true) → (rho, p_value)` | 包装 `scipy.stats.spearmanr` |

所有函数接受 numpy array 或 torch.Tensor（内部统一转 numpy）。

### 2.2 `evaluation/kstar_eval.py` — k\* 专项评估

| 函数 | 签名 | 职责 |
|---|---|---|
| `evaluate_kstar` | `(k_pred, k_true) → dict` | 返回 `{mae, spearman_rho, spearman_p, rmse}` |
| `evaluate_z_identification` | `(z_pred, z_true, p_pred, p_true) → dict` | 返回 `{z_spearman_rho, z_spearman_p, proxy_recon_r2}` |
| `evaluate_omega_distribution` | `(omega, kstar_true) → dict` | 返回 `{entropy_mean, peak_accuracy}` |

`evaluate_omega_distribution` 是额外的**诊断指标**：
- `entropy_mean`：ω 分布的平均熵。过高（≈ log K ≈ 2.3）说明均匀未分化，过低（≈ 0）说明过度集中
- `peak_accuracy`：`argmax(ω) + 1` 与 `round(k*)` 一致的实体比例

### 2.3 `experiments/run_synthetic.py` — 训练循环主脚本

**整体结构**：

```
1. parse_args()         — 命令行参数: scenario, seed, lr, epochs, patience, lambda_r, output_dir
2. setup_experiment()   — 生成数据, 创建模型/optimizer/loss, 初始化 MLflow
3. train_one_epoch()    — 前向 → 损失 → 反向 → 梯度裁剪 → 更新, 返回 loss 分项
4. evaluate()           — 模型 eval 模式, 跑全量数据, 返回所有指标 dict
5. run_e1a()            — linear 场景完整训练 + 评估
6. run_e1b()            — 复用 E1a 训练好的模型, 额外做 z_i 识别性评估
7. run_e1c()            — nonlinear 场景, 结构同 E1a
8. main()               — 串联 E1a → E1b → E1c, 汇总结果表
```

**关键设计决策**：

- **全量 batch 训练**：N=200, T=30 → 仅 6000 个样本点，直接 `model(entity_ids, X_it, p_i, s_i)` 全量前向
- **验证集划分**：按实体随机 80/20 split（160 训练 / 40 验证），而非按时间切（合成数据无时间趋势概念）
- **Early stopping**：监控验证集 task_loss，patience=20，保存最优权重（`torch.save` state_dict）
- **MLflow 记录**：每 epoch 记录 `train_loss`, `val_loss`, `kstar_mae`, `proxy_recon_r2`；训练结束记录最终指标

**模型调用方式**（基于已实现的 `CMDLModel.forward` 签名）：

```python
output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i)
# DomainAgnosticLoss 自动处理 warmup 对齐: Y_it[B,30] → 裁掉前 max_lag=10 步
```

### 2.4 `visualization/omega_heatmap.py`

函数 `plot_omega_heatmap(omega, z_values, kstar_true, save_path)`：
- 按 $z$ 值排序实体（y 轴），x 轴 = lag 1\~K
- `sns.heatmap(omega_sorted, cmap="YlOrRd", xticklabels=range(1, K+1))`
- 叠加 `kstar_true` 标记点（每行一个圆点标示真实 k\*）
- 预期视觉效果：左上角（高 z / 低 lag）和右下角（低 z / 高 lag）呈斜对角浓色带

### 2.5 `visualization/kstar_distribution.py`

函数 `plot_kstar_scatter(k_pred, k_true, save_path)`：
- 散点图 predicted k\* vs true k\*，叠加 y=x 对角线
- 标注 MAE 和 Spearman ρ 文字
- 可选：按 $z_{\text{true}}$ 分位数着色（四色）

---

## 三、训练流程依赖链

```
generate_cmdl_synthetic(cfg)  →  SyntheticPanel
         ↓
  80/20 entity split  →  train_ids / val_ids
         ↓
  CMDLModel(cfg)  +  DomainAgnosticLoss(λ_r, warmup=max_lag)
         ↓
  for epoch in range(200):
      train_one_epoch(model, panel, train_ids)
      val_metrics = evaluate(model, panel, val_ids)
      early_stopping.check(val_metrics["task_loss"])
         ↓
  final_metrics = evaluate(model, panel, all_ids)    ← E1a 指标
  z_identification = evaluate_z(model, panel)         ← E1b 指标
         ↓
  plot_omega_heatmap / plot_kstar_scatter              ← 图表
```

---

## 四、潜在问题与对策

### 问题 1：ω 分布坍缩为均匀分布（k\* ≈ 5.5 对所有实体）

**症状**：训练后所有实体的 ω ≈ 1/K，k\* 无差异，Spearman ≈ 0。

**原因分析**：
- 温度 τ 过高 → softmax 输出趋于均匀
- 位置偏置强度过大 → 梯度无法克服偏置，所有实体被钉在同一 lag
- $z_i$ 信号太弱 → Lag Gate 收到的条件信号无信息量

**对策**：
1. 降低温度：从默认 τ=1.0 尝试 {0.5, 0.3, 0.1}，强制 softmax 输出更尖锐
2. 减弱位置偏置：`lag_bias_strength` 从 1.0 降到 {0.5, 0.1, 0.0}
3. 监控诊断指标：每 20 epoch 打印 `entropy_mean(ω)`，若持续 ≈ log(10) ≈ 2.3 说明未分化
4. 检查 AC Encoder 梯度：`model.ac_encoder.latent_head.weight.grad.norm()` — 若接近 0 说明梯度断链

### 问题 2：ω 分布过度集中（所有权重在单一 lag，无平滑过渡）

**症状**：omega 呈 one-hot 形式，argmax 正确但 k\* 的连续值意义丧失。

**原因分析**：
- 温度 τ 过低 → softmax 近似 argmax
- 训练后期过拟合导致分布退化

**对策**：
1. 提高温度：τ ∈ {1.0, 2.0}
2. 可选加入 ω 熵正则：$L += \lambda_{\text{ent}} \cdot H(\omega)$，鼓励适度分散；λ\_ent 应很小（\~0.01），避免干扰 k\* 学习
3. 观察验证集指标是否同步退化——若验证集 k\* MAE 仍好，过度集中可能不是实质问题

### 问题 3：z\_i 与 z\_true 相关性低（E1b 不通过）

**症状**：proxy 重构 R² 高（>0.5），但 z\_i vs z\_true Spearman 低。

**原因分析**：
- $z_i$ 学到了 proxy 的某个非线性变换，保留了重构能力但丢失了与 $z_{\text{true}}$ 的单调关系
- AC Encoder 无符号约束，可能学到 $z_i \approx -z_{\text{true}}$

**对策**：
1. 画 `z_i` vs `z_true` 散点图：若呈单调但非线性曲线，Spearman 仍应高；若呈 U 型则说明编码器学反了
2. 取 `|spearman|` 或在评估时自动检测符号并翻转：若 $\rho_s < -0.5$，则将 $z_i$ 取负后重新计算
3. 增加 proxy 重构损失权重：$\lambda_r$ 从 0.1 升到 {0.2, 0.5}，强迫 z\_i 保留更多 proxy 语义

### 问题 4：Loss 不下降或 NaN

**症状**：前几个 epoch loss 不降或出现 NaN。

**原因分析**：
- 学习率过高（lr=1e-3 对小数据可能偏大）
- 数值溢出：softmax 前 logits 绝对值过大
- 梯度爆炸：LSTM 未正确初始化

**对策**：
1. 降低学习率：尝试 {3e-4, 1e-4}
2. 梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
3. 检查 logits 范围：训练时打印 `lag_gate_output.logits.abs().max()`，若 > 50 说明 ScalarGRN 输出过大
4. NaN 哨兵：在 loss 计算后加 `if torch.isnan(loss): break` 并保存当时的输入/输出用于调试

### 问题 5：非线性场景 E1c 效果显著差于 E1a

**症状**：linear 场景 k\* MAE < 1.0，nonlinear 场景 MAE > 2.0。

**原因分析**：
- $k^*(z)=10(1-z)^2$ 映射更陡峭，z 接近 1 时 k\* 变化很快
- AC Encoder → Lag Gate 的 MLP 表达力不足
- kstar\_true 分布偏向低 lag 端：`clip(round(10*(1-z)^2), 1, 10)` 在 z > 0.68 时全部 clip 到 k\*=1

**对策**：
1. 增加 ScalarGRN 隐藏维度：`hidden_dim` 从 16 升到 {32, 64}
2. 增加训练 epoch 和 patience：非线性场景可能需要更多迭代收敛，patience 从 20 升到 40
3. 分析误差分布：按 z\_true 分 4 分位数画每组的 k\* MAE，定位误差集中区
4. 在论文中注明 clipping 边界效应：高 z 区域 k\*=1 的扁平区可能影响学习

### 问题 6：Early stopping 过早触发

**症状**：训练不到 50 epoch 就停止，k\* MAE 仍高。

**原因分析**：
- 验证集 task\_loss 抖动大（N=40 实体样本量小）
- 模型在 proxy 重构上快速收敛但 task loss 还在下降

**对策**：
1. 监控 task\_loss 而非 total\_loss：避免被 recon\_loss 快速下降误导
2. 增大 patience：从 20 升到 {30, 50}
3. 对验证 loss 做 5-epoch 滑动平均后再判断是否 improve
4. 备选方案：合成数据无泛化需求，可全量训练后直接评估（需在论文中说明）

### 问题 7：MLflow 配置/端口冲突

**症状**：`mlflow.start_run()` 抛出连接错误。

**对策**：
1. 使用本地文件后端：`mlflow.set_tracking_uri("file:///c:/DevSpace/PyDevspace/CMDL/mlruns")`
2. 降级方案：若 MLflow 完全不可用，用 `json.dump` 保存指标到 `outputs/` 目录，不阻塞实验进度

---

## 五、超参搜索策略

Step 4 不做大范围搜索，仅在 E1a 不通过时**逐项排查**：

| 优先级 | 参数 | 默认值 | 搜索范围 | 触发条件 |
|---|---|---|---|---|
| 1 | `temperature` | 1.0 | {0.1, 0.3, 0.5, 1.0, 2.0} | ω 坍缩为均匀 或 过度集中 |
| 2 | `lag_bias_strength` | 1.0 | {0.0, 0.1, 0.5, 1.0} | ω 全部偏向 lag=1 |
| 3 | `lr` | 1e-3 | {3e-4, 1e-3, 3e-3} | loss 抖动或不降 |
| 4 | `lambda_r` | 0.1 | {0.01, 0.05, 0.1, 0.2} | E1b z\_i 识别性不佳 |

---

## 六、验证清单

- [ ] E1a: k\* MAE < 1.0 且 Spearman ρ\_s > 0.8（linear 场景）
- [ ] E1b: proxy 重构 R² > 0.5 且 z\_i—z\_true Spearman > 0.8
- [ ] E1c: k\* MAE < 1.0（nonlinear 场景）
- [ ] ω 热力图显示清晰的斜对角模式（高 z → 低 lag 权重集中，低 z → 高 lag）
- [ ] k\* 散点图点集紧贴 y=x 对角线
- [ ] 训练 loss 收敛曲线无异常（无 NaN、无长期平台期）
- [ ] MLflow 或 JSON 文件可复现查看所有指标

---

## 七、实现文件清单

| 文件 | 操作 | 依赖 |
|---|---|---|
| `evaluation/metrics.py` | **新建实现** | scipy, sklearn |
| `evaluation/kstar_eval.py` | **新建实现** | 依赖 metrics.py |
| `experiments/run_synthetic.py` | **新建实现** | model/\*, data/\*, evaluation/\*, visualization/\* |
| `visualization/omega_heatmap.py` | **新建实现** | seaborn, matplotlib |
| `visualization/kstar_distribution.py` | **新建实现** | matplotlib |
| `config/cmdl_config.py` | 不修改 | — |
| `model/cmdl_model.py` | 不修改 | — |
| `model/loss.py` | 不修改（`warmup_steps=max_lag` 已支持） | — |
| `model/lag_gate.py` | 可能微调 `temperature` / `lag_bias_strength` 默认值 | — |
| `data/synthetic/generate.py` | 不修改 | — |