# CMDL Step 4 改进计划

更新时间：2026-04-18（第三版，达标收束版）

本文档记录 Step 4 合成实验从首次失败到最终达标的完整修复链。第三版不再把后续阶段写成“待实施”，而是明确区分：哪些阶段已经完成、哪些旧判断已被证据推翻、以及当前如果还要继续推进，合理的方向是什么。

---

## 总体策略

最终有效的修复链不是单纯调参，而是“结构修复 + 场景局部校正 + 评估前小头重拟合”：

- 阶段 A-D：修复 backbone 捷径、梯度耦合、early stopping 和 lag bias，先把机制路径拉正。
- 阶段 F：仅对非线性场景局部收紧 temperature，解决 E1c 的数值量级问题。
- 阶段 G：在 best checkpoint 选出后，对 detached 的 proxy_reconstructor 做闭式最小二乘重拟合，解决 E1b。

当前 formal_target 结果：

| 子实验 | 指标 1 | 当前值 | 阈值 | 指标 2 | 当前值 | 阈值 | 状态 |
|---|---|---:|---|---|---:|---|---|
| E1a linear | kstar_mae | **0.9229** | < 1.0 | kstar_spearman_rho | **0.9805** | > 0.8 | ✅ |
| E1b identification | proxy_recon_r2 | **0.9439** | > 0.5 | z_spearman_rho | **0.9892** | > 0.8 | ✅ |
| E1c nonlinear | kstar_mae | **0.5348** | < 1.0 | kstar_spearman_rho | **0.9622** | > 0.8 | ✅ |

---

## 阶段 A：消除 backbone 捷径路径 — ✅ 已完成

问题：backbone LSTM 的 input_fusion 拼接了 current_sequence，模型可以绕过 lag_context_sequence 直接获取时序信息，omega 的梯度信号被削弱。

修改文件：model/backbone.py

修改内容：

1. input_fusion 维度从 5×d_model 改为 4×d_model。
2. fused_inputs 的 torch.cat 中移除 current_sequence。
3. current_sequence 参数保留但仅用于有效时间段对齐。

实际效果：

1. E1a 的 k* 恢复从“排序对、数值塌缩”转向可用区间。
2. omega 不再集中在单一固定 lag 上，为后续阶段奠定基础。

---

## 阶段 B：解耦 reconstructor 梯度 — ✅ 已完成

问题：proxy_reconstructor(z_i) 的梯度会回传到 encoder，在较大 lambda_r 下导致 z 的排序方向翻转。

修改文件：model/ac_encoder.py

修改内容：

1. proxy_reconstructor(z_i) → proxy_reconstructor(z_i.detach())。

实际效果：

1. z_spearman_rho 始终保持在高位。
2. 成功消除了 recon 分支反向污染 encoder 的风险。
3. 同时也暴露出一个新问题：重构头变成 detached 的极小线性层后，单靠训练中的 Adam 更新很难稳定追上 z_i 的漂移。

---

## 阶段 C：复合 early stopping — ✅ 已完成

问题：只监控 val_task_loss 会过早停在 task_loss 最优点，而不是 k* 恢复最优点。

修改文件：experiments/run_synthetic.py

修改内容：

1. 新增 best_val_score 变量。
2. 使用 val_task_loss + val_kstar_mae 作为模型选择与 patience 依据。

实际效果：

1. E1a best_epoch 从 50 拉到 80。
2. E1c 的有效训练轮数显著增加，后期的 kstar_mae 继续下降并最终达标。

---

## 阶段 D：移除有害 lag bias — ✅ 已完成

问题：lag_bias_strength=1.0 在 lag gate logits 上施加递增负偏置，系统性压制远端 lag。

修改文件：config/cmdl_config.py

修改内容：

1. lag_bias_strength 默认值从 1.0 改为 0.0。
2. synthetic preset 同步改为 0.0。

实际效果：

1. E1a 与 E1c 的 omega 分布不再锁死在前几个 lag。
2. E1a 最终通过；E1c 的后续优化空间被缩小到“只剩尖锐度问题”。

---

## 阶段 E：提高 lambda_r — ✅ 已实施，但已确认不是根因修复

最初假设：detach 后 reconstructor 的有效学习率变成 lr × lambda_r，因此把 lambda_r 从 0.1 提高到 1.0 应能修复 E1b。

已实施修改：

1. config/cmdl_config.py 中 lambda_r 默认值同步为 1.0。
2. notebooks/01_synthetic_verify.ipynb 的共享参数也同步为 1.0。

修正后的结论：

1. 该修改是安全的，但单独实施并不能修复 E1b。
2. 实测中 proxy_recon_r2 仍停留在约 -30，说明“有效学习率 = lr × lambda_r”的线性推断在当前 Adam 优化器下并不成立。
3. 这一阶段保留为配置一致性修改，但不再视为 E1b 的主修复。

---

## 阶段 F：局部降低非线性 temperature — ✅ 已完成

问题：非线性场景的 k* 分布更偏斜，temperature=1.0 的 softmax 不够尖锐，导致 E1c 仍有量级误差。

修改位置：notebooks/01_synthetic_verify.ipynb

修改内容：

1. 保持 E1a / E1b 使用 temperature=1.0。
2. 仅在 E1c 调用时构造一份 temperature=0.5 的局部参数副本。

实际效果：

1. E1c kstar_mae：1.3424 → 0.5348。
2. E1c 达到 formal_target 阈值。
3. 由于修改是局部化的，E1a 没有被额外扰动。

---

## 阶段 G：best checkpoint 后闭式重拟合 proxy head — ✅ 已完成

问题：在 detach 已生效的前提下，proxy_reconstructor 变成一个独立的 1→3 线性头。Notebook 诊断显示：

1. z_pred → proxy 的最优线性 R2 约为 0.944。
2. 训练得到的 proxy_recon_r2 却长期为 -30 左右。

这说明：

1. 问题不在 z 表示本身。
2. 问题也不在重构头表达能力。
3. 问题在于 detached 小线性头的训练路径，而不是继续调损失权重。

修改文件：experiments/run_synthetic.py、tests/test_step3_model.py

修改内容：

1. 新增 `refit_proxy_reconstructor(model, panel)`。
2. 在 best checkpoint 加载后、最终 full-panel evaluate 之前，用训练切分上的 z_i 与 p_i 对 proxy_reconstructor 直接做最小二乘闭式求解。
3. 将 refit 后的权重写回 checkpoint，并增加单测验证 refit 会降低重构误差。

实际效果：

1. E1b proxy_recon_r2：-30.02 → 0.9439。
2. model_proxy_recon_r2 与 best_linear_proxy_r2_from_z_pred 基本重合，说明该修复精确击中了真实瓶颈。
3. E1a / E1c 的 k* 指标不受影响。

---

## Notebook 整理状态 — ✅ 已完成

为保证 notebook 与最新代码路径一致，已完成以下整理：

1. 恢复了路径初始化单元，确保 repo_root、pandas、matplotlib、json、display 都在 notebook 内显式定义。
2. 第 2 个代码单元加入 `importlib.reload`，避免 notebook kernel 缓存旧版 experiments.run_synthetic。
3. 主实验单元统一使用最新 formal_target 配置，并保留 E1c 的局部 temperature 覆盖。
4. Notebook 已完整重跑，summary、图表、diagnostics 和 debug sweep 当前都与最新 formal_target 结果一致。

---

## 实施时间线

| 阶段 | 状态 | 作用 |
|---|---|---|
| A | ✅ 已完成 | 修复 backbone 捷径 |
| B | ✅ 已完成 | 切断 recon 对 encoder 的干扰 |
| C | ✅ 已完成 | 修正模型选择与 early stopping |
| D | ✅ 已完成 | 去除有害 lag bias |
| E | ✅ 已实施，但非主修复 | 同步 lambda_r=1.0，事实证明安全但不足以单独解决 E1b |
| F | ✅ 已完成 | 仅对 E1c 降低 temperature，解决非线性量级问题 |
| G | ✅ 已完成 | 在最终评估前重拟合 proxy head，解决 E1b |

---

## 当前收束结论

1. Step 4 合成实验已经达标，当前计划中的主修复阶段全部完成。
2. 阶段 E 的原始判断已被实证修正：提高 lambda_r 不是 E1b 的根因级修复。
3. 阶段 G 是最终闭合 E1b 的关键步骤，且改动面最小，不扰动已经通过的 E1a/E1c。
4. 当前如果还要继续做方法学增强，合理方向不是继续救火，而是决定是否需要一个“严格端到端 learned reconstructor”的替代版本。

---

## 后续可选方向（非当前 blocker）

1. 如果论文或报告必须强调“reconstructor 也是训练阶段端到端学出的”，可新增一个独立 param group 或独立优化器的对照实验，作为阶段 G 的补充版本，而不是替代当前达标实现。
2. 若要扩展到真实数据域，可复用当前 refit 流程，但需要明确记录：proxy head 是在 best checkpoint 后重估得到的。
3. 如果后续主要目标转向解释性，可继续优化 E1a 的 omega_peak_accuracy，而不是继续追逐已经达标的 kstar_mae。
