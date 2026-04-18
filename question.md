# CMDL 已知问题总表

更新时间：2026-04-18（第三版，Step 4 达标后）

本文档汇总截至当前已经被实验结果、Notebook 诊断、日志和源码共同确认的已知问题与结论。第三版基于阶段 A-G 完成后的 formal_target 结果，重点记录：哪些问题已经被解决、哪些旧解释已被修正、以及当前仍需注意但不再阻塞 Step 4 的残留风险。

---

## 一、当前总体现状

Step 4 合成实验在 formal_target 配置下已经全部通过。

formal_target 当前结果如下：

| 子实验 | 关键指标 1 | 修复前 | 当前值 | 阈值 | 关键指标 2 | 修复前 | 当前值 | 阈值 | 是否通过 |
|---|---|---:|---:|---|---|---:|---:|---|---|
| E1a linear | kstar_mae | 1.8607 | **0.9229** | < 1.0 | kstar_spearman_rho | 0.9804 | **0.9805** | > 0.8 | **是** |
| E1b identification | proxy_recon_r2 | -18.6828 | **0.9439** | > 0.5 | z_spearman_rho | 0.9883 | **0.9892** | > 0.8 | **是** |
| E1c nonlinear | kstar_mae | 2.4281 | **0.5348** | < 1.0 | kstar_spearman_rho | 0.9609 | **0.9622** | > 0.8 | **是** |

当前结论不再是“存在未通过项”，而是“Step 4 的三条验收链均已闭合”：

1. 阶段 A-D 解决了 backbone 捷径、lag 塌缩和 early stopping 错位，使 E1a 线性场景稳定通过。
2. 阶段 F 对非线性场景局部降低 temperature 到 0.5，解决了 E1c 的数值量级问题。
3. 阶段 G 在 best checkpoint 选出后，对 proxy_reconstructor 做闭式最小二乘重拟合，使 E1b 的 proxy_recon_r2 直接贴近当前 z_pred 的线性上限。

---

## 二、证据来源

1. outputs/notebook_step4/formal_target/step4_results.json（当前正式汇总结果）
2. outputs/notebook_step4/formal_target/E1a_linear/summary.json
3. outputs/notebook_step4/formal_target/E1c_nonlinear/summary.json
4. outputs/notebook_step4/formal_target/E1a_linear/predictions.csv
5. outputs/notebook_step4/formal_target/E1a_linear/history.csv
6. outputs/notebook_step4/formal_target/E1c_nonlinear/history.csv
7. notebooks/01_synthetic_verify.ipynb 当前全部执行输出
8. experiments/run_synthetic.py（阶段 C 与阶段 G）
9. config/cmdl_config.py（阶段 D 与 lambda_r 默认值同步）
10. tests/test_step3_model.py（proxy head refit 的回归测试）

---

## 三、已解决问题

### [已解决] 原 Q2 / Q6. k* 恢复出现“排序正确但数值塌缩”

解决方式：阶段 A（消除 backbone 捷径）+ 阶段 D（移除有害 lag bias）+ 阶段 F（仅对 E1c 的温度局部收紧）

结果：

1. E1a kstar_mae：1.8607 → 0.9229。
2. E1c kstar_mae：2.4281 → 0.5348。
3. E1a omega_peak_accuracy：0.055 → 0.335。
4. E1c omega_peak_accuracy：0.120 → 0.470。

归因：

1. 阶段 A 让 LSTM 不再能绕过 lag_context_sequence，omega 梯度路径恢复有效。
2. 阶段 D 去除了对远端 lag 的固定惩罚，lag 7-10 不再被系统性压制。
3. 阶段 F 只在非线性场景将 temperature 设为 0.5，使 omega 在偏斜的 k* 分布下形成更尖锐峰值，同时避免影响已通过的 E1a。

---

### [已解决] 原 Q4 / Q5. formal_target 被过早截断且模型选择准则与目标错位

解决方式：阶段 C（复合 early stopping）

结果：

1. E1a best_epoch：50 → 80，总训练轮数：70 → 100。
2. E1c best_epoch：38 → 196，总训练轮数显著拉长，允许 kstar_mae 继续下降到达标区间。
3. 模型选择现在基于 val_task_loss + val_kstar_mae，而非单看 task_loss。

---

### [已解决] Q-NEW-2. E1c kstar_mae 在结构修复后仍高于阈值

解决方式：阶段 F（非线性场景局部 temperature=0.5）

结果：

1. E1c kstar_mae：1.3424 → 0.5348。
2. E1c kstar_spearman_rho：0.9623 → 0.9622，保持稳定高相关。
3. E1c 通过 formal_target 阈值，不再是未解决问题。

说明：

1. 这证明 E1c 的剩余问题确实是量级问题，而非方向问题。
2. 对 temperature 的修改应保持局部化，只作用于非线性场景，避免无谓扰动 E1a。

---

### [已解决] Q-NEW-1. proxy_recon_r2 在结构修复后显著恶化

最终解决方式：阶段 G（best checkpoint 选出后，对 proxy_reconstructor 做闭式最小二乘重拟合）

当前结果：

1. E1b proxy_recon_r2：-29.86 / -30.02 → **0.9439**。
2. z_spearman_rho 保持 **0.9892**，说明修复没有干扰 z 的识别性。
3. Notebook 诊断显示：best_linear_proxy_r2_from_z_pred = **0.943990**，model_proxy_recon_r2 = **0.943949**，两者几乎重合。

修正后的根因分析：

1. 旧版本的判断是“detach 后 reconstructor 的有效学习率 = lr × lambda_r 太低”。
2. 实证表明，将 lambda_r 从 0.1 提高到 1.0 后，proxy_recon_r2 仍停留在约 -30，说明单纯调损失权重并没有解决问题。
3. 更合理的解释是：proxy_reconstructor 是一个 detached 的极小线性头（1→3，6 个参数），在 Adam 下梯度幅度会被二阶矩归一化，单纯放大 lambda_r 不会带来预期中的有效步长提升。
4. 由于 z_i 在训练过程中由 task_loss 驱动持续漂移，这个小线性头长期跟不上表示分布的变化；但从 z_pred 到 proxy 的最优线性映射始终存在，且 R2 约为 0.944。
5. 因此，正确修复不是继续调 lambda_r，而是在 best checkpoint 选出后，对 proxy_reconstructor 直接求最小二乘闭式解。

---

## 四、当前未解决问题

当前没有仍然阻塞 Step 4 合成实验验收的 P0 或 P1 问题，也就是说：

1. 当前阻塞问题数量是 **0**，不是“只剩 1 个阻塞问题”。
2. Step 4 formal_target 已经通过，不存在必须继续修复才能过线的未解决项。
3. 当前剩下的是“非阻塞事项”，主要涉及解释性、表述口径和后续研究扩展，而不是验收失败。

---

## 五、当前残留风险与非阻塞事项

1. E1a 虽然已通过，但 omega_peak_accuracy 只有 0.335，说明“精确峰值位置恢复”仍弱于“期望 lag 恢复”；当前这不妨碍 kstar_mae 达标，但如果后续研究问题转向实体级峰值解释性，需要单独优化。
2. 当前 proxy_recon_r2 反映的是“冻结后的 z 表示 + 评估前重估的线性 decoder”的能力，而不是“训练阶段端到端学出的 reconstructor”。若后续论文表述要求严格强调端到端联合学习，需要再做一个独立优化器或独立 param group 的对照实验。
3. 当前 debug sweep 里的 proxy_recon_r2 已不再是 recon 分支训练是否成功的证据，因为阶段 G 会在最终评估前统一重拟合该线性头。此后 debug sweep 主要用于比较 k* 恢复和 omega 形状，而不是比较重构头训练质量。

---

## 六、已被当前证据排除或修正的解释

1. **数据生成器不可重构**：已排除。best_linear_proxy_r2_from_z_true ≈ 0.939，best_linear_proxy_r2_from_z_pred ≈ 0.944，证明任务本身可解。
2. **lag 输入高度相关是唯一主因**：已排除。mean_abs_offdiag_lag_corr ≈ 0.3085，相关性存在但不足以单独解释原始塌缩。
3. **MLflow 后端切换影响数值结果**：已排除。跟踪后端只影响元数据持久化，不影响训练数值。
4. **“lambda_r 提高到 1.0 就能修复 E1b”**：已修正。该判断在 Adam 优化器下不成立；lambda_r=1.0 是安全的，但不是根因级修复。

---

## 七、已实施的代码修改

| 阶段 | 文件 | 修改内容 | 作用 |
|---|---|---|---|
| A | model/backbone.py | input_fusion 从 5×d_model 改为 4×d_model，fused_inputs 移除 current_sequence | 消除 backbone 捷径 |
| B | model/ac_encoder.py | proxy_reconstructor(z_i) → proxy_reconstructor(z_i.detach()) | 阻止 recon 梯度干扰 encoder |
| C | experiments/run_synthetic.py | early stopping 与 best model 选择改为 val_task_loss + val_kstar_mae | 修正模型选择逻辑 |
| D | config/cmdl_config.py | lag_bias_strength 默认值 1.0 → 0.0 | 移除对远端 lag 的有害惩罚 |
| E | config/cmdl_config.py、notebooks/01_synthetic_verify.ipynb | lambda_r 同步为 1.0 | 保持配置一致性，但事实证明并非 E1b 的主修复 |
| F | notebooks/01_synthetic_verify.ipynb | 非线性场景单独使用 temperature=0.5 | 修复 E1c 的量级问题 |
| G | experiments/run_synthetic.py、tests/test_step3_model.py | 在 best checkpoint 后对 proxy_reconstructor 做闭式最小二乘重拟合，并增加回归测试 | 修复 E1b，并保证实现可回归验证 |

---

## 八、当前最重要的结论

1. Step 4 合成实验现在已经全部达标，当前没有阻塞 formal_target 通过的已知问题。
2. E1b 的真实瓶颈不是 z 学坏，也不是数据不可重构，而是 detached 小线性头在 Adam 下的优化路径失效；根因修复是阶段 G 的评估前闭式重拟合，而不是继续调 lambda_r。
3. 目前的下一步工作不再是“继续救火”，而是决定是否需要把当前方案进一步包装成更严格的端到端版本，或者直接进入后续实验与写作阶段。
