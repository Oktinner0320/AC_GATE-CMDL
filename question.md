# CMDL 已知问题总表

更新时间：2026-04-17

本文档汇总截至当前已经被实验、日志、Notebook 诊断和源码检查共同确认的已知问题，目标是避免后续排障时重复走弯路。文中尽量区分三类信息：

1. 已经被证据直接支持的事实
2. 由事实推导出的高可信判断
3. 仍需源码级验证的疑似根因

---

## 一、当前总体现状

当前 Step 4 合成实验在 formal_target 配置下仍然未通过计划书阈值。

formal_target 当前结果如下：

| 子实验 | 关键指标 1 | 当前值 | 阈值 | 关键指标 2 | 当前值 | 阈值 | 是否通过 |
|---|---:|---:|---:|---:|---:|---:|---|
| E1a linear | kstar_mae | 1.8607 | < 1.0 | kstar_spearman_rho | 0.9804 | > 0.8 | 否 |
| E1b identification | proxy_recon_r2 | -18.6828 | > 0.5 | z_spearman_rho | 0.9883 | > 0.8 | 否 |
| E1c nonlinear | kstar_mae | 2.4281 | < 1.0 | kstar_spearman_rho | 0.9609 | > 0.8 | 否 |

从现有证据看，问题并不是“模型完全没学到东西”，而是至少同时存在两条独立失败链：

1. k* 的排序学到了，但数值幅度明显塌缩，导致 Spearman 很高而 MAE 很差。
2. z 的排序学到了，但 proxy 重构分支几乎没有把这个 z 用起来，导致 E1b 的 proxy_recon_r2 极差。

---

## 二、证据来源

本文件中的问题条目主要基于以下已验证来源：

1. outputs/notebook_step4/formal_target/step4_results.json
2. outputs/notebook_step4/formal_target/E1a_linear/summary.json
3. outputs/notebook_step4/formal_target/E1c_nonlinear/summary.json
4. outputs/notebook_step4/formal_target/E1a_linear/history.csv
5. outputs/notebook_step4/formal_target/E1c_nonlinear/history.csv
6. Notebook 中 formal_target 运行后的汇总表、热力图、散点图
7. Notebook 中的 Debug Diagnostics 单元
8. Notebook 中的 Debug Sweep 单元
9. experiments/run_synthetic.py、model/ac_encoder.py、model/lag_gate.py、model/cmdl_model.py、model/loss.py 的当前实现

---

## 三、已知问题清单

### Q1. Step 4 的 formal_target 仍未达到计划书阈值

状态：已确认

严重程度：P0

现象：

1. E1a 没有达到 kstar_mae < 1.0。
2. E1b 没有达到 proxy_recon_r2 > 0.5。
3. E1c 没有达到 kstar_mae < 1.0。

直接证据：

1. E1a linear：kstar_mae = 1.8607，kstar_spearman_rho = 0.9804。
2. E1b identification：proxy_recon_r2 = -18.6828，z_spearman_rho = 0.9883。
3. E1c nonlinear：kstar_mae = 2.4281，kstar_spearman_rho = 0.9609。

影响：

1. 当前结果不能作为 Step 4 的正式通过结果使用。
2. 后续若写报告或论文，必须明确说明当前实现尚未满足设计指标。

高可信判断：

1. 这不是单一超参轻微偏差，而是训练行为与目标之间存在结构性错位。

---

### Q2. k* 恢复出现“排序正确但数值塌缩”的失败模式

状态：已确认

严重程度：P0

现象：

1. E1a 和 E1c 的 kstar_spearman_rho 都很高，说明模型学到了大致排序。
2. 但 kstar_mae 仍然很大，说明模型没有学到正确的数值幅度。
3. 散点图显示预测 k* 被压在较窄区间，而不是覆盖真实的 1 到 10 范围。

直接证据：

1. E1a：kstar_spearman_rho = 0.9804，但 kstar_mae = 1.8607。
2. E1c：kstar_spearman_rho = 0.9609，但 kstar_mae = 2.4281。
3. E1a 散点图中，预测值大致集中在 4.8 到 5.5 左右，而真实值覆盖 3 到 10。
4. E1c 散点图中，预测值大致集中在 4.0 到 4.2 左右，而真实值覆盖 1 到 10。

热力图与峰值分布证据：

1. formal_target 线性场景中，omega_peak 的实体分布为 lag 3 占 170 个、lag 7 占 30 个。
2. formal_target 非线性场景中，omega_peak 的实体分布为 lag 3 占 200 个。
3. 线性场景 omega_peak_accuracy 只有 0.055。
4. 非线性场景 omega_peak_accuracy 只有 0.120。

影响：

1. 当前模型无法可靠恢复真实 lag 位置，只能给出“排序方向”而无法给出“正确 lag 值”。
2. E1a 与 E1c 的核心研究结论在当前实现下都不成立。

高可信判断：

1. 失败不是完全随机，因为排序相关性很高。
2. 失败主要表现为 lag gate 输出分布集中在少数几个固定峰值，导致期望滞后 k* 被压缩到中间区域。

疑似根因位置：

1. model/lag_gate.py 中的温度、相对位置偏置和 logits 形状。
2. experiments/run_synthetic.py 中当前只用 task_loss 做 early stopping 与模型选择。
3. CMDLModel 中 omega 只按实体级共享，不随时间步变化，可能降低数值恢复能力。

---

### Q3. E1b 的 proxy 重构分支严重失效，但这不是数据上限问题

状态：已确认

严重程度：P0

现象：

1. z 的排序几乎完好，但 proxy 重构 R2 极差。
2. 说明模型已经得到了有用的 z 表示，但 reconstructor 没有学会把这个 z 还原回 proxy 空间。

直接证据：

1. formal_target 下 z_spearman_rho = 0.9883。
2. formal_target 下 proxy_recon_r2 = -18.6828。
3. Notebook 诊断显示：如果直接用 z_true 对 proxy 做最优线性回归，R2 = 0.9392。
4. Notebook 诊断显示：如果直接用模型输出的 z_pred 对 proxy 做最优线性回归，R2 = 0.9487。
5. 但模型自己学到的 proxy_reconstructor 对同一 z_pred 只能得到 -18.6828。

影响：

1. E1b 当前失败的主因不是 z 无法识别，而是重构头或重构训练机制没有起作用。
2. 这意味着“提高 z 可识别性”不是当前首要矛盾；当前首要矛盾是“如何让重构头真的学到线性映射”。

高可信判断：

1. 数据生成器并没有把 E1b 设成不可能任务。
2. 单变量 z 到多变量 proxy 的线性映射本身是可行的。
3. 当前问题更像是优化路径、损失耦合、梯度分配或训练流程问题，而不是模型表达力上限问题。

疑似根因位置：

1. model/ac_encoder.py 中 proxy_reconstructor 的训练是否真正收到稳定梯度。
2. model/loss.py 中 recon_loss 的作用方式。
3. experiments/run_synthetic.py 中 task 与 recon 的共同优化是否让重构头始终跟不上表示层。

特别说明：

1. 这里不能再把问题简单解释成“lambda_r 太小”。
2. 因为后续 sweep 已验证，仅提高 lambda_r 并没有让 proxy_recon_r2 回到合理区间。

---

### Q4. formal_target 名义上是 200 epoch，但实际上被 early stopping 提前截断

状态：已确认

严重程度：P1

现象：

1. 虽然 formal_target 配置写的是 epochs = 200，patience = 20。
2. 但线性场景在第 70 轮提前停止，非线性场景在第 58 轮提前停止。

直接证据：

1. E1a_linear 的 best_epoch = 50，训练日志显示 early stopping at epoch 70。
2. E1c_nonlinear 的 best_epoch = 38，训练日志显示 early stopping at epoch 58。

影响：

1. formal_target 当前不是“完整训练 200 轮”的正式实验，而是“最多 200 轮、但按 val_task_loss 提前停止”的实验。
2. 如果后续要把 formal_target 解释成正式配置，需要明确它的真实停止条件。

高可信判断：

1. 这不一定是 bug，但它会误导实验解释。
2. 当前 formal_target 的失败不能简单归因于“训练轮数不够”，因为训练在现有准则下已经主动认为继续训练无收益。

疑似根因位置：

1. experiments/run_synthetic.py 中 early stopping 监控项只看 val_task_loss。

---

### Q5. 当前模型选择准则与研究目标错位

状态：已确认

严重程度：P0

现象：

1. val_task_loss 持续下降，但 kstar_mae 没有同步改善到目标区间。
2. proxy_recon_r2 在训练过程中一直为大幅负值，却没有阻止模型被选为最优。

直接证据：

1. E1a 的 val_task_loss 从 0.3727 下降到约 0.1029。
2. 但同一过程中 val_kstar_mae 只从 2.8633 改善到约 2.3661，仍远高于 1.0。
3. 同一过程中 val_proxy_recon_r2 始终为负，约在 -26.63 到 -19.94 区间波动。

影响：

1. 训练循环正在优化一个与 Step 4 验收标准并不一致的目标。
2. 这会导致“训练看起来稳定收敛，但实验始终不过线”的现象反复出现。

高可信判断：

1. 即使继续沿用当前 early stopping 与 best-model 选择方式，模型也很可能继续偏向较低 task_loss，而不是较好 k* 恢复或 proxy 重构。

疑似根因位置：

1. experiments/run_synthetic.py 中 best model 选择逻辑。
2. 现有训练日志只把 k* 和 proxy 结果当作观察指标，而不是模型选择指标。

---

### Q6. 非线性场景并没有显式暴露新的模式，而是重复了更严重的 lag 塌缩

状态：已确认

严重程度：P1

现象：

1. 非线性场景本应检验 MLP 对非线性 k*(z) 的拟合能力。
2. 但当前结果更像是线性场景失败模式的强化版，而不是一个新的、可区分的非线性拟合问题。

直接证据：

1. E1c kstar_mae = 2.4281，高于线性场景的 1.8607。
2. E1c 的 omega_peak 分布为 lag 3 占 200 个实体，没有任何明显分化。
3. E1c 散点图中预测值几乎压在 4.0 附近。

影响：

1. 目前无法判断 E1c 的失败究竟来自“非线性建模能力不足”，还是来自“lag gate 本身已经先失败”。
2. 在 E1a 尚未稳定通过之前，E1c 的解释价值较弱。

高可信判断：

1. 当前优先级不应先放在“增强非线性能力”，而应先解决线性场景下的 lag 恢复与重构分支问题。

---

### Q7. 简单调参不能解决根因，且部分调参会显著破坏 z 的方向稳定性

状态：已确认

严重程度：P1

已完成的最小 sweep：

| 配置 | lambda_r | temperature | lag_bias_strength | kstar_mae | kstar_spearman_rho | proxy_recon_r2 | omega_peak_accuracy | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_60 | 0.1 | 1.0 | 1.0 | 1.8607 | 0.9804 | -18.6828 | 0.055 | 基线失败 |
| sharp_gate | 0.1 | 0.3 | 0.1 | 1.6975 | 0.9805 | -18.7839 | 0.140 | k* 略有改善，但仍不过线 |
| recon_focus | 1.0 | 1.0 | 1.0 | 2.0383 | -0.9432 | -17.2341 | 0.055 | 排序方向反转，仍未修复 recon |
| combined | 1.0 | 0.3 | 0.1 | 3.1859 | -0.2135 | -17.9851 | 0.055 | 整体更差 |

关键结论：

1. 降低温度、减弱偏置可以让 kstar_mae 从 1.8607 改善到 1.6975，但仍远高于 1.0。
2. 单纯增大 lambda_r 不仅不能修复 proxy_recon_r2，还可能使 z 的排序方向发生翻转。
3. 因此，问题不是“还没找到一个幸运超参”，而是训练结构本身需要调整。

影响：

1. 不建议继续在当前训练逻辑上做盲目大范围网格搜索。
2. 优先级应从“调超参”转向“改训练代码和损失行为”。

---

### Q8. Notebook 的诊断区与主实验区并非自动同步，切换计划后诊断输出会过时

状态：已确认

严重程度：P2

现象：

1. 主实验单元切换到 formal_target 并重跑后，后面的诊断单元并不会自动刷新。
2. 如果不手动再运行诊断区，就会保留旧计划下的诊断输出。

影响：

1. 容易出现“主结果已经切到 formal_target，但诊断表还是旧值”的情况。
2. 这会误导后续对问题的判断，尤其是在对比 notebook_medium、formal_target 和 debug sweep 时。

高可信判断：

1. 这是 Notebook 组织层面的可用性问题，不是训练数值问题。
2. 但如果不记录，后续排障时很容易把旧诊断当成新结果。

建议处理方向：

1. 让诊断单元显式依赖当前 ACTIVE_PLAN。
2. 或在主实验单元末尾自动触发一次摘要刷新逻辑。

---

## 四、已被当前证据基本排除的解释

### E1b 失败不是因为数据生成器天生不可重构

证据：

1. best_linear_proxy_r2_from_z_true = 0.9392。
2. best_linear_proxy_r2_from_z_pred = 0.9487。

结论：

1. 数据生成器没有把 E1b 设成不可能任务。
2. 当前失败主要来自模型内部训练与重构分支行为，而不是数据上限。

### 当前 lag 输入相关性不是唯一主因

证据：

1. mean_abs_offdiag_lag_corr = 0.3085。

结论：

1. 不同 lag 之间存在一定相关性，但还不足以单独解释“所有实体几乎都压到 lag 3”的现象。
2. 说明模式塌缩更可能来自 gate 结构、偏置设置或训练目标错位，而不是纯数据不可辨识。

### MLflow 后端切换不是数值问题来源

证据：

1. 当前 tracking_backend 已稳定工作在 sqlite 本地后端。
2. 指标问题在 notebook_medium、formal_target 和 debug sweep 下都持续存在。

结论：

1. 训练失败与实验追踪后端无关。

---

## 五、受影响代码位置

以下位置与当前问题高度相关，应视为后续排障的重点区域：

1. experiments/run_synthetic.py
	- setup_experiment
	- train_one_epoch
	- evaluate
	- run_experiment
	- early stopping 与 best-model 选择逻辑

2. model/loss.py
	- DomainAgnosticLoss
	- task_loss 与 recon_loss 的组合方式

3. model/ac_encoder.py
	- AdaptiveACEncoder
	- proxy_reconstructor 的训练行为

4. model/lag_gate.py
	- ScaleInvariantLagGate
	- temperature 与 lag_bias_strength 对 logits 和 omega 的影响

5. model/cmdl_model.py
	- 实体级共享 omega 的使用方式
	- lag_context_sequence 的构造与聚合方式

6. notebooks/01_synthetic_verify.ipynb
	- 诊断区与主实验区的同步问题

---

## 六、建议的排障优先级

### 第一优先级

1. 修正训练循环的模型选择标准，不再只看 val_task_loss。
2. 直接诊断并修复 proxy_reconstructor 为什么在 z 已可用时仍学不出线性映射。

### 第二优先级

1. 调整 lag gate 的偏置、温度或目标耦合方式，解决固定峰值塌缩。
2. 让 k* 的数值恢复而不只是排序恢复。

### 第三优先级

1. 在 E1a 稳定通过后，再重新评价 E1c 的非线性拟合能力。
2. 最后再处理 Notebook 诊断自动同步等可用性问题。

---

## 七、当前最重要的结论

截至目前，最关键的已知问题只有两条，而且都已经有强证据支持：

1. lag gate 学到了排序，但没有学到正确的 lag 数值，导致 E1a 和 E1c 失败。
2. z 已经足够好，但 reconstructor 没有把它转成 proxy 重构能力，导致 E1b 失败。

换言之，当前瓶颈不在 Notebook，也不在 MLflow，也不在“还没跑够 epoch”，而在训练代码和损失行为本身。
