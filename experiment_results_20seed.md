# 20-Seed Experiment Results

本文件记录 `complete_20seed` 计划下的完整实验结果。结果来自 20 个 seeds (`0-19`) 的真实训练产物，不使用 notebook fallback 结果。

## 运行范围与完整性

训练入口为 [experiments/run_complete_20seed_suite.py](experiments/run_complete_20seed_suite.py)，产物根目录为 `outputs/notebook_*/complete_20seed/`。为缩短运行时间，实际执行时将 seeds 分为 `0-9` 与 `10-19` 两个分片并行运行；脚本按 `summary.json` 自动跳过已完成 run，支持断点续跑。

| Domain | Family | Expected summaries | Actual summaries |
| --- | --- | ---: | ---: |
| Synthetic | CMDL | 40 | 40 |
| Synthetic | Plain LSTM | 40 | 40 |
| Synthetic | Ablation | 120 | 120 |
| Economics | CMDL | 20 | 20 |
| Economics | Plain LSTM | 20 | 20 |
| Economics | Grouped ARDL | 20 | 20 |
| Economics | Ablation | 60 | 60 |
| Energy | CMDL | 20 | 20 |
| Energy | Plain LSTM | 20 | 20 |
| Energy | Grouped ARDL | 20 | 20 |
| Energy | Ablation | 60 | 60 |
| Total | All | 440 | 440 |

Notebook 汇总已重新生成：synthetic comparison rows = 200，economics comparison rows = 120，energy comparison rows = 120。

## 关键结论

1. Synthetic 域给出了稳定的机制验证：CMDL 在 linear 与 nonlinear 两个场景下均显著优于 Plain LSTM 的 post-hoc lag recovery，且 AC encoder / learned lag gate 的消融边界清晰。
2. Economics 域只能支持“局部机制迹象”，不能作为强机制证明：CMDL 有非退化 lag heterogeneity，但 anchor-adjusted direction 在 20 seeds 上不稳定，forecast 也不优于 Plain LSTM / Uniform Lag。
3. Energy 域不支持当前设定下的 AC-GATE 机制证明：CMDL 只略优于 matched Plain LSTM，但被 Grouped ARDL 大幅超过；proxy alignment 接近 0 且只有 45% seeds 为正。
4. No Recon Regularization 在 synthetic 与 energy 中几乎复现 CMDL，在 economics 中甚至比完整 CMDL 的 proxy alignment 更正向。这说明当前 reconstruction regularization 不是主要机制来源，也可能在真实域里与目标机制方向存在张力。

## Synthetic

主要文件：

- [outputs/notebook_synthetic/complete_20seed/comparison/synthetic_multiseed_summary.csv](outputs/notebook_synthetic/complete_20seed/comparison/synthetic_multiseed_summary.csv)
- [outputs/notebook_synthetic/complete_20seed/comparison/synthetic_result_log.csv](outputs/notebook_synthetic/complete_20seed/comparison/synthetic_result_log.csv)
- [outputs/notebook_synthetic/complete_20seed/comparison_plots/synthetic_multiseed_summary.png](outputs/notebook_synthetic/complete_20seed/comparison_plots/synthetic_multiseed_summary.png)

| Scenario | Model | Seeds | Task loss mean | k* MAE mean | k* rho mean | Proxy R2 mean | z rho mean | Positive seed share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | No AC Encoder | 20 | 0.063 | 1.932 | 0.000 | -0.001 | 0.000 | 0.000 |
| linear | No Recon Regularization | 20 | 0.037 | 1.173 | 0.935 | 0.888 | 0.944 | 1.000 |
| linear | Uniform Lag | 20 | 0.069 | 1.913 | 0.000 | 0.843 | 0.938 | 0.000 |
| linear | CMDL | 20 | 0.036 | 1.163 | 0.935 | 0.889 | 0.945 | 1.000 |
| linear | Plain LSTM | 20 | 0.069 | 1.707 | 0.355 |  |  | 1.000 |
| nonlinear | No AC Encoder | 20 | 0.066 | 2.435 | 0.000 | -0.001 | 0.000 | 0.000 |
| nonlinear | No Recon Regularization | 20 | 0.038 | 1.463 | 0.910 | 0.891 | 0.942 | 1.000 |
| nonlinear | Uniform Lag | 20 | 0.094 | 3.102 | 0.000 | 0.838 | 0.928 | 0.000 |
| nonlinear | CMDL | 20 | 0.038 | 1.463 | 0.910 | 0.892 | 0.942 | 1.000 |
| nonlinear | Plain LSTM | 20 | 0.093 | 2.612 | 0.342 |  |  | 1.000 |

Synthetic 结论：CMDL 的 k* MAE 明显低于 Plain LSTM，rank alignment 约为 `0.91-0.94`，20/20 seeds 都保持正向。No AC Encoder 与 Uniform Lag 的 k* rho 都退化为 0，说明机制恢复依赖 AC encoder 与 learned lag gate。No Recon Regularization 与 CMDL 几乎一致，说明在 synthetic 设定中 reconstruction regularization 不是必要条件。

## Economics

主要文件：

- [outputs/notebook_economics/complete_20seed/comparison/economics_compact_summary.csv](outputs/notebook_economics/complete_20seed/comparison/economics_compact_summary.csv)
- [outputs/notebook_economics/complete_20seed/comparison/economics_result_log.csv](outputs/notebook_economics/complete_20seed/comparison/economics_result_log.csv)
- [outputs/notebook_economics/complete_20seed/comparison/economics_per_proxy_audit_summary.csv](outputs/notebook_economics/complete_20seed/comparison/economics_per_proxy_audit_summary.csv)
- [outputs/notebook_economics/complete_20seed/comparison_plots/economics_multiseed_summary.png](outputs/notebook_economics/complete_20seed/comparison_plots/economics_multiseed_summary.png)

| Model | Seeds | Test R2 mean | Test R2 std | Anchor adjusted rho | Anchor positive share | Mean proxy adjusted rho | Mean proxy positive share | k* std | Lag sensitivity | Proxy R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No AC Encoder | 20 | 0.058 | 0.029 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| No Recon Regularization | 20 | 0.054 | 0.034 | 0.158 | 0.600 | 0.317 | 0.700 | 0.114 | 0.348 | 0.219 |
| Uniform Lag | 20 | 0.104 | 0.026 |  |  |  |  | 0.000 | 0.549 | 0.326 |
| CMDL | 20 | 0.053 | 0.037 | -0.090 | 0.400 | -0.033 | 0.400 | 0.170 | 0.508 | 0.300 |
| Grouped ARDL | 20 | -0.089 | 0.000 |  |  |  |  |  |  |  |
| Plain LSTM | 20 | 0.101 | 0.023 | -0.029 | 0.250 | -0.125 | 0.050 | 1.898 |  |  |

Per-proxy audit:

| Proxy | Expected sign | Adjusted rho mean | Positive seed share | Status |
| --- | ---: | ---: | ---: | --- |
| effective_labor_anchor | -1.000 | -0.090 | 0.400 | mixed_or_negative |
| employment_level | -1.000 | -0.083 | 0.400 | mixed_or_negative |
| hc_level | -1.000 | 0.125 | 0.700 | candidate_positive |
| hc_trend | -1.000 | -0.044 | 0.500 | mixed_or_negative |

Economics 结论：CMDL forecast R2 为 `0.053`，低于 Plain LSTM (`0.101`) 和 Uniform Lag (`0.104`)。相对简单基线，CMDL 高于 panel OLS 与 Grouped ARDL，但低于 persistence baseline。机制上，CMDL 的 lag gate 非退化（k* std `0.170`，lag sensitivity `0.508`，proxy R2 `0.300`），但 anchor adjusted rho 平均为 `-0.090`，正向 seed share 只有 `0.40`。只有 `hc_level` 是 candidate_positive。因此 economics 域可作为机制探索和局部证据，不能作为强机制证明。

## Energy

主要文件：

- [outputs/notebook_energy/complete_20seed/comparison/energy_compact_summary.csv](outputs/notebook_energy/complete_20seed/comparison/energy_compact_summary.csv)
- [outputs/notebook_energy/complete_20seed/comparison/energy_result_log.csv](outputs/notebook_energy/complete_20seed/comparison/energy_result_log.csv)
- [outputs/notebook_energy/complete_20seed/comparison/energy_per_proxy_audit_summary.csv](outputs/notebook_energy/complete_20seed/comparison/energy_per_proxy_audit_summary.csv)
- [outputs/notebook_energy/complete_20seed/comparison_plots/energy_multiseed_summary.png](outputs/notebook_energy/complete_20seed/comparison_plots/energy_multiseed_summary.png)

| Model | Seeds | Test R2 mean | Test R2 std | Anchor adjusted rho | Anchor positive share | Mean proxy adjusted rho | Mean proxy positive share | k* std | Lag sensitivity | Proxy R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No AC Encoder | 20 | -0.027 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| No Recon Regularization | 20 | -0.028 | 0.005 | 0.001 | 0.450 | 0.001 | 0.450 | 0.110 | 0.274 | 0.699 |
| Uniform Lag | 20 | -0.030 | 0.004 |  |  |  |  | 0.000 | 0.427 | 0.750 |
| CMDL | 20 | -0.028 | 0.005 | 0.001 | 0.450 | 0.001 | 0.450 | 0.110 | 0.274 | 0.699 |
| Grouped ARDL | 20 | 0.607 | 0.000 |  |  |  |  |  |  |  |
| Plain LSTM | 20 | -0.029 | 0.004 | 0.107 | 0.800 | 0.103 | 0.750 | 1.327 |  |  |

Per-proxy audit:

| Proxy | Expected sign | Adjusted rho mean | Positive seed share | Status |
| --- | ---: | ---: | ---: | --- |
| government_effectiveness | -1.000 | 0.001 | 0.450 | mixed_or_negative |
| regulatory_quality | -1.000 | 0.016 | 0.450 | mixed_or_negative |
| rule_of_law | -1.000 | -0.012 | 0.450 | mixed_or_negative |

Energy 结论：CMDL forecast R2 为 `-0.028`，仅略高于 matched Plain LSTM (`-0.029`)，但明显低于 Grouped ARDL (`0.607`)。机制上，CMDL 有非退化 lag gate（k* std `0.110`，lag sensitivity `0.274`，proxy R2 `0.699`），但 proxy direction 平均接近 0，positive seed share 只有 `0.45`，三项 proxy 全部为 mixed_or_negative。当前 energy 设定不能证明机制，说明这个数据域更适合用作反例/压力测试，而不是主证明域。

## Overall Assessment

当前 20-seed 结果支持如下判断：

- 机制主张应以 synthetic 为强证据来源：这里有 ground truth，CMDL 的 lag recovery、rank alignment、AC encoder 消融和 uniform lag 消融都稳定成立。
- economics 可以作为真实域审计案例：存在 heterogeneity 和部分 proxy 方向信号，但 seed 稳定性不足，不能写成“真实域证明”。
- energy 当前不应作为机制证明域：Grouped ARDL 预测显著更强，AC-GATE proxy alignment 也不稳。
- 后续如果要强化真实域证据，优先方向是重新审计 economics proxy bundle 与 anchor sign、单独测试 reconstruction loss 的作用，并在 energy 中重新设计目标/处理变量或加入更强的领域约束。