# 实验参数说明（Experiment Parameters Guide）

本文档说明三个精简版 notebook 共用的关键参数与阅读方式：

- `notebooks/01_synthetic_verify.ipynb`
- `notebooks/02_economics_results.ipynb`
- `notebooks/03_energy_results.ipynb`

这三个 notebook 现在遵循统一流程（shared workflow）：配置实验、按需运行训练、汇总产物（artifacts）、审计多种子稳定性（multi-seed stability），最后输出对比结果与可视化。

## 执行开关（Execution Switches）

| 参数（Parameter） | 含义（Meaning） | 使用方式（How to use） |
| --- | --- | --- |
| `RUN_CMDL` | 是否训练 CMDL / AC-GATE。 | 如果只读取已有产物，保持 `False`；如果要重新生成 CMDL 输出，设为 `True`。 |
| `RUN_BASELINE` | 是否训练匹配设置的 Plain LSTM 基线（baseline）。 | 应与 CMDL 使用相同的 `SEEDS`、数据契约（data contract）和数据切分（split）。 |
| `RUN_GROUPED_ARDL` | 是否运行确定性的 grouped-ARDL 基线。 | economics 与 energy notebook 中主要用于查看滞后趋势（lag trend）和做简单基线校准（baseline calibration）。 |
| `RUN_ABLATIONS` | 是否运行 no-AC、uniform-lag、no-reconstruction 三类消融（ablations）。 | 用于检验 AC-GATE 各组件是否真正改变了可解释机制（interpretable mechanism）。 |
| `ACTIVE_PLAN` | `outputs/notebook_*` 下的输出命名空间（output namespace）。 | 默认是 `complete_20seed`；若要查看更早的运行结果，可切换到如 `multiseed_audit` 等计划名。 |

这些开关默认都设为 `False`，因为完整 20 种子实验（complete 20-seed suite）开销较高。开关关闭时，notebook 只会读取 `OUTPUT_ROOT` 中已经存在的产物。

## 随机种子设计（Seed Design）

| 参数（Parameter） | 含义（Meaning） | 解释方式（Interpretation） |
| --- | --- | --- |
| `SEEDS = list(range(20))` | 所有随机模型计划使用的随机种子列表。 | 如果一个结论在这 20 个种子上都保持方向和量级稳定，那么这个结论更强。 |
| `n_seeds` | 在输出产物中实际找到的种子数。 | 在解释均值之前必须先看它。如果 `n_seeds < 20`，那么当前表格只是一份部分审计（partial audit）。 |
| `positive_seed_share` | 调整后机制指标（adjusted mechanism metric）为正的种子占比。 | `>= 2/3` 可视为实用上的正向阈值（practical candidate-positive threshold）；`1.0` 代表更强的一致性。 |

## 数据契约（Data Contracts）

| 领域（Domain） | 默认目标（Default target） | 主数据路径（Main data path） | 特征 / 代理契约（Feature / proxy contract） |
| --- | --- | --- | --- |
| Synthetic | generated target | generated in memory | 线性（linear）与非线性（nonlinear）两种场景，已知真实 `z`、`k*` 和 `omega`。 |
| Economics | `ctfp` | `data/economics/processed/economics_cleaned_long_v2.csv` | `effective_labor_aware`，以 effective labor 为主锚点（anchor），并辅以 employment 与 human-capital 代理变量。 |
| Energy | `co2_per_unit_energy` | `data/energy/raw/energy_wgi_merged.csv` | `minimal`，以 renewables share 作为序列输入（sequence input），并使用 WGI governance proxies。 |

## 训练参数（Training Parameters）

| 参数（Parameter） | 含义（Meaning） | 对比时的注意事项（Notes for comparison） |
| --- | --- | --- |
| `epochs` | 最大训练轮数（maximum training epochs）。 | synthetic 默认 `200`；真实数据 notebook 默认 `120`。如果提前停止（early stopping）触发，实际训练轮数会更少。 |
| `patience` | 提前停止耐心值（early-stopping patience）。 | 在同一领域内，CMDL 与 LSTM 应保持一致。 |
| `lr` | 学习率（learning rate）。 | 默认 `1e-3`。修改它会同时影响优化过程与结果可比性（comparability）。 |
| `grad_clip` | 梯度裁剪阈值（gradient clipping threshold）。 | 默认 `1.0`，用于提升训练稳定性。 |
| `grad_clip_mode` | 梯度裁剪作用的参数组（parameter groups）。 | economics 的完整契约使用 `split`；energy 使用 `global`。 |
| `device` | `auto`、`cpu` 或 `cuda`。 | 除非在调试复现性（reproducibility）或设备相关问题，否则建议使用 `auto`。 |

## AC-GATE 机制参数（AC-GATE Mechanism Parameters）

| 参数（Parameter） | 含义（Meaning） | 重点查看什么（What to inspect） |
| --- | --- | --- |
| `lambda_r` | 代理重构损失（proxy reconstruction loss）的权重。 | 将 CMDL 与 `no_recon_regularization` 对比；如果两者几乎相同，说明重构项不足以证明必要性（necessity）。 |
| `temperature` | 用于 lag weights 的 softmax / sparsemax 温度（temperature）。 | 值更低时，`omega` 更尖锐（sharper）；值更高时，`omega` 更平滑（smoother）。 |
| `omega_transform` | 滞后权重变换（lag-weight transform），通常是 `softmax`。 | `sparsemax` 能产生更稀疏（sparse）的滞后选择，但会改变结果可比性。 |
| `lag_bias_strength` | 滞后先验 / 偏置（lag prior / bias）的强度。 | 它会影响数据驱动适配（data-driven adaptation）发生前的初始滞后偏好。 |
| `lambda_omega_entropy` | 熵带约束违例惩罚（entropy-band violation penalty）。 | 用于防止 lag weights 完全均匀（fully uniform）或完全塌缩（fully collapsed）。 |
| `omega_entropy_min/max` | 允许的熵区间（allowed entropy band）。 | 需要结合 `omega_entropy_band_violation_share` 一起看。 |
| `lambda_z_anchor` | 将潜变量 `z` 直接对齐到锚点代理（anchor proxy）的可选惩罚项。 | 查看 `test_z_anchor_adjusted_rho` 与 `test_z_anchor_loss`。 |
| `z_anchor_target_sign` | `z-anchor` 惩罚所期待的方向（expected sign）。 | 应根据领域契约（domain contract）设为正或负。 |
| `recon_loss_mode` | 重构损失主要强调哪些代理（which proxies reconstruction emphasizes）。 | economics 的完整契约使用 `anchor_weighted`。 |
| `anchor_recon_weight` | 在 `anchor_weighted` 模式下，锚点代理（anchor proxy）的额外权重。 | 值越大，越优先重构锚点，而不是辅助代理（auxiliaries）。 |
| `reconstruction_detach` | 重构梯度是否从编码器路径（encoder path）分离。 | `False` 表示重构会反向影响 encoder；`True` 则更接近诊断式 / 事后式（diagnostic / post-hoc）使用。 |

## 核心指标（Core Metrics）

| 指标（Metric） | 含义（Meaning） | 如何解读（How to read it） |
| --- | --- | --- |
| `test_r2_mean` | 各种子上的平均预测 R2（mean forecast R2）。 | 这是预测证据（forecast evidence），不要把它与机制证据（mechanism evidence）混为一谈。 |
| `test_r2_std` | 各种子上的预测波动（forecast variation across seeds）。 | 如果 std 很高，说明预测排序（forecast ranking）不稳定。 |
| `anchor_adjusted_rho_mean` | 调整符号后的 `k*` 与 anchor 的 Spearman rho 均值。 | 为正说明学到的 `k*` 与预期锚点方向（expected anchor direction）一致。 |
| `anchor_positive_seed_share` | anchor-adjusted rho 为正的种子占比。 | 在很多情况下，它比单纯均值更有解释力。 |
| `mean_proxy_adjusted_rho_mean` | 与代理聚合量（proxy aggregate）的平均对齐程度。 | 当辅助代理（auxiliary proxies）带有信号、但 anchor 较弱时，这个指标尤其有用。 |
| `kstar_std_mean` | 学到的有效滞后（effective lag）在实体层面的平均变异。 | 接近 0 往往表示滞后行为塌缩（collapsed）或缺乏异质性（non-heterogeneous）。 |
| `entropy_mean` | `omega` 的平均熵（mean entropy）。 | 接近 `log(max_lag)` 说明滞后权重近似均匀；很低则说明接近单一滞后集中（single-lag concentration）。 |
| `lag_gate_sensitivity_range_mean` | lag gate 对潜变量 `z` 的敏感度（sensitivity）。 | 为正通常表示 lag gate 会响应实体级代理信息（entity-level proxy information）。 |
| `proxy_signal_r2_mean` | 代理重构 / 读出质量（proxy reconstruction/readout quality）。 | 强信号有助于解释 `z`，但它本身不是预测证据。 |

## 消融解释（Ablation Interpretation）

| 变体（Variant） | 如果 AC-GATE 机制确实重要，理论上应出现的现象（Expected behavior if AC-GATE mechanism matters） |
| --- | --- |
| `no_ac_encoder` | `kstar_std` 应明显塌缩，或机制指标（mechanism metrics）失效。 |
| `uniform_lag` | 熵（entropy）应接近 `log(max_lag)`，学习到的滞后异质性（lag heterogeneity）应基本消失。 |
| `no_recon_regularization` | 用来检验 reconstruction regularization 是否在任务损失（task loss）之外仍然是必要的。 |

一个相对干净的机制主张（clean mechanism claim）通常需要同时满足：CMDL 的 `k*` 不塌缩、调整后 rho 在多数种子上为正，而且退化型消融（degenerate ablations）确实失去机制结构。至于预测是否更优（forecast superiority），那是另一条独立主张。

## 每个 Notebook 重点看什么（Where To Look In Each Notebook）

1. 先看第一个代码单元（first code cell）中的设置表，确认 `SEEDS`、`ACTIVE_PLAN`、数据路径和各个运行开关。
2. 只有在你准备重新生成产物（regenerate artifacts）时，才运行可选训练单元（optional training cell）。
3. 先看 `compact_summary`，它给出预测与机制的核心均值摘要。
4. 再看 `result_log`，它给出紧凑的 yes / partial / no 审计结论。
5. 在真实数据 notebook 中，继续看 `per_proxy_audit_summary`，确认究竟是哪一个代理变量（proxy）在支持或削弱机制结论。
6. 在 economics / energy 中，再看 grouped-ARDL 的 lag trends，对比学习型滞后行为（learned lag behavior）与确定性滞后基线（deterministic lag baselines）。
7. 写报告或论文表格时，优先使用各自 `comparison/` 目录下保存的 CSV 结果。

## 实际主张强度（Practical Claim Strength）

| 证据模式（Evidence pattern） | 推荐表述（Suggested wording） |
| --- | --- |
| 预测 R2 更优，且机制指标稳定。 | 预测与机制同时为正的案例（forecast and mechanism positive case）。 |
| 预测较弱，但 adjusted rho 与消融保护（ablation guards）稳定。 | 机制对齐为正（mechanism-alignment positive case），但不能写成预测最优。 |
| 只有辅助代理稳定，anchor 结果混合。 | 部分机制证据（partial mechanism evidence）；必须明确写出边界。 |
| 指标高度依赖单个种子。 | 探索性案例（exploratory case）或边界案例（boundary case）；不要做强主张。 |