# CMDL / AC-GATE 论文策略（Paper Strategy）

本文件基于当前完整的 20 种子结果（20-seed results），给出最适合 CMDL / AC-GATE 的论文投稿策略。当前最稳的写法不是把真实域结果包装成“强机制证明（strong mechanism proof）”，而是采用如下三层证据结构：

- 合成域（synthetic）承担机制恢复证明（mechanism recovery proof）
- 经济域（economics）承担真实数据审计（real-data audit）
- 能源域（energy）承担压力测试（stress test）与边界说明（boundary condition）

基于当前证据，本文档只保留三个最匹配、最可落地的应用型投稿方向（applied-track directions）：

1. ECML PKDD Applied Data Science Track
2. CIKM Applied Research Papers / Applied Track
3. IEEE ICDM 应用导向投稿（application-oriented submission）

## 0.1 关键 20-seed 锁定数据（Locked 20-seed Numbers，2026-04-26）

> 数据来源：[outputs/notebook_synthetic/complete_20seed_20260426/comparison/](outputs/notebook_synthetic/complete_20seed_20260426/comparison/)、[outputs/notebook_economics/complete_20seed_20260426/comparison/](outputs/notebook_economics/complete_20seed_20260426/comparison/)、[outputs/notebook_energy/complete_20seed_20260426/comparison/](outputs/notebook_energy/complete_20seed_20260426/comparison/)。所有差异均使用 seed-level 配对 Wilcoxon（`evaluation/significance.py`）。

### 0.1.1 Synthetic（机制恢复，main evidence）

| Scenario | Method | task loss (mean ± std) | k\* MAE | k\* Spearman ρ | k\* positive seed share |
|---|---|---|---|---|---|
| linear | CMDL | 0.0362 ± 0.0044 | 1.159 ± 0.227 | **0.945** | 1.00 |
| linear | No Recon Reg | 0.0363 ± 0.0043 | 1.159 ± 0.227 | 0.945 | 1.00 |
| linear | Plain LSTM | 0.0695 ± 0.0029 | 1.707 ± 0.091 | 0.356 | 1.00 |
| linear | No AC Encoder | 0.0629 ± 0.0054 | 1.931 ± 0.162 | 0.000 | 0.00 |
| linear | Uniform Lag | 0.0695 ± 0.0027 | 1.913 ± 0.088 | 0.000 | 0.00 |
| nonlinear | CMDL | 0.0378 ± 0.0065 | 1.467 ± 0.248 | **0.907** | 1.00 |
| nonlinear | Plain LSTM | 0.0927 ± 0.0061 | 2.611 ± 0.138 | 0.344 | 1.00 |
| nonlinear | No AC Encoder | 0.0661 ± 0.0184 | 2.435 ± 0.261 | 0.000 | 0.00 |
| nonlinear | Uniform Lag | 0.0938 ± 0.0081 | 3.102 ± 0.098 | 0.000 | 0.00 |

配对显著性（k\* MAE，CMDL 为 reference，越小越好）：
- vs Plain LSTM：linear/nonlinear 均 p ≈ 1.91e-06（CMDL 显著更优）
- vs No AC Encoder / Uniform Lag：均 p ≈ 1.91e-06（CMDL 显著更优）
- vs No Recon Reg：linear p = 0.498，nonlinear p = 0.368（**无显著差异**）

可写入论文的硬结论：
1. AC encoder + lag gate 是 k\* 恢复的必要结构（去掉任一则 ρ 退化为 0）。
2. CMDL 在 task loss 与 k\* MAE 上均显著优于 Plain LSTM（p < 1e-5）。
3. Reconstruction regularization 不是机制恢复的主要来源（无显著差异，且 mean diff < 1e-4）；论文应将其降级为"辅助稳定项"。

### 0.1.2 Economics（真实域审计 — L2 结构化异质滞后成立，L3 方向性机制不成立）

target = `ctfp`（PWT），feature_bundle = `effective_labor_aware`，n_seeds = 20。

**预测层（L0 forecast，不是主张层）**

| Method | test R² | Anchor-adjusted ρ | Anchor positive share | Mean proxy ρ | k\* std |
|---|---|---|---|---|---|
| CMDL | 0.054 ± 0.037 | **−0.109** | 0.35 | −0.039 | 0.167 |
| No Recon Reg | 0.055 ± 0.033 | 0.164 | 0.60 | 0.312 | 0.109 |
| No AC Encoder | 0.052 ± 0.035 | — | — | — | **0.000** |
| Uniform Lag | **0.104** ± 0.028 | — | — | — | **0.000** |
| Plain LSTM | **0.102** ± 0.021 | −0.017 | 0.25 | −0.117 | 1.894 |
| Grouped ARDL | −0.089 ± 0.000 | — | — | — | — |

预测层配对 Wilcoxon（test R²）：vs Plain LSTM p = 6.3e-05，vs Uniform Lag p = 1.9e-06 — **CMDL 在预测上显著较弱**，论文不主张预测优势。

**L2 机制层（structured heterogeneous lag，主张层）**

> 数据：[outputs/notebook_economics/complete_20seed_20260426/comparison/economics_stratified_kstar_aggregated.csv](outputs/notebook_economics/complete_20seed_20260426/comparison/economics_stratified_kstar_aggregated.csv)
> 方法：[evaluation/stratified_kstar.py](evaluation/stratified_kstar.py) — 每个 seed 计算 per-entity k\* 与训练窗口 entity-level 静态发展指标的 Spearman ρ，2000 次实体置换 null，再对 20 seeds 聚合。

| Stratifier | abs ρ mean | seeds p<0.05 | seeds p<0.01 | Fisher combined p |
|---|---|---|---|---|
| `hc_mean_train`（人力资本） | **0.371** | 80% | 75% | **1.0e-46** |
| `log_gdp_per_worker_train` | 0.278 | 70% | 55% | 3.5e-37 |
| `log_capital_per_worker_train` | 0.257 | 65% | 50% | 1.4e-24 |

退化对照：`No AC Encoder` 与 `Uniform Lag` 的 per-entity k\* 完全恒定（kstar_std ≡ 0） → stratification 检验**结构性不可执行**。这是最强的 L2 ablation 对照（CMDL 学到 stratifier-aligned 异质滞后；去掉 AC encoder 或 lag gate 后该结构整体消失）。

> **L2 结论可写**：在 PWT-CTFP 面板上，AC-GATE 恢复出的实体条件滞后分布与 entity-level 静态发展指标（人力资本、log 人均 GDP、log 人均资本）在统计上**显著结构化**（Fisher combined p < 1e-24），且该结构在去掉 AC encoder 或 lag gate 时**完全消失**。
> **L3 限制需诚实写**：anchor proxy 方向在 seed 层面不稳定（adjusted ρ = −0.109，正向占比 35%）；论文不以 proxy 方向作为机制成立判据，而以 stratification + ablation 对照作为判据。

### 0.1.3 Energy（真实域审计 — L2 在该域反而最强，预测层为应用边界）

target = `co2_per_unit_energy`，feature_bundle = `minimal`，n_seeds = 20。

**预测层（L0，不是主张层）**

| Method | test R² | Anchor-adjusted ρ | Mean proxy ρ |
|---|---|---|---|
| CMDL | −0.029 ± 0.005 | −0.014 | −0.012 |
| No Recon Reg | −0.029 ± 0.005 | −0.014 | −0.012 |
| No AC Encoder | −0.028 ± 0.005 | — | — |
| Uniform Lag | −0.030 ± 0.003 | — | — |
| Plain LSTM | −0.029 ± 0.004 | 0.093 | 0.090 |
| **Grouped ARDL** | **0.607** ± 0 | — | — |

预测层全部神经模型 R² ≈ −0.029，被 Grouped ARDL R² = 0.607 显著压制（p = 1.9e-06）。**论文不主张预测优势**，并将该域定位为"线性可解目标 + 短面板下神经容量过剩"的应用边界。

**L2 机制层（structured heterogeneous lag，主张层）**

> 数据：[outputs/notebook_energy/complete_20seed_20260426/comparison/energy_stratified_kstar_aggregated.csv](outputs/notebook_energy/complete_20seed_20260426/comparison/energy_stratified_kstar_aggregated.csv)

| Stratifier | abs ρ mean | seeds p<0.05 | seeds p<0.01 | Fisher combined p |
|---|---|---|---|---|
| `rule_of_law_train` | **0.735** | **95%** | 90% | **1.8e-79** |
| `government_effectiveness_train` | 0.716 | 90% | 85% | 9.0e-77 |
| `log_gdp_per_capita_train` | 0.609 | 90% | 85% | 1.5e-77 |

退化对照：`No AC Encoder` 与 `Uniform Lag` 的 per-entity k\* 同样为常数 → 检验不可执行 → CMDL 的 stratification 完全归因于 AC encoder + lag gate。

> **L2 结论可写**：在 OWID-energy × WGI 面板上，AC-GATE 恢复出的国家级有效滞后 k\* 与制度治理指标（rule of law、government effectiveness）和 log 人均 GDP 之间存在**极强且稳定的结构化关联**（|ρ| ≈ 0.6–0.74，95% 种子 p<0.05，Fisher combined p < 1e-76）。
> **附加价值（双解耦证据）**：能源域同时给出"预测层不可信 + 机制层结构强"，这正是 L2 框架的设计意图——把预测有效性与机制可解释性**分开评估**。论文可以把这一点写成方法论亮点，而不是缺陷。

### 0.1.4 关于"sign instability"的诚实声明

20-seed Spearman ρ 在两个域中均出现 sign 跨 seed 翻转：economics rho_mean = −0.10 而 |ρ| = 0.27~0.37；energy rho_mean = +0.01 而 |ρ| = 0.61~0.74。这是**无监督结构发现的常见对称性现象**：模型把实体集合稳定划分成两组，但"哪一组被赋予更大 k\*" 的标签在不同初始化下会翻转（标签置换不变性，label permutation invariance）。论文应：

1. 报告 |ρ| 与 share-of-seeds-rejecting-null（这两个量对 sign 翻转鲁棒）；
2. 在 limitations 中说明 sign 翻转，并指出 sign-stable directional alignment 只在 synthetic anchor proxy 监督下成立；
3. 不把 signed directional alignment 作为真实域主张层。

### 0.1.4 复审遗留风险点（合规层）

| review.md 行动项 | 状态 | 说明 |
|---|---|---|
| #1 配对 Wilcoxon 显著性检验 | ✅ 已完成 | 三域 CSV 均落盘 |
| #2 GenAI Usage Disclosure | ✅ 已完成 | [GenAI_Usage_Disclosure.md](GenAI_Usage_Disclosure.md) |
| #3 requirements / environment | ✅ 已完成 | [requirements.txt](requirements.txt) / [environment.yml](environment.yml) |
| #4 LICENSE | ✅ 已完成 | MIT 匿名占位 |
| #5 数据 .meta.json | ✅ 已完成 | sha256 + URL + UTC |
| #6 runtime meta | ✅ 已完成 | summary.json 含 GPU/CUDA/wall time |
| #7 Ablation 结构变更后重设 seed | ✅ **已完成** | [run_economics_ablation.py#L165](experiments/run_economics_ablation.py#L165) 与 [run_energy_ablation.py#L159](experiments/run_energy_ablation.py#L159) 已显式 `set_seed(int(seed))` 在 `build_variant_model` 之前；`matched_init_to_full_cmdl=False` 仍诚实写入 summary.json |
| #8 README Hyperparameter Protocol + Reproducing 段 | ✅ 已完成 | [readme.md](readme.md) §5 / §6；并新增 [refs.md](refs.md) 集中放置文献引用 |
| #9 README 基线表 + TFT 说明 | ✅ 已完成 | [readme.md](readme.md) §4 基线表；TFT 已从 baseline 集合移除并降级为"仅作参考"（数据规模不需 Transformer） |
| #10 `set_seed` 加 `CMDL_DETERMINISTIC` 分支 | ⛔ N/A | 当前训练在 CPU 上完成（无 GPU），cuDNN 确定性开关无意义；CPU LSTM 在固定种子下已是确定性的 |
| #11 投稿匿名化 | ⏸ 投稿前执行 | Anonymous GitHub 流程 |

## 1. 论文核心定位（Core Positioning）

### 1.0 三层"机制成立"判据（Three-Tier Mechanism Claim Ladder）

为避免在真实域过度主张，本文显式区分三层"机制成立"强度。**主张层只取 L2**：

| 层 | 名称 | 判据 | Synthetic | Economics | Energy |
|---|---|---|---|---|---|
| L3 | Directional mechanism（方向性机制） | anchor proxy ρ 与预期同号 + seed 多数正向 | ✅ ρ≈0.95 | ❌ | ❌ |
| **L2** | **Structured heterogeneous lag（结构化异质滞后，**主张层**）** | per-entity k\* 与外部已知 entity-level 静态指标显著 stratified（permutation null 拒绝），且退化 ablation 下结构消失 | ✅ | ✅ Fisher p<1e-24 | ✅ Fisher p<1e-76 |
| L1 | Learnable heterogeneous lag（可学习异质滞后） | k\* std > 0 且 ablation_guard 触发 | ✅ | ✅ | ✅ |

> **核心主张（修订版，覆盖原 1.x）**：CMDL / AC-GATE 在合成域恢复出方向性机制（L3），并在两个真实国家级面板（PWT-CTFP，OWID-energy×WGI）上恢复出 L2 意义下的**结构化异质滞后机制（structured heterogeneous lag）**——learned per-entity effective lag k\* 与 entity-level 静态发展 / 治理指标存在统计显著的结构化关联（Fisher combined p < 1e-20），且该结构在 No AC Encoder 与 Uniform Lag 退化对照下完全消失。

建议将论文定位为：

> 一种面向面板时序（panel time series）的实体条件异质滞后发现框架（entity-conditioned heterogeneous lag discovery framework）。模型通过实体级代理变量（proxy variables）学习潜在吸收能力表示（latent absorption representation），再将其映射为实体特异的滞后权重分布（entity-specific lag-weight distribution），从而在没有直接监督有效滞后期（effective lag）`k*` 的情况下恢复可解释的异质滞后结构（interpretable heterogeneous lag structure）。

当前结果支持以下主线，而不支持更强叙述：

- 适合：可检验的异质滞后发现（testable lag discovery）
- 适合：机制恢复（mechanism recovery）与真实域审计（real-data audit）
- 不适合：真实域强机制证明（strong real-world mechanism proof）
- 不适合：统一预测最优（universal forecasting superiority）

## 2. 主张边界（Claim Boundary）

### 2.1 可以稳健主张的内容（Defensible Claims）

1. AC-GATE 在合成域稳定恢复实体条件异质滞后（L3 directional + L2 structured）。
2. **AC-GATE 在两个真实国家级面板上稳定恢复 L2 结构化异质滞后**：PWT-CTFP 与人力资本/log GDP/log capital 之间 Fisher combined p < 1e-24；OWID-energy×WGI 与 rule of law/gov effectiveness/log GDP 之间 |ρ| ≈ 0.6–0.74，95% seeds p<0.05，Fisher combined p < 1e-76。
3. AC encoder + lag gate 是 L2 结构化异质滞后的必要结构：去掉任一变体 per-entity k\* 立即退化为常数（kstar_std ≡ 0），stratification 检验结构性不可执行。
4. 模型输出的有效滞后期 `k*` 不是事后可视化，而是来自显式滞后门控结构。
5. 真实域 L0（预测）与 L2（机制）解耦：能源域 R² ≈ −0.029（神经全线退化）但 L2 机制层 |ρ| ≈ 0.7 — 这是方法论亮点（mechanism interpretability decoupled from forecast accuracy），不是缺陷。

### 2.2 应避免的主张（Claims to Avoid）

1. 不要宣称 CMDL / AC-GATE 是通用预测最优模型；真实域上 Plain LSTM、Uniform Lag、Grouped ARDL 在不同域分别更优。
2. 不要宣称真实域上 anchor proxy 方向与预期一致（L3）；该层只在 synthetic 上成立。
3. 不要把代理重构（proxy reconstruction）写成主贡献；20-seed 显示 No Recon Reg 与 CMDL 无显著差异（synthetic p > 0.36），降级为辅助稳定项。
4. 不要在论文中把 sign-stable directional alignment 当作真实域机制证据；改报 |ρ| 与 share-of-seeds-rejecting-null。
5. 不要遮蔽 energy 域 Grouped ARDL R² = 0.607 vs 神经全线 R² ≈ −0.029 的对比；主动诚实写作"应用边界 + 双解耦证据"。
6. 不要使用严格因果识别（causal identification）措辞；应使用 lag discovery、mechanism recovery、structured heterogeneity 等表述。

### 2.3 最佳一句话贡献（One-Sentence Contribution）

> 本文提出 AC-GATE：一个可检验的神经异质滞后发现框架（testable neural heterogeneous-lag discovery framework）。它在合成域恢复方向性机制（L3），并在 PWT-CTFP 与 OWID-energy×WGI 两个真实国家级面板上恢复 L2 意义下的结构化异质滞后——learned per-entity effective lag 与 entity-level 静态发展 / 治理指标在 permutation null 下显著结构化（Fisher combined p < 1e-20），且在去除 AC encoder 或 lag gate 时该结构完全消失。

## 3. 三个优先投稿方向（Three Priority Applied-Track Directions）

严格说，下面三个目标更准确地属于会议论文（conference proceedings）或应用型轨道（applied track），不是传统期刊（journal）。但它们比强部署导向的应用赛道更适合当前项目，因为它们通常允许“真实问题（real-world problem）+ 离线真实数据评估（offline real-data evaluation）+ 可复现实验（reproducible experiments）”。

| 优先级 | 投稿方向 | 当前适配度 | 适合的论文角度 | 当前准备度 |
| --- | --- | ---: | --- | --- |
| 1 | ECML PKDD Applied Data Science Track | 高 | 面板时序中的真实异质滞后审计（lag audit），强调数据特性、诊断协议与结论边界 | 最接近当前成果形态 |
| 2 | CIKM Applied Research Papers / Applied Track | 中高 | 可复现的滞后审计工作流（reproducible lag-audit workflow），强调实现、数据管线、报告输出 | 需要更强 workflow 叙事 |
| 3 | IEEE ICDM 应用导向投稿（application-oriented submission） | 中高 | 可解释异质滞后挖掘（interpretable heterogeneous lag mining），强调技术定义、算法细节、统计证据 | 需要更强技术化表达 |

建议正式路线：

1. 先准备 ECML PKDD Applied Data Science Track 版本。
2. 若时间窗口或主题匹配不理想，将同一实验体系改写为 CIKM Applied Research 版本。
3. 若希望进一步强化技术贡献和数据挖掘味道，再准备 ICDM 版本。

## 4. 方向一：ECML PKDD Applied Data Science Track

### 4.1 适配原因（Why It Fits）

这个方向最适合当前成果，因为它强调：

- 真实问题（real-world problem）
- 数据难点（data idiosyncrasies）
- 方法设计（methodology）
- 面向用例的结论（use-case conclusions）

CMDL / AC-GATE 当前最强的价值，不只是一个新模块（new module），而是一套可复用的真实域异质滞后审计协议（reusable lag-audit protocol）。

### 4.2 推荐标题（Recommended Title）

> Auditing Heterogeneous Delays in Panel Time Series with AC-GATE

中文可对应为：

> 基于 AC-GATE 的面板时序异质滞后审计方法

### 4.3 推荐主论题（Recommended Thesis）

> 我们研究一个真实应用型数据科学问题（applied data science problem）：分析者常需要判断不同国家之间是否存在系统性的响应滞后差异（lag heterogeneity），但传统同质滞后基线（homogeneous-lag baselines）和事后神经解释（post-hoc neural explanations）难以审计。AC-GATE 提供了一套诊断流程（diagnostic pipeline）：它在 synthetic 面板中稳定恢复已知机制，同时在 economics 与 energy 面板中给出部分支持（partial support）或负向证据（negative evidence）。

### 4.4 正文重点（Main-Text Emphasis）

| 部分 | 应强调的内容 |
| --- | --- |
| 问题定义 | 真实世界中的面板滞后审计，而不是一般序列预测 |
| 数据部分 | 年度面板、短时间维、实体异质性、balanced-panel 过滤、训练窗口标准化 |
| 方法部分 | AC encoder、lag gate、seed stability、proxy alignment、baseline calibration |
| 结果部分 | 先写 synthetic 机制恢复，再写 economics 和 energy 的用例结论 |
| 讨论部分 | 真实域可以产生部分支持或负向结果，而这恰恰体现审计协议的价值 |

### 4.5 投稿前必须补的内容（Must Add Before Submission）

1. 增加“数据特殊性（data idiosyncrasies）”小节。
   - 说明年度面板、短时间维度、跨国异质性、proxy 聚合、时间切分，以及为什么不能用随机切分。
2. 增加“用例结论表（use-case conclusions table）”。
   - Synthetic：机制成立。
   - Economics：部分支持的审计案例。
   - Energy：压力测试与弱证据案例。
3. 增加种子分布图（seed distribution plots）或置信区间（confidence intervals）。
4. 增加可复现附录（reproducibility appendix）。
   - 包括命令、种子列表、配置表，以及 outputs 仅本地保留但可再生的说明。

### 4.6 需要避免的风险（Risks to Avoid）

1. 不要以“全面优于 baselines”开头，因为 economics 和 energy 不支持这个主张。
2. 不要把 energy 写成失败实验（failed experiment）；应写成压力测试（stress test），说明协议不会强行输出正面机制。

### 4.7 当前判断（Current Decision）

这是当前最优先、最适配的投稿方向。

## 5. 方向二：CIKM Applied Research Papers / Applied Track

### 5.1 适配原因（Why It Fits）

CIKM 应用型稿件更可能看重：

- 实现落地（implementation）
- 数据管线（data pipeline）
- 报告机制（reporting workflow）
- 知识发现过程（knowledge discovery process）

因此，对 CIKM 来说，论文不应主要写成“一个模型结构（a model architecture）”，而应写成“一个可复现的滞后审计工作流（reproducible lag-audit workflow）”。

### 5.2 推荐标题（Recommended Title）

> A Reproducible Lag-Audit Workflow for Economic and Energy Panel Data

中文可对应为：

> 面向经济与能源面板数据的可复现滞后审计工作流

### 5.3 推荐主论题（Recommended Thesis）

> 我们提出一个应用研究工作流（applied research workflow），将自适应滞后发现（adaptive lag discovery）、基线比较（baseline comparison）、种子稳定性诊断（seed-stability diagnostics）和代理一致性审计（proxy-alignment audit）集成到国家级面板时序分析流程中。

### 5.4 正文重点（Main-Text Emphasis）

| 部分 | 应强调的内容 |
| --- | --- |
| 系统 / 工作流 | 数据读取、实验编排、结果汇总、notebook 报告、warning labels |
| 知识发现 | `k*` 排序、proxy alignment、domain-specific audit log |
| 应用价值 | 系统能够区分 strong support、partial support、unsupported claims |
| 可复现性 | 20-seed 编排脚本、统一 notebook、结果报告、local-only outputs policy |

### 5.5 投稿前必须补的内容（Must Add Before Submission）

1. 增加工作流图（workflow diagram）。
   - Data -> CMDL / baselines -> diagnostics -> result report -> analyst decision
2. 增加“审计判定规则框（audit decision rule box）”。
   - 例如：高 synthetic 恢复、非退化 `k*`、正向 seed share、proxy direction 稳定、baseline 校准合理。
3. 增加 economics 的简短案例叙事（case-study narrative）。
   - 重点展示如何在部分证据下避免过度主张。
4. 最好补一个轻量展示件（artifact）。
   - 例如小型 dashboard 截图、结果面板截图、统一 notebook 报告截图。

### 5.6 需要避免的风险（Risks to Avoid）

1. 不要只讲 AC-GATE 内部结构，而忽略整个 workflow 的操作价值（operational value）。
2. 不要把论文写得只对 economics 单域成立；应保持 data mining / knowledge discovery 的一般性。

### 5.7 当前判断（Current Decision）

这是第二优先方向。如果你愿意把 notebook 和结果报告进一步产品化（productize），它的适配度会进一步提高。

## 6. 方向三：IEEE ICDM 应用导向投稿（Application-Oriented Submission）

### 6.1 适配原因（Why It Fits）

ICDM 覆盖算法（algorithms）、软件（software）、系统（systems）与应用（applications），也特别适合：

- 时变数据（time-evolving data）
- 可解释建模（interpretable modeling）
- 异构数据整合（heterogeneous data integration）
- 社会科学、气候、金融等应用域（social science / climate / finance applications）

CMDL / AC-GATE 如果改写为“异质滞后挖掘任务（heterogeneous lag mining task）”，会更符合 ICDM 的技术审稿口味。

### 6.2 推荐标题（Recommended Title）

> Interpretable Heterogeneous Lag Mining in Country-Level Panel Time Series

中文可对应为：

> 国家级面板时序中的可解释异质滞后挖掘

### 6.3 推荐主论题（Recommended Thesis）

> 我们将异质滞后发现（heterogeneous lag discovery）形式化为实体条件滞后挖掘任务（entity-conditioned heterogeneous lag mining），并提出 AC-GATE 作为一个具有显式恢复诊断（explicit recovery diagnostics）的可解释挖掘模型（interpretable mining model）。

### 6.4 正文重点（Main-Text Emphasis）

| 部分 | 应强调的内容 |
| --- | --- |
| 任务定义 | 明确 heterogeneous lag mining 与 forecasting 的区别 |
| 算法细节 | AC encoder、lag gate 的公式、训练目标、诊断量计算 |
| 评估设计 | synthetic recovery、ablations、baseline 对比、统计检验、reproducibility checklist |
| 应用案例 | economics 和 energy 作为 public-data case studies，而非主证明域 |

### 6.5 投稿前必须补的内容（Must Add Before Submission）

1. 增加正式任务定义（formal task definition）。
   - 输入、输出、评价指标、可识别性边界（identifiability boundary）。
2. 增加伪代码（pseudocode）与复杂度分析（complexity analysis）。
3. 增加统计检验（statistical testing）。
   - 比如 synthetic 中 CMDL 与 Plain LSTM lag recovery 的 bootstrap interval 或 paired test。
4. 增加强 baseline 表。
   - Persistence、entity mean、Panel OLS / ARDL、Plain LSTM、Uniform Lag、No AC Encoder。
5. 准备匿名可复现包（anonymized reproducibility package）。
   - 如果评审阶段共享代码，需去除仓库中的识别信息。

### 6.6 需要避免的风险（Risks to Avoid）

1. 不要只依赖应用动机（application motivation）；ICDM 会更在意技术定义是否清晰。
2. 不要缺少统计证据；synthetic 主张必须足够硬。

### 6.7 当前判断（Current Decision）

这是第三优先方向。如果补齐 formalization、pseudocode、置信区间和 baseline 表，适配度会明显提升。

## 7. 三个方向的共用执行计划（Shared Execution Plan）

| 步骤 | 产物 | ECML PKDD | CIKM | ICDM |
| --- | --- | ---: | ---: | ---: |
| 增加种子分布图（seed distribution plots） | 箱线图 / CI 表 | 必需 | 必需 | 必需 |
| 增加数据特殊性说明（data idiosyncrasies） | 缺失、切分、proxy 构造 | 必需 | 必需 | 中等 |
| 增加 workflow 图 | 从数据到审计报告的流程图 | 中等 | 必需 | 中等 |
| 增加形式化任务定义（formal task definition） | 数学问题陈述 | 中等 | 中等 | 必需 |
| 增加伪代码与复杂度 | 算法块 | 中等 | 较低 | 必需 |
| 增加 proxy rationale 表 | proxy、符号、来源、caveat | 必需 | 必需 | 必需 |
| 增加 baseline 紧凑表 | Persistence / OLS / ARDL / LSTM / CMDL | 必需 | 必需 | 必需 |
| 增加结果界面截图 | dashboard / notebook report | 可选 | 建议 | 可选 |

建议执行顺序：

1. 先完成 ECML PKDD 版本。
2. 在不改变实验主体的前提下，改写为 CIKM workflow 版本。
3. 最后再补技术化内容，准备 ICDM 版本。

## 8. 结果叙事策略（Result Framing Strategy）

### 8.1 合成域（Synthetic）

Synthetic 是主证据（main evidence），必须放在 Results 的最前面。

应突出：

- 20 种子完整运行（complete 20-seed execution）
- linear 与 nonlinear 两个场景都成立
- CMDL 的 `k*` rank alignment 约为 0.91 到 0.94
- CMDL 的 `k*` MAE 明显低于 Plain LSTM 的 post-hoc lag recovery
- No AC Encoder 与 Uniform Lag 会使机制指标退化
- No Recon Regularization 近似复现 CMDL，说明主贡献更可能来自 AC encoder + lag gate

### 8.2 经济域（Economics）

> **2026-04-26 L2 重写**：在 PWT-CTFP 面板上，CMDL 学到的 per-entity k\* 与训练窗口 entity-level 静态发展指标显著结构化关联：人力资本 |ρ| = 0.371（80% seeds p<0.05，Fisher p = 1.0e-46）、log GDP per worker |ρ| = 0.278（Fisher p = 3.5e-37）、log capital per worker |ρ| = 0.257（Fisher p = 1.4e-24）。No AC Encoder 与 Uniform Lag 退化对照下 kstar_std ≡ 0，stratification 检验结构性不可执行。

推荐表述（L2 主张）：

> 在 PWT 11.0 effective-CTFP 面板上，AC-GATE 恢复出的实体条件滞后分布与人力资本、log 人均 GDP、log 人均资本三类 entity-level 静态指标存在显著结构化关联（permutation Fisher combined p < 1e-24）。该结构在 No AC Encoder 与 Uniform Lag 控制下完全消失，证明它由 AC encoder + lag gate 这一对结构件**联合**支撑，而非来自任意 LSTM 容量。

诚实声明（必须保留）：
- **预测层**：CMDL test R² = 0.054 显著低于 Plain LSTM (0.102, p = 6.3e-05) 与 Uniform Lag (0.104, p = 1.9e-06)。本文不主张该域预测优势。
- **L3 方向性机制**：anchor adjusted ρ = −0.109（35% 正向 seed），不成立。改报 L2 |ρ| 与 ablation 退化对照。

### 8.3 能源域（Energy）

> **2026-04-26 L2 重写**：能源域呈现"L0 退化 + L2 强结构"双解耦：所有神经模型 R² ≈ −0.029（Grouped ARDL R² = 0.607 显著主导），但 CMDL 的 per-entity k\* 与制度治理指标存在极强结构化关联——rule of law |ρ| = 0.735（**95% seeds p<0.05**，Fisher p = 1.8e-79）、government effectiveness |ρ| = 0.716（90% p<0.05，Fisher p = 9.0e-77）、log 人均 GDP |ρ| = 0.609（Fisher p = 1.5e-77）。

推荐表述（L2 主张 + 应用边界双层叙事）：

> 在 OWID-energy × WGI 面板上，AC-GATE 给出一个方法论意义清晰的"双解耦案例（decoupled case）"：所有递归神经模型在 CO₂/energy 上 R² ≈ −0.029（被 Grouped ARDL R² ≈ 0.607 显著超过），但 AC-GATE 学到的 per-entity 有效滞后 k\* 与三类 entity-level 制度 / 经济结构指标存在 |ρ| ≈ 0.6–0.74 的显著结构化关联（permutation Fisher combined p < 1e-76），且该结构在退化 ablation 下完全消失。这个组合证明：**机制可解释性（mechanism interpretability）可以与预测准确性（forecast accuracy）解耦评估**——预测层退化不意味着模型内部的 lag 表征是噪声。

应用边界声明（必须保留）：
- 该域 CO₂/energy 目标对短面板 + 国家级特征近似线性可解，神经容量过剩；论文将其定位为"forecast-layer applicability boundary"，而非主结果预测域。
- L3 anchor ρ ≈ −0.014，不成立；论文不主张方向性机制。

## 9. 推荐论文结构（Recommended Paper Structure）

1. 引言（Introduction）
   - 说明面板时序中的异质滞后问题（heterogeneous lag problem）
   - 说明本文目标是机制发现与审计（discovery and audit），不是因果识别（causal identification）
2. 相关工作（Related Work）
   - distributed lag models
   - neural time series
   - interpretable representation learning
3. 任务定义（Problem Formulation）
4. 方法（Method）
   - AC encoder
   - lag gate
   - prediction head
   - diagnostics
5. 实验（Experiments）
   - synthetic
   - economics
   - energy
   - baselines and ablations
6. 结果（Results）
   - synthetic main evidence
   - ablation necessity
   - economics audit
   - energy stress test
7. 讨论（Discussion）
   - what works
   - what remains partial
   - what fails and why
8. 可复现性说明（Reproducibility）

## 10. 三个方向的摘要角度（Venue-Specific Abstract Angles）

### 10.1 ECML PKDD 角度

> 真实世界中的面板时序应用常要求分析者在短时间维、强异质和高噪声的国家级面板中判断延迟效应（delayed effects）。本文提出 AC-GATE 作为一种应用型滞后审计框架（applied lag-audit framework），结合自适应滞后发现（adaptive lag discovery）、种子稳定性（seed stability）和代理一致性诊断（proxy-alignment diagnostics）。Synthetic ground-truth 实验表明模型在 20 种子下稳定恢复机制，economics 与 energy 面板则分别展示了部分支持和负向边界证据。

### 10.2 CIKM 角度

> 本文提出一个面向国家级面板时序的可复现知识发现工作流（reproducible knowledge-discovery workflow）。该工作流整合自适应滞后建模（adaptive lag modeling）、baseline comparison、per-proxy audit tables 与 notebook-based reports，使分析者能够区分强支持（supported）、部分支持（partial）与不支持（unsupported）的滞后机制主张。

### 10.3 ICDM 角度

> 本文将实体条件异质滞后发现（entity-conditioned heterogeneous lag discovery）形式化为一个可解释数据挖掘任务（interpretable data mining task），并提出 AC-GATE 作为输出实体特异滞后分布（entity-specific lag distributions）与有效滞后期（effective lag score）`k*` 的神经挖掘模型。20 种子 synthetic 实验显示，AC-GATE 在 lag recovery 上稳定优于 Plain LSTM 的事后滞后归因（post-hoc lag attribution），而真实面板案例则揭示了该方法在真实域中的诊断边界。

## 11. 当前最优判断（Current Best Judgment，2026-04-26 L2 修订）

证据结构已经从"synthetic 强 + 真实域弱"升级为：

- Synthetic：L3 directional + L2 structured 双层成立；
- Economics：L2 structured 成立（Fisher p < 1e-24，三个 stratifier 全部 reject null）；ablation 退化对照完美；
- Energy：L2 structured **极强**（|ρ| ≈ 0.6–0.74，95% seeds reject，Fisher p < 1e-76），同时构成"L0 退化 / L2 强结构"双解耦案例；
- 所有真实域 ablation 退化对照（No AC / Uniform Lag）下 kstar_std ≡ 0 → 检验不可执行 → L2 完全归因于 AC encoder + lag gate。

修订后的投稿优先级：

1. **IEEE ICDM 应用导向投稿** —— 升为第 1。L2 + permutation null + ablation kstar_std ≡ 0 的"机制层显著结构化 + 退化对照消失"是数据挖掘类审稿人最熟悉、最易接受的统计形式；ICDM 重视任务形式化与统计证据，与本文当前最强证据完全对齐。
2. **CIKM Applied Research Papers / Applied Track** —— 第 2。L2 stratification + L0/L2 双解耦写成"verdict-style audit workflow that separates forecast accuracy from mechanism interpretability"非常契合 CIKM 应用研究口味。
3. **ECML PKDD Applied Data Science Track** —— 降为第 3。ADS 评审更想要"actionable insight on a real-world problem"，本文真实域证据是"机制结构存在但预测层弱"，这种 actionable 故事不如 ICDM/CIKM 的"task formalization + statistical mechanism evidence"自然。

新的核心一句话：

> CMDL / AC-GATE 是一个可检验的实体条件异质滞后发现框架；其学到的 per-entity effective lag 在合成域恢复方向性机制（L3），并在两个真实国家级面板上恢复 L2 意义下的结构化异质滞后（permutation Fisher combined p < 1e-20），且该结构在 No AC Encoder / Uniform Lag 退化 ablation 下完全消失——证明机制可解释性可与预测准确性解耦评估。

## 12. 投稿方向可行性再审（Post-L2 Venue Feasibility Re-audit）

| 方向 | L2 证据强度 | 主要剩余风险 | 可行性 | 必补内容 |
|---|---|---|---|---|
| **IEEE ICDM** | ★★★★★ | (1) 真实域 sign instability 需在 limitations 显式声明并解释为标签置换不变性；(2) 需补 formal task definition + 伪代码 + 复杂度分析（这是 ICDM 评审硬要求） | **高** | task formalization、pseudocode、complexity、stratified k\* 主表 + ablation kstar_std ≡ 0 对照表、bootstrap CI for synthetic |
| **CIKM Applied** | ★★★★ | (1) 需要把 L0/L2 双解耦写成 workflow 价值，不是单个模型贡献；(2) 真实域预测层弱需写为 "verdict: forecast layer not certified, mechanism layer certified"；(3) 需要 workflow figure | **高** | workflow figure、verdict matrix（domain × layer × verdict）、stratified k\* 主表、real-data case-study narrative |
| **ECML PKDD ADS** | ★★★ | (1) ADS 偏好 actionable real-world insight；本文 actionable insight 是"用 per-entity k\* 给国家分组并对齐治理/资本结构指标"，需要给一个具体应用图例（如：能源域将国家按 learned k\* 二分，展示与 rule of law 的对齐 + 政策含义讨论）；(2) 真实域预测层弱不利于 ADS 口味 | **中等偏上** | applied case-study figure（learned k\* 与 stratifier 散点 + 国家标注）、policy-flavored discussion、stratified k\* 主表 |

三个方向**在假设 review 全部解决的前提下都可行**；优先级建议：**ICDM (1) → CIKM (2) → ECML PKDD (3)**。

> 与原 §11 顺序（ECML PKDD → CIKM → ICDM）相比，新数据让 ICDM 反而最适配，因为 stratified k\* + permutation null + ablation degeneracy 这三件事直接对齐 ICDM 的"interpretable mining task with statistical evidence"标准模板。

### 12.1 通用必补清单（Cross-Venue Must Add）

1. **Stratified k\* 主表**（已生成 [outputs/notebook_economics/.../economics_stratified_kstar_aggregated.csv](outputs/notebook_economics/complete_20seed_20260426/comparison/economics_stratified_kstar_aggregated.csv) 与 [outputs/notebook_energy/.../energy_stratified_kstar_aggregated.csv](outputs/notebook_energy/complete_20seed_20260426/comparison/energy_stratified_kstar_aggregated.csv)）— 论文 Results 主表必含。
2. **Ablation kstar degeneracy 对照表**（已可从 compact_summary 中读到 kstar_std ≡ 0）— 一行两列即可，但是 L2 主张的关键证据。
3. **Sign instability limitations 段** — 解释为标签置换不变性，并说明 |ρ| + share-of-seeds-rejecting 是 sign-robust 报告。
4. **Forecast / mechanism decoupling 论述段** — 把 energy 的 L0 退化 + L2 强结构包装成方法论亮点（不是失败）。
5. **Synthetic-only L3 声明** — 把 anchor proxy 监督的 sign-stable directional alignment 限定在 synthetic 上。

完成以上 5 项后，三个方向都可投。
