# CMDL / AC-GATE 论文策略（Paper Strategy）

本文件基于当前完整的 20 种子结果（20-seed results），给出最适合 CMDL / AC-GATE 的论文投稿策略。当前最稳的写法不是把真实域结果包装成“强机制证明（strong mechanism proof）”，而是采用如下三层证据结构：

- 合成域（synthetic）承担机制恢复证明（mechanism recovery proof）
- 经济域（economics）承担真实数据审计（real-data audit）
- 能源域（energy）承担压力测试（stress test）与边界说明（boundary condition）

基于当前证据，本文档只保留三个最匹配、最可落地的应用型投稿方向（applied-track directions）：

1. ECML PKDD Applied Data Science Track
2. CIKM Applied Research Papers / Applied Track
3. IEEE ICDM 应用导向投稿（application-oriented submission）

## 1. 论文核心定位（Core Positioning）

建议将论文定位为：

> 一种面向面板时序（panel time series）的实体条件异质滞后发现框架（entity-conditioned heterogeneous lag discovery framework）。模型通过实体级代理变量（proxy variables）学习潜在吸收能力表示（latent absorption representation），再将其映射为实体特异的滞后权重分布（entity-specific lag-weight distribution），从而在没有直接监督有效滞后期（effective lag）`k*` 的情况下恢复可解释的异质滞后结构（interpretable heterogeneous lag structure）。

当前结果支持以下主线，而不支持更强叙述：

- 适合：可检验的异质滞后发现（testable lag discovery）
- 适合：机制恢复（mechanism recovery）与真实域审计（real-data audit）
- 不适合：真实域强机制证明（strong real-world mechanism proof）
- 不适合：统一预测最优（universal forecasting superiority）

## 2. 主张边界（Claim Boundary）

### 2.1 可以稳健主张的内容（Defensible Claims）

1. AC-GATE 能在存在真实机制的数据生成过程中恢复实体条件异质滞后（entity-conditioned heterogeneous lags）。
2. 模型输出的有效滞后期 `k*` 不是单纯的事后可视化（post-hoc visualization），而是来自显式滞后门控（explicit lag gating）结构。
3. 自适应条件编码器（AC encoder）与学习型滞后门控（learned lag gate）是机制恢复的必要结构（necessary components）。
4. 真实数据域可以被写成机制审计（mechanism audit）案例，而不是一律写成机制确认（mechanism confirmation）。
5. 一个真实域即使没有形成正向机制证据，仍然可以构成有价值的负结果（negative result）或边界结果（boundary result），因为它说明审计协议不会强行制造正面结论。

### 2.2 应避免的主张（Claims to Avoid）

1. 不要宣称 CMDL / AC-GATE 是通用预测最优模型（universal forecasting SOTA model）。
2. 不要宣称 economics 域已经证明了真实机制（proved the mechanism in real data）。
3. 不要宣称 energy 域验证了所提机制（validated the mechanism）。
4. 不要暗示严格因果识别（causal identification）；应使用预测关联（predictive association）、滞后发现（lag discovery）、机制审计（mechanism audit）等表述。
5. 不要把代理重构（proxy reconstruction）写成充分证据（sufficient evidence），因为当前 20-seed 结果说明 reconstruction regularization 不是主要机制来源。

### 2.3 最佳一句话贡献（One-Sentence Contribution）

> 本文提出一个可检验的神经滞后发现框架（testable neural lag-discovery framework），将机制恢复（mechanism recovery）、预测校准（forecast calibration）与真实域审计（real-data audit）明确分离，并通过 20 种子实验展示 AC 条件滞后发现何时成立、何时减弱、何时失效。

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

Economics 应写为真实数据审计案例（real-data audit case），不是现实机制证明（real-data proof）。

推荐表述：

> Economics 面板显示出部分但不决定性的机制证据（partial but not decisive mechanism evidence）。AC-GATE 学到了非退化滞后异质性（non-degenerate lag heterogeneity），但预期代理方向（expected proxy direction）在种子层面不稳定，因此该域更适合作为审计案例，而非确认案例。

### 8.3 能源域（Energy）

Energy 应写为压力测试（stress test）或负向边界案例（negative boundary case）。

推荐表述：

> Energy 面板是一个有价值的反例式测试（falsification-style test）：当领域证据较弱时，AC-GATE 不会自动制造一个看似可信的正面机制信号（positive mechanism signal）。

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

## 11. 当前最优判断（Current Best Judgment）

基于当前内容，最优顺序是：

1. ECML PKDD Applied Data Science Track
2. CIKM Applied Research Papers / Applied Track
3. IEEE ICDM 应用导向投稿

原因不是这三个方向都更容易，而是它们与当前证据结构更匹配：

- 你已经有强 synthetic 机制证明
- 你已经有 economics 的部分支持型真实域审计
- 你已经有 energy 的负向 / 压力测试证据
- 你已经有统一的 20-seed 实验编排、notebook 汇总与结果报告

因此，当前最可发表的版本不是：

> CMDL / AC-GATE 在所有真实域都证明了机制。

而是：

> CMDL / AC-GATE 提供了一个可检验、可审计、可复现的实体条件异质滞后发现框架（testable, auditable, reproducible framework）；它在有真值的 synthetic 面板中稳定恢复机制，并在真实域中区分正向、部分和负向证据。