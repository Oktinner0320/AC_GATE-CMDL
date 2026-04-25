# Paper Strategy for CMDL / AC-GATE

本文件给出基于当前 20-seed 结果的发文策略。核心原则是：用 synthetic 域承担机制证明，用 economics 域承担真实数据审计，用 energy 域承担压力测试；不要把真实域包装成强机制证明或预测 SOTA。

## 1. Recommended Positioning

建议论文定位为：

> AC-GATE is a mechanism-oriented neural architecture for discovering entity-conditioned heterogeneous lag patterns in panel time series, with synthetic ground-truth validation and real-data audit cases.

中文表述：

> 本文提出一种面向面板时序的条件异质滞后发现框架。模型通过实体级调节代理学习潜在吸收能力表示，并将其映射到实体特异的滞后权重分布，从而在没有直接监督 `k*` 的情况下恢复可解释的滞后结构。

这一路线比“真实经济域证明机制”更稳。当前结果见 [experiment_results_20seed.md](experiment_results_20seed.md)：synthetic 证据很强，economics 证据局部成立，energy 结果偏负面。如果论文主张过强，专家很容易抓住 economics/energy 的不稳定性否定贡献；如果主动把真实域写成审计和边界条件，反而显得诚实、成熟、可信。

## 2. Claim Boundaries

### 2.1 Claims We Can Defend

1. AC-GATE can recover entity-conditioned lag heterogeneity when the data-generating process contains such a mechanism.
2. The recovered lag score `k*` is not merely a post-hoc visualization; it follows from an explicit lag-gating module and is tested against ground truth in synthetic data.
3. AC encoder and learned lag gate are necessary for mechanism recovery in the synthetic setting: removing AC encoder or forcing uniform lag collapses `k*` recovery.
4. Real domains show how to audit whether such a mechanism is present, stable, or absent.
5. The framework is valuable even when a real domain fails the mechanism test, because the failure is measurable through seed stability, proxy alignment, and baseline comparison.

### 2.2 Claims We Should Avoid

1. Do not claim AC-GATE is a universal forecasting SOTA model.
2. Do not claim economics proves the mechanism in real data.
3. Do not claim energy validates the proposed mechanism.
4. Do not imply causal identification. Use predictive association, lag discovery, and mechanism audit language instead.
5. Do not present proxy reconstruction as sufficient evidence of mechanism validity; the 20-seed results show reconstruction regularization is not the main source of recovery.

### 2.3 Best One-Sentence Contribution

> We introduce a testable neural lag-discovery framework that separates mechanism recovery, forecast calibration, and real-data audit, and we show through 20-seed experiments when AC-conditioned lag discovery succeeds, weakens, or fails.

## 3. Target Venue Strategy

严格说，下面三个高优先级目标主要是 conference proceedings / applied track，而不是传统 journal。它们比 KDD ADS / IAAI 更适合当前状态，因为它们通常允许“真实应用问题 + 离线真实数据评估 + 可复现实验”，不一定要求 live deployment 或 post-launch metrics。

| Priority | Venue / track | Fit now | Submission angle | Readiness |
| --- | --- | ---: | --- | --- |
| 1 | ECML PKDD Applied Data Science Track | High | A real-world panel time-series lag-audit method with explicit data idiosyncrasies, diagnostics, and use-case conclusions | Closest fit; needs stronger real-domain framing and reproducibility package |
| 2 | CIKM Applied Research Papers / Industry-style applied track | Medium-high | A reproducible lag-audit workflow for economic and energy panel data, emphasizing implementation, data pipeline, and analyst-facing diagnostics | Good fit if framed as workflow/system, not only model |
| 3 | IEEE ICDM application-oriented submission / Applied Track if available | Medium-high | Interpretable heterogeneous lag mining for country-level panel time series, with algorithmic clarity and public-data reproducibility | Needs stronger technical framing, significance tests, and clean anonymized artifacts |
| 4 | KDD / AAAI / IJCAI workshop | High backup | Mechanism-first method with transparent real-domain audit | Safe fallback |
| 5 | KDD ADS / IAAI | Low now | Deployed decision-support workbench with post-launch metrics | Not ready without deployment or controlled pilot |

最稳的正式投稿路线是：先准备 ECML PKDD ADS 版本；如果时间窗口或 scope 不合适，平移到 CIKM Applied Research；如果想强化技术贡献和数据挖掘味道，再准备 ICDM 版本。三个版本共用同一套实验，但论文叙事、主表和摘要重点不同。

### 3.1 Applied Track Reality Check

2026-04-25 检索 KDD 2026 官方 Applied Data Science Track CFP 后，需要特别注意：KDD ADS 不是“在真实数据上做应用实验”的泛化通道，而是非常强调真实部署的通道。官方页面要求论文描述 deployed applications，并要求量化 post-launch performance；没有 post-launch performance quantification 的投稿会被 desk-rejected。官方还明确说，仅在真实数据上做离线测试但没有 live user deployment，不满足 real-world deployment criteria。

参考页面：

- KDD 2026 Applied Data Science Track CFP: https://kdd2026.kdd.org/applied-data-science-ads-track-call-for-papers/
- KDD 2026 Research Track CFP: https://kdd2026.kdd.org/research-track-call-for-papers/

这意味着：当前 CMDL 项目如果按现在的状态直接投 KDD ADS，风险很高，主要原因不是模型不够，而是没有 deployment 和 post-launch 指标。当前版本更像 mechanism-oriented research / workshop / application-oriented research paper，而不是严格意义的 ADS deployment paper。

### 3.2 If We Want an Applied Track Paper

如果目标是 Applied Track，论文必须从“提出一个模型”升级为“部署一个可用的异质滞后审计系统”。建议命名为：

> AC-GATE Audit Workbench: A deployed decision-support system for auditing heterogeneous delays in panel time-series indicators.

核心转向：

| 当前版本 | Applied track 需要的版本 |
| --- | --- |
| Offline 20-seed experiments | Deployed or controlled-pilot system |
| Synthetic mechanism proof | System capability validation, placed after use case |
| Economics / energy as datasets | Domain workflow with concrete users and decisions |
| Model metrics only | Post-launch usage, expert utility, speed, quality, robustness metrics |
| Method contribution | Application significance + design tradeoffs + lessons learned |

Applied track 的主问题不应写成“Can AC-GATE recover hidden lags?”，而应写成：

> Can a deployed lag-audit workbench help analysts identify, compare, and stress-test heterogeneous policy-response delays across countries more reliably than manual lag specification and homogeneous baselines?

### 3.3 Minimum Work Needed for KDD-Style ADS

| Requirement | Current status | Required action |
| --- | --- | --- |
| Real deployment or controlled pilot | Missing | Build a small web dashboard or notebook-backed workbench and put it in front of real users |
| Post-launch quantification | Missing | Log usage and measure analyst outcomes after deployment |
| Domain significance | Partially present | Pick one primary use case: economics policy lag audit or energy transition lag audit |
| Design tradeoffs | Partially present | Document why neural lag gating, why ARDL baseline remains visible, why negative findings are surfaced |
| Real-world challenges | Partially present | Turn energy failure and economics instability into lessons about false-positive prevention |
| Reproducibility | Mostly present | Keep code/configs tracked; provide run command, seed list, and artifact regeneration protocol |
| Ethics and limitations | Partially present | Add no-causal-claim policy, public-data statement, and analyst-facing warning labels |

Minimum viable applied-track package:

1. Build an analyst-facing workbench.
   - Inputs: country panel, target, treatment/input variable, proxy bundle, split years, baseline set.
   - Outputs: forecast table, `k*` map/ranking, proxy alignment table, seed stability report, baseline comparison, warning labels.
   - Interface can be Streamlit, Dash, a lightweight local web app, or a polished notebook if it is used by actual analysts in a controlled pilot.
2. Run a controlled pilot with domain users.
   - Minimum: 3-5 users with economics, energy, policy, or applied data science background.
   - Ask them to complete realistic tasks: identify countries with short/long lag, compare CMDL vs ARDL, decide whether a mechanism claim is supportable, detect unstable proxy directions.
   - Record time, correctness against a rubric, confidence, perceived usefulness, and failure cases.
3. Quantify post-launch performance.
   - Usage: number of users, sessions, completed analyses, repeated use.
   - Efficiency: time-to-insight versus manual spreadsheet/ARDL workflow.
   - Decision quality: agreement with expert rubric or known synthetic ground truth in training tasks.
   - Reliability: seed-stability warnings reduce unsupported mechanism claims.
   - Model calibration: baseline-aware R2/RMSE and mechanism diagnostics from [experiment_results_20seed.md](experiment_results_20seed.md).
4. Add a lessons-learned section.
   - Economics: partial signal, unstable expected direction, good example of audit caution.
   - Energy: negative/stress-test case, shows the system does not always manufacture a positive mechanism.
   - Synthetic: used to train and validate analysts' trust in `k*`, not as the sole application proof.

如果无法获得真实用户或 pilot，建议不要投 KDD ADS。可以改投 Research Track 的 application-oriented paper、workshop、AI for social impact workshop、time-series workshop，或把 Applied Track 目标推迟到下一轮。

### 3.4 Applied Track Paper Storyline

Applied track 版本的标题应避免像纯方法论文，可以考虑：

- Auditing Heterogeneous Policy Delays in Panel Time Series with AC-GATE
- A Decision-Support Workbench for Entity-Specific Lag Discovery in Economic and Energy Panels
- From Lag Discovery to Lag Audit: Deploying AC-GATE for Interpretable Panel Time-Series Analysis

推荐摘要逻辑：

1. Real-world problem: analysts often need to choose lag windows for policy/economic indicators, but homogeneous lag assumptions hide cross-country heterogeneity.
2. Deployed solution: we built AC-GATE Audit Workbench, which combines adaptive lag discovery with baseline comparison and seed-stability warnings.
3. Post-launch evidence: report pilot usage, task-completion speed, expert agreement, and cases where the system prevented overclaiming.
4. Technical evidence: synthetic ground-truth recovery and 20-seed robustness.
5. Lessons: real domains contain partial and negative mechanism evidence; applied systems must surface uncertainty instead of forcing a positive narrative.

Applied track 的 Results 排序也应改变：

| Section | Applied track emphasis |
| --- | --- |
| Deployment setting | Who used the system, for what task, under what constraints |
| Post-launch metrics | Usage, speed, expert agreement, decision quality, failure detection |
| System design | Why these diagnostics are exposed; why baselines remain first-class citizens |
| Model validation | Synthetic 20-seed recovery and ablation support |
| Domain audit findings | Economics partial, energy negative, both framed as practical lessons |

### 3.5 Applied Track Metrics to Add

Do not rely only on R2, MAE, and Spearman rho. Add applied-system metrics:

| Metric family | Concrete metric | Why reviewers care |
| --- | --- | --- |
| Adoption | users, sessions, analyses completed | Shows the system was actually used |
| Efficiency | median time-to-insight vs manual baseline | Demonstrates workflow value |
| Decision quality | agreement with expert rubric | Shows outputs help analysts make better judgments |
| Safety | unsupported-claim rate before/after warnings | Shows the audit design reduces overclaiming |
| Robustness | seed-stability pass/fail rate | Connects ML reliability to user-facing decisions |
| Baseline awareness | fraction of cases where simpler baseline is recommended | Shows the system does not blindly promote CMDL |
| Qualitative value | expert ratings and quotes | Captures lessons that metrics miss |

For this project, the most defensible applied metric is “unsupported mechanism claim reduction”: give analysts compact result tables with and without seed-stability/proxy-direction warnings, then measure whether they stop claiming that economics or energy strongly proves the mechanism. That directly matches the current evidence and turns the negative/partial results into an applied contribution.

### 3.6 Applied Track Risk Decision

Decision rule:

| Condition | Recommendation |
| --- | --- |
| No deployed workbench, no user pilot | Do not submit to strict ADS track |
| Workbench exists but only author used it | Still weak for ADS; target workshop / applied research track |
| 3-5 domain users complete controlled pilot and post-launch metrics are reported | Possible ADS submission |
| Institutional or lab partner uses it for repeated analyses | Strong ADS submission |

Current state: **not yet ADS-ready**. Best applied-track path is to spend 2-4 weeks building a small audit workbench and running a controlled expert pilot. Without that, use the current 20-seed package for a workshop or application-oriented research submission instead.

### 3.7 Venue-Specific Revision Plan

#### 3.7.1 ECML PKDD Applied Data Science Track

ECML PKDD ADS should be treated as the first target. The fit is strong because this track historically asks for unique applications of machine learning / data mining to real-world problems, including the real-world difficulty, data idiosyncrasies, methodology, and conclusions for the use case. That maps well to CMDL: the contribution is not only a neural module, but a reproducible protocol for auditing whether heterogeneous lags are present in country-level panels.

Recommended title:

> Auditing Heterogeneous Delays in Panel Time Series with AC-GATE

Recommended thesis:

> We study a real-world applied data science problem: analysts need to reason about cross-country lag heterogeneity, but standard homogeneous-lag baselines and post-hoc neural explanations are hard to audit. AC-GATE provides a diagnostic pipeline that recovers known heterogeneous lags in synthetic panels and exposes partial or negative evidence in real economic and energy panels.

Main-text emphasis:

| Section | What ECML PKDD ADS should see |
| --- | --- |
| Problem | Real-world panel time-series lag audit, not generic sequence modeling |
| Data | Country-level annual panels, missingness, balanced-panel filtering, train-window standardization, proxy sign assumptions |
| Method | AC encoder + lag gate + audit diagnostics |
| Results | Synthetic ground truth first, then economics/energy as use-case conclusions |
| Lessons | Real data can produce partial or negative mechanism evidence; the method is useful because it surfaces that uncertainty |

Must add before submission:

1. A “Data idiosyncrasies” subsection.
   - Explain annual panels, short time dimension, cross-country heterogeneity, proxy aggregation, train/test time splits, and why simple random splits are invalid.
2. A “Use-case conclusions” table.
   - Synthetic: mechanism supported.
   - Economics: partial audit support.
   - Energy: stress-test / weak mechanism evidence.
3. Confidence intervals or seed-distribution plots.
   - ECML reviewers will expect the 20-seed claim to be visible, not buried in CSVs.
4. A reproducibility appendix.
   - Include command, seed list, config table, and note that `outputs/` is regenerated locally.

Risk to avoid:

- Do not lead with “we outperform baselines.” In economics and energy, that is not true enough.
- Do not call energy a failed experiment. Call it a stress-test case showing that the audit protocol does not force a positive mechanism.

Decision: **best current target**.

#### 3.7.2 CIKM Applied Research Papers / Applied Track

CIKM Applied Research is the second target. It is likely to value implementation, data pipeline, system design, and practical lessons. For CIKM, the paper should look less like “a model architecture paper” and more like “a reusable lag-audit workflow for knowledge discovery in panel data.”

Recommended title:

> A Reproducible Lag-Audit Workflow for Economic and Energy Panel Data

Recommended thesis:

> We present an applied research workflow that integrates adaptive lag discovery, baseline comparison, seed-stability diagnostics, and proxy-alignment audit for country-level panel time series.

Main-text emphasis:

| Section | What CIKM should see |
| --- | --- |
| System/workflow | Data loading, experiment orchestration, notebook reports, comparison tables, and warning labels |
| Knowledge discovery | `k*` ranking, proxy alignment, per-domain audit logs, and baseline-aware interpretation |
| Applied lesson | The system separates supported, partial, and unsupported mechanism claims |
| Reproducibility | 20-seed orchestration script, notebooks, result report, local-only outputs policy |

Must add before submission:

1. A workflow diagram.
   - Data -> CMDL / baselines -> diagnostics -> result report -> analyst decision.
2. A compact “audit decision rule” box.
   - Strong support requires high synthetic recovery, non-degenerate `k*`, positive seed share, proxy direction stability, and baseline sanity.
3. A short case-study narrative for economics.
   - Show how a reader should interpret partial evidence without overclaiming.
4. Optional but useful: a small Streamlit/Dash or notebook dashboard screenshot.
   - CIKM will likely tolerate offline evaluation, but an analyst-facing artifact improves the applied story.

Risk to avoid:

- Do not over-focus on AC-GATE internals while hiding the workflow. CIKM applied reviewers need to see the operational value.
- Do not make the paper too domain-specific to economics; keep the data mining / knowledge discovery angle broad.

Decision: **strong second target**, especially if a small dashboard or polished notebook report is added.

#### 3.7.3 IEEE ICDM Application-Oriented Submission

ICDM is the third target. It can fit because ICDM covers algorithms, software, systems, and applications of data mining, including time-evolving data, interpretable modeling, heterogeneous data integration, and applications in social science / climate / finance. The tradeoff is that ICDM will likely judge technical merit more sharply than ECML PKDD ADS or CIKM Applied Research.

Recommended title:

> Interpretable Heterogeneous Lag Mining in Country-Level Panel Time Series

Recommended thesis:

> We formulate heterogeneous lag discovery as a data mining problem over entity-indexed temporal panels and introduce AC-GATE as an interpretable mining model with explicit recovery diagnostics.

Main-text emphasis:

| Section | What ICDM should see |
| --- | --- |
| Mining task | Define heterogeneous lag mining clearly and distinguish it from forecasting |
| Algorithm | Give precise AC encoder / lag gate equations and pseudocode |
| Evaluation | Synthetic recovery, ablations, baseline comparison, statistical tests, reproducibility checklist |
| Applications | Economics and energy as public-data case studies, not primary proof |

Must add before submission:

1. Formal task definition.
   - Name the task: entity-conditioned heterogeneous lag mining.
   - Define input, output, evaluation, and what is identifiable.
2. Pseudocode and complexity.
   - Training loop, diagnostic computation, and runtime scaling with `N`, `T`, and `K`.
3. Statistical testing.
   - Paired or bootstrap confidence intervals for CMDL vs Plain LSTM post-hoc lag recovery on synthetic data.
4. Stronger baseline section.
   - Persistence, entity mean, Panel OLS / ARDL, Plain LSTM, Uniform Lag, No AC Encoder.
5. Anonymized reproducibility plan.
   - ICDM uses strict blind review in recent calls; remove identifying repo metadata if sharing code during review.

Risk to avoid:

- Do not rely on application motivation alone. ICDM needs a sharper technical contribution.
- Do not submit an applied narrative without statistical evidence; the synthetic mechanism claim should be statistically tight.

Decision: **good target if we add formalization, pseudocode, CIs, and a cleaner baseline table**.

### 3.8 Three-Target Execution Plan

| Step | Output | Needed for ECML PKDD | Needed for CIKM | Needed for ICDM |
| --- | --- | ---: | ---: | ---: |
| Add seed distribution plots | Boxplots / CI table for `k*` MAE, rho, R2 | Yes | Yes | Yes |
| Add data idiosyncrasy section | Missingness, splits, proxy construction | Yes | Yes | Medium |
| Add workflow diagram | Pipeline from data to audit report | Medium | Yes | Medium |
| Add formal task definition | Mathematical problem statement | Medium | Medium | Yes |
| Add pseudocode / complexity | Algorithm block | Medium | Low | Yes |
| Add proxy rationale table | Proxy, expected sign, source, caveat | Yes | Yes | Yes |
| Add baseline compact table | Persistence / OLS / ARDL / LSTM / CMDL | Yes | Yes | Yes |
| Add dashboard / report screenshot | Optional artifact | Medium | High | Low |
| Add controlled pilot | User study / analyst feedback | Optional | Optional-high | Optional |

Recommended order of work:

1. Prepare ECML PKDD ADS manuscript first.
2. Keep CIKM variant as a workflow-heavy rewrite of the same paper.
3. Prepare ICDM variant only after adding formalization, statistical testing, and algorithmic details.

### 3.9 Venue-Specific Abstract Angles

ECML PKDD ADS angle:

> Real-world panel time-series applications often require analysts to reason about delayed effects under short, noisy, and heterogeneous country panels. We introduce AC-GATE as an applied lag-audit framework that combines adaptive lag discovery with seed-stability and proxy-alignment diagnostics. Synthetic ground-truth experiments show stable mechanism recovery across 20 seeds, while economics and energy panels illustrate partial and negative mechanism evidence in realistic settings.

CIKM Applied Research angle:

> We present a reproducible workflow for knowledge discovery in country-level panel time series. The workflow orchestrates adaptive lag modeling, baseline comparison, per-proxy audit tables, and notebook-based reports, enabling analysts to distinguish supported, partial, and unsupported lag-mechanism claims.

ICDM angle:

> We formulate entity-conditioned heterogeneous lag mining as an interpretable data mining task and propose AC-GATE, a neural lag-gating model that outputs entity-specific lag distributions and expected lag scores. Across 20-seed synthetic experiments, AC-GATE recovers known lag mechanisms more accurately than post-hoc LSTM lag attribution, and real panel case studies demonstrate the diagnostic boundary of the approach.

## 4. Paper Thesis

建议主论点分三层：

1. Mechanism recovery: 在有 ground truth 的 synthetic panel 中，AC-GATE 能稳定恢复 `z_i -> omega_i -> k_i*` 的机制链。
2. Necessity: 消融显示 AC encoder 和 learned lag gate 是机制恢复的必要结构。
3. Auditability: 在真实数据中，AC-GATE 不只输出预测，还输出可审计的机制诊断；真实域结果可以被判定为支持、局部支持或不支持机制假设。

这三层中，第一层和第二层是强证据，第三层是方法价值。不要把第三层写成强实证结论。

## 5. Result Framing

### 5.1 Synthetic as Main Evidence

Synthetic 应放在 Results 的第一位，并作为机制证明的主证据。

必须突出：

- 20 seeds 完整运行。
- linear 与 nonlinear 均稳定成立。
- CMDL 的 `k*` rank alignment 约为 `0.91-0.94`。
- CMDL 的 `k*` MAE 明显低于 Plain LSTM post-hoc lag recovery。
- No AC Encoder 和 Uniform Lag 的 `k*` rho 退化为 0。
- No Recon Regularization 近似复现 CMDL，说明主要贡献来自 AC encoder + lag gate，而不是重构损失本身。

建议图表：

| Figure/Table | 内容 | 目的 |
| --- | --- | --- |
| Figure 1 | AC-GATE architecture | 让审稿人快速理解机制链 |
| Table 1 | Synthetic 20-seed summary | 主证据表 |
| Figure 2 | `k* true` vs `k* predicted` scatter | 直观看机制恢复 |
| Figure 3 | Ablation comparison | 证明结构必要性 |

### 5.2 Economics as Real-Data Audit

Economics 应写成“real-data audit case”，不是“real-data proof”。

可说：

- CMDL 学到非退化 lag heterogeneity。
- `k* std = 0.170`，lag sensitivity `0.508`，proxy R2 `0.300`，说明模型没有完全塌缩成同质滞后。
- 但 forecast R2 `0.053` 低于 Plain LSTM `0.101` 和 Uniform Lag `0.104`。
- anchor-adjusted rho 平均为 `-0.090`，positive seed share 只有 `0.40`。
- `hc_level` 是唯一 candidate_positive proxy。

推荐写法：

> The economics panel shows partial but not decisive mechanism evidence. AC-GATE learns non-degenerate lag heterogeneity, yet the expected proxy direction is not seed-stable. We therefore treat this domain as an audit case rather than a confirmation of the mechanism hypothesis.

这句话很重要。它主动承认边界，会降低专家对“过度解释”的攻击。

### 5.3 Energy as Stress Test

Energy 应写成压力测试或负例。

可说：

- CMDL 与 matched Plain LSTM 接近，R2 分别约为 `-0.028` 和 `-0.029`。
- Grouped ARDL 达到 `0.607`，说明该域中简单结构化模型显著更合适。
- CMDL 有非退化 lag gate，但 proxy direction 平均接近 0，positive seed share `0.45`。
- 三个 WGI proxy 全部为 mixed_or_negative。

推荐写法：

> The energy panel is a useful falsification-style test: AC-GATE does not manufacture a convincing mechanism signal when the domain evidence is weak. This behavior supports the audit protocol, while also showing that the current energy specification is not a positive validation case.

## 6. Reviewer Risk Map

| Reviewer concern | Risk level | Best response |
| --- | --- | --- |
| Synthetic evidence may be too easy | High | Include nonlinear scenario, multiple seeds, ablations, and lag recovery against Plain LSTM post-hoc baseline |
| Real domains do not validate the mechanism | High | Frame real domains as audit cases; claim the method provides diagnostics, not guaranteed confirmation |
| Forecasting performance is not SOTA | High | State forecasting is a calibration metric; the contribution is interpretable lag recovery |
| No causal identification | High | Explicitly avoid causal language; use predictive lag association and mechanism audit |
| Proxy choices may be arbitrary | Medium-high | Add proxy rationale, expected sign table, and per-proxy audit |
| Reconstruction loss is not decisive | Medium | Present No Recon Regularization honestly; say AC encoder + lag gate are the core mechanism |
| Energy results are negative | Medium | Use as stress test and boundary condition, not as failed validation hidden in appendix |
| Outputs are not in repository | Low | Provide scripts, seeds, configs, and summary report; raw artifacts are local due size and reproducibility policy |

## 7. Additional Experiments Before Submission

These are not all mandatory for a workshop paper, but they are ranked by review value.

| Priority | Experiment | Why it matters | Minimum acceptable version |
| --- | --- | --- | --- |
| P0 | Confidence intervals / seed distribution plots | Experts expect variability, not only means | Add CI or boxplot for key metrics |
| P0 | Matched-init no-recon ablation | Current no-recon comparison can be confounded by initialization | Re-seed before constructing no-recon model |
| P0 | Proxy sign rationale table | Defends real-domain interpretation | One table with proxy, expected sign, source, rationale |
| P1 | Alternative economics anchor bundle | Current economics sign is unstable | Test at least one revised proxy bundle |
| P1 | Persistence and simple baseline table in main text | Avoid “neural model without simple baselines” criticism | Include persistence, entity mean, panel OLS / ARDL |
| P1 | Runtime and parameter count | Workshop reviewers often ask practicality | One compact table |
| P2 | Statistical significance test for synthetic recovery | Strengthens main claim | Paired test or bootstrap CI over seeds |
| P2 | Energy respecification | Current energy is negative | Optional; only if time allows |

最关键的是 P0。尤其 matched-init no-recon ablation 很重要，因为当前结果已经显示 reconstruction regularization 不是主要机制来源。如果不补，论文里就不要把 reconstruction loss 写成核心贡献。

## 8. Recommended Paper Structure

1. Introduction
   - Motivate heterogeneous lags in panel time series.
   - State that the goal is mechanism discovery and audit, not causal proof.
   - Summarize synthetic success and real-domain audit outcomes.
2. Related Work
   - Distributed lag models and ARDL.
   - Neural time series and panel forecasting.
   - Interpretable representation learning / mechanism diagnostics.
3. Problem Formulation
   - Define panel units, treatment/input sequence, proxy vector, static features, target, lag distribution, and `k*`.
   - Define what is and is not identifiable.
4. Method
   - AC encoder.
   - Conditional lag gate.
   - Prediction head.
   - Diagnostics: `k*`, proxy alignment, lag sensitivity, entropy, seed stability.
5. Experiments
   - Synthetic ground-truth setup.
   - Economics audit setup.
   - Energy stress-test setup.
   - Baselines and ablations.
6. Results
   - Start with synthetic mechanism recovery.
   - Then ablations.
   - Then real-domain audits.
7. Discussion
   - Why synthetic proves recovery under known mechanism.
   - Why economics is partial evidence.
   - Why energy fails and what that means.
   - Limits: no causal identification, proxy dependence, annual panel constraints.
8. Reproducibility
   - Point to scripts, seeds, configs, and summary files.
   - State that large outputs are excluded from Git and regenerated locally.

## 9. Main Text vs Appendix

| Main text | Appendix |
| --- | --- |
| Architecture diagram | Full hyperparameter table |
| Synthetic main summary | All seed-level synthetic tables |
| Key ablation table | Training curves |
| Real-domain compact audit table | Per-proxy audit details |
| Claim boundary paragraph | Full negative results and artifact paths |

不要把 negative real-domain results 全放 appendix。专家看到后会觉得刻意回避。更好的写法是在正文给 compact audit table，附录给完整细节。

## 10. Abstract Draft Skeleton

> Panel time series often exhibit entity-dependent delays, yet standard distributed lag and neural forecasting models either assume homogeneous lags or provide only post-hoc interpretations. We propose AC-GATE, a conditional lag-discovery architecture that maps entity-level proxy variables to adaptive lag-weight distributions and produces an interpretable expected lag score. On synthetic panels with known ground-truth mechanisms, AC-GATE consistently recovers heterogeneous lag structure across 20 random seeds and outperforms post-hoc lag recovery from matched LSTM baselines. Ablations show that both the adaptive conditioning encoder and learned lag gate are necessary for recovery. On economics and energy panels, the same diagnostics reveal partial support in one domain and weak evidence in another, demonstrating that the framework can audit rather than assume mechanism validity. The results position AC-GATE as a transparent tool for mechanism-oriented lag discovery in panel forecasting tasks.

## 11. Rebuttal Preparation

Prepare short answers for these likely questions:

1. Why not claim causal effects?
   - Because the data are observational panels without identification assumptions; the paper studies predictive heterogeneous lag patterns.
2. Why include domains where AC-GATE is not best?
   - Because the method is designed to audit mechanism evidence. Negative or partial domains demonstrate diagnostic honesty and boundary conditions.
3. Why is synthetic central?
   - Because only synthetic data provide ground-truth `z` and `k*`, which are required for a clean mechanism recovery test.
4. Does reconstruction loss matter?
   - Current evidence suggests it is not the primary driver; the core mechanism is AC-conditioned lag gating. The paper should present reconstruction as an auxiliary diagnostic/regularizer, not as the main contribution.
5. Why exclude outputs from Git?
   - Raw artifacts are large and regenerable. The repository should track code, configs, notebooks, summary reports, and exact seeds; outputs remain local or can be archived separately for submission.

## 12. Submission Checklist

Before submission, complete the following:

- [ ] Add confidence intervals or seed-distribution plots for the main synthetic metrics.
- [ ] Re-run no-recon ablation with matched initialization or weaken claims about reconstruction loss.
- [ ] Add a proxy rationale table for economics and energy.
- [ ] Add a compact baseline table with persistence, entity mean, panel OLS / ARDL, Plain LSTM, and CMDL.
- [ ] Ensure all paper claims match [experiment_results_20seed.md](experiment_results_20seed.md).
- [ ] Keep `outputs/` local-only; do not commit generated artifacts.
- [ ] Provide a reproducibility command using [experiments/run_complete_20seed_suite.py](experiments/run_complete_20seed_suite.py).

## 13. Bottom Line

最可发表的版本不是“AC-GATE 在所有真实域都证明了机制”，而是：

> AC-GATE 提供了一个可检验的条件异质滞后发现框架；它在有真值的 synthetic 中稳定恢复机制，并在真实域中给出可审计的支持、局部支持或否定证据。

这个主张更窄，但更硬。专家评审通常更愿意接受边界清晰、负结果透明、可复现的机制型论文，而不是证据不足却试图讲成全域成功的故事。