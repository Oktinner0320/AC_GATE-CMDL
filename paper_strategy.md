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

| Tier | Venue type | Fit | Submission angle |
| --- | --- | --- | --- |
| Primary | KDD / AAAI / IJCAI workshop on time series, data mining, AI for social impact, or applied ML | High | Mechanism-first method with transparent real-domain audit |
| Secondary | Applied track short paper / symposium | Medium | Interpretable heterogeneous lag discovery for panel data |
| Risky | Main ML conference full paper | Low-medium | Needs stronger theory, stronger baselines, and a primary real domain with stable mechanism evidence |
| Not recommended yet | Economics or energy domain journal | Low | Current real-domain evidence is not strong enough for domain-causal claims |

最稳的路线是先投 workshop 或 applied short paper。该版本应强调方法论、诊断协议、可复现实验，而不是宣称在真实经济或能源问题上已经得到领域定论。

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