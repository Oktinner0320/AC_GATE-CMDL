#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [AC-gate],
  abstract: [
    add this part in the end
  ],
  authors: (
    (
      name: "tmp",
      department: [Co-Founder],
      organization: [Typst GmbH],
      location: [Berlin, Germany],
      email: "haug@typst.app"
    ),
    (
      name: "tmp",
      department: [Co-Founder],
      organization: [Typst GmbH],
      location: [Berlin, Germany],
      email: "maedje@typst.app"
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
)

// Your content goes below.

// ---------------------------------------------------------------------------
// Single-column figure helper and compact status chip for verdict summaries.
// ---------------------------------------------------------------------------
#let colimg(path, caption) = figure(
  placement: top,
  image(path, width: 100%),
  caption: caption,
)

#let verdict(fill, label) = box(
  width: 100%,
  fill: fill,
  inset: (x: 6pt, y: 4pt),
  radius: 3pt,
)[#align(center + horizon)[*#label*]]

= Introduction

// TODO: write Introduction. Hook = "predicting accurately and explaining the underlying lag mechanism are two orthogonal goals; existing panel models conflate them." Contributions: (i) by-construction entity-conditioned heterogeneous lag model, (ii) a three-tier (L1/L2/L3) audit protocol that separates forecast certification from mechanism certification, (iii) a dual-decoupled real-data case (energy) showing the protocol matters.

= Related Work

// TODO: distributed-lag and ARDL models [TODO-pesaran-smith-1995]; panel deep models / TFT [TODO-lim-tft-2021]; sparsemax / softmax gates [TODO-martins-sparsemax-2016]; gated residual networks [TODO-lim-tft-2021]; post-hoc interpretability vs by-construction structure [TODO-rudin-stop-2019]. Gap sentence: "Existing distributed-lag models assume homogeneous or manually-grouped lags, while panel deep models offer no by-construction entity-level lag structure and no protocol to test whether learned heterogeneity is genuine."

= Problem Formulation

== Setup

We consider a balanced panel time series with $N$ entities and $T$ time steps. For each entity $i in {1, dots, N}$ and time $t in {1, dots, T}$ we observe a sequential covariate $X_(i,t) in bb(R)^(d_x)$ and a scalar target $Y_(i,t) in bb(R)$. Each entity is additionally described by a time-invariant proxy vector $p_i in bb(R)^M$ that encodes prior beliefs about its absorptive capacity (e.g.~human capital, governance indices) and an optional vector of static features $s_i in bb(R)^(d_s)$. All splits are anchored on time: the train, validation and test windows correspond to disjoint year ranges, and per-entity z-scoring as well as all entity-level stratifiers are fitted on the train window only.

== Task: Entity-Conditioned Heterogeneous Lag Mining

Given a maximum lag horizon $K$, we seek, for each entity $i$, a distribution over lags
$ omega_i = (omega_(i,1), dots, omega_(i,K)) in Delta^(K-1), $
that is then used to aggregate past sequential covariates into a per-entity context vector
$ c_(i,t) = sum_(k=1)^K omega_(i,k) X_(i,t-k), $
which feeds a recurrent backbone predicting $hat(Y)_(i,t+1)$. The interpretable inferential output is the per-entity expected lag
$ k_i^* = sum_(k=1)^K k dot omega_(i,k), $
which, because $omega_i$ depends only on $p_i$ in our parameterisation, is constant across $t$ for a fixed entity within one trained model.

== Three-Tier Evidence Ladder

To avoid over-claiming on real-world panels, we assess the discovered lag structure on three explicit, increasingly stringent tiers.

- *L1: Learnable heterogeneous lag.* The cross-entity standard deviation of $k_i^*$ is strictly positive and the variant passes a degeneracy guard against constant-$omega$ ablations.

- *L2: Structured heterogeneous lag (main claim).* Per-entity $k_i^*$ is significantly aligned with at least one external, train-window-only entity-level stratifier under an entity-permutation null, and the alignment vanishes under L1-degenerate ablations.

- *L3: Directional mechanism.* Anchor-proxy Spearman $rho$ has a stable sign across seeds with a majority of seeds rejecting the null. This tier is restricted to the synthetic regime where the anchor proxy is supervised by ground truth.

The main claim of this paper is at L2: $k_i^*$ recovers a structured heterogeneity that is statistically aligned with established entity-level indicators and that is not reproducible by any of the degenerate ablations.

== Identifiability and Out-of-Scope Claims

The unsupervised recovery of $omega_i$ from $p_i$ is invariant to label permutations of the latent absorption score: across random initialisations, the model can stably partition entities into two groups while flipping which group is assigned the larger $k_i^*$. Consequently, signed Spearman correlations on real data are not directly comparable across seeds, and we report sign-robust quantities (mean $|rho|$, share of seeds rejecting the permutation null, and Fisher-combined $p$-values). Sign-stable directional claims are restricted to L3 in the synthetic regime.

We do not claim causal identification, nor universal forecasting superiority over classical or neural baselines; both are explicitly out of scope. Forecast performance is reported only as L0 calibration evidence under the audit protocol of Section~5.

= Method: AC-gate

We name our model AC-GATE: Adaptive-Conditioning encoder with a scale-invariant lag GATE on top of an LSTM backbone. The pipeline maps $p_i mapsto z_i mapsto omega_i mapsto c_(i,t) mapsto hat(Y)_(i,t+1)$.

== Adaptive Conditioning Encoder

A small multilayer perceptron $f_phi: bb(R)^M -> bb(R)$ summarises the proxy vector into a scalar latent absorption score
$ z_i = f_phi(p_i). $
A linear reconstruction head $g_psi: bb(R) -> bb(R)^M$ maps $z_i$ back to the proxy space, producing $hat(p)_i = g_psi(z_i)$. The reconstruction serves as an auxiliary anchor that prevents $z_i$ from drifting away from the input proxies during training; Section~6 shows empirically that removing it has no significant effect on mechanism recovery, so we treat it as a regulariser rather than a core mechanism.

== Scale-Invariant Lag Gate

Conditioned on $z_i$, a scalar gated residual network $h_theta: bb(R) -> bb(R)$ produces a single logit that is broadcast across the $K$ lag positions, then biased by a position-dependent prior:
$ omega(k mid(|) z_i) = "Softmax"_k ((h_theta(z_i) - lambda dot k slash K) / tau). $
Two design choices are worth highlighting. First, the position bias $lambda dot k slash K$ is *scale-invariant in $K$*: changing the lag horizon does not rescale the relative penalty on far lags. Second, a single scalar logit per entity is sufficient because the softmax over $K$ positions, combined with the linear bias, defines a one-parameter family of lag distributions; this keeps the gate identifiable from a small proxy vector and avoids representational competition with the backbone.

== Backbone and Information-Flow Discipline

The lag-weighted context $c_(i,t)$ defined in Section~3 is concatenated with a learned entity embedding $"emb"_i$, the static feature vector $s_i$, and an optional cross-sectional macro vector. The fused stream is processed by a two-layer LSTM whose initial hidden state is a learned linear projection of $z_i$, so that the absorption representation influences both the gate and the recurrent dynamics. Crucially, the *current* sequential observation $X_(i,t)$ is intentionally *excluded* from the fused backbone input: only past observations enter through $c_(i,t)$, which closes the natural shortcut path that would otherwise let an LSTM ignore the lag gate entirely.

== Training Objective

Training minimises the sum of a task term and a proxy-reconstruction term,
$ cal(L) = cal(L)_("task") + lambda_r dot cal(L)_("recon"), $
with $cal(L)_("task")$ the mean squared error between $hat(Y)_(i,t+1)$ and $Y_(i,t+1)$ on valid post-warmup steps, and $cal(L)_("recon") = norm(hat(p)_i - p_i)_2^2$ averaged over entities. For the economics domain we use an anchor-weighted variant of $cal(L)_("recon")$ that up-weights the human-capital channel; for synthetic and energy we use the plain MSE form. Optional $omega$-entropy and $z$-anchor penalties are available in the codebase but are set to zero in the locked 20-seed runs.

== Inferential Output

Because $omega_i = omega(dot mid(|) z_i)$ depends only on the time-invariant proxy $p_i$, the per-entity expected lag $k_i^* = sum_k k dot omega_(i,k)$ is a single scalar per entity per trained model, and its cross-entity distribution is the primary object the audit protocol examines.

= Audit Protocol

The protocol assigns a *verdict* to each (domain, layer) pair rather than a single accept/reject decision for the whole model. We use four building blocks; references to the implementation are `evaluation/significance.py` for paired tests and `evaluation/stratified_kstar.py` for the permutation test.

The shared reporting workflow is summarized in Fig.~@fig:workflow. It is intentionally separated from model training: the reporting stage consumes fixed experiment outputs, computes diagnostics, and packages the resulting artifacts for venue-specific variants without changing the underlying runs.

#colimg(
  "img/workflow_overview.png",
  [Shared reporting workflow used to turn locked experiment outputs into paper-facing artifacts. The diagram summarizes the sequence from data and splits to model outputs, diagnostics, artifact assembly, and venue-specific packaging.],
) <fig:workflow>

== L0 --- Forecast Calibration

For each method we report seed-level mean and standard deviation of test~$R^2$ over 20 seeds. Pairwise comparisons against CMDL use the two-sided Wilcoxon signed-rank test on seed-level differences. L0 is reported only as calibration: a method that fails L0 may still be certified at L2.

== L1 --- Degeneracy Guard

The cross-entity standard deviation of $k_i^*$, denoted "kstar_std", is a pre-condition for any heterogeneity claim. Variants for which $"kstar_std" equiv 0$ across all seeds (No-AC-Encoder collapses $z_i$ to a constant; Uniform-Lag fixes $omega_(i,k) = 1 slash K$) are *structurally unable* to be evaluated at L2 or L3, and any positive L2 result for CMDL on the same domain is therefore attributable to the joint AC-encoder / lag-gate structure rather than to LSTM capacity.

== L2 --- Stratified $k^*$ Permutation Test

For each seed we compute the Spearman correlation between the per-entity $k_i^*$ and a train-window-only entity-level stratifier $xi_i in bb(R)$ (e.g.~mean human capital). We then permute the entity identities of $xi$ 2000 times to obtain a per-seed $p$-value. The seed-level results are aggregated into three sign-robust statistics: mean $|rho|$, share of seeds with $p < 0.05$, and Fisher-combined $p$ over the 20 seeds. A stratifier is declared *L2-supported* when the Fisher combined $p$ is below $10^(-6)$ and at least 75% of seeds reject the null.

== L3 --- Anchor-Direction Test

When ground-truth supervision is available (synthetic), we additionally report the seed-level signed Spearman $rho$ between $z_i$ and the anchor proxy, and the share of seeds with $rho > 0$. L3 is declared supported only when both quantities are stable across seeds.

== Sign-Instability Disclosure

For real domains we report only $|rho|$ and the share of seeds rejecting the null, and we explicitly disclose that signed $rho$ is unstable across seeds because of the label-permutation symmetry of $z_i$. This is a property of unsupervised mechanism discovery, not a defect of the estimator.

= Experiments

== Setup

- *Synthetic.* Generator with known $k^*(z)$ in linear and nonlinear regimes; scalar target; split by $t in [0, 30)$ with a $0.70 / 0.15 / 0.15$ train/validation/test partition.

- *Economics.* Penn World Table 11.0 [TODO-feenstra-pwt-2015]; target `ctfp` under the `effective_labor_aware` bundle; anchored year split 1980--2007 / 2008--2013 / 2014--2023.

- *Energy.* OWID-energy $times$ World Bank WGI [TODO-kaufmann-wgi-2010]; target `co2_per_unit_energy` under the `minimal` bundle; anchored year split 1996--2011 / 2012--2017 / 2018--2023.

All methods share the same data loader, splits, optimiser (Adam, learning rate $10^(-3)$, gradient clipping $1.0$), and 20 seeds $"SEEDS" = {0, dots, 19}$. The shared backbone uses $d_("model") = 64$, two LSTM layers, dropout~$0.05$, and an entity embedding of dimension~$8$; AC-GATE-specific hyperparameters are $lambda_r = 0.1$, $tau = 1.0$, $lambda = 1.0$ and the softmax variant of the gate. *No hyperparameter search* was performed across domains: values are the synthetic defaults carried over to the real panels to enforce a strict apples-to-apples comparison. All locked runs were executed on CPU.

Baselines comprise Plain~LSTM (the AC-GATE backbone with the AC encoder and lag gate removed), Grouped ARDL on `linearmodels.PanelOLS` [TODO-pesaran-smith-1995], and three structural ablations of AC-GATE itself: *No-AC-Encoder* replaces $z_i$ with a constant, *Uniform-Lag* fixes $omega_(i,k) = 1 slash K$, and *No-Recon-Reg* sets $lambda_r = 0$. TFT [TODO-lim-tft-2021] is excluded from the locked suite: at the country-year scale of our real panels (under $200$ entities, single-digit years per split) Transformer-style attention is neither necessary nor identifiable, and we emphasise interpretability rather than forecast SOTA.

== Synthetic --- Mechanism Recovery (L3 + L2)

The two synthetic regimes are shown separately in Fig.~@fig:syn-linear and Fig.~@fig:syn-nonlinear so that the linear and nonlinear recovery patterns can be read without cross-panel compression.

#colimg(
  "img/synthetic_kstar_mae_seed_distribution_linear.png",
  [Seed-level distribution of synthetic linear $k^*$ MAE. Boxes show interquartile ranges, center lines medians, whiskers Tukey ranges, and points individual seeds.],
) <fig:syn-linear>

#colimg(
  "img/synthetic_kstar_mae_seed_distribution_nonlinear.png",
  [Seed-level distribution of synthetic nonlinear $k^*$ MAE. Boxes show interquartile ranges, center lines medians, whiskers Tukey ranges, and points individual seeds.],
) <fig:syn-nonlinear>

In both regimes CMDL recovers the ground-truth heterogeneous lag structure with Spearman $rho approx 0.91$--$0.95$, while *every* degenerate ablation collapses $rho$ to exactly zero. Paired Wilcoxon tests on $k^*$~MAE return $p approx 1.91 times 10^(-6)$ for CMDL versus No-AC-Encoder, Uniform-Lag, and Plain~LSTM, in both regimes. The No-Recon-Reg variant is statistically indistinguishable from CMDL ($p = 0.498$ in linear, $p = 0.368$ in nonlinear), confirming that the reconstruction term is auxiliary and that the mechanism is carried by the AC encoder and the lag gate jointly. Plain~LSTM, despite enjoying identical backbone capacity, can only recover a weak post-hoc lag signal ($rho approx 0.34$--$0.36$) that is roughly five times farther from the ground truth than CMDL on $k^*$~MAE.

== Economics --- Penn World Table CTFP (L2 supported, L3 not, L0 weaker)

#colimg(
  "img/economics_forecast_seed_distribution.png",
  [Seed-level distribution of economics test $R^2$. Boxes show interquartile ranges, center lines medians, whiskers Tukey ranges, and points individual seeds.],
) <fig:eco-l0>

The forecast layer is *not* certified for CMDL: paired Wilcoxon yields $p = 6.3 times 10^(-5)$ versus Plain~LSTM and $p = 1.91 times 10^(-6)$ versus Uniform-Lag, both in favour of the baselines. Conversely, the L1 degeneracy guard separates CMDL (kstar_std $approx 0.17$) from the No-AC-Encoder and Uniform-Lag variants, whose per-entity $k^*$ collapses to a constant (kstar_std $equiv 0$) and which therefore cannot be evaluated at L2 by construction.

#colimg(
  "img/economics_structured_mechanism_seed_distribution.png",
  [Seed-level distribution of economics $|rho|$ across train-window stratifiers. Boxes show interquartile ranges, center lines medians, whiskers Tukey ranges, and points individual seeds.],
) <fig:eco-l2>

All three economics stratifiers are L2-supported under our acceptance criterion. The strongest alignment is with mean human capital ($|rho| = 0.371$), which is consistent with the absorptive-capacity intuition that motivates the AC encoder. The anchor-direction test (L3) fails: signed adj.~$rho = -0.109$ with only 35% of seeds positive --- exactly the label-permutation symmetry disclosed in Section~5.5. The verdict for this domain is therefore *L0 not certified, L2 certified on three independent stratifiers, L3 not claimed*.

== Energy --- OWID $times$ WGI (Dual-Decoupled Case)

#colimg(
  "img/energy_forecast_seed_distribution.png",
  [Seed-level distribution of energy test $R^2$. Boxes show interquartile ranges, center lines medians, whiskers Tukey ranges, and points individual seeds.],
) <fig:eng-l0>

#colimg(
  "img/energy_structured_mechanism_seed_distribution.png",
  [Seed-level distribution of energy $|rho|$ across train-window stratifiers. Boxes show interquartile ranges, center lines medians, whiskers Tukey ranges, and points individual seeds.],
) <fig:eng-l2>

The energy domain is the cleanest illustration of the protocol's design intent. *No* recurrent neural model achieves a positive test $R^2$: CMDL, Plain~LSTM, and both ablations all sit at $R^2 approx -0.029$, while Grouped~ARDL reaches $R^2 = 0.607$ --- unsurprising, given that the $"CO"_2$ intensity target is approximately linear in the available country-level features over a short panel. By any classical reading, the neural family has "failed" on this domain.

Yet the L2 layer tells the opposite story: CMDL's per-entity $k^*$ aligns with three governance and development stratifiers at $|rho|$ between $0.61$ and $0.74$, with $90$--$95%$ of seeds rejecting the entity-permutation null and Fisher-combined $p$ values below $10^(-76)$. The L1-degenerate ablations again return undefined results, so the alignment is fully attributable to the AC-encoder / lag-gate pair. The verdict is *L0 ruled out, L2 strongly certified*: the same model is simultaneously a poor forecaster and a structurally informative lag miner. This is the dual-decoupled case that motivates the protocol.

== Cross-Domain Verdict Matrix

#figure(
  placement: top,
  grid(
    columns: (1.2fr, 0.8fr, 0.8fr, 0.9fr, 0.8fr),
    column-gutter: 6pt,
    row-gutter: 6pt,
    [*Domain*], [*L0*], [*L1*], [*L2*], [*L3*],
    [Synthetic],
    [#verdict(rgb(215, 242, 220), [yes])],
    [#verdict(rgb(215, 242, 220), [yes])],
    [#verdict(rgb(215, 242, 220), [yes])],
    [#verdict(rgb(215, 242, 220), [yes])],
    [Economics],
    [#verdict(rgb(255, 242, 204), [n/c])],
    [#verdict(rgb(215, 242, 220), [yes])],
    [#verdict(rgb(215, 242, 220), [yes])],
    [#verdict(rgb(230, 230, 230), [n/a])],
    [Energy],
    [#verdict(rgb(248, 215, 218), [no])],
    [#verdict(rgb(215, 242, 220), [yes])],
    [#verdict(rgb(215, 242, 220), [yes])],
    [#verdict(rgb(230, 230, 230), [n/a])],
  ),
  caption: [Single-column verdict matrix summarizing L0--L3 outcomes across domains. Green denotes supported, amber not certified, red ruled out, and gray not claimed.],
) <fig:verdict>

The verdict matrix is the punchline of the experimental section: across three domains, CMDL is the *only* method whose lag heterogeneity is both non-degenerate (L1) and stratifier-aligned (L2), while no method dominates the other two layers simultaneously. The protocol thus separates "the model captures something real about lag heterogeneity" from "the model forecasts well", and shows that the two questions admit different answers on the same data.

= Discussion

= Conclusion & Future Work
