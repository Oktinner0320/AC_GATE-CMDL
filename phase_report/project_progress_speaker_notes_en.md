# English PPT Speaker Notes (7 Slides)

## Slide 1: Research Progress Snapshot

This slide frames the current status. The project is now an implemented and reproducible framework for entity-conditioned heterogeneous lag discovery. The important point is that AC-GATE is not positioned as a universal forecasting winner. The main research progress is that the mechanism, baselines, ablations, and locked 20-seed outputs are all in place.

## Slide 2: Mechanism Completed

Here I explain the core mechanism. Entity-level proxy variables are encoded into a latent absorption-capacity score, `z_i`. That score conditions a lag-weight distribution over historical inputs, and the expected value of that distribution gives the effective lag, `k*`. This makes the lag estimate a structural model output rather than a post-hoc explanation.

## Slide 3: Experimental Protocol Locked

This slide summarizes the experimental protocol. We evaluate the same framework across three domains: synthetic panels with known ground truth, an economics panel from PWT 11.0, and an energy-governance panel from OWID-energy and WGI. All comparisons use 20 fixed seeds and the same set of neural, econometric, and ablation baselines.

## Slide 4: Synthetic Results

The synthetic experiments provide the cleanest mechanism evidence. CMDL recovers the true lag structure with high rank correlation: 0.945 in the linear setting and 0.907 in the nonlinear setting. The lag-recovery error is also lower than the Plain LSTM and degenerate ablations. This certifies that the core architecture can recover the intended mechanism when ground truth is available.

## Slide 5: Economics Results

The economics results are more nuanced. Forecasting is not the strongest layer: CMDL has a mean test R2 of 0.054, while the best competing mean is about 0.104. However, the learned effective lag is significantly stratified by human capital and development indicators. This means the economics panel supports structured lag heterogeneity, but the direction of proxy alignment is not fully stable across seeds.

## Slide 6: Energy Results

The energy domain shows an even sharper separation between forecasting and mechanism analysis. Neural forecasting is weak, and Grouped ARDL is much stronger on R2. At the same time, the learned `k*` is strongly structured by governance indicators, especially rule of law, with an absolute rho mean of 0.735 and Fisher p around 1.8e-79. This makes energy a useful boundary case: strong lag structure, but not neural forecasting superiority.

## Slide 7: Current Interpretation

This final slide states the defensible conclusion. The current evidence supports synthetic mechanism recovery and real-data structured heterogeneous lag discovery. It does not support claiming that CMDL is universally best for prediction, and real-data directional alignment remains mixed. The paper should therefore be written as a lag-audit and mechanism-discovery contribution with explicit boundaries.