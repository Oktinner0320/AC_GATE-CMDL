# Informal RQ Improvement Plan

This folder keeps the Informal RQ workflow isolated from the shared AC-GATE
experiments. The current implementation already includes the overlap suite, the
expanded full-span income bundle, sample audits, seed-aggregated Omega figures,
and a notebook entry point.

## Current Result Snapshot

The reliable 20-seed result set is currently:

- `RQ_res/outputs/informal_acgate/suite_region_proxy/comparison.csv`

The expanded full-span run has only passed a smoke check so far:

- `RQ_res/outputs/informal_acgate/smoke_fullspan_region_proxy/comparison.csv`

Main findings from the overlap 20-seed suite:

| Run group | Mean test MSE | Mean test MAE | Per-seed MSE wins |
| --- | ---: | ---: | ---: |
| Single CMDL | 23.46 | 4.84 | 19 / 20 |
| Uniform Lag | 28.02 | 5.27 | 1 / 20 |
| No AC Encoder | 28.86 | 5.35 | 0 / 20 |
| Multiseq CMDL | 28.99 | 5.37 | 0 / 20 |
| No Recon | 29.00 | 5.37 | 0 / 20 |
| Multiseq Plain LSTM | 29.35 | 5.41 | 0 / 20 |

Mechanism diagnostics are weak in the overlap suite:

- Multiseq CMDL adjusted proxy-kstar rho mean is about `0.035`.
- Positive seed share for adjusted proxy-kstar rho is `20%`.
- Region-level seed-mean kstar is nearly flat, about `1.41..1.48`.
- Omega varies much more across seeds than across regions.
- The overlap panel has only one post-warm-up prediction year per split.

Effective supervised sample counts after lag warm-up:

| Scenario | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| overlap, max_lag=2 | 21 | 21 | 21 |
| full-span income, max_lag=3 | 210 | 42 | 63 |
| full-span income, max_lag=2 | 231 | 42 | 63 |

Interpretation: the current overlap result does not yet support a stable
region-heterogeneous AC-conditioned lag mechanism. It mainly shows that the
overlap panel is underpowered and that the current multiseq/gate setup is not
stable enough in that setting.

## Data Boundary

Do not use KNN or any other imputation to fill the structural 2006-2018 RYDGDP
gaps in the main analysis. The support audit shows:

- `x_t` and RPCYD income are complete over 2006-2023.
- RYDGDP ratio and RYDGDP per-capita features are complete only over 2019-2023.
- The full-span main line should therefore use the stable income/RPCYD signal.
- Full-span multiseq should remain a sensitivity line unless unsupported years
  are explicitly excluded or handled by a train-window-only, support-bounded
  procedure.

## Improvement Roadmap

### Track A: Run the expanded full-span income suite

Goal: test whether AC-GATE becomes more stable once effective samples increase.

Primary run:

```powershell
python RQ_res/informal_acgate/suite.py --scenario fullspan_region_proxy --device cpu --force
```

Build figures:

```powershell
python RQ_res/informal_acgate/visualize.py --output-dir RQ_res/outputs/informal_acgate/suite_fullspan_region_proxy
```

Expected output:

- `RQ_res/outputs/informal_acgate/suite_fullspan_region_proxy/comparison.csv`
- `RQ_res/outputs/informal_acgate/figures_fullspan_region_proxy/figure_manifest.json`

Decision rule:

- Continue only if the full-span suite improves seed stability relative to the
  overlap suite.
- Do not claim mechanism support from lower MSE alone.
- Require nonzero kstar dispersion, lower seed-level Omega uncertainty, and a
  more stable proxy-kstar sign than in overlap.

### Track B: Add max_lag=2 full-span sensitivity

Goal: separate the effect of sample expansion from the effect of changing the
lag candidate set.

Run:

```powershell
python RQ_res/informal_acgate/suite.py --scenario fullspan_region_proxy --max-lag 2 --device cpu --force
```

Recommended output folder update before implementation:

- Add a suffix-aware suite output name such as `suite_fullspan_region_proxy_k2`.
- Avoid overwriting the default max_lag=3 full-span run.

Decision rule:

- If max_lag=2 and max_lag=3 behave similarly, sample expansion is the main
  driver.
- If max_lag=3 is unstable but max_lag=2 is stable, the model likely needs a
  smaller lag search space or stronger gate regularization.

### Track C: Multiseq feature subset ablation

Goal: determine whether multiseq underperformance is caused by noisy features,
collinearity, or insufficient sample size.

Candidate bundles:

- RPCYD only.
- RYDGDP ratio only, overlap years only.
- RPCYD plus RYDGDP ratio, overlap years only.
- Multiseq without per-capita columns.
- Compressed multiseq using PCA or another train-window-only projection.

Implementation notes:

- Add feature bundles in `RQ_res/informal_acgate/loader.py` only.
- Keep output folders separate, for example
  `suite_overlap_feature_subset_rpcyd_ratio`.
- Preserve the no-target and train-window-only construction boundary.

Decision rule:

- A useful multiseq subset should beat the current `Multiseq CMDL` and not lose
  badly to `Single CMDL`.
- A mechanism-positive subset should also improve kstar stability and proxy-kstar
  seed consistency.

### Track D: Reduce model capacity and regularize the gate

Goal: reduce seed-driven Omega flips and overfitting in small panels.

Recommended grid:

| Parameter | Values |
| --- | --- |
| `d_model` | 8, 16, 32 |
| `dropout` | 0.05, 0.15, 0.30 |
| `temperature` | 1.0, 1.5, 2.0 |
| `lambda_omega_entropy` | 0.0, small positive values |
| `omega_entropy_min` | optional lower-bound sensitivity |

Preferred first pass:

```powershell
python RQ_res/informal_acgate/runner.py --feature-bundle single_fullspan_region_proxy --model cmdl --d-model 16 --temperature 1.5 --device cpu --smoke
```

Decision rule:

- Prefer configurations that reduce seed-level Omega standard deviation without
  forcing uniform lag weights.
- Reject configurations that improve MSE only by collapsing kstar_std to zero.

### Track E: Add mechanism falsification checks

Goal: test whether the claimed AC/proxy mechanism is distinguishable from chance.

Recommended checks:

- Proxy permutation: shuffle entity proxy rows and rerun a small seed suite.
- Entity bootstrap: resample regions and recompute seed-mean kstar relations.
- Lag-label stability: report per-region peak-lag agreement across seeds.
- Null proxy bundle: replace proxy values with noise that preserves dimension.

Decision rule:

- Real proxy should outperform shuffled/noise proxy on proxy-kstar alignment.
- The seed-mean Omega heatmap should retain region structure after seeds are
  averaged, not only in a best-seed snapshot.

## Visualization Priorities

Primary figures should be seed-aggregated:

- `seed_mean_cmdl_omega_heatmap.png`
- `seed_std_cmdl_omega_heatmap.png`
- `year_lag_cmdl_omega_heatmap.png`
- `cmdl_kstar_distribution.png`
- `seed_mean_cmdl_proxy_kstar.png`

Best-seed Omega figures should be treated only as representative diagnostics.
They should not be used as the primary mechanism evidence.

Add the following comparison figures after the full-span 20-seed suite exists:

- Overlap vs full-span MSE/MAE by run group.
- Overlap vs full-span kstar_std and Omega seed uncertainty.
- Effective sample count bar chart next to mechanism metrics.
- Feature support heatmap for any proposed multiseq sensitivity.

## Recommended Execution Order

1. Run the full-span income 20-seed suite.
2. Rebuild full-span aggregation and figures.
3. Compare overlap and full-span metrics side by side.
4. If full-span is more stable, add max_lag=2 full-span sensitivity.
5. Add multiseq feature-subset bundles only after the full-span baseline is read.
6. Run a small low-capacity/gate-regularization grid.
7. Add proxy permutation or noise-proxy falsification checks.
8. Update the notebook with the selected final evidence path.

## Success Criteria

Treat an improved experiment as credible only if it satisfies all of the
following:

- Prediction metrics improve or remain competitive against simple baselines.
- kstar_std is nonzero but not dominated by seed noise.
- Seed-mean Omega shows region structure.
- Seed-std Omega is low enough to read the mean heatmap as stable.
- Proxy-kstar adjusted rho has a consistent sign across seeds.
- Proxy permutation/noise proxy weakens the mechanism metrics.
- The interpretation respects the data boundary: full-span income evidence is
  not the same as full-span RYDGDP multiseq evidence.

## Current Commands

Sample audit:

```powershell
python RQ_res/informal_acgate/sample_audit.py
```

Overlap 20-seed suite:

```powershell
python RQ_res/informal_acgate/suite.py --device cpu --force
```

Full-span smoke check:

```powershell
python RQ_res/informal_acgate/suite.py --scenario fullspan_region_proxy --smoke --device cpu --force --seeds 0
```

Full-span 20-seed suite:

```powershell
python RQ_res/informal_acgate/suite.py --scenario fullspan_region_proxy --device cpu --force
```

Build figures for a suite:

```powershell
python RQ_res/informal_acgate/visualize.py --output-dir RQ_res/outputs/informal_acgate/suite_region_proxy
```

Notebook entry point:

- `RQ_res/informal_rq_acgate_workflow.ipynb`