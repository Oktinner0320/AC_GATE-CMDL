# fullspan_income_k3 CMDL Archive

This folder isolates the RQ primary AC-GATE/CMDL setting:

- Variant: `fullspan_income_k3`
- Run group: `CMDL`
- Feature bundle: `single_fullspan_region_proxy`
- Max lag: `K=3`
- Seeds: 20
- Source matrix: `RQ_res/outputs/informal_acgate/improvement_matrix/full_all_improvements/fullspan_income_k3`

The archive copies only the CMDL seed runs. Plain LSTM, no-AC, no-recon, uniform-lag, and model checkpoint files are intentionally excluded from this isolated analysis folder.

## Folder Contents

- `build_archive_data.py`: reproducible script that rebuilds this archive from the full matrix outputs.
- `runs/seed_00` to `runs/seed_19`: copied per-seed JSON/CSV outputs.
- `data/runs_summary.csv`: 20 seed-level CMDL summary rows.
- `data/variant_summary_cmdl.csv`: one aggregate summary row for `fullspan_income_k3 + CMDL`.
- `data/predictions_all_seeds.csv`: concatenated entity-year predictions, with residual fields.
- `data/history_all_seeds.csv`: concatenated training/validation curves.
- `data/entity_summary_by_seed.csv`: per-seed, per-entity lag/proxy/error summary.
- `data/entity_summary_seed_mean.csv`: per-entity seed-mean lag/proxy/error summary.
- `data/baseline_metrics_by_seed.csv`: CMDL and internal baseline metrics by seed.
- `data/baseline_summary.csv`: aggregate CMDL vs internal baselines.
- `data/source_manifest.json`: archive provenance.

## Data Profile

- Seed-level runs: 20
- Prediction rows: 1260 = 20 seeds x 21 regions x 3 effective test years
- Entity count: 21 Swedish regions
- Effective prediction years: 2021, 2022, 2023
- Full panel years: 2006-2023
- Train statistics window: 2006-2018
- Validation years in setup: 2016-2020
- Configured test years: 2018-2023, but `K=3` leaves 2021-2023 as effective prediction years.
- Sequence feature: `x_t`
- Proxies: `proxy_income_level`, `proxy_income_recent_level`, `proxy_income_growth_signal`
- Static variables: `static_income_trend`, `static_income_volatility`

Important caveat: `target_entity_mean_degenerate=true`. The target value is effectively common across entities within a year. Region-level variation in prediction and error therefore reflects model response to regional proxies, not region-specific target observations.

## Key Results

- Mean test MSE: 1.5581
- Test MSE std across seeds: 0.2183
- Test MSE range: 1.1082 to 1.9124
- Mean test MAE: 1.2148
- Mean best epoch: 22

Year-wise prediction error:

| Year | y_true | mean y_pred | MSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 2.3389 | 1.4126 | 0.9026 | 0.9263 |
| 2022 | 2.8844 | 1.5931 | 1.7039 | 1.2913 |
| 2023 | 3.1086 | 1.6818 | 2.0678 | 1.4268 |

The model systematically underpredicts the rising test-period target.

Internal baseline comparison:

| Model | MSE | MAE |
| --- | ---: | ---: |
| CMDL | 1.5581 | 1.2148 |
| Train mean | 5.9220 | 2.4120 |
| Persistence | 0.1584 | 0.3755 |
| Panel OLS | 1.4850 | 1.2146 |
| Grouped ARDL | 1.4440 | 1.1958 |

CMDL improves strongly over the train-mean baseline but does not beat persistence, panel OLS, or grouped ARDL in this isolated full-span setting.

## Mechanism Diagnostics

- Seed-level mean `test_kstar_std`: 0.0413
- Seed-level mean proxy-`k*` adjusted Spearman rho: -0.0156
- Rho std across seeds: 0.6816
- Mean lag-gate sensitivity range: 0.1159
- Mean omega entropy: 0.7227

Across all seed-entity rows, peak-lag shares are:

| Peak lag | Share |
| --- | ---: |
| 1 | 0.4667 |
| 2 | 0.4024 |
| 3 | 0.1310 |

After seed-averaging by entity, every entity's modal peak lag becomes lag 1. This means lag preference is seed-sensitive and should be visualized with uncertainty instead of a single best-seed heatmap.

## Interpretation For RQ

This folder is suitable for the RQ main AC-GATE/CMDL specification because it preserves the full RQ mechanism inputs while expanding the effective sample. However, the visualization and text should not claim strong proxy-conditioned lag evidence. The safer RQ statement is that full-span income/RPCYD proxy construction stabilizes the RQ experiment, while learned lag-gate heterogeneity remains exploratory.

See `VISUALIZATION_PLAN.md` for the recommended figure set.