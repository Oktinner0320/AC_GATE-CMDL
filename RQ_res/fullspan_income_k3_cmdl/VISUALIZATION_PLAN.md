# Visualization Plan For `fullspan_income_k3 + CMDL`

## Design Goal

The visual story should support an RQ-safe interpretation:

1. The full-span income/RPCYD setup expands the effective sample and makes the RQ experiment stable enough to report.
2. CMDL improves over a naive train-mean baseline but is not the best predictive model against stronger time-series baselines.
3. The AC-GATE lag mechanism is weak and seed-sensitive, so mechanism plots should be diagnostic rather than confirmatory.

## Priority Figures

### 1. Data Window And Effective Sample Timeline

- Data source: `data/runs_summary.csv`, `runs/seed_00/panel_audit.json`
- Plot type: horizontal timeline band
- Show: full panel 2006-2023, train-stat fit window 2006-2018, validation window 2016-2020, configured test window 2018-2023, effective prediction years 2021-2023 after `K=3` warm-up.
- Why: makes the displayed lag construction explicit and explains why only 2021-2023 appear in `predictions_all_seeds.csv`.
- Main text suitability: high.

### 2. Seed-Level Forecast Robustness

- Data source: `data/runs_summary.csv`
- Plot type: strip plot or box plot for `test_mse` and `test_mae` across 20 seeds, with mean and standard deviation annotation.
- Key values: MSE mean 1.5581, std 0.2183, range 1.1082-1.9124.
- Why: shows the full 20-seed behavior rather than a best-seed snapshot.
- Main text suitability: high.

### 3. Test-Year Forecast Trajectory

- Data source: `data/predictions_all_seeds.csv`
- Plot type: line plot by year.
- Show: one thick line for `y_true`, one line for mean `y_pred`, and a ribbon or error bars for seed/entity prediction spread.
- Key pattern: underprediction grows from 2021 to 2023; MSE rises from 0.9026 to 2.0678.
- Why: explains where the prediction error comes from.
- Main text suitability: high.

### 4. CMDL Versus Internal Baselines

- Data source: `data/baseline_summary.csv`
- Plot type: horizontal dot/bar chart of test MSE and MAE.
- Show: CMDL, train mean, persistence, panel OLS, grouped ARDL.
- Important caveat: persistence MSE is 0.1584, much lower than CMDL's 1.5581.
- Why: prevents overclaiming AC-GATE predictive dominance.
- Main text suitability: high, especially if the RQ section discusses predictive credibility.

### 5. Region-Level Error Concentration

- Data source: `data/entity_summary_seed_mean.csv`
- Plot type: sorted lollipop/bar chart, optionally paired with a Swedish-region map if a reliable shape file is available.
- Show: `entity_test_mse_mean` or `entity_test_mae_mean` by region.
- Highest-MSE examples: Värmland, Halland, Stockholm, Kalmar, Norrbotten.
- Lowest-MSE examples: Gotland, Västerbotten, Blekinge, Skåne, Östergötland.
- Caveat: do not map `y_true` as a regional outcome, because the target is entity-degenerate.
- Main text suitability: medium; stronger as supplementary context.

### 6. Lag-Gate Omega Composition By Region

- Data source: `data/entity_summary_seed_mean.csv`, optionally `data/entity_summary_by_seed.csv`
- Plot type: stacked horizontal bars for seed-mean `omega_1_mean`, `omega_2_mean`, `omega_3_mean`, sorted by `proxy_income_level_true` or `k_star_mean`.
- Show: lag weights instead of only `omega_peak_mode`.
- Key pattern: seed-entity peak-lag shares are lag 1 = 0.4667, lag 2 = 0.4024, lag 3 = 0.1310; after seed averaging every entity's modal peak becomes lag 1.
- Why: communicates seed sensitivity better than a single omega heatmap.
- Main text suitability: medium, but phrase as diagnostic.

### 7. Proxy Versus Effective Lag Diagnostic

- Data source: `data/entity_summary_seed_mean.csv` and `data/runs_summary.csv`
- Plot type: scatter plot with `proxy_income_level_true` on x-axis and `k_star_mean` on y-axis, with optional vertical error bars from `k_star_std`.
- Add annotation: mean proxy-`k*` adjusted Spearman rho = -0.0156, std = 0.6816.
- Why: directly tests whether the learned lag gate aligns with the RQ proxy.
- Main text suitability: medium-low unless framed as weak evidence.

### 8. Mechanism Stability Across Seeds

- Data source: `data/runs_summary.csv`
- Plot type: three aligned box/strip plots for `test_kstar_std`, `test_kstar_proxy_spearman_adjusted_rho`, and `test_lag_gate_sensitivity_range`.
- Key pattern: low mean `k*` variation and unstable proxy-`k*` rho.
- Why: supports the cautious mechanism interpretation.
- Main text suitability: supplementary or diagnostic figure.

### 9. Training Dynamics

- Data source: `data/history_all_seeds.csv`
- Plot type: median validation MSE by epoch with interquartile ribbon; optionally overlay training task loss.
- Why: shows early stopping and training stability across seeds.
- Main text suitability: supplementary.

### 10. Proxy Reconstruction Diagnostic

- Data source: `data/predictions_all_seeds.csv`
- Plot type: true-versus-predicted scatter or small multiples for the three proxies.
- Show: reconstruction quality of income level, recent income level, and income growth signal.
- Why: checks whether the AC encoder/reconstruction component captures the proxy space.
- Main text suitability: supplementary.

## Recommended Main-Figure Bundle

For the RQ section, use a compact four-panel figure:

- Panel A: data window and effective prediction years.
- Panel B: test-year `y_true` versus mean `y_pred` with spread.
- Panel C: CMDL versus internal baselines by MSE.
- Panel D: seed-mean omega composition by region, with a caption saying the lag-gate pattern is exploratory.

This bundle shows sample expansion, forecast behavior, predictive limitations, and mechanism diagnostics without overstating the AC-GATE result.

## Visuals To Avoid Or De-emphasize

- Avoid a choropleth of `y_true`; the target is not region-specific in this dataset.
- Avoid a best-seed omega heatmap as the main mechanism figure; it will exaggerate seed-specific lag patterns.
- Avoid leading with R2; all reported R2 values are uninformative here because of the target variance setup.
- Avoid presenting CMDL as the best predictor; persistence and simpler time-series baselines are stronger in this isolated setting.

## Data Files To Use

- `data/runs_summary.csv`: seed-level prediction and mechanism metrics.
- `data/predictions_all_seeds.csv`: year/entity predictions and residuals.
- `data/entity_summary_seed_mean.csv`: region-level seed-averaged error and lag weights.
- `data/entity_summary_by_seed.csv`: uncertainty and seed sensitivity by region.
- `data/baseline_summary.csv`: CMDL versus internal baselines.
- `data/history_all_seeds.csv`: training dynamics.