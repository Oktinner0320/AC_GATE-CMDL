# Informal RQ AC-GATE Experiments

This folder is an isolated experiment area for the Informal RQ described in
`data/Informal/README.md`. It reuses the shared AC-GATE model and diagnostics,
but all new code and default outputs live under `RQ_res`.

## Layout

| File | Role |
| --- | --- |
| `loader.py` | Converts curated Informal CSVs into `X_it`, `p_i`, `s_i`, `Y_it` tensors. |
| `runner.py` | Runs one CMDL, plain LSTM, or CMDL ablation experiment. |
| `suite.py` | Runs the isolated 20-seed RQ suite with region-varying proxy bundles. |
| `aggregate.py` | Builds a compact `comparison.csv` from run-level `summary.json` files. |
| `aggregate_multi.py` | Aggregates one-shot matrix results across variant directories. |
| `experiment_matrix.py` | Defines all one-shot improvement variants and runner arguments. |
| `falsification.py` | Applies proxy shuffle/noise perturbations for mechanism checks. |
| `report_improvements.py` | Builds the matrix-level Markdown report and figures. |
| `suite_all_improvements.py` | Runs the one-shot improvement matrix with dry-run, smoke, resume, and report modes. |
| `visualize.py` | Builds seed summaries and RQ figures from the suite outputs. |
| `sample_audit.py` | Audits feature-year support and effective sample counts for overlap/full-span scenarios. |
| `../informal_rq_acgate_workflow.ipynb` | Notebook entry point for running, aggregating, plotting, and documenting the explicit lag construction. |

## Default Design

The default main experimental panel is `multiseq_overlap_region_proxy`, which reads
`data/Informal/processed/acgate_inputs/informal_acgate_multiseq_overlap_ready.csv`.
It covers 21 regions over 2019-2023, so the isolated runner defaults to:

- `max_lag=2`
- train evaluation years: 2021 after 2019-2020 warm-up
- validation evaluation year: 2022 after 2020-2021 warm-up
- test evaluation year: 2023 after 2021-2022 warm-up

This keeps the README plan's no-imputation overlap setup while satisfying the
AC-GATE lag-window contract.

The original workbook `proxy_*` and `static_*` columns are cross-sectionally
degenerate after entity aggregation. The current RQ suite therefore uses
region-varying proxy/static signals derived from train-window formal sequence
summaries:

- `proxy_formal_income_level`
- `proxy_formalization_ratio`
- `proxy_formal_capacity_signal`
- `static_formal_income_trend`
- `static_formal_income_volatility`

These derived signals use only the 2019-2021 reference window by default and do
not use `y_t`. The loader records this in `summary.json` under
`data.audit.proxy_construction`.

## Expanded Full-Span Design

The expanded effective-sample scenario is `single_fullspan_region_proxy`. It
uses `data/Informal/processed/acgate_inputs/informal_acgate_single_feature_ready.csv`
over 2006-2023 and derives region-varying AC/proxy signals from train-window
income/RPCYD summaries only:

- `proxy_income_level`
- `proxy_income_recent_level`
- `proxy_income_growth_signal`
- `static_income_trend`
- `static_income_volatility`

The default expanded split is train 2006-2018, validation 2019-2020, and test
2021-2023 with `max_lag=3`. Validation and test slices keep lag-context overlap,
so effective supervised samples are counted after warm-up as
`n_entities * (split_year_count - max_lag)`.

This full-span line increases statistical power, but it is narrower than the
dense multiseq overlap line: it studies long-horizon income/RPCYD lag structure,
not the full RYDGDP formalization bundle.

## Commands

Fast wiring check:

```powershell
python RQ_res/informal_acgate/suite.py --smoke --device cpu --force --seeds 0
```

Full isolated 20-seed suite:

```powershell
python RQ_res/informal_acgate/suite.py --device cpu --force
```

Audit sample expansion and feature-year support:

```powershell
python RQ_res/informal_acgate/sample_audit.py
```

Expanded full-span 20-seed suite:

```powershell
python RQ_res/informal_acgate/suite.py --scenario fullspan_region_proxy --device cpu --force
```

Expanded full-span smoke check:

```powershell
python RQ_res/informal_acgate/suite.py --scenario fullspan_region_proxy --smoke --device cpu --force --seeds 0
```

Single multiseq CMDL run:

```powershell
python RQ_res/informal_acgate/runner.py --feature-bundle multiseq_overlap_region_proxy --model cmdl --device cpu
```

Aggregate existing runs:

```powershell
python RQ_res/informal_acgate/aggregate.py --output-dir RQ_res/outputs/informal_acgate/suite_region_proxy
```

Build visualizations:

```powershell
python RQ_res/informal_acgate/visualize.py --output-dir RQ_res/outputs/informal_acgate/suite_region_proxy
```

For full-span outputs, the visualization script writes to
`RQ_res/outputs/informal_acgate/figures_fullspan_region_proxy` by default:

```powershell
python RQ_res/informal_acgate/visualize.py --output-dir RQ_res/outputs/informal_acgate/suite_fullspan_region_proxy
```

Dry-run the one-shot improvement matrix:

```powershell
python RQ_res/informal_acgate/suite_all_improvements.py --dry-run --device cpu
```

Smoke-test all one-shot variants:

```powershell
python RQ_res/informal_acgate/suite_all_improvements.py --smoke --device cpu --force --seeds 0 --screening-seeds 0
```

Run the default one-shot matrix with resume behavior:

```powershell
python RQ_res/informal_acgate/suite_all_improvements.py --device cpu --resume
```

Rebuild a matrix report:

```powershell
python RQ_res/informal_acgate/report_improvements.py --matrix-dir RQ_res/outputs/informal_acgate/improvement_matrix/default
```

Build matrix-level visualizations directly from aggregated runs:

```powershell
python RQ_res/informal_acgate/visualize.py --matrix-runs-csv RQ_res/outputs/informal_acgate/improvement_report/default/all_runs.csv
```

The notebook workflow is available at `RQ_res/informal_rq_acgate_workflow.ipynb`.
It contains the same runner and plotting interfaces plus a markdown explanation
of how the displayed effective lag `k_star = sum_k k * omega_k` is constructed
from explicit lag windows.

## Omega Visualization

The best-seed Omega heatmap is retained as a representative diagnostic only. It
selects the lowest-test-MSE CMDL run and is not seed averaged. The primary
mechanism figures are now seed-aggregated:

- `seed_mean_cmdl_omega_heatmap.png`
- `seed_std_cmdl_omega_heatmap.png`
- `year_lag_cmdl_omega_heatmap.png`
- `cmdl_kstar_distribution.png`
- `seed_mean_cmdl_proxy_kstar.png`

These plots summarize lag-gate behavior across seeds and, for full-span tests,
across multiple prediction years.

## Current Data Caveat

The full-span multiseq file still has structural RYDGDP feature missingness
before 2019, so the default multiseq RQ suite intentionally stays on the
no-imputation overlap panel. The expanded full-span main line uses the stable
single income/RPCYD signal instead. If KNN or other imputation is added later,
it should be fit on train-window information only, should not use `y_t`, and
should not fill unsupported 2006-2018 RYDGDP feature-years.
