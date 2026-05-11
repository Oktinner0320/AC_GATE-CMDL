# Full Matrix Evaluation

## Status

- Matrix: `full_all_improvements`
- Planned tasks: 600
- Completed summaries: 600
- Failed tasks: 0
- Variants with completed runs: 18
- Distinct seeds observed: 20

## Best Scheme By Objective

### Best prediction overall

The strongest predictive result is the full-span income/RPCYD line with `K=3` and the plain LSTM baseline:

- Variant: `fullspan_income_k3` / `feature_subset_rpcyd_fullspan_k3`
- Run group: `Plain LSTM`
- Seeds: 20
- Test MSE: 1.2055
- Test MAE: 1.0510

This is the best raw forecasting scheme, but it is not an AC-GATE mechanism result.

### Best CMDL / AC-GATE main result

The best 20-seed CMDL result is the full-span income/RPCYD line with `K=3`:

- Variant: `fullspan_income_k3` / `feature_subset_rpcyd_fullspan_k3`
- Run group: `CMDL`
- Seeds: 20
- Test MSE: 1.5581
- Test MAE: 1.2148
- Mean test `k*` std: 0.0413
- Mean proxy-`k*` Spearman rho: -0.0156

Compared with the original overlap references, this reduces MSE by:

- 94.63% vs `reference_overlap_multiseq_k2` CMDL: 28.9917 -> 1.5581
- 93.36% vs `reference_overlap_single_k2` CMDL: 23.4646 -> 1.5581

Within the full-span CMDL family, `K=3` improves over `K=2` by 4.02% MSE and 2.08% MAE.

### Best overlap-only CMDL result

If the analysis is restricted to the 2019-2023 overlap panel, the best CMDL feature-subset result is:

- Variant: `feature_subset_rydgdp_ratio_overlap_k2`
- Run group: `CMDL`
- Seeds: 20
- Test MSE: 22.3952
- Test MAE: 4.7258

This is only a modest improvement over `reference_overlap_single_k2` CMDL: 4.56% lower MSE and 2.32% lower MAE. It is not competitive with the full-span line.

### Best capacity/gate grid result

The best 5-seed CMDL grid setting is:

- Variant: `grid_fullspan_d32_temp10_dropout015`
- Seeds: 5
- Test MSE: 1.6116
- Test MAE: 1.2392

On the matched seeds 0-4, it is slightly better than default `fullspan_income_k3` CMDL: 1.6116 vs 1.6302 MSE. The gain is small and should be treated as a screening result, not a final 20-seed winner.

## Interpretation

The main improvement comes from expanding the effective sample with the safe full-span income/RPCYD construction. The overlap-only feature engineering variants still live in the high-MSE range, which supports the diagnosis that the original overlap setup is mostly limited by sample size and test-window fragility.

The AC-GATE mechanism itself is not the predictive winner. In the full-span `K=3` setting, `Plain LSTM` has lower error than CMDL, and `Uniform Lag` / `No AC Encoder` also remain competitive. This means the best forecasting improvement should be described as a data-window and proxy-design improvement, not as evidence that learned AC/proxy-conditioned lag gates improve prediction.

Mechanism evidence is weak. The best CMDL line has low average `k*` variation and near-zero proxy-`k*` correlation. The noise and shuffle falsification variants do not degrade the result enough to support a strong proxy-to-lag mechanism claim. AC-GATE outputs can still be reported as exploratory diagnostics, but not as a robust causal or behavioral mechanism result from this run.

## Recommended Use

- For best forecasting: use `fullspan_income_k3` / `feature_subset_rpcyd_fullspan_k3` with `Plain LSTM`.
- For the main CMDL scenario: use `fullspan_income_k3` CMDL, with the caveat that prediction gains mostly come from full-span sample expansion.
- For overlap-only comparison: report `feature_subset_rydgdp_ratio_overlap_k2` as the best CMDL overlap variant, but emphasize that it remains far weaker than full-span.
- For future AC-GATE tuning: only `grid_fullspan_d32_temp10_dropout015` is worth a focused 20-seed follow-up, and only if the goal is to improve CMDL itself.

## Source Files

- `variant_summary.csv`
- `track_summary.csv`
- `mechanism_summary.csv`
- `all_runs.csv`
- `README.md`