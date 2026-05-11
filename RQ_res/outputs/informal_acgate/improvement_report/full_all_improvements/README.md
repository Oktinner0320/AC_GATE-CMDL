# Informal RQ Improvement Matrix Report

## Run Status

- Matrix directory: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_matrix\full_all_improvements`
- Planned tasks: `600`
- Completed summaries: `600`
- Failed tasks: `0`
- Variants with completed runs: `18`
- Distinct seeds observed: `20`

## Best Prediction Rows

| track | variant_id | run_group | seed | test_mse | test_mae | feature_bundle | max_lag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fullspan_income | fullspan_income_k3 | Uniform Lag | 5 | 0.9406 | 0.9230783137064132 | single_fullspan_region_proxy | 3 |
| fullspan_income | fullspan_income_k3 | Plain LSTM | 7 | 0.9544 | 0.9240513718317426 | single_fullspan_region_proxy | 3 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | Plain LSTM | 7 | 0.9544 | 0.9240513718317426 | rpcyd_fullspan_region_proxy | 3 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | Plain LSTM | 3 | 0.9562 | 0.9272189764749436 | rpcyd_fullspan_region_proxy | 3 |
| fullspan_income | fullspan_income_k3 | Plain LSTM | 3 | 0.9562 | 0.9272189764749436 | single_fullspan_region_proxy | 3 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | Plain LSTM | 8 | 1.0027 | 0.9471488169261388 | rpcyd_fullspan_region_proxy | 3 |
| fullspan_income | fullspan_income_k3 | Plain LSTM | 8 | 1.0027 | 0.9471488169261388 | single_fullspan_region_proxy | 3 |
| fullspan_income | fullspan_income_k3 | Plain LSTM | 6 | 1.0102 | 0.9362591599661206 | single_fullspan_region_proxy | 3 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | Plain LSTM | 6 | 1.0102 | 0.9362591599661206 | rpcyd_fullspan_region_proxy | 3 |
| fullspan_income | fullspan_income_k2 | No AC Encoder | 13 | 1.0489 | 0.9789909370361812 | single_fullspan_region_proxy | 2 |
| fullspan_income | fullspan_income_k2 | No Recon | 13 | 1.0585 | 0.9804173140298752 | single_fullspan_region_proxy | 2 |
| fullspan_income | fullspan_income_k2 | CMDL | 13 | 1.0585 | 0.980435672260466 | single_fullspan_region_proxy | 2 |

## Mechanism-Oriented CMDL Rows

| track | variant_id | seed | proxy_perturbation | test_kstar_proxy_spearman_adjusted_rho | test_kstar_std | test_omega_entropy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| feature_subset | feature_subset_no_per_capita_overlap_k2 | 2 | none | 0.9623 | 0.3404 | 0.3462 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout030 | 2 | none | 0.9351 | 0.0024 | 0.9283 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | 11 | none | 0.9273 | 0.0983 | 0.9135 |
| fullspan_income | fullspan_income_k3 | 11 | none | 0.9273 | 0.0983 | 0.9135 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | 4 | none | 0.9143 | 0.0063 | 0.7843 |
| fullspan_income | fullspan_income_k3 | 4 | none | 0.9143 | 0.0063 | 0.7843 |
| capacity_gate_grid | grid_fullspan_d32_temp10_dropout015 | 4 | none | 0.9143 | 0.0063 | 0.7843 |
| falsification | falsification_proxy_shuffle_fullspan_k3 | 4 | shuffle | 0.9091 | 0.0071 | 0.7843 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015 | 2 | none | 0.9091 | 0.0034 | 0.9280 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015_entropy001 | 2 | none | 0.9091 | 0.0034 | 0.9280 |
| capacity_gate_grid | grid_fullspan_d16_temp20_dropout015 | 2 | none | 0.9026 | 0.0036 | 0.9851 |
| feature_subset | feature_subset_no_per_capita_overlap_k2 | 4 | none | 0.8792 | 0.0002 | 0.2727 |

## Variant Summary Files

- `all_runs.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\all_runs.csv`
- `variant_summary.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\variant_summary.csv`
- `track_summary.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\track_summary.csv`
- `mechanism_summary.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\mechanism_summary.csv`
- `failed_runs.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\failed_runs.csv`

## Built Figures

- `matrix_capacity_gate_grid`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\figures\matrix_capacity_gate_grid.png`
- `matrix_effective_samples`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\figures\matrix_effective_samples.png`
- `matrix_falsification_proxy_kstar`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\figures\matrix_falsification_proxy_kstar.png`
- `matrix_mechanism_by_variant`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\figures\matrix_mechanism_by_variant.png`
- `matrix_test_mse_by_variant`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\full_all_improvements\figures\matrix_test_mse_by_variant.png`

## Interpretation Guardrails

- Prediction wins and mechanism evidence are ranked separately.
- Seed-mean Omega and seed-std Omega are preferred over best-seed snapshots.
- Full-span income/RPCYD evidence is not full-span RYDGDP multiseq evidence.
- Structural 2006-2018 RYDGDP gaps are not imputed in the main analysis.
- Falsification variants should weaken proxy-kstar alignment before mechanism claims are treated as credible.

## Variant Summary Preview

| track | variant_id | run_group | test_mse_mean | test_mse_std | test_kstar_std_mean | run_count | seed_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015 | CMDL | 1.7266892424766396 | 0.2175492450088493 | 0.0138584343511923 | 5 | 5 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015_entropy001 | CMDL | 1.7266892424766396 | 0.2175492450088493 | 0.0138584343511923 | 5 | 5 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout030 | CMDL | 1.6728194928544131 | 0.2008443136469176 | 0.0140367349953932 | 5 | 5 |
| capacity_gate_grid | grid_fullspan_d16_temp20_dropout015 | CMDL | 1.7272141547550386 | 0.2189488527772025 | 0.0095237278713285 | 5 | 5 |
| capacity_gate_grid | grid_fullspan_d32_temp10_dropout015 | CMDL | 1.611619973859855 | 0.0953245061085672 | 0.0309560120099165 | 5 | 5 |
| capacity_gate_grid | grid_fullspan_d8_temp10_dropout005 | CMDL | 2.583534609873369 | 0.7671585624362599 | 0.0758746167718221 | 5 | 5 |
| falsification | falsification_noise_proxy_fullspan_k3 | CMDL | 1.6291693169821264 | 0.1032496674350928 | 0.0652057867057528 | 5 | 5 |
| falsification | falsification_proxy_shuffle_fullspan_k3 | CMDL | 1.636521155409179 | 0.1113303572126868 | 0.0531646356163227 | 5 | 5 |
| feature_subset | feature_subset_no_per_capita_overlap_k2 | CMDL | 27.802948563960733 | 3.14502150725142 | 0.0498491732710519 | 20 | 20 |
| feature_subset | feature_subset_no_per_capita_overlap_k2 | Plain LSTM | 27.086628884073317 | 2.595526854026406 | 0.0529831105295196 | 20 | 20 |
| feature_subset | feature_subset_pca1_overlap_k2 | CMDL | 25.954276388661636 | 2.202937014143827 | 1.839628966903226e-05 | 20 | 20 |
| feature_subset | feature_subset_pca1_overlap_k2 | Plain LSTM | 27.018723854525604 | 3.9151170546075353 | 0.00774491128942 | 20 | 20 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | CMDL | 1.5580880835646869 | 0.2182544183834278 | 0.0412626919919947 | 20 | 20 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | Plain LSTM | 1.2054945055836013 | 0.1676589980965747 | 0.0118989200698592 | 20 | 20 |
| feature_subset | feature_subset_rpcyd_overlap_k2 | CMDL | 23.464629266861007 | 1.5739125481849645 | 1.6004669922441226e-05 | 20 | 20 |
| feature_subset | feature_subset_rpcyd_overlap_k2 | Plain LSTM | 21.794293085681907 | 0.7305779187322293 | 0.2504460423228399 | 20 | 20 |
