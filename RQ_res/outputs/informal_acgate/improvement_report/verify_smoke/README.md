# Informal RQ Improvement Matrix Report

## Run Status

- Matrix directory: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_matrix\verify_smoke`
- Planned tasks: `36`
- Completed summaries: `36`
- Failed tasks: `0`
- Variants with completed runs: `18`
- Distinct seeds observed: `1`

## Best Prediction Rows

| track | variant_id | run_group | seed | test_mse | test_mae | feature_bundle | max_lag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fullspan_income | fullspan_income_k2 | No AC Encoder | 0 | 5.2800 | 2.247724311692374 | single_fullspan_region_proxy | 2 |
| fullspan_income | fullspan_income_k2 | CMDL | 0 | 5.2922 | 2.250386788968056 | single_fullspan_region_proxy | 2 |
| fullspan_income | fullspan_income_k2 | No Recon | 0 | 5.2922 | 2.250386800557848 | single_fullspan_region_proxy | 2 |
| fullspan_income | fullspan_income_k2 | Uniform Lag | 0 | 5.3003 | 2.2528406613402896 | single_fullspan_region_proxy | 2 |
| capacity_gate_grid | grid_fullspan_d8_temp10_dropout005 | CMDL | 0 | 8.0881 | 2.7908182385421934 | single_fullspan_region_proxy | 3 |
| fullspan_income | fullspan_income_k3 | Plain LSTM | 0 | 9.2225 | 2.981170939665938 | single_fullspan_region_proxy | 3 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | Plain LSTM | 0 | 9.2225 | 2.981170939665938 | rpcyd_fullspan_region_proxy | 3 |
| fullspan_income | fullspan_income_k2 | Plain LSTM | 0 | 9.3159 | 2.998179129960518 | single_fullspan_region_proxy | 2 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout030 | CMDL | 0 | 10.5552 | 3.208977557245701 | single_fullspan_region_proxy | 3 |
| capacity_gate_grid | grid_fullspan_d16_temp20_dropout015 | CMDL | 0 | 10.5554 | 3.2089913222524853 | single_fullspan_region_proxy | 3 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015_entropy001 | CMDL | 0 | 10.5568 | 3.209254080695765 | single_fullspan_region_proxy | 3 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015 | CMDL | 0 | 10.5568 | 3.209254080695765 | single_fullspan_region_proxy | 3 |

## Mechanism-Oriented CMDL Rows

| track | variant_id | seed | proxy_perturbation | test_kstar_proxy_spearman_adjusted_rho | test_kstar_std | test_omega_entropy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout030 | 0 | none | 0.3701 | 0.0282 | 0.9657 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015 | 0 | none | 0.3701 | 0.0282 | 0.9657 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015_entropy001 | 0 | none | 0.3701 | 0.0282 | 0.9657 |
| capacity_gate_grid | grid_fullspan_d16_temp20_dropout015 | 0 | none | 0.3701 | 0.0208 | 1.0182 |
| falsification | falsification_proxy_shuffle_fullspan_k3 | 0 | shuffle | 0.3286 | 0.1928 | 0.8328 |
| capacity_gate_grid | grid_fullspan_d32_temp10_dropout015 | 0 | none | 0.3286 | 0.1914 | 0.8340 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | 0 | none | 0.3286 | 0.1914 | 0.8340 |
| fullspan_income | fullspan_income_k3 | 0 | none | 0.3286 | 0.1914 | 0.8340 |
| capacity_gate_grid | grid_fullspan_d8_temp10_dropout005 | 0 | none | 0.3208 | 0.0176 | 0.8765 |
| feature_subset | feature_subset_no_per_capita_overlap_k2 | 0 | none | -0.0000 | 0.0000 | 0.4742 |
| reference | reference_overlap_multiseq_k2 | 0 | none | -0.0000 | 0.0000 | 0.4742 |
| feature_subset | feature_subset_rpcyd_ratio_overlap_k2 | 0 | none | -0.0000 | 0.0000 | 0.4759 |

## Variant Summary Files

- `all_runs.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\all_runs.csv`
- `variant_summary.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\variant_summary.csv`
- `track_summary.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\track_summary.csv`
- `mechanism_summary.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\mechanism_summary.csv`
- `failed_runs.csv`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\failed_runs.csv`

## Built Figures

- `matrix_capacity_gate_grid`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\figures\matrix_capacity_gate_grid.png`
- `matrix_effective_samples`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\figures\matrix_effective_samples.png`
- `matrix_falsification_proxy_kstar`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\figures\matrix_falsification_proxy_kstar.png`
- `matrix_mechanism_by_variant`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\figures\matrix_mechanism_by_variant.png`
- `matrix_test_mse_by_variant`: `C:\DevSpace\PyDevspace\CMDL\RQ_res\outputs\informal_acgate\improvement_report\verify_smoke\figures\matrix_test_mse_by_variant.png`

## Interpretation Guardrails

- Prediction wins and mechanism evidence are ranked separately.
- Seed-mean Omega and seed-std Omega are preferred over best-seed snapshots.
- Full-span income/RPCYD evidence is not full-span RYDGDP multiseq evidence.
- Structural 2006-2018 RYDGDP gaps are not imputed in the main analysis.
- Falsification variants should weaken proxy-kstar alignment before mechanism claims are treated as credible.

## Variant Summary Preview

| track | variant_id | run_group | test_mse_mean | test_mse_std | test_kstar_std_mean | run_count | seed_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015 | CMDL | 10.556847164581928 | nan | 0.0281887157822188 | 1 | 1 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout015_entropy001 | CMDL | 10.556847164581928 | nan | 0.0281887157822188 | 1 | 1 |
| capacity_gate_grid | grid_fullspan_d16_temp15_dropout030 | CMDL | 10.555198637880611 | nan | 0.0281928137923486 | 1 | 1 |
| capacity_gate_grid | grid_fullspan_d16_temp20_dropout015 | CMDL | 10.555379526805568 | nan | 0.0207796540426141 | 1 | 1 |
| capacity_gate_grid | grid_fullspan_d32_temp10_dropout015 | CMDL | 14.17099744268876 | nan | 0.1913937116025758 | 1 | 1 |
| capacity_gate_grid | grid_fullspan_d8_temp10_dropout005 | CMDL | 8.088106037240621 | nan | 0.0176411902836445 | 1 | 1 |
| falsification | falsification_noise_proxy_fullspan_k3 | CMDL | 14.167635875140505 | nan | 0.2011043883952872 | 1 | 1 |
| falsification | falsification_proxy_shuffle_fullspan_k3 | CMDL | 14.17047977536082 | nan | 0.1928239393867212 | 1 | 1 |
| feature_subset | feature_subset_no_per_capita_overlap_k2 | CMDL | 37.07891831995231 | nan | 4.531091031977211e-06 | 1 | 1 |
| feature_subset | feature_subset_no_per_capita_overlap_k2 | Plain LSTM | 41.844953860625175 | nan | 0.1702420890369172 | 1 | 1 |
| feature_subset | feature_subset_pca1_overlap_k2 | CMDL | 29.863344229666048 | nan | 4.3773763224476135e-05 | 1 | 1 |
| feature_subset | feature_subset_pca1_overlap_k2 | Plain LSTM | 40.53332151912 | nan | 0.0 | 1 | 1 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | CMDL | 14.171819935976757 | nan | 0.1913565399802221 | 1 | 1 |
| feature_subset | feature_subset_rpcyd_fullspan_k3 | Plain LSTM | 9.222490486370363 | nan | 0.0 | 1 | 1 |
| feature_subset | feature_subset_rpcyd_overlap_k2 | CMDL | 29.59657159822333 | nan | 2.881859831938404e-05 | 1 | 1 |
| feature_subset | feature_subset_rpcyd_overlap_k2 | Plain LSTM | 38.37870179905864 | nan | 0.0 | 1 | 1 |
