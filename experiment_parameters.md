# Experiment Parameters Guide

This guide explains the shared parameters used by the three compact notebooks:

- `notebooks/01_synthetic_verify.ipynb`
- `notebooks/02_economics_results.ipynb`
- `notebooks/03_energy_results.ipynb`

The notebooks now follow the same workflow: configure the experiment, optionally run training, consolidate outputs, audit multi-seed stability, and visualize the final comparison.

## Execution Switches

| Parameter | Meaning | How to use |
|---|---|---|
| `RUN_CMDL` | Whether to train CMDL / AC-GATE runs. | Keep `False` when only reading existing artifacts; set `True` to regenerate CMDL outputs. |
| `RUN_BASELINE` | Whether to train the matched plain-LSTM baseline. | Use with the same `SEEDS`, data contract, and split as CMDL. |
| `RUN_GROUPED_ARDL` | Whether to run the deterministic grouped-ARDL baseline. | Used in economics and energy notebooks for lag-trend and simple baseline calibration. |
| `RUN_ABLATIONS` | Whether to run no-AC, uniform-lag, and no-reconstruction controls. | Use when testing whether AC-GATE components change the interpretable mechanism. |
| `ACTIVE_PLAN` | Output namespace under `outputs/notebook_*`. | Default is `complete_20seed`; change it to inspect older runs such as `multiseed_audit`. |

The switches default to `False` because a complete 20-seed suite is expensive. With switches off, notebooks only read artifacts already present in `OUTPUT_ROOT`.

## Seed Design

| Parameter | Meaning | Interpretation |
|---|---|---|
| `SEEDS = list(range(20))` | The planned random seeds for all stochastic models. | A claim is stronger when its sign and magnitude are stable across these 20 seeds. |
| `n_seeds` | Number of seeds actually found in output artifacts. | Always check this before interpreting a mean. If `n_seeds < 20`, the table is a partial audit. |
| `positive_seed_share` | Share of seeds with a positive adjusted mechanism metric. | `>= 2/3` is a practical candidate-positive threshold; `1.0` is much stronger. |

## Data Contracts

| Domain | Default target | Main data path | Feature/proxy contract |
|---|---|---|---|
| Synthetic | generated target | generated in memory | linear and nonlinear scenarios with known true `z`, `k*`, and `omega`. |
| Economics | `ctfp` | `data/economics/processed/economics_cleaned_long_v2.csv` | `effective_labor_aware`, anchored on effective labor with employment and human-capital auxiliary proxies. |
| Energy | `co2_per_unit_energy` | `data/energy/raw/energy_wgi_merged.csv` | `minimal`, with renewables share as sequence input and WGI governance proxies. |

## Training Parameters

| Parameter | Meaning | Notes for comparison |
|---|---|---|
| `epochs` | Maximum training epochs. | Synthetic defaults to `200`; real-data notebooks default to `120`. Early stopping can stop earlier. |
| `patience` | Early-stopping patience. | Keep the same value across CMDL and LSTM within a domain. |
| `lr` | Learning rate. | Default `1e-3`. Changing it changes both optimization and comparability. |
| `grad_clip` | Gradient clipping threshold. | Default `1.0`; helps stabilize training. |
| `grad_clip_mode` | Which parameter groups are clipped. | Economics uses `split` in the complete contract; energy uses `global`. |
| `device` | `auto`, `cpu`, or `cuda`. | Use `auto` unless debugging reproducibility or device-specific issues. |

## AC-GATE Mechanism Parameters

| Parameter | Meaning | What to inspect |
|---|---|---|
| `lambda_r` | Weight of proxy reconstruction loss. | Compare CMDL with `no_recon_regularization`; if they collapse, reconstruction is not proving necessity. |
| `temperature` | Softmax/sparsemax temperature for lag weights. | Lower values sharpen `omega`; higher values smooth it. |
| `omega_transform` | Lag-weight transform, usually `softmax`. | `sparsemax` can create sparse lag choices but changes comparability. |
| `lag_bias_strength` | Strength of lag prior/bias. | Affects the initial lag preference before data-driven adaptation. |
| `lambda_omega_entropy` | Penalty for entropy-band violations. | Used to keep lag weights neither fully uniform nor fully collapsed. |
| `omega_entropy_min/max` | Allowed entropy band. | Inspect `omega_entropy_band_violation_share`. |
| `lambda_z_anchor` | Optional direct penalty aligning latent `z` to anchor proxy. | Inspect `test_z_anchor_adjusted_rho` and `test_z_anchor_loss`. |
| `z_anchor_target_sign` | Expected sign for the optional z-anchor penalty. | Positive or negative according to the domain contract. |
| `recon_loss_mode` | Which proxies reconstruction emphasizes. | Economics complete contract uses `anchor_weighted`. |
| `anchor_recon_weight` | Extra weight for the anchor proxy under `anchor_weighted`. | Larger values prioritize the anchor over auxiliaries. |
| `reconstruction_detach` | Whether reconstruction gradients detach from the encoder path. | `False` lets reconstruction affect the encoder; `True` makes it more diagnostic/post-hoc. |

## Core Metrics

| Metric | Meaning | How to read it |
|---|---|---|
| `test_r2_mean` | Mean forecast R2 across seeds. | Forecast evidence. Do not confuse this with mechanism evidence. |
| `test_r2_std` | Forecast variation across seeds. | High std means forecast ranking is unstable. |
| `anchor_adjusted_rho_mean` | Mean sign-adjusted k*-anchor Spearman rho. | Positive means learned `k*` follows the expected anchor direction. |
| `anchor_positive_seed_share` | Share of seeds with positive anchor-adjusted rho. | This is often more informative than the mean alone. |
| `mean_proxy_adjusted_rho_mean` | Mean alignment against the proxy aggregate. | Useful when auxiliary proxies carry signal but the anchor is weak. |
| `kstar_std_mean` | Mean entity-level variation in learned effective lag. | Near zero indicates collapsed or non-heterogeneous lag behavior. |
| `entropy_mean` | Mean entropy of `omega`. | `log(max_lag)` means near-uniform lag weights; very low means near-single-lag concentration. |
| `lag_gate_sensitivity_range_mean` | Sensitivity of lag gate to latent `z`. | Positive values indicate the lag gate responds to entity-level proxy information. |
| `proxy_signal_r2_mean` | Proxy reconstruction/readout quality. | Strong signal helps interpret `z`, but it is not itself forecast evidence. |

## Ablation Interpretation

| Variant | Expected behavior if AC-GATE mechanism matters |
|---|---|
| `no_ac_encoder` | `kstar_std` should collapse or mechanism metrics become invalid. |
| `uniform_lag` | Entropy should become `log(max_lag)` and learned lag heterogeneity should disappear. |
| `no_recon_regularization` | Tests whether reconstruction regularization is necessary beyond the task loss. |

A clean mechanism claim usually needs: CMDL has non-degenerate `k*`, positive adjusted rho in most seeds, and degenerate ablations lose the mechanism structure. Forecast superiority is a separate claim.

## Where To Look In Each Notebook

1. Check the settings table in the first code cell: confirm `SEEDS`, `ACTIVE_PLAN`, data path, and run switches.
2. Run the optional training cell only when you intend to regenerate artifacts.
3. Inspect `compact_summary` first for forecast and mechanism means.
4. Inspect `result_log` for the compact yes/partial/no audit.
5. Inspect `per_proxy_audit_summary` in real-data notebooks to see which proxy is actually supporting the mechanism.
6. Inspect grouped-ARDL lag trends in economics/energy to compare learned lag behavior against deterministic lag baselines.
7. Use saved CSVs in each `comparison/` folder when writing tables for reports or papers.

## Practical Claim Strength

| Evidence pattern | Suggested wording |
|---|---|
| Forecast R2 wins and mechanism metrics are stable. | Forecast and mechanism positive case. |
| Forecast is weak but adjusted rho and ablation guards are stable. | Mechanism-alignment positive case, not forecast-superiority case. |
| Only auxiliary proxies are stable, anchor is mixed. | Partial mechanism evidence; keep the boundary explicit. |
| Metrics depend heavily on one seed. | Exploratory or boundary case; do not make a strong claim. |
