# Project Progress PPT in English (7 Slides)

## PPT Generation Prompt (not counted as a slide)

Create a concise 16:9 research progress deck from Slide 1 to Slide 7 below. Use an academic but clean visual style. Keep each slide short, with 2-4 key points and minimal prose. Emphasize current research progress, completed experimental evidence, and brief analysis of what the data supports.

Use the following existing assets exactly where specified:

- Slide 2: mechanism workflow figure `workflow_overview.png`
- Slide 4: synthetic result figure `synthetic_kstar_mae_seed_distribution.png`
- Slide 5: economics figures `economics_structured_mechanism_seed_distribution.png` and `economics_forecast_seed_distribution.png`
- Slide 6: energy figures `energy_structured_mechanism_seed_distribution.png` and `energy_forecast_seed_distribution.png`

Reference tables:

- `synthetic_main_table.csv`
- `realdata_forecast_table.csv`
- `baseline_compact_table.csv`
- `economics_stratified_main_table.csv`
- `energy_stratified_main_table.csv`
- `verdict_matrix.csv`

---

# Slide 1: Review Progress Snapshot
- Follow the suggestion from teacher's guide and the feadback in Overleaf
- Using Boolean sentences to search papers connected to our topic
- Due to main searching string most cover on traditional methods like ARDL and VAR, we have to add another searching string to cover more papers on AI methods
- Now we finish the abstract screening which have 24 papers left, and begin to read the full text of these papers.
---
## Slide 2: Dev and Research Progress Snapshot

**AC-GATE / CMDL: entity-conditioned heterogeneous lag discovery**

- Mechanism, loaders, baselines, ablations, and reporting are implemented
- Locked 20-seed outputs are available for synthetic, economics, and energy domains
- The main claim is mechanism discovery, not universal forecasting superiority
- Current evidence supports structured heterogeneous lag discovery

---

## Slide 3: Mechanism Completed

![AC-GATE workflow](workflow_overview.png)

- Entity-level proxies are encoded into latent absorption capacity `z_i`
- `z_i` conditions a lag-weight distribution `omega(k | z_i)`
- The expected lag `k*` gives an interpretable entity-level delay
- The model links proxy structure, temporal dynamics, and forecast calibration

---

## Slide 4: Experimental Protocol Locked

**Three domains under one 20-seed protocol**

- Synthetic panels: ground-truth `k*` for mechanism recovery tests
- Economics: PWT 11.0, target `ctfp`, effective-labor-aware feature bundle
- Energy: OWID-energy × WGI, target `co2_per_unit_energy`
- Baselines: Plain LSTM, Grouped ARDL, No AC Encoder, Uniform Lag, No Recon Reg

---

## Slide 5: Synthetic Results

![Synthetic kstar distribution](synthetic_kstar_mae_seed_distribution.png)

- Linear setting: `k*` Spearman rho = 0.945, `k*` MAE = 1.159
- Nonlinear setting: `k*` Spearman rho = 0.907, `k*` MAE = 1.467
- Removing AC encoding or adaptive lag gating collapses lag recovery
- Analysis: synthetic results certify the core mechanism

---

## Slide 6: Economics Results

![Economics mechanism](economics_structured_mechanism_seed_distribution.png)
![Economics forecast](economics_forecast_seed_distribution.png)

- CMDL test R2 = 0.054; best competing R2 = 0.104
- Stratified `k*` is significant for human capital and development indicators
- Human-capital stratifier: abs rho mean = 0.371, Fisher p = 1.0e-46
- Analysis: structured lag evidence is present, but directional alignment is mixed

---

## Slide 7: Energy Results

![Energy mechanism](energy_structured_mechanism_seed_distribution.png)
![Energy forecast](energy_forecast_seed_distribution.png)

- Neural forecast R2 is near -0.029; Grouped ARDL reaches 0.607
- Stratified `k*` is very strong for governance indicators
- Rule of law: abs rho mean = 0.735, Fisher p = 1.8e-79
- Analysis: strong mechanism structure, clear forecasting boundary

---

## Slide 8: Current Interpretation

**What the current data supports**

- Certified: synthetic mechanism recovery and real-data structured lag discovery
- Mixed: real-data directional proxy alignment across random seeds
- Not claimed: CMDL as a universal forecasting SOTA model
- Next step: write the paper around lag audit, mechanism evidence, and honest boundaries
