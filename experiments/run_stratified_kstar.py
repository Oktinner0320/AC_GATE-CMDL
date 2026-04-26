"""Run stratified k* analysis on locked 20-seed economics & energy outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.stratified_kstar import (
    aggregate_per_method,
    build_economics_stratifiers,
    build_energy_stratifiers,
    collect_seed_dirs,
    evaluate_method,
)

REPO = Path(__file__).resolve().parents[1]
ECON_ROOT = REPO / "outputs" / "notebook_economics" / "complete_20seed_20260426"
ENERGY_ROOT = REPO / "outputs" / "notebook_energy" / "complete_20seed_20260426"

ECON_RAW = REPO / "data" / "economics" / "processed" / "economics_cleaned_long_v2.csv"
ENERGY_RAW = REPO / "data" / "energy" / "raw" / "energy_wgi_merged.csv"


def _run(domain: str, root: Path, raw_csv: Path, build_strats):
    strats = build_strats(raw_csv)
    methods = {
        "CMDL": collect_seed_dirs(root / "cmdl", f"{domain}_cmdl_"),
        "Plain LSTM": collect_seed_dirs(root / "plain_lstm", f"{domain}_lstm_"),
        "No AC Encoder": collect_seed_dirs(
            root / "ablation", f"{domain}_ablation_no_ac_encoder_"
        ),
        "Uniform Lag": collect_seed_dirs(
            root / "ablation", f"{domain}_ablation_uniform_lag_"
        ),
        "No Recon Reg": collect_seed_dirs(
            root / "ablation", f"{domain}_ablation_no_recon_regularization_"
        ),
    }
    per_seed_frames = []
    for name, dirs in methods.items():
        if not dirs:
            print(f"[{domain}] skip {name} (no seed dirs)")
            continue
        per_seed_frames.append(
            evaluate_method(name, dirs, strats, n_perm=2000)
        )
    per_seed = pd.concat(per_seed_frames, ignore_index=True)
    agg = aggregate_per_method(per_seed)
    out_dir = root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / f"{domain}_stratified_kstar_per_seed.csv", index=False)
    agg.to_csv(out_dir / f"{domain}_stratified_kstar_aggregated.csv", index=False)
    print(f"\n=== {domain} aggregated ===")
    print(agg.to_string(index=False))
    return per_seed, agg


if __name__ == "__main__":
    _run("economics", ECON_ROOT, ECON_RAW, build_economics_stratifiers)
    _run("energy", ENERGY_ROOT, ENERGY_RAW, build_energy_stratifiers)
