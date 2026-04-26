"""Stratified k* analysis for real-data L2 mechanism evidence.

Tests whether the per-entity effective lag k* learned by a model is
structured along an external, domain-known stratifier (continuous, e.g.
log GDP per capita, governance quality). For each seed we compute the
Spearman correlation between the per-entity k* vector and the stratifier
vector, plus a permutation null obtained by shuffling entity labels.

CMDL is expected to produce |rho| significantly larger than degenerate
ablations (No AC Encoder / Uniform Lag), supporting the L2 claim that
the recovered heterogeneous lag is domain-aligned, not noise-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StratifierSpec:
    """Per-entity continuous stratifier built from raw panel data."""

    name: str
    series: pd.Series  # index = entity_code, value = numeric stratifier


def build_economics_stratifiers(raw_csv: Path) -> list[StratifierSpec]:
    """Return per-entity stratifiers for the economics panel."""
    df = pd.read_csv(raw_csv)
    df = df[df["year"] <= 2011]  # train-window only, no leakage
    grouped = df.groupby("entity_code")
    log_gdp_pc = np.log(grouped["rgdpna"].mean() / grouped["emp"].mean())
    hc_mean = grouped["hc"].mean()
    log_capital_pc = np.log(grouped["ck"].mean() / grouped["emp"].mean())
    return [
        StratifierSpec("log_gdp_per_worker_train", log_gdp_pc.dropna()),
        StratifierSpec("hc_mean_train", hc_mean.dropna()),
        StratifierSpec("log_capital_per_worker_train", log_capital_pc.dropna()),
    ]


def build_energy_stratifiers(raw_csv: Path) -> list[StratifierSpec]:
    df = pd.read_csv(raw_csv)
    df = df[df["year"] <= 2011]
    grouped = df.groupby("entity_code")
    log_gdp_pc = np.log(grouped["gdp"].mean() / grouped["population"].mean())
    gov_eff = grouped["government_effectiveness"].mean()
    rule_of_law = grouped["rule_of_law"].mean()
    return [
        StratifierSpec("log_gdp_per_capita_train", log_gdp_pc.dropna()),
        StratifierSpec("government_effectiveness_train", gov_eff.dropna()),
        StratifierSpec("rule_of_law_train", rule_of_law.dropna()),
    ]


def _per_entity_kstar(predictions_csv: Path) -> pd.Series:
    df = pd.read_csv(predictions_csv)
    if "k_star" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("entity_code")["k_star"].first()


def _spearman_with_perm(
    kstar: np.ndarray,
    stratifier: np.ndarray,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    rho = stats.spearmanr(kstar, stratifier).correlation
    if not np.isfinite(rho):
        return float("nan"), float("nan")
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stats.spearmanr(rng.permutation(kstar), stratifier).correlation
    p = float((np.abs(null) >= abs(rho)).mean())
    return float(rho), p


def evaluate_method(
    method_name: str,
    seed_dirs: dict[int, Path],
    stratifiers: Iterable[StratifierSpec],
    n_perm: int = 2000,
) -> pd.DataFrame:
    rows = []
    for seed, run_dir in sorted(seed_dirs.items()):
        pred_csv = run_dir / "predictions.csv"
        if not pred_csv.exists():
            continue
        kstar_series = _per_entity_kstar(pred_csv)
        if kstar_series.empty or kstar_series.std() == 0 or kstar_series.isna().all():
            for spec in stratifiers:
                rows.append(
                    {
                        "method": method_name,
                        "seed": seed,
                        "stratifier": spec.name,
                        "n_entities": 0,
                        "spearman_rho": float("nan"),
                        "perm_p_two_sided": float("nan"),
                        "kstar_std": float(kstar_series.std()) if len(kstar_series) else 0.0,
                        "degenerate": True,
                    }
                )
            continue
        rng = np.random.default_rng(seed)
        for spec in stratifiers:
            joined = pd.concat(
                [kstar_series.rename("kstar"), spec.series.rename("strat")],
                axis=1,
                join="inner",
            ).dropna()
            if len(joined) < 5:
                rows.append(
                    {
                        "method": method_name,
                        "seed": seed,
                        "stratifier": spec.name,
                        "n_entities": len(joined),
                        "spearman_rho": float("nan"),
                        "perm_p_two_sided": float("nan"),
                        "kstar_std": float(kstar_series.std()),
                        "degenerate": False,
                    }
                )
                continue
            rho, p = _spearman_with_perm(
                joined["kstar"].to_numpy(),
                joined["strat"].to_numpy(),
                n_perm=n_perm,
                rng=rng,
            )
            rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "stratifier": spec.name,
                    "n_entities": len(joined),
                    "spearman_rho": rho,
                    "perm_p_two_sided": p,
                    "kstar_std": float(kstar_series.std()),
                    "degenerate": False,
                }
            )
    return pd.DataFrame(rows)


def aggregate_per_method(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return per_seed
    rows = []
    for (method, stratifier), g in per_seed.groupby(["method", "stratifier"]):
        valid = g.dropna(subset=["spearman_rho"])
        rho = valid["spearman_rho"].to_numpy()
        p = valid["perm_p_two_sided"].to_numpy()
        rows.append(
            {
                "method": method,
                "stratifier": stratifier,
                "n_seeds_total": len(g),
                "n_seeds_valid": len(valid),
                "rho_mean": float(np.mean(rho)) if len(rho) else float("nan"),
                "rho_median": float(np.median(rho)) if len(rho) else float("nan"),
                "abs_rho_mean": float(np.mean(np.abs(rho))) if len(rho) else float("nan"),
                "share_seeds_p_lt_05": float(np.mean(p < 0.05)) if len(p) else float("nan"),
                "share_seeds_p_lt_01": float(np.mean(p < 0.01)) if len(p) else float("nan"),
                "fisher_combined_p": (
                    float(stats.combine_pvalues(np.clip(p, 1e-6, 1.0))[1])
                    if len(p) >= 2
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_seed_dirs(root: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        tail = child.name[len(prefix):]
        if not tail.startswith("seed"):
            continue
        try:
            seed = int(tail.replace("seed", ""))
        except ValueError:
            continue
        out[seed] = child
    return out
