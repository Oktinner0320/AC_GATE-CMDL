"""Evaluate L2 stratifier alignment for proxy-shuffle negative controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.stratified_kstar import (
    aggregate_per_method,
    build_economics_stratifiers,
    build_energy_stratifiers,
    collect_seed_dirs,
    evaluate_method,
)
from experiments._proxy_shuffle_control import ensure_safe_negative_control_output_root


ORIGINAL_ECON_ROOT = WORKSPACE_ROOT / "outputs" / "notebook_economics" / "complete_20seed_20260426"
ORIGINAL_ENERGY_ROOT = WORKSPACE_ROOT / "outputs" / "notebook_energy" / "complete_20seed_20260426"
ECON_RAW = WORKSPACE_ROOT / "data" / "economics" / "processed" / "economics_cleaned_long_v2.csv"
ENERGY_RAW = WORKSPACE_ROOT / "data" / "energy" / "raw" / "energy_wgi_merged.csv"


def default_proxy_root() -> Path:
    return WORKSPACE_ROOT / "outputs" / "negative_controls" / f"proxy_shuffle_20seed_{date.today():%Y%m%d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proxy-shuffle L2 stratified k-star diagnostics.")
    parser.add_argument("--domain", choices=["all", "economics", "energy"], default="all")
    parser.add_argument("--proxy-root", type=str, default=str(default_proxy_root()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def _read_test_r2(seed_dirs: dict[int, Path]) -> dict[int, float]:
    values: dict[int, float] = {}
    for seed, run_dir in seed_dirs.items():
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        try:
            values[int(seed)] = float(payload["metrics"]["test"]["r2"])
        except (KeyError, TypeError, ValueError):
            values[int(seed)] = float("nan")
    return values


def _fisher_p(p_values: pd.Series) -> float:
    numeric = pd.to_numeric(p_values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size < 2:
        return float("nan")
    return float(stats.combine_pvalues(np.clip(numeric, 1e-6, 1.0))[1])


def _paired_wilcoxon_greater(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.allclose(left, right):
        return float("nan")
    try:
        return float(stats.wilcoxon(left, right, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def _bootstrap_ci(values: np.ndarray, samples: int, rng_seed: int = 0) -> tuple[float, float]:
    if values.size < 2 or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_summary(
    domain: str,
    per_seed: pd.DataFrame,
    original_r2: dict[int, float],
    proxy_r2: dict[int, float],
    bootstrap_samples: int,
) -> pd.DataFrame:
    original = per_seed[per_seed["method"] == "AC-GATE Original"].copy()
    proxy = per_seed[per_seed["method"] == "AC-GATE Proxy Shuffled"].copy()
    original = original.rename(
        columns={
            "spearman_rho": "rho_original",
            "perm_p_two_sided": "perm_p_original",
            "kstar_std": "kstar_std_original",
        }
    )
    proxy = proxy.rename(
        columns={
            "spearman_rho": "rho_proxy_shuffled",
            "perm_p_two_sided": "perm_p_proxy_shuffled",
            "kstar_std": "kstar_std_proxy_shuffled",
        }
    )
    merged = original.merge(
        proxy,
        on=["seed", "stratifier"],
        suffixes=("_original_row", "_proxy_row"),
    )
    rows = []
    for stratifier, group in merged.groupby("stratifier"):
        valid = group.dropna(subset=["rho_original", "rho_proxy_shuffled"])
        original_abs = valid["rho_original"].abs().to_numpy(dtype=float)
        proxy_abs = valid["rho_proxy_shuffled"].abs().to_numpy(dtype=float)
        delta = original_abs - proxy_abs
        ci_low, ci_high = _bootstrap_ci(delta, bootstrap_samples)
        seeds = valid["seed"].astype(int).tolist()
        original_r2_values = np.asarray([original_r2.get(seed, float("nan")) for seed in seeds], dtype=float)
        proxy_r2_values = np.asarray([proxy_r2.get(seed, float("nan")) for seed in seeds], dtype=float)
        rows.append(
            {
                "domain": domain,
                "stratifier": stratifier,
                "n_pairs": int(len(valid)),
                "original_abs_rho_mean": float(np.mean(original_abs)) if original_abs.size else float("nan"),
                "proxy_shuffled_abs_rho_mean": float(np.mean(proxy_abs)) if proxy_abs.size else float("nan"),
                "delta_abs_rho_mean": float(np.mean(delta)) if delta.size else float("nan"),
                "delta_abs_rho_ci_low": ci_low,
                "delta_abs_rho_ci_high": ci_high,
                "paired_wilcoxon_original_gt_proxy_p": _paired_wilcoxon_greater(original_abs, proxy_abs),
                "original_seed_p_lt_05_share": float((valid["perm_p_original"] < 0.05).mean()) if len(valid) else float("nan"),
                "proxy_shuffled_seed_p_lt_05_share": float((valid["perm_p_proxy_shuffled"] < 0.05).mean()) if len(valid) else float("nan"),
                "original_fisher_p": _fisher_p(valid["perm_p_original"]),
                "proxy_shuffled_fisher_p": _fisher_p(valid["perm_p_proxy_shuffled"]),
                "original_kstar_std_mean": float(valid["kstar_std_original"].mean()) if len(valid) else float("nan"),
                "proxy_shuffled_kstar_std_mean": float(valid["kstar_std_proxy_shuffled"].mean()) if len(valid) else float("nan"),
                "original_test_r2_mean": float(np.nanmean(original_r2_values)) if original_r2_values.size else float("nan"),
                "proxy_shuffled_test_r2_mean": float(np.nanmean(proxy_r2_values)) if proxy_r2_values.size else float("nan"),
                "delta_test_r2_mean": float(np.nanmean(original_r2_values - proxy_r2_values)) if original_r2_values.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _run_domain(
    domain: str,
    original_root: Path,
    proxy_root: Path,
    raw_csv: Path,
    build_strats,
    output_dir: Path,
    n_perm: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strats = build_strats(raw_csv)
    original_dirs = collect_seed_dirs(original_root / "cmdl", f"{domain}_cmdl_")
    proxy_dirs = collect_seed_dirs(proxy_root / domain / "cmdl", f"{domain}_proxy_shuffle_")
    if not original_dirs:
        raise FileNotFoundError(f"No original AC-GATE seed dirs found under {original_root / 'cmdl'}")
    if not proxy_dirs:
        raise FileNotFoundError(f"No proxy-shuffled AC-GATE seed dirs found under {proxy_root / domain / 'cmdl'}")

    per_seed = pd.concat(
        [
            evaluate_method("AC-GATE Original", original_dirs, strats, n_perm=n_perm),
            evaluate_method("AC-GATE Proxy Shuffled", proxy_dirs, strats, n_perm=n_perm),
        ],
        ignore_index=True,
    )
    aggregated = aggregate_per_method(per_seed)
    summary = _paired_summary(
        domain=domain,
        per_seed=per_seed,
        original_r2=_read_test_r2(original_dirs),
        proxy_r2=_read_test_r2(proxy_dirs),
        bootstrap_samples=bootstrap_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / f"{domain}_proxy_shuffle_stratified_kstar_per_seed.csv", index=False)
    aggregated.to_csv(output_dir / f"{domain}_proxy_shuffle_stratified_kstar_aggregated.csv", index=False)
    summary.to_csv(output_dir / f"{domain}_proxy_shuffle_summary.csv", index=False)
    print(f"\n=== {domain} proxy-shuffle paired summary ===")
    print(summary.to_string(index=False))
    return per_seed, aggregated, summary


def main() -> None:
    args = parse_args()
    proxy_root = ensure_safe_negative_control_output_root(args.proxy_root)
    output_dir = ensure_safe_negative_control_output_root(args.output_dir or (proxy_root / "comparison"))
    summaries = []
    if args.domain in {"all", "economics"}:
        _, _, economics_summary = _run_domain(
            "economics",
            ORIGINAL_ECON_ROOT,
            proxy_root,
            ECON_RAW,
            build_economics_stratifiers,
            output_dir,
            args.n_perm,
            args.bootstrap_samples,
        )
        summaries.append(economics_summary)
    if args.domain in {"all", "energy"}:
        _, _, energy_summary = _run_domain(
            "energy",
            ORIGINAL_ENERGY_ROOT,
            proxy_root,
            ENERGY_RAW,
            build_energy_stratifiers,
            output_dir,
            args.n_perm,
            args.bootstrap_samples,
        )
        summaries.append(energy_summary)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(output_dir / "proxy_shuffle_summary.csv", index=False)


if __name__ == "__main__":
    main()