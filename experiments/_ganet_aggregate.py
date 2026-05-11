"""Aggregate GA-Net 20-seed summaries for paper tables (v2)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
SYN_GA = ROOT / "outputs/notebook_synthetic/complete_20seed/ganet"
ECO_GA = ROOT / "outputs/notebook_economics/complete_20seed/ganet"
ENE_GA = ROOT / "outputs/notebook_energy/complete_20seed/ganet"
SYN_CMDL = ROOT / "outputs/notebook_synthetic/complete_20seed_20260426/cmdl"
ECO_CMDL = ROOT / "outputs/notebook_economics/complete_20seed_20260426/cmdl"
ENE_CMDL = ROOT / "outputs/notebook_energy/complete_20seed_20260426/cmdl"
SYN_LSTM = ROOT / "outputs/notebook_synthetic/complete_20seed_20260426/plain_lstm"
ECO_LSTM = ROOT / "outputs/notebook_economics/complete_20seed_20260426/plain_lstm"
ENE_LSTM = ROOT / "outputs/notebook_energy/complete_20seed_20260426/plain_lstm"
SYN_TFT = ROOT / "outputs/notebook_synthetic/complete_20seed/tft"
ECO_TFT = ROOT / "outputs/notebook_economics/complete_20seed/tft"
ENE_TFT = ROOT / "outputs/notebook_energy/complete_20seed/tft"

SEED_RE = re.compile(r"seed(\d+)")


def _seed_of(name: str) -> Optional[int]:
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else None


def _load_dir(dir_path: Path, pattern: str) -> List[dict]:
    if not dir_path.exists():
        return []
    out = []
    for sub in sorted(dir_path.iterdir()):
        if not sub.is_dir() or pattern not in sub.name:
            continue
        sj = sub / "summary.json"
        if sj.exists():
            d = json.load(open(sj, encoding="utf-8"))
            d.setdefault("seed", _seed_of(sub.name))
            out.append(d)
    return out


def _stat(values: Iterable[Optional[float]]) -> Optional[tuple]:
    xs = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return None
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return float(np.mean(xs)), sd, len(xs)


def _fmt(s: Optional[tuple]) -> str:
    if s is None:
        return "n/a"
    m, sd, n = s
    return f"{m:.4f}+/-{sd:.4f} (n={n})"


def _seedmap(runs: List[dict], extractor) -> Dict[int, float]:
    out = {}
    for r in runs:
        s = r.get("seed")
        v = extractor(r)
        if s is not None and v is not None and not (isinstance(v, float) and math.isnan(v)):
            out[int(s)] = float(v)
    return out


def _wilcoxon(map_a: Dict[int, float], map_b: Dict[int, float], label: str) -> None:
    seeds = sorted(set(map_a) & set(map_b))
    if len(seeds) < 5:
        print(f"     wilcoxon {label}: insufficient pairs (n={len(seeds)})")
        return
    a = np.array([map_a[s] for s in seeds])
    b = np.array([map_b[s] for s in seeds])
    diff = a - b
    if np.allclose(diff, 0):
        print(f"     wilcoxon {label}: all diffs zero")
        return
    res = sp_stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    print(f"     wilcoxon {label}: stat={res.statistic:.2f} p={res.pvalue:.4g} median(a-b)={np.median(diff):+.4f} n={len(seeds)}")


def syn_get(metric: str):
    return lambda r: r.get("metrics", {}).get(metric)


def syn_cmdl_get(metric: str):
    alias = {
        "task_r2": None,
        "posthoc_kstar_mae": "kstar_mae",
        "posthoc_kstar_spearman_rho": "kstar_spearman_rho",
        "posthoc_profile_entropy_mean": "omega_entropy_mean",
        "posthoc_profile_peak_accuracy": "omega_peak_accuracy",
    }
    key = alias.get(metric, metric)
    if key is None:
        return lambda r: None
    return lambda r: r.get("metrics", {}).get(key)


def real_get(metric: str):
    return lambda r: r.get("metrics", {}).get("test", {}).get(metric)


def synthetic_section() -> None:
    print("\n=== SYNTHETIC ===")
    for scen in ("linear", "nonlinear"):
        ganet = _load_dir(SYN_GA, f"_{scen}_seed")
        cmdl = _load_dir(SYN_CMDL, f"_{scen}_seed")
        lstm = _load_dir(SYN_LSTM, f"_{scen}_seed")
        tft = _load_dir(SYN_TFT, f"_{scen}_seed")
        print(f" -- {scen} --")
        for label, runs, get in (
            ("GA-Net", ganet, syn_get),
            ("CMDL", cmdl, syn_cmdl_get),
            ("LSTM", lstm, syn_get),
            ("TFT", tft, syn_get),
        ):
            if not runs:
                print(f"  {label:8s}: (no runs)"); continue
            print(f"  {label:8s}: r2={_fmt(_stat([get('task_r2')(r) for r in runs]))}  "
                  f"kstar_mae={_fmt(_stat([get('posthoc_kstar_mae')(r) for r in runs]))}  "
                  f"rho={_fmt(_stat([get('posthoc_kstar_spearman_rho')(r) for r in runs]))}  "
                  f"H={_fmt(_stat([get('posthoc_profile_entropy_mean')(r) for r in runs]))}  "
                  f"peakacc={_fmt(_stat([get('posthoc_profile_peak_accuracy')(r) for r in runs]))}")

        _wilcoxon(_seedmap(cmdl, syn_cmdl_get('posthoc_kstar_mae')),
                  _seedmap(ganet, syn_get('posthoc_kstar_mae')),
                  f"{scen} kstar_mae (CMDL-GA)")
        _wilcoxon(_seedmap(ganet, syn_get('task_r2')),
                  _seedmap(lstm, syn_get('task_r2')),
                  f"{scen} task_r2 (GA-LSTM)")
        _wilcoxon(_seedmap(ganet, syn_get('task_r2')),
                  _seedmap(tft, syn_get('task_r2')),
                  f"{scen} task_r2 (GA-TFT)")
        _wilcoxon(_seedmap(ganet, syn_get('posthoc_kstar_mae')),
                  _seedmap(lstm, syn_get('posthoc_kstar_mae')),
                  f"{scen} kstar_mae (GA-LSTM)")


def real_section(name, ga_dir, cm_dir, ls_dir, tf_dir, ga_pat, cm_pat, ls_pat, tf_pat) -> None:
    print(f"\n=== {name.upper()} ===")
    ganet = _load_dir(ga_dir, ga_pat)
    cmdl = _load_dir(cm_dir, cm_pat)
    lstm = _load_dir(ls_dir, ls_pat)
    tft = _load_dir(tf_dir, tf_pat)
    print(f"  counts: ganet={len(ganet)} cmdl={len(cmdl)} lstm={len(lstm)} tft={len(tft)}")

    def std_for_run(r):
        m = r.get("metrics", {}).get("test", {})
        return m.get("kstar_std") if m.get("kstar_std") is not None else m.get("posthoc_kstar_std")

    for label, runs in (("GA-Net", ganet), ("CMDL", cmdl), ("LSTM", lstm), ("TFT", tft)):
        if not runs:
            print(f"  {label:8s}: (no runs)"); continue
        print(f"  {label:8s}: r2={_fmt(_stat([real_get('r2')(r) for r in runs]))}  "
              f"mae={_fmt(_stat([real_get('mae')(r) for r in runs]))}  "
              f"kstar_sigma={_fmt(_stat([std_for_run(r) for r in runs]))}")

    _wilcoxon(_seedmap(cmdl, real_get('r2')), _seedmap(ganet, real_get('r2')),
              f"{name} test_r2 (CMDL-GA)")
    _wilcoxon(_seedmap(ganet, real_get('r2')), _seedmap(lstm, real_get('r2')),
              f"{name} test_r2 (GA-LSTM)")
    _wilcoxon(_seedmap(ganet, real_get('r2')), _seedmap(tft, real_get('r2')),
              f"{name} test_r2 (GA-TFT)")


def main() -> None:
    synthetic_section()
    real_section("economics", ECO_GA, ECO_CMDL, ECO_LSTM, ECO_TFT,
                 "economics_ganet_seed", "economics_cmdl_seed",
                 "economics_lstm_seed", "economics_tft_seed")
    real_section("energy", ENE_GA, ENE_CMDL, ENE_LSTM, ENE_TFT,
                 "energy_ganet_seed", "energy_cmdl_seed",
                 "energy_lstm_seed", "energy_tft_seed")


if __name__ == "__main__":
    main()
