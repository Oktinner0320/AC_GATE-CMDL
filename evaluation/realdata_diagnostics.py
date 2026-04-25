"""Shared diagnostics for real-data CMDL and baseline runs.

The helpers in this module keep economics and energy mechanism diagnostics on
the same anchor-aware and per-proxy convention.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from evaluation.metrics import compute_r2, compute_spearman


METRIC_EPS = 1e-10


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return np.asarray(array, dtype=np.float64)


def _population_std(value: Any) -> float:
    array = _to_numpy(value).reshape(-1)
    if array.size <= 1:
        return 0.0
    return float(np.std(array))


def _slug(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return normalized or "proxy"


def _safe_spearman(left: Any, right: Any) -> tuple[float, float, bool]:
    left_array = _to_numpy(left).reshape(-1)
    right_array = _to_numpy(right).reshape(-1)
    valid = (
        left_array.shape == right_array.shape
        and left_array.size > 1
        and np.std(left_array) > METRIC_EPS
        and np.std(right_array) > METRIC_EPS
    )
    if not valid:
        return float("nan"), float("nan"), False
    rho, p_value = compute_spearman(left_array, right_array)
    return float(rho), float(p_value), True


def infer_proxy_metadata(metadata: dict[str, Any] | None, n_proxies: int) -> dict[str, Any]:
    """Return normalized proxy anchor metadata with backward-compatible defaults."""

    source = metadata or {}
    proxy_columns = list(source.get("proxy_columns") or [f"proxy_{index + 1}" for index in range(n_proxies)])
    if len(proxy_columns) != n_proxies:
        proxy_columns = [f"proxy_{index + 1}" for index in range(n_proxies)]

    anchor_proxy_index = int(source.get("anchor_proxy_index", 0))
    anchor_proxy_index = min(max(anchor_proxy_index, 0), max(n_proxies - 1, 0))
    anchor_proxy_name = str(source.get("anchor_proxy_name", proxy_columns[anchor_proxy_index]))
    anchor_expected_sign = float(source.get("anchor_expected_sign", -1.0))
    if anchor_expected_sign == 0.0:
        anchor_expected_sign = -1.0

    proxy_expected_signs = source.get("proxy_expected_signs")
    if not isinstance(proxy_expected_signs, list) or len(proxy_expected_signs) != n_proxies:
        proxy_expected_signs = [anchor_expected_sign for _ in range(n_proxies)]
    proxy_expected_signs = [float(sign if float(sign) != 0.0 else anchor_expected_sign) for sign in proxy_expected_signs]

    auxiliary_proxy_names = list(source.get("auxiliary_proxy_names") or [])
    if not auxiliary_proxy_names:
        auxiliary_proxy_names = [name for index, name in enumerate(proxy_columns) if index != anchor_proxy_index]

    return {
        "proxy_columns": proxy_columns,
        "anchor_proxy_index": anchor_proxy_index,
        "anchor_proxy_name": anchor_proxy_name,
        "anchor_expected_sign": anchor_expected_sign,
        "proxy_expected_signs": proxy_expected_signs,
        "auxiliary_proxy_names": auxiliary_proxy_names,
        "proxy_aggregate_name": str(source.get("proxy_aggregate_name", "proxy_mean")),
    }


def proxy_metadata_payload(metadata: dict[str, Any] | None, n_proxies: int) -> dict[str, Any]:
    """Build JSON-safe proxy metadata for summary payloads."""

    normalized = infer_proxy_metadata(metadata, n_proxies)
    return {
        "anchor_proxy_name": normalized["anchor_proxy_name"],
        "anchor_proxy_index": int(normalized["anchor_proxy_index"]),
        "anchor_expected_sign": float(normalized["anchor_expected_sign"]),
        "auxiliary_proxy_names": list(normalized["auxiliary_proxy_names"]),
        "proxy_expected_signs": [float(value) for value in normalized["proxy_expected_signs"]],
        "proxy_aggregate_name": normalized["proxy_aggregate_name"],
    }


def compute_proxy_alignment_metrics(
    effective_kstar: Any,
    proxies: Any,
    metadata: dict[str, Any] | None,
    prefix: str = "kstar",
) -> dict[str, float]:
    """Compute anchor, aggregate, and per-proxy k* alignment metrics."""

    kstar_array = _to_numpy(effective_kstar).reshape(-1)
    proxy_array = _to_numpy(proxies)
    if proxy_array.ndim == 1:
        proxy_array = proxy_array.reshape(-1, 1)
    if proxy_array.ndim != 2:
        raise ValueError(f"Expected proxies with shape [N, M], got {proxy_array.shape}")
    if proxy_array.shape[0] != kstar_array.shape[0]:
        raise ValueError("effective_kstar and proxies must agree on entity count")

    n_proxies = proxy_array.shape[1]
    proxy_meta = infer_proxy_metadata(metadata, n_proxies)
    anchor_index = int(proxy_meta["anchor_proxy_index"])
    anchor_sign = float(proxy_meta["anchor_expected_sign"])
    metrics: dict[str, float] = {
        f"{prefix}_std": _population_std(kstar_array),
        f"{prefix}_anchor_proxy_index": float(anchor_index),
        f"{prefix}_anchor_expected_sign": anchor_sign,
    }

    anchor_rho, anchor_p, anchor_valid = _safe_spearman(kstar_array, proxy_array[:, anchor_index])
    metrics.update(
        {
            f"{prefix}_proxy_spearman_rho": anchor_rho,
            f"{prefix}_proxy_spearman_p": anchor_p,
            f"{prefix}_proxy_spearman_adjusted_rho": anchor_sign * anchor_rho if anchor_valid else float("nan"),
            f"{prefix}_proxy_metric_valid": float(anchor_valid),
        }
    )

    mean_signal = np.mean(proxy_array, axis=1)
    mean_rho, mean_p, mean_valid = _safe_spearman(kstar_array, mean_signal)
    metrics.update(
        {
            f"{prefix}_proxy_mean_spearman_rho": mean_rho,
            f"{prefix}_proxy_mean_spearman_p": mean_p,
            f"{prefix}_proxy_mean_spearman_adjusted_rho": anchor_sign * mean_rho if mean_valid else float("nan"),
            f"{prefix}_proxy_mean_metric_valid": float(mean_valid),
        }
    )

    for proxy_index, proxy_name in enumerate(proxy_meta["proxy_columns"]):
        proxy_sign = float(proxy_meta["proxy_expected_signs"][proxy_index])
        rho, p_value, valid = _safe_spearman(kstar_array, proxy_array[:, proxy_index])
        slug = _slug(str(proxy_name))
        common = {
            f"{prefix}_proxy_{proxy_index + 1}_spearman_rho": rho,
            f"{prefix}_proxy_{proxy_index + 1}_spearman_p": p_value,
            f"{prefix}_proxy_{proxy_index + 1}_spearman_adjusted_rho": proxy_sign * rho if valid else float("nan"),
            f"{prefix}_proxy_{proxy_index + 1}_metric_valid": float(valid),
        }
        named = {
            f"{prefix}_{slug}_spearman_rho": rho,
            f"{prefix}_{slug}_spearman_adjusted_rho": proxy_sign * rho if valid else float("nan"),
        }
        metrics.update(common)
        metrics.update(named)

    return metrics


def compute_proxy_reconstruction_metrics(
    proxy_predictions: Any,
    proxies: Any,
    metadata: dict[str, Any] | None,
    metric_valid: bool = True,
) -> dict[str, float]:
    """Compute aggregate, anchor, and per-proxy reconstruction R2 metrics."""

    pred_array = _to_numpy(proxy_predictions)
    true_array = _to_numpy(proxies)
    if pred_array.ndim == 1:
        pred_array = pred_array.reshape(-1, 1)
    if true_array.ndim == 1:
        true_array = true_array.reshape(-1, 1)
    if pred_array.shape != true_array.shape:
        raise ValueError(f"proxy_predictions and proxies must match; got {pred_array.shape} vs {true_array.shape}")

    n_proxies = true_array.shape[1]
    proxy_meta = infer_proxy_metadata(metadata, n_proxies)
    anchor_index = int(proxy_meta["anchor_proxy_index"])
    if not metric_valid:
        nan_metrics = {
            "proxy_recon_r2": float("nan"),
            "proxy_anchor_recon_r2": float("nan"),
            "proxy_metric_valid": 0.0,
        }
        for proxy_index, proxy_name in enumerate(proxy_meta["proxy_columns"]):
            slug = _slug(str(proxy_name))
            nan_metrics[f"proxy_{proxy_index + 1}_recon_r2"] = float("nan")
            nan_metrics[f"proxy_{slug}_recon_r2"] = float("nan")
        return nan_metrics

    metrics: dict[str, float] = {
        "proxy_recon_r2": float(compute_r2(pred_array, true_array)),
        "proxy_anchor_recon_r2": float(compute_r2(pred_array[:, anchor_index], true_array[:, anchor_index])),
        "proxy_metric_valid": 1.0,
    }
    for proxy_index, proxy_name in enumerate(proxy_meta["proxy_columns"]):
        slug = _slug(str(proxy_name))
        score = float(compute_r2(pred_array[:, proxy_index], true_array[:, proxy_index]))
        metrics[f"proxy_{proxy_index + 1}_recon_r2"] = score
        metrics[f"proxy_{slug}_recon_r2"] = score
    return metrics


def compute_latent_proxy_metrics(z_values: Any, proxies: Any, metadata: dict[str, Any] | None) -> dict[str, float]:
    """Compute z spread and z-proxy alignment diagnostics."""

    z_array = _to_numpy(z_values).reshape(-1)
    proxy_array = _to_numpy(proxies)
    if proxy_array.ndim == 1:
        proxy_array = proxy_array.reshape(-1, 1)
    proxy_meta = infer_proxy_metadata(metadata, proxy_array.shape[1])
    anchor_index = int(proxy_meta["anchor_proxy_index"])
    anchor_sign = float(proxy_meta["anchor_expected_sign"])
    rho, p_value, valid = _safe_spearman(z_array, proxy_array[:, anchor_index])
    return {
        "z_std": _population_std(z_array),
        "z_proxy_spearman_rho": rho,
        "z_proxy_spearman_p": p_value,
        "z_proxy_spearman_adjusted_rho": anchor_sign * rho if valid else float("nan"),
        "z_proxy_metric_valid": float(valid),
    }


def compute_omega_metrics(omega: Any) -> dict[str, float]:
    """Compute omega entropy, peak concentration, and peak histogram metrics."""

    omega_array = _to_numpy(omega)
    if omega_array.ndim != 2:
        raise ValueError(f"Expected omega with shape [N, K], got {omega_array.shape}")
    clipped = np.clip(omega_array, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    peaks = np.argmax(omega_array, axis=1)
    counts = np.bincount(peaks, minlength=omega_array.shape[1]).astype(np.float64)
    shares = counts / max(float(omega_array.shape[0]), 1.0)
    metrics: dict[str, float] = {
        "omega_entropy_mean": float(np.mean(entropy)),
        "omega_entropy_std": float(np.std(entropy)) if entropy.size > 1 else 0.0,
        "omega_top1_share": float(np.max(shares)) if shares.size else 0.0,
    }
    for lag_index, share in enumerate(shares, start=1):
        metrics[f"omega_peak_share_{lag_index}"] = float(share)
    return metrics


def compute_kstar_quantile_metrics(
    effective_kstar: Any,
    proxies: Any,
    metadata: dict[str, Any] | None,
    prefix: str = "kstar",
) -> dict[str, float]:
    """Summarize effective k* by anchor proxy quartile."""

    kstar_array = _to_numpy(effective_kstar).reshape(-1)
    proxy_array = _to_numpy(proxies)
    if proxy_array.ndim == 1:
        proxy_array = proxy_array.reshape(-1, 1)
    proxy_meta = infer_proxy_metadata(metadata, proxy_array.shape[1])
    anchor = proxy_array[:, int(proxy_meta["anchor_proxy_index"])]
    if anchor.size < 4 or np.std(anchor) <= METRIC_EPS:
        return {f"{prefix}_anchor_q{index}_mean": float("nan") for index in range(1, 5)}

    quantiles = np.quantile(anchor, [0.25, 0.5, 0.75])
    groups = np.digitize(anchor, quantiles, right=True)
    metrics: dict[str, float] = {}
    for group_index in range(4):
        mask = groups == group_index
        metrics[f"{prefix}_anchor_q{group_index + 1}_mean"] = float(np.mean(kstar_array[mask])) if np.any(mask) else float("nan")
        metrics[f"{prefix}_anchor_q{group_index + 1}_count"] = float(np.sum(mask))
    return metrics


def compute_lag_gate_sensitivity(lag_gate: Any, z_values: Any, n_grid: int = 9) -> dict[str, float]:
    """Probe the learned lag gate with a z grid and summarize k* sensitivity."""

    z_array = _to_numpy(z_values).reshape(-1)
    if z_array.size < 2 or np.std(z_array) <= METRIC_EPS:
        return {
            "lag_gate_sensitivity_slope": float("nan"),
            "lag_gate_sensitivity_range": 0.0,
            "lag_gate_grid_min_kstar": float("nan"),
            "lag_gate_grid_max_kstar": float("nan"),
        }

    lower, upper = np.quantile(z_array, [0.05, 0.95])
    if not np.isfinite(lower) or not np.isfinite(upper) or abs(float(upper - lower)) <= METRIC_EPS:
        lower, upper = float(np.min(z_array)), float(np.max(z_array))
    if abs(float(upper - lower)) <= METRIC_EPS:
        return {
            "lag_gate_sensitivity_slope": float("nan"),
            "lag_gate_sensitivity_range": 0.0,
            "lag_gate_grid_min_kstar": float("nan"),
            "lag_gate_grid_max_kstar": float("nan"),
        }

    was_training = getattr(lag_gate, "training", False)
    device = next(lag_gate.parameters()).device
    dtype = next(lag_gate.parameters()).dtype
    grid = torch.linspace(float(lower), float(upper), steps=n_grid, device=device, dtype=dtype).unsqueeze(1)
    lag_gate.eval()
    with torch.no_grad():
        gate_output = lag_gate(grid)
        kstar_grid = gate_output.k_star.detach().cpu().numpy().reshape(-1)
    if was_training:
        lag_gate.train()

    grid_np = grid.detach().cpu().numpy().reshape(-1)
    slope = float(np.polyfit(grid_np, kstar_grid, deg=1)[0]) if grid_np.size > 1 else float("nan")
    return {
        "lag_gate_sensitivity_slope": slope,
        "lag_gate_sensitivity_range": float(np.max(kstar_grid) - np.min(kstar_grid)),
        "lag_gate_grid_min_kstar": float(np.min(kstar_grid)),
        "lag_gate_grid_max_kstar": float(np.max(kstar_grid)),
    }


def build_realdata_diagnostics(
    effective_kstar: Any,
    proxies: Any,
    metadata: dict[str, Any] | None,
    prefix: str = "kstar",
    omega: Any | None = None,
    z_values: Any | None = None,
    proxy_predictions: Any | None = None,
    proxy_metric_valid: bool = True,
    lag_gate: Any | None = None,
) -> dict[str, float]:
    """Build the full real-data diagnostics dictionary for one split."""

    metrics = compute_proxy_alignment_metrics(effective_kstar, proxies, metadata, prefix=prefix)
    metrics.update(compute_kstar_quantile_metrics(effective_kstar, proxies, metadata, prefix=prefix))
    if omega is not None:
        metrics.update(compute_omega_metrics(omega))
    if z_values is not None:
        metrics.update(compute_latent_proxy_metrics(z_values, proxies, metadata))
        if lag_gate is not None:
            metrics.update(compute_lag_gate_sensitivity(lag_gate, z_values))
    if proxy_predictions is not None:
        metrics.update(compute_proxy_reconstruction_metrics(proxy_predictions, proxies, metadata, metric_valid=proxy_metric_valid))
    return metrics


__all__ = [
    "build_realdata_diagnostics",
    "compute_lag_gate_sensitivity",
    "compute_omega_metrics",
    "compute_proxy_alignment_metrics",
    "compute_proxy_reconstruction_metrics",
    "infer_proxy_metadata",
    "proxy_metadata_payload",
]