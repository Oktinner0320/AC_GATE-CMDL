"""Simple panel forecast baselines for real-data difficulty calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.metrics import compute_mae, compute_mse, compute_r2


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def aligned_targets(panel: Any, max_lag: int) -> np.ndarray:
    y = _to_numpy(panel.Y_it)
    if y.ndim != 2 or y.shape[1] <= max_lag:
        raise ValueError("panel.Y_it must have shape [N, T] with T > max_lag")
    return y[:, max_lag:]


def _previous_targets(panel: Any, max_lag: int) -> np.ndarray:
    y = _to_numpy(panel.Y_it)
    return y[:, max_lag - 1 : -1]


def _score(prefix: str, pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        f"baseline_{prefix}_mse": float(compute_mse(pred, true)),
        f"baseline_{prefix}_mae": float(compute_mae(pred, true)),
        f"baseline_{prefix}_r2": float(compute_r2(pred, true)),
    }


def _target_variance_metrics(target: np.ndarray) -> dict[str, float]:
    entity_variances = np.var(target, axis=1) if target.shape[1] > 1 else np.zeros(target.shape[0])
    time_variances = np.var(target, axis=0) if target.shape[0] > 1 else np.zeros(target.shape[1])
    return {
        "target_variance": float(np.var(target)),
        "target_entity_variance_mean": float(np.mean(entity_variances)),
        "target_time_variance_mean": float(np.mean(time_variances)),
    }


def _lagged_design(panel: Any, max_lag: int, entity_indices: np.ndarray | None = None) -> np.ndarray:
    x = _to_numpy(panel.X_it)
    p = _to_numpy(panel.p_i)
    s = _to_numpy(panel.s_i)
    if x.ndim != 3:
        raise ValueError("panel.X_it must have shape [N, T, F]")
    if entity_indices is not None:
        x = x[entity_indices]
        p = p[entity_indices]
        s = s[entity_indices]

    n_entities, seq_length, _ = x.shape
    valid_steps = seq_length - max_lag
    rows: list[np.ndarray] = []
    time_grid = np.linspace(-1.0, 1.0, valid_steps, dtype=np.float64)
    for time_offset, time_index in enumerate(range(max_lag, seq_length)):
        lag_block = x[:, time_index - max_lag : time_index, :].reshape(n_entities, -1)
        current_x = x[:, time_index, :]
        time_feature = np.full((n_entities, 1), time_grid[time_offset], dtype=np.float64)
        rows.append(np.concatenate([current_x, lag_block, p, s, time_feature], axis=1))
    return np.concatenate(rows, axis=0)


def _flatten_target(panel: Any, max_lag: int, entity_indices: np.ndarray | None = None) -> np.ndarray:
    target = aligned_targets(panel, max_lag)
    if entity_indices is not None:
        target = target[entity_indices]
    return target.T.reshape(-1)


def _fit_panel_ols(train_panel: Any, max_lag: int, ridge: float = 1e-6, entity_indices: np.ndarray | None = None) -> np.ndarray:
    x_train = _lagged_design(train_panel, max_lag, entity_indices=entity_indices)
    y_train = _flatten_target(train_panel, max_lag, entity_indices=entity_indices)
    design = np.concatenate([np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1)
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y_train)


def _predict_panel_ols(panel: Any, max_lag: int, coefficients: np.ndarray) -> np.ndarray:
    x_eval = _lagged_design(panel, max_lag)
    design = np.concatenate([np.ones((x_eval.shape[0], 1), dtype=np.float64), x_eval], axis=1)
    flat_pred = design @ coefficients
    valid_steps = _to_numpy(panel.Y_it).shape[1] - max_lag
    return flat_pred.reshape(valid_steps, -1).T


def _anchor_proxy(panel: Any) -> np.ndarray:
    proxies = _to_numpy(panel.p_i)
    metadata = getattr(panel, "metadata", {}) or {}
    anchor_index = int(metadata.get("anchor_proxy_index", 0))
    anchor_index = min(max(anchor_index, 0), proxies.shape[1] - 1)
    return proxies[:, anchor_index]


def _group_indices(values: np.ndarray, thresholds: np.ndarray) -> list[np.ndarray]:
    groups = np.digitize(values, thresholds, right=True)
    return [np.flatnonzero(groups == group_index) for group_index in range(3)]


def _lag_strength_summary(coefficients: np.ndarray, seq_features: int, max_lag: int) -> tuple[float, float]:
    lag_start = 1 + seq_features
    lag_end = lag_start + max_lag * seq_features
    lag_coefficients = coefficients[lag_start:lag_end].reshape(max_lag, seq_features)
    strengths = np.sum(np.abs(lag_coefficients), axis=1)
    if not np.any(np.isfinite(strengths)) or float(np.sum(strengths)) <= 0.0:
        return float("nan"), float("nan")
    lag_indices = np.arange(1, max_lag + 1, dtype=np.float64)
    best_lag = float(np.argmax(strengths) + 1)
    effective_lag = float(np.sum(lag_indices * strengths) / np.sum(strengths))
    return best_lag, effective_lag


def evaluate_grouped_ardl_baseline(train_panel: Any, eval_panel: Any, max_lag: int) -> dict[str, float]:
    """Evaluate an anchor-quantile grouped distributed-lag OLS baseline."""

    eval_target = aligned_targets(eval_panel, max_lag)
    global_coefficients = _fit_panel_ols(train_panel, max_lag)
    fallback_pred = _predict_panel_ols(eval_panel, max_lag, global_coefficients)
    predictions = fallback_pred.copy()

    train_anchor = _anchor_proxy(train_panel)
    eval_anchor = _anchor_proxy(eval_panel)
    if train_anchor.size < 3 or float(np.std(train_anchor)) <= 1e-10:
        metrics = _score("grouped_ardl", fallback_pred, eval_target)
        metrics.update(
            {
                "baseline_grouped_ardl_group_count": 1.0,
                "baseline_grouped_ardl_best_lag_mean": float("nan"),
                "baseline_grouped_ardl_effective_lag_mean": float("nan"),
            }
        )
        return metrics

    thresholds = np.quantile(train_anchor, [1.0 / 3.0, 2.0 / 3.0])
    train_groups = _group_indices(train_anchor, thresholds)
    eval_groups = _group_indices(eval_anchor, thresholds)
    seq_features = _to_numpy(train_panel.X_it).shape[2]
    best_lags: list[float] = []
    effective_lags: list[float] = []
    group_labels = ["low", "mid", "high"]
    group_count = 0

    for label, train_indices, eval_indices in zip(group_labels, train_groups, eval_groups):
        if train_indices.size == 0 or eval_indices.size == 0:
            continue
        try:
            coefficients = _fit_panel_ols(train_panel, max_lag, entity_indices=train_indices)
            group_pred = _predict_panel_ols(eval_panel, max_lag, coefficients=np.asarray(coefficients))
            predictions[eval_indices] = group_pred[eval_indices]
            best_lag, effective_lag = _lag_strength_summary(coefficients, seq_features, max_lag)
            best_lags.append(best_lag)
            effective_lags.append(effective_lag)
            group_count += 1
        except Exception:
            continue

    metrics = _score("grouped_ardl", predictions, eval_target)
    metrics["baseline_grouped_ardl_group_count"] = float(group_count)
    for label, best_lag, effective_lag in zip(group_labels, best_lags, effective_lags):
        metrics[f"baseline_grouped_ardl_{label}_best_lag"] = float(best_lag)
        metrics[f"baseline_grouped_ardl_{label}_effective_lag"] = float(effective_lag)
    finite_best = [value for value in best_lags if np.isfinite(value)]
    finite_effective = [value for value in effective_lags if np.isfinite(value)]
    metrics["baseline_grouped_ardl_best_lag_mean"] = float(np.mean(finite_best)) if finite_best else float("nan")
    metrics["baseline_grouped_ardl_effective_lag_mean"] = float(np.mean(finite_effective)) if finite_effective else float("nan")
    return metrics


def evaluate_panel_forecast_baselines(train_panel: Any, eval_panel: Any, max_lag: int) -> dict[str, float]:
    """Evaluate simple baselines on one split using train split statistics only."""

    train_target = aligned_targets(train_panel, max_lag)
    eval_target = aligned_targets(eval_panel, max_lag)
    metrics = _target_variance_metrics(eval_target)

    train_mean_pred = np.full_like(eval_target, float(np.mean(train_target)))
    metrics.update(_score("train_mean", train_mean_pred, eval_target))

    train_entity_means = np.mean(train_target, axis=1)
    if train_entity_means.shape[0] == eval_target.shape[0]:
        entity_mean_pred = np.repeat(train_entity_means[:, None], eval_target.shape[1], axis=1)
        metrics.update(_score("entity_mean", entity_mean_pred, eval_target))

    persistence_pred = _previous_targets(eval_panel, max_lag)
    metrics.update(_score("persistence", persistence_pred, eval_target))

    try:
        coefficients = _fit_panel_ols(train_panel, max_lag)
        panel_ols_pred = _predict_panel_ols(eval_panel, max_lag, coefficients)
        metrics.update(_score("panel_ols", panel_ols_pred, eval_target))
    except Exception:
        metrics.update(
            {
                "baseline_panel_ols_mse": float("nan"),
                "baseline_panel_ols_mae": float("nan"),
                "baseline_panel_ols_r2": float("nan"),
            }
        )

    try:
        metrics.update(evaluate_grouped_ardl_baseline(train_panel, eval_panel, max_lag))
    except Exception:
        metrics.update(
            {
                "baseline_grouped_ardl_mse": float("nan"),
                "baseline_grouped_ardl_mae": float("nan"),
                "baseline_grouped_ardl_r2": float("nan"),
                "baseline_grouped_ardl_group_count": float("nan"),
                "baseline_grouped_ardl_best_lag_mean": float("nan"),
                "baseline_grouped_ardl_effective_lag_mean": float("nan"),
            }
        )

    r2_values = [value for key, value in metrics.items() if key.startswith("baseline_") and key.endswith("_r2") and np.isfinite(value)]
    metrics["baseline_best_simple_r2"] = float(max(r2_values)) if r2_values else float("nan")
    return metrics


def add_forecast_calibration(
    metrics: dict[str, float],
    train_panel: Any,
    eval_panel: Any,
    max_lag: int,
) -> dict[str, float]:
    """Return metrics enriched with simple baseline and model-vs-baseline deltas."""

    enriched = dict(metrics)
    baseline_metrics = evaluate_panel_forecast_baselines(train_panel, eval_panel, max_lag)
    enriched.update(baseline_metrics)
    model_r2 = enriched.get("r2")
    for baseline_name in ("train_mean", "entity_mean", "persistence", "panel_ols", "grouped_ardl"):
        baseline_r2 = enriched.get(f"baseline_{baseline_name}_r2")
        if model_r2 is None or baseline_r2 is None or not np.isfinite(float(baseline_r2)):
            enriched[f"r2_delta_vs_{baseline_name}"] = float("nan")
        else:
            enriched[f"r2_delta_vs_{baseline_name}"] = float(model_r2) - float(baseline_r2)
    return enriched


__all__ = [
    "add_forecast_calibration",
    "aligned_targets",
    "evaluate_grouped_ardl_baseline",
    "evaluate_panel_forecast_baselines",
]