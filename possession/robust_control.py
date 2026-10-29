"""Symmetric 98-feature control head; the v1.2 schema stays unchanged.

The deterministic optimizer follows possession.control. Zero-intercept fitting
and uncentred training-only RMS scaling preserve H/A team exchange symmetry.
"""
import json
from pathlib import Path
import numpy as np
from . import control, changes, missing_ball

FEATURE_NAMES = (control.FEATURE_NAMES + tuple("change_" + n for n in changes.FEATURE_NAMES)
                 + missing_ball.FEATURE_NAMES)


def feature_matrix(tracking, features, blocks, pitch, fps):
    return np.column_stack([module.feature_matrix(tracking, features, blocks, pitch, fps)
                            for module in (control, changes, missing_ball)])


def _mask(values):
    result = np.asarray(values)
    if result.shape != (len(FEATURE_NAMES),) or result.dtype.kind != "b" or not result.any():
        raise ValueError("Feature mask must be a matching Boolean vector with an enabled feature")
    return result


def _matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError(f"Control features must be finite n by {len(FEATURE_NAMES)}")
    return result


def _sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def fit_model(X, labels, l2=0.001, sample_weight=None, metadata=None, max_iter=80, feature_mask=None):
    """Fit deterministic symmetric logistic control, using H/A rows only.

    With no intercept or centering, exchanging teams complements every probability,
    even with an imbalanced sample. RMS scales use positive-weight training rows.
    """
    X = _matrix(X)
    mask = _mask(np.ones(len(FEATURE_NAMES), dtype=bool) if feature_mask is None else feature_mask)
    X = X * mask
    labels = np.asarray(labels)
    if labels.shape != (len(X),) or not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Control labels must be a matching H/A/D/X vector")
    if isinstance(l2, bool) or not np.isfinite(l2) or l2 <= 0:
        raise ValueError("Regularization must be positive and finite")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    weights = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (len(X),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Sample weights must be finite and nonnegative")
    use = np.isin(labels, ["H", "A"]) & (weights > 0)
    if not use.any() or len(np.unique(labels[use])) != 2:
        raise ValueError("Control fitting needs both known team classes")
    x, y, w = X[use], (labels[use] == "H").astype(float), weights[use]
    w = w / w.sum()
    scale = np.sqrt(np.sum(x ** 2 * w[:, None], axis=0))
    scale[scale < 1e-8] = 1
    z = x / scale
    beta = np.zeros(z.shape[1])

    def objective(b):
        eta = z @ b
        return float(np.sum(w * (np.logaddexp(0, eta) - y * eta)) + 0.5 * l2 * (b @ b))

    loss = objective(beta)
    for iteration in range(max_iter):
        probabilities = _sigmoid(z @ beta)
        gradient = z.T @ (w * (probabilities - y)) + l2 * beta
        if np.max(np.abs(gradient)) < 1e-8:
            break
        hessian = z.T @ ((w * probabilities * (1 - probabilities))[:, None] * z)
        hessian.flat[::len(beta) + 1] += l2 + 1e-12
        step = np.linalg.solve(hessian, gradient)
        factor = 1.0
        for _ in range(30):
            candidate = beta - factor * step
            candidate_loss = objective(candidate)
            if candidate_loss <= loss + 1e-14:
                break
            factor *= 0.5
        else:
            raise ValueError("Control optimizer could not find a decreasing step")
        beta, loss = candidate, candidate_loss
    gradient = z.T @ (w * (_sigmoid(z @ beta) - y)) + l2 * beta
    if np.max(np.abs(gradient)) >= 1e-7:
        raise ValueError("Control optimizer did not converge")
    result = {
        "schema_version": 1, "kind": "symmetric_logistic_control_v3",
        "feature_names": list(FEATURE_NAMES), "feature_mask": mask.tolist(), "scale": scale.tolist(),
        "coefficients": beta.tolist(), "intercept": 0.0,
        "training": {"rows": int(use.sum()), "home_rows": int(np.sum(y == 1)),
                     "away_rows": int(np.sum(y == 0)), "excluded_rows": int((~use).sum()),
                     "l2": float(l2), "iterations": iteration + 1, "objective": loss,
                     "converged": True, "gradient_max_abs": float(np.max(np.abs(gradient)))},
        "metadata": {} if metadata is None else metadata,
    }
    return validate_model(result)


def validate_model(model):
    if (not isinstance(model, dict) or model.get("schema_version") != 1
            or model.get("kind") != "symmetric_logistic_control_v3"):
        raise ValueError("Unsupported control model schema")
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Control feature schema differs from implementation")
    for key in ("scale", "coefficients"):
        values = np.asarray(model.get(key), dtype=float)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid control model {key}")
        if key == "scale" and (values <= 0).any():
            raise ValueError("Control scales must be positive")
    mask = _mask(model.get("feature_mask"))
    if np.any(np.asarray(model["coefficients"])[~mask] != 0):
        raise ValueError("Excluded features must have zero coefficients")
    value = model.get("intercept")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value != 0:
        raise ValueError("Control intercept must be exactly zero to preserve symmetry")
    if "mean" in model:
        raise ValueError("Control models must not center signed features")
    json.dumps(model, allow_nan=False)
    return model


def predict_home(model, X):
    validate_model(model)
    x = _matrix(X)
    return _sigmoid((x / np.asarray(model["scale"])) @ np.asarray(model["coefficients"]))


def save_model(path, model):
    validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_model(path):
    return validate_model(json.loads(Path(path).read_text(encoding="utf-8-sig")))
