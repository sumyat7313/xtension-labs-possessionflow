"""Team-equivariant learned control evidence from tracking observations only.

All feature columns change sign when H/A membership is exchanged. A zero-intercept
logistic model with uncentred, training-only RMS scaling preserves that symmetry.
No trajectory identity is assumed: every frame uses permutation-invariant team
distance aggregates. Temporal context is offline, short, and half-local.
"""

import json
from pathlib import Path

import numpy as np

from .features import smooth_in_halves


BASE_NAMES = (
    "nearest_margin", "nearest_log_ratio", "near_1p5m", "near_3m", "near_6m",
    "second_margin", "third_margin", "local_density", "geometric_control",
    "slow_ball_control", "fast_ball_control", "ball_contact_control",
)
SMOOTH_SECONDS = (0.3, 0.8, 2.0)
CONTEXT_SECONDS = (0.4, 1.2)
FEATURE_NAMES = BASE_NAMES + tuple(
    f"{name}_mean_{seconds:g}s" for seconds in SMOOTH_SECONDS for name in BASE_NAMES
) + tuple(f"near_3m_{side}_{seconds:g}s" for seconds in CONTEXT_SECONDS
          for side in ("past", "future"))


def _ball_contact(ball, blocks, fps):
    """Change in ball velocity supports contact without requiring ball height."""
    result = np.zeros(len(ball))
    radius = max(1, round(0.12 * fps))
    for _, start, last in blocks:
        if last - start + 1 <= 2 * radius:
            continue
        center = ball[start + radius:last - radius + 1]
        before = (center - ball[start:last - 2 * radius + 1]) * fps / radius
        after = (ball[start + 2 * radius:last + 1] - center) * fps / radius
        good = (np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
                & (np.linalg.norm(before, axis=1) <= 45)
                & (np.linalg.norm(after, axis=1) <= 45))
        change = np.zeros(len(center))
        change[good] = np.clip(np.linalg.norm(after[good] - before[good], axis=1) / 10, 0, 1)
        result[start + radius:last - radius + 1] = change
    return result


def _directional_mean(values, blocks, radius, future=False):
    result = np.zeros(len(values))
    for _, start, last in blocks:
        section = values[start:last + 1]
        if future:
            section = section[::-1]
        prefix = np.r_[0.0, np.cumsum(section)]
        right = np.arange(1, len(section) + 1)
        left = np.maximum(0, right - radius)
        out = (prefix[right] - prefix[left]) / (right - left)
        result[start:last + 1] = out[::-1] if future else out
    return result


def feature_matrix(tracking, features, blocks, pitch, fps):
    """Return finite n-by-feature signed H-minus-A observation features.

    Coordinates are converted from normalized pitch fractions to physical metres.
    Distances are capped at 30 m; absent members are deliberately weak evidence.
    No labels, team names, event outcomes or persistent player IDs are inspected.
    """
    n = len(tracking.ball)
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("Pitch dimensions must be two positive finite metre values")
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    teams = np.asarray(tracking.teams)
    if not np.isin(teams, [0, 1]).all() or not all((teams == t).any() for t in (0, 1)):
        raise ValueError("Control observations need both team memberships")
    base = np.zeros((n, len(BASE_NAMES)))
    contact = _ball_contact(np.asarray(features.ball), blocks, fps)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        points = np.asarray(tracking.players[start:end], dtype=float)
        valid = (np.isfinite(points).all(axis=2) & (points >= -0.2).all(axis=2)
                 & (points <= 1.2).all(axis=2))
        distance = np.linalg.norm(points * pitch - features.ball[start:end, None], axis=2)
        distance[~valid | ~np.isfinite(distance)] = np.inf
        nearest = []
        ranked = []
        density = []
        available = []
        for team in (0, 1):
            d = np.sort(distance[:, teams == team], axis=1)
            nearest.append(np.minimum(d[:, 0], 30))
            # Fewer slots are allowed in synthetic examples; pad missing ranks.
            top = np.full((end - start, 3), 30.0)
            width = min(3, d.shape[1])
            top[:, :width] = np.minimum(d[:, :width], 30)
            ranked.append(top)
            density.append(np.exp(-np.minimum(d, 300) / 3).sum(axis=1))
            available.append(np.isfinite(d[:, 0]))
        home, away = nearest
        reliable = np.asarray(features.reliability[start:end], dtype=float)
        coverage = np.where(available[0] & available[1], 1.0, 0.25)
        support = reliable * coverage
        near = [np.exp(-home / r) - np.exp(-away / r) for r in (1.5, 3, 6)]
        margin = np.tanh((away - home) / 3)
        ratio = np.clip((np.log(away + 0.5) - np.log(home + 0.5)) / 3, -1, 1)
        slow = 1 / (1 + (np.nan_to_num(features.ball_speed[start:end], nan=15) / 8) ** 2)
        density_difference = (density[0] - density[1]) / (1 + (density[0] + density[1]))
        block = np.column_stack((
            margin, ratio, *near,
            np.tanh((ranked[1][:, 1] - ranked[0][:, 1]) / 5),
            np.tanh((ranked[1][:, 2] - ranked[0][:, 2]) / 5),
            density_difference, np.asarray(features.team_log_odds[start:end]) / 4,
            near[1] * slow, near[1] * (1 - slow), near[1] * contact[start:end],
        ))
        base[start:end] = block * support[:, None]
    active = np.zeros(n, dtype=bool)
    for _, start, last in blocks:
        active[start:last + 1] = True
    base[~active] = 0
    columns = [base]
    for seconds in SMOOTH_SECONDS:
        radius = max(1, round(seconds * fps / 2))
        columns.append(np.column_stack([smooth_in_halves(base[:, i], blocks, radius)
                                        for i in range(base.shape[1])]))
    for seconds in CONTEXT_SECONDS:
        radius = max(1, round(seconds * fps))
        for future in (False, True):
            columns.append(_directional_mean(base[:, 3], blocks, radius, future)[:, None])
    result = np.column_stack(columns)
    if result.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Invalid control feature matrix")
    return result


def _matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError(f"Control features must be finite n by {len(FEATURE_NAMES)}")
    return result


def _sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def fit_model(X, labels, l2=0.01, sample_weight=None, metadata=None, max_iter=80):
    """Fit deterministic symmetric logistic control, using H/A rows only.

    With no intercept or centering, exchanging teams complements every probability,
    even with an imbalanced sample. RMS scales use positive-weight training rows.
    """
    X = _matrix(X)
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
        "schema_version": 1, "kind": "symmetric_logistic_control",
        "feature_names": list(FEATURE_NAMES), "scale": scale.tolist(),
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
            or model.get("kind") != "symmetric_logistic_control"):
        raise ValueError("Unsupported control model schema")
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Control feature schema differs from implementation")
    for key in ("scale", "coefficients"):
        values = np.asarray(model.get(key), dtype=float)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid control model {key}")
        if key == "scale" and (values <= 0).any():
            raise ValueError("Control scales must be positive")
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
