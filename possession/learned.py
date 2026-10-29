"""Learned live/dead evidence from other matches, with label-free inference.

Only fit_model accepts reference labels. Neither feature construction nor decoding
can see them. Model JSON contains all learned scaling and coefficients; inference
never fits or normalizes from the prediction batch.
"""

import json
from pathlib import Path

import numpy as np

from .decoder import decode
from .features import extract, runs, smooth_in_halves
from .tags import build


BASE_NAMES = (
    "ball_observed", "ball_reliability", "ball_log_speed", "ball_speed_known",
    "nearest_distance", "nearest_known", "outside_support", "boundary_distance",
    "boundary_known", "player_coverage", "player_p25_speed", "player_median_speed",
    "player_p75_speed", "player_mean_speed", "players_below_1mps",
    "players_below_2mps", "players_below_3mps", "stationary_support", "missing_support",
)
SMOOTH_COLUMNS = (0, 2, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17)
SMOOTH_SECONDS = (0.6, 2.0, 6.0)
FEATURE_NAMES = BASE_NAMES + tuple(
    f"{BASE_NAMES[column]}_mean_{seconds:g}s"
    for seconds in SMOOTH_SECONDS for column in SMOOTH_COLUMNS
)


def feature_matrix(tracking, features, blocks, pitch, fps):
    """Finite features from observations only; independent of team ordering.

    Speed summaries use visible valid slots, with robust quartiles alongside means.
    Windows are centred and half-local because this is offline inference. No player
    identity, team name, possession field, event outcome or tactical tag is used.
    """
    n = len(tracking.ball)
    speeds = np.asarray(tracking.player_speed, dtype=np.float64)
    points = np.asarray(tracking.players)
    valid = (np.isfinite(points).all(axis=2) & (points >= -0.2).all(axis=2)
             & (points <= 1.2).all(axis=2) & np.isfinite(speeds)
             & (speeds >= 0) & (speeds <= 13))
    count = valid.sum(axis=1)
    masked = np.where(valid, speeds, np.nan)
    quantiles = np.zeros((n, 3))
    selected = count > 0
    if selected.any():
        quantiles[selected] = np.nanpercentile(masked[selected], [25, 50, 75], axis=1).T
    mean = np.divide(np.where(valid, speeds, 0).sum(axis=1), count,
                     out=np.zeros(n), where=count > 0)
    fractions = [np.divide((valid & (speeds < threshold)).sum(axis=1), count,
                           out=np.zeros(n), where=count > 0) for threshold in (1, 2, 3)]
    speed = np.asarray(features.ball_speed)
    known_speed = np.isfinite(speed)
    closest = np.min(features.nearest, axis=1)
    known_nearest = np.isfinite(closest)
    ball = np.asarray(features.ball)
    known_ball = np.isfinite(ball).all(axis=1)
    # Distance to the nearest pitch boundary, absolute for either side of the line.
    edge = np.min(np.abs(np.column_stack((ball[:, 0], pitch[0] - ball[:, 0],
                                         ball[:, 1], pitch[1] - ball[:, 1]))), axis=1)
    base = np.column_stack((
        features.observed.astype(float), features.reliability,
        np.log1p(np.clip(np.nan_to_num(speed, nan=0), 0, 45)) / np.log(46),
        known_speed.astype(float), np.clip(np.nan_to_num(closest, posinf=20), 0, 20) / 20,
        known_nearest.astype(float), features.outside,
        np.clip(np.nan_to_num(edge, nan=5), 0, 5) / 5, known_ball.astype(float),
        np.clip(count / 22, 0, 1),
        quantiles[:, 0] / 13, quantiles[:, 1] / 13, quantiles[:, 2] / 13,
        mean / 13, *fractions, features.stationary, features.missing,
    ))
    columns = [base]
    for seconds in SMOOTH_SECONDS:
        radius = max(1, round(seconds * fps / 2))
        columns.append(np.column_stack([
            smooth_in_halves(base[:, index], blocks, radius)
            for index in SMOOTH_COLUMNS
        ]))
    result = np.column_stack(columns)
    if result.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Learned feature matrix has invalid dimensions or values")
    return result


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def _matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected n by {len(FEATURE_NAMES)} learned features")
    if not np.isfinite(result).all():
        raise ValueError("Learned features must be finite")
    return result


def fit_model(X, labels, l2=0.01, sample_weight=None, metadata=None, max_iter=80):
    """Deterministic weighted ridge logistic regression, fit on known rows only.

    H/A are live targets, D is dead, and X is excluded before any statistics are
    computed. Zero-weight rows are likewise excluded. Regularization applies to
    standardized slopes, not the intercept. Newton steps use a monotone line search.
    """
    X = _matrix(X)
    labels = np.asarray(labels)
    if labels.shape != (len(X),) or not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Training labels must be a matching H/A/D/X vector")
    if not np.isfinite(l2) or l2 <= 0 or isinstance(l2, bool):
        raise ValueError("Regularization must be positive and finite")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    weights = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (len(X),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Sample weights must be finite and nonnegative")
    use = (labels != "X") & (weights > 0)
    if not use.any() or len(np.unique(labels[use] == "D")) != 2:
        raise ValueError("Training needs both known live and dead examples")
    x, y, w = X[use], (labels[use] == "D").astype(float), weights[use]
    w = w / w.sum()
    mean = np.sum(x * w[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * w[:, None], axis=0))
    scale[scale < 1e-8] = 1.0
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.zeros(z.shape[1])
    prior = float(np.sum(w * y))
    beta[0] = np.log(prior / (1 - prior))
    penalty = np.r_[0.0, np.full(z.shape[1] - 1, float(l2))]

    def objective(b):
        eta = z @ b
        return float(np.sum(w * (np.logaddexp(0, eta) - y * eta)) + .5 * np.sum(penalty * b * b))

    converged = False
    previous_loss = objective(beta)
    for iteration in range(max_iter):
        p = _sigmoid(z @ beta)
        gradient = z.T @ (w * (p - y)) + penalty * beta
        if np.max(np.abs(gradient)) < 1e-8:
            converged = True
            break
        hessian = z.T @ ((w * p * (1 - p))[:, None] * z)
        hessian.flat[::len(beta) + 1] += penalty + 1e-12
        step = np.linalg.solve(hessian, gradient)
        factor = 1.0
        for _ in range(30):
            candidate = beta - factor * step
            loss = objective(candidate)
            if loss <= previous_loss + 1e-14:
                break
            factor *= .5
        else:
            raise ValueError("Logistic optimizer could not find a decreasing step")
        beta = candidate
        previous_loss = loss
    # Recheck the final accepted iterate, including max_iter exit.
    gradient = z.T @ (w * (_sigmoid(z @ beta) - y)) + penalty * beta
    converged = bool(np.max(np.abs(gradient)) < 1e-7)
    if not converged:
        raise ValueError("Logistic optimizer did not converge; model was not produced")
    result = {
        "schema_version": 1, "kind": "logistic_live_dead", "decoder_mode": "joint",
        "feature_names": list(FEATURE_NAMES), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": beta[1:].tolist(), "intercept": float(beta[0]),
        "training": {"rows": int(use.sum()), "live_rows": int(np.sum(y == 0)),
                     "dead_rows": int(np.sum(y == 1)), "excluded_rows": int((~use).sum()),
                     "l2": float(l2), "iterations": iteration + 1,
                     "objective": previous_loss, "converged": converged,
                     "gradient_max_abs": float(np.max(np.abs(gradient)))},
        "metadata": {} if metadata is None else metadata,
    }
    validate_model(result)
    return result


def validate_model(model):
    if not isinstance(model, dict) or model.get("schema_version") != 1 or model.get("kind") != "logistic_live_dead":
        raise ValueError("Unsupported learned model schema")
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Learned model feature schema differs from implementation")
    if model.get("decoder_mode", "joint") not in ("joint", "separate"):
        raise ValueError("Unknown learned decoder mode")
    settings = model.get("decoder_settings", {})
    if not isinstance(settings, dict) or set(settings) - {"team_switch_cost", "min_team_seconds"}:
        raise ValueError("Unsupported learned decoder settings")
    if settings:
        from dataclasses import replace
        from .config import Config
        replace(Config(), **settings).validate()
    if "control_model" in model:
        from .control import validate_model as validate_control
        validate_control(model["control_model"])
    for key in ("mean", "scale", "coefficients"):
        a = np.asarray(model.get(key), dtype=float)
        if a.shape != (len(FEATURE_NAMES),) or not np.isfinite(a).all():
            raise ValueError(f"Invalid learned model {key}")
        if key == "scale" and (a <= 0).any():
            raise ValueError("Learned model scales must be positive")
    value = model.get("intercept")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError("Invalid learned model intercept")
    # Also reject non-serializable or nonfinite metadata without executing it.
    json.dumps(model, allow_nan=False)
    return model


def save_model(path, model):
    validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_model(path):
    return validate_model(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def predict_dead(X, model):
    validate_model(model)
    x = _matrix(X)
    return _sigmoid(((x - np.asarray(model["mean"])) / np.asarray(model["scale"]))
                    @ np.asarray(model["coefficients"]) + model["intercept"])


def _decode_binary(probabilities, durations, costs, evidence_scale):
    """Exact two-state duration Viterbi; half-edge segments may be truncated."""
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 2 or not len(probabilities)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Binary decoder needs finite normalized n by 2 probabilities")
    durations = np.asarray(durations, dtype=int)
    if durations.shape != (2,) or (durations < 1).any() or durations.sum() > 65535:
        raise ValueError("Invalid binary state durations")
    starts = np.r_[0, durations[0]]
    ends = np.cumsum(durations) - 1
    owners = np.repeat(np.arange(2), durations)
    size = int(durations.sum())
    age = np.array([i for i in range(size) if i not in starts], dtype=int)
    emission = -np.log(np.maximum(probabilities, 1e-8)) * evidence_scale
    previous = np.full(size, np.inf)
    previous[ends] = emission[0]
    back = np.zeros((len(probabilities), size), dtype=np.uint16)
    for frame in range(1, len(probabilities)):
        current = np.full(size, np.inf)
        current[age] = previous[age - 1] + emission[frame, owners[age]]
        back[frame, age] = age - 1
        for state in range(2):
            stop = ends[state]
            stay = previous[stop] + emission[frame, state]
            if stay <= current[stop]:
                current[stop], back[frame, stop] = stay, stop
            other = 1 - state
            switch = previous[ends[other]] + costs[other, state] + emission[frame, state]
            if switch < current[starts[state]]:
                current[starts[state]], back[frame, starts[state]] = switch, ends[other]
        current -= np.min(current)
        previous = current
    path = np.empty(len(probabilities), dtype=np.uint8)
    state = int(np.argmin(previous))
    for frame in range(len(path) - 1, -1, -1):
        path[frame] = owners[state]
        state = int(back[frame, state])
    return path


def resolve_config(config, model):
    """Apply validated model-selected decoding settings to the recorded config."""
    from dataclasses import replace
    validate_model(model)
    return replace(config, **model.get("decoder_settings", {})).validate()


def learned_probabilities(features, tags, X, model, control_matrix=None):
    """Observation-only emissions shared by campaign and production inference."""
    dead = predict_dead(X, model)
    if "control_model" in model:
        from .control import predict_home
        if control_matrix is None or len(control_matrix) != len(dead):
            raise ValueError("Combined model requires a matching control feature matrix")
        home = np.clip(predict_home(model["control_model"], control_matrix), 1e-8, 1 - 1e-8)
        team_logit = np.log(home) - np.log1p(-home)
    else:
        team_logit = features.team_log_odds
    home = _sigmoid(np.clip(team_logit + tags.team_log_odds, -8, 8))
    return np.column_stack(((1 - dead) * home, (1 - dead) * (1 - home), dead))


def decode_learned(features, tags, X, blocks, fps, config, model, mode=None, control_matrix=None):
    """Reuse an extracted feature matrix across fixed candidate comparisons."""
    mode = model.get("decoder_mode", "joint") if mode is None else mode
    if mode not in ("joint", "separate"):
        raise ValueError("Learned decoder mode must be joint or separate")
    config = resolve_config(config, model)
    probabilities = learned_probabilities(features, tags, X, model, control_matrix)
    dead = probabilities[:, 2]
    home = np.divide(probabilities[:, 0], 1 - dead, out=np.full(len(dead), 0.5), where=dead < 1)
    labels = np.full(len(dead), "X", dtype="<U1")
    warnings = []
    team_duration = max(1, round(config.min_team_seconds * fps))
    dead_duration = max(1, round(config.min_dead_seconds * fps))
    scale = config.effective_observations_per_second / fps
    for half, start, last in blocks:
        stop = last + 1
        evidence = (features.reliability[start:stop].sum() + tags.live[start:stop].sum()
                    + tags.restart_live[start:stop].sum())
        if evidence == 0:
            warnings.append(f"Half {half} has no usable ball or tactical evidence; returned X")
            continue
        if mode == "joint":
            labels[start:stop] = np.array(["H", "A", "D"])[decode(probabilities[start:stop], fps, config)]
            continue
        is_dead = _decode_binary(np.column_stack((1 - dead[start:stop], dead[start:stop])),
                                 [team_duration, dead_duration],
                                 np.array([[np.inf, config.stop_cost], [config.restart_cost, np.inf]]), scale)
        labels[start:stop] = "D"
        for a, b in runs(is_dead == 0):
            p = home[start + a:start + b]
            team = _decode_binary(np.column_stack((p, 1 - p)), [team_duration, team_duration],
                                  np.array([[np.inf, config.team_switch_cost],
                                            [config.team_switch_cost, np.inf]]), scale)
            labels[start + a:start + b] = np.array(["H", "A"])[team]
    return labels, probabilities, warnings


def infer_learned(tracking, events, blocks, pitch, fps, config, model, mode=None):
    config = resolve_config(config, model)
    features = extract(tracking, pitch, blocks, fps, config)
    tags = build(events, features, tracking.team_names, len(tracking.ball), fps, blocks, config)
    X = feature_matrix(tracking, features, blocks, pitch, fps)
    control_matrix = None
    if "control_model" in model:
        from .control import feature_matrix as control_features
        control_matrix = control_features(tracking, features, blocks, pitch, fps)
    labels, probabilities, warnings = decode_learned(features, tags, X, blocks, fps, config, model, mode,
                                                    control_matrix=control_matrix)
    return labels, probabilities, features, tags, warnings
