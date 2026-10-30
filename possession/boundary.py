"""Observation-only H/A boundary evidence for bounded decoder switch costs.

Boundary probability describes a short neighborhood, not an exact-frame hazard.
All inputs and operations are invariant to exchanging H/A teams and half-local.
The optional model never changes minimum durations or dead/restart penalties.
"""
import json
import numpy as np

CONTROL_WIDTH = 98
SIGNALS = ("home_margin", "nearest_margin", "near_3m", "geometry", "contact", "reach")
RADII = (.2, .6)
FEATURE_NAMES = ("dead_probability", "home_certainty", "base_support", "change_support", "missing_support") + tuple(
    "absolute_change_" + str(i) for i in range(8)
) + tuple(name for seconds in RADII for name in (
    f"past_support_{seconds:g}", f"future_support_{seconds:g}",
    *(f"{signal}_{kind}_{seconds:g}" for signal in SIGNALS for kind in ("difference", "reversal", "agreement"))
))


def _blocks(blocks, n):
    active = np.zeros(n, dtype=bool)
    result = []
    for block in blocks:
        if len(block) != 3: raise ValueError("Invalid boundary block")
        _, start, last = block
        if not isinstance(start, (int, np.integer)) or not isinstance(last, (int, np.integer)):
            raise ValueError("Boundary indices must be integers")
        if not 0 <= start <= last < n or active[start:last+1].any(): raise ValueError("Invalid boundary blocks")
        active[start:last+1] = True; result.append((int(start), int(last)))
    return result, active


def feature_matrix(control_matrix, probabilities, blocks, fps):
    """Return n x 53 unsigned features; no labels enter this function."""
    x, p = np.asarray(control_matrix, dtype=float), np.asarray(probabilities, dtype=float)
    if x.ndim != 2 or x.shape[1] != CONTROL_WIDTH or not np.isfinite(x).all():
        raise ValueError("Boundary observations must be finite n by 98")
    if p.shape != (len(x), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Boundary probabilities must be normalized n by 3")
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0: raise ValueError("FPS must be positive")
    sections, active = _blocks(blocks, len(x))
    live = 1 - p[:, 2]
    home = np.divide(p[:, 0], live, out=np.full(len(p), .5), where=live > 1e-8)
    margin = 2 * home - 1
    columns = [p[:, 2], np.abs(margin),
               np.minimum(np.linalg.norm(x[:, :12], axis=1), 4)/4,
               np.minimum(np.linalg.norm(x[:, 52:80], axis=1), 4)/4,
               np.minimum(np.linalg.norm(x[:, 80:], axis=1), 4)/4]
    columns.extend(np.minimum(np.abs(x[:, 52+i]), 2)/2 for i in range(8))
    signals = np.column_stack((margin, np.clip(x[:, (0, 3, 8, 11, 54)], -1, 1)))
    for seconds in RADII:
        radius = max(1, round(seconds * fps))
        past, future = np.zeros_like(signals), np.zeros_like(signals)
        past_support, future_support = np.zeros(len(x)), np.zeros(len(x))
        for start, last in sections:
            values = signals[start:last+1]; n = len(values); indices = np.arange(n)
            prefix = np.vstack((np.zeros((1, len(SIGNALS))), np.cumsum(values, axis=0)))
            left, right = np.maximum(0, indices-radius), np.minimum(n, indices+radius)
            a, b = indices-left, right-indices
            # Edge context is unavailable, never copied from an adjacent half.
            past[start:last+1] = np.divide(prefix[indices]-prefix[left], a[:, None],
                                          out=np.zeros_like(values), where=a[:, None] > 0)
            future[start:last+1] = (prefix[right]-prefix[indices])/b[:, None]
            past_support[start:last+1], future_support[start:last+1] = a/radius, b/radius
        columns.extend((past_support, future_support))
        support = np.minimum(past_support, future_support)
        for i in range(len(SIGNALS)):
            product = past[:, i] * future[:, i]
            columns.extend((np.abs(past[:, i]-future[:, i])/2 * support,
                            np.maximum(-product, 0) * support,
                            np.maximum(product, 0) * support))
    result = np.column_stack(columns)
    result[~active] = 0
    if result.shape != (len(x), len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Invalid boundary features")
    return result


def validate_model(model):
    if not isinstance(model, dict) or model.get("kind") != "local_possession_boundary" or model.get("schema_version") != 1:
        raise ValueError("Unsupported boundary model")
    if model.get("feature_names") != list(FEATURE_NAMES) or model.get("target_radius_seconds") != .16:
        raise ValueError("Boundary schema differs")
    for key in ("mean", "scale", "coefficients"):
        value = np.asarray(model.get(key), dtype=float)
        if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all(): raise ValueError(f"Invalid boundary {key}")
        if key == "scale" and (value <= 0).any(): raise ValueError("Boundary scales must be positive")
    for key in ("intercept", "prior"):
        value = model.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid boundary {key}")
    if not 0 < model["prior"] < 1 or model.get("maximum_cost_reduction") != .75:
        raise ValueError("Invalid boundary probability prior or reduction")
    json.dumps(model, allow_nan=False)
    return model


def predict_from_features(model, matrix):
    validate_model(model)
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all(): raise ValueError("Invalid boundary matrix")
    z = np.clip((x-np.asarray(model["mean"]))/np.asarray(model["scale"]), -8, 8)
    logit = z @ np.asarray(model["coefficients"]) + model["intercept"]
    return 1/(1+np.exp(-np.clip(logit, -35, 35)))


def predict_probability(model, control_matrix, probabilities, blocks, fps):
    result = predict_from_features(model, feature_matrix(control_matrix, probabilities, blocks, fps))
    sections, active = _blocks(blocks, len(result))
    result[~active] = 0
    for start, _ in sections: result[start] = 0  # no transition into a half
    return result


def switch_costs(model, control_matrix, probabilities, blocks, fps, base_cost):
    """Lower H/A switch cost only with stronger-than-prior boundary evidence."""
    if isinstance(base_cost, bool) or not np.isfinite(base_cost) or base_cost < 0: raise ValueError("Invalid base switch cost")
    probability = predict_probability(model, control_matrix, probabilities, blocks, fps)
    prior = model["prior"]
    high = max(.5, (1+prior)/2 if prior >= .5 else .5)
    strength = np.clip((probability-prior)/(high-prior), 0, 1)
    return float(base_cost) * (1-model["maximum_cost_reduction"]*strength)
