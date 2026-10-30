"""Portable, odd temporal residual network and reliability-gated expert mixture.

Training uses PyTorch in a separate optional environment. Inference uses NumPy
only. All operations are half-local; exchanging teams negates logits. No stable
player-slot identities, target labels, or recurrent cross-period state are used.
"""
import json
import numpy as np
from .robust_control import FEATURE_NAMES

DILATIONS = (1, 3, 9)
HIDDEN = 16
GATE_NAMES = ("bias", "base_confidence", "expert_confidence", "disagreement",
              "missing_context", "geometry_support")


def validate_model(model):
    if not isinstance(model, dict) or model.get("kind") != "odd_temporal_residual" or model.get("schema_version") != 1:
        raise ValueError("Unsupported temporal model")
    if model.get("feature_names") != list(FEATURE_NAMES) or model.get("sample_hz") != 5:
        raise ValueError("Temporal feature schema or sampling differs")
    if model.get("dilations") != list(DILATIONS): raise ValueError("Temporal dilation schema differs")
    shapes = {"scale": (len(FEATURE_NAMES),), "input_weight": (HIDDEN, len(FEATURE_NAMES)),
              "output_weight": (1, HIDDEN), **{f"conv_{i}": (HIDDEN, HIDDEN, 3) for i in range(3)}}
    for key, shape in shapes.items():
        value = np.asarray(model.get(key), dtype=float)
        if value.shape != shape or not np.isfinite(value).all(): raise ValueError(f"Invalid temporal {key}")
        if key == "scale" and (value <= 0).any(): raise ValueError("Temporal scales must be positive")
    json.dumps(model, allow_nan=False)
    return model


def low_rate_residual(model, matrix):
    validate_model(model)
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid temporal observation matrix")
    h = np.tanh(np.clip(x / np.asarray(model["scale"]), -8, 8) @ np.asarray(model["input_weight"]).T)
    for i, dilation in enumerate(DILATIONS):
        weight = np.asarray(model[f"conv_{i}"])
        value = h @ weight[:, :, 1].T
        if len(h) > dilation:
            value[dilation:] += h[:-dilation] @ weight[:, :, 0].T
            value[:-dilation] += h[dilation:] @ weight[:, :, 2].T
        h = np.tanh(h + value)
    return (h @ np.asarray(model["output_weight"]).T).ravel()


def residual(model, matrix, blocks, fps):
    validate_model(model)
    if not np.isfinite(fps) or fps < 5 or not np.isclose(fps / 5, round(fps / 5)):
        raise ValueError("Temporal inference requires FPS divisible by 5")
    step = round(fps / 5)
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid temporal observation matrix")
    out = np.zeros(len(x)); active = np.zeros(len(x), dtype=bool)
    for _, start, last in blocks:
        if not 0 <= start <= last < len(x) or active[start:last+1].any(): raise ValueError("Invalid temporal blocks")
        active[start:last+1] = True
        indices = np.arange(start, last+1, step)
        values = low_rate_residual(model, x[indices])
        out[start:last+1] = np.interp(np.arange(start,last+1), indices, values)
    return out


def gate_features(base_logit, expert_logit, matrix):
    base, expert = np.asarray(base_logit), np.asarray(expert_logit)
    x = np.asarray(matrix)
    if base.shape != (len(x),) or expert.shape != base.shape or x.shape[1:] != (len(FEATURE_NAMES),):
        raise ValueError("Invalid expert gate inputs")
    result = np.column_stack((np.ones(len(x)), np.minimum(np.abs(base), 8)/8,
        np.minimum(np.abs(expert), 8)/8, np.minimum(np.abs(expert-base), 8)/8,
        np.minimum(np.linalg.norm(x[:,80:],axis=1), 2)/2,
        np.minimum(np.linalg.norm(x[:,:12],axis=1), 2)/2))
    if not np.isfinite(result).all(): raise ValueError("Nonfinite expert gate inputs")
    return result


def validate_gate(model):
    if not isinstance(model,dict) or model.get("kind") != "reliability_expert_gate" or model.get("feature_names") != list(GATE_NAMES):
        raise ValueError("Invalid reliability gate schema")
    c = np.asarray(model.get("coefficients"), dtype=float)
    if c.shape != (len(GATE_NAMES),) or not np.isfinite(c).all(): raise ValueError("Invalid reliability gate weights")
    json.dumps(model, allow_nan=False)
    return model


def predict_home(base_logit, matrix, blocks, fps, model, gate=None):
    base = np.asarray(base_logit,dtype=float)
    if base.shape != (len(matrix),) or not np.isfinite(base).all(): raise ValueError("Invalid base logits")
    expert = base + residual(model,matrix,blocks,fps)
    sigmoid = lambda value: 1/(1+np.exp(-np.clip(value,-35,35)))
    if gate is None: return sigmoid(expert)
    validate_gate(gate)
    weight = sigmoid(gate_features(base,expert,matrix) @ np.asarray(gate["coefficients"]))
    return (1-weight)*sigmoid(base) + weight*sigmoid(expert)
