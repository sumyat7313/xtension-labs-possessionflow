"""Half-local joint H/A/D temporal residuals, with NumPy-only inference.

Even observation context feeds both the live/dead head and an odd team-control
branch. Team exchange negates the latter and leaves live/dead evidence unchanged.
The original model's logits are offsets, so zero residuals preserve its emissions.
"""
import json
import numpy as np
from .robust_control import FEATURE_NAMES as CONTROL_NAMES
from .learned import FEATURE_NAMES as DEAD_NAMES

HIDDEN = 12
DILATIONS = (1, 3, 9)
CONTEXT_NAMES = tuple(DEAD_NAMES) + tuple("abs_" + name for name in CONTROL_NAMES)
RESIDUAL_LIMIT = 4.0


def observations(control, dead):
    x, z = np.asarray(control, dtype=float), np.asarray(dead, dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(CONTROL_NAMES)
            or z.shape != (len(x), len(DEAD_NAMES))
            or not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError("Joint observations require finite n by 98 control and n by 58 dead features")
    return x, z


def context_features(control, dead):
    x, z = observations(control, dead)
    return np.column_stack((z, np.abs(x)))


def validate_model(model):
    if (not isinstance(model, dict) or model.get("schema_version") != 1
            or model.get("kind") != "equivariant_joint_temporal_residual"
            or model.get("sample_hz") != 5 or model.get("dilations") != list(DILATIONS)
            or model.get("residual_limit") != RESIDUAL_LIMIT
            or model.get("control_names") != list(CONTROL_NAMES)
            or model.get("context_names") != list(CONTEXT_NAMES)):
        raise ValueError("Unsupported joint temporal schema")
    shapes = {"control_scale": (len(CONTROL_NAMES),), "context_mean": (len(CONTEXT_NAMES),),
              "context_scale": (len(CONTEXT_NAMES),), "odd_input": (HIDDEN, len(CONTROL_NAMES)),
              "even_input": (HIDDEN, len(CONTEXT_NAMES)), "even_input_bias": (HIDDEN,),
              "home_output": (1, HIDDEN), "dead_output": (1, HIDDEN), "dead_output_bias": (1,)}
    for i in range(len(DILATIONS)):
        shapes.update({f"odd_conv_{i}": (HIDDEN, HIDDEN, 3),
                       f"even_conv_{i}": (HIDDEN, HIDDEN, 3), f"even_conv_bias_{i}": (HIDDEN,)})
    for key, shape in shapes.items():
        value = np.asarray(model.get(key), dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid joint temporal {key}")
        if key.endswith("scale") and (value <= 0).any():
            raise ValueError("Joint temporal scales must be positive")
    json.dumps(model, allow_nan=False)
    return model


def normalized_inputs(model, control, dead):
    x, z = observations(control, dead)
    odd = np.clip(x / np.asarray(model["control_scale"]), -8, 8)
    even = np.clip((context_features(x, z) - np.asarray(model["context_mean"]))
                   / np.asarray(model["context_scale"]), -8, 8)
    return odd, even


def _convolve(values, weights, dilation):
    value = values @ weights[:, :, 1].T
    if len(values) > dilation:
        value[dilation:] += values[:-dilation] @ weights[:, :, 0].T
        value[:-dilation] += values[dilation:] @ weights[:, :, 2].T
    return value


def low_rate_residual(model, control, dead):
    """Return residual [conditional-home logit, dead logit] at native 5 Hz."""
    validate_model(model)
    odd, even = normalized_inputs(model, control, dead)
    even = np.tanh(even @ np.asarray(model["even_input"]).T + model["even_input_bias"])
    odd = np.tanh(odd @ np.asarray(model["odd_input"]).T) * (1 + .25 * even)
    for i, dilation in enumerate(DILATIONS):
        even = np.tanh(even + _convolve(even, np.asarray(model[f"even_conv_{i}"]), dilation)
                       + model[f"even_conv_bias_{i}"])
        odd = np.tanh(odd + _convolve(odd, np.asarray(model[f"odd_conv_{i}"]), dilation)) * (1 + .25 * even)
    home = (odd @ np.asarray(model["home_output"]).T).ravel()
    death = (even @ np.asarray(model["dead_output"]).T).ravel() + model["dead_output_bias"][0]
    return RESIDUAL_LIMIT * np.tanh(np.column_stack((home, death)) / RESIDUAL_LIMIT)


def _blocks(blocks, length, fps):
    if not np.isfinite(fps) or fps < 5 or not np.isclose(fps / 5, round(fps / 5)):
        raise ValueError("Joint temporal inference requires FPS divisible by 5")
    active = np.zeros(length, dtype=bool)
    for block in blocks:
        if len(block) != 3:
            raise ValueError("Invalid joint temporal blocks")
        _, start, last = block
        if (not isinstance(start, (int, np.integer)) or not isinstance(last, (int, np.integer))
                or not 0 <= start <= last < length or active[start:last + 1].any()):
            raise ValueError("Invalid joint temporal blocks")
        active[start:last + 1] = True
    return active, round(fps / 5)


def residual(model, control, dead, blocks, fps):
    validate_model(model)
    x, z = observations(control, dead)
    blocks = list(blocks)
    _, step = _blocks(blocks, len(x), fps)
    output = np.zeros((len(x), 2))
    for _, start, last in blocks:
        indices = np.arange(start, last + 1, step)
        values = low_rate_residual(model, x[indices], z[indices])
        for column in range(2):
            output[start:last + 1, column] = np.interp(np.arange(start, last + 1), indices, values[:, column])
    return output


def predict_probabilities(base_home, base_dead, control, dead, blocks, fps, model):
    """H/A/D probabilities; ungraded gaps are zero, never fabricated live states.

    base_home is the conditional H-given-live logit; base_dead is the D logit.
    Callers should apply any automatic tag fusion consistently after this step.
    """
    x, z = observations(control, dead)
    home, death = np.asarray(base_home, dtype=float), np.asarray(base_dead, dtype=float)
    if home.shape != (len(x),) or death.shape != home.shape or not np.isfinite(home).all() or not np.isfinite(death).all():
        raise ValueError("Invalid joint temporal baseline logits")
    blocks = list(blocks)
    active, _ = _blocks(blocks, len(x), fps)
    adjustment = residual(model, x, z, blocks, fps)
    sigmoid = lambda value: 1 / (1 + np.exp(-np.clip(value, -35, 35)))
    h = sigmoid(home + adjustment[:, 0]); d = sigmoid(death + adjustment[:, 1])
    output = np.column_stack(((1 - d) * h, (1 - d) * (1 - h), d))
    output[~active] = 0
    return output
