"""Bounded transition evidence with optional, locally gated short-team exits.

The classifier estimates a rare boundary neighborhood. Its probability is
converted to odds relative to the training prior, rather than requiring an
absolute probability near one half before reducing a switch cost. This is
evidence strength, not a claim of calibrated transition hazard. All context is
half-local and exchanging H/A leaves both outputs unchanged.
"""
import json
import numpy as np
from . import boundary


POLICY = {
    "maximum_cost_reduction": .75,
    "saturation_odds_ratio": 8.,
    "early_exit_minimum_strength": .75,
    "context_seconds": .2,
    "minimum_context_margin": .2,
    "maximum_context_dead_probability": .25,
    "minimum_short_team_seconds": .16,
    "early_exit_penalty": 1.,
}


def from_boundary(classifier, metadata=None):
    """Wrap a training-only fitted classifier in the fixed v6 inference policy."""
    import copy
    return validate_model({
        "schema_version": 1, "kind": "evidence_gated_team_transitions",
        "classifier": copy.deepcopy(boundary.validate_model(classifier)),
        "policy": dict(POLICY), "metadata": {} if metadata is None else metadata,
    })


def validate_model(model):
    if (not isinstance(model, dict) or model.get("schema_version") != 1
            or model.get("kind") != "evidence_gated_team_transitions"):
        raise ValueError("Unsupported transition-evidence model")
    policy = model.get("policy")
    if not isinstance(policy, dict) or set(policy) != set(POLICY):
        raise ValueError("Transition policy schema differs")
    if any(isinstance(policy[key], bool) or not isinstance(policy[key], (int, float))
           or not np.isfinite(policy[key]) or policy[key] != value
           for key, value in POLICY.items()):
        raise ValueError("Transition policy differs from the fixed experiment")
    boundary.validate_model(model.get("classifier"))
    json.dumps(model, allow_nan=False)
    return model


def _context(probabilities, blocks, fps):
    p = np.asarray(probabilities, dtype=float)
    sections, active = boundary._blocks(blocks, len(p))
    live = 1-p[:, 2]
    margin = np.divide(p[:, 0]-p[:, 1], live, out=np.zeros(len(p)), where=live > 1e-8)
    values = np.column_stack((margin, p[:, 2]))
    past, future = np.zeros_like(values), np.zeros_like(values)
    complete = np.zeros(len(p), dtype=bool)
    radius = max(1, round(POLICY["context_seconds"]*fps))
    for start, last in sections:
        local = values[start:last+1]
        n = len(local); indices = np.arange(n)
        left, right = np.maximum(0, indices-radius), np.minimum(n, indices+radius)
        prefix = np.vstack((np.zeros((1, 2)), np.cumsum(local, axis=0)))
        a, b = indices-left, right-indices
        past[start:last+1] = np.divide(prefix[indices]-prefix[left], a[:, None],
            out=np.zeros_like(local), where=a[:, None] > 0)
        future[start:last+1] = (prefix[right]-prefix[indices])/b[:, None]
        complete[start:last+1] = (a == radius) & (b == radius)
    return past, future, complete & active


def evidence(model, control_matrix, probabilities, blocks, fps, base_cost):
    """Return bounded H/A switch costs and an optional early-exit Boolean mask.

    The decoder still enforces a .16s floor and adds a one-unit penalty to any
    enabled exit before its ordinary .4s minimum. This module does not change
    dead/restart transitions, predict missing ball as dead, or consume labels.
    """
    validate_model(model)
    if isinstance(base_cost, bool) or not np.isfinite(base_cost) or base_cost < 0:
        raise ValueError("Base team switch cost must be finite and nonnegative")
    p = np.asarray(probabilities, dtype=float)
    classifier = model["classifier"]
    probability = boundary.predict_probability(classifier, control_matrix, p, blocks, fps)
    prior = classifier["prior"]
    bounded = np.clip(probability, 1e-12, 1-1e-12)
    # Log-odds subtraction is stable and avoids dividing by a tiny event prior.
    relative_log_odds = np.log(bounded)-np.log1p(-bounded)-np.log(prior)+np.log1p(-prior)
    strength = np.clip(relative_log_odds / np.log(POLICY["saturation_odds_ratio"]), 0, 1)
    sections, active = boundary._blocks(blocks, len(p))
    strength[~active] = 0
    for start, _ in sections: strength[start] = 0
    past, future, complete = _context(p, blocks, fps)
    margin = POLICY["minimum_context_margin"]
    reversal = ((past[:, 0] >= margin) & (future[:, 0] <= -margin)) | (
        (past[:, 0] <= -margin) & (future[:, 0] >= margin))
    maximum_dead = POLICY["maximum_context_dead_probability"]
    live_context = (past[:, 1] <= maximum_dead) & (future[:, 1] <= maximum_dead) & (p[:, 2] <= maximum_dead)
    # V5's unrestricted boundary discount increased short-spell recall while
    # losing precision. Require supported opposite team context for *any*
    # discount; the larger strength threshold then gates the shorter exit.
    supported = complete & reversal & live_context
    costs = float(base_cost)*(1-POLICY["maximum_cost_reduction"]*strength*supported)
    allowed = supported & (strength >= POLICY["early_exit_minimum_strength"])
    return {"team_switch_costs": costs, "early_exit_allowed": allowed}


def switch_costs(model, control_matrix, probabilities, blocks, fps, base_cost):
    return evidence(model, control_matrix, probabilities, blocks, fps, base_cost)["team_switch_costs"]


def short_exit_allowed(model, control_matrix, probabilities, blocks, fps):
    return evidence(model, control_matrix, probabilities, blocks, fps, 0.)["early_exit_allowed"]
