"""Training-only labels for a portable, bounded possession-boundary model."""
import numpy as np
from possession.boundary import FEATURE_NAMES, _blocks, validate_model


def targets(labels, blocks, fps, radius_seconds=.16):
    """Mark neighborhoods of direct H/A changes; never bridge D/X or periods."""
    y = np.asarray(labels)
    if y.ndim != 1 or not np.isin(y, list("HADX")).all(): raise ValueError("Invalid boundary labels")
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0 or radius_seconds != .16:
        raise ValueError("Invalid boundary target settings")
    sections, active = _blocks(blocks, len(y)); result = np.zeros(len(y))
    usable = active & np.isin(y, ["H", "A"])
    radius = max(1, round(radius_seconds * fps))
    for start, last in sections:
        usable[start] = False
        for index in range(start+1, last+1):
            if y[index] in ("H", "A") and y[index-1] in ("H", "A") and y[index] != y[index-1]:
                left, right = index, index
                # Stop at unknown/dead barriers even inside the target window.
                while left > max(start, index-radius) and y[left-1] in ("H", "A"): left -= 1
                while right < min(last, index+radius) and y[right+1] in ("H", "A"): right += 1
                result[left:right+1] = 1
    return result, usable


def fit(items, *, metadata=None, stride=5, l2=.01):
    """Each training item: X=boundary features, y=labels, blocks, fps, optional w.

    Caller excludes outer/inner evaluation matches. Sampling restarts each half.
    Balanced logistic training is corrected back to the training class prior;
    this is an estimated neighborhood probability, not a calibration guarantee.
    """
    from scipy.optimize import minimize
    if not items or not isinstance(stride, int) or stride < 1 or not np.isfinite(l2) or l2 <= 0:
        raise ValueError("Invalid boundary training settings")
    matrices, labels, weights = [], [], []
    for item in items:
        x = np.asarray(item["X"], dtype=float); y = np.asarray(item["y"])
        if x.shape != (len(y), len(FEATURE_NAMES)) or not np.isfinite(x).all(): raise ValueError("Invalid training matrix")
        target, usable = targets(y, item["blocks"], item["fps"])
        w = np.asarray(item.get("w", np.ones(len(y))), dtype=float)
        if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any(): raise ValueError("Invalid boundary weights")
        sections, _ = _blocks(item["blocks"], len(y))
        indices = np.concatenate([np.arange(a+1, b+1, stride) for a, b in sections]) if sections else np.zeros(0, dtype=int)
        indices = indices[usable[indices] & (w[indices] > 0)]
        if len(indices):
            matrices.append(x[indices]); labels.append(target[indices]); weights.append(w[indices])
    if not matrices: raise ValueError("No live boundary training observations")
    x, y, original = np.concatenate(matrices), np.concatenate(labels), np.concatenate(weights)
    positive, negative = original[y == 1].sum(), original[y == 0].sum()
    if positive <= 0 or negative <= 0: raise ValueError("Boundary fitting requires positive and negative examples")
    normalized = original / original.sum()
    mean = np.sum(x * normalized[:, None], axis=0)
    scale = np.sqrt(np.sum((x-mean)**2 * normalized[:, None], axis=0)); scale[scale < 1e-8] = 1
    x = np.column_stack((np.clip((x-mean)/scale, -8, 8), np.ones(len(x))))
    w = original * np.where(y == 1, .5/positive, .5/negative)
    def objective(c):
        z = x @ c; p = 1/(1+np.exp(-np.clip(z, -35, 35)))
        loss = np.sum(w * (np.logaddexp(0, z)-y*z)) + .5*l2*np.sum(c[:-1]**2)
        gradient = x.T @ (w*(p-y)); gradient[:-1] += l2*c[:-1]
        return loss, gradient
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter":300,"ftol":1e-11,"gtol":1e-7})
    if not result.success or not np.isfinite(result.x).all(): raise ValueError("Boundary optimization failed: "+result.message)
    prior = float(positive/(positive+negative))
    model = {"schema_version":1,"kind":"local_possession_boundary","feature_names":list(FEATURE_NAMES),
        "mean":mean.tolist(),"scale":scale.tolist(),"coefficients":result.x[:-1].tolist(),
        "intercept":float(result.x[-1]+np.log(positive/negative)),"prior":prior,
        "target_radius_seconds":.16,"maximum_cost_reduction":.75,
        "training":{"sample_stride":stride,"rows":len(y),"positive_rows":int(y.sum()),"l2":l2,
                    "objective":float(result.fun),"balanced_fit_prior_corrected":True,
                    "calibration_claimed":False},"metadata":metadata or {}}
    return validate_model(model)
