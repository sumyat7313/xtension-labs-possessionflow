"""Deterministic v6 loss ablation: emphasize stoppage duration and boundaries.

Uses the unchanged v5 joint network/export schema. Separate D-loss weights preserve
both class masses per half; the conditional H/A objective and scaling stay unchanged.
This module consumes training references only and is never imported by inference.
"""
import numpy as np
from possession import joint_temporal as joint
from tools.train_joint import make_network, export_model

WEIGHTING = {
    "dead_duration_anchor_seconds": 5.0,
    "boundary_radius_seconds": 1.0,
    "boundary_multiplier": 2.0,
    "multiplier_minimum": 0.5,
    "multiplier_maximum": 3.0,
    "class_normalization": "preserve original weighted mass separately for D/live in each half",
    "conditional_home_loss": "unchanged v5 weights",
    "unknown_policy": "X and zero-weight observations split valid runs",
}


def _spans(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def stoppage_weights(labels, base_weights, fps=5):
    """Redistribute D BCE attention without changing each class's total mass.

    For D spells, use sqrt(5 seconds/duration). Multiply neighborhoods within
    one second of an observed live/D transition by two, including both sides.
    Repeated neighborhoods do not compound. X or zero-weight gaps split runs.
    Then normalize each class with multipliers bounded to [.5, 3]. Never call
    this function on held-out labels before prediction; it is training-only.
    """
    y, w = np.asarray(labels), np.asarray(base_weights, dtype=float)
    if (isinstance(fps, (bool, np.bool_)) or not isinstance(fps, (int, float, np.number))
            or not np.isfinite(fps) or fps <= 0 or y.ndim != 1 or w.shape != y.shape
            or not np.isin(y, list("HADX")).all() or not np.isfinite(w).all() or (w < 0).any()):
        raise ValueError("Invalid stoppage training labels, weights or FPS")
    active = (y != "X") & (w > 0)
    raw = np.ones(len(y), dtype=float)
    near = np.zeros(len(y), dtype=bool)
    radius = round(WEIGHTING["boundary_radius_seconds"] * fps)
    for start, stop in _spans(active):
        dead = y[start:stop] == "D"
        for a, b in _spans(dead):
            raw[start+a:start+b] = np.sqrt(WEIGHTING["dead_duration_anchor_seconds"] * fps / (b-a))
        for index in np.flatnonzero(dead[1:] != dead[:-1]) + start + 1:
            near[max(start, index-radius):min(stop, index+radius)] = True
    raw[near] *= WEIGHTING["boundary_multiplier"]
    output = np.zeros(len(y), dtype=float)
    lower, upper = WEIGHTING["multiplier_minimum"], WEIGHTING["multiplier_maximum"]
    for cls in (y == "D", y != "D"):
        chosen = active & cls
        if not chosen.any():
            continue
        r, weight = raw[chosen], w[chosen]
        target = weight.sum()
        low, high = 0.0, upper / r.min()
        for _ in range(80):
            mid = (low + high) / 2
            if np.dot(weight, np.clip(mid*r, lower, upper)) < target:
                low = mid
            else:
                high = mid
        output[chosen] = weight * np.clip((low+high)/2*r, lower, upper)
    return output


def fit(sequences, *, seed=10527, epochs=8, metadata=None):
    """Each sequence is one half at 5 Hz: X, Z, y, w, base_home, base_dead.

    Positive weights must include live and dead examples. X references and zero
    weight rows do not affect scaling or the objective. Inputs remain observation
    context inside each half. The caller is responsible for whole-match splits.
    """
    import torch
    from torch.nn import functional as F
    if not sequences or type(epochs) is not int or epochs < 1 or type(seed) is not int:
        raise ValueError("Need sequences, an integer seed and positive epochs")
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    total = 0.; odd_square = np.zeros(len(joint.CONTROL_NAMES)); ctx_sum = np.zeros(len(joint.CONTEXT_NAMES)); ctx_square = ctx_sum.copy()
    rows = live_rows = dead_rows = 0
    class_rows = {label: 0 for label in "HAD"}
    checked = []
    for item in sequences:
        x, z = joint.observations(item["X"], item["Z"])
        y, w = np.asarray(item["y"]), np.asarray(item["w"], dtype=float)
        bh, bd = np.asarray(item["base_home"], dtype=float), np.asarray(item["base_dead"], dtype=float)
        if (y.shape != (len(x),) or w.shape != y.shape or bh.shape != y.shape or bd.shape != y.shape
                or not np.isin(y, ["H", "A", "D", "X"]).all() or not np.isfinite(w).all() or (w < 0).any()
                or not np.isfinite(bh).all() or not np.isfinite(bd).all()):
            raise ValueError("Invalid joint training labels, weights or baseline logits")
        good = (y != "X") & (w > 0); context = joint.context_features(x, z)
        odd_square += np.sum(x[good] ** 2 * w[good, None], axis=0)
        ctx_sum += np.sum(context[good] * w[good, None], axis=0)
        ctx_square += np.sum(context[good] ** 2 * w[good, None], axis=0); total += w[good].sum()
        rows += int(good.sum()); live_rows += int((good & (y != "D")).sum()); dead_rows += int((good & (y == "D")).sum())
        for label in class_rows:
            class_rows[label] += int((good & (y == label)).sum())
        checked.append((x, z, y, w * good, bh, bd))
    if total <= 0 or not all(class_rows.values()):
        raise ValueError("Joint training requires positive-weight H, A and D examples")
    control_scale = np.sqrt(odd_square / total); control_scale[control_scale < 1e-8] = 1
    context_mean = ctx_sum / total
    context_scale = np.sqrt(np.maximum(0, ctx_square / total - context_mean ** 2)); context_scale[context_scale < 1e-8] = 1
    net = make_network(); optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
    scaling = {"control_scale": control_scale, "context_mean": context_mean, "context_scale": context_scale}
    prepared = []; windows = []; width = 128; halo = sum(joint.DILATIONS)
    for i, (x, z, y, w, bh, bd) in enumerate(checked):
        odd, even = joint.normalized_inputs(scaling, x, z)
        prepared.append((odd.astype(np.float32), even.astype(np.float32), (y == "H").astype(np.float32),
                         (y == "D").astype(np.float32), w.astype(np.float32), bh.astype(np.float32), bd.astype(np.float32),
                         stoppage_weights(y, w, fps=5).astype(np.float32)))
        windows.extend((i, start) for start in range(0, len(y), width) if w[start:start + width].sum() > 0)
    rng = np.random.default_rng(seed); losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(windows)); loss_sum = weight_sum = 0.
        for at in range(0, len(order), 16):
            batch = [windows[k] for k in order[at:at + 16]]; size = len(batch)
            bx = np.zeros((size, width + 2 * halo, len(joint.CONTROL_NAMES)), dtype=np.float32)
            bz = np.zeros((size, width + 2 * halo, len(joint.CONTEXT_NAMES)), dtype=np.float32)
            bm = np.zeros((size, width + 2 * halo), dtype=np.float32)
            yh = np.zeros((size, width), dtype=np.float32); yd = yh.copy(); bw = yh.copy(); bh = yh.copy(); bd = yh.copy(); bdw = yh.copy()
            for j, (i, start) in enumerate(batch):
                x, z, h, d, w, home, death, dw = prepared[i]; stop = min(start + width, len(x))
                left, right = max(0, start - halo), min(len(x), start + width + halo)
                region = slice(left - start + halo, right - start + halo)
                bx[j, region] = x[left:right]; bz[j, region] = z[left:right]; bm[j, region] = 1
                for target, source in ((yh, h), (yd, d), (bw, w), (bh, home), (bd, death), (bdw, dw)):
                    target[j, :stop - start] = source[start:stop]
            optimizer.zero_grad(set_to_none=True)
            adjustment = net(torch.from_numpy(bx), torch.from_numpy(bz), torch.from_numpy(bm))[:, halo:halo + width]
            death = torch.from_numpy(yd); weight = torch.from_numpy(bw)
            dead_loss = F.binary_cross_entropy_with_logits(adjustment[:, :, 1] + torch.from_numpy(bd), death, reduction="none")
            home_loss = F.binary_cross_entropy_with_logits(adjustment[:, :, 0] + torch.from_numpy(bh), torch.from_numpy(yh), reduction="none")
            loss = (torch.from_numpy(bdw) * dead_loss + weight * (1 - death) * home_loss).sum() / weight.sum()
            if not torch.isfinite(loss): raise ValueError("Joint training produced a nonfinite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            loss_sum += float(loss.detach()) * float(weight.sum()); weight_sum += float(weight.sum())
        losses.append(loss_sum / weight_sum)
        print(f"  boundary-focused joint epoch {epoch + 1}/{epochs}: weighted H/A/D loss {losses[-1]:.6f}", flush=True)
    return export_model(net, control_scale, context_mean, context_scale, metadata=metadata, training={
        "seed": seed, "epochs": epochs, "positive_weight_rows": rows, "live_rows": live_rows, "dead_rows": dead_rows,
        "class_rows": class_rows,
        "losses": losses, "objective": "boundary-focused D BCE plus unchanged v5 H-given-live BCE",
        "stoppage_weighting": WEIGHTING,
        "optimizer": "AdamW lr=.002 weight_decay=.01 clip_grad_norm=5", "cpu_threads": 1, "torch_version": torch.__version__})
