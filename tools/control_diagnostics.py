"""External-label diagnostics and a training-only spell weighting utility.

This module is never imported by inference. Call training_spell_weights only on
training-match references; diagnostics may inspect evaluation labels after scoring.
"""

import numpy as np

from evaluation import possession_metrics as pm


def _codes(labels):
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("Labels must be a one-dimensional H/A/D/X vector")
    if np.issubdtype(values.dtype, np.integer):
        if not np.isin(values, [pm.H, pm.A, pm.D, pm.X]).all():
            raise ValueError("Integer label codes must be in 0..3")
        return values.astype(np.int8, copy=True)
    if not np.isin(values, pm.LABELS).all():
        raise ValueError("Labels must be H/A/D/X or integer codes 0..3")
    return pm.label_codes(values)


def _fps(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("FPS must be positive and finite")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("FPS must be positive and finite")
    return float(value)


def diagnostic(reference, prediction, fps=25):
    """Return turnover timing and official IoU=.5 recall by GT spell duration.

    Only adjacent H-to-A or A-to-H changes count as direct turnovers. Raw predicted
    X/D gaps are not bridged for this diagnostic. Matching is one-to-one, typed by
    both origin and destination, within one second and within the same contiguous
    known-reference run, so even a one-frame GT X gap prevents cross-gap matching.
    Earliest-to-earliest matching maximizes cardinality; latency is diagnostic,
    not a claim of minimum total timing error.

    Spell recall follows the official evaluator, including prediction abstention
    filling inside GT-known runs. Empty duration bins have recall=None.
    """
    fps = _fps(fps)
    gt, pred = _codes(reference), _codes(prediction)
    if gt.shape != pred.shape:
        raise ValueError("Reference and prediction lengths differ")
    known_runs = pm.valid_runs(gt)
    pairs, missed, spurious = [], [], []
    n_gt = n_pred = 0
    for run in known_runs:
        actual = [t for t in pm.transitions(gt, [run]) if t[1] < pm.D and t[2] < pm.D]
        estimated = [t for t in pm.transitions(pred, [run]) if t[1] < pm.D and t[2] < pm.D]
        matched, absent, extra = pm.match_transitions(actual, estimated, fps, typed="pair")
        n_gt += len(actual)
        n_pred += len(estimated)
        pairs.extend(matched)
        missed.extend(absent)
        spurious.extend(extra)
    precision = len(pairs) / n_pred if n_pred else 0.0
    recall = len(pairs) / n_gt if n_gt else 0.0
    errors = [abs(b[0] - a[0]) / fps for a, b in pairs]
    transitions = {
        "tolerance_seconds": 1.0, "tolerance_frames": fps,
        "n_gt": n_gt, "n_pred": n_pred, "n_matched": len(pairs),
        "n_missed": len(missed), "n_spurious": len(spurious),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "median_absolute_timing_error_seconds": float(np.median(errors)) if errors else None,
    }
    gt_spells = pm.possession_spells(gt, known_runs)
    filled = pm.fill_abstentions(pred, known_runs)
    pred_spells = pm.possession_spells(filled, known_runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matches = pm.matched_gt_flags(overlaps, len(gt_spells), 0.5)
    durations = np.asarray([(stop - start) / fps for start, stop, _ in gt_spells])
    strata = {}
    for name, low, high in (("<2s", 0, 2), ("2-5s", 2, 5),
                            ("5-15s", 5, 15), (">=15s", 15, np.inf)):
        selected = (durations >= low) & (durations < high)
        count = int(selected.sum())
        hits = int(matches[selected].sum())
        strata[name] = {"n_gt": count, "n_matched": hits,
                        "recall": hits / count if count else None}
    return {"transitions": transitions, "spell_iou": 0.5,
            "n_gt_spells": len(gt_spells), "n_pred_spells": len(pred_spells),
            "spell_recall_by_duration": strata}


def training_spell_weights(reference, fps=25):
    """Training-only capped inverse-square-root live-spell duration weights.

    Weight for a live spell of d seconds is clip(c/sqrt(d), .5, 3).
    A deterministic monotone solve chooses c so the mean over live frames is one;
    unlike normalize-after-clipping, this preserves both final caps. Dead/unknown
    frames receive zero. Empty/all-dead references return all-zero weights.
    """
    fps = _fps(fps)
    codes = _codes(reference)
    weights = np.zeros(len(codes), dtype=float)
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    if not spells:
        return weights
    lengths = np.asarray([stop - start for start, stop, _ in spells], dtype=float)
    inverse = np.sqrt(fps / lengths)
    low, high = 0.0, 3.0 / float(inverse.min())
    live_frames = float(lengths.sum())
    # Eighty fixed steps make normalization deterministic and machine-accurate.
    for _ in range(80):
        mid = (low + high) / 2
        mean = float(np.sum(lengths * np.clip(mid * inverse, .5, 3)) / live_frames)
        if mean < 1:
            low = mid
        else:
            high = mid
    per_spell = np.clip(((low + high) / 2) * inverse, .5, 3)
    for (start, stop, _), weight in zip(spells, per_spell):
        weights[start:stop] = weight
    return weights
