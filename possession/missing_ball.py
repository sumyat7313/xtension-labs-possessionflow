"""Weak, symmetric possession evidence when the ball is not observed.

This offline branch does not invent a ball trajectory. It supplies bounded context
from the last/next reliable observation and permutation-invariant team geometry.
All columns are H-minus-A, zero for a reliable current ball, and half-local.
Coefficients must be learned on separate matches; these features are not labels.
"""

import numpy as np


DECAY_SECONDS = (0.7, 2.0, 5.0)
MAX_CONTEXT_SECONDS = 8.0
SHAPE_NAMES = (
    "x_spread", "y_spread", "radial_spread", "mean_speed", "fast_fraction",
    "speed_spread", "centroid_motion", "x_expansion", "y_expansion",
)
FEATURE_NAMES = tuple(
    f"missing_control_{side}_{seconds:g}s" for seconds in DECAY_SECONDS
    for side in ("past", "future")
) + ("missing_past_anchor_support", "missing_future_anchor_support",
     "missing_agreeing_context") + tuple(f"missing_{name}" for name in SHAPE_NAMES)


def _context_indices(good, blocks, fps):
    """Nearest reliable observation on either side, never beyond eight seconds."""
    n = len(good)
    past, future = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
    for _, start, last in blocks:
        index = np.arange(start, last + 1)
        before = np.maximum.accumulate(np.where(good[start:last + 1], index, -n - 1))
        after = np.minimum.accumulate(
            np.where(good[start:last + 1], index, 2 * n + 1)[::-1])[::-1]
        past[start:last + 1] = np.where((before >= start) & (index - before <= MAX_CONTEXT_SECONDS * fps), before, -1)
        future[start:last + 1] = np.where((after <= last) & (after - index <= MAX_CONTEXT_SECONDS * fps), after, -1)
    return past, future


def _mean_sorted(values, valid):
    """Sorting before reduction makes slot permutations numerically invariant."""
    count = valid.sum(axis=1)
    total = np.sort(np.where(valid, values, 0), axis=1).sum(axis=1)
    return np.divide(total, count, out=np.zeros(len(values)), where=count > 0)


def _team_geometry(tracking, pitch, blocks, fps):
    n = len(tracking.ball)
    summaries = np.zeros((n, 2, len(SHAPE_NAMES)))
    centers = np.zeros((n, 2, 2))
    counts = np.zeros((n, 2))
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        for team in (0, 1):
            slots = np.asarray(tracking.teams) == team
            points = np.asarray(tracking.players[start:end, slots], dtype=float)
            valid = (np.isfinite(points).all(axis=2) & (points >= -.2).all(axis=2)
                     & (points <= 1.2).all(axis=2))
            points = points * pitch
            count = valid.sum(axis=1)
            counts[start:end, team] = count
            center = np.column_stack([_mean_sorted(points[:, :, axis], valid) for axis in (0, 1)])
            centers[start:end, team] = center
            variances = np.column_stack([
                _mean_sorted((points[:, :, axis] - center[:, axis, None]) ** 2, valid)
                for axis in (0, 1)])
            spread = np.sqrt(variances)
            speed = np.asarray(tracking.player_speed[start:end, slots], dtype=float)
            speed_valid = valid & np.isfinite(speed) & (speed >= 0) & (speed <= 13)
            mean_speed = _mean_sorted(speed, speed_valid)
            speed_spread = np.sqrt(_mean_sorted((speed - mean_speed[:, None]) ** 2, speed_valid))
            summaries[start:end, team, :6] = np.column_stack((
                spread[:, 0] / 20, spread[:, 1] / 15,
                np.sqrt(variances.sum(axis=1)) / 25, mean_speed / 5,
                _mean_sorted((speed > 4).astype(float), speed_valid), speed_spread / 3,
            ))
    radius = max(1, round(.4 * fps))
    for _, start, last in blocks:
        if last - start < 2 * radius:
            continue
        center = slice(start + radius, last - radius + 1)
        before = slice(start, last - 2 * radius + 1)
        after = slice(start + 2 * radius, last + 1)
        # Centroid motion is a team aggregate, never a presumed persistent ID.
        # Strong visibility changes make its derivative unreliable.
        stable = ((np.minimum(counts[before], counts[after]) >= 4)
                  & (np.abs(counts[after] - counts[before]) <= 1))
        velocity = np.linalg.norm(centers[after] - centers[before], axis=2) * fps / (2 * radius)
        summaries[center, :, 6] = np.minimum(velocity / 5, 2) * stable
        summaries[center, :, 7:9] = np.clip(
            (summaries[after, :, :2] - summaries[before, :, :2]) * fps / (2 * radius),
            -2, 2) * stable[:, :, None]
    coverage = np.minimum(1, counts.min(axis=1) / 5)
    return np.clip(summaries[:, 0] - summaries[:, 1], -2, 2) * coverage[:, None], counts


def feature_matrix(tracking, features, blocks, pitch, fps):
    """Return finite n-by-18 signed features using observations only.

    Past/future evidence expires after eight seconds. Team geometry is downweighted
    with incomplete coverage; current player distance to an old ball observation
    is an anchor-support feature, explicitly not a recovered ball location.
    """
    n = len(tracking.ball)
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("Pitch dimensions must be two positive finite metre values")
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    teams = np.asarray(tracking.teams)
    if not np.isin(teams, [0, 1]).all() or not all((teams == t).any() for t in (0, 1)):
        raise ValueError("Missing-ball observations need both team memberships")
    if (np.asarray(tracking.players).shape != (n, len(teams), 2)
            or np.asarray(tracking.player_speed).shape != (n, len(teams))):
        raise ValueError("Invalid player observation shapes")
    active = np.zeros(n, dtype=bool)
    for _, start, last in blocks:
        if not 0 <= start <= last < n or active[start:last + 1].any():
            raise ValueError("Blocks must be disjoint inclusive frame ranges")
        active[start:last + 1] = True
    reliable = np.clip(np.nan_to_num(features.reliability, nan=0), 0, 1)
    good = (np.asarray(features.observed, dtype=bool) & (reliable >= .75)
            & np.isfinite(features.ball).all(axis=1) & active)
    past, future = _context_indices(good, blocks, fps)
    indices = (past, future)
    ages = [np.abs(np.arange(n) - index) / fps for index in indices]
    valid = [index >= 0 for index in indices]
    evidence = np.clip(np.nan_to_num(features.team_log_odds, nan=0), -4, 4) / 4
    context = [evidence[np.maximum(0, index)] * ok for index, ok in zip(indices, valid)]
    columns = [value * np.exp(-age / seconds) for seconds in DECAY_SECONDS
               for value, age in zip(context, ages)]
    shape, counts = _team_geometry(tracking, pitch, blocks, fps)
    coverage = np.minimum(1, counts.min(axis=1) / 3)
    for index, age, ok in zip(indices, ages, valid):
        anchor = np.asarray(features.ball)[np.maximum(0, index)]
        nearest = np.full((n, 2), 30.0)
        for start in range(0, n, 4096):
            end = min(n, start + 4096)
            points = np.asarray(tracking.players[start:end], dtype=float)
            available = (np.isfinite(points).all(axis=2) & (points >= -.2).all(axis=2)
                         & (points <= 1.2).all(axis=2))
            distance = np.linalg.norm(points * pitch - anchor[start:end, None], axis=2)
            distance[~available | ~np.isfinite(distance)] = np.inf
            for team in (0, 1):
                nearest[start:end, team] = np.minimum(30, distance[:, teams == team].min(axis=1))
        support = ((np.exp(-nearest[:, 0] / 3) - np.exp(-nearest[:, 1] / 3))
                   * coverage * np.exp(-age / 2) * ok)
        columns.append(support)
    agreement = (context[0] * context[1] > 0)
    columns.append((columns[2] + columns[3]) * .5 * agreement)
    result = np.column_stack((*columns, shape))
    result *= ((1 - reliable) * active)[:, None]
    if result.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Invalid missing-ball feature matrix")
    return result
