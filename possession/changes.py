"""Observation-only evidence for contested touches and possession boundaries.

All columns reverse sign on exchanging teams. Short offline temporal windows may
look forwards, but never cross a period boundary. Player slots need not retain
identity: only per-frame team minima are used. These are candidate learned-model
inputs, not hard rules that force a turnover at every touch.
"""

import numpy as np

from .control import _directional_mean


BASE_NAMES = (
    "uncontested_control", "contested_control", "reach_advantage",
    "contact_control", "deceleration_control", "acceleration_control",
    "direction_change_control", "close_fast_control",
)
CONTEXT_COLUMNS = (1, 2, 3, 6)
CONTEXT_SECONDS = (0.25, 0.75)
TRAJECTORY_SECONDS = (-0.6, -0.2, 0.2, 0.6)
FEATURE_NAMES = BASE_NAMES + tuple(
    f"{BASE_NAMES[column]}_{side}_{seconds:g}s"
    for seconds in CONTEXT_SECONDS for side in ("past", "future")
    for column in CONTEXT_COLUMNS
) + tuple(f"trajectory_reach_{seconds:+g}s" for seconds in TRAJECTORY_SECONDS)


def _shift(values, blocks, offset):
    """Return values at t+offset, with unavailable context explicitly missing."""
    output = np.full_like(values, np.nan, dtype=float)
    for _, start, last in blocks:
        left, right = max(start, start - offset), min(last + 1, last + 1 - offset)
        if right > left:
            output[left:right] = values[left + offset:right + offset]
    return output


def _reach(points, speeds, teams, ball, pitch):
    """Team distance/reach evidence using bounded speed, never slot velocity."""
    valid = (np.isfinite(points).all(axis=2) & (points >= -0.2).all(axis=2)
             & (points <= 1.2).all(axis=2))
    distance = np.linalg.norm(points * pitch - ball[:, None], axis=2)
    distance[~valid | ~np.isfinite(distance)] = np.inf
    # A nonzero reaction/reach speed avoids overconfident infinite arrival times
    # for stationary players. This is a feature, not a physical interception model.
    speed = np.where(np.isfinite(speeds) & (speeds >= 0) & (speeds <= 13), speeds, 2.)
    travel = distance / (2 + np.minimum(speed, 8))
    nearest = np.column_stack([np.min(distance[:, teams == team], axis=1) for team in (0, 1)])
    arrival = np.column_stack([np.min(travel[:, teams == team], axis=1) for team in (0, 1)])
    coverage = np.where(np.isfinite(nearest).all(axis=1), 1., .25)
    nearest = np.minimum(nearest, 50)
    arrival = np.minimum(arrival, 20)
    reach = np.tanh((arrival[:, 1] - arrival[:, 0]) / .75)
    # No nearby players means little evidence even if distant teams differ.
    reach *= np.exp(-np.min(nearest, axis=1) / 8) * coverage
    return nearest, reach, coverage


def feature_matrix(tracking, features, blocks, pitch, fps):
    """Return n x 28 signed features, without labels or event outcome fields.

    Pitch dimensions are metres; Tracking positions are normalized fractions and
    Features.ball is already cleaned into metres. No fitted state is used here.
    """
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("Pitch dimensions must be two positive finite metre values")
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    teams = np.asarray(tracking.teams)
    if not np.isin(teams, [0, 1]).all() or not all((teams == team).any() for team in (0, 1)):
        raise ValueError("Change observations need both team memberships")
    n = len(tracking.ball)
    ball = np.asarray(features.ball, dtype=float)
    reliability = np.clip(np.nan_to_num(features.reliability, nan=0), 0, 1)
    if ball.shape != (n, 2) or reliability.shape != (n,):
        raise ValueError("Change observations have inconsistent shapes")
    radius = max(1, round(.12 * fps))
    before = (ball - _shift(ball, blocks, -radius)) * fps / radius
    after = (_shift(ball, blocks, radius) - ball) * fps / radius
    incoming, outgoing = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    good = np.isfinite(incoming) & np.isfinite(outgoing) & (incoming <= 45) & (outgoing <= 45)
    contact = np.zeros(n)
    deceleration = np.zeros(n)
    acceleration = np.zeros(n)
    turn = np.zeros(n)
    contact[good] = np.clip(np.linalg.norm(after[good] - before[good], axis=1) / 10, 0, 1)
    deceleration[good] = np.clip((incoming[good] - outgoing[good]) / 10, 0, 1)
    acceleration[good] = np.clip((outgoing[good] - incoming[good]) / 10, 0, 1)
    moving = good & (incoming > 2) & (outgoing > 2)
    cosine = np.sum(before[moving] * after[moving], axis=1) / (incoming[moving] * outgoing[moving])
    turn[moving] = np.clip((1 - cosine) / 2, 0, 1)
    base = np.zeros((n, len(BASE_NAMES)))
    trajectories = np.zeros((n, len(TRAJECTORY_SECONDS)))
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        points = np.asarray(tracking.players[start:end], dtype=float)
        speeds = np.asarray(tracking.player_speed[start:end], dtype=float)
        near, reach, coverage = _reach(points, speeds, teams, ball[start:end], pitch)
        proximity = np.exp(-near[:, 0] / 2) - np.exp(-near[:, 1] / 2)
        pressure = np.exp(-np.max(near, axis=1) / 4)
        support = reliability[start:end] * coverage
        fast = np.clip(np.nan_to_num(features.ball_speed[start:end], nan=0) / 15, 0, 1)
        base[start:end] = np.column_stack((
            proximity * (1 - pressure), proximity * pressure, reach,
            proximity * contact[start:end], proximity * deceleration[start:end],
            proximity * acceleration[start:end], proximity * turn[start:end], proximity * fast,
        )) * support[:, None]
    # Current players' reach to nearby past/future ball locations. No player slot
    # is compared across frames; this is robust to tracking identity fragmentation.
    for column, seconds in enumerate(TRAJECTORY_SECONDS):
        offset = max(1, round(abs(seconds) * fps)) * (1 if seconds > 0 else -1)
        context = _shift(ball, blocks, offset)
        context_support = np.nan_to_num(_shift(reliability, blocks, offset), nan=0)
        for start in range(0, n, 4096):
            end = min(start + 4096, n)
            _, reach, _ = _reach(np.asarray(tracking.players[start:end], dtype=float),
                                np.asarray(tracking.player_speed[start:end], dtype=float),
                                teams, context[start:end], pitch)
            trajectories[start:end, column] = reach * reliability[start:end] * context_support[start:end]
    active = np.zeros(n, dtype=bool)
    for _, start, last in blocks:
        active[start:last + 1] = True
    base[~active] = 0
    trajectories[~active] = 0
    columns = [base]
    for seconds in CONTEXT_SECONDS:
        radius = max(1, round(seconds * fps))
        for future in (False, True):
            columns.append(np.column_stack([
                _directional_mean(base[:, column], blocks, radius, future)
                for column in CONTEXT_COLUMNS
            ]))
    columns.append(trajectories)
    result = np.column_stack(columns)
    if result.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Invalid possession-change feature matrix")
    return result
