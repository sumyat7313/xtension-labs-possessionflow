"""Geometry and motion evidence; no player identities or match labels required."""

from dataclasses import dataclass
import numpy as np


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def rolling_mean(values, radius):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
    counts = np.r_[0, np.cumsum(valid)]
    index = np.arange(len(values))
    left = np.maximum(0, index - radius)
    right = np.minimum(len(values), index + radius + 1)
    number = counts[right] - counts[left]
    return np.divide(total[right] - total[left], number,
                     out=np.full(len(values), np.nan), where=number > 0)


def smooth_in_halves(values, blocks, radius, fill=0.0):
    result = np.full(len(values), fill, dtype=np.float64)
    for _, start, last in blocks:
        result[start:last + 1] = rolling_mean(values[start:last + 1], radius)
    return result


@dataclass
class Features:
    ball: np.ndarray
    observed: np.ndarray
    reliability: np.ndarray
    ball_speed: np.ndarray
    nearest: np.ndarray
    team_log_odds: np.ndarray
    stationary: np.ndarray
    still_players: np.ndarray
    outside: np.ndarray
    moving: np.ndarray
    missing: np.ndarray
    player_count: np.ndarray
    rejected_spikes: int
    interpolated_frames: int


def clean_ball(raw, pitch, blocks, fps, config):
    ball = np.asarray(raw, dtype=np.float64).copy()
    plausible = np.isfinite(ball).all(axis=1) & (ball >= -0.15).all(axis=1) & (ball <= 1.15).all(axis=1)
    ball[~plausible] = np.nan
    ball *= np.asarray(pitch)
    rejected = 0
    for _, start, last in blocks:
        if last - start < 2:
            continue
        section = ball[start:last + 1]
        left = np.linalg.norm(section[1:-1] - section[:-2], axis=1)
        right = np.linalg.norm(section[1:-1] - section[2:], axis=1)
        bypass = np.linalg.norm(section[2:] - section[:-2], axis=1)
        step = config.max_ball_speed_mps / fps
        spikes = (left > step) & (right > step) & (bypass <= 2 * step)
        section[1:-1][spikes] = np.nan
        rejected += int(spikes.sum())
    observed = np.isfinite(ball).all(axis=1)
    weight = observed.astype(np.float64)
    max_gap = max(1, round(config.interpolation_seconds * fps))
    interpolated = 0
    for _, start, last in blocks:
        for left, right in runs(~observed[start:last + 1]):
            a, b = start + left, start + right
            if a == start or b > last or b - a > max_gap:
                continue
            elapsed = (b - a + 1) / fps
            if np.linalg.norm(ball[b] - ball[a - 1]) / elapsed > config.max_ball_speed_mps:
                continue
            fractions = np.arange(1, b - a + 1)[:, None] / (b - a + 1)
            ball[a:b] = ball[a - 1] + fractions * (ball[b] - ball[a - 1])
            weight[a:b] = config.interpolated_weight
            interpolated += int(b - a)
    return ball, observed, weight, rejected, interpolated


def extract(tracking, pitch, blocks, fps, config):
    n = len(tracking.ball)
    ball, observed, weight, rejected, interpolated = clean_ball(tracking.ball, pitch, blocks, fps, config)
    speed = np.asarray(tracking.ball_speed, dtype=np.float64).copy()
    speed[(speed < 0) | (speed > config.max_ball_speed_mps) | ~observed] = np.nan
    # A short central derivative avoids depending on the speed field alone.
    geometric = np.full(n, np.nan)
    for _, start, last in blocks:
        if last - start >= 4:
            geometric[start + 2:last - 1] = np.linalg.norm(
                ball[start + 4:last + 1] - ball[start:last - 3], axis=1) * fps / 4
    geometric[geometric > config.max_ball_speed_mps] = np.nan
    both = np.isfinite(speed) & np.isfinite(geometric)
    speed[both] = (speed[both] + geometric[both]) / 2
    only_geometry = ~np.isfinite(speed) & np.isfinite(geometric)
    speed[only_geometry] = geometric[only_geometry]

    nearest = np.full((n, 2), np.inf)
    counts = np.zeros(n, dtype=np.int16)
    player_motion = np.full(n, np.nan)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        points = tracking.players[start:end].astype(np.float64)
        valid = np.isfinite(points).all(axis=2) & (points >= -0.2).all(axis=2) & (points <= 1.2).all(axis=2)
        points[~valid] = np.nan
        distance = np.linalg.norm(points * pitch - ball[start:end, None], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        for team in (0, 1):
            nearest[start:end, team] = np.min(distance[:, tracking.teams == team], axis=1)
        counts[start:end] = valid.sum(axis=1)
        ps = tracking.player_speed[start:end]
        speed_ok = valid & np.isfinite(ps) & (ps >= 0) & (ps <= 13)
        denominator = speed_ok.sum(axis=1)
        player_motion[start:end] = np.divide(np.where(speed_ok, ps, 0).sum(axis=1), denominator,
                                            out=np.full(end - start, np.nan), where=denominator > 0)

    closest = np.min(nearest, axis=1)
    control = weight * np.exp(-np.maximum(closest - config.control_radius_m, 0) / config.control_decay_m)
    control *= 0.35 + 0.65 / (1 + (np.nan_to_num(speed, nan=0) / config.fast_ball_mps) ** 2)
    margin = np.zeros(n)
    both_teams = np.isfinite(nearest).all(axis=1)
    margin[both_teams] = (nearest[both_teams, 1] - nearest[both_teams, 0]) / config.distance_margin_m
    # Incomplete team coverage is weak evidence, not an infinitely decisive win.
    margin[np.isfinite(nearest[:, 0]) & ~np.isfinite(nearest[:, 1])] = 1
    margin[np.isfinite(nearest[:, 1]) & ~np.isfinite(nearest[:, 0])] = -1
    team_log_odds = np.clip(margin, -4, 4) * control

    radius = max(1, round(config.evidence_window_seconds * fps / 2))
    speed_smoothed = smooth_in_halves(speed, blocks, radius, fill=np.nan)
    visibility = smooth_in_halves(observed.astype(float), blocks, radius)
    stationary = np.clip(1 - np.nan_to_num(speed_smoothed, nan=100) / config.stationary_ball_mps, 0, 1) * visibility
    motion = smooth_in_halves(player_motion, blocks, radius, fill=np.nan)
    coverage = np.clip(counts / 12, 0, 1)
    still_players = np.clip(1 - np.nan_to_num(motion, nan=100) / config.still_players_mps, 0, 1) * coverage
    outside_raw = np.isfinite(ball).all(axis=1) & (
        (ball[:, 0] < -config.outside_margin_m) | (ball[:, 0] > pitch[0] + config.outside_margin_m) |
        (ball[:, 1] < -config.outside_margin_m) | (ball[:, 1] > pitch[1] + config.outside_margin_m))
    outside = smooth_in_halves(outside_raw.astype(float) * weight, blocks, radius)
    moving = np.clip(np.nan_to_num(speed_smoothed, nan=0) / 3, 0, 1) * visibility * (1 - outside)
    missing = 1 - smooth_in_halves(observed.astype(float), blocks, max(radius, fps))
    return Features(ball, observed, weight, speed, nearest, team_log_odds, stationary,
                    still_players, outside, moving, missing, counts, rejected, interpolated)
