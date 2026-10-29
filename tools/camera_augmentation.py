"""Input-only virtual-camera augmentation for external development matches.

This is a synthetic camera stress model, not a calibrated reconstruction of the
challenge camera. Its parameters are fixed without challenge possession labels.
Coordinates entering and leaving this module use the Tracking normalized pitch
convention. There are deliberately no possession labels in the API.
"""
from __future__ import annotations

import numpy as np

from possession.io import Tracking


CAMERA_AUGMENTATION_SPEC = {
    "version": 1,
    "camera_knot_seconds": 1.2,
    "camera_lag_seconds": 0.4,
    "field_width_metres": [42.0, 76.0],
    "field_height_metres": [32.0, 58.0],
    "camera_jitter_sd_metres": [8.0, 5.0],
    "position_noise_sd_metres": 0.15,
    "speed_noise_sd_metres_per_second": 0.3,
    "occlusion_block_seconds": 2.0,
    "player_occlusion_probability": 0.08,
    "ball_gap_probability_per_block": 0.22,
    "ball_gap_seconds": [0.2, 2.4],
    "slot_permutation_seconds": 6.0,
    "duplicate_probability_per_block": 0.16,
    "duplicate_seconds": [0.12, 0.4],
    "false_ball_probability_per_block": 0.035,
    "false_ball_seconds": [0.04, 0.12],
}


def _validate(tracking, blocks, pitch, fps, seed):
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("pitch must contain two finite positive dimensions")
    if not np.isscalar(fps) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n = len(tracking.ball)
    k = len(tracking.teams)
    for name, shape in (("players", (n, k, 2)), ("player_speed", (n, k)),
                        ("ball", (n, 2)), ("ball_speed", (n,))):
        values = np.asarray(getattr(tracking, name))
        if values.shape != shape or np.isinf(values).any():
            raise ValueError(f"Invalid {name} shape or infinite value")
    if np.asarray(tracking.teams).shape != (k,) or not np.isin(tracking.teams, [0, 1]).all():
        raise ValueError("teams must be a one-dimensional array of 0/1 values")
    blocks = list(blocks)
    previous = -1
    for block in blocks:
        if len(block) != 3:
            raise ValueError("blocks must contain (half, start, last-inclusive)")
        _, start, last = block
        if (not isinstance(start, (int, np.integer)) or
                not isinstance(last, (int, np.integer)) or
                not 0 <= start <= last < n or start <= previous):
            raise ValueError("blocks must be ordered, disjoint, and in bounds")
        previous = last
    return blocks, pitch


def _camera_path(ball_metres, players_metres, fps, rng):
    """Half-local delayed anchor with interpolated motions between camera knots.

    The knot interpolation is offline and can use the next knot; this is not
    presented as a causal live camera simulator.
    """
    spec = CAMERA_AUGMENTATION_SPEC
    n = len(ball_metres)
    valid_players = np.isfinite(players_metres).all(axis=2)
    count = valid_players.sum(axis=1)
    total = np.where(valid_players[..., None], players_metres, 0).sum(axis=1)
    centroid = np.divide(total, count[:, None], out=np.full((n, 2), np.nan),
                         where=count[:, None] > 0)
    anchor = np.where(np.isfinite(ball_metres).all(axis=1)[:, None], ball_metres, centroid)
    # Forward-fill observations only within this half. The fallback is supplied
    # by the caller, so even completely empty input has no warning or NaN path.
    valid = np.isfinite(anchor).all(axis=1)
    previous = np.maximum.accumulate(np.where(valid, np.arange(n), -1))
    held = anchor[np.maximum(previous, 0)].copy()
    held[previous < 0] = 0
    lag = max(0, round(spec["camera_lag_seconds"] * fps))
    held = held[np.maximum(np.arange(n) - lag, 0)]
    step = max(1, round(spec["camera_knot_seconds"] * fps))
    knots = np.unique(np.r_[np.arange(0, n, step), n - 1])
    locations = held[knots] + rng.normal(0, spec["camera_jitter_sd_metres"], (len(knots), 2))
    path = np.column_stack([np.interp(np.arange(n), knots, locations[:, j]) for j in range(2)])
    widths = rng.uniform(*spec["field_width_metres"], len(knots))
    heights = rng.uniform(*spec["field_height_metres"], len(knots))
    extent = np.column_stack((np.interp(np.arange(n), knots, widths),
                              np.interp(np.arange(n), knots, heights))) / 2
    return path, extent


def corrupt_camera_tracking(tracking, blocks, pitch, fps, seed):
    """Return a new Tracking with reproducible camera and detector corruption.

    ``blocks`` are ordered (half, first, last-inclusive) spans. Frames outside
    those spans are copied unchanged. Independent RNG streams and camera state
    prevent information or random-state propagation between halves. A virtual
    camera follows delayed input observations, clipping both ball and players
    to a rectangular field of view. Detector gaps, position/speed noise, track
    slot permutations, same-team duplicates in vacant slots, and brief false
    balls at visible player positions then simulate imperfect visual tracking.

    No existing NaN ball is recovered from its hidden original coordinate;
    inserted false detections use only surviving visible player observations.
    Slot count and team metadata are retained. This augmentation is an external
    development stress regime, not evidence of calibrated target-camera error.
    """
    blocks, pitch = _validate(tracking, blocks, pitch, fps, seed)
    result = Tracking(*(np.array(getattr(tracking, name), dtype=float, copy=True)
                        for name in ("players", "player_speed", "ball", "ball_speed")),
                      np.array(tracking.teams, copy=True), tuple(tracking.team_names))
    spec = CAMERA_AUGMENTATION_SPEC
    for block_index, (_, start, last) in enumerate(blocks):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), block_index]))
        stop = last + 1
        length = stop - start
        players = result.players[start:stop]
        speeds = result.player_speed[start:stop]
        ball = result.ball[start:stop]
        ball_speed = result.ball_speed[start:stop]
        # Centering lets an empty half use pitch center as its camera anchor.
        center, extent = _camera_path((ball - .5) * pitch,
                                       (players - .5) * pitch, fps, rng)
        player_visible = (np.abs((players - .5) * pitch - center[:, None]) <= extent[:, None]).all(axis=2)
        ball_visible = (np.abs((ball - .5) * pitch - center) <= extent).all(axis=1)
        players[~player_visible] = np.nan
        speeds[~player_visible] = np.nan
        ball[~ball_visible] = np.nan
        ball_speed[~ball_visible] = np.nan
        players += rng.normal(0, spec["position_noise_sd_metres"], players.shape) / pitch
        ball += rng.normal(0, spec["position_noise_sd_metres"], ball.shape) / pitch
        speeds[:] = np.maximum(0, speeds + rng.normal(0, spec["speed_noise_sd_metres_per_second"], speeds.shape))
        ball_speed[:] = np.maximum(0, ball_speed + rng.normal(0, spec["speed_noise_sd_metres_per_second"], ball_speed.shape))
        step = max(1, round(spec["occlusion_block_seconds"] * fps))
        for a in range(0, length, step):
            b = min(a + step, length)
            absent = rng.random(len(result.teams)) < spec["player_occlusion_probability"]
            players[a:b, absent] = np.nan
            speeds[a:b, absent] = np.nan
            if rng.random() < spec["ball_gap_probability_per_block"]:
                first = int(rng.integers(a, b))
                end = min(length, first + max(1, round(rng.uniform(*spec["ball_gap_seconds"]) * fps)))
                ball[first:end] = np.nan
                ball_speed[first:end] = np.nan
            if rng.random() < spec["duplicate_probability_per_block"]:
                first = int(rng.integers(a, b))
                end = min(b, first + max(1, round(rng.uniform(*spec["duplicate_seconds"]) * fps)))
                # Copy into a vacant slot of the same team; never overwrite a
                # different real player or create a new team/slot identity.
                for side in (0, 1):
                    visible = np.isfinite(players[first]).all(axis=1)
                    sources = np.flatnonzero((result.teams == side) & visible)
                    destinations = np.flatnonzero((result.teams == side) & ~visible)
                    if len(sources) and len(destinations):
                        src, dst = int(rng.choice(sources)), int(rng.choice(destinations))
                        eligible = ~np.isfinite(players[first:end, dst]).all(axis=1)
                        frames = np.arange(first, end)[eligible]
                        players[frames, dst] = players[frames, src]
                        speeds[frames, dst] = speeds[frames, src]
            if rng.random() < spec["false_ball_probability_per_block"]:
                first = int(rng.integers(a, b))
                end = min(b, first + max(1, round(rng.uniform(*spec["false_ball_seconds"]) * fps)))
                sources = np.flatnonzero(np.isfinite(players[first]).all(axis=1))
                if len(sources):
                    src = int(rng.choice(sources))
                    ball[first:end] = players[first:end, src]
                    ball_speed[first:end] = speeds[first:end, src]
        # Track IDs are not trusted identities. Permutations stay within teams
        # and move positions and speed together, without consulting labels.
        step = max(1, round(spec["slot_permutation_seconds"] * fps))
        for a in range(0, length, step):
            b = min(a + step, length)
            for side in (0, 1):
                slots = np.flatnonzero(result.teams == side)
                order = rng.permutation(slots)
                players[a:b, slots] = players[a:b, order]
                speeds[a:b, slots] = speeds[a:b, order]
    return result
