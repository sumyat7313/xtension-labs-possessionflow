"""Label-free aggregate camera calibration and external-match augmentation.

Homographies map NORMALIZED pitch coordinates to pixels. Inverse image corners
give a ground-plane quadrilateral; metres are used only after that inversion.
The exported profile retains quantized population summaries, never a frame's
homography, player/ball position, camera pose, timestamp, or visibility sequence.
It must remain local under runs because its source is restricted match data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from possession.io import Tracking
from tools.camera_augmentation import _validate

MAX_REPROJECTION_ERROR_PIXELS = 5.0
QUANTIZATION_METRES = 2.0
MIN_AGGREGATE_FRAMES = 16
DURATION_EDGES_SECONDS = (0., .08, .16, .32, .64, 1.28, 2.56, 5.12, 10.24,
                          20.48, 40.96, 81.92, 163.84, 327.68, 655.36)


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _histogram_runs(mask, blocks, fps):
    durations = []
    for _, start, last in blocks:
        durations.extend((b - a) / fps for a, b in _runs(mask[start:last + 1]))
    edges = np.asarray(DURATION_EDGES_SECONDS)
    clipped = np.minimum(durations, np.nextafter(edges[-1], 0))
    return np.histogram(clipped, edges)[0].astype(int).tolist()


def _centroid(players):
    valid = np.isfinite(players).all(axis=2)
    # Sorting prevents the summation order from depending on slot identities.
    total = np.sort(np.where(valid[..., None], players, 0), axis=1).sum(axis=1)
    count = valid.sum(axis=1)
    return np.divide(total, count[:, None], out=np.full((len(players), 2), np.nan),
                     where=count[:, None] > 0)


def project_points(matrices, points):
    """Project n-by-k-by-2 normalized points with n-by-3-by-3 homographies."""
    matrices, points = np.asarray(matrices, dtype=float), np.asarray(points, dtype=float)
    if (matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or points.ndim != 3
            or points.shape[0] != len(matrices) or points.shape[2] != 2):
        raise ValueError("Projection needs n-by-3-by-3 matrices and n-by-k-by-2 points")
    homogeneous = np.concatenate((points, np.ones((*points.shape[:2], 1))), axis=2)
    pixels = np.einsum("nij,nkj->nki", matrices, homogeneous)
    denominator = pixels[..., 2:]
    return np.divide(pixels[..., :2], denominator, out=np.full_like(points, np.nan),
                     where=np.abs(denominator) > 1e-10)


def _convex(polygons):
    edges = np.roll(polygons, -1, axis=1) - polygons
    following = np.roll(edges, -1, axis=1)
    cross = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
    return (cross > 1e-6).all(axis=1) | (cross < -1e-6).all(axis=1)


def _inside(points, polygons):
    """Convex quadrilateral membership, independent of clockwise orientation."""
    edges = np.roll(polygons, -1, axis=1) - polygons
    relative = points[:, :, None] - polygons[:, None]
    cross = (edges[:, None, :, 0] * relative[..., 1]
             - edges[:, None, :, 1] * relative[..., 0])
    return ((cross >= -1e-8).all(axis=2) | (cross <= 1e-8).all(axis=2)) & np.isfinite(points).all(axis=2)


def fit_profile(tracking, matrices, valid, errors, blocks, pitch, fps, width=1920, height=1080):
    """Fit a compact aggregate camera profile without possession references.

    Only half spans contribute. Valid finite homographies with reprojection error
    at most 5 pixels are eligible. Image-corner inversions crossing the projective
    horizon or producing nonconvex/unbounded polygons are excluded. Within zoom
    quartiles and ball/centroid anchor groups, at least 16 frames are averaged and
    each relative vertex is quantized to 2 metres. No individual polygon survives.

    Ball observed/missing duration histograms and player-count histograms describe
    detector availability, not ground-truth recall. Missing-ball position is not
    known, so a conditional in-view miss rate is explicitly NOT estimated.
    """
    blocks, pitch = _validate(tracking, blocks, pitch, fps, 0)
    if isinstance(fps, (bool, np.bool_)):
        raise ValueError("fps must be positive and finite")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be finite and positive")
    n = len(tracking.ball)
    matrices, valid, errors = np.asarray(matrices, float), np.asarray(valid), np.asarray(errors, float)
    if matrices.shape != (n, 3, 3) or valid.shape != (n,) or errors.shape != (n,):
        raise ValueError("Homography records must align exactly with tracking frames")
    if not np.isin(valid, [0, 1, False, True]).all():
        raise ValueError("Homography valid flags must be boolean or 0/1")
    active = np.zeros(n, dtype=bool)
    for _, start, last in blocks:
        active[start:last + 1] = True
    if active.sum() < MIN_AGGREGATE_FRAMES:
        raise ValueError("At least 16 active frames are needed for aggregate calibration")
    usable = (active & valid.astype(bool) & np.isfinite(matrices).all(axis=(1, 2))
              & np.isfinite(errors) & (errors >= 0) & (errors <= MAX_REPROJECTION_ERROR_PIXELS))
    indices = np.flatnonzero(usable)
    good = np.abs(np.linalg.det(matrices[indices])) > 1e-10
    indices = indices[good]
    inverse = np.linalg.inv(matrices[indices])
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    raw = np.einsum("nij,kj->nki", inverse, corners)
    denominator = raw[..., 2]
    good = ((denominator > 1e-9).all(axis=1) | (denominator < -1e-9).all(axis=1))
    polygons = np.divide(raw[..., :2], denominator[..., None],
                         out=np.full_like(raw[..., :2], np.nan), where=np.abs(denominator[..., None]) > 1e-9) * pitch
    good &= np.isfinite(polygons).all(axis=(1, 2)) & (np.abs(polygons) <= 10 * pitch).all(axis=(1, 2))
    good &= _convex(polygons)
    indices, polygons = indices[good], polygons[good]
    usable[:] = False
    usable[indices] = True
    ball = np.asarray(tracking.ball, float)
    players = np.asarray(tracking.players, float)
    ball_observed = (np.isfinite(ball).all(axis=1) & (ball >= -.2).all(axis=1) & (ball <= 1.2).all(axis=1))
    player_valid = (np.isfinite(players).all(axis=2) & (players >= -.2).all(axis=2) & (players <= 1.2).all(axis=2))
    centers = _centroid(np.where(player_valid[..., None], players, np.nan)) * pitch
    anchor = np.where(ball_observed[:, None], ball * pitch, centers)
    relative = polygons - anchor[indices, None]
    good = np.isfinite(relative).all(axis=(1, 2))
    indices, relative = indices[good], relative[good]
    if len(indices) < MIN_AGGREGATE_FRAMES:
        raise ValueError("Too few valid anchored homographies for aggregate calibration")
    span = np.ptp(relative[..., 0], axis=1)
    cuts = np.quantile(span, [.25, .5, .75])
    groups = np.searchsorted(cuts, span, side="right")
    prototypes = []
    for kind in ("ball", "centroid"):
        for group in range(4):
            chosen = (ball_observed[indices] == (kind == "ball")) & (groups == group)
            if chosen.sum() < MIN_AGGREGATE_FRAMES:
                continue
            # Averaging all group frames, then coarse quantization, prevents an
            # individual target frame geometry from appearing in the profile.
            polygon = np.round(np.mean(relative[chosen], axis=0) / QUANTIZATION_METRES) * QUANTIZATION_METRES
            if not _convex(polygon[None])[0]:
                continue
            prototypes.append({"anchor": kind, "count": int(chosen.sum()), "relative_polygon_metres": polygon.tolist()})
    if not prototypes:
        raise ValueError("No sufficiently aggregated convex camera footprint could be fitted")
    projected_observed = projected_inside = observed_players = inside_players = 0
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        keep = usable[start:end]
        if not keep.any():
            continue
        point = project_points(matrices[start:end][keep], ball[start:end][keep, None])[:, 0]
        known = ball_observed[start:end][keep]
        within = ((point >= 0) & (point <= [width, height])).all(axis=1)
        projected_observed += int(known.sum())
        projected_inside += int((known & within).sum())
        point = project_points(matrices[start:end][keep], players[start:end][keep])
        known = player_valid[start:end][keep]
        within = ((point >= 0) & (point <= [width, height])).all(axis=2)
        observed_players += int(known.sum())
        inside_players += int((known & within).sum())
    counts = np.minimum(player_valid.sum(axis=1), 22)
    profile = {
        "schema_version": 1, "kind": "aggregate_calibrated_camera",
        "coordinate_convention": "homography_normalized_pitch_to_pixels",
        "image_size": [int(width), int(height)], "source_pitch_metres": pitch.tolist(),
        "quantization_metres": QUANTIZATION_METRES, "minimum_group_frames": MIN_AGGREGATE_FRAMES,
        "max_reprojection_error_pixels": MAX_REPROJECTION_ERROR_PIXELS,
        "active_frames": int(active.sum()), "usable_homography_fraction": round(float(usable[active].mean()), 3),
        "ball_observed_fraction": round(float(ball_observed[active].mean()), 3),
        "observed_ball_inside_image_fraction": round(projected_inside / max(1, projected_observed), 3),
        "observed_players_inside_image_fraction": round(inside_players / max(1, observed_players), 3),
        "player_count_histogram": np.bincount(counts[active], minlength=23).astype(int).tolist(),
        "run_duration_edges_seconds": list(DURATION_EDGES_SECONDS),
        "ball_observed_run_histogram": _histogram_runs(ball_observed, blocks, fps),
        "ball_missing_run_histogram": _histogram_runs(~ball_observed, blocks, fps),
        "homography_invalid_run_histogram": _histogram_runs(~usable, blocks, fps),
        "footprints": prototypes,
        "augmentation_constants": {"camera_knot_seconds": 2., "camera_lag_seconds": .4,
                                   "player_count_block_seconds": 2., "slot_permutation_seconds": 6.},
        "limitations": ["availability is not ground-truth detector recall",
            "missing-ball in-view status is unknown; no conditional miss-rate estimate",
            "relative footprints are aggregate approximations; target frame geometry is discarded",
            "player counts cap at 22; duplicate identities are not inferred",
            "ball missingness is a total-rate target applied after geometric clipping; clipping may exceed it",
            "camera motion and player masking persistence are fixed, not learned from trajectories",
            "local restricted-data aggregate; do not publish without organizer permission"],
    }
    return validate_profile(profile)


def validate_profile(profile):
    if not isinstance(profile, dict) or profile.get("kind") != "aggregate_calibrated_camera" or profile.get("schema_version") != 1:
        raise ValueError("Unsupported calibrated camera profile")
    for key in ("ball_observed_fraction", "usable_homography_fraction"):
        value = profile.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key}")
    edges = np.asarray(profile.get("run_duration_edges_seconds"), dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or edges[0] != 0 or (np.diff(edges) <= 0).any():
        raise ValueError("Invalid run duration bins")
    for key, length in (("player_count_histogram", 23), ("ball_missing_run_histogram", len(edges) - 1),
                        ("ball_observed_run_histogram", len(edges) - 1), ("homography_invalid_run_histogram", len(edges) - 1)):
        counts = np.asarray(profile.get(key))
        if counts.shape != (length,) or counts.dtype.kind not in "iu" or (counts < 0).any():
            raise ValueError(f"Invalid {key}")
    if sum(profile["player_count_histogram"]) <= 0:
        raise ValueError("Player count distribution is empty")
    footprints = profile.get("footprints")
    if not isinstance(footprints, list) or not footprints or len(footprints) > 8:
        raise ValueError("Invalid aggregate footprints")
    for item in footprints:
        polygon = np.asarray(item.get("relative_polygon_metres"), float)
        if (item.get("anchor") not in ("ball", "centroid") or
                not isinstance(item.get("count"), int) or item["count"] < MIN_AGGREGATE_FRAMES or
                polygon.shape != (4, 2) or not np.isfinite(polygon).all() or not _convex(polygon[None])[0]):
            raise ValueError("Invalid aggregate footprint")
    constants = profile.get("augmentation_constants", {})
    for key in ("camera_knot_seconds", "camera_lag_seconds", "player_count_block_seconds", "slot_permutation_seconds"):
        value = constants.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid camera augmentation constants")
    json.dumps(profile, allow_nan=False)
    return profile


def _duration(rng, histogram, edges, fps):
    counts = np.asarray(histogram, float)
    if counts.sum() == 0:
        return 1
    index = int(rng.choice(len(counts), p=counts / counts.sum()))
    return max(1, round(rng.uniform(edges[index], edges[index + 1]) * fps))


def _add_missing(mask, target_fraction, histogram, edges, fps, rng):
    """Add contiguous empirical-duration gaps until the total rate is reached.

    Existing geometric/input gaps cannot be undone. Candidate starts are drawn
    without replacement, making completion bounded even with very long runs.
    """
    result = mask.copy()
    needed = max(0, round(target_fraction * len(mask)) - int(result.sum()))
    if not needed:
        return result
    for first in rng.permutation(len(mask)):
        if not needed:
            break
        stop = min(len(mask), first + _duration(rng, histogram, edges, fps))
        available = np.flatnonzero(~result[first:stop]) + first
        chosen = available[:needed]
        result[chosen] = True
        needed -= len(chosen)
    return result


def corrupt_calibrated_tracking(tracking, blocks, pitch, fps, seed, profile):
    """Apply aggregate-calibrated geometry, coverage and structured dropout.

    External observations alone set camera anchors. The fitted profile contains
    no target sequence to replay. Footprints are sampled independently every two
    seconds, linearly interpolated, and clipped to actual external player/ball
    positions. Added ball gaps target the measured total missing fraction; this
    avoids treating unknown out-of-view balls as proven detector failures.
    Every half has a fresh deterministic random stream and motion state.
    """
    blocks, pitch = _validate(tracking, blocks, pitch, fps, seed)
    if isinstance(fps, (bool, np.bool_)):
        raise ValueError("fps must be positive and finite")
    profile = validate_profile(profile)
    result = Tracking(*(np.array(getattr(tracking, name), dtype=float, copy=True)
                        for name in ("players", "player_speed", "ball", "ball_speed")),
                      np.array(tracking.teams, copy=True), tuple(tracking.team_names))
    footprints = profile["footprints"]
    probability = np.asarray([item["count"] for item in footprints], float)
    probability /= probability.sum()
    player_count = np.asarray(profile["player_count_histogram"], float)
    player_count /= player_count.sum()
    edges = profile["run_duration_edges_seconds"]
    constants = profile["augmentation_constants"]
    for block_index, (_, start, last) in enumerate(blocks):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), block_index]))
        stop, length = last + 1, last + 1 - start
        players, speeds = result.players[start:stop], result.player_speed[start:stop]
        ball, ball_speed = result.ball[start:stop], result.ball_speed[start:stop]
        center = _centroid(players) * pitch
        center[~np.isfinite(center).all(axis=1)] = pitch / 2
        anchor = np.where(np.isfinite(ball).all(axis=1)[:, None], ball * pitch, center)
        knot_step = max(1, round(constants["camera_knot_seconds"] * fps))
        knots = np.unique(np.r_[np.arange(0, length, knot_step), length - 1])
        picked = rng.choice(len(footprints), len(knots), p=probability)
        lag = max(0, round(constants["camera_lag_seconds"] * fps))
        knot_polygons = []
        for frame, choice in zip(knots, picked):
            item = footprints[int(choice)]
            index = max(0, frame - lag)
            origin = anchor[index] if item["anchor"] == "ball" else center[index]
            knot_polygons.append(origin + np.asarray(item["relative_polygon_metres"]))
        knot_polygons = np.asarray(knot_polygons)
        polygons = np.stack([np.interp(np.arange(length), knots, knot_polygons[:, j, axis])
                              for j in range(4) for axis in range(2)], axis=1).reshape(length, 4, 2)
        for a in range(0, length, 4096):
            b = min(a + 4096, length)
            player_visible = _inside(players[a:b] * pitch, polygons[a:b])
            ball_visible = _inside(ball[a:b, None] * pitch, polygons[a:b])[:, 0]
            players[a:b][~player_visible] = np.nan
            speeds[a:b][~player_visible] = np.nan
            ball[a:b][~ball_visible] = np.nan
            ball_speed[a:b][~ball_visible] = np.nan
        invalid = _add_missing(np.zeros(length, bool), 1 - profile["usable_homography_fraction"],
                               profile["homography_invalid_run_histogram"], edges, fps, rng)
        players[invalid] = np.nan
        speeds[invalid] = np.nan
        ball[invalid] = np.nan
        ball_speed[invalid] = np.nan
        step = max(1, round(constants["player_count_block_seconds"] * fps))
        for a in range(0, length, step):
            b = min(a + step, length)
            desired = int(rng.choice(23, p=player_count))
            # Random priorities persist across each block, so missing slots form
            # fragments rather than independent frame-level salt-and-pepper.
            priority = rng.random(len(result.teams))
            available = np.isfinite(players[a:b]).all(axis=2)
            ranks = np.argsort(np.argsort(np.where(available, priority[None], np.inf), axis=1), axis=1)
            remove = ranks >= desired
            players[a:b][remove] = np.nan
            speeds[a:b][remove] = np.nan
        missing = _add_missing(~np.isfinite(ball).all(axis=1), 1 - profile["ball_observed_fraction"],
                               profile["ball_missing_run_histogram"], edges, fps, rng)
        ball[missing] = np.nan
        ball_speed[missing] = np.nan
        step = max(1, round(constants["slot_permutation_seconds"] * fps))
        for a in range(0, length, step):
            b = min(a + step, length)
            for team in (0, 1):
                slots = np.flatnonzero(result.teams == team)
                order = rng.permutation(slots)
                players[a:b, slots] = players[a:b, order]
                speeds[a:b, slots] = speeds[a:b, order]
    return result


if __name__ == "__main__":
    from games import GAMES, half_video_blocks
    from possession.io import read_tracking, write_json, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Authorized challenge data directory")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/calibrated-camera/profile.json")
    args = parser.parse_args()
    if ROOT / "runs" not in args.out.resolve().parents:
        parser.error("Restricted-data aggregate profiles must be saved under this repository's runs directory")
    if args.out.exists():
        parser.error("Use a new output profile path; an existing profile is never overwritten")
    folder = args.data_dir / "tactical"
    stem = folder / "BAR-ATH_tactical"
    meta, raw, homography = [Path(str(stem) + suffix) for suffix in ("_atd.xml", "_atd.txt", "_homography.csv")]
    tracking = read_tracking(meta, raw, GAMES["BAR-ATH"]["n_video_frames"])
    table = np.loadtxt(homography, delimiter=",", skiprows=1)
    if table.shape != (len(tracking.ball), 12) or not np.array_equal(table[:, 0], np.arange(len(tracking.ball))):
        raise ValueError("Homography CSV frame indices do not align with tracking")
    profile = fit_profile(tracking, table[:, 1:10].reshape(-1, 3, 3), table[:, 11], table[:, 10],
                          half_video_blocks("BAR-ATH"), GAMES["BAR-ATH"]["pitch"], GAMES["BAR-ATH"]["fps"])
    profile["source_sha256"] = {p.name: sha256(p) for p in (meta, raw, homography)}
    write_json(args.out, profile)
    print(json.dumps({"profile": str(args.out), "footprints": len(profile["footprints"]),
                      "ball_observed_fraction": profile["ball_observed_fraction"],
                      "usable_homography_fraction": profile["usable_homography_fraction"]}))
