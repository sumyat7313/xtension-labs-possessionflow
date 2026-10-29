"""Soft tactical priors. Defensive tags imply possession by the OTHER side."""

from dataclasses import dataclass
import re
import unicodedata
import numpy as np

from .features import rolling_mean


OFFENSIVE = {"build up": 1.0, "progression": 0.9, "creating chances": 0.9,
             "attacking transition": 0.9, "long ball": 0.65,
             "ball in final third": 0.55, "ball in the box": 0.55, "players in the box": 0.3}
DEFENSIVE = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


def normalize(value):
    value = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def team_index(name, team_names):
    key = normalize(name)
    if not key or key in {"n a", "na", "none", "unknown"}:
        return None
    aliases = [{normalize(team_names[0])}, {normalize(team_names[1])}]
    # Published club spellings in this dataset; no inferred or human labels.
    for index, actual in enumerate(team_names):
        if "barcelona" in normalize(actual):
            aliases[index].update({"barcelona", "fc barcelona", "barca"})
        if "athletic" in normalize(actual) or "atletic" in normalize(actual):
            aliases[index].update({"athletic club", "atletic club", "athletic bilbao"})
    matches = [index for index in (0, 1) if key in aliases[index]]
    return matches[0] if len(matches) == 1 else None


@dataclass
class TagEvidence:
    team_log_odds: np.ndarray
    live: np.ndarray
    dead: np.ndarray
    restart_live: np.ndarray
    unknown_team_events: int
    restart_candidates: int


def _taper(length, width):
    index = np.arange(length)
    return np.minimum(1.0, np.minimum(index + 1, length - index) / max(1, width))


def build(events, features, team_names, n_frames, fps, blocks, config):
    support = np.zeros((n_frames, 2))
    live = np.zeros(n_frames)
    dead = np.zeros(n_frames)
    restart_live = np.zeros(n_frames)
    unknown = detected = 0
    taper = max(1, round(config.tag_taper_seconds * fps))
    for event in events:
        a, b = event["start"], event["end"]
        side = team_index(event["tags"].get("team", ""), team_names)
        if side is None:
            unknown += 1
        code = normalize(event["code"])
        confidence = OFFENSIVE.get(code, 0)
        if code in DEFENSIVE:
            confidence = 0.8
            side = None if side is None else 1 - side
        if confidence:
            strength = confidence * _taper(b - a, taper)
            live[a:b] = np.maximum(live[a:b], strength)
            if side is not None:
                support[a:b, side] = np.maximum(support[a:b, side], strength)
        if code != "set pieces":
            continue
        # Work in the containing half. Intervals are windows around restarts,
        # so an interval's start/end is never copied as a hard state transition.
        block = next(((start, last + 1) for _, start, last in blocks
                      if a < last + 1 and b > start), None)
        if block is None:
            continue
        lo, hi = block
        a, b = max(a, lo), min(b, hi)
        context = max(lo, a - round(config.restart_context_seconds * fps))
        before = max(1, round(config.restart_before_seconds * fps))
        after = max(1, round(config.restart_after_seconds * fps))
        candidate = None
        values = features.ball_speed
        for frame in range(a, b):
            if frame - before < lo or frame + after > hi:
                continue
            pre, post = values[frame - before:frame], values[frame:frame + after]
            if np.isfinite(pre).mean() < 0.6 or np.isfinite(post).mean() < 0.6:
                continue
            if np.nanmean(pre) <= config.restart_stationary_mps and np.nanmean(post) >= config.restart_moving_mps:
                candidate = frame
                break
        end_dead = candidate if candidate is not None else b
        if end_dead > context:
            corroboration = np.maximum.reduce([
                features.stationary[context:end_dead], features.outside[context:end_dead],
                features.missing[context:end_dead] * features.still_players[context:end_dead]])
            time_weight = np.linspace(0.25, 1.0, end_dead - context)
            if candidate is None:
                # Ambiguous set-piece intervals contribute only weak evidence.
                time_weight *= 0.55
            dead[context:end_dead] = np.maximum(dead[context:end_dead], time_weight * corroboration)
        if candidate is not None:
            detected += 1
            end = min(hi, candidate + round(config.tag_taper_seconds * fps))
            restart_live[candidate:end] = 1
            if side is not None:
                support[candidate:end, side] = np.maximum(support[candidate:end, side], 0.85)
    return TagEvidence(config.tag_log_odds * (support[:, 0] - support[:, 1]),
                       live, dead, restart_live, unknown, detected)
