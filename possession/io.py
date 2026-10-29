"""Strict local readers. Restricted input records are never printed."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Tracking:
    players: np.ndarray
    player_speed: np.ndarray
    ball: np.ndarray
    ball_speed: np.ndarray
    teams: np.ndarray
    team_names: tuple


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xml_root(path):
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    return root


def read_metadata(path):
    root = xml_root(path)
    teams = list(root.iter("Team"))
    ids = [team.get("id") for team in teams]
    if len(ids) != 2 or None in ids or len(set(ids)) != 2:
        raise ValueError("ATD metadata must contain two distinct Team ids")
    slots = list(root.iter("Player"))
    if not slots:
        raise ValueError("ATD metadata contains no track slots")
    if any(player.get("teamId") not in ids for player in slots):
        raise ValueError("ATD metadata has a track slot with an unknown teamId")
    sides = np.array([ids.index(player.get("teamId")) for player in slots], dtype=np.int8)
    if set(sides.tolist()) != {0, 1}:
        raise ValueError("Both teams must have track slots")
    names = tuple((team.findtext("Name") or "").strip() for team in teams)
    return sides, names


def _triple(text, row, kind):
    parts = text.strip().split(",")
    if len(parts) != 3:
        raise ValueError(f"ATD row {row}: {kind} must have x,y,speed")
    try:
        values = tuple(float(value) for value in parts)
    except ValueError:
        raise ValueError(f"ATD row {row}: nonnumeric {kind} value") from None
    if any(np.isinf(value) for value in values):
        raise ValueError(f"ATD row {row}: infinite {kind} value")
    return values


def read_tracking(meta_path, raw_path, n_frames):
    sides, names = read_metadata(meta_path)
    count = len(sides)
    players = np.full((n_frames, count, 2), np.nan, dtype=np.float32)
    player_speed = np.full((n_frames, count), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    seen = 0
    with Path(raw_path).open(encoding="utf-8-sig") as handle:
        for row, line in enumerate(handle, 1):
            if row > n_frames:
                raise ValueError(f"ATD has more than {n_frames} records; wrong recording?")
            parts = line.strip().split(":")
            if len(parts) != 3:
                raise ValueError(f"ATD row {row}: expected frame:players:ball")
            try:
                frame_id = int(parts[0])
            except ValueError:
                raise ValueError(f"ATD row {row}: invalid frame id") from None
            if frame_id != row:
                raise ValueError(f"ATD row {row}: expected sequential frame id {row}; do not add offsets")
            groups = parts[1].rstrip(";").split(";")
            if len(groups) != count or any(group.count(",") != 2 for group in groups):
                raise ValueError(f"ATD row {row}: track-slot count disagrees with metadata")
            # Parse a complete numeric block at once, avoiding Python work per player.
            flat = np.fromstring(",".join(groups), sep=",", dtype=np.float32)
            if flat.size != count * 3 or np.isinf(flat).any():
                raise ValueError(f"ATD row {row}: invalid player x,y,speed block")
            values = flat.reshape(count, 3)
            players[row - 1] = values[:, :2]
            player_speed[row - 1] = values[:, 2]
            bx, by, speed = _triple(parts[2].rstrip(";"), row, "ball")
            ball[row - 1] = bx, by
            ball_speed[row - 1] = speed
            seen = row
    if seen != n_frames:
        raise ValueError(f"ATD has {seen} records, expected {n_frames}; dataset is incomplete")
    return Tracking(players, player_speed, ball, ball_speed, sides, names)


def read_events(path, n_frames, fps):
    events = []
    for node in xml_root(path).iter("instance"):
        try:
            a = float(node.findtext("start"))
            b = float(node.findtext("end"))
        except (ValueError, TypeError):
            raise ValueError("Smart Tagging contains an invalid time interval") from None
        if not np.isfinite([a, b]).all() or b <= a:
            raise ValueError("Smart Tagging contains a reversed or nonfinite interval")
        start, end = round(a * fps), round(b * fps)
        if end <= 0 or start >= n_frames:
            raise ValueError("Smart Tagging is outside the tactical timeline")
        tags = {(label.findtext("group") or "").strip().casefold():
                (label.findtext("text") or "").strip()
                for label in node.findall("label")}
        events.append({"start": max(0, start), "end": min(n_frames, end),
                       "code": (node.findtext("code") or "").strip().casefold(), "tags": tags})
    if not events:
        raise ValueError("Smart Tagging contains no instances")
    return sorted(events, key=lambda e: (e["start"], e["end"], e["code"]))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_submission(path, labels):
    path = Path(path)
    if not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Invalid output label")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("frame,label\n")
            handle.writelines(f"{i},{label}\n" for i, label in enumerate(labels))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
