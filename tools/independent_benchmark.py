"""Independent-source sparse pass-origin team audit; NOT dense possession truth.

Metrica sample Game 1 is adapter development. Game 2 is reserved for a single
frozen-candidate audit. Events never enter tracking conversion or inference.
Public source data stays in ignored runs/; only aggregate results may be shipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.config import load_config
from possession.io import Tracking, sha256, write_json, write_submission

SOURCE = "https://github.com/metrica-sports/sample-data"
FPS = 25
PITCH = (105., 68.)
CATALOG = {
    1: {
        "RawEventsData.csv": (139446, "3c426dd4f442dd433bccf89e3634f91aa310cd8f"),
        "RawTrackingData_Away_Team.csv": (32782073, "ff409a5680fd75acc7cebcf786827214c687478b"),
        "RawTrackingData_Home_Team.csv": (32782961, "ce2e91c0e633426cec1146f4e728b72417454e3d"),
    },
    2: {
        "RawEventsData.csv": (153897, "decccd47c1d53960d290cb6679337eec83a8a97f"),
        "RawTrackingData_Away_Team.csv": (29627711, "e7334d8d762954a215c49e55f211855000243fdc"),
        "RawTrackingData_Home_Team.csv": (31890338, "8b4d3648888cc8b033e7142919ebaea04f6328e0"),
    },
}
SPLIT = {
    "schema_version": 1, "source": SOURCE,
    "adapter_development": [1], "reserved_final_sparse_audit": [2],
    "point_definition": "One H/A reference at each unique PASS Start Frame; contradictory teams excluded; no interval filling.",
    "metric": "Exact-frame pass-origin team accuracy and per-team recall; D/X predictions count as wrong.",
    "not_measured": ["dense possession", "spellF1", "stoppage precision or recall"],
    "rule": "Freeze candidate artifacts and inference source hashes before reading Game 2 events. Do not tune on its audit results.",
    "license": "Provider README permits community analytics and requests attribution; no standard LICENSE is supplied. Raw data is not redistributed.",
}


def filename(game, suffix):
    return f"Sample_Game_{game}_{suffix}"


def verify_file(path, size, git_sha):
    """Validate published Git blob identity, including its length header."""
    path = Path(path)
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected size for {path.name}")
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != git_sha:
        raise ValueError(f"Published Git blob mismatch for {path.name}")


def freeze_split(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "split-before-reference-access.json"
    value = dict(SPLIT, catalog=CATALOG)
    # JSON round-trip normalizes integer dictionary keys for identity checks.
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError("Existing independent-data split differs; do not overwrite")
    else:
        write_json(path, value)
    return path


def download(folder):
    folder = Path(folder)
    freeze_split(folder)
    manifest = []
    for game, entries in CATALOG.items():
        for suffix, (size, git_sha) in entries.items():
            name = filename(game, suffix)
            url = f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{game}/{name}"
            path = folder / name
            if not path.exists():
                temporary = folder / (name + ".partial")
                received = 0
                with urlopen(url, timeout=90) as response, temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > size:
                            raise ValueError("Download exceeds published bounded size")
                        handle.write(chunk)
                verify_file(temporary, size, git_sha)
                temporary.replace(path)
            verify_file(path, size, git_sha)
            manifest.append({"game": game, "file": name, "bytes": size, "git_blob_sha1": git_sha,
                             "sha256": sha256(path), "url": url})
            print(f"Verified {name}: {size:,} bytes", flush=True)
    write_json(folder / "download-manifest.json", {"source": SOURCE, "files": manifest,
               "events_parsed_during_download": False, "split_sha256": sha256(freeze_split(folder))})
    return manifest


def _table(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        next(rows); next(rows)
        header = next(rows)
    if header[:3] != ["Period", "Frame", "Time [s]"]:
        raise ValueError("Unexpected Metrica tracking header")
    values = np.genfromtxt(path, delimiter=",", skip_header=3, dtype=float)
    values = np.atleast_2d(values)
    if values.shape[1] != len(header) or (values.shape[1] - 5) % 2 or values.shape[1] < 7:
        raise ValueError("Invalid tracking coordinate columns")
    if not np.isfinite(values[:, :3]).all():
        raise ValueError("Missing tracking frame metadata")
    return values


def _speed(points, blocks):
    result = np.full(points.shape[:-1], np.nan)
    for _, start, last in blocks:
        if last - start >= 4:
            result[start + 2:last - 1] = np.linalg.norm(
                (points[start + 4:last + 1] - points[start:last - 3]) * np.asarray(PITCH), axis=-1) * FPS / 4
    return result


def read_tracking(folder, game):
    """Read observations only, without opening any event or reference file."""
    folder = Path(folder)
    arrays = []
    for side in ("Home", "Away"):
        suffix = f"RawTrackingData_{side}_Team.csv"
        path = folder / filename(game, suffix)
        verify_file(path, *CATALOG[game][suffix])
        arrays.append(_table(path))
    home, away = arrays
    if home.shape[0] != away.shape[0] or not np.array_equal(home[:, :3], away[:, :3]):
        raise ValueError("Home and away tracking timelines differ")
    frame_ids = home[:, 1].astype(np.int64)
    if not np.array_equal(frame_ids, home[:, 1]) or len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("Frame identifiers must be unique integers")
    periods = home[:, 0].astype(int)
    if not np.isin(periods, [1, 2]).all() or not np.array_equal(periods, home[:, 0]):
        raise ValueError("Only regulation halves are supported")
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(periods)]
    blocks = [(int(periods[a]), int(a), int(b - 1)) for a, b in zip(edges[:-1], edges[1:])]
    if len({half for half, _, _ in blocks}) != len(blocks):
        raise ValueError("Repeated disjoint half blocks")
    for _, a, b in blocks:
        if not np.allclose(np.diff(home[a:b + 1, 2]), 1 / FPS, atol=1e-5):
            raise ValueError("Tracking timeline is not contiguous at 25 fps")
    home_points = home[:, 3:-2].reshape(len(home), -1, 2)
    away_points = away[:, 3:-2].reshape(len(away), -1, 2)
    players = np.concatenate([home_points, away_points], axis=1)
    home_ball, away_ball = home[:, -2:], away[:, -2:]
    valid_h, valid_a = np.isfinite(home_ball).all(axis=1), np.isfinite(away_ball).all(axis=1)
    if not np.allclose(home_ball[valid_h & valid_a], away_ball[valid_h & valid_a], atol=1e-7):
        raise ValueError("Home and away ball observations disagree")
    ball = np.where(valid_h[:, None], home_ball, away_ball)
    ball[~(valid_h | valid_a)] = np.nan
    tracking = Tracking(players, _speed(players, blocks), ball, _speed(ball, blocks),
                        np.r_[np.zeros(home_points.shape[1], dtype=int), np.ones(away_points.shape[1], dtype=int)],
                        ("Home", "Away"))
    alignment = {"frame_ids": frame_ids, "periods": periods, "fps": FPS,
                 "frame_mapping": "Exact published Start Frame lookup; no inferred offset"}
    return tracking, blocks, PITCH, alignment


def point_references(event_path, frame_ids, periods):
    """Sparse references from PASS actor team, never filled into dense spells."""
    if len(frame_ids) != len(periods) or len(set(frame_ids.tolist())) != len(frame_ids):
        raise ValueError("Invalid observation frame mapping")
    lookup = {int(frame): index for index, frame in enumerate(frame_ids)}
    labels = np.full(len(frame_ids), "X", dtype="<U1")
    points = {}
    counts = {"pass_events": 0, "unknown_team": 0, "unmapped_frame": 0, "conflicting_frames": 0,
              "duplicate_same_team": 0, "other_events_excluded": 0}
    with Path(event_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        required = {"Type", "Team", "Start Frame", "Period"}
        if not required <= set(rows.fieldnames or []):
            raise ValueError("Unexpected Metrica event header")
        for event in rows:
            if event["Type"] != "PASS":
                counts["other_events_excluded"] += 1
                continue
            counts["pass_events"] += 1
            team = {"Home": "H", "Away": "A"}.get(event["Team"])
            if team is None:
                counts["unknown_team"] += 1
                continue
            raw_frame = float(event["Start Frame"])
            if not np.isfinite(raw_frame) or raw_frame != int(raw_frame):
                raise ValueError("Event frames must be finite integers")
            index = lookup.get(int(raw_frame))
            if index is None:
                counts["unmapped_frame"] += 1
                continue
            if float(event["Period"]) != periods[index]:
                raise ValueError("Event period does not match tracking frame")
            if index in points and points[index] == {team}:
                counts["duplicate_same_team"] += 1
            points.setdefault(index, set()).add(team)
    for index, teams in points.items():
        if len(teams) == 1:
            labels[index] = next(iter(teams))
        else:
            counts["conflicting_frames"] += 1
    counts["scored_frames"] = int(np.sum(labels != "X"))
    if counts["unmapped_frame"] > max(1, .01 * counts["pass_events"]):
        raise ValueError("Too many pass events have no matching tracking frame")
    return labels, counts


def score_points(reference, predictions):
    reference, predictions = np.asarray(reference), np.asarray(predictions)
    if reference.shape != predictions.shape or reference.ndim != 1:
        raise ValueError("Reference and prediction shapes differ")
    if not np.isin(reference, list("HAX")).all() or not np.isin(predictions, list("HADX")).all():
        raise ValueError("Invalid point labels")
    use = reference != "X"
    if not use.any():
        raise ValueError("No pass-origin points to score")
    per_team = {}
    for team in "HA":
        selected = reference == team
        per_team[team] = {"n": int(selected.sum()), "recall": float(np.mean(predictions[selected] == team)) if selected.any() else None}
    recalls = [v["recall"] for v in per_team.values() if v["recall"] is not None]
    live = use & np.isin(predictions, ["H", "A"])
    return {"metric": "sparse_pass_origin_team_accuracy", "n": int(use.sum()),
            "accuracy": float(np.mean(reference[use] == predictions[use])),
            "balanced_team_recall": float(np.mean(recalls)), "per_team": per_team,
            "predicted_dead_fraction": float(np.mean(predictions[use] == "D")),
            "predicted_unknown_fraction": float(np.mean(predictions[use] == "X")),
            "predicted_live_coverage": float(live.sum() / use.sum()),
            "conditional_live_team_accuracy": float(np.mean(reference[live] == predictions[live])) if live.any() else None,
            "dense_possession_truth": False, "spellF1": None}


def _check_unused_audit(folder, game):
    if game == 2 and (Path(folder) / "game2-reference-access.json").exists():
        raise ValueError("Game 2 references have already been accessed; this is no longer an untouched audit")


def _mark_reference_access(folder, game, out):
    if game == 2:
        # Exclusive creation prevents two concurrent runs consuming the same test.
        with (Path(folder) / "game2-reference-access.json").open("x", encoding="utf-8") as handle:
            json.dump({"audit_directory": str(Path(out).resolve()),
                       "protocol_sha256": sha256(Path(out) / "frozen-before-reference-access.json"),
                       "status": "reference_access_started"}, handle, sort_keys=True, indent=2)
            handle.write("\n")


def evaluate(folder, game, model_path, config_path, out):
    """Freeze artifacts, infer from tracking, save predictions, THEN open events."""
    from possession.learned import load_model
    from possession.pipeline import infer
    folder, out = Path(folder), Path(out)
    _check_unused_audit(folder, game)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty audit output directory")
    out.mkdir(parents=True, exist_ok=True)
    split_path = freeze_split(folder)
    model_path, config_path = Path(model_path), Path(config_path)
    protocol = {"game": game, "role": "adapter_development" if game == 1 else "reserved_final_sparse_audit",
                "split_sha256": sha256(split_path), "model_sha256": sha256(model_path),
                "config_sha256": sha256(config_path), "sources": {
                    p.relative_to(ROOT).as_posix(): sha256(p)
                    for p in [Path(__file__), *sorted((ROOT / "possession").glob("*.py"))]},
                "point_definition": SPLIT["point_definition"], "no_parameter_tuning_in_this_command": True}
    write_json(out / "frozen-before-reference-access.json", protocol)
    tracking, blocks, pitch, alignment = read_tracking(folder, game)
    predictions, _, _, _, warnings = infer(tracking, [], blocks, pitch, FPS,
                                           load_config(config_path), load_model(model_path))
    write_submission(out / "predictions.csv", predictions)
    event_path = folder / filename(game, "RawEventsData.csv")
    verify_file(event_path, *CATALOG[game]["RawEventsData.csv"])
    if sha256(model_path) != protocol["model_sha256"] or sha256(config_path) != protocol["config_sha256"]:
        raise ValueError("Frozen artifact changed before reference access")
    if any(sha256(ROOT / name) != digest for name, digest in protocol["sources"].items()):
        raise ValueError("Frozen inference source changed before reference access")
    _mark_reference_access(folder, game, out)
    reference, counts = point_references(event_path, alignment["frame_ids"], alignment["periods"])
    if sha256(model_path) != protocol["model_sha256"] or sha256(config_path) != protocol["config_sha256"]:
        raise ValueError("Frozen artifact changed during audit")
    if any(sha256(ROOT / name) != digest for name, digest in protocol["sources"].items()):
        raise ValueError("Frozen inference source changed during audit")
    result = {"protocol": protocol, "scores": score_points(reference, predictions), "event_counts": counts,
              "prediction_sha256": sha256(out / "predictions.csv"), "event_file_sha256": sha256(event_path),
              "warnings": warnings, "reference_events_passed_to_inference": False}
    write_json(out / "summary.json", result)
    print(json.dumps(result["scores"], indent=2), flush=True)
    return result


def evaluate_pair(folder, game, candidate_model, baseline_model, config_path, out,
                  baseline_config_path=None):
    """Freeze both artifacts, save both predictions, then read references once.

    This is a paired audit, not a candidate selection loop. Game 2 consumption is
    recorded exclusively in the dataset directory to prevent accidental reuse.
    Model dispatch stays in production pipeline.infer, including optional heads.
    """
    from possession.learned import load_model
    from possession.pipeline import infer
    folder, out = Path(folder), Path(out)
    _check_unused_audit(folder, game)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty paired audit output directory")
    out.mkdir(parents=True, exist_ok=True)
    split_path = freeze_split(folder)
    artifacts = {
        "baseline": (Path(baseline_model), Path(baseline_config_path or config_path)),
        "candidate": (Path(candidate_model), Path(config_path)),
    }
    protocol = {
        "game": game, "role": "adapter_development" if game == 1 else "reserved_final_sparse_audit",
        "comparison": "one frozen candidate against one frozen baseline; no selection or tuning",
        "split_sha256": sha256(split_path),
        "artifacts": {name: {"model_sha256": sha256(model), "config_sha256": sha256(config)}
                      for name, (model, config) in artifacts.items()},
        "sources": {p.relative_to(ROOT).as_posix(): sha256(p)
                    for p in [Path(__file__), *sorted((ROOT / "possession").glob("*.py"))]},
        "point_definition": SPLIT["point_definition"], "no_parameter_tuning_in_this_command": True,
        "reference_read_order": "both prediction CSVs saved before one shared event-reference read",
    }
    write_json(out / "frozen-before-reference-access.json", protocol)

    def unchanged():
        for name, (model, config) in artifacts.items():
            if (sha256(model) != protocol["artifacts"][name]["model_sha256"]
                    or sha256(config) != protocol["artifacts"][name]["config_sha256"]):
                raise ValueError("Frozen paired-audit artifact changed")
        if any(sha256(ROOT / name) != digest for name, digest in protocol["sources"].items()):
            raise ValueError("Frozen paired-audit inference source changed")

    tracking, blocks, pitch, alignment = read_tracking(folder, game)
    predictions, warnings = {}, {}
    for name, (model_path, config) in artifacts.items():
        predictions[name], _, _, _, warnings[name] = infer(
            tracking, [], blocks, pitch, FPS, load_config(config), load_model(model_path))
        write_submission(out / f"{name}-predictions.csv", predictions[name])
    unchanged()
    event_path = folder / filename(game, "RawEventsData.csv")
    verify_file(event_path, *CATALOG[game]["RawEventsData.csv"])
    _mark_reference_access(folder, game, out)
    reference, counts = point_references(event_path, alignment["frame_ids"], alignment["periods"])
    unchanged()
    scores = {name: score_points(reference, values) for name, values in predictions.items()}
    deltas = {metric: scores["candidate"][metric] - scores["baseline"][metric]
              for metric in ("accuracy", "balanced_team_recall", "predicted_live_coverage",
                             "predicted_dead_fraction", "predicted_unknown_fraction")}
    result = {
        "protocol": protocol, "scores": scores, "candidate_minus_baseline": deltas,
        "event_counts": counts, "event_file_sha256": sha256(event_path),
        "prediction_sha256": {name: sha256(out / f"{name}-predictions.csv") for name in artifacts},
        "warnings": warnings, "reference_events_passed_to_inference": False,
        "reference_read_count": 1, "dense_possession_truth": False,
    }
    write_json(out / "summary.json", result)
    print(json.dumps({"scores": scores, "candidate_minus_baseline": deltas}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "evaluate", "evaluate-pair"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "runs/independent-data")
    parser.add_argument("--game", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model", type=Path, default=ROOT / "models/submission.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/independent-audit-game1")
    args = parser.parse_args()
    if args.command == "download":
        download(args.data_dir)
    elif args.command == "evaluate-pair":
        if args.baseline_model is None:
            parser.error("evaluate-pair requires --baseline-model")
        evaluate_pair(args.data_dir, args.game, args.model, args.baseline_model,
                      args.config, args.out, args.baseline_config)
    else:
        evaluate(args.data_dir, args.game, args.model, args.config, args.out)
