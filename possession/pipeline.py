"""End-to-end inference. No ground-truth files are opened here."""

from pathlib import Path
import csv
import hashlib
import json
import platform
import time

import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks
from . import __version__
from .config import load_config
from .decoder import decode, observation_probabilities
from .features import extract
from .io import read_events, read_tracking, sha256, write_json, write_submission
from .tags import build

ROOT = Path(__file__).resolve().parents[1]


def code_digest():
    digest = hashlib.sha256()
    files = [ROOT / "games.py", *sorted((ROOT / "possession").glob("*.py"))]
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_csv(path, expected):
    from evaluation.possession_metrics import read_label_csv, validate_submission
    table = read_label_csv(path)
    reason = validate_submission(table, expected)
    if reason:
        raise ValueError(f"Submission validation failed: {reason}")
    return len(table)


def infer(tracking, events, blocks, pitch, fps, config, model=None):
    if model is not None:
        from .learned import infer_learned
        return infer_learned(tracking, events, blocks, pitch, fps, config, model,
                             mode=model.get("decoder_mode", "joint"))
    features = extract(tracking, pitch, blocks, fps, config)
    tags = build(events, features, tracking.team_names, len(tracking.ball), fps, blocks, config)
    probabilities = observation_probabilities(features, tags, config)
    labels = np.full(len(tracking.ball), "X", dtype="<U1")
    warnings = []
    for half, start, last in blocks:
        stop = last + 1
        evidence = features.reliability[start:stop].sum() + tags.live[start:stop].sum() + tags.restart_live[start:stop].sum()
        if evidence == 0:
            warnings.append(f"Half {half} has no usable ball or tactical evidence; returned X")
            continue
        labels[start:stop] = np.array(["H", "A", "D"])[decode(probabilities[start:stop], fps, config)]
    return labels, probabilities, features, tags, warnings


def segments(labels, probabilities, fps):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    for a, b in zip(edges[:-1], edges[1:]):
        label = str(labels[a])
        score = None if label == "X" else float(probabilities[a:b, "HAD".index(label)].mean())
        yield {"start_frame": int(a), "end_frame_exclusive": int(b), "label": label,
               "duration_seconds": round((b - a) / fps, 3),
               "mean_observation_support": None if score is None else round(score, 6)}


def predict(data_dir=None, out_dir="runs/possessionflow", config_path=None,
            tracking_only=False, allow_synthetic=False, progress=print, model_path=None):
    config = load_config(config_path)
    model = None
    if model_path is not None:
        from .learned import load_model
        model = load_model(model_path)
    paths = {name: Path(path) for name, path in asset_paths(data_dir=data_dir).items()}
    root = paths["atd_raw"].parent.parent
    synthetic = (root / ".synthetic-fixture.json").is_file()
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic inputs are demo-only. Use the demo command; they cannot be packaged as a submission.")
    required = ["atd_meta", "atd_raw"] + ([] if tracking_only else ["smart_tagging"])
    missing = [paths[name].name for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("Missing tactical assets: " + ", ".join(missing) +
                                ". Put tactical/ under --data-dir, or set MTAH_DATA_DIR.")
    destination = Path(out_dir).resolve()
    marker = destination / ".prediction-run.json"
    if destination.exists() and any(destination.iterdir()) and not (destination / "manifest.json").is_file() and not marker.is_file():
        raise ValueError("Output directory is nonempty and is not an existing prediction run")
    if destination == root.resolve() or root.resolve() in destination.parents:
        raise ValueError("Choose an output directory outside the input data folder")
    started = time.perf_counter()
    spec = GAMES[GAME]
    n, fps = spec["n_video_frames"], spec["fps"]
    blocks = half_video_blocks(GAME)
    progress("Reading and checking tactical tracking locally...")
    tracking = read_tracking(paths["atd_meta"], paths["atd_raw"], n)
    events = [] if tracking_only else read_events(paths["smart_tagging"], n, fps)
    progress("Computing geometry, tactical evidence, and temporal states...")
    labels, probabilities, features, tags, warnings = infer(tracking, events, blocks, spec["pitch"], fps, config, model)
    if tracking_only:
        warnings.append("Tracking-only mode: automatic tactical and restart evidence is unavailable")
    if synthetic:
        warnings.append("SYNTHETIC DEMO: these predictions are not for the competition match")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(marker, {"status": "writing", "solution": "PossessionFlow", "synthetic": synthetic})
    csv_path = destination / "predictions.csv"
    write_submission(csv_path, labels)
    validate_csv(csv_path, n)
    rows = list(segments(labels, probabilities, fps))
    with (destination / "segments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    observation_entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-8))).sum(axis=1) / np.log(3)
    report = {
        "synthetic": synthetic, "rows": n, "graded_frames": int(graded.sum()),
        "label_counts": {label: int(np.sum(labels == label)) for label in "HADX"},
        "possession_segments": sum(row["label"] in ("H", "A") for row in rows),
        "dead_ball_segments": sum(row["label"] == "D" for row in rows),
        "observed_ball_fraction_in_graded_frames": round(float(features.observed[graded].mean()), 6),
        "interpolated_frames": features.interpolated_frames,
        "rejected_isolated_ball_spikes": features.rejected_spikes,
        "tag_events": len(events), "unknown_or_na_team_tags": tags.unknown_team_events,
        "detected_restarts": tags.restart_candidates,
        "mean_observation_entropy_in_graded_frames": round(float(observation_entropy[graded].mean()), 6),
        "support_note": "Observation support and entropy are heuristic evidence, not calibrated confidence or accuracy.",
        "elapsed_seconds": round(time.perf_counter() - started, 3), "warnings": warnings,
    }
    write_json(destination / "diagnostics.json", report)
    write_json(destination / "config.json", config.as_dict())
    if model is not None:
        write_json(destination / "model.json", model)
    manifest = {
        "schema_version": 1, "solution_version": __version__, "game": GAME,
        "synthetic": synthetic, "tracking_only": tracking_only,
        "code_sha256": code_digest(), "config_sha256": sha256(destination / "config.json"),
        "inputs": {name: {"filename": paths[name].name, "sha256": sha256(paths[name]),
                          "bytes": paths[name].stat().st_size} for name in required},
        "output": {"filename": "predictions.csv", "sha256": sha256(csv_path), "rows": n},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "training_data": "none" if model is None else model.get("metadata", {}),
        "ground_truth_used_for_inference": False,
    }
    if model is not None:
        manifest["learned_model"] = {
            "filename": "model.json", "sha256": sha256(destination / "model.json"),
            "decoder_mode": model.get("decoder_mode", "joint"),
        }
    write_json(destination / "manifest.json", manifest)
    marker.unlink()
    progress(f"Validated {n:,} rows. Run saved to {destination}")
    return manifest, report
