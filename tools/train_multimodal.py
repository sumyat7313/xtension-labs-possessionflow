"""Fixed calibrated-camera and temporal-expert development campaign.

All seven Sportec matches are exposed development data. Comparisons use v1.3
out-of-fold control heads, never its all-seven refit on a scored match. Target
possession labels are not read. An independent-provider audit is a separate step
after selection is frozen; it must not select a runner-up or tune this campaign.
"""
from __future__ import annotations

import argparse
import ast
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from possession import robust_control, temporal
from possession.config import load_config
from possession.features import extract
from possession.io import sha256, write_json
from possession.learned import (feature_matrix as dead_features, predict_dead,
                                fuse_control_probabilities, load_model, validate_model)
from tools.calibrated_camera import corrupt_calibrated_tracking, validate_profile
from tools.control_diagnostics import training_spell_weights
from tools.train_control import CATALOG, SEEDS, read_external, score_job
from tools import train_sequence

BASELINE = "baseline-v1.3"
CANDIDATES = ("calibrated-control", "temporal", "gated-temporal")
PRIMARY_REGIMES = ("clean", "degraded", "camera")
REGIMES = PRIMARY_REGIMES + ("calibrated",)
FOLDS = (("J03WN1", "J03WMX"), ("J03WPY", "J03WQQ"), ("J03WR9",))
CONTROL_STRIDE = 10
TEMPORAL_STRIDE = 5
TEMPORAL_SEED = 9207
EPOCHS = 8
BASELINE_SHA256 = "f4feb096face6a2a3eb1214079a5f34ea8b495147478b7a92b6b224a42dc0573"
CONFIG_SHA256 = "2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b"
ARCHIVE_SHA256 = "6bba5e359ec0c425aba15a764bf8c7e76c42ee804004de6eca311e2af7d2decb"


def scientific_hashes():
    files = sorted((ROOT / "possession").glob("*.py")) + [ROOT / name for name in (
        "tools/train_multimodal.py", "tools/train_sequence.py", "tools/calibrated_camera.py",
        "tools/camera_augmentation.py", "tools/train_robust.py", "tools/train_control.py",
        "tools/control_diagnostics.py", "tools/train_stoppages.py", "tools/external_benchmark.py",
        "evaluation/possession_metrics.py")]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def _score(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value) and 0 <= value <= 1


def select(records):
    """Fixed complete-cell gate; rank on the three pre-existing conditions only.

    Calibrated condition is a non-regression stress cell, not an opportunity to
    win by matching the newly introduced augmentation distribution alone.
    """
    names = (BASELINE,) + CANDIDATES
    cells = sorted((m, r) for fold in FOLDS for m in fold for r in REGIMES)
    primary = [cell for cell in cells if cell[1] in PRIMARY_REGIMES]
    grouped = {name: {} for name in names}
    for row in records:
        name, key = row["candidate"], (row["match"], row["regime"])
        if name not in grouped or key not in cells or key in grouped[name]:
            raise ValueError("Unknown candidate/cell or duplicated evaluation cell")
        try:
            values = [row[k] for k in ("spellF1", "macroF1", "dead_recall")]
            values += [row["control_diagnostics"]["transitions"]["recall"],
                       row["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"]]
        except (KeyError, TypeError):
            raise ValueError("Missing required evaluation diagnostic") from None
        if not all(_score(value) for value in values):
            raise ValueError("Scores and required diagnostics must be finite probabilities")
        grouped[name][key] = row
    if any(set(rows) != set(cells) for rows in grouped.values()):
        raise ValueError("Every fixed candidate requires all 20 match/regime cells")
    gates, means = {}, {}
    baseline = grouped[BASELINE]
    for name, rows in grouped.items():
        gains = {f"{m}/{r}": float(rows[m, r]["spellF1"] - baseline[m, r]["spellF1"]) for m, r in cells}
        primary_gain = float(np.mean([rows[k]["spellF1"] - baseline[k]["spellF1"] for k in primary]))
        death = min(rows[k]["dead_recall"] - baseline[k]["dead_recall"] for k in cells)
        transition = float(np.mean([rows[k]["control_diagnostics"]["transitions"]["recall"]
                                    - baseline[k]["control_diagnostics"]["transitions"]["recall"] for k in cells]))
        short = float(np.mean([rows[k]["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"]
                               - baseline[k]["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"] for k in cells]))
        means[name] = {"primary_spellF1": float(np.mean([rows[k]["spellF1"] for k in primary])),
                       "primary_macroF1": float(np.mean([rows[k]["macroF1"] for k in primary])),
                       "all_conditions_spellF1": float(np.mean([rows[k]["spellF1"] for k in cells]))}
        gates[name] = {"passed": bool(primary_gain >= .01 and min(gains.values()) >= -.02
                                      and death >= -.01 and transition >= 0 and short >= 0),
                       "primary_mean_spellF1_gain": primary_gain, "cell_gains": gains,
                       "minimum_dead_recall_delta": float(death),
                       "mean_transition_recall_gain": transition, "mean_short_spell_recall_gain": short}
    ranking = sorted(names, key=lambda name: (-means[name]["primary_spellF1"], -means[name]["primary_macroF1"], name))
    selected = next(name for name in ranking if name == BASELINE or gates[name]["passed"])
    return {"selected": selected, "ranking": ranking, "gates": gates, "means": means,
            "independent_audit_used_for_selection": False}


def partition(held_out):
    """Outer training and deterministic inner gate holdout, both match-separated."""
    held = set(held_out)
    if not held or not held <= set(CATALOG):
        raise ValueError("Outer held-out matches must belong to the fixed catalog")
    train = sorted(set(CATALOG) - held)
    if len(train) < 3:
        raise ValueError("Need at least three outer-training matches")
    return train, train[0], train[1:]


def base_logits(head, matrix):
    robust_control.validate_model(head)
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(robust_control.FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid control observation matrix")
    return (x / np.asarray(head["scale"])) @ np.asarray(head["coefficients"])


def _function_ast(source, name):
    tree = ast.parse(source)
    return ast.dump(next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name), include_attributes=False)


def verify_history(previous, baseline_path, config_path, archive_path):
    """Authenticate frozen sources/model provenance before reusing local caches.

    Historical v3 did not record cache-file hashes. This runner records them now,
    checks cached subsamples/reference/provenance internally, rereads raw labels
    for exact equality, and later reproduces the incumbent's historical scores.
    It does not falsely claim those new fingerprints existed in the old release.
    """
    if sha256(baseline_path) != BASELINE_SHA256 or sha256(config_path) != CONFIG_SHA256 or sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Frozen v1.3 model, configuration, or source archive differs")
    summary = json.loads((previous / "summary.json").read_text())
    if summary != json.loads((ROOT / "models/robust-selection.json").read_text()):
        raise ValueError("Historical local campaign differs from committed v1.3 evidence")
    if summary["release_model_sha256"] != BASELINE_SHA256 or summary["selection"]["selected"] != "camera-context":
        raise ValueError("Historical selected model does not identify v1.3")
    if sha256(previous / "protocol.json") != summary["protocol_sha256"]:
        raise ValueError("Historical protocol hash differs")
    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in summary["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archived historical scientific source differs: {name}")
        old_learned = archive.read("possession/learned.py").decode("utf-8-sig")
    unchanged = ("possession/changes.py", "possession/config.py", "possession/control.py", "possession/decoder.py",
                 "possession/features.py", "possession/io.py", "possession/missing_ball.py", "possession/robust_control.py",
                 "possession/tags.py", "tools/camera_augmentation.py", "tools/control_diagnostics.py",
                 "tools/external_benchmark.py", "tools/train_stoppages.py", "evaluation/possession_metrics.py")
    for name in unchanged:
        if sha256(ROOT / name) != summary["source_hashes"][name]:
            raise ValueError(f"Cached-observation scientific dependency changed: {name}")
    current_learned = (ROOT / "possession/learned.py").read_text(encoding="utf-8-sig")
    for name in ("feature_matrix", "predict_dead", "fuse_control_probabilities"):
        if _function_ast(old_learned, name) != _function_ast(current_learned, name):
            raise ValueError(f"Frozen dead/features or fusion implementation changed: {name}")
    fold_hashes = {}
    for index, held in enumerate(FOLDS):
        path = previous / "fold-models" / str(index) / "camera-context.json"
        head = robust_control.load_model(path)
        metadata = head["metadata"]
        if (metadata["training_matches"] != sorted(set(CATALOG) - set(held))
                or metadata["protocol_sha256"] != summary["protocol_sha256"]
                or metadata["training_regimes"] != list(PRIMARY_REGIMES) or metadata["fold"] != index):
            raise ValueError("Historical incumbent fold-model separation/provenance differs")
        fold_hashes[str(index)] = sha256(path)
    return summary, {"historical_summary_sha256": sha256(previous / "summary.json"),
                     "historical_protocol_sha256": summary["protocol_sha256"],
                     "source_archive_sha256": ARCHIVE_SHA256, "incumbent_fold_model_sha256": fold_hashes,
                     "cache_hashes_previously_recorded": False}


def prepare_cache(out, previous, data_dir, baseline, config, profile):
    provenance = {}
    for match in sorted(CATALOG):
        print(f"Preparing calibrated observations and verifying {match}", flush=True)
        old, folder = previous / "cache" / match, out / "cache" / match
        folder.mkdir(parents=True)
        names = ["reference.npy", "metadata.json"] + [f"{r}-{suffix}" for r in PRIMARY_REGIMES
                    for suffix in ("features.npy", "baseline.npy", "train.npz", "meta.json")]
        fingerprints = {}
        for name in names:
            fingerprints[name] = sha256(old / name)
            shutil.copyfile(old / name, folder / name)
            if sha256(folder / name) != fingerprints[name]:
                raise ValueError("Historical cache copy changed bytes")
        reference = np.load(folder / "reference.npy", allow_pickle=False)
        weights = training_spell_weights(reference, 25)
        np.save(folder / "weights.npy", weights)
        for regime in PRIMARY_REGIMES:
            matrix = np.load(folder / f"{regime}-features.npy", mmap_mode="r", allow_pickle=False)
            if matrix.shape != (len(reference), len(robust_control.FEATURE_NAMES)) or not np.isfinite(matrix).all():
                raise ValueError("Invalid historical feature cache")
            probabilities = np.load(folder / f"{regime}-baseline.npy", mmap_mode="r", allow_pickle=False)
            if probabilities.shape != (len(reference), 3) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
                raise ValueError("Invalid historical emission cache")
            with np.load(folder / f"{regime}-train.npz", allow_pickle=False) as subsample:
                for key, expected in (("X", matrix[::CONTROL_STRIDE]), ("y", reference[::CONTROL_STRIDE]), ("w", weights[::CONTROL_STRIDE])):
                    if not np.array_equal(subsample[key], expected):
                        raise ValueError("Historical training cache disagrees with full observation/reference cache")
        tracking, fresh_reference, blocks, pitch, alignment, inputs = read_external(data_dir, match)
        metadata = json.loads((folder / "metadata.json").read_text())
        if (not np.array_equal(reference, fresh_reference) or metadata["inputs"] != inputs
                or metadata["blocks"] != [list(b) for b in blocks]
                or metadata["pitch"] != list(pitch)
                or metadata["alignment"] != json.loads(json.dumps(alignment))):
            raise ValueError("Historical cached references/alignment/provenance differ from verified raw inputs")
        observed = corrupt_calibrated_tracking(tracking, blocks, pitch, 25, SEEDS[match] + 192071, profile)
        features = extract(observed, pitch, blocks, 25, config)
        matrix = robust_control.feature_matrix(observed, features, blocks, pitch, 25)
        dead = predict_dead(dead_features(observed, features, blocks, pitch, 25), baseline)
        np.save(folder / "calibrated-features.npy", matrix)
        np.save(folder / "calibrated-dead.npy", dead)
        np.savez(folder / "calibrated-train.npz", X=matrix[::CONTROL_STRIDE], y=reference[::CONTROL_STRIDE], w=weights[::CONTROL_STRIDE])
        write_json(folder / "calibrated-meta.json", {"usable": [bool(features.reliability[a:b + 1].sum()) for _, a, b in blocks],
                                                     "ball_observed_fraction": float(features.observed.mean())})
        provenance[match] = {"copied_cache_sha256": fingerprints, "inputs": inputs,
                             "fresh_reference_equals_cached": True,
                             "calibrated_files_sha256": {p.name: sha256(p) for p in sorted(folder.glob("calibrated-*"))}}
        print(f"  calibrated: {len(reference):,} frames, ball observed {features.observed.mean():.3f}", flush=True)
        del tracking, observed, features, matrix, dead, probabilities, weights
        write_json(out / "cache-provenance.json", provenance)
    return provenance


def fit_control(out, matches, metadata):
    xs, ys, ws = [], [], []
    for match in sorted(matches):
        for regime in REGIMES:
            with np.load(out / "cache" / match / f"{regime}-train.npz", allow_pickle=False) as item:
                xs.append(item["X"]); ys.append(item["y"]); ws.append(item["w"])
    print("Fitting calibrated control on " + ",".join(sorted(matches)), flush=True)
    return robust_control.fit_model(np.concatenate(xs), np.concatenate(ys), l2=.001,
        sample_weight=np.concatenate(ws), metadata=dict(metadata, training_matches=sorted(matches), training_regimes=list(REGIMES)))


def sequences(out, matches, head):
    """One 5-Hz sequence per half and regime; sampling resets at half start."""
    result = []
    for match in sorted(matches):
        folder = out / "cache" / match
        reference = np.load(folder / "reference.npy", allow_pickle=False)
        weights = np.load(folder / "weights.npy", allow_pickle=False)
        blocks = json.loads((folder / "metadata.json").read_text())["blocks"]
        for regime in REGIMES:
            matrix = np.load(folder / f"{regime}-features.npy", mmap_mode="r", allow_pickle=False)
            for half, start, last in blocks:
                indices = np.arange(start, last + 1, TEMPORAL_STRIDE)
                x = np.array(matrix[indices], copy=True)
                result.append({"X": x, "y": reference[indices], "w": weights[indices],
                               "base": base_logits(head, x), "match": match, "regime": regime, "half": half})
    return result


def fit_temporal(out, matches, head, metadata):
    print("Fitting temporal residual on " + ",".join(sorted(matches)), flush=True)
    return train_sequence.fit(sequences(out, matches, head), seed=TEMPORAL_SEED, epochs=EPOCHS,
        metadata=dict(metadata, training_matches=sorted(matches), training_regimes=list(REGIMES)))


def fit_inner_gate(out, train, metadata, destination):
    inner_holdout, inner_train = sorted(train)[0], sorted(train)[1:]
    if not inner_train:
        raise ValueError("Inner gate calibration must exclude an entire training match")
    inner_metadata = dict(metadata, stage="inner-gate", inner_holdout=inner_holdout)
    head = fit_control(out, inner_train, inner_metadata)
    network = fit_temporal(out, inner_train, head, inner_metadata)
    robust_control.save_model(destination / "inner-control.json", head)
    write_json(destination / "inner-temporal.json", network)
    folder = out / "cache" / inner_holdout
    reference = np.load(folder / "reference.npy", allow_pickle=False)
    weights = np.load(folder / "weights.npy", allow_pickle=False)
    blocks = json.loads((folder / "metadata.json").read_text())["blocks"]
    indices = np.concatenate([np.arange(a, b + 1, TEMPORAL_STRIDE) for _, a, b in blocks])
    matrices, bases, experts, labels, ws = [], [], [], [], []
    for regime in REGIMES:
        matrix = np.load(folder / f"{regime}-features.npy", mmap_mode="r", allow_pickle=False)
        base = base_logits(head, matrix)
        expert = base + temporal.residual(network, matrix, blocks, 25)
        matrices.append(np.array(matrix[indices])); bases.append(base[indices]); experts.append(expert[indices])
        labels.append(reference[indices]); ws.append(weights[indices])
    gate = train_sequence.fit_gate(np.concatenate(matrices), np.concatenate(bases), np.concatenate(experts),
        np.concatenate(labels), np.concatenate(ws), metadata=dict(inner_metadata,
            expert_training_matches=inner_train, gate_training_matches=[inner_holdout], training_regimes=list(REGIMES)))
    write_json(destination / "gate.json", gate)
    return gate


def evaluate(pool, out, previous, match, index, head, network, gate, config):
    folder = out / "cache" / match
    reference = np.load(folder / "reference.npy", allow_pickle=False)
    blocks = json.loads((folder / "metadata.json").read_text())["blocks"]
    incumbent = robust_control.load_model(previous / "fold-models" / str(index) / "camera-context.json")
    rows = []
    for regime in REGIMES:
        matrix = np.load(folder / f"{regime}-features.npy", mmap_mode="r", allow_pickle=False)
        dead = (np.load(folder / "calibrated-dead.npy", allow_pickle=False) if regime == "calibrated" else
                np.load(folder / f"{regime}-baseline.npy", allow_pickle=False)[:, 2])
        usable = json.loads((folder / f"{regime}-meta.json").read_text())["usable"]
        base = base_logits(head, matrix)
        probabilities = {
            BASELINE: robust_control.predict_home(incumbent, matrix),
            "calibrated-control": robust_control.predict_home(head, matrix),
            "temporal": temporal.predict_home(base, matrix, blocks, 25, network),
            "gated-temporal": temporal.predict_home(base, matrix, blocks, 25, network, gate=gate),
        }
        jobs = [pool.submit(score_job, out / "cv" / match / regime / candidate, match, candidate, regime,
                            reference, fuse_control_probabilities(dead, home, np.zeros(len(reference))), blocks,
                            config, usable, 0) for candidate, home in probabilities.items()]
        for job in as_completed(jobs):
            row, _ = job.result()
            row["fold"] = index
            rows.append(row)
    return rows


def verify_incumbent_scores(records, historical):
    expected = {(row["match"], row["regime"]): row for row in historical["records"] if row["candidate"] == "camera-context"}
    rows = [row for row in records if row["candidate"] == BASELINE and row["regime"] in PRIMARY_REGIMES]
    if {(row["match"], row["regime"]) for row in rows} != set(expected) or len(rows) != 15:
        raise ValueError("Historical incumbent score comparison is incomplete")
    for row in rows:
        old = expected[row["match"], row["regime"]]
        for metric in ("spellF1", "macroF1", "dead_recall", "dead_precision"):
            if not np.isclose(row[metric], old[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Incumbent historical score changed: {row['match']}/{row['regime']}/{metric}")
        if row["prediction_sha256"] != old["prediction_sha256"]:
            raise ValueError("Incumbent historical predictions changed")


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new empty output directory; campaign evidence is immutable")
    historical, provenance = verify_history(args.previous, args.baseline, args.config, args.archive)
    baseline, config = load_model(args.baseline), load_config(args.config)
    config = replace(config, team_switch_cost=1., min_team_seconds=.4).validate()
    profile = validate_profile(json.loads(args.profile.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = scientific_hashes()
    protocol = {
        "schema_version": 4, "source_hashes": frozen, "baseline": BASELINE, "baseline_sha256": BASELINE_SHA256,
        "baseline_comparison": "Historical v1.3 out-of-fold heads on scored folds; never all-seven refit.",
        "config_sha256": CONFIG_SHA256, "profile_sha256": sha256(args.profile), "historical_provenance": provenance,
        "folds": FOLDS, "all_matches": sorted(CATALOG), "candidates": [BASELINE, *CANDIDATES], "regimes": REGIMES,
        "primary_regimes": PRIMARY_REGIMES, "control_stride": CONTROL_STRIDE, "control_l2": .001,
        "temporal_sample_hz": 5, "temporal_seed": TEMPORAL_SEED, "temporal_epochs": EPOCHS,
        "calibrated_seeds": {m: SEEDS[m] + 192071 for m in sorted(CATALOG)},
        "inner_gate_policy": "Within each outer training set, lexicographically first match is held out; fit inner control and temporal experts on remaining matches, then gate on this inner match. Refit outer experts on all outer training matches.",
        "gate": {"primary_mean_spellF1_gain": .01, "maximum_any_cell_spellF1_drop": .02,
                 "maximum_any_cell_dead_recall_drop": .01, "minimum_all_cell_mean_transition_recall_gain": 0,
                 "minimum_all_cell_mean_under2s_spell_recall_gain": 0},
        "selection": "Rank eligible candidates by primary-condition equal-cell spell F1, then macro F1, then name; baseline fallback.",
        "exposure": "All seven Sportec matches previously exposed. Development CV only, not an untouched test.",
        "dead_head": "Frozen v1.3 dead weights; WOH/WOY used in its historical training and never scored.",
        "refit": "Freeze selection before all-seven refit; final artifact itself has no Sportec holdout.",
        "independent_audit_policy": "A separate Game2/provider audit may compare only frozen selected candidate vs baseline after selection; it must not tune, select a runner-up, or silently trigger retraining.",
        "target_calibration": "Unlabeled aggregate tracking/homography profile; no target possession labels, frame-geometry publication, or event labels.",
        "competition_labels_used": False, "independent_audit_used_for_selection": False,
    }
    write_json(args.out / "protocol.json", protocol)
    protocol_hash = sha256(args.out / "protocol.json")
    incumbent_snapshot = args.out / "incumbent"
    for index in range(len(FOLDS)):
        relative = Path("fold-models") / str(index) / "camera-context.json"
        destination = incumbent_snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.previous / relative, destination)
        if sha256(destination) != provenance["incumbent_fold_model_sha256"][str(index)]:
            raise ValueError("Incumbent fold model changed before its immutable snapshot")
    cache = prepare_cache(args.out, args.previous, args.data_dir, baseline, config, profile)
    if scientific_hashes() != frozen:
        raise ValueError("Scientific source changed before fitting")
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, held in enumerate(FOLDS):
            train, inner, inner_train = partition(held)
            meta = {"protocol_sha256": protocol_hash, "fold": index, "held_out_matches": list(held)}
            destination = args.out / "fold-models" / str(index)
            gate = fit_inner_gate(args.out, train, meta, destination)
            head = fit_control(args.out, train, meta)
            network = fit_temporal(args.out, train, head, meta)
            robust_control.save_model(destination / "calibrated-control.json", head)
            write_json(destination / "temporal.json", network)
            for match in held:
                records.extend(evaluate(pool, args.out, incumbent_snapshot, match, index, head, network, gate, config))
            write_json(args.out / f"fold-{index}-records.json", records)
    if scientific_hashes() != frozen:
        raise ValueError("Scientific source changed during CV")
    verify_incumbent_scores(records, historical)
    selection = select(records)
    write_json(args.out / "selection-before-independent-audit.json", {"protocol_sha256": protocol_hash, **selection})
    selected = selection["selected"]
    release = copy.deepcopy(baseline)
    if selected != BASELINE:
        meta = {"protocol_sha256": protocol_hash, "development_cv_selection": selected, "stage": "all-seven-refit"}
        train = sorted(CATALOG)
        gate = fit_inner_gate(args.out, train, meta, args.out / "refit-models") if selected == "gated-temporal" else None
        head = fit_control(args.out, train, meta)
        release["robust_control_model"] = head
        release.pop("temporal_model", None); release.pop("expert_gate", None)
        if selected in ("temporal", "gated-temporal"):
            release["temporal_model"] = fit_temporal(args.out, train, head, meta)
        if gate is not None:
            release["expert_gate"] = gate
        release["candidate_id"] = selected
        release["decoder_settings"] = {"team_switch_cost": 1., "min_team_seconds": .4}
    validate_model(release)
    write_json(args.out / "release-candidate-model.json", release)
    if selected == BASELINE:
        (args.out / "release-candidate-model.json").write_bytes(args.baseline.read_bytes())
    write_json(args.out / "release-candidate-config.json", config.as_dict())
    if scientific_hashes() != frozen or sha256(args.profile) != protocol["profile_sha256"]:
        raise ValueError("Scientific source changed during refit")
    summary = {"protocol": protocol, "protocol_sha256": protocol_hash, "source_hashes": frozen,
        "records": sorted(records, key=lambda row: (row["match"], row["regime"], row["candidate"])),
        "selection": selection, "historical_provenance": provenance, "cache_provenance": cache,
        "incumbent_historical_predictions_reproduced": True, "development_candidate_selected": selected != BASELINE,
        "independent_audit_status": "pending-separate-audit", "competition_labels_used": False,
        "independent_test_claimed": False, "release_candidate_model_sha256": sha256(args.out / "release-candidate-model.json")}
    write_json(args.out / "summary.json", summary)
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/multimodal-campaign-v4")
    parser.add_argument("--previous", type=Path, default=ROOT / "runs/robust-campaign-v3")
    parser.add_argument("--profile", type=Path, default=ROOT / "runs/calibrated-camera/profile.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/baselines/v1.3.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines/v1.3.json")
    parser.add_argument("--archive", type=Path, default=ROOT / "models/baselines/v1.3-source.zip")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    run(parser.parse_args())
