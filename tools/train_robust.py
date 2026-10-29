"""Fixed, grouped development CV for camera, possession-change and missing-ball heads.

All seven public Sportec matches are previously exposed. These results are NOT a
fresh test estimate. The frozen dead head's two training matches are never scored.
No competition reference labels are loaded by this command.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import control, robust_control
from possession.config import load_config
from possession.features import extract
from possession.io import sha256, write_json
from possession.learned import feature_matrix as dead_features, learned_probabilities, load_model, fuse_control_probabilities
from possession.tags import build
from tools.camera_augmentation import corrupt_camera_tracking, CAMERA_AUGMENTATION_SPEC
from tools.control_diagnostics import training_spell_weights
from tools.train_control import CATALOG, SEEDS, read_external, score_job
from tools.train_stoppages import corrupt_tracking

BASELINE = "release-1.2"
REGIMES = ("clean", "degraded", "camera")
FOLDS = (("J03WN1", "J03WMX"), ("J03WPY", "J03WQQ"), ("J03WR9",))
CANDIDATES = {
    "camera-control": {"regimes": REGIMES, "base_only": True, "cost": 1., "duration": .4},
    "context-control": {"regimes": REGIMES[:2], "base_only": False, "cost": 1., "duration": .4},
    "camera-context": {"regimes": REGIMES, "base_only": False, "cost": 1., "duration": .4},
    "camera-context-fast": {"regimes": REGIMES, "base_only": False, "cost": .5, "duration": .2},
}
STRIDE = 10


def scientific_hashes():
    files = sorted((ROOT / "possession").glob("*.py")) + [ROOT / name for name in (
        "tools/train_robust.py", "tools/camera_augmentation.py", "tools/train_control.py",
        "tools/control_diagnostics.py", "tools/train_stoppages.py",
        "tools/external_benchmark.py", "evaluation/possession_metrics.py")]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def select(records):
    """Require all fixed out-of-fold cells; compare each to the same frozen baseline."""
    names = {BASELINE, *CANDIDATES}
    expected = {(m, r) for fold in FOLDS for m in fold for r in REGIMES}
    grouped = {name: {} for name in names}
    for row in records:
        name, key = row["candidate"], (row["match"], row["regime"])
        if name not in names or key in grouped[name]:
            raise ValueError("Unknown candidate or duplicate evaluation cell")
        for metric in ("spellF1", "macroF1", "dead_recall"):
            if not np.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError("Invalid evaluation score")
        grouped[name][key] = row
    if any(set(rows) != expected for rows in grouped.values()):
        raise ValueError("Missing predeclared out-of-fold cells")
    gates = {}
    baseline = grouped[BASELINE]
    for name, rows in grouped.items():
        gains = {f"{m}/{r}": rows[m,r]["spellF1"] - baseline[m,r]["spellF1"] for m,r in sorted(expected)}
        mean = float(np.mean(list(gains.values())))
        death_drop = min(rows[k]["dead_recall"] - baseline[k]["dead_recall"] for k in expected)
        transition_gain = float(np.mean([rows[k]["control_diagnostics"]["transitions"]["recall"] - baseline[k]["control_diagnostics"]["transitions"]["recall"] for k in expected]))
        short_gain = float(np.mean([rows[k]["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"] - baseline[k]["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"] for k in expected]))
        gates[name] = {"passed": bool(mean >= .02 and min(gains.values()) >= -.02 and death_drop >= -.01 and transition_gain >= 0 and short_gain >= 0),
            "mean_spellF1_gain": mean, "cell_gains": gains, "minimum_dead_recall_delta": float(death_drop),
            "mean_transition_recall_gain": transition_gain, "mean_short_spell_recall_gain": short_gain}
    ranking = sorted(names, key=lambda name: (-np.mean([r["spellF1"] for r in grouped[name].values()]),
                                            -np.mean([r["macroF1"] for r in grouped[name].values()]), name))
    return {"selected": next(name for name in ranking if name == BASELINE or gates[name]["passed"]),
            "ranking": ranking, "gates": gates}


def prepare_cache(out, data_dir, baseline, config):
    for match in sorted(CATALOG):
        print(f"Preparing input-only observations: {match}", flush=True)
        tracking, reference, blocks, pitch, alignment, provenance = read_external(data_dir, match)
        folder = out / "cache" / match
        folder.mkdir(parents=True)
        np.save(folder / "reference.npy", reference)
        weights = training_spell_weights(reference, 25)
        write_json(folder / "metadata.json", {"blocks": blocks, "pitch": list(pitch), "inputs": provenance, "alignment": alignment})
        for regime in REGIMES:
            observed = tracking if regime == "clean" else (
                corrupt_tracking(tracking, blocks, pitch, 25, SEEDS[match]) if regime == "degraded" else
                corrupt_camera_tracking(tracking, blocks, pitch, 25, SEEDS[match] + 91001))
            features = extract(observed, pitch, blocks, 25, config)
            tags = build([], features, observed.team_names, len(observed.ball), 25, blocks, config)
            matrix = robust_control.feature_matrix(observed, features, blocks, pitch, 25)
            old = learned_probabilities(features, tags, dead_features(observed, features, blocks, pitch, 25),
                                        baseline, matrix[:, :len(control.FEATURE_NAMES)])
            usable = [bool(features.reliability[a:b+1].sum()) for _, a,b in blocks]
            np.save(folder / f"{regime}-features.npy", matrix)
            np.save(folder / f"{regime}-baseline.npy", old)
            np.savez(folder / f"{regime}-train.npz", X=matrix[::STRIDE], y=reference[::STRIDE], w=weights[::STRIDE])
            write_json(folder / f"{regime}-meta.json", {"usable": usable, "ball_observed_fraction": float(features.observed.mean())})
            print(f"  {regime}: {len(matrix):,} frames, ball observed {features.observed.mean():.3f}", flush=True)
            del observed, features, matrix, old


def fit(out, matches, candidate, metadata):
    spec = CANDIDATES[candidate]
    xs, ys, ws = [], [], []
    for match in sorted(matches):
        for regime in spec["regimes"]:
            with np.load(out / "cache" / match / f"{regime}-train.npz") as item:
                xs.append(item["X"]); ys.append(item["y"]); ws.append(item["w"])
    mask = np.ones(len(robust_control.FEATURE_NAMES), dtype=bool)
    if spec["base_only"]: mask[len(control.FEATURE_NAMES):] = False
    print(f"Fitting {candidate} on {','.join(sorted(matches))}", flush=True)
    model = robust_control.fit_model(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws),
        l2=.001, feature_mask=mask, metadata=dict(metadata, training_matches=sorted(matches), training_regimes=list(spec["regimes"])))
    return model


def evaluate(pool, out, match, heads, config, fold):
    folder = out / "cache" / match
    reference = np.load(folder / "reference.npy")
    meta = json.loads((folder / "metadata.json").read_text())
    rows = []
    for regime in REGIMES:
        matrix = np.load(folder / f"{regime}-features.npy", mmap_mode="r")
        old = np.load(folder / f"{regime}-baseline.npy")
        usable = json.loads((folder / f"{regime}-meta.json").read_text())["usable"]
        jobs = []
        for candidate in (BASELINE, *CANDIDATES):
            effective, probabilities = config, old
            if candidate != BASELINE:
                spec = CANDIDATES[candidate]
                effective = replace(config, team_switch_cost=spec["cost"], min_team_seconds=spec["duration"])
                home = robust_control.predict_home(heads[candidate], matrix)
                dead = old[:, 2]
                probabilities = fuse_control_probabilities(dead, home, np.zeros(len(home)))
            jobs.append(pool.submit(score_job, out / "cv" / match / regime / candidate, match, candidate,
                regime, reference, probabilities, meta["blocks"], effective, usable, 0))
        for job in as_completed(jobs):
            row, _ = job.result()
            row["fold"] = fold
            rows.append(row)
    return rows


def run(args):
    out = args.out
    if out.exists() and any(out.iterdir()): raise ValueError("Use a new empty campaign directory")
    out.mkdir(parents=True, exist_ok=True)
    baseline = load_model(args.baseline)
    config = load_config(args.config)
    if sha256(args.baseline) != "115c192cd4aa71be837e1c4bd2137d91add6dcea4b006a809b13232e62f43c0b":
        raise ValueError("Campaign comparator must be frozen v1.2")
    frozen = scientific_hashes()
    protocol = {"schema_version": 3, "baseline_sha256": sha256(args.baseline), "config_sha256": sha256(args.config),
        "source_hashes": frozen, "candidates": CANDIDATES, "folds": FOLDS, "all_matches": sorted(CATALOG),
        "regimes": REGIMES, "train_stride": STRIDE, "l2": .001, "camera_spec": CAMERA_AUGMENTATION_SPEC,
        "seeds": {m: SEEDS[m] + 91001 for m in CATALOG}, "features": list(robust_control.FEATURE_NAMES),
        "exposure": "All seven matches have previously been inspected. Development CV only; no untouched final test.",
        "split_policy": "Control training excludes every match in the scored fold. Frozen dead head trained WOH/WOY; neither is scored. Train on remaining catalog matches.",
        "selection": "Complete 15-cell out-of-fold evidence; equal-cell spellF1 then macroF1 then name among eligible models, baseline always eligible.",
        "gate": {"mean_spellF1_gain": .02, "maximum_cell_spellF1_drop": .02, "maximum_cell_dead_recall_drop": .01,
                 "minimum_mean_transition_recall_gain": 0, "minimum_mean_under2s_spell_recall_gain": 0},
        "refit": "If eligible, refit selected head on all seven public matches at fixed settings. CV measures training procedure, not this final artifact on unseen matches.",
        "competition_labels_used": False, "external_pretrained_checkpoints_used": False}
    write_json(out / "protocol.json", protocol)
    protocol_hash = sha256(out / "protocol.json")
    prepare_cache(out, args.data_dir, baseline, config)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, held_out in enumerate(FOLDS):
            train = set(CATALOG) - set(held_out)
            heads = {}
            for candidate in CANDIDATES:
                if candidate == "camera-context-fast":
                    heads[candidate] = heads["camera-context"]
                else:
                    heads[candidate] = fit(out, train, candidate, {"protocol_sha256": protocol_hash, "fold": index})
                robust_control.save_model(out / "fold-models" / str(index) / f"{candidate}.json", heads[candidate])
            for match in held_out:
                records.extend(evaluate(pool, out, match, heads, config, index))
            write_json(out / f"fold-{index}-records.json", records)
    if scientific_hashes() != frozen: raise ValueError("Scientific source changed during campaign")
    selection = select(records)
    write_json(out / "selection-before-refit.json", selection)
    selected = selection["selected"]
    release = copy.deepcopy(baseline)
    if selected != BASELINE:
        head = fit(out, set(CATALOG), selected, {"protocol_sha256": protocol_hash, "development_cv_selection": selected})
        release["robust_control_model"] = head
        release["decoder_settings"] = {"team_switch_cost": CANDIDATES[selected]["cost"], "min_team_seconds": CANDIDATES[selected]["duration"]}
        release["candidate_id"] = selected
    write_json(out / "release-model.json", release)
    if selected == BASELINE:
        (out / "release-model.json").write_bytes(args.baseline.read_bytes())
    write_json(out / "release-config.json", replace(config, **release.get("decoder_settings", {})).as_dict())
    if scientific_hashes() != frozen: raise ValueError("Scientific source changed during refit")
    summary = {"protocol": protocol, "protocol_sha256": protocol_hash, "source_hashes": frozen,
        "records": records, "selection": selection, "promoted": selected != BASELINE,
        "release_model_sha256": sha256(out / "release-model.json"), "competition_labels_used": False,
        "independent_test_claimed": False}
    write_json(out / "summary.json", summary)
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/robust-campaign-v3")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/baselines/v1.2.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines/v1.2.json")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    run(parser.parse_args())
