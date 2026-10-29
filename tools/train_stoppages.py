"""Frozen, match-separated external training campaign; never reads challenge labels."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.external_benchmark import MATCHES, SOURCE, LICENSE, digest, read_match
from possession.config import load_config
from possession.features import extract
from possession.io import Tracking, sha256, write_json, write_submission
from possession.tags import build
from possession.decoder import decode, observation_probabilities
from evaluation.possession_metrics import (evaluate, label_codes, valid_runs, fill_abstentions,
    possession_spells, spell_overlaps, matched_gt_flags, SPELL_IOUS)

CATALOG = dict(MATCHES, **{
    "J03WOY": {"metadata": (51643481, 11952, "7b021e577fca917a63111d55cc29ae5b"),
                "tracking": (51643523, 362567859, "4e1063e0a0ae499920cf68ac1aa3ab4e")},
    "J03WMX": {"metadata": (51643475, 11930, "bc5d5b21dbcf36aa30f4cd0428309f6b"),
                "tracking": (51643514, 418524833, "e4273a2adf5dbd9d4ec42def2f01905c")},
})
SPLITS = {"train": ["J03WOH", "J03WOY"], "validation": ["J03WN1"], "final": ["J03WMX"]}
SEEDS = {"J03WOH": 41021, "J03WOY": 41039, "J03WN1": 51021, "J03WMX": 61021}
REGIMES = ("clean", "degraded")
L2_VALUES = (0.001, 0.01, 0.1)
MODES = ("joint", "separate")
MIN_MEAN_GAIN = 0.02
MAX_REGIME_DROP = 0.02


def corrupt_tracking(tracking, blocks, pitch, fps, seed):
    """Input-only structured corruption, independent of all target labels.

    Each half alternates visible 2--8 s and missing 0.4--6 s ball windows;
    random player slots are absent in 12 s blocks with probability .15. Gaussian
    position noise has SD .15 metres; speed noise .3 m/s. No target enters this API.
    """
    rng = np.random.default_rng(seed)
    result = copy.deepcopy(tracking)
    pitch = np.asarray(pitch, dtype=float)
    result.players = result.players.astype(np.float64)
    result.ball = result.ball.astype(np.float64)
    result.player_speed = result.player_speed.astype(np.float64)
    result.ball_speed = result.ball_speed.astype(np.float64)
    for _, start, last in blocks:
        stop = last + 1
        length = stop - start
        result.players[start:stop] += rng.normal(0, .15, (length, result.players.shape[1], 2)) / pitch
        result.ball[start:stop] += rng.normal(0, .15, (length, 2)) / pitch
        result.player_speed[start:stop] = np.maximum(0, result.player_speed[start:stop] + rng.normal(0, .3, result.player_speed[start:stop].shape))
        result.ball_speed[start:stop] = np.maximum(0, result.ball_speed[start:stop] + rng.normal(0, .3, length))
        cursor = start
        while cursor < stop:
            cursor += int(rng.integers(round(2 * fps), round(8 * fps) + 1))
            gap_end = min(stop, cursor + int(rng.integers(round(.4 * fps), round(6 * fps) + 1)))
            result.ball[cursor:gap_end] = np.nan
            result.ball_speed[cursor:gap_end] = np.nan
            cursor = gap_end
        for a in range(start, stop, round(12 * fps)):
            b = min(a + round(12 * fps), stop)
            absent = rng.random(result.players.shape[1]) < .15
            result.players[a:b, absent] = np.nan
            result.player_speed[a:b, absent] = np.nan
    return result


def candidate_id(l2, mode):
    return f"l2-{l2:g}_{mode}"


def promotion_gate(candidate, baseline):
    """Apply the preregistered paired clean/degraded non-regression gate."""
    gains = {regime: float(candidate[regime]["spellF1"] - baseline[regime]["spellF1"])
             for regime in REGIMES}
    mean_gain = float(np.mean(list(gains.values())))
    passed = mean_gain >= MIN_MEAN_GAIN and min(gains.values()) >= -MAX_REGIME_DROP
    return {"passed": bool(passed), "mean_spellF1_gain": mean_gain, "regime_gains": gains,
            "required_mean_gain": MIN_MEAN_GAIN, "maximum_regime_drop": MAX_REGIME_DROP}


def select_candidate(records):
    """Validation-only selection, including the baseline as an eligible fallback.

    records is a list of candidate, regime, spellF1, macroF1 dictionaries. Incomplete
    or duplicated regime records fail instead of silently changing the objective.
    """
    grouped = {}
    for row in records:
        cid, regime = row["candidate"], row["regime"]
        if regime not in REGIMES or regime in grouped.setdefault(cid, {}):
            raise ValueError("Unknown or duplicate candidate regime")
        if not all(np.isfinite(row[k]) for k in ("spellF1", "macroF1")):
            raise ValueError("Nonfinite candidate score")
        grouped[cid][regime] = row
    if "baseline" not in grouped or any(set(v) != set(REGIMES) for v in grouped.values()):
        raise ValueError("Every candidate must have clean and degraded validation scores")
    ranked = sorted(grouped, key=lambda cid: (-np.mean([r["spellF1"] for r in grouped[cid].values()]),
                                             -np.mean([r["macroF1"] for r in grouped[cid].values()]), cid))
    gates = {cid: promotion_gate(grouped[cid], grouped["baseline"]) for cid in grouped}
    eligible = [cid for cid in ranked if cid == "baseline" or gates[cid]["passed"]]
    winner = eligible[0]
    return {"selected": winner, "highest_validation_candidate": ranked[0],
            "ranking": ranked, "eligible_ranking": eligible,
            "candidate_gates": gates, "gate": gates[winner]}


def source_hashes():
    names = ["possession/config.py", "possession/io.py", "possession/features.py", "possession/tags.py",
             "possession/decoder.py", "possession/learned.py", "tools/external_benchmark.py",
             "tools/train_stoppages.py", "evaluation/possession_metrics.py"]
    return {name: sha256(ROOT / name) for name in names}


def read_external(data_dir, match):
    paths = []
    for kind in ("metadata", "tracking"):
        _, size, md5 = CATALOG[match][kind]
        path = data_dir / f"{match}_{kind}.xml"
        if not path.is_file() or path.stat().st_size != size or digest(path) != md5:
            raise ValueError(f"Missing or invalid external dataset file: {path.name}")
        paths.append(path)
    data, labels, blocks, pitch, alignment = read_match(*paths, match)
    provenance = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths}
    return data, labels, blocks, pitch, alignment, provenance


def download_inputs(data_dir):
    """Download pinned public inputs with sizes and published checksum verification."""
    data_dir.mkdir(parents=True, exist_ok=True)
    def fetch(pair):
        match, kind = pair
        ident, size, expected = CATALOG[match][kind]
        path = data_dir / f'{match}_{kind}.xml'
        if not path.exists():
            tmp = path.with_suffix('.download')
            print(f'Downloading {match} {kind} ({size:,} bytes)', flush=True)
            with urllib.request.urlopen(f'https://ndownloader.figshare.com/files/{ident}', timeout=60) as response, tmp.open('wb') as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if tmp.stat().st_size != size or digest(tmp) != expected:
                raise ValueError(f'Download checksum mismatch: {tmp.name}')
            tmp.replace(path)
        if path.stat().st_size != size or digest(path) != expected:
            raise ValueError(f'Existing input checksum mismatch: {path.name}')
        print(f'Verified {path.name}', flush=True)
    pairs = [(match, kind) for group in SPLITS.values() for match in group for kind in ('metadata','tracking')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, pairs))


def paired_spell_bootstrap(reference, baseline, candidate, draws=1000):
    """Same reference-possession resamples for both predictions, official matching.

    This uses the official scorer's fixed-n-pred bootstrap approximation. It is a
    diagnostic interval, not the predeclared release gate or a challenge rank.
    """
    truth = label_codes(reference)
    spans = valid_runs(truth)
    gt_spells = possession_spells(truth, spans)
    if not gt_spells:
        return {'draws':0,'mean_delta':None,'percentiles_5_95':[None,None]}
    comparisons=[]
    for labels in (baseline,candidate):
        filled=fill_abstentions(label_codes(labels),spans)
        predicted=possession_spells(filled,spans)
        overlaps=spell_overlaps(gt_spells,predicted)
        flags=np.vstack([matched_gt_flags(overlaps,len(gt_spells),threshold) for threshold in SPELL_IOUS])
        comparisons.append((flags,len(predicted)))
    rng=np.random.RandomState(62171)
    deltas=[]
    for _ in range(draws):
        picks=rng.randint(0,len(gt_spells),len(gt_spells))
        scores=[]
        for flags,n_pred in comparisons:
            tp=flags[:,picks].sum(axis=1)
            fp=np.maximum(0,n_pred-tp)
            fn=len(gt_spells)-tp
            scores.append(float(np.mean(np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(SPELL_IOUS)),where=(2*tp+fp+fn)>0))))
        deltas.append(scores[1]-scores[0])
    return {'draws':draws,'seed':62171,'mean_delta':float(np.mean(deltas)),
            'percentiles_5_95':[float(x) for x in np.percentile(deltas,[5,95])]}


def prepared(data, blocks, pitch, regime, match, config):
    from possession.learned import feature_matrix
    observations = data if regime == "clean" else corrupt_tracking(data, blocks, pitch, 25, SEEDS[match])
    features = extract(observations, pitch, blocks, 25, config)
    tags = build([], features, observations.team_names, len(observations.ball), 25, blocks, config)
    matrix = feature_matrix(observations, features, blocks, pitch, 25)
    return features, tags, matrix


def baseline_labels(features, tags, blocks, config):
    probabilities = observation_probabilities(features, tags, config)
    labels = np.full(len(probabilities), "X", dtype="<U1")
    for _, a, b in blocks:
        evidence = features.reliability[a:b + 1].sum() + tags.live[a:b + 1].sum() + tags.restart_live[a:b + 1].sum()
        if evidence:
            labels[a:b + 1] = np.array(list("HAD"))[decode(probabilities[a:b + 1], 25, config)]
    return labels


def measure(destination, match, candidate, regime, reference, labels, bootstrap=0):
    destination.mkdir(parents=True, exist_ok=True)
    result = evaluate(label_codes(reference), label_codes(labels), bootstrap_n=bootstrap)
    write_submission(destination / "predictions.csv", labels)
    write_json(destination / "metrics.json", result)
    row = {"match": match, "candidate": candidate, "regime": regime,
           "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
           "dead_precision": result["per_class"]["D"]["precision"],
           "dead_recall": result["per_class"]["D"]["recall"],
           "predicted_spells": result["n_pred_spells"], "reference_spells": result["n_gt_spells"],
           "prediction_sha256": sha256(destination / "predictions.csv")}
    if bootstrap:
        row["spellF1_bootstrap_5_95"] = [result["spell_bootstrap_low"], result["spell_bootstrap_high"]]
    print(json.dumps(row), flush=True)
    return row


def run(args):
    from possession.learned import fit_model, save_model, decode_learned, FEATURE_NAMES
    config = load_config(args.config)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose a fresh output folder; campaign records are immutable")
    protocol = {
        "dataset": SOURCE, "license": LICENSE, "splits": SPLITS, "input_catalog": CATALOG,
        "source_hashes": source_hashes(), "config": config.as_dict(), "config_sha256": sha256(args.config),
        "feature_names": list(FEATURE_NAMES), "candidate_l2": list(L2_VALUES), "candidate_decoders": list(MODES),
        "baseline_is_candidate": True, "train_stride": 5, "regimes": list(REGIMES), "seeds": SEEDS,
        "corruption": {"ball_visible_seconds": [2, 8], "ball_missing_seconds": [.4, 6],
                       "player_absent_probability": .15, "player_absent_blocks_seconds": 12,
                       "position_noise_metres_sd": .15, "speed_noise_mps_sd": .3},
        "selection": "Gate each validation candidate first (baseline always eligible); rank eligible by mean spellF1 over clean/degraded, then mean macroF1, then lexical ID",
        "promotion_gate": {"minimum_mean_spellF1_gain": MIN_MEAN_GAIN, "maximum_any_regime_drop": MAX_REGIME_DROP},
        "final_policy": "Freeze winner after validation, compare only winner and baseline on untouched final; same gate. Failure retains baseline; no runner-up search.",
        "tagging": "No external automatic tags; trained dead model has no tag features. Competition team tags remain separate.",
        "target_separation": "BallStatus/Possession are training targets or scoring only; corruption, features and decoding blocks never consume them.",
        "competition_public_labels_used": False,
    }
    write_json(out / "protocol.json", protocol)
    frozen_hashes = protocol["source_hashes"]
    training_x, training_y, training_meta = [], [], {}
    for match in SPLITS["train"]:
        print(f"Reading TRAIN {match}", flush=True)
        data, reference, blocks, pitch, alignment, provenance = read_external(args.data_dir, match)
        write_json(out / f"{match}-alignment.json", alignment)
        training_meta[match] = {"inputs": provenance, "reference_sha256": __import__('hashlib').sha256(reference.tobytes()).hexdigest(),
                                "known_frames": int(np.isin(reference, list("HAD")).sum()), "regimes": {}}
        for regime in REGIMES:
            _, _, matrix = prepared(data, blocks, pitch, regime, match, config)
            indices = np.arange(0, len(reference), 5)
            training_x.append(matrix[indices])
            training_y.append(reference[indices])
            training_meta[match]["regimes"][regime] = {"rows_before_unknown_mask": len(indices), "seed": None if regime == "clean" else SEEDS[match]}
        del data, reference, matrix
    X, y = np.concatenate(training_x), np.concatenate(training_y)
    del training_x, training_y
    models = {}
    for l2 in L2_VALUES:
        print(f"Fitting training-only logistic model l2={l2}", flush=True)
        model = fit_model(X, y, l2=l2, metadata={"dataset": SOURCE, "license": LICENSE,
                          "training_matches": training_meta, "validation_matches": SPLITS['validation'],
                          "final_test_matches": SPLITS['final'], "protocol_sha256": sha256(out / 'protocol.json')})
        for mode in MODES:
            candidate = candidate_id(l2, mode)
            record = copy.deepcopy(model)
            record["decoder_mode"] = mode
            record["candidate_id"] = candidate
            save_model(out / "models" / f"{candidate}.json", record)
            models[candidate] = record
    del X, y
    validation = []
    for match in SPLITS["validation"]:
        print(f"Reading VALIDATION {match}", flush=True)
        data, reference, blocks, pitch, alignment, provenance = read_external(args.data_dir, match)
        write_json(out / f"{match}-inputs.json", provenance)
        write_json(out / f"{match}-alignment.json", alignment)
        for regime in REGIMES:
            features, tags, matrix = prepared(data, blocks, pitch, regime, match, config)
            labels = baseline_labels(features, tags, blocks, config)
            validation.append(measure(out / "validation" / regime / "baseline", match, "baseline", regime, reference, labels))
            for candidate, model in models.items():
                labels, _, _ = decode_learned(features, tags, matrix, blocks, 25, config, model, mode=model['decoder_mode'])
                validation.append(measure(out / "validation" / regime / candidate, match, candidate, regime, reference, labels))
        del data, reference, matrix, features, tags
    selection = select_candidate(validation)
    selection["validation_records"] = validation
    selection["selected_model_sha256"] = None if selection['selected'] == 'baseline' else sha256(out / 'models' / f"{selection['selected']}.json")
    write_json(out / "selection-frozen-before-final.json", selection)
    print("Frozen selection: " + selection['selected'], flush=True)
    selected = selection['selected']
    final = []
    paired_uncertainty = {}
    for match in SPLITS["final"]:
        print(f"Reading UNTOUCHED FINAL {match}, after selection frozen", flush=True)
        data, reference, blocks, pitch, alignment, provenance = read_external(args.data_dir, match)
        write_json(out / f"{match}-inputs.json", provenance)
        write_json(out / f"{match}-alignment.json", alignment)
        for regime in REGIMES:
            features, tags, matrix = prepared(data, blocks, pitch, regime, match, config)
            labels = baseline_labels(features, tags, blocks, config)
            baseline = labels.copy()
            final.append(measure(out / 'final' / regime / 'baseline', match, 'baseline', regime, reference, labels, 1000))
            if selected != 'baseline':
                model = models[selected]
                labels, _, _ = decode_learned(features, tags, matrix, blocks, 25, config, model, mode=model['decoder_mode'])
                final.append(measure(out / 'final' / regime / selected, match, selected, regime, reference, labels, 1000))
                paired_uncertainty[regime] = paired_spell_bootstrap(reference, baseline, labels)
    base = {row['regime']: row for row in final if row['candidate'] == 'baseline'}
    learned = {row['regime']: row for row in final if row['candidate'] == selected}
    gate = promotion_gate(learned, base)
    promoted = selected != 'baseline' and gate['passed']
    if source_hashes() != frozen_hashes:
        raise ValueError("Scientific source changed during campaign; results cannot be released")
    summary = {"selected_on_validation": selected, "promoted": promoted, "final_gate": gate,
               "final_records": final, "validation_records": validation, "training_matches": training_meta,
               "source_hashes": frozen_hashes, "protocol_sha256": sha256(out / 'protocol.json'),
               "protocol": protocol, "paired_final_uncertainty": paired_uncertainty,
               "winner_frozen_before_final_sha256": sha256(out / 'selection-frozen-before-final.json'),
               "competition_public_labels_used": False, "final_set_retuning": False,
               "release_model": selected if promoted else 'baseline'}
    write_json(out / 'summary.json', summary)
    if promoted:
        (ROOT / 'models').mkdir(exist_ok=True)
        shutil.copyfile(out / 'models' / f'{selected}.json', ROOT / 'models/stoppages.json')
        summary['release_model_sha256'] = sha256(ROOT / 'models/stoppages.json')
        write_json(out / 'summary.json', summary)
    (ROOT / 'models').mkdir(exist_ok=True)
    write_json(ROOT / 'models/selection.json', summary)
    print(json.dumps({'selected':selected,'promoted':promoted,'final_gate':gate}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/external/idsse')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/stoppage-campaign-v1')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/default.json')
    parser.add_argument('--download', action='store_true', help='Download/check pinned public inputs before training')
    args=parser.parse_args()
    if args.download:
        download_inputs(args.data_dir)
    run(args)


if __name__=='__main__':
    main()
