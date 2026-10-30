"""Sealed v7 phase regression audit. Never validates or promotes stoppage changes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json, write_submission
from possession.config import load_config
from tools import fresh_benchmark as old

SC_COMMIT = old.SC_COMMIT
SC = {
    1953632: {
        'match.json': (27767, 'b63200aace34ada146ab615e776d9a022f1a72ad'),
        'phases_of_play.csv': (114478, 'bd5c07b7010bc00c73cedcef497aaaadf12a6455'),
        'tracking_extrapolated.jsonl': (90027965, '1c2014f97c3eff1ce2475203512c9d126930d7599c036105c9ac442d1de1667e')},
    1959846: {
        'match.json': (32946, '3513a7cd787f59e3f8027ba17365c17d78878969'),
        'phases_of_play.csv': (128571, '9ba4d7dff66a1b73ae9fbc39a899c678e308f5e9'),
        'tracking_extrapolated.jsonl': (87710167, '5ba2d87f6fd1ba3138301b369bd9266be69416a2d799a3fba5b5ec8a52e1284b')},
}
SPLIT = {
    'schema_version': 1, 'skillcorner_commit': SC_COMMIT,
    'final_silver_matches': list(SC), 'final_dense_matches': [],
    'already_exposed_skillcorner': [1874553, 1886347, 1899585, 1925299, 1927964],
    'adapter_development': 'Synthetic fixtures and previously exposed 1874553 only.',
    'rule': 'One frozen candidate and baseline; no training, runner-up, retuning or repeated audit. Moving files does not restore freshness.',
}
POLICY = {
    'silver_mean_accuracy_minimum_delta': 0.,
    'silver_mean_balanced_recall_minimum_delta': 0.,
    'silver_maximum_match_accuracy_drop': .01,
    'fresh_stoppage_validation_available': False,
    'fresh_short_possession_validation_available': False,
    'promotion_eligible': False,
    'rule': 'Phase regression gate is descriptive only. Without fresh explicit stoppage and short-possession truth the v7 candidate cannot replace the submission, even if this gate passes.',
}


def freeze_split(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in [('split-before-reference-access.json', SPLIT), ('policy-before-reference-access.json', POLICY)]:
        path = folder / name
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8-sig')) != value:
                raise ValueError('Fresh split or policy changed')
        else:
            write_json(path, value)
    return folder / 'split-before-reference-access.json'


def catalog(folder):
    rows = []
    for match, files in SC.items():
        for suffix, expected in files.items():
            name = f'{match}_{suffix}'
            host = 'media.githubusercontent.com/media' if suffix.endswith('jsonl') else 'raw.githubusercontent.com'
            rows.append((Path(folder) / name, expected, f'https://{host}/SkillCorner/opendata/{SC_COMMIT}/data/matches/{match}/{name}'))
    return rows


def download(folder):
    freeze_split(folder)
    manifest = []
    for path, expected, url in catalog(folder):
        if not path.exists():
            temporary = path.with_name(path.name + '.partial')
            received = 0
            with urlopen(url, timeout=120) as response, temporary.open('wb') as out:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > expected[0]:
                        raise ValueError('Download exceeded catalog size')
                    out.write(block)
            old.check_file(temporary, expected)
            temporary.replace(path)
        old.check_file(path, expected)
        manifest.append({'file': path.name, 'bytes': expected[0], 'sha256': sha256(path), 'source_identity': expected[1], 'url': url})
        print('Verified ' + path.name, flush=True)
    write_json(Path(folder) / 'download-manifest.json', {'files': manifest, 'reference_values_parsed': False})


def read_skillcorner(folder, match):
    """Existing validated whitelist reads detected XY; reference fields are ignored."""
    folder = Path(folder)
    for suffix in ('match.json', 'tracking_extrapolated.jsonl'):
        old.check_file(folder / f'{match}_{suffix}', SC[match][suffix])
    metadata = json.loads((folder / f'{match}_match.json').read_text(encoding='utf-8'))
    with (folder / f'{match}_tracking_extrapolated.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    return old._skillcorner_observations(metadata, rows)


def scientific_hashes():
    """Exact local inference/adapter/scoring closure; excludes release packaging."""
    names = ['tools/fresh_benchmark_v7.py', 'tools/fresh_benchmark.py', 'tools/independent_benchmark.py']
    paths = [*[ROOT / n for n in names], *sorted(ROOT.glob('possession/*.py'))]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def decide(scores):
    if set(scores) != {f'skillcorner{m}' for m in SC}:
        raise ValueError('Every frozen match is required')
    deltas = {}
    for name, pair in scores.items():
        if set(pair) != {'baseline', 'candidate'} or pair['baseline']['n'] != pair['candidate']['n']:
            raise ValueError('Paired reference support differs')
        for value in pair.values():
            if value['n'] <= 0 or value['dense_had_truth'] is not False or value['reference_quality'] != 'provider_silver':
                raise ValueError('Invalid reference scope')
            for metric in ('accuracy', 'balanced_team_recall'):
                v = value[metric]
                if isinstance(v, bool) or not np.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError('Invalid audit score')
        deltas[name] = {k: pair['candidate'][k] - pair['baseline'][k] for k in ('accuracy', 'balanced_team_recall')}
    accuracy = [d['accuracy'] for d in deltas.values()]
    balanced = [d['balanced_team_recall'] for d in deltas.values()]
    phase_passed = (np.mean(accuracy) >= POLICY['silver_mean_accuracy_minimum_delta']
                    and np.mean(balanced) >= POLICY['silver_mean_balanced_recall_minimum_delta']
                    and min(accuracy) >= -POLICY['silver_maximum_match_accuracy_drop'])
    return {'passed': False, 'phase_regression_passed': bool(phase_passed), 'promotion_eligible': False,
            'fresh_stoppage_validation_available': False, 'fresh_short_possession_validation_available': False,
            'candidate_minus_baseline': deltas, 'silver_mean_accuracy_delta': float(np.mean(accuracy)),
            'silver_mean_balanced_recall_delta': float(np.mean(balanced)), 'policy': POLICY}


def evaluate_pair(folder, candidate_model, baseline_model, config_path, out, baseline_config_path=None):
    from possession.learned import load_model
    from possession.pipeline import infer
    folder, out = Path(folder), Path(out)
    split = freeze_split(folder)
    marker = folder / 'final-reference-access.json'
    if marker.exists():
        raise ValueError('Fresh final matches already consumed')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty audit output directory')
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {'baseline': (Path(baseline_model), Path(baseline_config_path or config_path)),
                 'candidate': (Path(candidate_model), Path(config_path))}
    sources = scientific_hashes()
    data = {}
    for path, expected, _ in catalog(folder):
        old.check_file(path, expected)  # Integrity only: never parse labels here.
        data[path.name] = sha256(path)
    protocol = {'schema_version': 1, 'split': SPLIT, 'split_sha256': sha256(split), 'policy': POLICY,
                'policy_sha256': sha256(folder / 'policy-before-reference-access.json'), 'sources': sources, 'data': data,
                'artifacts': {k: {'model_sha256': sha256(m), 'config_sha256': sha256(c)} for k, (m, c) in artifacts.items()},
                'selection_frozen_before_reference_access': True, 'no_training_or_retuning': True,
                'reference_order': 'All four paired prediction CSVs sealed before any phase reference parsing. Access marker precedes loading JSON with embedded ignored possession fields.',
                'dense_human_had_validation': False, 'fresh_stoppage_validation_available': False}
    frozen = out / 'frozen-before-reference-access.json'
    write_json(frozen, protocol)
    frozen_hash = sha256(frozen)

    def unchanged():
        if sha256(frozen) != frozen_hash or sha256(split) != protocol['split_sha256'] or sha256(folder / 'policy-before-reference-access.json') != protocol['policy_sha256']:
            raise ValueError('Frozen protocol changed')
        if scientific_hashes() != sources:
            raise ValueError('Frozen source changed')
        if any(sha256(folder / n) != v for n, v in data.items()):
            raise ValueError('Frozen data changed')
        for k, (m, c) in artifacts.items():
            if sha256(m) != protocol['artifacts'][k]['model_sha256'] or sha256(c) != protocol['artifacts'][k]['config_sha256']:
                raise ValueError('Frozen artifacts changed')

    with marker.open('x', encoding='utf-8') as handle:
        json.dump({'protocol_sha256': frozen_hash, 'audit_directory': str(out.resolve()), 'status': 'access_started'}, handle, sort_keys=True)
    predictions, alignments, warnings = {}, {}, {}
    for match in SC:
        name = f'skillcorner{match}'
        print('Inferring frozen pair on ' + name, flush=True)
        unchanged()
        tracking, blocks, pitch, alignment = read_skillcorner(folder, match)
        alignments[name] = alignment
        predictions[name], warnings[name] = {}, {}
        for key, (model, config) in artifacts.items():
            unchanged()
            labels, *_, warn = infer(tracking, [], blocks, pitch, alignment['fps'], load_config(config), load_model(model))
            labels = np.asarray(labels).copy()
            if labels.shape != (len(alignment['periods']),) or not np.isin(labels, list('HADX')).all():
                raise ValueError('Invalid prediction timeline or label')
            predictions[name][key], warnings[name][key] = labels, warn
            write_submission(out / f'{name}-{key}.csv', labels)
    prediction_hashes = {p.name: sha256(p) for p in sorted(out.glob('*.csv'))}
    if len(prediction_hashes) != 4:
        raise ValueError('All four predictions must exist before references')
    write_json(out / 'predictions-before-reference-access.json', prediction_hashes)
    unchanged()
    scores, counts = {}, {}
    for match in SC:
        name = f'skillcorner{match}'
        reference, counts[name] = old.phase_references(folder / f'{match}_phases_of_play.csv', alignments[name])
        if np.any(reference == 'D'):
            raise ValueError('Phase gaps must never become dead-ball truth')
        scores[name] = {k: old.score_reference(reference, v, 'provider_silver') for k, v in predictions[name].items()}
    unchanged()
    if any(sha256(out / n) != v for n, v in prediction_hashes.items()):
        raise ValueError('Sealed predictions changed')
    result = {'protocol': protocol, 'scores': scores, 'gate': decide(scores), 'reference_counts': counts,
              'prediction_sha256': prediction_hashes, 'warnings': warnings,
              'reference_fields_used_for_inference': False, 'reference_read_count': len(SC),
              'dense_human_had_validation': False, 'fresh_stoppage_validation_available': False}
    write_json(out / 'summary.json', result)
    print(json.dumps(result['gate'], indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('download', 'evaluate-pair'))
    p.add_argument('--data-dir', type=Path, default=ROOT / 'runs/fresh-data-v7')
    p.add_argument('--model', type=Path)
    p.add_argument('--baseline-model', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    p.add_argument('--baseline-config', type=Path)
    p.add_argument('--out', type=Path, default=ROOT / 'runs/fresh-audit-v7')
    args = p.parse_args()
    if args.command == 'download':
        download(args.data_dir)
    else:
        if args.model is None or args.baseline_model is None:
            p.error('Both frozen model paths are required')
        evaluate_pair(args.data_dir, args.model, args.baseline_model, args.config, args.out, args.baseline_config)
