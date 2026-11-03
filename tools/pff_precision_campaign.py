"""Bounded PFF development calibration with a locked, one-shot final audit."""
from pathlib import Path
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from possession.learned import load_model
from tools import pff_campaign as prior, pff_validation as original
from tools import pff_precision_model as correction, pff_precision_validation as validation
from tools.train_temporal_v6 import CONFIG_SHA256

PARENT = ROOT / 'runs/pff-campaign-v3'
DEFAULT = ROOT / 'runs/pff-precision-v1'
NEW_SOURCES = ('tools/pff_precision_model.py', 'tools/pff_precision_validation.py',
               'tools/pff_precision_campaign.py', 'tools/pff_false_stoppage_diagnostics.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sources():
    return dict(prior.sources(), **{n: sha256(ROOT / n) for n in NEW_SOURCES},
                **{'configs/baselines/v1.5.json': sha256(ROOT / 'configs/baselines/v1.5.json')})


def context():
    reservation = original.verify_protocol(read(ROOT / 'models/pff-protocol.json'))
    old = read(ROOT / 'models/pff-research.json')
    from tools.verify_pff_research import verify_bundle
    verify_bundle(old)
    if (sha256(PARENT / 'candidate.json') != validation.PARENT_SHA256
        or sha256(ROOT / 'configs/baselines/v1.5.json') != CONFIG_SHA256
        or read(PARENT / 'development-summary.json') != old['development']['summary']):
        raise ValueError('Frozen parent artifacts differ')
    return reservation, old['development']['summary']['records']


def initialize(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Choose a fresh study directory')
    reservation, old = context()
    out.mkdir(parents=True)
    declarations = []
    for number, penalty in enumerate(validation.PENALTIES):
        ident = f'penalty-{number:02d}'
        path = out / 'corrections' / (ident + '.json')
        artifact = {'schema_version': 1, 'parent_sha256': validation.PARENT_SHA256,
                    'parameters': {'dead_logit_penalty': penalty}, 'policy': correction.POLICY}
        write_json(path, artifact)
        declarations.append({'candidate_id': ident, 'artifact_sha256': sha256(path),
                             'parameters': artifact['parameters']})
    study = validation.build_protocol(reservation, old, declarations, sources())
    validation.freeze_protocol(out / 'protocol.json', study, reservation, old)
    return study


def authenticate(out):
    reservation, old = context()
    study = validation.verify_protocol(read(out / 'protocol.json'), reservation, old)
    if study['source_hashes'] != sources():
        raise ValueError('Precision scientific sources changed')
    for declared in study['candidates']:
        path = out / 'corrections' / (declared['candidate_id'] + '.json')
        artifact = read(path)
        if (sha256(path) != declared['artifact_sha256']
            or artifact['parent_sha256'] != validation.PARENT_SHA256
            or artifact['parameters'] != declared['parameters'] or artifact['policy'] != correction.POLICY):
            raise ValueError('Correction artifact changed')
    return study, reservation, old


def require_final(out):
    study, reservation, old = authenticate(out)
    summary = read(out / 'development-summary.json')
    if summary['protocol_sha256'] != study['protocol_sha256'] or summary['source_hashes'] != sources():
        raise ValueError('Development provenance differs')
    selection = validation.select_candidate(study, summary['records'], old, reservation)
    if summary['selection'] != selection or not selection['eligible_for_final_audit']:
        raise ValueError('Development failed; final matches stay locked')
    freeze_path = out / 'development-predictions-before-references.json'
    access_path = out / 'development-reference-access.json'
    frozen, marker = read(freeze_path), read(access_path)
    if (sha256(freeze_path) != summary['prediction_freeze_sha256']
        or sha256(access_path) != summary['reference_access_sha256']
        or marker['prediction_freeze_sha256'] != summary['prediction_freeze_sha256']
        or marker['protocol_sha256'] != study['protocol_sha256']
        or marker['source_hashes'] != sources() or marker['part'] != 'development'):
        raise ValueError('Development prediction/reference provenance changed')
    verify_predictions(out, 'development', frozen, study, reservation)
    if summary['parent_records'] != old:
        raise ValueError('Development parent records changed')
    details = {(r['match'], r['candidate']): r for r in summary['details']}
    expected = {(str(mid), name) for mid in reservation['split']['development']
                for name in ('baseline', 'parent', *(d['candidate_id'] for d in study['candidates']))}
    if set(details) != expected or len(details) != len(summary['details']):
        raise ValueError('Development diagnostic cohort differs')
    for ident, records in summary['records'].items():
        for record in records:
            for name, field in (('baseline', 'baseline'), (ident, 'candidate')):
                values = prior.metrics(details[str(record['match_id']), name])
                values['short_recall'] = values.pop('short')
                if values != record[field]:
                    raise ValueError('Development metrics differ from diagnostics')
    return selection


def begin_access(path, marker):
    # A failed partial pass remains consumed; never silently rerun scoring.
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)


def observation_root(out, part):
    return PARENT if part == 'development' else out / 'final-observations'


def prepare_final(out, workers):
    require_final(out)
    _, reservation, _ = authenticate(out)
    cache = observation_root(out, 'final')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prior.prepare_one, cache, mid) for mid in reservation['split']['final']]
        for future in as_completed(futures):
            print(json.dumps({'prepared_final': future.result()['match_id']}), flush=True)


def predict_one(out, part, mid, declarations):
    out = Path(out)
    study, reservation, _ = authenticate(out)
    if part not in ('development', 'final') or mid not in reservation['split'][part]:
        raise ValueError('Prediction fixture outside declared cohort')
    expected = study['candidates']
    if part == 'final':
        winner = require_final(out)['selected_candidate_id']
        expected = [d for d in expected if d['candidate_id'] == winner]
    if declarations != expected:
        raise ValueError('Prediction candidates differ from declared selection')
    item = prior.item_at(observation_root(out, part), mid)
    model = load_model(PARENT / 'candidate.json')
    probabilities = correction.probabilities_from_cache(item, model)
    folder = out / part / str(mid)
    folder.mkdir(parents=True, exist_ok=True)
    parent, _ = correction.decode_probabilities(probabilities, item, model)
    if part == 'development':
        previous = np.load(PARENT / 'development' / str(mid) / 'candidate.npy', allow_pickle=False)
        if not np.array_equal(parent, previous):
            raise ValueError('Zero-penalty predictions differ from the frozen parent')
    np.save(folder / 'parent.npy', parent, allow_pickle=False)
    baseline = np.load(observation_root(out, part) / 'cache' / str(mid) / 'baseline.npy', allow_pickle=False)
    np.save(folder / 'baseline.npy', baseline, allow_pickle=False)
    files = {'parent': sha256(folder / 'parent.npy'), 'baseline': sha256(folder / 'baseline.npy')}
    for declared in declarations:
        ident = declared['candidate_id']; penalty = declared['parameters']['dead_logit_penalty']
        labels = parent if penalty == 0 else correction.decode_probabilities(probabilities, item, model, penalty)[0]
        np.save(folder / (ident + '.npy'), labels, allow_pickle=False)
        files[ident] = sha256(folder / (ident + '.npy'))
    return {'match_id': mid, 'prediction_sha256': files,
            'observation_manifest': read(observation_root(out, part) / 'cache' / str(mid) / 'complete.json')}


def predict(out, part, workers):
    if part not in ('development', 'final'):
        raise ValueError('Only declared evaluation cohorts are allowed')
    study, reservation, _ = authenticate(out)
    if (out / (part + '-reference-access.json')).exists():
        raise ValueError('References already consumed; predictions cannot be regenerated')
    path = out / (part + '-predictions-before-references.json')
    if path.exists():
        raise ValueError('Cohort predictions already frozen')
    declarations = study['candidates']
    if part == 'final':
        selection = require_final(out)
        declarations = [d for d in declarations if d['candidate_id'] == selection['selected_candidate_id']]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(predict_one, out, part, mid, declarations) for mid in reservation['split'][part]]
        for future in as_completed(futures):
            result = future.result(); records.append(result)
            print(json.dumps({'predicted': result['match_id'], 'part': part}), flush=True)
    records.sort(key=lambda row: row['match_id'])
    freeze = {'part': part, 'records': records, 'candidates': declarations,
              'protocol_sha256': study['protocol_sha256'], 'source_hashes': sources(),
              'references_accessed_in_this_study': False,
              'development_previously_consumed': part == 'development'}
    authenticate(out)
    write_json(path, freeze)
    return freeze


def verify_predictions(out, part, frozen, study, reservation):
    expected = study['candidates']
    if part == 'final':
        selected = require_final(out)['selected_candidate_id']
        expected = [d for d in expected if d['candidate_id'] == selected]
    if (frozen['part'] != part or frozen['candidates'] != expected
        or frozen['protocol_sha256'] != study['protocol_sha256'] or frozen['source_hashes'] != sources()
        or frozen['references_accessed_in_this_study'] is not False
        or [r['match_id'] for r in frozen['records']] != sorted(reservation['split'][part])):
        raise ValueError('Prediction freeze or cohort changed')
    names = {'baseline', 'parent', *(d['candidate_id'] for d in expected)}
    for row in frozen['records']:
        if set(row['prediction_sha256']) != names:
            raise ValueError('Prediction model coverage differs')
        mid = row['match_id']
        if row['observation_manifest'] != read(observation_root(out, part) / 'cache' / str(mid) / 'complete.json'):
            raise ValueError('Observation provenance changed')
        for name, expected_hash in row['prediction_sha256'].items():
            if sha256(out / part / str(mid) / (name + '.npy')) != expected_hash:
                raise ValueError('Frozen prediction bytes changed')


def score(out, part):
    if part not in ('development', 'final'):
        raise ValueError('Only declared evaluation cohorts are allowed')
    study, reservation, old = authenticate(out)
    if (out / (part + '-summary.json')).exists() or (out / (part + '-reference-access.json')).exists():
        raise ValueError('No repeated or partial held-out scoring')
    if part == 'final':
        require_final(out)
    path = out / (part + '-predictions-before-references.json')
    frozen = read(path)
    verify_predictions(out, part, frozen, study, reservation)
    marker = {'part': part, 'prediction_freeze_sha256': sha256(path),
              'protocol_sha256': study['protocol_sha256'], 'source_hashes': sources(),
              'development_previously_consumed': part == 'development'}
    begin_access(out / (part + '-reference-access.json'), marker)
    records = {d['candidate_id']: [] for d in frozen['candidates']}
    details, parent_records, supports = [], [], {}
    for saved in frozen['records']:
        mid = saved['match_id']; folder = out / part / str(mid)
        item = prior.item_at(observation_root(out, part), mid)
        y, supports[str(mid)] = prior.reference(observation_root(out, part), mid)
        values = {}
        for name in saved['prediction_sha256']:
            labels = np.load(folder / (name + '.npy'), allow_pickle=False)
            row, values[name] = prior.score_one(out, part, mid, name, y, labels, item['blocks'])
            details.append(row)
        diagnostics = row['temporal_diagnostics']
        shared = {'match_id': mid, 'condition': 'native', 'baseline_sha256': original.BASELINE_SHA256,
                  'baseline': values['baseline'], 'reference_dead_segments': diagnostics['dead_segments']['n_gt'],
                  'reference_short_spells': diagnostics['short_possessions']['n_gt']}
        parent_records.append(dict(shared, candidate=values['parent'], candidate_sha256=validation.PARENT_SHA256))
        for declared in frozen['candidates']:
            ident = declared['candidate_id']
            records[ident].append(dict(shared, candidate=values[ident], candidate_sha256=declared['artifact_sha256']))
    evidence = {'dense_truth_validated': False, 'independent_observations': True, 'identity_checked': True,
                'usage_permitted': False, 'development_passed': part == 'final', 'observations_only': True,
                'predictions_frozen_before_references': True, 'adequate_label_support': False}
    result = {'part': part, 'records': records, 'parent_records': parent_records, 'details': details,
              'reference_support': supports, 'source_hashes': sources(), 'protocol_sha256': study['protocol_sha256'],
              'prediction_freeze_sha256': sha256(path),
              'reference_access_sha256': sha256(out / (part + '-reference-access.json')),
              'evidence': evidence, 'promoted': False, 'retained_model_sha256': original.BASELINE_SHA256}
    if part == 'development':
        if original.digest(parent_records) != original.digest(old):
            raise ValueError('Rescored development parent differs; preserve consumed attempt')
        result['selection'] = validation.select_candidate(study, records, old, reservation)
        print(json.dumps(result['selection']), flush=True)
    else:
        selection = require_final(out)
        result['gate'] = validation.final_gate(records[selection['selected_candidate_id']], parent_records,
                                               reservation, evidence, selection)
        print(json.dumps(result['gate']), flush=True)
    authenticate(out)
    write_json(out / (part + '-summary.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['initialize', 'predict', 'score', 'prepare-final'])
    parser.add_argument('--split', choices=['development', 'final'], default='development')
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(); out = args.out.resolve()
    if args.workers not in (1, 2):
        raise ValueError('Use one or two workers to bound memory')
    if args.stage == 'initialize': initialize(out)
    elif args.stage == 'prepare-final': prepare_final(out, args.workers)
    elif args.stage == 'predict': predict(out, args.split, args.workers)
    else: score(out, args.split)


if __name__ == '__main__':
    main()
