"""Resumable Sportec-only control/joint rebuild; never a submission command.

Every run call consumes one bounded stage. A failed stage remains consumed; use a
new run directory rather than silently retrying a partially observed experiment.
Only source code, pinned external assets and fresh synthetic augmentations enter.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import robust_control
from possession.config import load_config
from possession.features import extract
from possession.io import sha256, write_json
from possession.learned import (feature_matrix, load_model, predict_dead,
                                fuse_control_probabilities, joint_probabilities,
                                validate_model)
from tools.camera_augmentation import corrupt_camera_tracking, CAMERA_AUGMENTATION_SPEC
from tools.control_diagnostics import training_spell_weights
from tools.external_benchmark import digest
from tools.train_control import CATALOG, SEEDS, read_external, score_job
from tools.train_stoppages import corrupt_tracking
from tools import train_joint
from tools.improvement_policy import audit_lineage, authenticate_models

BASE_SHA = '115c192cd4aa71be837e1c4bd2137d91add6dcea4b006a809b13232e62f43c0b'
CONFIG_SHA = '2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b'
REGIMES = ('clean', 'degraded', 'camera')
FOLDS = (('J03WN1', 'J03WMX'), ('J03WPY', 'J03WQQ'), ('J03WR9',))
DEFAULT_RECIPE = dict(schema_version=1, seed=10527, epochs=8,
                      control_l2=.001, transition_weight=1.5)
NAMES = ('external-control', 'external-joint')


def recipe(value=None):
    value = DEFAULT_RECIPE.copy() if value is None else copy.deepcopy(value)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_RECIPE):
        raise ValueError('Recipe requires exactly the declared external-only parameters')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Unsupported recipe schema')
    if type(value['seed']) is not int or not 0 <= value['seed'] < 2**31:
        raise ValueError('Seed must be an integer in [0, 2**31)')
    if type(value['epochs']) is not int or value['epochs'] not in (4, 8, 12):
        raise ValueError('Epochs must be 4, 8 or 12')
    for key, choices in [('control_l2', (.001, .01)), ('transition_weight', (1., 1.5, 2.))]:
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or value[key] not in choices:
            raise ValueError(f'Unsupported {key}')
    return value


def check_match(match):
    if not isinstance(match, str) or match not in CATALOG:
        raise ValueError('Only the seven registered external Sportec fixtures are permitted')
    return match


def split(index):
    if type(index) is not int or not 0 <= index < len(FOLDS):
        raise ValueError('Invalid external development fold')
    held = list(FOLDS[index])
    train = sorted(set(CATALOG) - set(held))
    if set(held) & {'J03WOH', 'J03WOY'}:
        raise ValueError('Frozen dead-head training fixtures cannot be scored')
    return train, held


def no_links(path):
    """Reject symlinks/junctions, including existing ancestors of new outputs."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Linked filesystem paths are not accepted')
        if part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
            raise ValueError('Reparse-point filesystem paths are not accepted')


def paths(data_dir, out):
    no_links(data_dir)
    no_links(out)
    data_dir, out = Path(data_dir).resolve(), Path(out).resolve()
    permitted = {(ROOT / 'data/external/idsse').resolve(),
                 (ROOT.parent / 'more-than-a-hack-2027/data/external/idsse').resolve()}
    if data_dir not in permitted:
        raise ValueError('Data directory must be the registered external Sportec location')
    root = (ROOT / 'runs').resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('Output must be a dedicated local runs subdirectory')
    return data_dir, out


def source_hashes():
    names = ('tools/external_rebuild.py', 'tools/camera_augmentation.py',
             'tools/control_diagnostics.py', 'tools/external_benchmark.py',
             'tools/train_control.py', 'tools/train_stoppages.py',
             'tools/train_joint.py', 'tools/improvement_policy.py',
             'tools/pff_precision_validation.py', 'tools/pff_validation.py',
             'evaluation/possession_metrics.py')
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for p in [*sorted((ROOT / 'possession').glob('*.py')),
                      *[ROOT / name for name in names]]}


def base():
    model_path = ROOT / 'models/baselines/v1.2.json'
    config_path = ROOT / 'configs/baselines/v1.2.json'
    if sha256(model_path) != BASE_SHA or sha256(config_path) != CONFIG_SHA:
        raise ValueError('Pinned pre-calibration model/configuration differs')
    model = load_model(model_path)
    # Reuse the external live/dead head only. Do not inherit an old control head.
    model = {key: copy.deepcopy(model[key]) for key in (
        'schema_version', 'kind', 'feature_names', 'mean', 'scale',
        'coefficients', 'intercept')}
    model.update(decoder_mode='joint', metadata={
        'dead_parent_sha256': BASE_SHA, 'dead_training_matches': ['J03WOH', 'J03WOY'],
        'source': 'Sportec/IDSSE', 'external_only': True})
    return validate_model(model), load_config(config_path)


def plan(parameters=None):
    base()  # authenticate before accepting even a dry-run plan
    return dict(schema_version=1, recipe=recipe(parameters),
                matches=sorted(CATALOG), folds=[list(x) for x in FOLDS],
                regimes=list(REGIMES), camera_spec=CAMERA_AUGMENTATION_SPEC,
                source_hashes=source_hashes(), dead_parent_sha256=BASE_SHA,
                config_sha256=CONFIG_SHA, external_only=True, final_accessed=False,
                promoted=False, submission_eligible=False,
                stages=[*[f'prepare-{m}' for m in sorted(CATALOG)],
                        *[f'fold-{i}' for i in range(len(FOLDS))], 'refit'],
                limitations=['Previously exposed development fixtures only',
                             'No fresh-test or hidden-score claim',
                             'Historical dead head and configuration require separate lineage review',
                             'No challenge data, target camera profile, or PFF data is accessed'])


def preflight(data_dir, out, parameters=None):
    data_dir, _ = paths(data_dir, out)
    result = plan(parameters)
    inputs = {}
    for match in sorted(CATALOG):
        for kind in ('metadata', 'tracking'):
            _, size, expected = CATALOG[match][kind]
            p = data_dir / f'{match}_{kind}.xml'
            no_links(p)
            if not p.is_file() or p.stat().st_size != size or digest(p) != expected:
                raise ValueError(f'External input missing or mismatched: {p.name}')
            inputs[p.name] = {'sha256': sha256(p), 'bytes': size, 'published_md5': expected}
    result['inputs'] = inputs
    return result


def _atomic(path, value):
    temporary = Path(path).with_suffix('.pending')
    write_json(temporary, value)
    temporary.replace(path)


def _seal(folder):
    files = {p.relative_to(folder).as_posix(): sha256(p)
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'complete.json'}
    _atomic(folder / 'complete.json', files)
    return sha256(folder / 'complete.json')


def _verify_seal(folder):
    no_links(folder)
    for path in folder.rglob('*'):
        no_links(path)
    manifest = json.loads((folder / 'complete.json').read_text())
    actual = {p.relative_to(folder).as_posix(): sha256(p)
              for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'complete.json'}
    if manifest != actual:
        raise ValueError('Completed stage artifact bytes changed')


def prepare(match, data_dir, folder, config):
    check_match(match)
    no_links(folder)
    for kind in ('metadata', 'tracking'):
        no_links(data_dir / f'{match}_{kind}.xml')
    # The registered reader authenticates file identities before parsing.
    tracking, y, blocks, pitch, alignment, inputs = read_external(data_dir, match)
    folder.mkdir(parents=True)
    np.save(folder / 'reference.npy', y)
    write_json(folder / 'metadata.json', dict(blocks=blocks, pitch=pitch,
               alignment=alignment, inputs=inputs, match=match))
    for regime in REGIMES:
        observed = (tracking if regime == 'clean' else
                    corrupt_tracking(tracking, blocks, pitch, 25, SEEDS[match]) if regime == 'degraded' else
                    corrupt_camera_tracking(tracking, blocks, pitch, 25, SEEDS[match] + 91001))
        features = extract(observed, pitch, blocks, 25, config)
        x = robust_control.feature_matrix(observed, features, blocks, pitch, 25)
        z = feature_matrix(observed, features, blocks, pitch, 25)
        np.save(folder / f'{regime}-x.npy', x)
        np.save(folder / f'{regime}-z.npy', z)
        write_json(folder / f'{regime}-support.json',
                   {'usable': [bool(features.reliability[a:b+1].sum()) for _, a, b in blocks]})
        del observed, features, x, z
    _seal(folder)


def cached(out, match, regime):
    check_match(match)
    if regime not in REGIMES:
        raise ValueError('Only fixed synthetic external regimes are accepted')
    folder = out / 'cache' / match
    meta = json.loads((folder / 'metadata.json').read_text())
    return (np.load(folder / f'{regime}-x.npy', mmap_mode='r', allow_pickle=False),
            np.load(folder / f'{regime}-z.npy', mmap_mode='r', allow_pickle=False),
            np.load(folder / 'reference.npy', allow_pickle=False), meta['blocks'])


def fit_control(out, matches, parameters, metadata):
    xs, ys, ws = [], [], []
    for match in sorted(matches):
        check_match(match)
        for regime in REGIMES:
            x, _, y, _ = cached(out, match, regime)
            xs.append(np.array(x[::10])); ys.append(y[::10])
            ws.append(training_spell_weights(y, 25)[::10])
    return robust_control.fit_model(np.concatenate(xs), np.concatenate(ys),
        sample_weight=np.concatenate(ws), l2=parameters['control_l2'],
        metadata=dict(metadata, training_matches=sorted(matches), training_regimes=list(REGIMES)))


def _logit(p):
    p = np.clip(p, 1e-8, 1-1e-8)
    return np.log(p) - np.log1p(-p)


def joint_weights(y, blocks, strength):
    weights = training_spell_weights(y, 25)
    weights[y == 'D'] = 1.
    active = np.zeros(len(y), dtype=bool)
    for _, a, b in blocks:
        active[a:b+1] = True
        for i in range(a+1, b+1):
            if y[i] != 'X' and y[i-1] != 'X' and y[i] != y[i-1]:
                lo, hi = max(a, i-10), min(b+1, i+11)
                weights[lo:hi] = np.maximum(weights[lo:hi], strength)
    weights[(y == 'X') | ~active] = 0
    return weights


def fit_joint(out, matches, model, parameters, metadata):
    sequences = []
    for match in sorted(matches):
        check_match(match)
        for regime in REGIMES:
            x, z, y, blocks = cached(out, match, regime)
            home = robust_control.predict_home(model['robust_control_model'], x)
            dead = predict_dead(z, model)
            weights = joint_weights(y, blocks, parameters['transition_weight'])
            for _, a, b in blocks:
                ix = np.arange(a, b+1, 5)
                sequences.append(dict(X=np.array(x[ix]), Z=np.array(z[ix]),
                    y=y[ix], w=weights[ix], base_home=_logit(home[ix]), base_dead=_logit(dead[ix])))
    return train_joint.fit(sequences, seed=parameters['seed'], epochs=parameters['epochs'],
        metadata=dict(metadata, training_matches=sorted(matches), training_regimes=list(REGIMES)))


def run_fold(out, index, parameters, protocol_hash):
    train, held = split(index)
    model, config = base()
    folder = out / 'folds' / str(index)
    folder.mkdir(parents=True)
    metadata = dict(protocol_sha256=protocol_hash, fold=index, held_out_matches=held,
                    external_only=True, challenge_inputs_used=False)
    model['robust_control_model'] = fit_control(out, train, parameters, metadata)
    write_json(folder / 'control.json', model)
    model['joint_model'] = fit_joint(out, train, model, parameters, metadata)
    write_json(folder / 'joint.json', model)
    rows = []
    for match in held:
        for regime in REGIMES:
            x, z, y, blocks = cached(out, match, regime)
            dead = predict_dead(z, model)
            home = robust_control.predict_home(model['robust_control_model'], x)
            p = fuse_control_probabilities(dead, home, np.zeros(len(y)))
            q = joint_probabilities(dead, home, x, z, blocks, 25, model['joint_model'], np.zeros(len(y)))
            usable = json.loads((out / 'cache' / match / f'{regime}-support.json').read_text())['usable']
            for name, probability in zip(NAMES, (p, q)):
                row, _ = score_job(folder / 'scores' / match / regime / name,
                    match, name, regime, y, probability, blocks, config, usable, 0)
                row['fold'] = index
                rows.append(row)
    write_json(folder / 'records.json', rows)
    _seal(folder)


def select(records):
    cells = {(m, r) for fold in FOLDS for m in fold for r in REGIMES}
    grouped = {n: {} for n in NAMES}
    for row in records:
        name, cell = row['candidate'], (row['match'], row['regime'])
        if name not in grouped or cell not in cells or cell in grouped[name]:
            raise ValueError('Invalid or duplicate external development record')
        values = [row[k] for k in ('spellF1', 'macroF1', 'dead_precision', 'dead_recall')]
        values.extend([row['control_diagnostics']['transitions']['recall'],
                       row['control_diagnostics']['spell_recall_by_duration']['<2s']['recall']])
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Invalid metric')
        grouped[name][cell] = row
    if any(set(rows) != cells for rows in grouped.values()):
        raise ValueError('Every fixed external development cell is required')
    left, right = [grouped[n] for n in NAMES]
    delta = lambda k: [right[c][k] - left[c][k] for c in sorted(cells)]
    spell = delta('spellF1')
    transition = np.mean([right[c]['control_diagnostics']['transitions']['recall'] - left[c]['control_diagnostics']['transitions']['recall'] for c in cells])
    short = np.mean([right[c]['control_diagnostics']['spell_recall_by_duration']['<2s']['recall'] - left[c]['control_diagnostics']['spell_recall_by_duration']['<2s']['recall'] for c in cells])
    checks = dict(spell_gain=np.mean(spell) >= .01, no_spell_cell_regression=min(spell) >= -.02,
                  no_dead_recall_cell_regression=min(delta('dead_recall')) >= -.01,
                  dead_precision=np.mean(delta('dead_precision')) >= 0,
                  macro=np.mean(delta('macroF1')) >= 0, transition=transition >= 0, short_recall=short >= 0)
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(selected=NAMES[1] if all(checks.values()) else NAMES[0],
                development_gate_passed=all(checks.values()), checks=checks,
                comparison='New joint head against newly fitted external control; not v1.5 promotion',
                means={n: {k: float(np.mean([r[k] for r in rows.values()]))
                           for k in ('spellF1', 'macroF1', 'dead_precision', 'dead_recall')}
                       for n, rows in grouped.items()})


def lineage(frozen, candidate_hash, candidate_id):
    artifacts = {}
    def node(identity, kind, scope, identifier, parents=()):
        artifacts[identity] = dict(sha256=identity, kind=kind, scope=scope,
                                   identifier=identifier, parents=list(parents))
    node(BASE_SHA, 'model', 'external', 'Pinned v1.2 external Sportec dead head')
    node(CONFIG_SHA, 'configuration', 'synthetic', 'Pre-camera-fit fixed v1.2 configuration')
    for name, identity in frozen['source_hashes'].items():
        node(identity, 'code', 'synthetic', name)
    for name, evidence in frozen['inputs'].items():
        if name not in {f'{m}_{kind}.xml' for m in CATALOG for kind in ('metadata', 'tracking')}:
            raise ValueError('Lineage contains an unregistered external file')
        node(evidence['sha256'], 'dataset', 'external', 'sportec-idsse-28196177')
    parents = sorted(artifacts)
    node(candidate_hash, 'model', 'external', candidate_id, parents)
    result = dict(schema_version=1, candidate_sha256=candidate_hash,
        comparator_sha256='3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0',
        fit_inputs=[BASE_SHA, CONFIG_SHA, *sorted(v['sha256'] for v in frozen['inputs'].values())],
        artifacts=list(artifacts.values()))
    audit_lineage(result)
    return result


def refit(out, parameters, protocol_hash):
    records = [r for i in range(len(FOLDS))
               for r in json.loads((out / 'folds' / str(i) / 'records.json').read_text())]
    selection = select(records)
    # Freeze grouped-CV selection before the all-development refit.
    write_json(out / 'selection-before-refit.json', selection)
    model, config = base()
    meta = dict(protocol_sha256=protocol_hash, stage='all-external-development-refit',
                external_only=True, challenge_inputs_used=False)
    model['robust_control_model'] = fit_control(out, sorted(CATALOG), parameters, meta)
    if selection['selected'] == NAMES[1]:
        model['joint_model'] = fit_joint(out, sorted(CATALOG), model, parameters, meta)
    model['candidate_id'] = selection['selected']
    validate_model(model)
    folder = out / 'refit'
    write_json(folder / 'candidate.json', model)
    write_json(folder / 'config.json', config.as_dict())
    write_json(folder / 'results.json', dict(selection=selection, records=records))
    frozen = json.loads((out / 'protocol.json').read_text())
    candidate_hash = sha256(folder / 'candidate.json')
    manifest = lineage(frozen, candidate_hash, selection['selected'])
    write_json(folder / 'lineage.json', manifest)
    model_paths = {BASE_SHA: 'models/baselines/v1.2.json',
                   candidate_hash: (folder / 'candidate.json').relative_to(ROOT).as_posix()}
    authenticate_models(audit_lineage(manifest), model_paths, ROOT)
    write_json(folder / 'model-bindings.json', model_paths)
    _seal(folder)
    return selection


def _authenticate(out, parameters):
    frozen = json.loads((out / 'protocol.json').read_text())
    expected = plan(parameters)
    if {k: frozen[k] for k in expected} != expected:
        raise ValueError('Frozen sources, parameters or lineage changed; start a new study')
    state = json.loads((out / 'state.json').read_text())
    if state['protocol_sha256'] != sha256(out / 'protocol.json'):
        raise ValueError('Protocol identity differs')
    if (not isinstance(state.get('completed'), dict) or
            set(state['completed']) != set(frozen['stages'][:len(state['completed'])])):
        raise ValueError('Stages must complete in the frozen order')
    for stage in state['completed']:
        folder = (out / 'cache' / stage.removeprefix('prepare-') if stage.startswith('prepare-') else
                  out / 'folds' / stage.removeprefix('fold-') if stage.startswith('fold-') else out / 'refit')
        if sha256(folder / 'complete.json') != state['completed'][stage]:
            raise ValueError('Completed stage identity differs')
        _verify_seal(folder)
    return frozen, state


def run(data_dir, out, parameters=None):
    parameters = recipe(parameters)
    data_dir, out = paths(data_dir, out)
    if not out.exists():
        protocol = preflight(data_dir, out, parameters)
        out.mkdir(parents=True)
        write_json(out / 'protocol.json', protocol)
        write_json(out / 'state.json', dict(protocol_sha256=sha256(out / 'protocol.json'), completed={}))
    frozen, state = _authenticate(out, parameters)
    pending = [s for s in frozen['stages'] if s not in state['completed']]
    if not pending:
        return json.loads((out / 'summary.json').read_text())
    stage = pending[0]
    access = out / f'{stage}-started.json'
    # Exclusive access marker survives errors, preventing silent retries.
    with access.open('x', encoding='utf-8') as f:
        json.dump(dict(stage=stage, protocol_sha256=state['protocol_sha256'], pid=os.getpid()), f)
    print(f'External-only stage: {stage}', flush=True)
    if stage.startswith('prepare-'):
        match = stage.removeprefix('prepare-')
        folder = out / 'cache' / match
        prepare(match, data_dir, folder, base()[1])
    elif stage.startswith('fold-'):
        index = int(stage.removeprefix('fold-'))
        folder = out / 'folds' / str(index)
        run_fold(out, index, parameters, state['protocol_sha256'])
    else:
        folder = out / 'refit'
        refit(out, parameters, state['protocol_sha256'])
    if source_hashes() != frozen['source_hashes']:
        raise ValueError('Scientific sources changed during execution')
    state['completed'][stage] = sha256(folder / 'complete.json')
    summary = dict(schema_version=1, protocol_sha256=state['protocol_sha256'],
                   stage_completed=stage, status='complete' if len(pending) == 1 else 'in_progress',
                   next_stage=pending[1] if len(pending) > 1 else None,
                   external_only=True, final_accessed=False, promoted=False, submission_eligible=False)
    if len(pending) == 1:
        results = json.loads((out / 'refit/results.json').read_text())
        summary.update(results['selection'], records=results['records'],
                       release_candidate_model_sha256=sha256(out / 'refit/candidate.json'),
                       model_paths=json.loads((out / 'refit/model-bindings.json').read_text()),
                       lineage_sha256=sha256(out / 'refit/lineage.json'))
    _atomic(out / 'summary.json', summary)
    _atomic(out / 'state.json', state)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'preflight', 'run'))
    local = ROOT / 'data/external/idsse'
    p.add_argument('--data-dir', type=Path, default=local if local.is_dir() else ROOT.parent / 'more-than-a-hack-2027/data/external/idsse')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--stage', choices=('next',), default='next')
    args = p.parse_args()
    parameters = recipe(json.loads(args.recipe.read_text()) if args.recipe else None)
    paths(args.data_dir, args.out)
    result = plan(parameters) if args.action == 'plan' else preflight(args.data_dir, args.out, parameters) if args.action == 'preflight' else run(args.data_dir, args.out, parameters)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
