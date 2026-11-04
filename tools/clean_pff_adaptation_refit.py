"""One verified inner winner, refitted once on original40 plus clean Sportec7.

No CLI, development/final access or promotion. The outer operator must bind this
module and enforce subprocess time/resource limits. All fitted artifacts local.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools import clean_pff_adaptation_fit as kernel
from tools import clean_pff_adaptation_inputs as inputs
from tools import clean_pff_adaptation_protocol as protocol
from tools import clean_pff_adaptation_stages as stages
from tools import experiment_harness as storage

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = 'clean-pff-adaptation-refit-v1'
FLAGS = dict(development_accessed=False, final_accessed=False,
             final_access_authorized=False, promoted=False, submission_eligible=False,
             artifact_delivery_rights_verified=False, fitted_weights_must_remain_local=True)


def _require_fixture(match):
    if type(match) is not str or match not in protocol.fit_ids('refit'):
        raise ValueError('Only original40 plus clean Sportec7 may enter winner refit')
    return match


def _verified_selection(inner_out):
    stages.location(inner_out, root=ROOT)
    from tools import clean_pff_adaptation_campaign as campaign
    proof = campaign.verify_completed(inner_out)
    selection = proof.get('selection', {})
    if (proof.get('artifacts_verified') is not True or proof.get('lifecycle_complete') is not True or
            proof.get('protocol_sha256') != stages.SNAPSHOT_PROTOCOL_SHA256 or
            selection.get('protocol_sha256') != stages.SNAPSHOT_PROTOCOL_SHA256 or
            proof.get('selection_sha256') != protocol.original.digest(selection) or
            not protocol.original._sha(proof.get('inner_ledger_tip')) or
            selection.get('selected_candidate_id') not in protocol.CANDIDATES or
            selection.get('inner_numerical_passed') is not True or
            any(selection.get(key) is not False for key in protocol.FLAGS) or
            any(proof.get(key) is not False for key in ('refit_authorized', 'promoted', 'final_access_authorized'))):
        raise ValueError('Complete actual inner artifacts and a qualified fixed winner are required')
    return copy.deepcopy(proof)


def _refit_context():
    ctx = inputs._evidence()  # exact clean ancestry; no arrays or references
    old = inputs._json('models/pff-research.json', inputs.dev.PARENT_BUNDLE_SHA)
    manifests = [m for m in old['cache_manifests'] if m['match_id'] in protocol.TRAIN_IDS]
    pff = {f'pff{m["match_id"]}': m for m in manifests}
    if len(manifests) != 40 or set(pff) != {f'pff{mid}' for mid in protocol.TRAIN_IDS}:
        raise ValueError('Original forty native cache manifests required')
    if any(m['identity']['sources'] != ctx['producer_sources'] for m in manifests):
        raise ValueError('Refit observation producer identity differs')
    return dict(ctx, pff=pff)


def _pff_record(match, ctx):
    _require_fixture(match)
    manifest = ctx['pff'][match]
    mid = int(match[3:]); folder = f'{inputs.PFF_CACHE}/{mid}'
    raw = inputs._path(f'{folder}/complete.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() not in inputs.dev.old_verifier._serialized_hashes(manifest) or
            json.loads(raw) != manifest):
        raise ValueError('Refit native cache differs from pinned original provenance')
    item = inputs._json(f'{folder}/item.json', manifest['files']['item.json'])
    if (set(item) != inputs.PFF_ITEM_KEYS or item['match'] != match or
            item['reference_fields_used_for_inference'] is not False):
        raise ValueError('Native observation-only refit metadata required')
    inputs._period_metadata(item['blocks'], item['usable'], item['fps'], manifest['rows'])
    alignment = inputs._json(f'{folder}/alignment.json', manifest['files']['alignment.json'])
    if (str(alignment.get('match')) != str(mid) or alignment.get('reference_fields_used_for_inference') is not False or
            alignment.get('fps') != 25 or len(alignment.get('source_frames', [])) != manifest['rows']):
        raise ValueError('Refit alignment belongs to another observation timeline')
    x = inputs._array(f'{folder}/x.npy', manifest['files']['x.npy'])
    z = inputs._array(f'{folder}/z.npy', manifest['files']['z.npy'])
    inputs._validate_features(x, z, manifest['rows'])
    events = inputs._json(f'data/pff/Event Data/{mid}.json', manifest['identity']['events'])
    y, _ = inputs.prior.pff_adapter.references(events, alignment)
    return inputs._validate_record(dict(match=match, x=x, z=z, y=y, blocks=item['blocks'], fps=item['fps'],
                                        reference_fields_used_for_inference=False))


def _record(match, ctx):
    _require_fixture(match)  # before any source/record I/O
    for key in ('producer_sources', 'clean_sources'):
        inputs._sources(ctx[key])
    inputs._sources(ctx['snapshot']['source_hashes'])
    return _pff_record(match, ctx) if match.startswith('pff') else inputs._sportec_record(match, ctx)


def _sample(record, match):
    """Same mathematical sampling as frozen inner kernel; different strict roster.

    Kept here rather than monkeypatching the frozen kernel's held-out barrier.
    Parity tests require exact arrays, weights and support for overlapping IDs.
    """
    _require_fixture(match)
    if not isinstance(record, dict) or record.get('match') != match or set(record) != kernel.RECORD_KEYS:
        raise ValueError('Exact observation/reference refit record required')
    if record['reference_fields_used_for_inference'] is not False or type(record['fps']) not in (int, float) or record['fps'] != 25:
        raise ValueError('Fixed observation-only 25 Hz timeline required')
    x, z, y = np.asarray(record['x']), np.asarray(record['z']), np.asarray(record['y'])
    if (x.ndim != 2 or x.shape[1] != len(kernel.robust_control.FEATURE_NAMES) or
            z.shape != (len(x), len(kernel.learned.FEATURE_NAMES)) or y.shape != (len(x),) or
            not np.isfinite(x).all() or not np.isfinite(z).all() or not np.isin(y, list('HADX')).all()):
        raise ValueError('Invalid refit observations or references')
    parameters = protocol.recipe()
    blocks = kernel._blocks(record['blocks'], len(x))
    weights = np.zeros(len(y), dtype=float)
    stride = int(record['fps'] / parameters['sample_hz'])
    radius = round(parameters['boundary_radius_seconds'] * record['fps'])
    indices = []
    for _, a, b in blocks:
        local = y[a:b+1]
        w = kernel.training_spell_weights(local, record['fps'])
        w[local == 'D'] = 1.
        transitions = np.flatnonzero((local[1:] != local[:-1]) &
                                    (local[1:] != 'X') & (local[:-1] != 'X')) + 1
        for j in transitions:
            lo, hi = max(0, j-radius), min(len(w), j+radius+1)
            w[lo:hi] = np.maximum(w[lo:hi], parameters['boundary_weight'])
        w[local == 'X'] = 0.
        weights[a:b+1] = w
        indices.append(np.arange(a, b+1, stride))
    mass = float(sum(weights[ix].sum() for ix in indices))
    if mass <= 0:
        raise ValueError('Every fitting fixture needs positive sampled known-label mass')
    weights *= parameters['fixture_mass'] / mass
    sequences = [dict(X=np.array(x[ix], dtype=float, copy=True),
                      Z=np.array(z[ix], dtype=float, copy=True),
                      y=np.array(y[ix], copy=True), w=np.array(weights[ix], copy=True)) for ix in indices]
    support = dict(periods=len(sequences), sampled_rows=sum(len(s['y']) for s in sequences),
                   class_counts={c: sum(int(np.sum(s['y'] == c)) for s in sequences) for c in 'HADX'},
                   sampled_weight=float(sum(s['w'].sum() for s in sequences)))
    return sequences, support


def _metadata(candidate, parent_model, proof_sha256):
    return dict(candidate=candidate, phase='winner-refit', inner_proof_sha256=proof_sha256,
        protocol_sha256=protocol.fixed_protocol_sha256(), training_matches=list(protocol.fit_ids('refit')),
        clean_parent_file_sha256=kernel.PARENT_FILE_SHA256, clean_parent_content_sha256=kernel.PARENT_CONTENT_SHA256,
        excluded_pff_matches=[f'pff{mid}' for mid in sorted((*protocol.DEVELOPMENT_IDS, *protocol.FINAL_IDS))],
        fitting_data='Original PFF forty plus seven clean Sportec fixtures',
        use_scope='local analytical research; delivery rights unresolved',
        frozen_control_content_sha256=kernel.content_digest(parent_model['robust_control_model']),
        parent_dead_parameters_content_sha256=kernel.content_digest({k: parent_model[k] for k in kernel.DEAD_PARAMETERS}),
        discarded_joint_content_sha256=kernel.content_digest(parent_model['joint_model']),
        upstream_stacking='in-sample-fit-only', old_joint_used=False, external_only=True, **FLAGS)


def _fit(candidate, record_factory, parent_model, proof_sha256):
    if candidate not in protocol.CANDIDATES or type(candidate) is not str or not protocol.original._sha(proof_sha256):
        raise ValueError('Fixed selected candidate and proof identity required')
    kernel.validate_parent(parent_model)
    saved_parent = kernel.content_digest(parent_model)
    model = copy.deepcopy(parent_model); model.pop('joint_model')
    parameters = protocol.recipe(); roster = protocol.fit_ids('refit')
    sequences, support = [], {}
    for match in roster:
        sampled, counts = _sample(record_factory(match), match)
        sequences.extend(sampled); support[match] = counts
    if not all(sum(support[m]['class_counts'][c] for m in roster) > 0 for c in 'HAD'):
        raise ValueError('Refit cohort requires sampled H, A and D support')
    metadata = _metadata(candidate, parent_model, proof_sha256)
    if candidate == 'B':
        dead = kernel.learned.fit_model(np.concatenate([s['Z'] for s in sequences]),
            np.concatenate([s['y'] for s in sequences]), l2=parameters['dead_l2'],
            sample_weight=np.concatenate([s['w'] for s in sequences]), max_iter=parameters['dead_max_iter'], metadata=metadata)
        for key in kernel.DEAD_PARAMETERS: model[key] = copy.deepcopy(dead[key])
        model['training'] = copy.deepcopy(dead['training'])
    for sequence in sequences:
        sequence['base_home'] = kernel._logit(kernel.robust_control.predict_home(model['robust_control_model'], sequence['X']))
        sequence['base_dead'] = kernel._logit(kernel.learned.predict_dead(sequence['Z'], model))
    model['joint_model'] = kernel.train_joint.fit(sequences, seed=parameters['seed'], epochs=parameters['epochs'], metadata=metadata)
    model['candidate_id'] = f'clean-pff-adaptation-{candidate}-winner-refit'
    model['metadata'] = dict(adaptation=metadata, frozen_parent_metadata=copy.deepcopy(parent_model['metadata']))
    _validate_fitted_model(model, parent_model, metadata, candidate)
    if (kernel.content_digest(parent_model) != saved_parent or
            model['robust_control_model'] != parent_model['robust_control_model'] or
            model['decoder_mode'] != parent_model['decoder_mode'] or
            (candidate == 'A' and any(model[k] != parent_model[k] for k in kernel.DEAD_PARAMETERS))):
        raise ValueError('Fixed parent components changed during winner refit')
    report = dict(metadata, recipe=parameters, fitting_fixtures=len(roster), fixture_support=support,
        new_joint_training=copy.deepcopy(model['joint_model'].get('training', {})),
        new_dead_training=copy.deepcopy(model.get('training')) if candidate == 'B' else None)
    return model, report


def _validate_fitted_model(model, parent, metadata, candidate):
    kernel.learned.validate_model(model)
    expected_keys = set(parent) | ({'training'} if candidate == 'B' else set())
    joint = model['joint_model']
    training = joint.get('training', {})
    parameters = protocol.recipe()
    if (set(model) != expected_keys or
            model['candidate_id'] != f'clean-pff-adaptation-{candidate}-winner-refit' or
            storage.canonical(model['metadata']) != storage.canonical(dict(adaptation=metadata, frozen_parent_metadata=parent['metadata'])) or
            storage.canonical(joint.get('metadata')) != storage.canonical(metadata) or
            training.get('seed') != parameters['seed'] or training.get('epochs') != parameters['epochs'] or
            kernel.content_digest(model['robust_control_model']) != kernel.content_digest(parent['robust_control_model']) or
            model['decoder_mode'] != parent['decoder_mode'] or
            (candidate == 'A' and kernel.content_digest({key:model[key] for key in kernel.DEAD_PARAMETERS}) !=
             kernel.content_digest({key:parent[key] for key in kernel.DEAD_PARAMETERS}))):
        raise ValueError('Refit model ancestry, joint fit schedule or fixed components differ')
    if candidate == 'B':
        dead = model['training']
        if (not isinstance(dead, dict) or type(dead.get('l2')) not in (int, float) or
                dead['l2'] != parameters['dead_l2'] or type(dead.get('iterations')) is not int or
                not 1 <= dead['iterations'] <= parameters['dead_max_iter']):
            raise ValueError('Refit dead-head regularization or iteration bound differs')


def _sources():
    result = stages.source_hashes(ROOT)
    pending = [ROOT/name for name in ('tools/clean_pff_adaptation_refit.py', 'tools/clean_pff_adaptation_campaign.py')]
    seen = set()
    def add(name):
        parts = name.split('.')
        if any(not part.isidentifier() for part in parts): return
        for end in range(1, len(parts)+1):
            path = ROOT.joinpath(*parts[:end])/'__init__.py'
            if path.is_file() and path not in seen: pending.append(path)
        path = ROOT.joinpath(*parts).with_suffix('.py')
        if path.is_file() and path not in seen: pending.append(path)
    while pending:
        path = stages._no_links(pending.pop())
        if path in seen: continue
        if not path.is_relative_to(ROOT): raise ValueError('Refit source escapes repository')
        seen.add(path)
        package = list(path.relative_to(ROOT).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                for alias in node.names: add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                if node.module: base.extend(node.module.split('.'))
                if base:
                    add('.'.join(base))
                    for alias in node.names:
                        if alias.name != '*': add('.'.join([*base, alias.name]))
        result[path.relative_to(ROOT).as_posix()] = storage.file_hash(path)
    return result


def run_refit(inner_out):
    """Consume the single refit only after actual complete inner verification.

    No candidate override, output override, retry or release path exists. The
    bounded outer operator must impose the wall-clock limit on this function.
    """
    stages.location(inner_out, root=ROOT)
    if any(os.environ.get(name) != '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Refit worker must start with one BLAS/OpenMP thread')
    proof = _verified_selection(inner_out)
    out = stages._no_links(ROOT/'runs'/OUT_NAME)
    out.mkdir(exist_ok=True)
    with storage.locked(out):
        if set(p.name for p in out.iterdir()) - {'.lock'}:
            raise ValueError('Winner-refit attempt already consumed; no retry or runner-up')
        sources = _sources()
        identity = protocol.original.digest(proof)
        marker = dict(schema_version=1, inner_proof_sha256=identity, inner_proof=proof,
                      selected_candidate_id=proof['selection']['selected_candidate_id'],
                      source_hashes=sources, runtime_versions=stages.runtime_versions(),
                      recipe=protocol.recipe(), fitting_fixture_ids=list(protocol.fit_ids('refit')),
                      **FLAGS)
        stages._exclusive(out/'attempt.json', marker)
        try:
            ctx = _refit_context()
            parent = inputs._json(f'{inputs.REBUILD}/refit/candidate.json', protocol.CLEAN_SHA256)
            model, report = _fit(marker['selected_candidate_id'], lambda match: _record(match, ctx), parent, identity)
            if _sources() != sources or protocol.original.digest(_verified_selection(inner_out)) != identity:
                raise ValueError('Source or verified inner winner changed during refit')
            if stages._read(out/'attempt.json') != marker:
                raise ValueError('Consumed refit marker changed')
            stages._exclusive(out/'model.json', model)
            stages._exclusive(out/'report.json', report)
            result = dict(schema_version=1, selected_candidate_id=marker['selected_candidate_id'],
                          inner_proof_sha256=identity, attempt_sha256=storage.file_hash(out/'attempt.json'),
                          model_sha256=storage.file_hash(out/'model.json'), report_sha256=storage.file_hash(out/'report.json'),
                          fitting_fixtures=47, refit_complete=True, **FLAGS)
            stages._exclusive(out/'complete.json', result)
            return _verify_result_locked(inner_out, out)
        except BaseException as error:
            stages._exclusive(out/'failed.json', dict(exception_type=type(error).__name__,
                inner_proof_sha256=identity, attempt_sha256=storage.file_hash(out/'attempt.json'), **FLAGS))
            raise


def _verify_result_locked(inner_out, out):
    proof = _verified_selection(inner_out)
    identity = protocol.original.digest(proof)
    if {p.name for p in out.iterdir()} != {'.lock', 'attempt.json', 'model.json', 'report.json', 'complete.json'}:
        raise ValueError('Complete unfailed refit artifacts required')
    marker = stages._read(out/'attempt.json')
    candidate = proof['selection']['selected_candidate_id']
    expected_marker = dict(schema_version=1, inner_proof_sha256=identity, inner_proof=proof,
        selected_candidate_id=candidate, source_hashes=_sources(), runtime_versions=stages.runtime_versions(), recipe=protocol.recipe(),
        fitting_fixture_ids=list(protocol.fit_ids('refit')), **FLAGS)
    if storage.canonical(marker) != storage.canonical(expected_marker):
        raise ValueError('Refit marker no longer binds actual selected inner result and sources')
    complete = stages._read(out/'complete.json')
    expected = dict(schema_version=1, selected_candidate_id=candidate, inner_proof_sha256=identity,
        attempt_sha256=storage.file_hash(out/'attempt.json'), model_sha256=storage.file_hash(out/'model.json'),
        report_sha256=storage.file_hash(out/'report.json'), fitting_fixtures=47, refit_complete=True, **FLAGS)
    if storage.canonical(complete) != storage.canonical(expected):
        raise ValueError('Refit output bytes differ from the completion receipt')
    parent = inputs._json(f'{inputs.REBUILD}/refit/candidate.json', protocol.CLEAN_SHA256)
    kernel.validate_parent(parent)
    model = stages._read(out/'model.json'); report = stages._read(out/'report.json')
    metadata = _metadata(candidate, parent, identity)
    _validate_fitted_model(model, parent, metadata, candidate)
    expected_report = dict(metadata, recipe=protocol.recipe(), fitting_fixtures=47,
        fixture_support=report.get('fixture_support'),
        new_joint_training=model['joint_model'].get('training', {}),
        new_dead_training=model.get('training') if candidate == 'B' else None)
    if storage.canonical(report) != storage.canonical(expected_report):
        raise ValueError('Refit report does not match model training and exact recipe')
    support = report['fixture_support']
    if not isinstance(support, dict) or set(support) != set(protocol.fit_ids('refit')):
        raise ValueError('Refit report lacks exact47 fitting support')
    for counts in support.values():
        if (not isinstance(counts, dict) or set(counts) != {'periods', 'sampled_rows', 'class_counts', 'sampled_weight'} or
                type(counts['periods']) is not int or counts['periods'] < 1 or
                type(counts['sampled_rows']) is not int or counts['sampled_rows'] < 1 or
                not isinstance(counts['class_counts'], dict) or set(counts['class_counts']) != set('HADX') or
                any(type(v) is not int or v < 0 for v in counts['class_counts'].values()) or
                sum(counts['class_counts'].values()) != counts['sampled_rows'] or
                type(counts['sampled_weight']) not in (int, float) or
                not np.isfinite(counts['sampled_weight']) or
                not np.isclose(counts['sampled_weight'], 10000., rtol=0, atol=1e-7)):
            raise ValueError('Invalid refit aggregate fitting support')
    return expected


def verify_result(inner_out):
    """Read-only actual-byte verification; never starts/resumes a refit."""
    stages.location(inner_out, root=ROOT)
    out = stages._no_links(ROOT/'runs'/OUT_NAME)
    with storage.locked(out):
        return _verify_result_locked(inner_out, out)
