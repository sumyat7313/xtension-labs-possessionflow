"""Verify an external-only candidate's fixed PFF development comparison.

Portable verification recomputes aggregate evidence. Local verification also
hashes actual artifacts without printing their contents. Neither mode opens
final fixtures, fits a model, certifies provider rights, or promotes a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from tools import external_rebuild as rebuild
from tools import improvement_policy as policy
from tools import pff_precision_validation as precision
from tools import pff_validation as original
from tools import verify_pff_research as parent_verifier
from tools import external_pff_development as campaign

CANDIDATE_SHA = '7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47'
CONFIG_SHA = '2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b'


def read(path):
    rebuild.no_links(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _context():
    """Authenticate public aggregate ancestors; no ignored fitted model needed."""
    parent_path = ROOT / 'models/pff-research.json'
    parent = read(parent_path)
    parent_verifier.verify_bundle(parent)
    reservation = parent['protocol']
    old = parent['development']['summary']['records']
    need(reservation['protocol_sha256'] == policy.RESERVATION_SHA256,
         'Original reservation differs')
    need(original.digest(old) == policy.ORIGINAL_PARENT_RECORDS_SHA256,
         'Original PFF control evidence differs')
    rebuild_path = ROOT / 'models/external-rebuild-001-research.json'
    evidence = read(rebuild_path)
    need(evidence['schema_version'] == 1 and evidence['trial_id'] == 'external-rebuild-001',
         'External reconstruction identity differs')
    expected = rebuild.plan(evidence['protocol']['recipe'])
    need({key: evidence['protocol'][key] for key in expected} == expected,
         'Reconstruction source or protocol differs')
    need(set(evidence['completed_stage_seals']) == set(expected['stages'])
         and all(original._sha(value) for value in evidence['completed_stage_seals'].values()),
         'Every reconstruction stage must be recorded')
    summary = evidence['summary']
    selection = rebuild.select(summary['records'])
    need(all(summary[key] == value for key, value in selection.items()),
         'Reconstruction selection does not reproduce')
    need(summary['status'] == 'complete' and summary['stage_completed'] == 'refit'
         and summary['next_stage'] is None and summary['external_only'] is True
         and all(summary[field] is False for field in ('final_accessed', 'promoted', 'submission_eligible'))
         and summary['release_candidate_model_sha256'] == CANDIDATE_SHA,
         'Reconstruction completion or candidate differs')
    need(summary['protocol_sha256'] in parent_verifier._serialized_hashes(evidence['protocol']),
         'Reconstruction protocol hash differs')
    ancestry = policy.audit_lineage(evidence['lineage'])
    need(ancestry['candidate_sha256'] == CANDIDATE_SHA
         and summary['lineage_sha256'] in parent_verifier._serialized_hashes(evidence['lineage']),
         'Reconstruction lineage identity differs')
    return {'reservation': reservation, 'prior_records': old,
            'lineage': evidence['lineage'], 'model_paths': summary['model_paths'],
            'old_bundle': parent, 'cache_manifests': parent['cache_manifests'],
            'parent_research_sha256': sha256(parent_path),
            'rebuild_bundle': evidence, 'rebuild_research_sha256': sha256(rebuild_path)}


def _metrics(detail):
    values = parent_verifier.campaign.metrics(detail)
    values['short_recall'] = values.pop('short')
    return values


def _verify_details(summary, ctx):
    """Cross-derive metric records and both historical controls from diagnostics."""
    mids = sorted(ctx['reservation']['split']['development'])
    old = ctx['old_bundle']['development']['summary']
    expected = {(str(mid), name) for mid in mids for name in ('baseline', 'parent', 'candidate')}
    details = {}
    for detail in summary['details']:
        need(set(detail) == set(old['details'][0]), 'Unexpected diagnostic fields')
        key = detail['match'], detail['candidate']
        need(key in expected and key not in details and detail['regime'] == 'native'
             and original._sha(detail['prediction_sha256']), 'Foreign, duplicate or invalid diagnostic identity')
        details[key] = detail
    need(set(details) == expected, 'Every model requires all twelve diagnostic records')
    need(summary['parent_records'] == ctx['prior_records'], 'Historical PFF control metrics changed')
    need(summary['reference_support'] == old['reference_support'], 'Historical reference support changed')
    old_details = {(d['match'], d['candidate']): d for d in old['details']}
    hashes = {'parent': precision.PARENT_SHA256, 'candidate': CANDIDATE_SHA}
    for name, records in (('parent', summary['parent_records']), ('candidate', summary['records'])):
        need([row['match_id'] for row in records] == mids, 'Metric records must cover the ordered development cohort')
        for row in records:
            need(set(row) == set(ctx['prior_records'][0]), 'Unexpected metric record fields')
            need(row['candidate_sha256'] == hashes[name]
                 and row['baseline_sha256'] == original.BASELINE_SHA256
                 and row['condition'] == 'native', 'Metric model identity or condition differs')
            mid = str(row['match_id'])
            for side, which in (('baseline', 'baseline'), ('candidate', name)):
                detail = details[mid, which]
                need(_metrics(detail) == row[side], 'Metric record differs from detailed diagnostics')
                for key, section in (('reference_dead_segments', 'dead_segments'),
                                     ('reference_short_spells', 'short_possessions')):
                    need(row[key] == detail['temporal_diagnostics'][section]['n_gt'],
                         'Reference event support differs from diagnostics')
    for mid in mids:
        for name, old_name in (('baseline', 'baseline'), ('parent', 'candidate')):
            detail = details[str(mid), name]
            historical = old_details[str(mid), old_name]
            need(_metrics(detail) == _metrics(historical)
                 and detail['prediction_sha256'] == historical['prediction_sha256'],
                 'Historical control diagnostics or scored prediction identity changed')
    return details


def _forbidden_activity(out):
    """Names only: do not inspect any unexpected final/training artifact."""
    out = Path(out)
    rebuild.no_links(out)
    for path in out.iterdir():
        rebuild.no_links(path)
        if path.name.startswith('final') or path.name in ('fit-access.json', 'training-plan.json'):
            return True
        if path.name.endswith('-started.json') and path.name.removesuffix('-started.json') not in campaign.STAGES:
            return True
        if path.is_dir() and path.name not in ('predictions', 'scores', 'stages'):
            return True
    for name in ('predictions', 'scores'):
        folder = out / name
        if folder.exists() and any(path.name not in {str(mid) for mid in campaign.IDS} for path in folder.iterdir()):
            return True
    return False


def _flags(value):
    need(all(value.get(key) is expected for key, expected in campaign.FLAGS.items()),
         'Research, final-access or promotion flags differ')


def _proofs(out, state):
    return {stage: read(Path(out) / 'stages' / (stage + '.json')) for stage in state['completed']}


def bundle_from_run(out):
    out = Path(out)
    state = read(out / 'state.json')
    need(set(state['completed']) == set(campaign.STAGES),
         'Incomplete study cannot produce a completed evidence bundle')
    return {
        'schema_version': 1, 'protocol': read(out / 'protocol.json'), 'state': state,
        'stage_proofs': _proofs(out, state),
        'start_markers': {stage: read(out / (stage + '-started.json')) for stage in campaign.STAGES},
        'run_summary': read(out / 'summary.json'),
        'development': {'summary': read(out / 'development-summary.json'),
                        'prediction_freeze': read(out / 'predictions-before-references.json'),
                        'reference_access': read(out / 'reference-access.json')},
        'lineage': read(ROOT / 'models/external-rebuild-001-research.json')['lineage'],
        'model_paths': read(ROOT / 'models/external-rebuild-001-research.json')['summary']['model_paths'],
        'parent_research_sha256': sha256(ROOT / 'models/pff-research.json'),
        'rebuild_research_sha256': sha256(ROOT / 'models/external-rebuild-001-research.json'),
        'weights_included': False, 'fit_performed': False,
        'final_artifacts_present': _forbidden_activity(out),
        'final_access_authorized': False, 'promoted': False, 'submission_eligible': False,
    }


def _verify_protocol(study, ctx):
    need(study == campaign.build_protocol(ctx, study['recipe']),
         'Fixed candidate, cohort, source, ancestry or numerical policy differs')
    need(study['parent_research_sha256'] == ctx['parent_research_sha256'] == campaign.PARENT_BUNDLE_SHA
         and study['rebuild_research_sha256'] == ctx['rebuild_research_sha256'] == campaign.REBUILD_BUNDLE_SHA,
         'Published ancestor file hashes differ')


def _prediction_proof(proof, study, ctx):
    fields = {'match_id', 'protocol_sha256', 'candidate_sha256', 'cache_manifest_sha256', 'prediction_sha256'}
    need(set(proof) == fields and type(proof['match_id']) is int and proof['match_id'] in campaign.IDS,
         'Prediction proof has foreign identity or fields')
    mid = proof['match_id']
    need(proof['protocol_sha256'] == study['protocol_sha256']
         and proof['candidate_sha256'] == CANDIDATE_SHA
         and proof['cache_manifest_sha256'] == campaign.CACHE_HASHES[str(mid)]
         and set(proof['prediction_sha256']) == set(campaign.NAMES)
         and all(original._sha(value) for value in proof['prediction_sha256'].values()),
         'Prediction or observation identity differs')
    old = next(r for r in ctx['old_bundle']['development']['prediction_freeze']['records'] if r['match_id'] == mid)
    need(proof['prediction_sha256']['baseline'] == old['baseline_prediction_sha256']
         and proof['prediction_sha256']['parent'] == old['candidate_prediction_sha256'],
         'Historical prediction controls changed')


def _verify_freeze(frozen, study, ctx):
    need([r['match_id'] for r in frozen['records']] == list(campaign.IDS),
         'All twelve predictions must be frozen before reference access')
    for proof in frozen['records']:
        _prediction_proof(proof, study, ctx)
    need(frozen == campaign.freeze_value(study, frozen['records']),
         'Prediction-before-reference declaration differs')


def _expected_access(study, freeze_hash):
    return dict(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
                candidate_sha256=CANDIDATE_SHA, source_hashes=study['source_hashes'],
                prediction_freeze_sha256=freeze_hash, references_consumed=True,
                development_previously_consumed=True, **campaign.FLAGS)


def _verify_screen(screen, gate, ctx):
    ancestry = policy.audit_lineage(ctx['lineage'])
    datasets = sorted(set(ancestry['external_dataset_ids']) | {policy.EVALUATION_DATASET})
    expected = {
        'policy': dict(policy.POLICY), 'policy_sha256': original.digest(policy.POLICY),
        'numeric_gate': gate, 'lineage': ancestry,
        'artifact_bindings': {'model_bytes_authenticated': True, 'n_models': len(ctx['model_paths'])},
        'evidence_receipts': policy.inspect_receipts([], datasets, ROOT),
        'ready_for_independent_final_review': False, 'final_access_authorized': False, 'promoted': False,
        'required_before_final_access': [
            'Independent substantive review of dataset rights, reference quality and full lineage.',
            'Run-level verification of frozen protocol, complete bounded selection and all predictions before references.',
            'Exactly one eligible candidate; final cohort remains one-shot and cannot become development data.',
        ],
    }
    need(screen == expected, 'Evidence screening overstates rights, lineage or final eligibility')


def _verify_stage_state(study, state, proofs, markers, *, complete):
    need(set(state) == {'protocol_sha256', 'completed'}
         and state['protocol_sha256'] == study['protocol_sha256']
         and isinstance(state['completed'], dict), 'Stage state differs')
    count = len(state['completed'])
    need(set(state['completed']) == set(campaign.STAGES[:count])
         and set(proofs) == set(state['completed'])
         and (not complete or count == len(campaign.STAGES)), 'Stages are missing, foreign or out of order')
    need(set(markers) == set(state['completed']), 'Each completed stage requires its exclusive start marker')
    for stage, proof in proofs.items():
        need(set(proof) == {'stage', 'protocol_sha256', 'artifacts'} and proof['stage'] == stage
             and proof['protocol_sha256'] == study['protocol_sha256']
             and isinstance(proof['artifacts'], dict) and proof['artifacts']
             and all(original._sha(value) for value in proof['artifacts'].values()), 'Stage proof differs')
        need(state['completed'][stage] in parent_verifier._serialized_hashes(proof), 'Stage completion hash differs')
        marker = markers[stage]
        need(set(marker) == {'stage', 'protocol_sha256', 'candidate_sha256', 'source_hashes', 'pid'}
             and marker['stage'] == stage and marker['protocol_sha256'] == study['protocol_sha256']
             and marker['candidate_sha256'] == CANDIDATE_SHA and marker['source_hashes'] == study['source_hashes']
             and type(marker['pid']) is int and marker['pid'] > 0, 'Exclusive stage marker differs')


def _verify_stage_artifacts(bundle):
    study = bundle['protocol']; proofs = bundle['stage_proofs']
    block = bundle['development']; frozen = block['prediction_freeze']; summary = block['summary']
    need(set(proofs['initialize']['artifacts']) == {'protocol.json'}
         and proofs['initialize']['artifacts']['protocol.json'] in parent_verifier._serialized_hashes(study),
         'Initialization is not bound to the frozen protocol file')
    for row in frozen['records']:
        mid = row['match_id']; artifacts = proofs[f'predict-{mid}']['artifacts']
        expected = {f'predictions/{mid}/{name}.npy': value for name, value in row['prediction_sha256'].items()}
        path = f'predictions/{mid}/complete.json'
        need(set(artifacts) == {*expected, path}
             and all(artifacts[key] == value for key, value in expected.items())
             and artifacts[path] in parent_verifier._serialized_hashes(row), 'Prediction stage proof differs')
    need(proofs['freeze']['artifacts'] == {'predictions-before-references.json': summary['prediction_freeze_sha256']},
         'Freeze stage identity differs')
    scores = proofs['score']['artifacts']
    expected = {'development-summary.json', 'reference-access.json'}
    for detail in summary['details']:
        prefix = f"scores/{detail['match']}/{detail['candidate']}"
        expected.update((prefix + '/predictions.csv', prefix + '/metrics.json'))
        need(scores.get(prefix + '/predictions.csv') == detail['prediction_sha256'], 'Scored CSV and diagnostics differ')
    need(set(scores) == expected and scores['development-summary.json'] in parent_verifier._serialized_hashes(summary)
         and scores['reference-access.json'] == summary['reference_access_sha256'], 'Scoring stage artifacts differ')


def verify_bundle(bundle):
    fields = {'schema_version', 'protocol', 'state', 'stage_proofs', 'start_markers', 'run_summary',
              'development', 'lineage', 'model_paths', 'parent_research_sha256', 'rebuild_research_sha256',
              'weights_included', 'fit_performed', 'final_artifacts_present', 'final_access_authorized',
              'promoted', 'submission_eligible'}
    need(isinstance(bundle, dict) and set(bundle) == fields and type(bundle['schema_version']) is int
         and bundle['schema_version'] == 1, 'Completed bundle schema differs')
    need(all(bundle[key] is False for key in ('weights_included', 'fit_performed', 'final_artifacts_present',
                                             'final_access_authorized', 'promoted', 'submission_eligible')),
         'Study cannot contain fitting, final access or promotion')
    ctx = _context(); study = bundle['protocol']; _verify_protocol(study, ctx)
    need(bundle['lineage'] == ctx['lineage'] and bundle['model_paths'] == ctx['model_paths']
         and bundle['parent_research_sha256'] == ctx['parent_research_sha256']
         and bundle['rebuild_research_sha256'] == ctx['rebuild_research_sha256'], 'Fitted ancestry or ancestor bundle differs')
    _verify_stage_state(study, bundle['state'], bundle['stage_proofs'], bundle['start_markers'], complete=True)
    block = bundle['development']
    need(set(block) == {'summary', 'prediction_freeze', 'reference_access'}, 'Completed development evidence missing')
    summary, frozen, access = block['summary'], block['prediction_freeze'], block['reference_access']
    summary_fields = {'schema_version', 'part', 'protocol_sha256', 'candidate_sha256', 'source_hashes',
                      'records', 'parent_records', 'details', 'reference_support', 'gate', 'evidence_screen',
                      'development_gate_passed', 'prediction_freeze_sha256', 'reference_access_sha256',
                      'reference_quality', *campaign.FLAGS}
    need(set(summary) == summary_fields, 'Development summary fields differ')
    _verify_freeze(frozen, study, ctx)
    _flags(summary)
    need(summary['schema_version'] == 1 and summary['part'] == 'development'
         and summary['protocol_sha256'] == study['protocol_sha256']
         and summary['candidate_sha256'] == CANDIDATE_SHA and summary['source_hashes'] == study['source_hashes']
         and summary['reference_quality'] == 'PFF provider-event-derived silver; previously used development',
         'Scored development provenance differs')
    need(summary['prediction_freeze_sha256'] in parent_verifier._serialized_hashes(frozen)
         and access == _expected_access(study, summary['prediction_freeze_sha256'])
         and summary['reference_access_sha256'] in parent_verifier._serialized_hashes(access, False),
         'Exclusive reference access is not bound to the complete prediction freeze')
    _verify_details(summary, ctx)
    gate = precision.development_gate(summary['records'], ctx['prior_records'], ctx['reservation'])
    need(summary['gate'] == gate and summary['development_gate_passed'] is gate['numeric_gate_passed'],
         'Reported numerical gate does not reproduce')
    _verify_screen(summary['evidence_screen'], gate, ctx)
    run = bundle['run_summary']; _flags(run)
    need(set(run) == {'schema_version', 'status', 'stage_completed', 'next_stage', 'protocol_sha256',
                     'protocol_file_sha256', 'candidate_sha256', 'source_hashes',
                     'development_gate_passed', 'development_summary_sha256', *campaign.FLAGS},
         'Run summary fields differ')
    need(run['status'] == 'complete' and run['stage_completed'] == 'score' and run['next_stage'] is None
         and run['protocol_sha256'] == study['protocol_sha256']
         and run['protocol_file_sha256'] in parent_verifier._serialized_hashes(study)
         and run['candidate_sha256'] == CANDIDATE_SHA and run['source_hashes'] == study['source_hashes']
         and run['development_gate_passed'] is gate['numeric_gate_passed']
         and run['development_summary_sha256'] in parent_verifier._serialized_hashes(summary),
         'Run completion summary differs')
    _verify_stage_artifacts(bundle)
    return {'status': 'complete', 'development_gate_passed': gate['numeric_gate_passed'],
            'n_development_matches': 12, 'candidate_sha256': CANDIDATE_SHA,
            'aggregate_gates_recomputed': True, 'historical_controls_reproduced': True,
            'local_artifact_bytes_verified': False, 'final_access_authorized': False,
            'final_evaluated': False, 'promoted': False, 'submission_eligible': False}


def verify_local(bundle, out):
    result = verify_bundle(bundle); out = Path(out)
    need(bundle_from_run(out) == bundle, 'Local run evidence differs from bundle')
    ctx = campaign.context()  # exact local model/config/lineage, no match records
    policy.authenticate_models(policy.audit_lineage(ctx['lineage']), ctx['model_paths'], ROOT)
    campaign.checked_hash(campaign.PARENT / 'candidate.json', precision.PARENT_SHA256)
    for row in bundle['development']['prediction_freeze']['records']:
        mid = str(row['match_id']); manifests = campaign.manifests(ctx)
        for name, identity in manifests[mid]['files'].items():
            campaign.checked_hash(campaign.PARENT / 'cache' / mid / name, identity)
    for proof in bundle['stage_proofs'].values():
        for name, identity in proof['artifacts'].items():
            campaign.checked_hash(out / name, identity)
    need(sha256(out / 'protocol.json') == bundle['run_summary']['protocol_file_sha256'], 'Protocol file bytes differ')
    return dict(result, local_artifact_bytes_verified=True)


def inspect_run(out):
    """Authenticate incomplete status without calling it a completed experiment."""
    out = Path(out); need(not _forbidden_activity(out), 'Unexpected final or fitting activity')
    ctx = campaign.context(); study = read(out / 'protocol.json'); _verify_protocol(study, ctx)
    study, state, _ = campaign.authenticate(out, study['recipe'], ctx)
    proofs = _proofs(out, state)
    markers = {stage: read(out / (stage + '-started.json')) for stage in state['completed']}
    _verify_stage_state(study, state, proofs, markers, complete=False)
    pending = [stage for stage in campaign.STAGES if stage not in state['completed']]
    if not pending:
        return verify_local(bundle_from_run(out), out)
    starts = {p.name.removesuffix('-started.json') for p in out.glob('*-started.json')}
    need(starts <= set(state['completed']) | {pending[0]}, 'An unreported later stage has already been attempted')
    for stage in state['completed']:
        if stage.startswith('predict-'):
            mid = int(stage[8:]); _prediction_proof(read(out / 'predictions' / str(mid) / 'complete.json'), study, ctx)
    if 'freeze' in state['completed']:
        frozen = read(out / 'predictions-before-references.json'); _verify_freeze(frozen, study, ctx)
    consumed = (out / 'reference-access.json').exists()
    if consumed:
        need('freeze' in state['completed'] and pending[0] == 'score', 'References accessed before the complete freeze')
        access = read(out / 'reference-access.json')
        need(access == _expected_access(study, sha256(out / 'predictions-before-references.json')),
             'Incomplete scoring access marker differs')
    attempted = (out / (pending[0] + '-started.json')).exists()
    if attempted:
        marker = read(out / (pending[0] + '-started.json'))
        need(marker.get('stage') == pending[0] and marker.get('protocol_sha256') == study['protocol_sha256']
             and marker.get('candidate_sha256') == CANDIDATE_SHA and marker.get('source_hashes') == study['source_hashes'],
             'Pending attempt identity differs')
    need(not (out / 'development-summary.json').exists() or pending[0] == 'score', 'Unreported score before scoring stage')
    return {'status': 'consumed_incomplete' if attempted or consumed else 'in_progress',
            'completed_stages': len(state['completed']), 'total_stages': len(campaign.STAGES),
            'next_stage': pending[0], 'candidate_sha256': CANDIDATE_SHA,
            'reference_access_consumed': consumed, 'continuation_allowed': not attempted and not consumed,
            'development_gate_passed': None, 'aggregate_gates_recomputed': False,
            'local_completed_artifacts_verified': True, 'final_access_authorized': False,
            'final_evaluated': False, 'promoted': False, 'submission_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'models/external-pff-development-research.json')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--write', type=Path)
    parser.add_argument('--allow-in-progress', action='store_true')
    args = parser.parse_args()
    if args.write and not args.run:
        parser.error('--write requires --run')
    if args.allow_in_progress:
        if not args.run or args.write:
            parser.error('--allow-in-progress requires --run and cannot publish a bundle')
        result = inspect_run(args.run)
    else:
        bundle = bundle_from_run(args.run) if args.run else read(args.evidence)
        result = verify_local(bundle, args.run) if args.run else verify_bundle(bundle)
        if args.write:
            need(not args.write.exists(), 'Completed evidence cannot overwrite a prior bundle')
            write_json(args.write, bundle)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
