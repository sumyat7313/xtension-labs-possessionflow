"""Independent checks for the single clean-refit development comparison.

Aggregate verification cannot establish hidden-challenge accuracy, provider
rights, historical truth, or authorize final-fixture access or submission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from tools import pff_validation as original
from tools import pff_precision_validation as precision
from tools import improvement_policy as policy
from tools import verify_pff_research as historical

IDS = (3813, 3816, 3818, 3823, 3831, 3844, 3850, 3854, 3858, 10502, 10514, 10517)
FINAL_IDS = frozenset((3814, 3815, 3824, 3830, 3837, 3840, 3841, 3848, 3852, 10506, 10510, 10511))
RETAIN = ('spell', 'macro', 'short_precision', 'short_recall',
          'dead_segment', 'dead_boundary', 'dead_recall')
CONFIG_SHA = '2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b'
CLEAN_SHA = '7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47'
ROOT = Path(__file__).resolve().parents[1]
OUT = 'runs/clean-pff-adaptation-development-v1'
REFIT = 'runs/clean-pff-adaptation-refit-v1'
STAGES = ('initialize', *(f'predict-{mid}' for mid in IDS), 'freeze',
          *(f'score-{mid}' for mid in IDS), 'summarize')
FLAGS = dict(external_only=True, final_accessed=False, final_access_authorized=False,
             promoted=False, submission_eligible=False)


def need(value, message):
    if not value:
        raise ValueError(message)


def require_match(value):
    need(type(value) is int and value in IDS, 'Only the fixed twelve development fixtures are permitted')
    return value


def fixed_policy():
    """Do not let a changed module-level policy silently weaken verification."""
    expected = dict(schema_version=1, minimum_dead_precision_gain=.002,
                    minimum_retained_positive_gain_fraction=.9,
                    minimum_development_matches=12, original_gates_required=True,
                    automatic_promotion=False, automatic_final_access=False)
    need(policy.POLICY == expected, 'Original evidence policy changed')
    need(precision.POLICY['minimum_dead_precision_gain'] == .002
         and precision.POLICY['minimum_retained_positive_gain_fraction'] == .9
         and precision.POLICY['retained_metrics'] == list(RETAIN)
         and tuple(precision.RETAIN_METRICS) == RETAIN,
         'Precision margin or protected-gain policy changed')
    return expected


def metrics(detail):
    values = historical.campaign.metrics(detail)
    values['short_recall'] = values.pop('short')
    return values


def verify_details(summary, ctx, candidate_sha256):
    """Derive both paired records from detailed metrics and prior controls."""
    need(original._sha(candidate_sha256) and candidate_sha256 not in policy.PROHIBITED_FIT_MODELS
         and candidate_sha256 != CLEAN_SHA, 'Expected a new clean winner-refit identity')
    need(ctx['reservation']['split']['development'] == list(IDS), 'Development cohort changed')
    old = ctx['old_bundle']['development']['summary']
    expected = {(str(mid), name) for mid in IDS for name in ('baseline', 'parent', 'candidate')}
    details = {}
    for detail in summary['details']:
        need(isinstance(detail, dict) and set(detail) == set(old['details'][0]), 'Diagnostic schema changed')
        key = detail['match'], detail['candidate']
        need(key in expected and key not in details and detail['regime'] == 'native'
             and original._sha(detail['prediction_sha256']), 'Foreign, duplicate or malformed diagnostic')
        details[key] = detail
    need(set(details) == expected, 'All three models require all twelve diagnostic records')
    need(summary['parent_records'] == ctx['prior_records'], 'Original retention anchor changed')
    need(summary['reference_support'] == old['reference_support'], 'Reference support changed')
    for name, records, identity in (
            ('parent', summary['parent_records'], precision.PARENT_SHA256),
            ('candidate', summary['records'], candidate_sha256)):
        need([require_match(row['match_id']) for row in records] == list(IDS),
             'Metric rows must cover the ordered development cohort exactly')
        for row in records:
            need(set(row) == set(ctx['prior_records'][0]), 'Paired metric schema changed')
            need(row['candidate_sha256'] == identity and row['baseline_sha256'] == original.BASELINE_SHA256
                 and row['condition'] == 'native', 'Paired model identity changed')
            for side, label in (('baseline', 'baseline'), ('candidate', name)):
                detail = details[str(row['match_id']), label]
                need(metrics(detail) == row[side], 'Paired metrics differ from detailed diagnostics')
                for field, section in (('reference_dead_segments', 'dead_segments'),
                                       ('reference_short_spells', 'short_possessions')):
                    need(row[field] == detail['temporal_diagnostics'][section]['n_gt'],
                         'Event support differs from scored diagnostics')
    prior_details = {(row['match'], row['candidate']): row for row in old['details']}
    for mid in IDS:
        for name, prior_name in (('baseline', 'baseline'), ('parent', 'candidate')):
            actual = details[str(mid), name]
            prior = prior_details[str(mid), prior_name]
            need(metrics(actual) == metrics(prior)
                 and actual['prediction_sha256'] == prior['prediction_sha256'],
                 'Historical control metrics or prediction bytes changed')
    return details


def recompute_gate(summary, ctx, candidate_sha256):
    fixed_policy()
    verify_details(summary, ctx, candidate_sha256)
    need(ctx['reservation']['protocol_sha256'] == policy.RESERVATION_SHA256,
         'Original reservation identity changed')
    need(original.digest(ctx['prior_records']) == policy.ORIGINAL_PARENT_RECORDS_SHA256,
         'Original PFF improvement anchor changed')
    gate = precision.development_gate(summary['records'], ctx['prior_records'], ctx['reservation'])
    need(gate['candidate_sha256'] == candidate_sha256, 'Gate candidate differs from fixed refit')
    need(summary['gate'] == gate and summary['development_gate_passed'] is gate['numeric_gate_passed'],
         'Reported development decision does not reproduce')
    return gate


def serialized_sha(value):
    from tools import experiment_harness as storage
    return hashlib.sha256(storage.canonical(value) + b'\n').hexdigest()


def flags(value):
    need(all(value.get(name) is expected for name, expected in FLAGS.items()),
         'Research-only or final-access flags changed')


def verify_handoff(ctx):
    """Rebuild the declared graph independently of the execution constructor."""
    from tools import clean_pff_adaptation_protocol as adaptation
    from tools import clean_pff_adaptation_stages as stages
    from tools import clean_pff_adaptation_refit as refit
    complete, inner, marker = ctx['refit_proof'], ctx['inner_proof'], ctx['refit_marker']
    selected = inner['selection']
    need(inner.get('artifacts_verified') is True and inner.get('lifecycle_complete') is True
         and inner.get('protocol_sha256') == stages.SNAPSHOT_PROTOCOL_SHA256
         and inner.get('selection_sha256') == original.digest(selected)
         and selected.get('protocol_sha256') == stages.SNAPSHOT_PROTOCOL_SHA256
         and selected.get('selected_candidate_id') == 'A'
         and selected.get('inner_numerical_passed') is True
         and all(selected.get(name) is False for name in adaptation.FLAGS),
         'Complete independently verified A selection required')
    need(complete.get('selected_candidate_id') == 'A' and complete.get('refit_complete') is True
         and type(complete.get('fitting_fixtures')) is int and complete['fitting_fixtures'] == 47
         and complete.get('inner_proof_sha256') == original.digest(inner)
         and complete.get('model_sha256') == ctx['candidate_sha256']
         and all(complete.get(name) is expected for name, expected in refit.FLAGS.items()),
         'Selected single-refit completion differs')
    expected_marker = dict(schema_version=1, inner_proof_sha256=original.digest(inner), inner_proof=inner,
                           selected_candidate_id='A', source_hashes=refit._sources(),
                           runtime_versions=stages.runtime_versions(), recipe=adaptation.recipe(),
                           fitting_fixture_ids=list(adaptation.fit_ids('refit')), **refit.FLAGS)
    need(marker == expected_marker
         and serialized_sha(marker) == complete['attempt_sha256'],
         'Refit recipe, roster or actual marker binding differs')
    candidate = ctx['candidate_sha256']
    need(original._sha(candidate) and candidate not in policy.PROHIBITED_FIT_MODELS
         and candidate != CLEAN_SHA, 'Wrong refit candidate identity')
    clean = ctx['rebuild_bundle']['lineage']
    need(clean['candidate_sha256'] == CLEAN_SHA, 'Wrong clean reconstruction ancestor')
    graph = copy.deepcopy(clean)
    manifests = [row for row in ctx['cache_manifests'] if row['match_id'] in adaptation.TRAIN_IDS]
    need(len(manifests) == 40 and {row['match_id'] for row in manifests} == set(adaptation.TRAIN_IDS),
         'Exactly original forty training manifests required')
    data = original.digest(dict(train_ids=list(adaptation.TRAIN_IDS),
                                manifests=sorted(manifests, key=lambda row: row['match_id'])))
    source = original.digest(marker['source_hashes'])
    recipe = original.digest(adaptation.recipe())
    selection = complete['inner_proof_sha256']
    graph['artifacts'].extend([
        dict(sha256=data, kind='dataset', scope='external', identifier=policy.EVALUATION_DATASET, parents=[]),
        dict(sha256=source, kind='code', scope='synthetic', identifier='Fixed adaptation refit producer source closure', parents=[]),
        dict(sha256=recipe, kind='configuration', scope='synthetic', identifier='Fixed adaptation fitting recipe', parents=[source]),
        dict(sha256=selection, kind='configuration', scope='external',
             identifier='Verified fixed A selection on original training partition; not a weight warm start',
             parents=[data, source]),
        dict(sha256=candidate, kind='model', scope='external', identifier='Clean-parent A refit original forty plus Sportec seven',
             parents=[CLEAN_SHA, data, source, recipe, selection]),
    ])
    graph['candidate_sha256'] = candidate
    graph['fit_inputs'] = sorted(set(graph['fit_inputs']) | {CLEAN_SHA, data, source, recipe, selection})
    need(ctx['lineage'] == graph, 'Connected lineage differs from authenticated fitting inputs')
    audit = policy.audit_lineage(graph)
    need(audit['external_dataset_ids'] == ['pff-world-cup-2022', 'sportec-idsse-28196177'],
         'Foreign fitting dataset ancestry')
    expected_bindings = dict(ctx['rebuild_bundle']['summary']['model_paths'],
                             **{candidate: f'{REFIT}/model.json'})
    need(ctx['model_paths'] == expected_bindings, 'Actual model binding roster differs')
    inputs = ctx['input_hashes']
    expected_inputs = {f'{REFIT}/{name}' for name in ('attempt.json', 'model.json', 'report.json', 'complete.json')} | {
        'runs/clean-pff-adaptation-inner-v1/manifest.json',
        'runs/clean-pff-adaptation-operator-v2/manifest.json', 'models/pff-research.json',
        'models/external-rebuild-001-research.json', 'configs/baselines/v1.2.json'}
    trees = ctx['authenticated_tree_files']
    expected_trees = {'runs/clean-pff-adaptation-inner-v1', REFIT,
                      'runs/clean-pff-adaptation-operator-v2', 'runs/clean-pff-adaptation-operator-v1'}
    need(set(trees) == expected_trees, 'Authenticated predecessor tree roster differs')
    for tree, members in trees.items():
        need(isinstance(members, list) and members == sorted(set(members)) and members
             and all(type(name) is str and name.startswith(tree + '/') and
                     all(part not in ('', '.', '..') for part in name.split('/')) and '\\' not in name
                     and not Path(name).is_absolute() and Path(name).name != '.lock' for name in members),
             'Completed predecessor input inventory is not bounded')
        expected_inputs.update(members)
    expected_inputs.update(expected_bindings.values())
    need(set(inputs) == expected_inputs and all(original._sha(value) for value in inputs.values())
         and inputs.get(f'{REFIT}/model.json') == candidate
         and inputs.get(f'{REFIT}/attempt.json') == complete['attempt_sha256']
         and inputs.get(f'{REFIT}/report.json') == complete['report_sha256']
         and inputs.get(f'{REFIT}/complete.json') == serialized_sha(complete)
         and inputs.get('configs/baselines/v1.2.json') == CONFIG_SHA,
         'Actual model, recipe or completion input binding differs')
    return audit


def verify_predictions(frozen, study, ctx):
    expected_fields = {'match_id', 'protocol_sha256', 'candidate_sha256',
                       'cache_manifest_sha256', 'prediction_sha256'}
    need([require_match(row['match_id']) for row in frozen['records']] == list(IDS),
         'Entire development prediction cohort must be frozen first')
    prior = {row['match_id']: row for row in ctx['old_bundle']['development']['prediction_freeze']['records']}
    for row in frozen['records']:
        need(set(row) == expected_fields and row['protocol_sha256'] == study['protocol_sha256']
             and row['candidate_sha256'] == study['candidate_sha256']
             and row['cache_manifest_sha256'] == study['cache_manifest_sha256'][str(row['match_id'])]
             and set(row['prediction_sha256']) == {'candidate', 'baseline', 'parent'}
             and all(original._sha(value) for value in row['prediction_sha256'].values()),
             'Prediction proof identity differs')
        before = prior[row['match_id']]
        need(row['prediction_sha256']['baseline'] == before['baseline_prediction_sha256']
             and row['prediction_sha256']['parent'] == before['candidate_prediction_sha256'],
             'Historical frozen prediction controls changed')
    expected = dict(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
                    candidate_sha256=study['candidate_sha256'], source_hashes=study['source_hashes'],
                    records=frozen['records'], references_accessed_in_this_study=False,
                    development_previously_consumed=True, **FLAGS)
    need(frozen == expected, 'Prediction-before-reference declaration differs')
    return serialized_sha(frozen)


def expected_access(study, mid, freeze_sha):
    require_match(mid)
    return dict(schema_version=1, part='development', match_id=mid,
                protocol_sha256=study['protocol_sha256'], candidate_sha256=study['candidate_sha256'],
                prediction_freeze_sha256=freeze_sha, references_consumed=True,
                development_previously_consumed=True, **FLAGS)


def expected_screen(gate, ctx):
    audit = policy.audit_lineage(ctx['lineage'])
    datasets = sorted(set(audit['external_dataset_ids']) | {policy.EVALUATION_DATASET})
    expected = dict(policy=fixed_policy(), policy_sha256=original.digest(policy.POLICY),
                    numeric_gate=gate, lineage=audit,
                    artifact_bindings=dict(model_bytes_authenticated=True, n_models=len(ctx['model_paths'])),
                    evidence_receipts=policy.inspect_receipts([], datasets, ROOT),
                    ready_for_independent_final_review=False, final_access_authorized=False, promoted=False,
                    required_before_final_access=[
                        'Independent substantive review of dataset rights, reference quality and full lineage.',
                        'Run-level verification of frozen protocol, complete bounded selection and all predictions before references.',
                        'Exactly one eligible candidate; final cohort remains one-shot and cannot become development data.',
                    ])
    return expected


def verify_screen(screen, gate, ctx):
    need(screen == expected_screen(gate, ctx),
         'Unresolved rights/label quality or submission hold overstated')


def score_files(mid):
    require_match(mid)
    return {f'scores/{mid}/access.json', f'scores/{mid}/aggregate.json'} | {
        f'details/{mid}/{name}/{filename}' for name in ('baseline', 'parent', 'candidate')
        for filename in ('predictions.csv', 'metrics.json')}


def expected_stage_files(stage):
    need(stage in STAGES, 'Foreign stage')
    if stage == 'initialize': return {'lineage.json', 'model-bindings.json', 'handoff.json'}
    if stage == 'freeze': return {'predictions-before-references.json'}
    if stage == 'summarize': return {'development-summary.json'}
    kind, value = stage.split('-')
    mid = require_match(int(value))
    if kind == 'score': return score_files(mid)
    return {f'predictions/{mid}/{name}' for name in ('candidate.npy', 'baseline.npy', 'parent.npy', 'complete.json')}


def verify_evidence(bundle, ctx):
    """Pure complete-bundle validation; local execution adds actual-byte checks."""
    from tools import clean_pff_adaptation_development_v1 as run
    fields = {'schema_version', 'protocol', 'state', 'stage_proofs', 'start_markers', 'prediction_freeze',
              'score_aggregates', 'reference_access', 'lineage', 'model_paths', 'handoff', 'development_summary'}
    need(isinstance(bundle, dict) and set(bundle) == fields and type(bundle['schema_version']) is int
         and bundle['schema_version'] == 1, 'Completed bundle schema differs')
    fixed_policy()
    verify_handoff(ctx)
    study = bundle['protocol']
    need(study == run.build_protocol(ctx), 'Frozen sources, runtime, candidate or inputs changed')
    need(run.IDS == IDS and run.STAGES == STAGES and study['matches'] == list(IDS)
         and study['stages'] == list(STAGES) and study['candidate_budget'] == 1
         and study['fitting_permitted'] is False and study['config_sha256'] == CONFIG_SHA
         and study['limits'] == dict(data_workers=1, blas_threads=1, torch_threads=1, stage_timeout_seconds=1800)
         and study['artifact_delivery_rights_verified'] is False and study['fitted_weights_must_remain_local'] is True,
         'Study scope or resource limits changed')
    flags(study)
    need(bundle['lineage'] == ctx['lineage'] and bundle['model_paths'] == ctx['model_paths']
         and bundle['handoff'] == dict(refit_proof=ctx['refit_proof'], inner_proof=ctx['inner_proof'],
                                       input_hashes=ctx['input_hashes'], **FLAGS),
         'Frozen handoff or learned ancestry differs')
    state, proofs, markers = bundle['state'], bundle['stage_proofs'], bundle['start_markers']
    need(set(state) == {'protocol_sha256', 'completed'} and state['protocol_sha256'] == study['protocol_sha256']
         and set(state['completed']) == set(STAGES) and set(proofs) == set(STAGES) and set(markers) == set(STAGES),
         'All fixed stages and exclusive starts must complete')
    for stage in STAGES:
        proof, marker = proofs[stage], markers[stage]
        need(set(proof) == {'stage', 'protocol_sha256', 'artifacts'} and proof['stage'] == stage
             and proof['protocol_sha256'] == study['protocol_sha256']
             and set(proof['artifacts']) == expected_stage_files(stage)
             and all(original._sha(value) for value in proof['artifacts'].values())
             and state['completed'][stage] == serialized_sha(proof), 'Stage completion or artifact inventory differs')
        need(set(marker) == {'stage', 'protocol_sha256', 'candidate_sha256', 'source_hashes', 'pid'}
             and marker['stage'] == stage and marker['protocol_sha256'] == study['protocol_sha256']
             and marker['candidate_sha256'] == study['candidate_sha256'] and marker['source_hashes'] == study['source_hashes']
             and type(marker['pid']) is int and marker['pid'] > 0, 'Exclusive stage marker differs')
    declared_objects = {'lineage.json': bundle['lineage'], 'model-bindings.json': bundle['model_paths'],
                        'handoff.json': bundle['handoff'], 'predictions-before-references.json': bundle['prediction_freeze'],
                        'development-summary.json': bundle['development_summary']}
    freeze_sha = verify_predictions(bundle['prediction_freeze'], study, ctx)
    for row in bundle['prediction_freeze']['records']:
        mid = row['match_id']
        declared_objects[f'predictions/{mid}/complete.json'] = row
        for name, identity in row['prediction_sha256'].items():
            need(proofs[f'predict-{mid}']['artifacts'][f'predictions/{mid}/{name}.npy'] == identity,
                 'Prediction stage differs from frozen cohort')
    summary = bundle['development_summary']
    summary_fields = {'schema_version', 'part', 'protocol_sha256', 'candidate_sha256', 'source_hashes',
                      'records', 'parent_records', 'details', 'reference_support', 'gate', 'evidence_screen',
                      'development_gate_passed', 'prediction_freeze_sha256', 'reference_access_sha256',
                      'reference_quality', *FLAGS}
    need(set(summary) == summary_fields and summary['schema_version'] == 1 and summary['part'] == 'development'
         and summary['protocol_sha256'] == study['protocol_sha256'] and summary['candidate_sha256'] == study['candidate_sha256']
         and summary['source_hashes'] == study['source_hashes'] and summary['prediction_freeze_sha256'] == freeze_sha
         and summary['reference_quality'] == 'PFF provider-event-derived silver; previously used development',
         'Development summary scope or identity differs')
    flags(summary)
    need(set(bundle['score_aggregates']) == {str(mid) for mid in IDS}
         and set(bundle['reference_access']) == {str(mid) for mid in IDS}, 'All twelve independent scores/access proofs required')
    records, parents, details, support, accesses = [], [], [], {}, {}
    for mid in IDS:
        score, access = bundle['score_aggregates'][str(mid)], bundle['reference_access'][str(mid)]
        need(access == expected_access(study, mid, freeze_sha), 'Reference access not bound to entire frozen cohort')
        need(set(score) == {'match_id', 'protocol_sha256', 'candidate_sha256', 'candidate_record', 'parent_record',
                            'details', 'reference_support', 'prediction_freeze_sha256', 'reference_access_sha256'}
             and type(score['match_id']) is int and score['match_id'] == mid
             and score['protocol_sha256'] == study['protocol_sha256'] and score['candidate_sha256'] == study['candidate_sha256']
             and score['prediction_freeze_sha256'] == freeze_sha and score['reference_access_sha256'] == serialized_sha(access),
             'Per-fixture score or access identity differs')
        need(len(score['details']) == 3 and {d['candidate'] for d in score['details']} == {'baseline', 'parent', 'candidate'}
             and all(d['match'] == str(mid) for d in score['details']), 'Per-fixture score detail cohort differs')
        records.append(score['candidate_record']); parents.append(score['parent_record'])
        details.extend(score['details']); support[str(mid)] = score['reference_support']; accesses[str(mid)] = serialized_sha(access)
        declared_objects[f'scores/{mid}/aggregate.json'] = score
        declared_objects[f'scores/{mid}/access.json'] = access
        for detail in score['details']:
            name = f"details/{mid}/{detail['candidate']}/predictions.csv"
            need(proofs[f'score-{mid}']['artifacts'][name] == detail['prediction_sha256'], 'Scored prediction CSV differs from diagnostics')
    need(summary['records'] == records and summary['parent_records'] == parents and summary['details'] == details
         and summary['reference_support'] == support and summary['reference_access_sha256'] == accesses,
         'Aggregate summary differs from all twelve frozen score artifacts')
    for proof in proofs.values():
        for name, identity in proof['artifacts'].items():
            if name in declared_objects:
                need(identity == serialized_sha(declared_objects[name]), 'Aggregate content differs from sealed stage artifact')
    gate = recompute_gate(summary, ctx, study['candidate_sha256'])
    verify_screen(summary['evidence_screen'], gate, ctx)
    return dict(status='complete', n_development_matches=12, candidate_sha256=study['candidate_sha256'],
                development_gate_passed=gate['numeric_gate_passed'], aggregate_gates_recomputed=True,
                historical_controls_reproduced=True, local_artifact_bytes_verified=False,
                artifact_delivery_rights_verified=False, fitted_weights_must_remain_local=True,
                ready_for_independent_final_review=False, final_evaluated=False, **FLAGS)


def bundle_from_run():
    from tools import clean_pff_adaptation_development_v1 as run
    need(run.IDS == IDS and run.STAGES == STAGES and run.OUT == OUT, 'Fixed execution scope changed')
    state = run.state()
    need(set(state['completed']) == set(STAGES), 'Partial or consumed attempts cannot become complete evidence')
    return dict(schema_version=1, protocol=run.read('protocol.json'), state=state,
                stage_proofs={stage: run.read(f'stages/{stage}.json') for stage in STAGES},
                start_markers={stage: run.read(stage + '-started.json') for stage in STAGES},
                prediction_freeze=run.read('predictions-before-references.json'),
                score_aggregates={str(mid): run.read(f'scores/{mid}/aggregate.json') for mid in IDS},
                reference_access={str(mid): run.read(f'scores/{mid}/access.json') for mid in IDS},
                lineage=run.read('lineage.json'), model_paths=run.read('model-bindings.json'),
                handoff=run.read('handoff.json'), development_summary=run.read('development-summary.json'))


def verify_owned_history(attempts, study, markers, proofs):
    """Independent completed-receipt audit; live PID checks belong to admission."""
    from tools import owned_python_launch as owned
    need(isinstance(attempts, list) and len(attempts) == len(STAGES),
         'Every consumed owned attempt must have one complete terminal proof')
    previous = '0'*64
    nonces = set()
    for index, (stage, attempt) in enumerate(zip(STAGES, attempts)):
        need(set(attempt) == {'request', 'result', 'worker', 'launch', 'claim'}, 'Owned attempt inventory differs')
        request, result, worker, launch, claim = (attempt[key] for key in ('request', 'result', 'worker', 'launch', 'claim'))
        need(set(request) == {'index', 'stage', 'nonce', 'parent_pid', 'protocol_sha256', 'previous_result_sha256'}
             and type(request['index']) is int and request['index'] == index and request['stage'] == stage
             and request['protocol_sha256'] == study['protocol_sha256']
             and request['previous_result_sha256'] == previous
             and type(request['parent_pid']) is int and request['parent_pid'] > 0
             and type(request['nonce']) is str and re.fullmatch('[0-9a-f]{32}', request['nonce'])
             and request['nonce'] not in nonces, 'Consumed stage chain or nonce differs')
        nonces.add(request['nonce'])
        request_sha = original.digest(request)
        need(set(launch) == {'schema_version', 'platform', 'request_sha256', 'receipt_path', 'parent_pid',
                             'owned_pid', 'runtime', 'parent_identity', 'owned_identity'}
             and type(launch['schema_version']) is int and launch['schema_version'] == 1
             and launch['platform'] == ('windows' if os.name == 'nt' else 'posix')
             and launch['receipt_path'] == owned._normal(ROOT/OUT/'operator'/'attempts'/f'{index:03d}.launch.json')
             and launch['request_sha256'] == request_sha and launch['parent_pid'] == request['parent_pid']
             and type(launch['owned_pid']) is int and launch['owned_pid'] > 0
             and launch['runtime'] == study['runtime_identity'], 'Owned launch path or runtime differs')
        launch_sha = original.digest(launch)
        need(set(claim) == {'request_sha256', 'worker_pid', 'launch_receipt_sha256', 'admission', 'resources'}
             and claim['request_sha256'] == request_sha and claim['launch_receipt_sha256'] == launch_sha
             and type(claim['worker_pid']) is int and claim['worker_pid'] > 0
             and markers[stage]['pid'] == claim['worker_pid'], 'Stage was not produced by its claimed worker')
        admission = claim['admission']
        need(admission == dict(request_sha256=request_sha, parent_pid=request['parent_pid'],
                                owned_pid=launch['owned_pid'], worker_pid=claim['worker_pid'],
                                mode=admission.get('mode'), runtime_verified=True, index=index,
                                nonce=request['nonce'], stage=stage), 'Owned admission differs')
        if launch['platform'] == 'posix':
            need(admission['mode'] == 'direct' and claim['worker_pid'] == launch['owned_pid']
                 and launch['parent_identity'] is None and launch['owned_identity'] is None,
                 'POSIX direct ancestry required')
        else:
            parent, child = launch['parent_identity'], launch['owned_identity']
            need(all(isinstance(row, dict) and set(row) == {'pid', 'parent_pid', 'image', 'creation_token'}
                     and type(row['creation_token']) is int and row['creation_token'] > 0 for row in (parent, child))
                 and parent['pid'] == request['parent_pid'] and child['pid'] == launch['owned_pid']
                 and child['parent_pid'] == parent['pid'] and child['creation_token'] >= parent['creation_token']
                 and parent['image'] == study['runtime_identity']['base_executable']
                 and child['image'] in (study['runtime_identity']['base_executable'], study['runtime_identity']['executable']),
                 'Windows parent/launcher ancestry differs')
            need((admission['mode'] == 'direct' and claim['worker_pid'] == launch['owned_pid']) or
                 (admission['mode'] == 'windows-venv' and claim['worker_pid'] != launch['owned_pid']
                  and child['image'] == study['runtime_identity']['executable'] != study['runtime_identity']['base_executable']),
                 'Only the authenticated direct child or one Windows launcher hop is permitted')
        resources = claim['resources']
        need(resources == dict(blas_env={name:'1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                               torch_threads=1, torch_interop_threads=1)
             and type(resources['torch_threads']) is int and type(resources['torch_interop_threads']) is int,
             'Observed worker thread limits differ')
        need(worker == dict(request_sha256=request_sha, artifacts=proofs[stage], launch_receipt_sha256=launch_sha)
             and result == dict(request_sha256=request_sha, protocol_sha256=study['protocol_sha256'], status='complete',
                                detail=dict(worker_receipt_sha256=original.digest(worker))),
             'Failed, partial, replayed or altered terminal attempt cannot pass')
        previous = original.digest(result)
    return dict(owned_execution_verified=True, completed_owned_stages=len(STAGES),
                observed_data_workers=1, observed_blas_threads=1, observed_torch_threads=1)


def verify_completed():
    """Authorized local verification only; outputs aggregate flags, never arrays."""
    from tools import clean_pff_adaptation_development_v1 as run
    from tools import clean_pff_adaptation_development_operator_v1 as operator
    ctx = run.context()
    bundle = bundle_from_run()
    result = verify_evidence(bundle, ctx)
    for stage in STAGES:
        need(run.verify_stage(stage) == bundle['stage_proofs'][stage], 'Actual stage proof differs')
    policy.authenticate_models(policy.audit_lineage(ctx['lineage']), ctx['model_paths'], run.ROOT)
    for name, identity in bundle['protocol']['input_hashes'].items():
        target = run.inner_store._no_links(run.ROOT / name)
        need(target.is_relative_to(run.ROOT) and run.storage.file_hash(target) == identity,
             'Fixed handoff input bytes changed')
    status = operator.status()
    need(status.get('complete') is True and status.get('blocked') is False,
         'Every bounded owned-worker attempt must be complete and verified')
    attempts = [{kind: operator._read(operator.file(index, kind + '.json'))
                 for kind in ('request', 'result', 'worker', 'launch', 'claim')} for index in range(len(STAGES))]
    ownership = verify_owned_history(attempts, bundle['protocol'], bundle['start_markers'], bundle['stage_proofs'])
    need(bundle_from_run() == bundle and run.context() == ctx, 'Artifacts changed during independent verification')
    return dict(result, local_artifact_bytes_verified=True, **ownership)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(verify_completed(), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # No provider records, paths, fitted coefficients or parser data in output.
        print(json.dumps(dict(status='verification_failed', exception_type=type(error).__name__)))
        raise SystemExit(2)
