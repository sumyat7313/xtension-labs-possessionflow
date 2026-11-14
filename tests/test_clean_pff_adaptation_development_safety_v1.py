"""Independent synthetic candidate/aggregate mutations; no provider records."""
import copy
import hashlib
from functools import lru_cache
import unittest
from unittest.mock import patch

from tools import verify_clean_pff_adaptation_development_v1 as verify
from tools import verify_external_pff_development as prior_verify

CANDIDATE = hashlib.sha256(b'invented selected clean refit for tests only').hexdigest()


@lru_cache(maxsize=1)
def anchors():
    # This authenticates published aggregate controls and code only. It never
    # enters prior_verify.verify_local/context paths that read fitted artifacts.
    return prior_verify._context()


def fixture(precision_gain=.003):
    ctx = copy.deepcopy(anchors())
    old = ctx['old_bundle']['development']['summary']
    summary = dict(records=copy.deepcopy(old['records']),
                   parent_records=copy.deepcopy(old['records']),
                   reference_support=copy.deepcopy(old['reference_support']), details=[])
    for detail in old['details']:
        if detail['candidate'] == 'baseline':
            summary['details'].append(copy.deepcopy(detail))
        else:
            parent = copy.deepcopy(detail)
            parent['candidate'] = 'parent'
            summary['details'].extend((parent, copy.deepcopy(detail)))
    for row in summary['records']:
        row['candidate_sha256'] = CANDIDATE
        row['candidate']['dead_precision'] = row['baseline']['dead_precision'] + precision_gain
        detail = next(d for d in summary['details']
                      if d['match'] == str(row['match_id']) and d['candidate'] == 'candidate')
        detail['dead_precision'] = row['candidate']['dead_precision']
    gate = verify.precision.development_gate(summary['records'], ctx['prior_records'], ctx['reservation'])
    summary.update(gate=gate, development_gate_passed=gate['numeric_gate_passed'])
    return summary, ctx


@lru_cache(maxsize=1)
def handoff_fixture():
    from tools import clean_pff_adaptation_development_v1 as run
    from tools import clean_pff_adaptation_protocol as adaptation
    from tools import clean_pff_adaptation_stages as stages
    from tools import clean_pff_adaptation_refit as refit
    ctx = copy.deepcopy(anchors())
    selected = dict(protocol_sha256=stages.SNAPSHOT_PROTOCOL_SHA256, selected_candidate_id='A',
                    inner_numerical_passed=True, **adaptation.FLAGS)
    inner = dict(selection=selected, selection_sha256=verify.original.digest(selected),
                 artifacts_verified=True, lifecycle_complete=True,
                 protocol_sha256=stages.SNAPSHOT_PROTOCOL_SHA256)
    marker = dict(schema_version=1, inner_proof_sha256=verify.original.digest(inner), inner_proof=inner,
                  selected_candidate_id='A', source_hashes=refit._sources(),
                  runtime_versions=stages.runtime_versions(), recipe=adaptation.recipe(),
                  fitting_fixture_ids=list(adaptation.fit_ids('refit')), **refit.FLAGS)
    complete = dict(schema_version=1, selected_candidate_id='A', refit_complete=True, fitting_fixtures=47,
                    inner_proof_sha256=verify.original.digest(inner), model_sha256=CANDIDATE,
                    report_sha256='a'*64, attempt_sha256=verify.serialized_sha(marker), **refit.FLAGS)
    graph, bindings = run.connected_lineage(ctx, complete, marker)
    inputs = {f'{verify.REFIT}/attempt.json': complete['attempt_sha256'],
              f'{verify.REFIT}/model.json': CANDIDATE, f'{verify.REFIT}/report.json': complete['report_sha256'],
              f'{verify.REFIT}/complete.json': verify.serialized_sha(complete),
              f'{run.INNER}/manifest.json': 'b'*64,
              'runs/clean-pff-adaptation-operator-v2/manifest.json': 'c'*64,
              'models/pff-research.json': ctx['parent_research_sha256'],
              'models/external-rebuild-001-research.json': ctx['rebuild_research_sha256'],
              'configs/baselines/v1.2.json': verify.CONFIG_SHA}
    trees = {tree: sorted(name for name in inputs if name.startswith(tree + '/')) for tree in run.AUTHENTICATED_TREES}
    for tree in trees:
        if not trees[tree]:
            trees[tree] = [tree + '/synthetic-proof.json']
            inputs[trees[tree][0]] = 'd'*64
    for identity, name in bindings.items(): inputs[name] = identity
    ctx.update(lineage=graph, model_paths=bindings, candidate_sha256=CANDIDATE,
               refit_proof=complete, refit_marker=marker, inner_proof=inner, input_hashes=inputs,
               authenticated_tree_files=trees)
    return ctx


def seal_bundle(bundle):
    """Bind synthetic aggregate objects; no arrays or frame records are created."""
    objects = {'lineage.json': bundle['lineage'], 'model-bindings.json': bundle['model_paths'],
               'handoff.json': bundle['handoff'], 'predictions-before-references.json': bundle['prediction_freeze'],
               'development-summary.json': bundle['development_summary']}
    for row in bundle['prediction_freeze']['records']:
        mid = row['match_id']
        objects[f'predictions/{mid}/complete.json'] = row
        for name, identity in row['prediction_sha256'].items():
            bundle['stage_proofs'][f'predict-{mid}']['artifacts'][f'predictions/{mid}/{name}.npy'] = identity
    for mid, score in bundle['score_aggregates'].items():
        objects[f'scores/{mid}/aggregate.json'] = score
        objects[f'scores/{mid}/access.json'] = bundle['reference_access'][mid]
        for detail in score['details']:
            bundle['stage_proofs'][f'score-{mid}']['artifacts'][f"details/{mid}/{detail['candidate']}/predictions.csv"] = detail['prediction_sha256']
    for stage, proof in bundle['stage_proofs'].items():
        for name in proof['artifacts']:
            if name in objects: proof['artifacts'][name] = verify.serialized_sha(objects[name])
        bundle['state']['completed'][stage] = verify.serialized_sha(proof)


@lru_cache(maxsize=1)
def complete_fixture():
    from tools import clean_pff_adaptation_development_v1 as run
    summary, _ = fixture()
    ctx = handoff_fixture()
    study = run.build_protocol(ctx)
    rows = []
    for old in ctx['old_bundle']['development']['prediction_freeze']['records']:
        mid = old['match_id']
        rows.append(dict(match_id=mid, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE,
                         cache_manifest_sha256=study['cache_manifest_sha256'][str(mid)],
                         prediction_sha256=dict(candidate='e'*64, baseline=old['baseline_prediction_sha256'],
                                                parent=old['candidate_prediction_sha256'])))
    frozen = run.freeze_value(study, rows)
    freeze_sha = verify.serialized_sha(frozen)
    scores, accesses, details = {}, {}, []
    for index, mid in enumerate(verify.IDS):
        midstr = str(mid)
        access = verify.expected_access(study, mid, freeze_sha)
        part = [d for d in summary['details'] if d['match'] == midstr]
        details.extend(part)
        scores[midstr] = dict(match_id=mid, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE,
                             candidate_record=summary['records'][index], parent_record=summary['parent_records'][index],
                             details=part, reference_support=summary['reference_support'][midstr],
                             prediction_freeze_sha256=freeze_sha, reference_access_sha256=verify.serialized_sha(access))
        accesses[midstr] = access
    summary.update(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
                   candidate_sha256=CANDIDATE, source_hashes=study['source_hashes'], details=details,
                   prediction_freeze_sha256=freeze_sha,
                   reference_access_sha256={mid: verify.serialized_sha(access) for mid, access in accesses.items()},
                   reference_quality='PFF provider-event-derived silver; previously used development',
                   evidence_screen=verify.expected_screen(summary['gate'], ctx), **verify.FLAGS)
    bundle = dict(schema_version=1, protocol=study,
                  state=dict(protocol_sha256=study['protocol_sha256'], completed={}),
                  stage_proofs={stage: dict(stage=stage, protocol_sha256=study['protocol_sha256'],
                                           artifacts={name:'d'*64 for name in verify.expected_stage_files(stage)})
                                for stage in verify.STAGES},
                  start_markers={stage: dict(stage=stage, protocol_sha256=study['protocol_sha256'],
                                            candidate_sha256=CANDIDATE, source_hashes=study['source_hashes'], pid=123)
                                 for stage in verify.STAGES},
                  prediction_freeze=frozen, score_aggregates=scores, reference_access=accesses,
                  lineage=ctx['lineage'], model_paths=ctx['model_paths'],
                  handoff=dict(refit_proof=ctx['refit_proof'], inner_proof=ctx['inner_proof'],
                               input_hashes=ctx['input_hashes'], **verify.FLAGS), development_summary=summary)
    seal_bundle(bundle)
    return bundle, ctx


def owned_fixture(bundle):
    from tools import owned_python_launch as owned
    study = bundle['protocol']
    previous, attempts = '0'*64, []
    for index, stage in enumerate(verify.STAGES):
        request = dict(index=index, stage=stage, nonce=f'{index:032x}', parent_pid=456,
                       protocol_sha256=study['protocol_sha256'], previous_result_sha256=previous)
        request_sha = verify.original.digest(request)
        launch = dict(schema_version=1, platform='windows' if verify.os.name == 'nt' else 'posix',
                      request_sha256=request_sha,
                      receipt_path=owned._normal(verify.ROOT/verify.OUT/'operator'/'attempts'/f'{index:03d}.launch.json'),
                      parent_pid=456, owned_pid=123, runtime=study['runtime_identity'],
                      parent_identity=None, owned_identity=None)
        if launch['platform'] == 'windows':
            launch['parent_identity'] = dict(pid=456, parent_pid=789, image=study['runtime_identity']['base_executable'], creation_token=100)
            launch['owned_identity'] = dict(pid=123, parent_pid=456, image=study['runtime_identity']['base_executable'], creation_token=101)
        launch_sha = verify.original.digest(launch)
        claim = dict(request_sha256=request_sha, worker_pid=123, launch_receipt_sha256=launch_sha,
                     admission=dict(request_sha256=request_sha, parent_pid=456, owned_pid=123, worker_pid=123,
                                    mode='direct', runtime_verified=True, index=index, nonce=request['nonce'], stage=stage),
                     resources=dict(blas_env={name:'1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                                    torch_threads=1, torch_interop_threads=1))
        worker = dict(request_sha256=request_sha, artifacts=bundle['stage_proofs'][stage], launch_receipt_sha256=launch_sha)
        result = dict(request_sha256=request_sha, protocol_sha256=study['protocol_sha256'], status='complete',
                      detail=dict(worker_receipt_sha256=verify.original.digest(worker)))
        attempts.append(dict(request=request, launch=launch, claim=claim, worker=worker, result=result))
        previous = verify.original.digest(result)
    return attempts


class IndependentDevelopmentSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools import clean_pff_adaptation_refit as refit
        from tools import clean_pff_adaptation_development_v1 as run
        # Cache immutable code-only audits for these pure synthetic mutations.
        # Production verifier never replaces its actual-byte checks this way.
        cls.refit_patch = patch.object(refit, '_sources', return_value=refit._sources())
        cls.source_patch = patch.object(run, 'sources', return_value=run.sources())
        cls.refit_patch.start(); cls.source_patch.start()
        cls.addClassCleanup(cls.refit_patch.stop); cls.addClassCleanup(cls.source_patch.stop)

    def test_owned_execution_proofs_reject_partial_replay_wrong_pid_runtime_and_threads(self):
        bundle, _ = complete_fixture()
        attempts = owned_fixture(bundle)
        checked = verify.verify_owned_history(attempts, bundle['protocol'], bundle['start_markers'], bundle['stage_proofs'])
        self.assertEqual(checked['completed_owned_stages'], len(verify.STAGES))
        for mutation in ('partial', 'failed', 'replayed', 'chain', 'worker_pid', 'runtime', 'receipt_path',
                         'admission', 'blas', 'torch', 'bool_count', 'foreign_artifact'):
            changed = copy.deepcopy(attempts)
            first = changed[0]
            if mutation == 'partial': changed.pop()
            elif mutation == 'failed': first['result']['status'] = 'failed'
            elif mutation == 'replayed': changed[1] = copy.deepcopy(first)
            elif mutation == 'chain': changed[1]['request']['previous_result_sha256'] = 'f'*64
            elif mutation == 'worker_pid': first['claim']['worker_pid'] = 321
            elif mutation == 'runtime': first['launch']['runtime'] = {}
            elif mutation == 'receipt_path': first['launch']['receipt_path'] += '.moved'
            elif mutation == 'admission': first['claim']['admission']['runtime_verified'] = False
            elif mutation == 'blas': first['claim']['resources']['blas_env']['OMP_NUM_THREADS'] = '2'
            elif mutation == 'torch': first['claim']['resources']['torch_threads'] = 2
            elif mutation == 'bool_count': first['claim']['resources']['torch_threads'] = True
            else: first['worker']['artifacts']['artifacts']['final/3814.npy'] = 'f'*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.verify_owned_history(changed, bundle['protocol'], bundle['start_markers'], bundle['stage_proofs'])

    def test_complete_synthetic_lifecycle_retains_all_holds(self):
        bundle, ctx = complete_fixture()
        result = verify.verify_evidence(bundle, ctx)
        self.assertTrue(result['development_gate_passed'])
        self.assertTrue(result['aggregate_gates_recomputed'])
        for field in ('promoted', 'submission_eligible', 'final_access_authorized',
                      'ready_for_independent_final_review', 'local_artifact_bytes_verified'):
            self.assertFalse(result[field], field)

    def test_clean_handoff_rejects_selection_roster_sources_and_ancestry_tampering(self):
        ctx = handoff_fixture()
        verify.verify_handoff(ctx)
        for field in ('winner', 'roster', 'source', 'held_ancestor', 'extra_input', 'weights', 'binding'):
            changed = copy.deepcopy(ctx)
            if field == 'winner': changed['inner_proof']['selection']['selected_candidate_id'] = 'B'
            elif field == 'roster': changed['refit_marker']['fitting_fixture_ids'].pop()
            elif field == 'source': changed['refit_marker']['source_hashes']['tools/fake.py'] = 'f'*64
            elif field == 'held_ancestor': changed['lineage']['artifacts'][-1]['parents'].append(verify.original.BASELINE_SHA256)
            elif field == 'extra_input': changed['input_hashes']['../unbounded.json'] = 'f'*64
            elif field == 'weights': changed['candidate_sha256'] = 'f'*64
            else: changed['model_paths'][CANDIDATE] = 'models/default.json'
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.verify_handoff(changed)

    def test_partial_extra_foreign_and_rehashed_stage_proofs_rejected(self):
        original_bundle, ctx = complete_fixture()
        for mutation in ('missing', 'extra', 'marker', 'inventory', 'identity'):
            bundle = copy.deepcopy(original_bundle)
            if mutation == 'missing': bundle['stage_proofs'].pop('freeze')
            elif mutation == 'extra': bundle['stage_proofs']['predict-3814'] = {}
            elif mutation == 'marker': bundle['start_markers']['freeze']['source_hashes'] = {}
            elif mutation == 'inventory': bundle['stage_proofs']['freeze']['artifacts']['final/3814.npy'] = 'f'*64
            else: bundle['stage_proofs']['freeze']['artifacts']['predictions-before-references.json'] = 'f'*64
            for stage, proof in bundle['stage_proofs'].items():
                bundle['state']['completed'][stage] = verify.serialized_sha(proof)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.verify_evidence(bundle, ctx)

    def test_all_predictions_must_precede_access_and_cache_controls_are_fixed(self):
        original_bundle, ctx = complete_fixture()
        for mutation in ('partial', 'early_access', 'cache', 'control', 'reference_access', 'final_access'):
            bundle = copy.deepcopy(original_bundle)
            frozen = bundle['prediction_freeze']
            if mutation == 'partial': frozen['records'].pop()
            elif mutation == 'early_access': frozen['references_accessed_in_this_study'] = True
            elif mutation == 'cache': frozen['records'][0]['cache_manifest_sha256'] = 'f'*64
            elif mutation == 'control': frozen['records'][0]['prediction_sha256']['baseline'] = 'f'*64
            elif mutation == 'reference_access': bundle['reference_access']['3813']['prediction_freeze_sha256'] = 'f'*64
            else: bundle['reference_access']['3813']['final_access_authorized'] = True
            seal_bundle(bundle)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.verify_evidence(bundle, ctx)

    def test_sources_resources_and_false_rights_clearance_fail_closed(self):
        original_bundle, ctx = complete_fixture()
        for mutation in ('source', 'worker', 'blas', 'torch', 'budget', 'fit', 'rights'):
            bundle = copy.deepcopy(original_bundle)
            study = bundle['protocol']
            if mutation == 'source': study['source_hashes'] = {}
            elif mutation in ('worker', 'blas', 'torch'):
                study['limits'][{'worker':'data_workers','blas':'blas_threads','torch':'torch_threads'}[mutation]] = 2
            elif mutation == 'budget': study['candidate_budget'] = 2
            elif mutation == 'fit': study['fitting_permitted'] = True
            else: bundle['development_summary']['evidence_screen']['ready_for_independent_final_review'] = True
            seal_bundle(bundle)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.verify_evidence(bundle, ctx)

    def test_exact_development_ids_reject_final_training_boolean_and_paths(self):
        for match in verify.IDS:
            self.assertEqual(verify.require_match(match), match)
        for match in (*verify.FINAL_IDS, 3812, '3813', True, '../3813', None):
            with self.subTest(match=match), self.assertRaises(ValueError):
                verify.require_match(match)

    def test_original_gate_recomputed_without_claiming_promotion(self):
        summary, ctx = fixture()
        result = verify.recompute_gate(summary, ctx, CANDIDATE)
        self.assertTrue(result['numeric_gate_passed'])
        self.assertFalse(result['promoted'])
        self.assertEqual(set(result['retention_checks']), set(verify.RETAIN))

    def test_sub_margin_precision_cannot_pass(self):
        summary, ctx = fixture(.001999)
        result = verify.recompute_gate(summary, ctx, CANDIDATE)
        self.assertFalse(result['checks']['dead_precision_margin'])
        self.assertFalse(result['numeric_gate_passed'])

    def test_each_original_gain_requires_ninety_percent_retention(self):
        for metric in verify.RETAIN:
            summary, ctx = fixture()
            original_gain = summary['gate']['parent_mean_gains'][metric]
            self.assertGreater(original_gain, 0)
            prior = {row['match_id']: row for row in ctx['prior_records']}
            for row in summary['records']:
                original_fixture_gain = prior[row['match_id']]['candidate'][metric] - row['baseline'][metric]
                row['candidate'][metric] = row['baseline'][metric] + .89 * original_fixture_gain
            gate = verify.precision.development_gate(summary['records'], ctx['prior_records'], ctx['reservation'])
            self.assertFalse(gate['retention_checks'][metric], metric)
            self.assertFalse(gate['numeric_gate_passed'], metric)

    def test_malformed_candidate_rows_or_nonfinite_metrics_rejected(self):
        for change in ('missing', 'duplicate', 'final', 'second_model', 'nan', 'extra_field'):
            summary, ctx = fixture()
            rows = summary['records']
            if change == 'missing': rows.pop()
            elif change == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
            elif change == 'final': rows[0]['match_id'] = min(verify.FINAL_IDS)
            elif change == 'second_model': rows[0]['candidate_sha256'] = 'd' * 64
            elif change == 'nan': rows[0]['candidate']['spell'] = float('nan')
            else: rows[0]['retuned_threshold'] = .1
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify.recompute_gate(summary, ctx, CANDIDATE)

    def test_purported_gate_and_changed_policy_cannot_override_checks(self):
        summary, ctx = fixture(.001)
        for field in ('gate', 'development_gate_passed'):
            altered = copy.deepcopy(summary)
            if field == 'gate': altered['gate']['numeric_gate_passed'] = True
            else: altered[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.recompute_gate(altered, ctx, CANDIDATE)
        with patch.dict(verify.policy.POLICY, minimum_dead_precision_gain=0):
            with self.assertRaises(ValueError): verify.fixed_policy()
        with patch.dict(verify.precision.POLICY, minimum_retained_positive_gain_fraction=.8):
            with self.assertRaises(ValueError): verify.fixed_policy()

    def test_changed_support_or_rehashed_control_details_rejected(self):
        for change in ('support', 'control_score', 'control_hash', 'event_count', 'detail_missing'):
            summary, ctx = fixture()
            if change == 'support': summary['reference_support'] = {}
            elif change == 'control_score':
                summary['records'][0]['baseline']['spell'] += .01
                baseline = next(d for d in summary['details'] if d['candidate'] == 'baseline')
                baseline['spell_f1_mean'] = summary['records'][0]['baseline']['spell']
            elif change == 'control_hash':
                next(d for d in summary['details'] if d['candidate'] == 'parent')['prediction_sha256'] = 'd' * 64
            elif change == 'event_count': summary['records'][0]['reference_dead_segments'] += 1
            else: summary['details'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify.recompute_gate(summary, ctx, CANDIDATE)

    def test_known_held_models_cannot_masquerade_as_refit_candidate(self):
        summary, ctx = fixture()
        for identity in (*verify.policy.PROHIBITED_FIT_MODELS, verify.CLEAN_SHA):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                verify.verify_details(summary, ctx, identity)


if __name__ == '__main__':
    unittest.main()
