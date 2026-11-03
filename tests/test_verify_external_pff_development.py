"""Synthetic aggregate tampering; no provider observations or labels are read."""
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import verify_external_pff_development as verifier


@lru_cache(maxsize=1)
def context():
    return verifier._context()


def summary_fixture():
    ctx = copy.deepcopy(context())
    old = ctx['old_bundle']['development']['summary']
    summary = {'records': copy.deepcopy(old['records']), 'parent_records': copy.deepcopy(old['records']),
               'reference_support': copy.deepcopy(old['reference_support']), 'details': []}
    for row in summary['records']:
        row['candidate_sha256'] = verifier.CANDIDATE_SHA
    for detail in old['details']:
        if detail['candidate'] == 'baseline':
            summary['details'].append(copy.deepcopy(detail))
        else:
            parent = copy.deepcopy(detail); parent['candidate'] = 'parent'
            summary['details'].extend([parent, copy.deepcopy(detail)])
    return summary, ctx


def serialized(value, newline=True, crlf=False):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + ('\n' if newline else '')
    if crlf:
        text = text.replace('\n', '\r\n')
    return hashlib.sha256(text.encode()).hexdigest()


def completed_fixture(passing=False, crlf=False):
    summary, ctx = summary_fixture()
    study = verifier.campaign.build_protocol(ctx)
    if passing:
        for row in summary['records']:
            row['candidate']['dead_precision'] = row['baseline']['dead_precision'] + .003
            detail = next(d for d in summary['details'] if d['match'] == str(row['match_id']) and d['candidate'] == 'candidate')
            detail['dead_precision'] = row['candidate']['dead_precision']
    old_predictions = {r['match_id']: r for r in ctx['old_bundle']['development']['prediction_freeze']['records']}
    rows = []
    for mid in verifier.campaign.IDS:
        old = old_predictions[mid]
        rows.append({'match_id': mid, 'protocol_sha256': study['protocol_sha256'],
                     'candidate_sha256': verifier.CANDIDATE_SHA,
                     'cache_manifest_sha256': verifier.campaign.CACHE_HASHES[str(mid)],
                     'prediction_sha256': {'baseline': old['baseline_prediction_sha256'],
                                           'parent': old['candidate_prediction_sha256'],
                                           'candidate': hashlib.sha256(str(mid).encode()).hexdigest()}})
    frozen = verifier.campaign.freeze_value(study, rows)
    frozen_hash = serialized(frozen, crlf=crlf)
    access = verifier._expected_access(study, frozen_hash)
    gate = verifier.precision.development_gate(summary['records'], ctx['prior_records'], ctx['reservation'])
    with patch.object(verifier.policy, 'authenticate_models', return_value={'model_bytes_authenticated': True, 'n_models': 2}):
        screen = verifier.policy.screen_development(summary['records'], ctx['prior_records'], ctx['reservation'],
            ctx['lineage'], [], verifier.ROOT, model_paths=ctx['model_paths'], artifact_root=verifier.ROOT)
    summary.update(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
                   candidate_sha256=verifier.CANDIDATE_SHA, source_hashes=study['source_hashes'], gate=gate,
                   evidence_screen=screen, development_gate_passed=gate['numeric_gate_passed'],
                   prediction_freeze_sha256=frozen_hash, reference_access_sha256=serialized(access, False, crlf),
                   reference_quality='PFF provider-event-derived silver; previously used development', **verifier.campaign.FLAGS)
    run = dict(schema_version=1, status='complete', stage_completed='score', next_stage=None,
               protocol_sha256=study['protocol_sha256'], protocol_file_sha256=serialized(study, crlf=crlf),
               candidate_sha256=verifier.CANDIDATE_SHA, source_hashes=study['source_hashes'],
               development_gate_passed=gate['numeric_gate_passed'],
               development_summary_sha256=serialized(summary, crlf=crlf), **verifier.campaign.FLAGS)
    artifacts = {'initialize': {'protocol.json': serialized(study, crlf=crlf)},
                 'freeze': {'predictions-before-references.json': frozen_hash},
                 'score': {'development-summary.json': serialized(summary, crlf=crlf),
                           'reference-access.json': serialized(access, False, crlf)}}
    for row in rows:
        mid = row['match_id']
        artifacts[f'predict-{mid}'] = {f'predictions/{mid}/{name}.npy': value for name, value in row['prediction_sha256'].items()}
        artifacts[f'predict-{mid}'][f'predictions/{mid}/complete.json'] = serialized(row, crlf=crlf)
    for detail in summary['details']:
        prefix = f"scores/{detail['match']}/{detail['candidate']}"
        artifacts['score'][prefix + '/predictions.csv'] = detail['prediction_sha256']
        artifacts['score'][prefix + '/metrics.json'] = 'f' * 64
    proofs = {stage: {'stage': stage, 'protocol_sha256': study['protocol_sha256'], 'artifacts': artifacts[stage]}
              for stage in verifier.campaign.STAGES}
    state = {'protocol_sha256': study['protocol_sha256'],
             'completed': {stage: serialized(proof, crlf=crlf) for stage, proof in proofs.items()}}
    markers = {stage: {'stage': stage, 'protocol_sha256': study['protocol_sha256'],
                       'candidate_sha256': verifier.CANDIDATE_SHA, 'source_hashes': study['source_hashes'], 'pid': 123}
               for stage in verifier.campaign.STAGES}
    return {'schema_version': 1, 'protocol': study, 'state': state, 'stage_proofs': proofs, 'start_markers': markers,
            'run_summary': run, 'development': {'summary': summary, 'prediction_freeze': frozen, 'reference_access': access},
            'lineage': ctx['lineage'], 'model_paths': ctx['model_paths'],
            'parent_research_sha256': ctx['parent_research_sha256'], 'rebuild_research_sha256': ctx['rebuild_research_sha256'],
            'weights_included': False, 'fit_performed': False, 'final_artifacts_present': False,
            'final_access_authorized': False, 'promoted': False, 'submission_eligible': False}, ctx


class CompletedEvidenceTests(unittest.TestCase):
    def verify(self, bundle, ctx):
        with patch.object(verifier, '_context', return_value=ctx):
            return verifier.verify_bundle(bundle)

    def test_failed_numeric_comparison_is_complete_but_ineligible(self):
        bundle, ctx = completed_fixture()
        result = self.verify(bundle, ctx)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['development_gate_passed'])
        self.assertFalse(result['final_access_authorized'])

    def test_numeric_pass_cannot_unlock_final_or_promotion(self):
        bundle, ctx = completed_fixture(passing=True)
        result = self.verify(bundle, ctx)
        self.assertTrue(result['development_gate_passed'])
        self.assertFalse(result['final_access_authorized'])
        self.assertFalse(result['promoted'])
        self.assertFalse(result['submission_eligible'])
        self.assertFalse(result['local_artifact_bytes_verified'])

    def test_windows_and_posix_serializations_preserve_canonical_identity(self):
        for crlf in (False, True):
            bundle, ctx = completed_fixture(crlf=crlf)
            self.verify(bundle, ctx)
            self.assertNotEqual(bundle['protocol']['protocol_sha256'], bundle['run_summary']['protocol_file_sha256'])

    def test_gate_booleans_cannot_override_recomputation(self):
        for field in ('gate', 'development_gate_passed', 'screen'):
            bundle, ctx = completed_fixture()
            summary = bundle['development']['summary']
            if field == 'gate':
                summary['gate']['numeric_gate_passed'] = True
            elif field == 'screen':
                summary['evidence_screen']['ready_for_independent_final_review'] = True
            else:
                summary[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(bundle, ctx)

    def test_candidate_or_training_lineage_cannot_be_swapped(self):
        for kind in ('candidate', 'lineage', 'fit'):
            bundle, ctx = completed_fixture()
            if kind == 'candidate':
                bundle['protocol']['candidate_sha256'] = 'a' * 64
            elif kind == 'lineage':
                bundle['lineage']['artifacts'][0]['scope'] = 'challenge'
            else:
                bundle['fit_performed'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.verify(bundle, ctx)

    def test_partial_prediction_freeze_and_early_reference_claim_rejected(self):
        for mode in ('partial', 'early', 'access'):
            bundle, ctx = completed_fixture()
            frozen = bundle['development']['prediction_freeze']
            if mode == 'partial':
                frozen['records'].pop()
            elif mode == 'early':
                frozen['references_accessed_in_this_study'] = True
            else:
                bundle['development']['reference_access']['prediction_freeze_sha256'] = 'a' * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.verify(bundle, ctx)

    def test_historical_prediction_controls_cannot_be_rehashed_into_validity(self):
        for name in ('baseline', 'parent'):
            bundle, ctx = completed_fixture()
            bundle['development']['prediction_freeze']['records'][0]['prediction_sha256'][name] = 'a' * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.verify(bundle, ctx)

    def test_final_and_false_evidence_flags_rejected(self):
        for field in ('final_artifacts_present', 'promoted', 'final_access_authorized', 'submission_eligible', 'weights_included'):
            bundle, ctx = completed_fixture(); bundle[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(bundle, ctx)

    def test_stage_artifacts_and_completion_order_are_bound(self):
        for mode in ('proof', 'missing', 'marker', 'csv'):
            bundle, ctx = completed_fixture()
            if mode == 'proof':
                bundle['stage_proofs']['initialize']['artifacts']['protocol.json'] = 'a' * 64
            elif mode == 'missing':
                del bundle['state']['completed']['freeze']
            elif mode == 'marker':
                bundle['start_markers']['score']['candidate_sha256'] = 'a' * 64
            else:
                key = next(k for k in bundle['stage_proofs']['score']['artifacts'] if k.endswith('predictions.csv'))
                bundle['stage_proofs']['score']['artifacts'][key] = 'a' * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.verify(bundle, ctx)

    def test_nonfinite_scores_are_rejected(self):
        bundle, ctx = completed_fixture(); bundle['development']['summary']['records'][0]['candidate']['spell'] = float('nan')
        with self.assertRaises(ValueError):
            self.verify(bundle, ctx)

    def test_local_artifact_bytes_are_checked_without_reading_model_contents(self):
        bundle, ctx = completed_fixture(); out = Path('synthetic-run')
        with patch.object(verifier, 'verify_bundle', return_value={}), \
             patch.object(verifier, 'bundle_from_run', return_value=bundle), \
             patch.object(verifier.campaign, 'context', return_value=ctx), \
             patch.object(verifier.policy, 'authenticate_models', return_value={}) as models, \
             patch.object(verifier.campaign, 'checked_hash') as checked, \
             patch.object(verifier, 'sha256', return_value=bundle['run_summary']['protocol_file_sha256']):
            result = verifier.verify_local(bundle, out)
            self.assertTrue(result['local_artifact_bytes_verified'])
            models.assert_called_once()
            self.assertGreater(checked.call_count, 12 * 3)
            checked.side_effect = ValueError('changed bytes')
            with self.assertRaises(ValueError):
                verifier.verify_local(bundle, out)


class IncompleteStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.bundle, self.ctx = completed_fixture()

    def write(self, relative, value):
        path = self.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')

    def stage(self, count):
        study = self.bundle['protocol']
        stages = verifier.campaign.STAGES[:count]
        state = {'protocol_sha256': study['protocol_sha256'],
                 'completed': {name: self.bundle['state']['completed'][name] for name in stages}}
        self.write('protocol.json', study); self.write('state.json', state)
        for name in stages:
            self.write('stages/' + name + '.json', self.bundle['stage_proofs'][name])
            self.write(name + '-started.json', self.bundle['start_markers'][name])
            if name.startswith('predict-'):
                mid = int(name[8:])
                row = next(r for r in self.bundle['development']['prediction_freeze']['records'] if r['match_id'] == mid)
                self.write(f'predictions/{mid}/complete.json', row)
        if 'freeze' in stages:
            self.write('predictions-before-references.json', self.bundle['development']['prediction_freeze'])
        return study, state

    def inspect(self, study, state):
        with patch.object(verifier.campaign, 'context', return_value=self.ctx), \
             patch.object(verifier.campaign, 'authenticate', return_value=(study, state, self.ctx)):
            return verifier.inspect_run(self.out)

    def test_incomplete_status_has_no_invented_metrics(self):
        study, state = self.stage(2)
        result = self.inspect(study, state)
        self.assertEqual(result['status'], 'in_progress')
        self.assertEqual(result['completed_stages'], 2)
        self.assertTrue(result['continuation_allowed'])
        self.assertIsNone(result['development_gate_passed'])
        self.assertFalse(result['aggregate_gates_recomputed'])
        self.assertFalse(result['final_evaluated'])
        with self.assertRaises(ValueError):
            verifier.bundle_from_run(self.out)

    def test_incomplete_reference_attempt_stays_consumed(self):
        study, state = self.stage(14)
        self.write('score-started.json', self.bundle['start_markers']['score'])
        access = verifier._expected_access(study, verifier.sha256(self.out / 'predictions-before-references.json'))
        self.write('reference-access.json', access)
        result = self.inspect(study, state)
        self.assertEqual(result['status'], 'consumed_incomplete')
        self.assertTrue(result['reference_access_consumed'])
        self.assertFalse(result['continuation_allowed'])
        self.assertIsNone(result['development_gate_passed'])

    def test_reference_marker_before_complete_freeze_rejected(self):
        study, state = self.stage(1)
        self.write('reference-access.json', self.bundle['development']['reference_access'])
        with self.assertRaises(ValueError):
            self.inspect(study, state)

    def test_unreported_later_attempt_rejected(self):
        study, state = self.stage(1)
        self.write('score-started.json', self.bundle['start_markers']['score'])
        with self.assertRaises(ValueError):
            self.inspect(study, state)

    def test_foreign_fixture_directory_rejected_without_opening_records(self):
        study, state = self.stage(1)
        (self.out / 'predictions/3814').mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.inspect(study, state)

    def test_final_or_fit_activity_rejected(self):
        for name in ('final-observations', 'fit-access.json'):
            with self.subTest(name=name):
                path = self.out / name; path.write_text('Must not be read.', encoding='utf-8')
                self.assertTrue(verifier._forbidden_activity(self.out))
                path.unlink()


class DiagnosticBindingTests(unittest.TestCase):
    def test_all_three_models_are_bound(self):
        summary, ctx = summary_fixture()
        details = verifier._verify_details(summary, ctx)
        self.assertEqual(len(details), 36)

    def test_metric_edits_cannot_be_hidden_in_declared_records(self):
        for name in ('baseline', 'parent', 'candidate'):
            summary, ctx = summary_fixture()
            detail = next(d for d in summary['details'] if d['candidate'] == name)
            detail['dead_precision'] -= .01
            with self.subTest(name=name), self.assertRaises(ValueError):
                verifier._verify_details(summary, ctx)

    def test_both_historical_controls_are_pinned(self):
        for name in ('baseline', 'parent'):
            summary, ctx = summary_fixture()
            detail = next(d for d in summary['details'] if d['candidate'] == name)
            detail['prediction_sha256'] = 'a' * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                verifier._verify_details(summary, ctx)

    def test_reference_support_mismatch_rejected(self):
        for mode in ('global', 'record', 'diagnostic'):
            summary, ctx = summary_fixture()
            if mode == 'global':
                mid = next(iter(summary['reference_support']))
                summary['reference_support'][mid]['class_counts']['D'] += 1
            elif mode == 'record':
                summary['records'][0]['reference_dead_segments'] += 1
            else:
                summary['details'][0]['temporal_diagnostics']['dead_segments']['n_gt'] += 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verifier._verify_details(summary, ctx)

    def test_missing_duplicate_foreign_and_reordered_cohort_rejected(self):
        for mode in ('missing', 'duplicate', 'foreign', 'reordered'):
            summary, ctx = summary_fixture()
            if mode == 'missing':
                summary['details'].pop()
            elif mode == 'duplicate':
                summary['details'].append(copy.deepcopy(summary['details'][0]))
            elif mode == 'foreign':
                summary['details'][0]['match'] = '3814'
            else:
                summary['records'].reverse()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verifier._verify_details(summary, ctx)

    def test_candidate_identity_and_condition_rejected(self):
        for key, value in (('candidate_sha256', 'a' * 64), ('baseline_sha256', 'b' * 64), ('condition', 'camera')):
            summary, ctx = summary_fixture(); summary['records'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier._verify_details(summary, ctx)


if __name__ == '__main__':
    unittest.main()
