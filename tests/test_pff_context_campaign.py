"""Independent bindings for the context runner, with synthetic fixture bytes."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from possession.io import sha256, write_json
from tools import pff_context_campaign as campaign


class PFFContextCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'study'
        self.parent = self.root / 'parent'
        real_root = campaign.ROOT
        self.reservation = json.loads((real_root / 'models/pff-protocol.json').read_text())
        baseline = dict.fromkeys(campaign.original.METRICS, .6)
        parent = dict.fromkeys(campaign.original.METRICS, .65)
        parent['dead_precision'] = .59
        self.old = [dict(match_id=mid, condition='native', baseline_sha256=campaign.original.BASELINE_SHA256,
                         candidate_sha256=campaign.validation.PARENT_SHA256,
                         baseline=baseline.copy(), candidate=parent.copy(), reference_dead_segments=100,
                         reference_short_spells=100) for mid in self.reservation['split']['development']]
        write_json(self.root / 'models/pff-precision-research.json',
                   {'development': {'summary': {'selection': {'eligible_for_final_audit': False}}}, 'final': None})
        (self.root / 'tools').mkdir()
        (self.root / 'tools/pff_context_prepare.py').write_text('synthetic producer')
        for target, value in [('ROOT', self.root), ('PARENT', self.parent)]:
            active = patch.object(campaign, target, value)
            active.start(); self.addCleanup(active.stop)
        for target, value in [('context', (self.reservation, self.old)), ('sources', {'synthetic': 'a' * 64})]:
            active = patch.object(campaign, target, return_value=value)
            active.start(); self.addCleanup(active.stop)
        with patch('tools.verify_pff_precision.verify_bundle'):
            self.plan = campaign.initialize(self.out)
        records = []
        for mid in self.plan['training_matches']:
            folder = self.out / 'training-probabilities' / str(mid)
            folder.mkdir(parents=True)
            np.save(folder / 'parent_probabilities.npy', np.tile([.2, .3, .5], (3, 1)))
            write_json(folder / 'manifest.json', {'match_id': mid, 'synthetic': True})
            records.append({'match_id': mid, 'manifest_sha256': sha256(folder / 'manifest.json'),
                            'probability_sha256': sha256(folder / 'parent_probabilities.npy'), 'rows': 3})
        self.complete = {'records': records, 'parent_candidate_sha256': campaign.validation.PARENT_SHA256,
                         'protocol_sha256': self.reservation['protocol_sha256'],
                         'preparation_source_sha256': sha256(self.root / 'tools/pff_context_prepare.py'),
                         'references_read': False}
        write_json(self.out / 'training-probabilities/complete.json', self.complete)

    def fake_head(self, plan):
        return {'schema_version': 1, 'kind': campaign.correction.POLICY['kind'],
                'policy': copy.deepcopy(campaign.correction.POLICY),
                'feature_names': list(campaign.correction.FEATURE_NAMES),
                'mean': [0.] * 60, 'scale': [1.] * 60, 'input_weight': np.zeros((32, 60)).tolist(),
                'input_bias': [0.] * 32, 'output_weight': [0.] * 32, 'output_bias': 0.,
                'metadata': {'parent_sha256': campaign.validation.PARENT_SHA256,
                             'reference_fields_used_for_inference': False,
                             'protocol_sha256': plan['protocol_sha256'],
                             'training_matches': sorted(plan['training_matches']),
                             'held_out_matches': sorted(plan['held_out_matches'])}, 'training': {}}

    def finish_fit(self, fail=False):
        self.seen = []
        def train(items, **kwargs):
            self.assertTrue((self.out / 'fit-access.json').exists())
            if fail:
                raise RuntimeError('synthetic interrupted fit')
            for item in items:
                self.seen.append(item['match'])
            return self.fake_head(self.plan)
        def reference(source, mid):
            self.assertTrue((self.out / 'fit-access.json').exists())
            self.assertIn(mid, self.plan['training_matches'])
            return np.array(list('HAD')), {'synthetic': True}
        with patch.object(campaign, 'load_model', return_value={}), \
             patch.object(campaign.preparation, 'prepare_match', side_effect=lambda s, o, mid, p, m: {'match_id': mid}), \
             patch.object(campaign.prior, 'item_at', side_effect=lambda *args: {'z': np.zeros((3, 58)), 'fps': 25, 'blocks': [(1, 0, 2)]}), \
             patch.object(campaign.prior, 'reference', side_effect=reference), \
             patch.object(campaign.correction, 'fit', side_effect=train):
            self.study = campaign.fit(self.out)
        return self.study

    def test_plan_exists_before_fit_and_rejects_source_policy_membership_drift(self):
        self.assertFalse((self.out / 'fit-access.json').exists())
        self.assertEqual(campaign.training_plan(self.out), self.plan)
        for field, value in [('source_hashes', {'changed': 'b' * 64}), ('training_matches', self.plan['training_matches'][:-1]),
                             ('fit_policy', {}), ('blend_grid', [0., 1.]), ('parent_sha256', 'b' * 64)]:
            changed = copy.deepcopy(self.plan); changed[field] = value
            body = dict(changed); body.pop('protocol_sha256'); changed['protocol_sha256'] = campaign.original.digest(body)
            write_json(self.out / 'training-plan.json', changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                campaign.training_plan(self.out)
        write_json(self.out / 'training-plan.json', self.plan)

    def test_fit_requires_all_forty_prepared_matches_before_reference_access(self):
        changed = copy.deepcopy(self.complete); changed['records'].pop()
        write_json(self.out / 'training-probabilities/complete.json', changed)
        with patch.object(campaign.prior, 'reference') as reference, self.assertRaises(ValueError):
            campaign.fit(self.out)
        reference.assert_not_called()
        self.assertFalse((self.out / 'fit-access.json').exists())

    def test_last_training_probability_tamper_fails_before_any_reference(self):
        mid = self.plan['training_matches'][-1]
        (self.out / 'training-probabilities' / str(mid) / 'parent_probabilities.npy').write_bytes(b'changed')
        with patch.object(campaign, 'load_model', return_value={}), \
             patch.object(campaign.preparation, 'prepare_match', return_value={}), \
             patch.object(campaign.prior, 'reference') as reference:
            with self.assertRaises(ValueError): campaign.fit(self.out)
            reference.assert_not_called()
        self.assertFalse((self.out / 'fit-access.json').exists())

    def test_failed_fit_is_consumed_and_cannot_be_retried(self):
        with self.assertRaises(RuntimeError): self.finish_fit(fail=True)
        self.assertTrue((self.out / 'fit-access.json').exists())
        with patch.object(campaign.correction, 'fit') as fit, self.assertRaises(ValueError):
            campaign.fit(self.out)
        fit.assert_not_called()

    def test_fit_records_exact_training_membership_and_authenticates(self):
        self.finish_fit()
        self.assertEqual(self.seen, self.plan['training_matches'])
        self.assertEqual(len(self.seen), 40)
        self.assertEqual(campaign.authenticate(self.out)[0], self.study)
        self.assertFalse(campaign.read(self.out / 'fit.json')['held_out_references_read'])

    def test_fitted_head_bytes_and_independent_metadata_drift_fail(self):
        self.finish_fit()
        head = campaign.read(self.out / 'context-head.json')
        changed = copy.deepcopy(head); changed['metadata']['training_matches'].pop()
        with patch.object(campaign.correction, 'validate_head', return_value=changed), self.assertRaises(ValueError):
            campaign.authenticate(self.out)
        (self.out / 'context-head.json').write_bytes(b'{}')
        with self.assertRaises(ValueError): campaign.authenticate(self.out)

    def test_final_entrypoints_and_worker_block_before_any_data_access(self):
        self.finish_fit()
        mid = self.reservation['split']['final'][0]
        with patch.object(campaign, 'require_final', side_effect=ValueError('locked')), \
             patch.object(campaign, 'ProcessPoolExecutor') as pool, \
             patch.object(campaign.prior, 'item_at') as item, \
             patch.object(campaign.prior, 'reference') as reference:
            for action in (lambda: campaign.prepare_final(self.out, 1), lambda: campaign.predict(self.out, 'final', 1),
                           lambda: campaign.score(self.out, 'final'), lambda: campaign.predict_one(self.out, 'final', mid, self.study['candidates'])):
                with self.assertRaises(ValueError): action()
            pool.assert_not_called(); item.assert_not_called(); reference.assert_not_called()

    def test_no_development_summary_cannot_open_final(self):
        self.finish_fit()
        with patch.object(campaign.prior, 'item_at') as item, patch.object(campaign.prior, 'reference') as reference:
            with self.assertRaises(FileNotFoundError): campaign.require_final(self.out)
            item.assert_not_called(); reference.assert_not_called()

    def test_zero_blend_parent_mismatch_fails_before_prediction_publication(self):
        self.finish_fit()
        mid = self.reservation['split']['development'][0]
        folder = self.parent / 'development' / str(mid); folder.mkdir(parents=True)
        np.save(folder / 'candidate.npy', np.array(list('HAD')))
        with patch.object(campaign.prior, 'item_at', return_value={}), patch.object(campaign, 'load_model', return_value={}), \
             patch.object(campaign.precision, 'probabilities_from_cache', return_value=np.ones((3, 3)) / 3), \
             patch.object(campaign.correction, 'residuals', return_value=np.zeros(3)), \
             patch.object(campaign.precision, 'decode_probabilities', return_value=(np.array(list('DAD')), None)):
            with self.assertRaises(ValueError): campaign.predict_one(self.out, 'development', mid, self.study['candidates'])
        self.assertFalse((self.out / 'development' / str(mid) / 'parent.npy').exists())

    def test_consumed_scoring_blocks_reprediction_before_workers_or_references(self):
        self.finish_fit()
        campaign.begin_access(self.out / 'development-reference-access.json', {'consumed': True})
        with patch.object(campaign, 'ProcessPoolExecutor') as pool, patch.object(campaign.prior, 'reference') as reference:
            with self.assertRaises(ValueError): campaign.score(self.out, 'development')
            with self.assertRaises(ValueError): campaign.predict(self.out, 'development', 1)
            pool.assert_not_called(); reference.assert_not_called()

    def completed_development(self):
        self.finish_fit()
        records = []
        names = ['baseline', 'parent', *(d['candidate_id'] for d in self.study['candidates'])]
        for mid in sorted(self.reservation['split']['development']):
            folder = self.out / 'development' / str(mid); folder.mkdir(parents=True)
            for name in names:
                np.save(folder / (name + '.npy'), np.array(list('HAD')))
            manifest = {'synthetic': True, 'match_id': mid}
            write_json(self.parent / 'cache' / str(mid) / 'complete.json', manifest)
            records.append({'match_id': mid, 'prediction_sha256': {name: sha256(folder / (name + '.npy')) for name in names},
                            'observation_manifest': manifest})
        frozen = {'part': 'development', 'records': records, 'candidates': self.study['candidates'],
                  'protocol_sha256': self.study['protocol_sha256'], 'source_hashes': campaign.sources(),
                  'references_accessed_in_this_study': False, 'development_previously_consumed': True}
        write_json(self.out / 'development-predictions-before-references.json', frozen)
        def reference(*args):
            self.assertTrue((self.out / 'development-reference-access.json').exists())
            return np.array(list('HAD')), {'synthetic': True}
        def score(out, part, mid, name, y, labels, blocks):
            if name == 'baseline':
                values = self.old[0]['baseline'].copy()
            elif name in ('parent', 'context-00'):
                values = self.old[0]['candidate'].copy()
            else:
                values = dict.fromkeys(campaign.original.METRICS, .65); values['dead_precision'] = .605
            detail = {'match': str(mid), 'candidate': name, 'regime': 'native', 'spellF1': values['spell'],
                      'macroF1': values['macro'], 'dead_precision': values['dead_precision'], 'dead_recall': values['dead_recall'],
                      'temporal_diagnostics': {'dead_segments': {'n_gt': 100, 'f1': values['dead_segment']},
                          'dead_onsets': {'f1': values['dead_boundary']}, 'dead_offsets': {'f1': values['dead_boundary']},
                          'short_possessions': {'n_gt': 100, 'recall': values['short_recall']}},
                      'duration_precision_recall': {'bins': {'<0.4s': {'n_pred': 100, 'n_matched_pred': round(100 * values['short_precision'])},
                                                           '0.4-1s': {'n_pred': 0, 'n_matched_pred': 0}, '1-2s': {'n_pred': 0, 'n_matched_pred': 0}}}}
            return detail, values
        with patch.object(campaign.prior, 'item_at', return_value={'blocks': [(1, 0, 2)]}), \
             patch.object(campaign.prior, 'reference', side_effect=reference), \
             patch.object(campaign.prior, 'score_one', side_effect=score), contextlib.redirect_stdout(io.StringIO()):
            return campaign.score(self.out, 'development')

    def test_final_guard_recomputes_complete_development_instead_of_trusting_flag(self):
        result = self.completed_development()
        self.assertTrue(campaign.require_final(self.out)['eligible_for_final_audit'])
        for rows in result['records'].values():
            for row in rows:
                row['candidate']['dead_precision'] = .59
        write_json(self.out / 'development-summary.json', result)
        with self.assertRaises(ValueError): campaign.require_final(self.out)

    def test_final_guard_rejects_reference_marker_and_last_prediction_tampering(self):
        self.completed_development()
        path = self.out / 'development-reference-access.json'; original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaises(ValueError): campaign.require_final(self.out)
        path.write_bytes(original)
        mid = self.reservation['split']['development'][-1]
        (self.out / 'development' / str(mid) / 'context-05.npy').write_bytes(b'changed')
        with self.assertRaises(ValueError): campaign.require_final(self.out)

    def test_final_guard_derives_selected_metrics_from_complete_paired_details(self):
        result = self.completed_development()
        result['details'][0]['spellF1'] = .123
        write_json(self.out / 'development-summary.json', result)
        with self.assertRaises(ValueError): campaign.require_final(self.out)


if __name__ == '__main__':
    unittest.main()
