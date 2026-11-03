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
from tools import pff_precision_campaign as campaign


class PFFPrecisionCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'study'
        self.parent = self.root / 'parent'
        self.reservation = json.loads((campaign.ROOT / 'models/pff-protocol.json').read_text(encoding='utf-8'))
        self.baseline = dict.fromkeys(campaign.original.METRICS, .6)
        self.parent_metrics = dict.fromkeys(campaign.original.METRICS, .65)
        self.parent_metrics['dead_precision'] = .59
        self.old = [dict(match_id=mid, condition='native', baseline_sha256=campaign.original.BASELINE_SHA256,
                         candidate_sha256=campaign.validation.PARENT_SHA256,
                         baseline=self.baseline.copy(), candidate=self.parent_metrics.copy(),
                         reference_dead_segments=100, reference_short_spells=100)
                    for mid in self.reservation['split']['development']]
        for target, value in [('context', (self.reservation, self.old)), ('sources', {'synthetic': 'a' * 64})]:
            active = patch.object(campaign, target, return_value=value)
            active.start(); self.addCleanup(active.stop)
        active = patch.object(campaign, 'PARENT', self.parent)
        active.start(); self.addCleanup(active.stop)
        self.study = campaign.initialize(self.out)

    def predictions(self):
        records = []
        names = ['baseline', 'parent', *(d['candidate_id'] for d in self.study['candidates'])]
        for mid in sorted(self.reservation['split']['development']):
            folder = self.out / 'development' / str(mid)
            folder.mkdir(parents=True)
            for name in names:
                np.save(folder / (name + '.npy'), np.array(list('HAD')), allow_pickle=False)
            manifest = {'synthetic': True, 'match_id': mid}
            write_json(self.parent / 'cache' / str(mid) / 'complete.json', manifest)
            records.append({'match_id': mid, 'prediction_sha256': {name: sha256(folder / (name + '.npy')) for name in names},
                            'observation_manifest': manifest})
        frozen = {'part': 'development', 'records': records, 'candidates': self.study['candidates'],
                  'protocol_sha256': self.study['protocol_sha256'], 'source_hashes': campaign.sources(),
                  'references_accessed_in_this_study': False, 'development_previously_consumed': True}
        write_json(self.out / 'development-predictions-before-references.json', frozen)
        return frozen

    def fake_score(self, out, part, mid, name, y, labels, blocks):
        if name == 'baseline':
            values = self.baseline.copy()
        elif name in ('parent', 'penalty-00'):
            values = self.parent_metrics.copy()
        else:
            values = dict.fromkeys(campaign.original.METRICS, .65)
            values['dead_precision'] = .605
        details = {'match': str(mid), 'candidate': name, 'regime': 'native',
                   'spellF1': values['spell'], 'macroF1': values['macro'],
                   'dead_precision': values['dead_precision'], 'dead_recall': values['dead_recall'],
                   'temporal_diagnostics': {'dead_segments': {'n_gt': 100, 'f1': values['dead_segment']},
                                            'dead_onsets': {'f1': values['dead_boundary']},
                                            'dead_offsets': {'f1': values['dead_boundary']},
                                            'short_possessions': {'n_gt': 100, 'recall': values['short_recall']}},
                   'duration_precision_recall': {'bins': {
                       '<0.4s': {'n_pred': 100, 'n_matched_pred': round(100 * values['short_precision'])},
                       '0.4-1s': {'n_pred': 0, 'n_matched_pred': 0},
                       '1-2s': {'n_pred': 0, 'n_matched_pred': 0}}}}
        return details, values

    def completed_score(self):
        self.predictions()
        def reference(*args):
            self.assertTrue((self.out / 'development-reference-access.json').exists())
            return np.array(list('HAD')), {'synthetic': True}
        with patch.object(campaign.prior, 'item_at', return_value={'blocks': [(1, 0, 2)]}), \
             patch.object(campaign.prior, 'reference', side_effect=reference), \
             patch.object(campaign.prior, 'score_one', side_effect=self.fake_score), \
             contextlib.redirect_stdout(io.StringIO()):
            return campaign.score(self.out, 'development')

    def test_final_entrypoints_fail_before_any_data_or_worker_access(self):
        with patch.object(campaign, 'require_final', side_effect=ValueError('locked')), \
             patch.object(campaign, 'ProcessPoolExecutor') as pool, \
             patch.object(campaign.prior, 'item_at') as item, \
             patch.object(campaign.prior, 'reference') as reference:
            for action in (lambda: campaign.prepare_final(self.out, 1),
                           lambda: campaign.predict(self.out, 'final', 1),
                           lambda: campaign.score(self.out, 'final')):
                with self.assertRaises(ValueError): action()
            pool.assert_not_called(); item.assert_not_called(); reference.assert_not_called()

    def test_direct_final_worker_cannot_bypass_development_guard(self):
        mid = self.reservation['split']['final'][0]
        with patch.object(campaign, 'require_final', side_effect=ValueError('locked')) as guard, \
             patch.object(campaign.prior, 'item_at') as item:
            with self.assertRaises(ValueError):
                campaign.predict_one(self.out, 'final', mid, self.study['candidates'])
            guard.assert_called_once(); item.assert_not_called()

    def test_direct_worker_rejects_wrong_cohort_and_candidate_grid(self):
        with patch.object(campaign.prior, 'item_at') as item:
            for part, mid, declared in [('development', self.reservation['split']['final'][0], self.study['candidates']),
                                        ('development', self.reservation['split']['development'][0], self.study['candidates'][:-1])]:
                with self.assertRaises(ValueError): campaign.predict_one(self.out, part, mid, declared)
            item.assert_not_called()

    def test_final_guard_recomputes_selection_instead_of_trusting_flag(self):
        result = self.completed_score()
        self.assertTrue(campaign.require_final(self.out)['eligible_for_final_audit'])
        for rows in result['records'].values():
            for row in rows:
                row['candidate']['dead_precision'] = .59
        write_json(self.out / 'development-summary.json', result)
        with self.assertRaises(ValueError): campaign.require_final(self.out)

    def test_final_guard_authenticates_reference_access_and_predictions(self):
        self.completed_score()
        access = self.out / 'development-reference-access.json'
        original = access.read_bytes()
        access.write_bytes(original + b' ')
        with self.assertRaises(ValueError): campaign.require_final(self.out)
        access.write_bytes(original)
        mid = self.reservation['split']['development'][-1]
        (self.out / 'development' / str(mid) / 'penalty-07.npy').write_bytes(b'changed')
        with self.assertRaises(ValueError): campaign.require_final(self.out)

    def test_final_guard_derives_metrics_from_paired_details(self):
        result = self.completed_score()
        result['details'][0]['spellF1'] = .123
        write_json(self.out / 'development-summary.json', result)
        with self.assertRaises(ValueError): campaign.require_final(self.out)

    def test_last_prediction_tamper_rejected_before_first_reference(self):
        self.predictions()
        mid = self.reservation['split']['development'][-1]
        (self.out / 'development' / str(mid) / 'penalty-07.npy').write_bytes(b'altered')
        with patch.object(campaign.prior, 'reference') as reference:
            with self.assertRaises(ValueError): campaign.score(self.out, 'development')
            reference.assert_not_called()
        self.assertFalse((self.out / 'development-reference-access.json').exists())

    def test_prediction_freeze_provenance_and_coverage_are_checked(self):
        frozen = self.predictions()
        for field, value in [('part', 'final'), ('source_hashes', {'changed': 'b' * 64}),
                             ('protocol_sha256', 'b' * 64), ('references_accessed_in_this_study', True),
                             ('records', frozen['records'][:-1]), ('candidates', self.study['candidates'][:-1])]:
            changed = copy.deepcopy(frozen); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                campaign.verify_predictions(self.out, 'development', changed, self.study, self.reservation)

    def test_partial_scoring_attempt_is_consumed_and_blocks_reprediction(self):
        self.predictions()
        def fail(*args):
            self.assertTrue((self.out / 'development-reference-access.json').exists())
            raise RuntimeError('synthetic interrupted reference read')
        with patch.object(campaign.prior, 'item_at', return_value={'blocks': [(1, 0, 2)]}), \
             patch.object(campaign.prior, 'reference', side_effect=fail):
            with self.assertRaises(RuntimeError): campaign.score(self.out, 'development')
        with patch.object(campaign.prior, 'reference') as reference, patch.object(campaign, 'ProcessPoolExecutor') as pool:
            with self.assertRaises(ValueError): campaign.score(self.out, 'development')
            with self.assertRaises(ValueError): campaign.predict(self.out, 'development', 1)
            reference.assert_not_called(); pool.assert_not_called()

    def test_exclusive_marker_cannot_be_overwritten(self):
        path = self.out / 'exclusive.json'
        campaign.begin_access(path, {'attempt': 1})
        with self.assertRaises(FileExistsError): campaign.begin_access(path, {'attempt': 2})
        self.assertEqual(campaign.read(path), {'attempt': 1})

    def test_source_and_correction_drift_rejected(self):
        with patch.object(campaign, 'sources', return_value={'synthetic': 'b' * 64}):
            with self.assertRaises(ValueError): campaign.authenticate(self.out)
        path = self.out / 'corrections' / 'penalty-01.json'
        data = campaign.read(path); data['parameters']['dead_logit_penalty'] = .5
        write_json(path, data)
        with self.assertRaises(ValueError): campaign.authenticate(self.out)

    def test_zero_parent_parity_failure_stops_before_publishing_predictions(self):
        mid = self.reservation['split']['development'][0]
        folder = self.parent / 'development' / str(mid); folder.mkdir(parents=True)
        np.save(folder / 'candidate.npy', np.array(list('HAD')), allow_pickle=False)
        with patch.object(campaign.prior, 'item_at', return_value={}), \
             patch.object(campaign, 'load_model', return_value={}), \
             patch.object(campaign.correction, 'probabilities_from_cache', return_value=np.ones((3, 3))/3), \
             patch.object(campaign.correction, 'decode_probabilities', return_value=(np.array(list('DAD')), None)):
            with self.assertRaises(ValueError): campaign.predict_one(self.out, 'development', mid, self.study['candidates'])
        self.assertFalse((self.out / 'development' / str(mid) / 'parent.npy').exists())

    def test_completed_score_records_identity_and_silver_limitations(self):
        result = self.completed_score()
        self.assertFalse(result['promoted'])
        self.assertFalse(result['evidence']['dense_truth_validated'])
        self.assertFalse(result['evidence']['usage_permitted'])
        self.assertEqual(result['parent_records'], self.old)
        self.assertEqual(result['reference_access_sha256'], sha256(self.out / 'development-reference-access.json'))
        self.assertEqual(result['selection']['selected_candidate_id'], 'penalty-01')


if __name__ == '__main__':
    unittest.main()
