import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.io import sha256, write_json
from tools import pff_campaign as campaign, pff_validation as validation


def reservation():
    return json.loads((campaign.ROOT / 'models/pff-protocol.json').read_text(encoding='utf-8'))


def prediction_fixture(out, protocol, part='development'):
    (out / 'candidate.json').write_text('{}', encoding='utf-8')
    records = []
    for mid in protocol['split'][part]:
        folder = out / part / str(mid)
        folder.mkdir(parents=True)
        for name in ('baseline', 'candidate'):
            np.save(folder / (name + '.npy'), np.array(list('HAD')), allow_pickle=False)
        records.append({'match_id': mid, **{name + '_prediction_sha256': sha256(folder / (name + '.npy'))
                                           for name in ('baseline', 'candidate')}})
    frozen = {'part': part, 'candidate_sha256': sha256(out / 'candidate.json'),
              'baseline_sha256': validation.BASELINE_SHA256,
              'split_protocol_sha256': protocol['protocol_sha256'], 'source_hashes': {'test': 'a' * 64},
              'references_accessed': False, 'records': records}
    path = out / (part + '-predictions-before-references.json')
    write_json(path, frozen)
    return frozen, path


def fake_score(out, part, mid, which, y, labels, blocks):
    details = {'temporal_diagnostics': {'dead_segments': {'n_gt': 100}, 'short_possessions': {'n_gt': 100}}}
    return details, dict.fromkeys(validation.METRICS, .7 if which == 'baseline' else .72)


class PFFCampaignTests(unittest.TestCase):
    def setUp(self):
        self.protocol = reservation()
        self.source_patch = patch.object(campaign, 'sources', return_value={'test': 'a' * 64})
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def test_final_gate_enforced_by_direct_evaluator_before_prediction_or_reference(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(campaign, 'predict_split') as predict, patch.object(campaign, 'reference') as reference:
            with self.assertRaises(ValueError):
                campaign.evaluate_split(Path(folder), self.protocol, 'final')
            predict.assert_not_called()
            reference.assert_not_called()

    def test_train_cannot_be_reported_as_fresh(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(campaign, 'reference') as reference:
            with self.assertRaises(ValueError):
                campaign.evaluate_split(Path(folder), self.protocol, 'train')
            reference.assert_not_called()

    def test_last_fixture_prediction_tamper_detected_before_first_reference(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(campaign, 'reference') as reference:
            out = Path(folder)
            prediction_fixture(out, self.protocol)
            last = self.protocol['split']['development'][-1]
            (out / 'development' / str(last) / 'candidate.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                campaign.evaluate_split(out, self.protocol, 'development')
            reference.assert_not_called()
            self.assertFalse((out / 'development-reference-access.json').exists())

    def test_wrong_freeze_identity_rejected_before_reference(self):
        for key, value in [('part', 'final'), ('baseline_sha256', 'b' * 64),
                           ('split_protocol_sha256', 'b' * 64), ('references_accessed', True)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder, patch.object(campaign, 'reference') as reference:
                out = Path(folder)
                frozen, path = prediction_fixture(out, self.protocol)
                frozen[key] = value
                write_json(path, frozen)
                with self.assertRaises(ValueError):
                    campaign.evaluate_split(out, self.protocol, 'development')
                reference.assert_not_called()

    def test_failed_reference_attempt_is_consumed(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            prediction_fixture(out, self.protocol)
            def fail_reference(*args):
                self.assertTrue((out / 'development-reference-access.json').exists())
                raise RuntimeError('Synthetic interrupted scoring')
            with patch.object(campaign, 'item_at', return_value={'blocks': [(1, 0, 2)]}), patch.object(campaign, 'reference', side_effect=fail_reference):
                with self.assertRaises(RuntimeError):
                    campaign.evaluate_split(out, self.protocol, 'development')
            with patch.object(campaign, 'reference') as reference:
                with self.assertRaises(ValueError):
                    campaign.evaluate_split(out, self.protocol, 'development')
                reference.assert_not_called()
            with self.assertRaises(ValueError):
                campaign.predict_split(out, self.protocol, 'development')

    def test_all_predictions_exist_and_marker_precedes_every_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            frozen, path = prediction_fixture(out, self.protocol)
            def reference(*args):
                marker = json.loads((out / 'development-reference-access.json').read_text())
                self.assertEqual(marker['prediction_freeze_sha256'], sha256(path))
                self.assertTrue(marker['references_consumed'])
                for saved in frozen['records']:
                    for which in ('baseline', 'candidate'):
                        self.assertEqual(sha256(out / 'development' / str(saved['match_id']) / (which + '.npy')),
                                         saved[which + '_prediction_sha256'])
                return np.array(list('HAD')), {'reference_quality': 'synthetic'}
            with patch.object(campaign, 'item_at', return_value={'blocks': [(1, 0, 2)]}), patch.object(campaign, 'reference', side_effect=reference) as ref, patch.object(campaign, 'score_one', side_effect=fake_score):
                campaign.evaluate_split(out, self.protocol, 'development')
            self.assertEqual(ref.call_count, 12)
            result = json.loads((out / 'development-summary.json').read_text())
            self.assertTrue(result['gate']['eligible_for_final_audit'])
            self.assertFalse(result['promoted'])
            self.assertFalse(result['evidence']['dense_truth_validated'])
            self.assertFalse(result['evidence']['usage_permitted'])
            self.assertEqual(result['reference_access_sha256'], sha256(out / 'development-reference-access.json'))

    def test_final_guard_recomputes_development_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            prediction_fixture(out, self.protocol)
            with patch.object(campaign, 'item_at', return_value={'blocks': [(1, 0, 2)]}), patch.object(campaign, 'reference', return_value=(np.array(list('HAD')), {})), patch.object(campaign, 'score_one', side_effect=fake_score):
                campaign.evaluate_split(out, self.protocol, 'development')
            campaign.require_final_eligible(out, self.protocol)
            path = out / 'development-summary.json'
            summary = json.loads(path.read_text())
            summary['records'][0]['candidate']['spell'] = 0.
            write_json(path, summary)
            with self.assertRaises(ValueError):
                campaign.require_final_eligible(out, self.protocol)

    def test_scientific_freeze_cannot_change_after_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            campaign.freeze(out, self.protocol)
            with patch.object(campaign, 'sources', return_value={'test': 'b' * 64}):
                with self.assertRaises(ValueError):
                    campaign.freeze(out, self.protocol)

    def test_real_scoring_adapter_maps_perfect_period_separated_labels(self):
        y = np.array(list('H' * 30 + 'D' * 40 + 'A' * 10 + 'H' * 20 + 'X' + 'A' * 80 + 'D' * 50))
        with tempfile.TemporaryDirectory() as folder:
            details, values = campaign.score_one(Path(folder), 'development', 3813, 'candidate',
                                                 y, y.copy(), [(1, 0, 99), (2, 101, len(y) - 1)])
        self.assertEqual(set(values), set(validation.METRICS))
        for name in ('spell', 'macro', 'short_precision', 'short_recall', 'dead_segment', 'dead_precision', 'dead_recall'):
            self.assertAlmostEqual(values[name], 1., msg=name)
        self.assertEqual(details['temporal_diagnostics']['dead_segments']['n_gt'], 2)


if __name__ == '__main__':
    unittest.main()
