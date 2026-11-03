import copy
from pathlib import Path
import tempfile
import unittest

from tools import pff_validation as pff


def protocol():
    rows = [{'match_id': i, 'date': '2022-11-21' if i < 10000 else '2022-12-18',
             'home_team': f'Team{i}', 'away_team': 'Opponent',
             'stage': 'group' if i < 10000 else 'knockout',
             'competition': 'FIFA World Cup 2022'} for i in pff.EXPECTED_IDS]
    return pff.build_protocol(rows, {str(i): 'a' * 64 for i in pff.EXPECTED_IDS}, 'b' * 64)


def scores(reservation):
    base = dict.fromkeys(pff.METRICS, .7)
    return [{'match_id': i, 'condition': 'native', 'baseline_sha256': pff.BASELINE_SHA256,
             'candidate_sha256': 'c' * 64, 'baseline': dict(base),
             'candidate': dict.fromkeys(pff.METRICS, .72),
             'reference_dead_segments': 100, 'reference_short_spells': 100}
            for i in reservation['split']['final']]


class PFFValidationTests(unittest.TestCase):
    def test_order_invariant_and_stage_balanced(self):
        first = protocol()
        second = pff.build_protocol(first['metadata'][::-1], first['metadata_sha256'], 'b' * 64)
        self.assertEqual(first, second)
        self.assertEqual([len(first['split'][k]) for k in ('train', 'development', 'final')], [40, 12, 12])
        self.assertEqual(sum(i < 10000 for i in first['split']['final']), 9)
        self.assertIn(3812, first['split']['train'])
        self.assertFalse(first['labels_used_for_split'])

    def test_label_fields_rejected(self):
        first = protocol()
        first['metadata'][0]['possession'] = 'H'
        with self.assertRaises(ValueError):
            pff.build_protocol(first['metadata'], first['metadata_sha256'], 'b' * 64)

    def test_identity_overlap_rejected(self):
        first = protocol()
        first['metadata'][1]['home_team'] = first['metadata'][0]['away_team']
        first['metadata'][1]['away_team'] = first['metadata'][0]['home_team']
        with self.assertRaises(ValueError):
            pff.build_protocol(first['metadata'], first['metadata_sha256'], 'b' * 64)

    def test_missing_fixture_or_hash_rejected(self):
        first = protocol()
        with self.assertRaises(ValueError):
            pff.build_protocol(first['metadata'][:-1], first['metadata_sha256'], 'b' * 64)
        first['metadata_sha256']['3812'] = 'bad'
        with self.assertRaises(ValueError):
            pff.build_protocol(first['metadata'], first['metadata_sha256'], 'b' * 64)

    def test_split_reassignment_cannot_be_rehashed(self):
        first = protocol()
        first['split']['train'][0], first['split']['final'][0] = first['split']['final'][0], first['split']['train'][0]
        first['protocol_sha256'] = pff.digest({k: v for k, v in first.items() if k != 'protocol_sha256'})
        with self.assertRaises(ValueError):
            pff.verify_protocol(first)

    def test_reservation_immutable(self):
        first = protocol()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'protocol.json'
            pff.freeze_protocol(target, first)
            pff.freeze_protocol(target, first)
            first['metadata'][0]['home_team'] = 'Renamed'
            changed = pff.build_protocol(first['metadata'], first['metadata_sha256'], 'b' * 64)
            with self.assertRaises(ValueError):
                pff.freeze_protocol(target, changed)

    def test_positive_comparison_requires_every_evidence_flag(self):
        first = protocol()
        rows = scores(first)
        evidence = dict.fromkeys(pff.EVIDENCE, True)
        self.assertTrue(pff.promotion_gate(rows, first, evidence)['promoted'])
        for name in pff.EVIDENCE:
            insufficient = dict(evidence, **{name: False})
            self.assertFalse(pff.promotion_gate(rows, first, insufficient)['promoted'])

    def test_segment_regression_blocks_other_improvements(self):
        first = protocol()
        rows = scores(first)
        for row in rows:
            row['candidate']['dead_segment'] = .699
        self.assertFalse(pff.promotion_gate(rows, first, dict.fromkeys(pff.EVIDENCE, True))['promoted'])

    def test_no_dropping_bad_fixture_or_multiple_candidates(self):
        first = protocol()
        rows = scores(first)
        with self.assertRaises(ValueError):
            pff.promotion_gate(rows[:-1], first, {})
        rows[0]['candidate_sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            pff.promotion_gate(rows, first, {})

    def test_sparse_or_invalid_metrics_not_promotable(self):
        first = protocol()
        rows = scores(first)
        rows[0]['baseline']['dead_precision'] = None
        with self.assertRaises(ValueError):
            pff.promotion_gate(rows, first, {})
        rows = scores(first)
        for row in rows:
            row['reference_short_spells'] = 1
        self.assertFalse(pff.promotion_gate(rows, first, dict.fromkeys(pff.EVIDENCE, True))['promoted'])

    def test_reference_count_bool_and_wrong_baseline_rejected(self):
        first = protocol()
        rows = scores(first)
        rows[0]['reference_short_spells'] = True
        with self.assertRaises(ValueError):
            pff.promotion_gate(rows, first, {})
        rows = scores(first)
        rows[0]['baseline_sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            pff.promotion_gate(rows, first, {})

    def test_bootstrap_reproducible(self):
        first = protocol()
        rows = scores(first)
        evidence = dict.fromkeys(pff.EVIDENCE, True)
        self.assertEqual(pff.promotion_gate(rows, first, evidence),
                         pff.promotion_gate(rows[::-1], first, evidence))

    def test_development_gate_never_promotes(self):
        first = protocol()
        rows = scores(first)
        for row, match in zip(rows, first['split']['development']):
            row['match_id'] = match
        result = pff.development_gate(rows, first)
        self.assertTrue(result['eligible_for_final_audit'])
        self.assertFalse(result['promoted'])
        rows[0]['candidate']['spell'] = .4
        self.assertFalse(pff.development_gate(rows, first)['eligible_for_final_audit'])

    def test_development_and_final_cannot_be_interchanged(self):
        first = protocol()
        with self.assertRaises(ValueError):
            pff.development_gate(scores(first), first)


if __name__ == '__main__':
    unittest.main()
