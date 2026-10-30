import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools import fresh_benchmark_v7 as fresh


def scores():
    item = {'n': 100, 'accuracy': .8, 'balanced_team_recall': .8,
            'dense_had_truth': False, 'reference_quality': 'provider_silver'}
    return {f'skillcorner{m}': {'baseline': copy.deepcopy(item), 'candidate': copy.deepcopy(item)} for m in fresh.SC}


class FreshV7Tests(unittest.TestCase):
    def test_only_new_matches_reserved_and_immutable_catalog(self):
        self.assertFalse(set(fresh.SC) & set(fresh.SPLIT['already_exposed_skillcorner']))
        self.assertEqual(set(fresh.SC), {1953632, 1959846})
        self.assertEqual(len(fresh.catalog('unused')), 6)
        for _, expected, url in fresh.catalog('unused'):
            self.assertGreater(expected[0], 0)
            self.assertIn(fresh.SC_COMMIT, url)

    def test_scientific_source_closure_excludes_packaging(self):
        hashes = fresh.scientific_hashes()
        for name in ('tools/fresh_benchmark_v7.py', 'tools/fresh_benchmark.py', 'tools/independent_benchmark.py', 'possession/pipeline.py'):
            self.assertIn(name, hashes)
        self.assertTrue(all(n.startswith('possession/') or n in {
            'tools/fresh_benchmark_v7.py', 'tools/fresh_benchmark.py', 'tools/independent_benchmark.py'} for n in hashes))

    def test_phase_success_never_promotes_stoppage_changes(self):
        values = scores()
        for pair in values.values():
            pair['candidate']['accuracy'] = .9
            pair['candidate']['balanced_team_recall'] = .9
        result = fresh.decide(values)
        self.assertTrue(result['phase_regression_passed'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['promotion_eligible'])
        self.assertFalse(result['fresh_stoppage_validation_available'])
        self.assertFalse(result['fresh_short_possession_validation_available'])

    def test_mean_and_per_match_regression_checks(self):
        for field in ('accuracy', 'balanced_team_recall'):
            values = scores()
            next(iter(values.values()))['candidate'][field] -= .001
            self.assertFalse(fresh.decide(values)['phase_regression_passed'])
        values = scores()
        pairs = list(values.values())
        pairs[0]['candidate']['accuracy'] -= .011
        pairs[1]['candidate']['accuracy'] += .02
        self.assertFalse(fresh.decide(values)['phase_regression_passed'])

    def test_gate_rejects_invalid_scope_support_or_metric(self):
        for key, bad in [('n', 0), ('dense_had_truth', True), ('reference_quality', 'human'),
                         ('accuracy', float('nan')), ('balanced_team_recall', True), ('accuracy', 1.1)]:
            values = scores()
            next(iter(values.values()))['candidate'][key] = bad
            with self.assertRaises(ValueError):
                fresh.decide(values)
        values = scores()
        values.pop(next(iter(values)))
        with self.assertRaises(ValueError):
            fresh.decide(values)

    def test_split_and_policy_cannot_change(self):
        with tempfile.TemporaryDirectory() as d:
            fresh.freeze_split(d)
            fresh.freeze_split(d)
            (Path(d) / 'policy-before-reference-access.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'policy changed'):
                fresh.freeze_split(d)

    def test_phase_absence_is_unknown_and_never_dead(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'phase.csv'
            path.write_text('frame_start,frame_end,period,team_in_possession_id\n1,3,1,10\n5,7,2,20\n')
            alignment = {'periods': np.array([1] * 4 + [2] * 4), 'team_ids': [10, 20]}
            reference, counts = fresh.old.phase_references(path, alignment)
            self.assertEqual(''.join(reference), 'XHHXXAAX')
            self.assertEqual(counts['dead_reference_frames'], 0)

    def _audit(self, mutate=False, invalid_prediction=False):
        with tempfile.TemporaryDirectory() as d:
            folder, out = Path(d) / 'data', Path(d) / 'audit'
            folder.mkdir()
            for path, _, _ in fresh.catalog(folder):
                path.write_text('unread synthetic source')
            for name in ('model.json', 'config.json'):
                (folder / name).write_text('{}')
            labels = np.array(list('HHHAAAXXHHAAAAXX'))
            obs = (None, [(1, 0, 7), (2, 8, 15)], (105., 68.),
                   {'fps': 10, 'periods': np.array([1] * 8 + [2] * 8), 'team_ids': [10, 20]})
            calls = []

            def infer(*args):
                self.assertEqual(args[4], 10)
                self.assertTrue((folder / 'final-reference-access.json').exists())
                calls.append('infer')
                if mutate:
                    (folder / 'model.json').write_text('{"changed":true}')
                return labels[:-1] if invalid_prediction else labels.copy(), None, None, []

            def reference(*args):
                self.assertEqual(calls.count('infer'), 4)
                self.assertEqual(len(list(out.glob('*.csv'))), 4)
                self.assertTrue((out / 'predictions-before-reference-access.json').exists())
                calls.append('reference')
                return labels.copy(), {'dead_reference_frames': 0}

            with patch.object(fresh, 'read_skillcorner', return_value=obs), patch.object(fresh.old, 'check_file'), \
                 patch.object(fresh.old, 'phase_references', side_effect=reference), \
                 patch('possession.learned.load_model', return_value={}), patch.object(fresh, 'load_config', return_value={}), \
                 patch('possession.pipeline.infer', side_effect=infer):
                if mutate or invalid_prediction:
                    with self.assertRaisesRegex(ValueError, 'Frozen artifacts changed|Invalid prediction'):
                        fresh.evaluate_pair(folder, folder / 'model.json', folder / 'model.json', folder / 'config.json', out)
                    self.assertNotIn('reference', calls)
                else:
                    result = fresh.evaluate_pair(folder, folder / 'model.json', folder / 'model.json', folder / 'config.json', out)
                    self.assertEqual(calls, ['infer'] * 4 + ['reference'] * 2)
                    self.assertFalse(result['gate']['passed'])
                    self.assertTrue(result['gate']['phase_regression_passed'])
                    self.assertFalse(result['reference_fields_used_for_inference'])
                with self.assertRaisesRegex(ValueError, 'already consumed'):
                    fresh.evaluate_pair(folder, folder / 'model.json', folder / 'model.json', folder / 'config.json', Path(d) / 'other')

    def test_all_predictions_sealed_before_references_and_one_shot(self):
        self._audit()

    def test_model_tampering_fails_before_reference(self):
        self._audit(mutate=True)

    def test_invalid_prediction_fails_before_reference(self):
        self._audit(invalid_prediction=True)


if __name__ == '__main__':
    unittest.main()
