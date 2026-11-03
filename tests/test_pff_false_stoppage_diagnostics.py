import unittest
import json
import numpy as np
from tools.pff_false_stoppage_diagnostics import boundary_regions, summarize, add_tree


class FalseStoppageDiagnosticsTests(unittest.TestCase):
    def test_unknown_gaps_stop_boundary_attribution(self):
        y = np.array(list('HHXHHDDHAXHA'))
        before, after = boundary_regions(y, [(1, 0, 11)], 50)
        self.assertEqual(np.flatnonzero(before).tolist(), [3, 4])
        self.assertEqual(np.flatnonzero(after).tolist(), [7, 8])

    def test_no_cross_period_boundary(self):
        before, after = boundary_regions(np.array(list('HDDH')), [(1, 0, 0), (2, 1, 3)])
        self.assertFalse(before.any())
        self.assertEqual(np.flatnonzero(after).tolist(), [3])

    def test_added_removed_recovered_and_unknown_are_separate(self):
        y = np.array(list('HADDAX'))
        baseline = np.array(list('DHHDDD'))
        candidate = np.array(list('HDDDHD'))
        probabilities = np.tile([.2, .2, .6], (6, 1))
        z = np.zeros((6, 58))
        result = summarize(y, baseline, candidate, probabilities, z, [(1, 0, 5)])
        self.assertEqual(result['added_false_dead']['frames'], 1)
        self.assertEqual(result['removed_false_dead'], 2)
        self.assertEqual(result['recovered_true_dead']['frames'], 1)
        self.assertEqual(result['candidate_false_dead'], 1)
        self.assertEqual(result['candidate_true_dead'], 2)
        self.assertEqual(result['all_false_dead']['dead_probability_below']['0.6'], 0)
        self.assertEqual(result['all_false_dead']['dead_probability_below']['0.7'], 1)
        self.assertEqual(result['predicted_dead_segments_with_false_frames'], {'overlap_reference_dead': 1, 'no_reference_dead_overlap': 0})
        self.assertEqual(''.join(y), 'HADDAX')
        json.dumps(result, allow_nan=False)

    def test_aggregate_counts_exactly(self):
        result = {}
        add_tree(result, {'a': 2, 'nested': {'b': 3}})
        add_tree(result, {'a': 5, 'nested': {'b': 7}})
        self.assertEqual(result, {'a': 7, 'nested': {'b': 10}})


if __name__ == '__main__':
    unittest.main()
