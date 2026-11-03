import copy
import unittest

import numpy as np

from tools import pff_combined_model as combined, pff_context_model as context
from tools import pff_precision_model as precision, pff_training


def item():
    return {'x': np.zeros((80, 98)), 'z': np.zeros((80, 58)),
            'base_home': np.full(80, .6), 'base_dead': np.full(80, .4),
            'blocks': [(1, 0, 29), (2, 35, 79)], 'fps': 25., 'usable': [True, True],
            'reference_fields_used_for_inference': False}


def head():
    return {'schema_version': 1, 'kind': context.POLICY['kind'], 'policy': copy.deepcopy(context.POLICY),
            'feature_names': list(context.FEATURE_NAMES), 'mean': [0.]*60, 'scale': [1.]*60,
            'input_weight': np.full((32, 60), .01).tolist(), 'input_bias': [.1]*32,
            'output_weight': [.05]*32, 'output_bias': .2,
            'metadata': {'parent_sha256': context.PARENT_SHA256, 'reference_fields_used_for_inference': False},
            'training': {}}


class PFFCombinedModelTests(unittest.TestCase):
    def setUp(self):
        self.item, self.head, self.parent = item(), head(), pff_training.baseline_model()
        self.p = combined.probabilities_from_cache(self.item, self.parent)
        self.r = combined.residuals(self.item, self.p, self.head)

    def test_zero_blend_zero_penalty_is_exact_parent(self):
        expected, p = pff_training.predict_cache(self.item, self.parent)
        labels, q = combined.decode_probabilities(self.p, self.item, self.parent, self.head, 0., 0.)
        np.testing.assert_array_equal(p, self.p)
        np.testing.assert_array_equal(labels, expected); np.testing.assert_array_equal(q, p)

    def test_full_blend_zero_penalty_is_exact_context_parent(self):
        expected, p = context.predict_cache(self.item, self.parent, self.head, blend=1., parent_probabilities=self.p)
        labels, q = combined.decode_probabilities(self.p, self.item, self.parent, self.head, 1., 0., self.r)
        np.testing.assert_array_equal(labels, expected); np.testing.assert_array_equal(q, p)

    def test_cached_and_direct_paths_identical_without_mutation(self):
        saved_p, saved_r, saved_head = self.p.copy(), self.r.copy(), copy.deepcopy(self.head)
        direct, p = combined.decode_probabilities(self.p, self.item, self.parent, self.head, 1., .125)
        cached, q = combined.decode_probabilities(self.p, self.item, self.parent, self.head, 1., .125, self.r)
        np.testing.assert_array_equal(cached, direct); np.testing.assert_array_equal(q, p)
        np.testing.assert_array_equal(self.p, saved_p); np.testing.assert_array_equal(self.r, saved_r)
        self.assertEqual(self.head, saved_head)

    def test_conditional_team_odds_preserved_and_only_two_offsets_composed(self):
        _, q = combined.decode_probabilities(self.p, self.item, self.parent, self.head, .75, .125, self.r)
        np.testing.assert_allclose(q[:, 0]/q[:, 1], self.p[:, 0]/self.p[:, 1], rtol=1e-14)
        np.testing.assert_allclose(q[:, 2]/(1-q[:, 2]),
                                   np.exp(.75*self.r-.125)*self.p[:, 2]/(1-self.p[:, 2]), rtol=1e-13)

    def test_references_do_not_affect_inference_and_period_gaps_stay_unknown(self):
        labels, p = combined.decode_probabilities(self.p, self.item, self.parent, self.head, 1., .125, self.r)
        self.item['y'] = np.full(80, 'D'); self.item['events'] = [{'team': 'away'}]
        altered, q = combined.decode_probabilities(self.p, self.item, self.parent, self.head, 1., .125)
        np.testing.assert_array_equal(labels, altered); np.testing.assert_array_equal(p, q)
        self.assertTrue(np.all(labels[30:35] == 'X'))

    def test_invalid_parameters_and_cached_residual_rejected(self):
        for blend, penalty, r in [(True, .1, self.r), (1.1, .1, self.r), (1., -.1, self.r),
                                   (1., 2.1, self.r), (1., .1, np.zeros(79)),
                                   (1., .1, np.full(80, np.nan)), (1., .1, np.full(80, 2.01))]:
            with self.subTest(blend=blend, penalty=penalty), self.assertRaises(ValueError):
                combined.decode_probabilities(self.p, self.item, self.parent, self.head, blend, penalty, r)
        self.item['reference_fields_used_for_inference'] = True
        with self.assertRaises(ValueError):
            combined.decode_probabilities(self.p, self.item, self.parent, self.head, 1., .1, self.r)


if __name__ == '__main__':
    unittest.main()
