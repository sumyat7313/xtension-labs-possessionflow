import copy
import unittest

import numpy as np

from tools import pff_precision_model as precision, pff_training


def item(n=100):
    return {'x': np.zeros((n, 98)), 'z': np.zeros((n, 58)),
            'base_home': np.full(n, .6), 'base_dead': np.full(n, .35),
            'blocks': [(1, 0, 39), (2, 45, n - 1)], 'fps': 25.,
            'usable': [True, True], 'reference_fields_used_for_inference': False}


class PFFPrecisionTests(unittest.TestCase):
    def test_zero_is_exact_copy_and_nonzero_does_not_mutate(self):
        p = np.random.default_rng(19).dirichlet([1, 2, 3], 100)
        before = p.copy()
        q = precision.adjust_probabilities(p, 0.)
        np.testing.assert_array_equal(q, p)
        self.assertFalse(np.shares_memory(p, q))
        precision.adjust_probabilities(p, .75)
        np.testing.assert_array_equal(p, before)

    def test_bounded_penalty_preserves_live_odds_and_binary_logit_shift(self):
        p = np.random.default_rng(81).dirichlet([2, 3, 4], 200)
        for value in (.125, .5, 1., 2.):
            q = precision.adjust_probabilities(p, value)
            np.testing.assert_allclose(q[:, 0] / q[:, 1], p[:, 0] / p[:, 1], rtol=1e-14)
            np.testing.assert_allclose(q[:, 2] / (1-q[:, 2]), np.exp(-value)*p[:, 2]/(1-p[:, 2]), rtol=1e-14)
            self.assertTrue(np.all(q[:, 2] <= p[:, 2]))
            np.testing.assert_allclose(q.sum(axis=1), 1, rtol=0, atol=1e-15)

    def test_pure_classes_and_zero_live_mass_are_not_invented(self):
        p = np.eye(3)
        np.testing.assert_array_equal(precision.adjust_probabilities(p, 2.), p)
        p = np.array([[0., .4, .6], [.8, 0., .2]])
        self.assertEqual(precision.adjust_probabilities(p, .3)[0, 0], 0)
        self.assertEqual(precision.adjust_probabilities(p, .3)[1, 1], 0)

    def test_invalid_penalties_rejected(self):
        for value in (-.1, 2.001, np.nan, np.inf, True, np.bool_(False), None, '1', [1], np.array([1.])):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                precision.adjust_probabilities([[.3, .3, .4]], value)

    def test_invalid_probabilities_rejected(self):
        for p in ([], [.3, .3, .4], [[.5, .5]], [[.5, .5, .2]], [[-.1, .6, .5]],
                  [[np.nan, .5, .5]], [[np.inf, 0, 0]], [[1.1, 0, -.1]]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                precision.adjust_probabilities(p, .5)

    def test_exact_frozen_cache_parity_and_reference_independence(self):
        model = pff_training.baseline_model()
        cache = item(); before = copy.deepcopy(cache)
        expected, p = pff_training.predict_cache(cache, model)
        actual_p = precision.probabilities_from_cache(cache, model)
        np.testing.assert_array_equal(actual_p, p)
        labels, q = precision.predict_cache(cache, model)
        np.testing.assert_array_equal(labels, expected)
        np.testing.assert_array_equal(q, p)
        cache['y'] = np.full(100, 'D'); cache['events'] = [{'home': False}]
        changed, r = precision.predict_cache(cache, model)
        np.testing.assert_array_equal(labels, changed)
        np.testing.assert_array_equal(q, r)
        for key in ('x', 'z', 'base_home', 'base_dead'):
            np.testing.assert_array_equal(cache[key], before[key])

    def test_period_separators_and_unsupported_periods_remain_unknown(self):
        cache = item(); model = pff_training.baseline_model()
        cache['usable'][1] = False
        labels, _ = precision.decode_probabilities(np.tile([.01, .01, .98], (100, 1)), cache, model, .5)
        self.assertTrue(np.all(labels[:40] == 'D'))
        self.assertTrue(np.all(labels[40:] == 'X'))

    def test_invalid_observation_cache_rejected(self):
        model = pff_training.baseline_model()
        for key, value in [('reference_fields_used_for_inference', True), ('x', np.zeros((100, 97))),
                           ('z', np.full((100, 58), np.nan)), ('base_dead', np.full(100, 1.1)),
                           ('base_home', np.zeros(99)), ('fps', 29.97), ('usable', [True]),
                           ('usable', [True, 1]), ('blocks', [(1, 0, 50), (2, 45, 99)])]:
            cache = item(); cache[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                precision.probabilities_from_cache(cache, model)


if __name__ == '__main__':
    unittest.main()
