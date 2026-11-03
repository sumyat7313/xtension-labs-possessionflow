import contextlib
import copy
import importlib.util
import io
import unittest

import numpy as np

from tools import pff_context_model as context, pff_training, pff_precision_model as precision


def item(mid=1, n=100, value=1.):
    return {'match': mid, 'z': np.full((n, 58), value),
            'probabilities': np.tile([.3, .2, .5], (n, 1)),
            'y': np.array(list('H'*(n//2)+'D'*(n-n//2))), 'blocks': [(1, 0, n-1)],
            'fps': 25., 'usable': [True], 'reference_fields_used_for_inference': False}


def head():
    return {'schema_version': 1, 'kind': context.POLICY['kind'], 'policy': copy.deepcopy(context.POLICY),
            'feature_names': list(context.FEATURE_NAMES), 'mean': [0.]*60, 'scale': [1.]*60,
            'input_weight': np.full((32, 60), .01).tolist(), 'input_bias': [0.]*32,
            'output_weight': [.05]*32, 'output_bias': 0.,
            'metadata': {'parent_sha256': context.PARENT_SHA256, 'reference_fields_used_for_inference': False},
            'training': {}}


class PFFContextModelTests(unittest.TestCase):
    def test_context_features_are_team_invariant(self):
        rng = np.random.default_rng(8)
        z = rng.normal(size=(100, 58)); p = rng.dirichlet([2, 3, 4], 100)
        np.testing.assert_array_equal(context.features(z, p), context.features(z, p[:, [1, 0, 2]]))

    def test_adjustment_preserves_team_odds_and_zero_is_exact_copy(self):
        p = np.random.default_rng(13).dirichlet([2, 2, 2], 100)
        saved = p.copy(); r = np.linspace(-2, 2, len(p))
        q = context.adjust_probabilities(p, r, .75)
        np.testing.assert_allclose(q[:, 0]/q[:, 1], p[:, 0]/p[:, 1], rtol=1e-14)
        np.testing.assert_allclose(q[:, 2]/(1-q[:, 2]), np.exp(.75*r)*p[:, 2]/(1-p[:, 2]), rtol=1e-14)
        zero = context.adjust_probabilities(p, r, 0.)
        np.testing.assert_array_equal(zero, p); self.assertFalse(np.shares_memory(zero, p))
        np.testing.assert_array_equal(saved, p)

    def test_inference_ignores_reference_payload_and_residual_is_bounded(self):
        a = item(); model = head(); before = copy.deepcopy(model)
        r = context.residuals(a, a['probabilities'], model)
        a['y'][:] = 'A'; a['events'] = [{'team': 'away'}]
        np.testing.assert_array_equal(r, context.residuals(a, a['probabilities'], model))
        self.assertTrue(np.all(np.abs(r) <= 2.)); self.assertEqual(model, before)

    def test_invalid_residual_blend_and_head_rejected(self):
        p = np.tile([.3, .3, .4], (5, 1))
        for blend in (-1, 1.01, True, np.nan, '1', np.array([.5])):
            with self.subTest(blend=repr(blend)), self.assertRaises(ValueError):
                context.adjust_probabilities(p, np.zeros(5), blend)
        for r in (np.zeros(4), np.full(5, np.nan), np.full(5, 2.1)):
            with self.assertRaises(ValueError): context.adjust_probabilities(p, r, .5)
        for key, value in [('scale', [0.]*60), ('output_weight', [0.]), ('output_bias', float('nan')),
                           ('feature_names', ['wrong']), ('policy', {})]:
            h = head(); h[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): context.validate_head(h)

    def test_whole_fixture_membership_excludes_heldout_duplicate_and_missing(self):
        for items, train, held in [([item(2)], [1], [2]), ([item()], [1], [1]),
                                    ([item(), item()], [1], [2]), ([item()], [1, 2], []),
                                    ([item()], ['1'], [])]:
            with self.assertRaises(ValueError): context.training_samples(items, train, held)

    def test_fit_requires_frozen_protocol_before_reading_items(self):
        def forbidden_items():
            raise AssertionError('Invalid protocol must not consume a training item')
            yield
        for value in (None, '', 'z'*64, 'a'*63):
            with self.subTest(value=value), self.assertRaises(ValueError):
                context.fit(forbidden_items(), training_matches=[1], held_out_matches=[2], protocol_sha256=value)

    def test_scaler_is_train_only_unknown_excluded_and_fixture_mass_equal(self):
        a, b = item(1, 100, 1.), item(2, 200, 3.)
        a['y'][0] = 'X'; a['z'][0] = 100000
        x, y, offset, w, mean, scale, support = context.training_samples(iter([a, b]), [1, 2], [3])
        np.testing.assert_allclose(mean[:58], 2., rtol=0, atol=1e-12)
        self.assertEqual(len(x), 59)
        self.assertEqual(support['1']['sampled_rows'], 19)
        self.assertEqual(support['2']['sampled_rows'], 40)
        self.assertAlmostEqual(float(w[:19].sum()), float(w[19:].sum()), places=5)
        self.assertAlmostEqual(float(w.sum()), len(w), places=5)

    def test_period_sampling_excludes_gaps_and_unsupported_periods(self):
        a = item(n=100)
        a['blocks'] = [(1, 0, 19), (2, 40, 99)]; a['usable'] = [False, True]
        _, _, _, _, _, _, support = context.training_samples([a], [1], [])
        self.assertEqual(support['1']['sampled_rows'], 12)

    def test_empty_known_support_and_single_class_fail(self):
        for change in ('unknown', 'dead', 'unsupported'):
            a = item()
            if change == 'unknown': a['y'][:] = 'X'
            elif change == 'dead': a['y'][:] = 'D'
            else: a['usable'] = [False]
            with self.subTest(change=change), self.assertRaises(ValueError):
                context.training_samples([a], [1], [])

    def test_zero_blend_matches_parent_decoder_and_keeps_gaps_unknown(self):
        a = item(); a.update(x=np.zeros((100, 98)), base_home=np.full(100, .6), base_dead=np.full(100, .35))
        a['blocks'] = [(1, 0, 39), (2, 45, 99)]
        a['usable'] = [True, False]
        parent = pff_training.baseline_model(); p = precision.probabilities_from_cache(a, parent)
        expected, _ = precision.decode_probabilities(p, a, parent)
        labels, q = context.predict_cache(a, parent, head(), blend=0, parent_probabilities=p)
        np.testing.assert_array_equal(labels, expected); np.testing.assert_array_equal(q, p)
        self.assertTrue(np.all(labels[40:] == 'X'))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional training dependency Torch unavailable')
    def test_numpy_inference_matches_nonzero_torch_head(self):
        import torch
        torch.set_num_threads(1); torch.manual_seed(5)
        net = context.make_network()
        with torch.no_grad():
            net.output.weight.normal_(std=.2); net.output.bias.fill_(.3)
        meta = head()['metadata']
        exported = context.export_head(net, np.zeros(60), np.ones(60), meta, {})
        x = np.random.default_rng(99).normal(size=(80, 60))
        expected = net(torch.from_numpy(x.astype(np.float32))).detach().numpy()
        np.testing.assert_allclose(context.forward(x, exported), expected, rtol=1e-5, atol=2e-7)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional training dependency Torch unavailable')
    def test_fixed_fit_is_deterministic_and_records_training_membership(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = context.fit(iter([item(1), item(2, value=2.)]), training_matches=[1, 2], held_out_matches=[3], protocol_sha256='a'*64)
            second = context.fit(iter([item(1), item(2, value=2.)]), training_matches=[1, 2], held_out_matches=[3], protocol_sha256='a'*64)
        self.assertEqual(first, second)
        self.assertEqual(first['metadata']['training_matches'], [1, 2])
        self.assertEqual(first['metadata']['held_out_matches'], [3])
        self.assertEqual(len(first['training']['losses']), 4)
        self.assertFalse(first['training']['conditional_home_away_trained'])


if __name__ == '__main__':
    unittest.main()
