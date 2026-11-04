"""Synthetic-only tests of fixed A/B fitting, isolation and immutable parents."""
import copy
import importlib.util
import unittest
from unittest.mock import Mock, patch

import numpy as np

from possession import learned, robust_control
from tools import clean_pff_adaptation_fit as fit
from tests.test_joint_temporal import invented_model


def toy_parent():
    return dict(schema_version=1, kind='logistic_live_dead', decoder_mode='joint',
                feature_names=list(learned.FEATURE_NAMES), mean=[0.] * 58,
                scale=[1.] * 58, coefficients=[0.] * 58, intercept=-.3,
                metadata={'synthetic': True}, candidate_id='synthetic-parent',
                robust_control_model=dict(schema_version=1, kind='symmetric_logistic_control_v3',
                    feature_names=list(robust_control.FEATURE_NAMES), feature_mask=[True]*98,
                    scale=[1.]*98, coefficients=[.2]+[0.]*97, intercept=0.),
                joint_model=invented_model())


def toy_record(match, n=90):
    y = np.resize(np.repeat(list('HAD'), 30), n)
    x, z = np.zeros((n, 98)), np.zeros((n, 58))
    x[:, 0] = np.where(y == 'H', 1., -1.)
    z[:, 0] = (y == 'D').astype(float)
    return dict(match=match, x=x, z=z, y=y, blocks=[[1, 0, n-1]], fps=25,
                reference_fields_used_for_inference=False)


class Explosive:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Disallowed arrays were processed')


class CleanPFFAdaptationFitTests(unittest.TestCase):
    def setUp(self):
        self.ids = fit.protocol.fit_ids('inner')
        self.parent = toy_parent()
        self.kw = dict(fixture_ids=self.ids, parent_model=self.parent,
                       protocol_sha256=fit.protocol.fixed_protocol_sha256())

    def test_bad_rosters_fail_before_factory_or_parent(self):
        cases = [list(self.ids[:-1]), list(self.ids)+[self.ids[0]], list(reversed(self.ids)),
                 [3812]+list(self.ids[1:])]
        for held in (fit.protocol.INNER_VALIDATION_IDS[0], 3813, 3814, 99999):
            cases.append([f'pff{held}']+list(self.ids[1:]))
        for ids in cases:
            loader = Mock()
            with patch.object(fit, 'validate_parent') as parent:
                with self.assertRaises(ValueError):
                    fit.fit_candidate('A', loader, **dict(self.kw, fixture_ids=ids))
                parent.assert_not_called(); loader.assert_not_called()

    def test_refit_unknown_candidate_or_missing_binding_cannot_read(self):
        for candidate, change in [('C', {}), ('A', {'phase': 'refit'}),
                                  ('A', {'protocol_sha256': 'x'}), ('A', {'protocol_sha256': 'a'*64})]:
            loader = Mock()
            with self.assertRaises(ValueError):
                fit.fit_candidate(candidate, loader, **dict(self.kw, **change))
            loader.assert_not_called()

    def test_wrong_parent_content_rejected_before_records(self):
        loader = Mock()
        with self.assertRaisesRegex(ValueError, 'parent content'):
            fit.fit_candidate('A', loader, **self.kw)
        loader.assert_not_called()

    def test_parent_digest_is_content_identity_not_metadata_declaration(self):
        parent = copy.deepcopy(self.parent)
        parent['metadata']['claimed_sha256'] = fit.PARENT_FILE_SHA256
        with self.assertRaises(ValueError): fit.validate_parent(parent)
        parent['coefficients'][0] = 1.
        self.assertNotEqual(fit.content_digest(parent), fit.content_digest(self.parent))

    def test_wrong_record_and_cached_probabilities_fail_before_arrays(self):
        for change in ({'match': 'pff3814'}, {'base_home': Explosive()}, {'base_dead': Explosive()},
                       {'reference_fields_used_for_inference': True}, {'fps': 24}):
            record = toy_record(self.ids[0]); record.update(change)
            record['x'] = Explosive()
            with self.assertRaises(ValueError):
                fit.sample_record(record, self.ids[0], fit.protocol.recipe())
        with self.assertRaises(ValueError):
            fit.sample_record({}, 'pff3814', fit.protocol.recipe())

    def test_samples_are_half_local_owned_copies_and_unknown_has_zero_weight(self):
        record = toy_record(self.ids[0], 100)
        record['blocks'] = [[1, 3, 42], [2, 57, 96]]
        record['y'][3] = 'X'
        record['x'][:, 1] = np.arange(100)
        sequence, support = fit.sample_record(record, self.ids[0], fit.protocol.recipe())
        np.testing.assert_array_equal(sequence[0]['X'][:, 1], np.arange(3, 43, 5))
        np.testing.assert_array_equal(sequence[1]['X'][:, 1], np.arange(57, 97, 5))
        self.assertEqual(sequence[0]['w'][0], 0)
        self.assertAlmostEqual(sum(s['w'].sum() for s in sequence), 10000.)
        self.assertEqual(support['sampled_rows'], 16)
        before = sequence[0]['X'].copy(); record['x'][:] = 999
        np.testing.assert_array_equal(sequence[0]['X'], before)

    def test_invalid_periods_shapes_and_unsupported_labels_rejected(self):
        for change in ({'blocks': [[1, 0, 30], [2, 30, 50]]}, {'blocks': [[1, 0, 30], [1, 31, 50]]},
                       {'blocks': []}, {'blocks': [[1, True, 5]]},
                       {'y': np.repeat('X', 90)}, {'y': np.repeat('Q', 90)},
                       {'x': np.zeros((90, 97))}, {'z': np.full((90, 58), np.nan)}):
            record = toy_record(self.ids[0]); record.update(change)
            with self.assertRaises(ValueError): fit.sample_record(record, self.ids[0], fit.protocol.recipe())

    def test_each_fixture_mass_equal_despite_different_lengths(self):
        for n in (90, 180):
            _, support = fit.sample_record(toy_record(self.ids[0], n), self.ids[0], fit.protocol.recipe())
            self.assertAlmostEqual(support['sampled_weight'], 10000.)

    def test_A_replaces_joint_preserves_heads_and_traverses_factory_once(self):
        saved = copy.deepcopy(self.parent); calls = []
        def loader(mid):
            calls.append(mid); return toy_record(mid)
        fresh = invented_model(); fresh['metadata'] = {'new': True}
        with patch.object(fit, 'validate_parent'), patch.object(fit.learned, 'fit_model') as dfit, \
             patch.object(fit.train_joint, 'fit', return_value=fresh) as joint:
            model, report = fit.fit_candidate('A', loader, **self.kw)
        self.assertEqual(calls, list(self.ids)); dfit.assert_not_called()
        self.assertEqual(self.parent, saved)
        for key in fit.DEAD_PARAMETERS: self.assertEqual(model[key], saved[key])
        self.assertEqual(model['robust_control_model'], saved['robust_control_model'])
        self.assertEqual(model['joint_model'], fresh)
        self.assertEqual(joint.call_args.kwargs['epochs'], 8)
        self.assertEqual(joint.call_args.kwargs['seed'], 20261101)
        for s in joint.call_args.args[0]:
            np.testing.assert_allclose(s['base_dead'], -.3, atol=1e-14)
            np.testing.assert_allclose(s['base_home'], .2*s['X'][:, 0], atol=1e-14)
        self.assertFalse(report['old_joint_used']); self.assertFalse(report['refit_authorized'])

    def test_B_refits_dead_before_recomputing_logits_and_joint(self):
        replacement = copy.deepcopy(self.parent)
        replacement['intercept'] = 1.2
        replacement['training'] = {'synthetic_fit': True}
        with patch.object(fit, 'validate_parent'), \
             patch.object(fit.learned, 'fit_model', return_value=replacement) as dead, \
             patch.object(fit.train_joint, 'fit', return_value=invented_model()) as joint:
            model, report = fit.fit_candidate('B', toy_record, **self.kw)
        self.assertEqual(dead.call_args.kwargs['l2'], .01)
        self.assertEqual(dead.call_args.kwargs['max_iter'], 80)
        self.assertEqual(dead.call_args.args[0].shape, (39*18, 58))
        self.assertAlmostEqual(dead.call_args.kwargs['sample_weight'].sum(), 39*10000.)
        for s in joint.call_args.args[0]: np.testing.assert_allclose(s['base_dead'], 1.2, atol=1e-14)
        self.assertEqual(model['intercept'], 1.2)
        self.assertEqual(model['robust_control_model'], self.parent['robust_control_model'])
        self.assertEqual(report['upstream_stacking'], 'in-sample-fit-only')

    def test_missing_class_stops_before_any_fit(self):
        def loader(mid):
            record = toy_record(mid); record['y'][:] = 'D'; return record
        with patch.object(fit, 'validate_parent'), patch.object(fit.learned, 'fit_model') as dead, \
             patch.object(fit.train_joint, 'fit') as joint:
            with self.assertRaises(ValueError): fit.fit_candidate('B', loader, **self.kw)
        dead.assert_not_called(); joint.assert_not_called()

    def test_actual_logistic_scaler_excludes_unknown_extremes(self):
        def loader(mid):
            record = toy_record(mid)
            record['y'][0] = 'X'; record['z'][0, 1] = 1e10
            return record
        with patch.object(fit, 'validate_parent'), \
             patch.object(fit.train_joint, 'fit', return_value=invented_model()) as joint:
            model, report = fit.fit_candidate('B', loader, **self.kw)
        self.assertEqual(model['mean'][1], 0.)
        self.assertEqual(model['scale'][1], 1.)
        self.assertTrue(model['training']['converged'])
        self.assertEqual(report['new_dead_training']['excluded_rows'], 39)
        self.assertTrue(all(s['w'][0] == 0 for s in joint.call_args.args[0]))

    def test_report_contains_aggregate_support_not_frame_records(self):
        with patch.object(fit, 'validate_parent'), \
             patch.object(fit.train_joint, 'fit', return_value=invented_model()):
            _, report = fit.fit_candidate('A', toy_record, **self.kw)
        self.assertEqual(report['fitting_fixtures'], 39)
        self.assertFalse(report['final_accessed']); self.assertFalse(report['promoted'])
        self.assertFalse(report['cache_bytes_authenticated_by_kernel'])
        for value in report['fixture_support'].values():
            self.assertEqual(set(value), {'periods', 'sampled_rows', 'class_counts', 'sampled_weight'})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional local Torch training dependency')
    def test_actual_eight_epoch_joint_fit_on_tiny_synthetic_cohort(self):
        with patch.object(fit, 'validate_parent'):
            model, report = fit.fit_candidate('A', toy_record, **self.kw)
        learned.validate_model(model)
        self.assertEqual(model['joint_model']['training']['epochs'], 8)
        self.assertEqual(len(model['joint_model']['training']['losses']), 8)
        self.assertTrue(np.isfinite(model['joint_model']['training']['losses']).all())
        self.assertEqual(model['robust_control_model'], self.parent['robust_control_model'])
        self.assertEqual(report['new_joint_training']['positive_weight_rows'], 39*18)


if __name__ == '__main__':
    unittest.main()
