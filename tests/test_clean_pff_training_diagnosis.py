"""Synthetic component identities, fixed strata and diagnostic accounting."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from possession.io import write_json
from tools import clean_pff_training_diagnosis as diag


class CleanPFFTrainingDiagnosisTests(unittest.TestCase):
    def item(self, n=8, blocks=None):
        return dict(x=np.zeros((n, len(diag.robust_control.FEATURE_NAMES))),
                    z=np.zeros((n, len(diag.FEATURE_NAMES))), blocks=blocks or [[1, 0, n-1]],
                    fps=25., usable=[True] * len(blocks or [[1, 0, n-1]]), pitch=[105., 68.],
                    reference_fields_used_for_inference=False)

    def model(self):
        return dict(robust_control_model={'clean': True}, joint_model={'clean': True}, feature_names=list(diag.FEATURE_NAMES))

    def test_ha_only_keeps_base_dead_exact_and_full_conditional_team(self):
        full = np.array([[.7, .2, .1], [.01, .04, .95], [.4, .4, .2], [1e-10, 2e-10, 1-3e-10]])
        saved = full.copy()
        dead = np.array([0., .7, 1e-12, 1-1e-12])
        result = diag.ha_only_probabilities(full, dead)
        np.testing.assert_array_equal(result[:, 2], dead)
        np.testing.assert_allclose(result[:, :2]/result[:, :2].sum(axis=1, keepdims=True),
                                   full[:, :2]/full[:, :2].sum(axis=1, keepdims=True), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(result[:, :2].argmax(axis=1), full[:, :2].argmax(axis=1))
        np.testing.assert_array_equal(full, saved)

    def test_ha_only_rejects_undefined_conditional_or_bad_probabilities(self):
        for full, dead in [(np.array([[0., 0., 1.]]), np.array([.5])),
                           (np.array([[.3, .3, .4]]), np.array([1.])),
                           (np.array([[.3, .3, .3]]), np.array([.5])),
                           (np.array([[.3, .3, .4]]), np.array([np.nan]))]:
            with self.assertRaises(ValueError): diag.ha_only_probabilities(full, dead)

    def test_three_ablations_use_recomputed_heads_and_full_production_path(self):
        item, model = self.item(8, [[1, 1, 3], [2, 5, 7]]), self.model()
        item['usable'][1] = False
        home, dead = np.full(8, .7), np.full(8, .2)
        full = np.tile([.1, .3, .6], (8, 1))
        with patch.object(diag.robust_control, 'predict_home', return_value=home) as h, \
             patch.object(diag, 'predict_dead', return_value=dead) as d, \
             patch.object(diag, 'joint_probabilities', return_value=full) as joint, \
             patch.object(diag, 'decode', side_effect=lambda p, fps, config: p.argmax(axis=1)):
            labels, p = diag.ablations(item, model, None)
        np.testing.assert_array_equal(p['full'], full)
        np.testing.assert_array_equal(p['ha_only'][:, 2], dead)
        np.testing.assert_array_equal(p['base'][:, 2], dead)
        self.assertEqual(''.join(labels['full']), 'XDDDXXXX')
        self.assertEqual(''.join(labels['ha_only']), 'XAAAXXXX')
        self.assertEqual(''.join(labels['base']), 'XHHHXXXX')
        self.assertIs(h.call_args.args[0], model['robust_control_model'])
        self.assertIs(d.call_args.args[1], model)
        np.testing.assert_array_equal(joint.call_args.args[0], dead)
        np.testing.assert_array_equal(joint.call_args.args[1], home)

    def test_old_upstream_arrays_and_references_cannot_enter_ablation_inputs(self):
        for key in ('base_dead', 'base_home', 'y', 'reference'):
            item = self.item(); item[key] = np.zeros(8)
            with patch.object(diag.robust_control, 'predict_home') as fit:
                with self.assertRaises(ValueError): diag.ablations(item, self.model(), None)
                fit.assert_not_called()

    def test_confidence_player_and_availability_boundaries(self):
        item = self.item(8)
        z = item['z']
        z[:, diag.FEATURE_NAMES.index('player_coverage')] = np.array([0, 1, 5, 6, 11, 12, 17, 22])/22
        z[:, diag.FEATURE_NAMES.index('ball_observed')] = [1, 0, 0, 1, 1, 1, 1, 1]
        z[:, diag.FEATURE_NAMES.index('ball_reliability')] = [1, .45, 0, 1, 1, 1, 1, 1]
        result = diag.observation_bins(item, np.array([0, .2, .4, .6, .8, 1, .19, .99]))
        np.testing.assert_array_equal(result['base_dead_confidence'], [0, 1, 2, 3, 4, 4, 0, 4])
        np.testing.assert_array_equal(result['valid_player_motion_count'], [0, 1, 1, 2, 2, 3, 3, 4])
        np.testing.assert_array_equal(result['ball_availability'], [0, 1, 2, 0, 0, 0, 0, 0])

    def test_missing_run_boundaries_and_full_run_offline_length(self):
        for frames, expected in [(15, 1), (16, 2), (50, 2), (51, 3), (125, 3), (126, 4), (200, 4), (201, 5)]:
            item = self.item(frames+2)
            item['z'][[0, -1], 0] = 1
            bins = diag.observation_bins(item, np.zeros(frames+2))
            np.testing.assert_array_equal(bins['missing_run_seconds'][1:-1], expected)
            np.testing.assert_array_equal(bins['missing_run_edge_censor'][1:-1], 1)

    def test_missing_runs_reset_at_adjacent_periods_and_gaps_are_unscored(self):
        item = self.item(44, [[1, 1, 20], [2, 21, 40]])
        bins = diag.observation_bins(item, np.zeros(44))
        self.assertTrue((bins['missing_run_seconds'][1:41] == 2).all())
        self.assertTrue((bins['missing_run_edge_censor'][1:41] == 4).all())
        for value in bins.values():
            self.assertTrue((value[[0, 41, 42, 43]] == 255).all())

    def test_missing_left_right_edges_are_separate(self):
        item = self.item(8); item['z'][3:5, 0] = 1
        bins = diag.observation_bins(item, np.zeros(8))
        np.testing.assert_array_equal(bins['missing_run_edge_censor'], [2, 2, 2, 0, 0, 3, 3, 3])

    def test_invalid_observation_semantics_fail_before_diagnosis(self):
        for column, value in [('ball_observed', .2), ('player_coverage', .123), ('ball_reliability', np.nan)]:
            item = self.item(); item['z'][0, diag.FEATURE_NAMES.index(column)] = value
            with self.assertRaises(ValueError): diag.observation_bins(item, np.zeros(8))

    def short(self, reference, prediction=None, blocks=None):
        reference = np.array(list(reference))
        prediction = reference.copy() if prediction is None else np.array(list(prediction))
        return diag.short_false_dead(reference, {n: prediction.copy() for n in diag.NAMES}, blocks or [[1, 0, len(reference)-1]], 25)

    def test_short_exact_two_second_exclusion(self):
        self.assertEqual(self.short('D'+'H'*49+'D')['n_gt'], 1)
        self.assertEqual(self.short('D'+'H'*50+'D')['n_gt'], 0)

    def test_short_spells_do_not_bridge_adjacent_periods_or_unknown(self):
        halves = self.short('H'*40, blocks=[[1, 0, 19], [2, 20, 39]])
        self.assertEqual(halves['n_gt'], 2)
        self.assertEqual(halves['touches_scored_run_edge'], 2)
        self.assertEqual(halves['censored_reference_frames'], 40)
        unknown = self.short('D'+'H'*10+'X'+'H'*10+'D')
        self.assertEqual(unknown['n_gt'], 2)
        self.assertEqual(unknown['touches_scored_run_edge'], 2)

    def test_short_interruption_and_erasure_distinguish_abstention_team_errors(self):
        ref = 'D'+'H'*10+'D'
        for middle, erased, touched, mass in [('D'*10, 1, 1, 10), ('X'*10, 0, 0, 0),
                                              ('A'*10, 0, 0, 0), ('H'*5+'D'*5, 0, 1, 5)]:
            result = self.short(ref, 'D'+middle+'D')
            self.assertEqual(result['touches_scored_run_edge'], 0)
            value = result['models']['full']
            self.assertEqual(value['spells_erased_by_dead'], erased)
            self.assertEqual(value['spells_with_any_dead'], touched)
            self.assertEqual(value['false_dead_frames'], mass)
            self.assertEqual(value['erasure_rate'], erased)
            self.assertEqual(value['interruption_rate'], touched)

    def test_empty_short_support_rates_are_none(self):
        value = self.short('DDDD')['models']['base']
        self.assertIsNone(value['false_dead_frame_fraction'])
        self.assertIsNone(value['erasure_rate'])
        self.assertIsNone(value['interruption_rate'])

    def test_diagnostic_bins_partition_same_known_mask_and_x_errors_count(self):
        item = self.item(8)
        reference = np.array(list('HADHXADX'))
        labels = {n: np.array(list('DDXAXHDX')) for n in diag.NAMES}
        probs = {n: np.tile([.7, .2, .1], (8, 1)) for n in diag.NAMES}
        bins = diag.observation_bins(item, np.full(8, .1))
        result = diag.aggregate_fixture(reference, labels, probs, bins, item['blocks'], 25)
        total = result['models']['base']
        self.assertEqual(total['known_frames'], 6)
        self.assertEqual(total['false_dead'], 2)
        self.assertEqual(total['missed_dead'], 1)
        self.assertEqual(total['predicted_unknown'], 1)
        self.assertEqual(total['decoded_team_correct'], 0)
        for grouping in result['bins'].values():
            self.assertEqual(sum(row['base']['known_frames'] for row in grouping.values()), total['known_frames'])
            self.assertEqual(sum(row['base']['false_dead'] for row in grouping.values()), total['false_dead'])

    def test_paired_error_attribution_is_directional(self):
        item = self.item(4)
        y = np.array(list('HADD'))
        labels = dict(base=np.array(list('HADH')), ha_only=np.array(list('DADD')), full=np.array(list('HADX')))
        p = {n: np.tile([.4, .4, .2], (4, 1)) for n in diag.NAMES}
        result = diag.aggregate_fixture(y, labels, p, diag.observation_bins(item, np.full(4, .2)), item['blocks'], 25)
        self.assertEqual(result['paired']['base->ha_only'], dict(added_false_dead=1, removed_false_dead=0, recovered_true_dead=1, lost_true_dead=0))
        self.assertEqual(result['paired']['ha_only->full'], dict(added_false_dead=0, removed_false_dead=1, recovered_true_dead=0, lost_true_dead=1))

    def test_full_freeze_required_before_reference_marker(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            with patch.object(diag, 'verify_freeze', side_effect=ValueError('missing predictions')), \
                 patch.object(diag, 'read_reference') as reference:
                with self.assertRaises(ValueError): diag.score_one(out, diag.TRAIN_IDS[0], {}, {})
                reference.assert_not_called()
                self.assertFalse((out/'reference-access.json').exists())

    def test_fixture_scoring_failure_is_consumed_even_when_called_directly(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            with patch.object(diag, 'verify_freeze'), patch.object(diag, 'access_value', return_value={}), \
                 patch.object(diag, 'read_reference', side_effect=RuntimeError('failure')) as reference:
                with self.assertRaises(RuntimeError): diag.score_one(out, diag.TRAIN_IDS[0], {'protocol_sha256': 'synthetic'}, {})
                with self.assertRaises(FileExistsError): diag.score_one(out, diag.TRAIN_IDS[0], {'protocol_sha256': 'synthetic'}, {})
                self.assertEqual(reference.call_count, 1)


if __name__ == '__main__': unittest.main()
