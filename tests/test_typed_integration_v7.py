"""Production dispatch checks using invented tracking, never match references."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from possession import learned, robust_control, transition, typed_boundary
from possession.config import Config
from possession.decoder import decode
from tests.test_boundary import invented_model as invented_boundary
from tests.test_control import invented_tracking
from tests.test_typed_boundary_v7 import invented_model as invented_typed


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = [(1, 0, 89), (2, 100, 199)]
PITCH = (105.3, 68)
FPS = 25


def baseline():
    return learned.load_model(ROOT / 'models/baselines/v1.5.json')


def candidate(contact=False):
    model = baseline()
    head = invented_typed(contact)
    # Different subprior evidence makes all three vectors nontrivial and distinct.
    head['intercepts'] = (np.asarray(head['intercepts']) - [1., 2., 3.]).tolist()
    head['coefficients'][0] = [.08, -.03, .05]
    if contact:
        head['coefficients'][-1] = [.03, -.02, .01]
    model['typed_boundary_model'] = head
    return model


class TypedIntegrationV7Tests(unittest.TestCase):
    def test_production_dispatches_three_vectors_and_preserves_joint_emissions(self):
        tracking = invented_tracking()
        old = learned.infer_learned(tracking, [], BLOCKS, PITCH, FPS, Config(), baseline())
        for contact in (False, True):
            with self.subTest(contact=contact):
                model = candidate(contact)
                learned.validate_model(model)
                config = learned.resolve_config(Config(), model)
                with patch('possession.learned.decode', wraps=decode) as dispatch:
                    actual = learned.infer_learned(tracking, [], BLOCKS, PITCH, FPS, Config(), model)
                np.testing.assert_array_equal(actual[1], old[1])
                x = robust_control.feature_matrix(tracking, actual[2], BLOCKS, PITCH, FPS)
                z = learned.feature_matrix(tracking, actual[2], BLOCKS, PITCH, FPS)
                signals = typed_boundary.evidence(model['typed_boundary_model'], x, z, actual[1], BLOCKS, FPS, config)
                self.assertEqual(dispatch.call_count, len(BLOCKS))
                expected = np.full(200, 'X', dtype='<U1')
                for call, (_, start, last) in zip(dispatch.call_args_list, BLOCKS):
                    local = {key: values[start:last + 1] for key, values in signals.items()}
                    for key, values in local.items():
                        np.testing.assert_array_equal(call.kwargs[key], values)
                    self.assertIsNone(call.kwargs['early_exit_allowed'])
                    np.testing.assert_array_equal(call.args[0], actual[1][start:last + 1])
                    expected[start:last + 1] = np.array(list('HAD'))[decode(
                        actual[1][start:last + 1], FPS, config, **local)]
                for key, base_cost in (('team_switch_costs', config.team_switch_cost),
                                       ('stop_costs', config.stop_cost),
                                       ('restart_costs', config.restart_cost)):
                    self.assertTrue(np.any(signals[key] != base_cost), key)
                np.testing.assert_array_equal(actual[0], expected)
                np.testing.assert_array_equal(actual[0][90:100], 'X')
                self.assertEqual(actual[4], [])

    def test_second_half_and_gap_changes_cannot_change_first_half(self):
        tracking = invented_tracking()
        changed = copy.deepcopy(tracking)
        changed.ball[90:] = [.9, .9]
        changed.players[90:] = [.1, .1]
        changed.player_speed[90:] = 11.
        changed.ball_speed[90:] = 35.
        model = candidate(contact=True)
        results, calls = [], []
        for value in (tracking, changed):
            with patch('possession.learned.decode', wraps=decode) as dispatch:
                results.append(learned.infer_learned(value, [], BLOCKS, PITCH, FPS, Config(), model))
            calls.append(dispatch.call_args_list)
        for index in (0, 1):
            np.testing.assert_array_equal(results[0][index][:90], results[1][index][:90])
        for key in ('team_switch_costs', 'stop_costs', 'restart_costs'):
            np.testing.assert_array_equal(calls[0][0].kwargs[key], calls[1][0].kwargs[key])
        self.assertGreater(np.max(np.abs(results[0][1][100:] - results[1][1][100:])), .001)
        for result in results:
            np.testing.assert_array_equal(result[0][90:100], 'X')

    def test_absent_head_uses_unchanged_baseline_and_prior_head_is_exactly_neutral(self):
        tracking = invented_tracking()
        model = baseline()
        config = learned.resolve_config(Config(), model)
        with patch('possession.typed_boundary.evidence', side_effect=AssertionError('disabled head called')):
            with patch('possession.learned.decode', wraps=decode) as dispatch:
                actual = learned.infer_learned(tracking, [], BLOCKS, PITCH, FPS, Config(), model)
        expected = np.full(200, 'X', dtype='<U1')
        for call, (_, start, last) in zip(dispatch.call_args_list, BLOCKS):
            for key in ('team_switch_costs', 'stop_costs', 'restart_costs', 'early_exit_allowed'):
                self.assertIsNone(call.kwargs[key])
            expected[start:last + 1] = np.array(list('HAD'))[decode(actual[1][start:last + 1], FPS, config)]
        np.testing.assert_array_equal(actual[0], expected)
        model['typed_boundary_model'] = invented_typed()
        neutral = learned.infer_learned(tracking, [], BLOCKS, PITCH, FPS, Config(), model)
        for index in (0, 1):
            np.testing.assert_array_equal(neutral[index], actual[index])
        self.assertEqual(neutral[4], actual[4])

    def test_validation_rejects_conflicting_heads_and_invalid_typed_schema(self):
        for conflict in ('legacy', 'transition', 'separate', 'missing_robust', 'schema'):
            with self.subTest(conflict=conflict):
                model = candidate()
                if conflict == 'legacy':
                    model['boundary_model'] = invented_boundary()
                elif conflict == 'transition':
                    model['transition_model'] = transition.from_boundary(invented_boundary())
                elif conflict == 'separate':
                    model['decoder_mode'] = 'separate'
                elif conflict == 'missing_robust':
                    model.pop('robust_control_model')
                    for key in ('temporal_model', 'expert_gate', 'joint_model'):
                        model.pop(key, None)
                else:
                    model['typed_boundary_model']['classes'] = ['none', 'restart', 'stop', 'team_change']
                with self.assertRaises(ValueError):
                    learned.validate_model(model)

    def test_runtime_separate_override_cannot_bypass_typed_joint_requirement(self):
        with self.assertRaisesRegex(ValueError, 'Typed boundaries require joint decoding'):
            learned.infer_learned(invented_tracking(), [], BLOCKS, PITCH, FPS, Config(), candidate(), mode='separate')


if __name__ == '__main__':
    unittest.main()
