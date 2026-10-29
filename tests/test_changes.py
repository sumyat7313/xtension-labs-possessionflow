"""Possession-boundary features use invented observations, never match labels."""

import copy
import unittest

import numpy as np

from possession import changes
from possession.config import Config
from possession.features import extract
from possession.io import Tracking


def tracking_example():
    n = 180
    rng = np.random.default_rng(4829)
    players = np.empty((n, 6, 2))
    players[:, :3] = [.48, .5]
    players[:, 3:] = [.53, .51]
    players += rng.normal(0, .003, players.shape)
    ball = np.column_stack((.44 + np.minimum(np.arange(n), 80) * .0005,
                            np.full(n, .5)))
    ball[120:140] = np.nan
    return Tracking(players, rng.uniform(0, 5, (n, 6)), ball, np.full(n, 1.3),
                    np.repeat([0, 1], 3), ("H", "A"))


def matrix(tracking, blocks=None):
    blocks = blocks or [(1, 0, len(tracking.ball) - 1)]
    features = extract(tracking, (105., 68.), blocks, 25, Config())
    return changes.feature_matrix(tracking, features, blocks, (105., 68.), 25)


class ChangeFeatureTests(unittest.TestCase):
    def test_finite_deterministic_and_input_unchanged(self):
        original = tracking_example()
        saved = copy.deepcopy(original)
        result = matrix(original)
        self.assertEqual(result.shape, (180, 28))
        self.assertEqual(len(set(changes.FEATURE_NAMES)), 28)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, matrix(original))
        for attribute in ("players", "player_speed", "ball", "ball_speed", "teams"):
            np.testing.assert_array_equal(getattr(original, attribute), getattr(saved, attribute))

    def test_team_swap_sign_and_pitch_reflection(self):
        original = tracking_example()
        swapped = copy.deepcopy(original)
        swapped.teams = 1 - swapped.teams
        np.testing.assert_array_equal(matrix(original), -matrix(swapped))
        reflected = copy.deepcopy(original)
        reflected.players = 1 - reflected.players
        reflected.ball = 1 - reflected.ball
        np.testing.assert_allclose(matrix(original), matrix(reflected), atol=2e-12)

    def test_per_frame_slot_permutation_does_not_require_identity(self):
        original = tracking_example()
        changed = copy.deepcopy(original)
        for frame in range(len(changed.ball)):
            order = np.r_[np.roll(np.arange(3), frame % 3), np.roll(np.arange(3, 6), frame % 3)]
            changed.players[frame] = original.players[frame, order]
            changed.player_speed[frame] = original.player_speed[frame, order]
        np.testing.assert_array_equal(matrix(original), matrix(changed))

    def test_halves_are_isolated_and_separator_zero(self):
        original = tracking_example()
        changed = copy.deepcopy(original)
        changed.ball[100:] = .9
        changed.players[100:] = .05
        changed.player_speed[100:] = 12
        blocks = [(1, 0, 89), (2, 100, 179)]
        a, b = matrix(original, blocks), matrix(changed, blocks)
        np.testing.assert_array_equal(a[:100], b[:100])
        np.testing.assert_array_equal(a[90:100], np.zeros((10, 28)))

    def test_missing_observations_and_empty_periods_are_neutral(self):
        example = tracking_example()
        example.ball[:] = np.nan
        np.testing.assert_array_equal(matrix(example), np.zeros((180, 28)))
        example = tracking_example()
        example.players[:] = np.nan
        np.testing.assert_array_equal(matrix(example), np.zeros((180, 28)))
        features = extract(example, (105., 68.), [], 25, Config())
        np.testing.assert_array_equal(changes.feature_matrix(example, features, [], (105., 68.), 25),
                                      np.zeros((180, 28)))

    def test_contact_evidence_responds_to_ball_deceleration(self):
        example = tracking_example()
        result = matrix(example)
        contact = changes.FEATURE_NAMES.index("deceleration_control")
        # Ball stops beside the home team at frame 80; constant-speed frames have
        # zero deceleration evidence, while the contact neighbourhood is positive.
        self.assertGreater(result[80, contact], .01)
        np.testing.assert_allclose(result[20:50, contact], 0, atol=1e-12)

    def test_bad_geometry_and_rates_fail_early(self):
        example = tracking_example()
        blocks = [(1, 0, 179)]
        features = extract(example, (105., 68.), blocks, 25, Config())
        for pitch, fps in (((105, 0), 25), ((105, 68), 0), ((105, 68), float("nan"))):
            with self.assertRaises(ValueError):
                changes.feature_matrix(example, features, blocks, pitch, fps)


if __name__ == "__main__":
    unittest.main()
