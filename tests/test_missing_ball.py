"""Missing-ball behavior on invented observations, never challenge labels."""

import copy
import unittest

import numpy as np

from possession import missing_ball
from possession.config import Config
from possession.features import extract
from possession.io import Tracking


def invented_tracking():
    n = 600
    rng = np.random.default_rng(6122)
    players = np.empty((n, 12, 2))
    players[:, :6] = [.25, .4]
    players[:, 6:] = [.75, .6]
    players += rng.normal(0, .015, players.shape)
    ball = players[:, 0].copy()
    ball[40:280] = np.nan
    ball[300:] = players[300:, 6]
    ball[340:420] = np.nan
    speeds = rng.uniform(1, 6, (n, 12))
    return Tracking(players, speeds, ball, np.full(n, 2.),
                    np.repeat([0, 1], 6), ("Home", "Away"))


def matrix(tracking, blocks=None):
    blocks = blocks if blocks is not None else [(1, 0, len(tracking.ball) - 1)]
    features = extract(tracking, (105, 68), blocks, 25, Config())
    return missing_ball.feature_matrix(tracking, features, blocks, (105, 68), 25)


class MissingBallTests(unittest.TestCase):
    def test_team_swap_negates_every_feature(self):
        original = invented_tracking()
        changed = copy.deepcopy(original)
        changed.teams = 1 - changed.teams
        np.testing.assert_array_equal(matrix(original), -matrix(changed))

    def test_slot_permutations_and_coordinate_reflection(self):
        original = invented_tracking()
        changed = copy.deepcopy(original)
        order = np.array([8, 3, 0, 7, 10, 1, 4, 11, 5, 2, 6, 9])
        changed.players = changed.players[:, order]
        changed.player_speed = changed.player_speed[:, order]
        changed.teams = changed.teams[order]
        np.testing.assert_array_equal(matrix(original), matrix(changed))
        # Permutations may also change at every frame within each team: no IDs.
        changed = copy.deepcopy(original)
        changed.players[::2, :6] = changed.players[::2, :6][:, ::-1]
        changed.player_speed[::2, :6] = changed.player_speed[::2, :6][:, ::-1]
        np.testing.assert_array_equal(matrix(original), matrix(changed))
        changed = copy.deepcopy(original)
        changed.players = 1 - changed.players
        changed.ball = 1 - changed.ball
        np.testing.assert_allclose(matrix(original), matrix(changed), atol=1e-12)

    def test_half_isolation_and_outside_blocks(self):
        original = invented_tracking()
        changed = copy.deepcopy(original)
        changed.ball[300:] = np.nan
        changed.players[300:] = np.nan
        blocks = [(1, 0, 280), (2, 300, 599)]
        np.testing.assert_array_equal(matrix(original, blocks)[:300], matrix(changed, blocks)[:300])
        np.testing.assert_array_equal(matrix(original, blocks)[281:300], 0)
        # No observation at all in half two: it must not inherit half one's ball.
        np.testing.assert_array_equal(matrix(changed, blocks)[300:], 0)

    def test_context_decays_expires_and_never_fills_ball(self):
        tracking = invented_tracking()
        tracking.ball[40:] = np.nan
        tracking.players[:] = tracking.players[0]
        tracking.player_speed[:] = 2
        before = tracking.ball.copy()
        values = matrix(tracking)
        self.assertGreater(values[45, 0], values[60, 0])
        self.assertGreater(values[60, 0], values[100, 0])
        np.testing.assert_array_equal(values[240:, :9], 0)
        np.testing.assert_array_equal(values[40:, [1, 3, 5, 7, 8]], 0)
        np.testing.assert_array_equal(tracking.ball, before)

    def test_reliable_ball_disables_branch_and_no_observations_neutral(self):
        tracking = invented_tracking()
        values = matrix(tracking)
        features = extract(tracking, (105, 68), [(1, 0, 599)], 25, Config())
        self.assertTrue((features.reliability == 1).any())
        np.testing.assert_array_equal(values[features.reliability == 1], 0)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values.shape, (600, len(missing_ball.FEATURE_NAMES)))
        tracking.players[:] = np.nan
        tracking.ball[:] = np.nan
        tracking.player_speed[:] = np.nan
        np.testing.assert_array_equal(matrix(tracking), 0)

    def test_missing_team_coverage_suppresses_shape_and_anchor_support(self):
        tracking = invented_tracking()
        tracking.players[40:280, 6:] = np.nan
        tracking.player_speed[40:280, 6:] = np.nan
        values = matrix(tracking)
        np.testing.assert_array_equal(values[45:270, 9:], 0)
        np.testing.assert_array_equal(values[45:270, 6:8], 0)

    def test_short_missing_clip_never_uses_sentinel_as_index(self):
        source = invented_tracking()
        for n in (1, 2, 10, 100):
            item = Tracking(source.players[:n].copy(), source.player_speed[:n].copy(),
                            np.full((n, 2), np.nan), source.ball_speed[:n].copy(),
                            source.teams.copy(), source.team_names)
            values = matrix(item)
            np.testing.assert_array_equal(values[:, :9], 0)
            self.assertTrue(np.isfinite(values).all())

    def test_invalid_parameters_and_blocks(self):
        tracking = invented_tracking()
        blocks = [(1, 0, 599)]
        features = extract(tracking, (105, 68), blocks, 25, Config())
        for fps in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                missing_ball.feature_matrix(tracking, features, blocks, (105, 68), fps)
        for pitch in ((0, 68), (105,), (105, float("inf"))):
            with self.assertRaises(ValueError):
                missing_ball.feature_matrix(tracking, features, blocks, pitch, 25)
        with self.assertRaises(ValueError):
            matrix(tracking, [(1, 0, 301), (2, 300, 599)])


if __name__ == "__main__":
    unittest.main()
