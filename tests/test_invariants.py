"""Independent symmetry checks; no competition records or labels are opened."""

import copy
import unittest

import numpy as np

from possession.config import Config
from possession.io import Tracking
from possession.pipeline import infer


def invented_match():
    n = 400
    rng = np.random.default_rng(72851)
    teams = np.repeat([0, 1], 4)
    players = np.empty((n, 8, 2))
    players[:, :4] = [0.25, 0.4]
    players[:, 4:] = [0.75, 0.6]
    players += rng.normal(0, 0.003, players.shape)
    ball = players[:, 0].copy()
    ball[175:325] = players[175:325, 4]
    ball[60:80] = np.nan
    return Tracking(players, np.full((n, 8), 2.0), ball, np.full(n, 2.0),
                    teams, ("Home", "Away"))


def labels(data, blocks=None):
    blocks = blocks or [(1, 0, len(data.ball) - 1)]
    return infer(data, [], blocks, (105.3, 68), 25, Config())[0]


class InvarianceTests(unittest.TestCase):
    def test_pitch_reflection_preserves_decisions(self):
        original = invented_match()
        reflected = copy.deepcopy(original)
        reflected.players = 1 - reflected.players
        reflected.ball = 1 - reflected.ball
        np.testing.assert_array_equal(labels(original), labels(reflected))

    def test_swapping_team_names_and_membership_swaps_labels(self):
        original = invented_match()
        swapped = copy.deepcopy(original)
        swapped.teams = 1 - swapped.teams
        swapped.team_names = swapped.team_names[::-1]
        expected = np.array([{"H": "A", "A": "H", "D": "D", "X": "X"}[x]
                             for x in labels(original)])
        np.testing.assert_array_equal(expected, labels(swapped))

    def test_track_slot_order_is_irrelevant(self):
        original = invented_match()
        reordered = copy.deepcopy(original)
        order = np.array([7, 1, 5, 3, 0, 2, 6, 4])
        reordered.players = reordered.players[:, order]
        reordered.player_speed = reordered.player_speed[:, order]
        reordered.teams = reordered.teams[order]
        np.testing.assert_array_equal(labels(original), labels(reordered))

    def test_per_frame_identity_switches_within_team_are_irrelevant(self):
        original = invented_match()
        switched = copy.deepcopy(original)
        rng = np.random.default_rng(3912)
        for frame in range(len(original.ball)):
            order = np.r_[rng.permutation(4), rng.permutation(4) + 4]
            switched.players[frame] = original.players[frame, order]
            switched.player_speed[frame] = original.player_speed[frame, order]
        np.testing.assert_array_equal(labels(original), labels(switched))

    def test_later_half_cannot_change_previous_half(self):
        original = invented_match()
        changed = copy.deepcopy(original)
        changed.ball[200:] = np.nan
        changed.players[200:] = np.nan
        changed.player_speed[200:] = np.nan
        blocks = [(1, 0, 179), (2, 200, 399)]
        np.testing.assert_array_equal(labels(original, blocks)[:200], labels(changed, blocks)[:200])


if __name__ == "__main__":
    unittest.main()
