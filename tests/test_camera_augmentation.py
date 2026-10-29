"""Synthetic checks for camera corruption, with no challenge data or labels."""
import copy
import inspect
import unittest

import numpy as np

from possession.io import Tracking
from tools.camera_augmentation import corrupt_camera_tracking


def invented_tracking(n=2500):
    t = np.arange(n) / 25
    ball = np.column_stack((.5 + .28 * np.sin(t / 6), .5 + .2 * np.sin(t / 11)))
    # Two local players and a grid distributed over the complete pitch make
    # field-of-view masking and detector corruption observable in this fixture.
    grid = np.array([[x, y] for x in (.1, .3, .5, .7, .9) for y in (.2, .8)])
    players = np.broadcast_to(grid, (n, len(grid), 2)).copy()
    players[:, 0] = ball + [.005, 0]
    players[:, 5] = ball - [.015, 0]
    return Tracking(players, np.full((n, len(grid)), 2.), ball, np.full(n, 3.),
                    np.repeat([0, 1], 5), ("Home", "Away"))


class CameraAugmentationTests(unittest.TestCase):
    def run_camera(self, tracking, seed=982, blocks=None, fps=25):
        if blocks is None:
            blocks = [(1, 0, len(tracking.ball) - 1)]
        return corrupt_camera_tracking(tracking, blocks, (105., 68.), fps, seed)

    def assert_tracking_equal(self, left, right):
        for name in ("players", "player_speed", "ball", "ball_speed", "teams"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        self.assertEqual(left.team_names, right.team_names)

    def test_determinism_no_mutation_and_preserved_shapes(self):
        tracking = invented_tracking()
        before = copy.deepcopy(tracking)
        first = self.run_camera(tracking)
        self.assert_tracking_equal(first, self.run_camera(tracking))
        self.assert_tracking_equal(tracking, before)
        for name in ("players", "player_speed", "ball", "ball_speed", "teams"):
            self.assertEqual(getattr(first, name).shape, getattr(tracking, name).shape)
            self.assertFalse(np.shares_memory(getattr(first, name), getattr(tracking, name)))
        other = self.run_camera(tracking, seed=983)
        self.assertFalse(np.array_equal(first.ball, other.ball, equal_nan=True))

    def test_camera_masks_observations_and_keeps_valid_values(self):
        result = self.run_camera(invented_tracking())
        players_visible = np.isfinite(result.players).all(axis=2)
        ball_visible = np.isfinite(result.ball).all(axis=1)
        self.assertGreater(players_visible.mean(), .1)
        self.assertLess(players_visible.mean(), .8)
        self.assertGreater(ball_visible.mean(), .3)
        self.assertLess(ball_visible.mean(), .98)
        np.testing.assert_array_equal(np.isfinite(result.players[..., 0]),
                                      np.isfinite(result.players[..., 1]))
        self.assertTrue(np.isnan(result.player_speed[~players_visible]).all())
        self.assertTrue(np.isnan(result.ball_speed[~ball_visible]).all())
        self.assertTrue((result.player_speed[players_visible] >= 0).all())
        self.assertTrue((result.ball_speed[ball_visible] >= 0).all())

    def test_halves_isolated_and_ungraded_frames_untouched(self):
        tracking = invented_tracking(700)
        blocks = [(1, 10, 299), (2, 350, 689)]
        original = self.run_camera(tracking, blocks=blocks)
        changed = copy.deepcopy(tracking)
        changed.players[:300] = np.nan
        changed.player_speed[:300] = np.nan
        changed.ball[:300] = np.nan
        changed.ball_speed[:300] = np.nan
        altered = self.run_camera(changed, blocks=blocks)
        for name in ("players", "player_speed", "ball", "ball_speed"):
            np.testing.assert_array_equal(getattr(original, name)[350:690],
                                          getattr(altered, name)[350:690])
            indices = np.r_[0:10, 300:350, 690:700]
            np.testing.assert_array_equal(getattr(original, name)[indices],
                                          getattr(tracking, name)[indices])
        changed = copy.deepcopy(tracking)
        changed.players[350:] = np.nan
        changed.ball[350:] = np.nan
        altered = self.run_camera(changed, blocks=blocks)
        np.testing.assert_array_equal(original.players[10:300], altered.players[10:300])
        np.testing.assert_array_equal(original.ball[10:300], altered.ball[10:300])

    def test_empty_observations_do_not_create_information(self):
        tracking = invented_tracking(125)
        for name in ("players", "player_speed", "ball", "ball_speed"):
            getattr(tracking, name)[:] = np.nan
        with np.errstate(all="raise"):
            result = self.run_camera(tracking)
        for name in ("players", "player_speed", "ball", "ball_speed"):
            self.assertTrue(np.isnan(getattr(result, name)).all())

    def test_one_frame_and_low_fps_are_valid(self):
        result = self.run_camera(invented_tracking(1), fps=.1)
        self.assertEqual(result.ball.shape, (1, 2))
        self.assertFalse(np.isinf(result.players).any())

    def test_duplicate_detections_exist_in_same_team(self):
        # Duplicates are exact copies in an otherwise independently noisy
        # stream; this checks that the configured detector effect is exercised.
        result = self.run_camera(invented_tracking(10000))
        duplicated = 0
        for side in (0, 1):
            p = result.players[:, result.teams == side]
            for left in range(p.shape[1]):
                for right in range(left):
                    duplicated += int(np.all(p[:, left] == p[:, right], axis=1).sum())
        self.assertGreater(duplicated, 0)

    def test_false_ball_detections_use_surviving_players(self):
        result = self.run_camera(invented_tracking(10000))
        # Noise is otherwise independent, so exact equality identifies the
        # false-ball branch. Its source is an observed player, never a label.
        coincident = np.all(result.players == result.ball[:, None], axis=2).any(axis=1)
        self.assertGreater(int(coincident.sum()), 0)

    def test_api_has_no_reference_label_input(self):
        self.assertEqual(list(inspect.signature(corrupt_camera_tracking).parameters),
                         ["tracking", "blocks", "pitch", "fps", "seed"])

    def test_invalid_inputs_fail(self):
        tracking = invented_tracking(10)
        for fps in (0, -1, np.nan, np.inf):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.run_camera(tracking, fps=fps)
        for blocks in ([(1, -1, 4)], [(1, 0, 10)], [(1, 0, 5), (2, 5, 8)]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                self.run_camera(tracking, blocks=blocks)
        with self.assertRaises(ValueError):
            self.run_camera(tracking, seed=-1)
        with self.assertRaises(ValueError):
            corrupt_camera_tracking(tracking, [(1, 0, 9)], (105, np.nan), 25, 1)
        tracking.player_speed = np.ones((10, 9))
        with self.assertRaises(ValueError):
            self.run_camera(tracking)


if __name__ == "__main__":
    unittest.main()
