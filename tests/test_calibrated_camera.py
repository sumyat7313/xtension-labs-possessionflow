"""Camera calibration checks using invented tracking and analytic homographies."""
import copy
import inspect
import json
import unittest

import numpy as np

from possession.io import Tracking
from tools.calibrated_camera import (fit_profile, corrupt_calibrated_tracking,
                                     project_points, validate_profile, _inside)


def fixture(n=400):
    points = np.array([[x, y] for x in (.3, .4, .5, .6, .7) for y in (.4, .6)])
    players = np.broadcast_to(points, (n, len(points), 2)).copy()
    ball = np.tile([.5, .5], (n, 1))
    ball[n // 3:n // 2] = np.nan
    players[n // 2:, [1, 4, 7]] = np.nan
    speeds = np.where(np.isfinite(players).all(axis=2), 2., np.nan)
    tracking = Tracking(players, speeds, ball, np.where(np.isfinite(ball).all(axis=1), 3., np.nan),
                        np.repeat([0, 1], 5), ("Home", "Away"))
    matrices = np.broadcast_to(np.diag([1920., 1080., 1.]), (n, 3, 3)).copy()
    return tracking, matrices, np.ones(n, bool), np.full(n, .25)


class CalibratedCameraTests(unittest.TestCase):
    def fit(self, tracking=None, matrices=None, valid=None, errors=None, blocks=None):
        if tracking is None:
            tracking, matrices, valid, errors = fixture()
        return fit_profile(tracking, matrices, valid, errors,
                           blocks or [(1, 0, len(tracking.ball) - 1)], (105., 68.), 25)

    def corrupt(self, tracking, profile, seed=152, blocks=None):
        return corrupt_calibrated_tracking(tracking, blocks or [(1, 0, len(tracking.ball) - 1)],
                                           (105., 68.), 25, seed, profile)

    def test_normalized_projection_and_inverse_footprints(self):
        matrix = np.diag([1920., 1080., 1.])[None]
        result = project_points(matrix, np.array([[[.25, .75], [.5, .5]]]))
        np.testing.assert_allclose(result, [[[480., 810.], [960., 540.]]])
        profile = self.fit()
        for item in profile["footprints"]:
            polygon = np.asarray(item["relative_polygon_metres"])
            # Full-pitch inverse footprint, coarsened to two-metre quantization.
            self.assertLessEqual(abs(np.ptp(polygon[:, 0]) - 105), 2)
            self.assertLessEqual(abs(np.ptp(polygon[:, 1]) - 68), 2)

    def test_profile_is_compact_aggregate_without_frame_sequences(self):
        profile = self.fit()
        self.assertEqual(profile, self.fit())
        self.assertEqual(profile["active_frames"], 400)
        self.assertAlmostEqual(profile["ball_observed_fraction"], 333 / 400, places=3)
        self.assertLess(len(json.dumps(profile)), 8000)
        self.assertLessEqual(len(profile["footprints"]), 8)
        for item in profile["footprints"]:
            self.assertGreaterEqual(item["count"], 16)
            np.testing.assert_allclose(np.asarray(item["relative_polygon_metres"]) % 2, 0)
            self.assertEqual(set(item), {"anchor", "count", "relative_polygon_metres"})
        self.assertNotIn("frames", profile)
        self.assertNotIn("matrices", profile)
        self.assertNotIn("ball_positions", profile)
        self.assertNotIn("player_positions", profile)

    def test_projective_polygon_membership_matches_pixel_bounds(self):
        matrix = np.array([[1800., 100., 20.], [20., 1000., 50.], [.05, .03, 1.]])
        corners = np.array([[0, 0, 1], [1920, 0, 1], [1920, 1080, 1], [0, 1080, 1]])
        homogeneous = corners @ np.linalg.inv(matrix).T
        polygon = homogeneous[:, :2] / homogeneous[:, 2:]
        points = np.random.default_rng(122).uniform(-.5, 1.5, (1, 1000, 2))
        pixels = project_points(matrix[None], points)
        expected = ((pixels >= 0) & (pixels <= [1920, 1080])).all(axis=2)
        np.testing.assert_array_equal(_inside(points, polygon[None]), expected)

    def test_profile_is_independent_of_slot_and_team_naming(self):
        tracking, matrices, valid, errors = fixture()
        before = self.fit(tracking, matrices, valid, errors)
        order = np.random.default_rng(197).permutation(len(tracking.teams))
        tracking.players = tracking.players[:, order]
        tracking.player_speed = tracking.player_speed[:, order]
        tracking.teams = 1 - tracking.teams[order]
        tracking.team_names = ("Different", "Names")
        self.assertEqual(before, self.fit(tracking, matrices, valid, errors))

    def test_quality_filter_and_no_cross_half_run_concatenation(self):
        tracking, matrices, valid, errors = fixture(200)
        tracking.ball[:] = np.nan
        tracking.ball_speed[:] = np.nan
        valid[:20] = False
        matrices[:20] = np.nan
        errors[20:40] = 20
        profile = self.fit(tracking, matrices, valid, errors, [(1, 0, 99), (2, 100, 199)])
        self.assertEqual(profile["usable_homography_fraction"], .8)
        self.assertEqual(sum(profile["ball_missing_run_histogram"]), 2)
        edges = np.asarray(profile["run_duration_edges_seconds"])
        four_seconds_bin = np.searchsorted(edges, 4, side="right") - 1
        self.assertEqual(profile["ball_missing_run_histogram"][four_seconds_bin], 2)

    def test_halftime_and_outside_frames_do_not_fit_profile(self):
        tracking, matrices, valid, errors = fixture(400)
        blocks = [(1, 50, 149), (2, 250, 349)]
        before = self.fit(tracking, matrices, valid, errors, blocks)
        outside = np.ones(400, bool)
        outside[50:150] = outside[250:350] = False
        tracking.ball[outside] = [100, 100]
        tracking.players[outside] = [-100, -100]
        matrices[outside] = 0
        errors[outside] = 100
        after = self.fit(tracking, matrices, valid, errors, blocks)
        self.assertEqual(before, after)

    def test_no_mutation_determinism_and_no_recovered_inputs(self):
        tracking, *_ = fixture(1000)
        before = copy.deepcopy(tracking)
        profile = self.fit()
        result = self.corrupt(tracking, profile)
        repeated = self.corrupt(tracking, profile)
        for name in ("players", "player_speed", "ball", "ball_speed", "teams"):
            np.testing.assert_array_equal(getattr(tracking, name), getattr(before, name))
            np.testing.assert_array_equal(getattr(result, name), getattr(repeated, name))
            self.assertFalse(np.shares_memory(getattr(result, name), getattr(tracking, name)))
        self.assertTrue(np.isnan(result.ball[np.isnan(tracking.ball).any(axis=1)]).all())
        self.assertTrue(np.isnan(result.ball_speed[~np.isfinite(result.ball).all(axis=1)]).all())
        self.assertTrue(np.isnan(result.player_speed[~np.isfinite(result.players).all(axis=2)]).all())

    def test_calibrated_missing_rate_is_applied_without_double_counting(self):
        tracking, *_ = fixture(2000)
        tracking.ball[:] = [.5, .5]
        tracking.ball_speed[:] = 3
        profile = self.fit()
        profile["ball_observed_fraction"] = .6
        # Analytic full-pitch footprint excludes no central ball; missingness
        # therefore reaches exactly the calibrated total, not geometry + rate.
        result = self.corrupt(tracking, profile)
        self.assertAlmostEqual(np.isfinite(result.ball).all(axis=1).mean(), .6)
        tracking.ball[:1000] = np.nan
        result = self.corrupt(tracking, profile)
        self.assertAlmostEqual(np.isfinite(result.ball).all(axis=1).mean(), .5)

    def test_half_isolation_and_ungraded_frames_unchanged(self):
        tracking, *_ = fixture(500)
        blocks = [(1, 10, 199), (2, 250, 489)]
        profile = self.fit()
        original = self.corrupt(tracking, profile, blocks=blocks)
        changed = copy.deepcopy(tracking)
        changed.players[:200] = np.nan
        changed.ball[:200] = np.nan
        altered = self.corrupt(changed, profile, blocks=blocks)
        for name in ("players", "player_speed", "ball", "ball_speed"):
            np.testing.assert_array_equal(getattr(original, name)[250:490], getattr(altered, name)[250:490])
            outside = np.r_[0:10, 200:250, 490:500]
            np.testing.assert_array_equal(getattr(original, name)[outside], getattr(tracking, name)[outside])

    def test_all_missing_input_stays_missing(self):
        tracking, *_ = fixture(100)
        for name in ("players", "player_speed", "ball", "ball_speed"):
            getattr(tracking, name)[:] = np.nan
        with np.errstate(all="raise"):
            result = self.corrupt(tracking, self.fit())
        for name in ("players", "player_speed", "ball", "ball_speed"):
            self.assertTrue(np.isnan(getattr(result, name)).all())

    def test_malformed_inputs_fail_closed(self):
        tracking, matrices, valid, errors = fixture()
        for invalid in (np.zeros_like(matrices), np.full_like(matrices, np.nan)):
            with self.assertRaises(ValueError):
                self.fit(tracking, invalid, valid, errors)
        with self.assertRaises(ValueError):
            self.fit(tracking, matrices[:-1], valid, errors)
        with self.assertRaises(ValueError):
            self.fit(tracking, matrices, np.full(len(valid), 2), errors)
        profile = self.fit()
        profile["footprints"][0]["relative_polygon_metres"] = [[0, 0]] * 4
        with self.assertRaises(ValueError):
            validate_profile(profile)

    def test_apis_do_not_accept_labels(self):
        self.assertEqual(list(inspect.signature(fit_profile).parameters),
                         ["tracking", "matrices", "valid", "errors", "blocks", "pitch", "fps", "width", "height"])
        self.assertEqual(list(inspect.signature(corrupt_calibrated_tracking).parameters),
                         ["tracking", "blocks", "pitch", "fps", "seed", "profile"])


if __name__ == "__main__":
    unittest.main()
