"""Checks for the learned team-control head using invented observations only."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession import control
from possession.config import Config
from possession.features import extract
from possession.io import Tracking


def invented_tracking():
    n = 200
    rng = np.random.default_rng(782)
    players = np.empty((n, 8, 2))
    players[:, :4] = [0.25, 0.4]
    players[:, 4:] = [0.75, 0.6]
    players += rng.normal(0, 0.002, players.shape)
    ball = players[:, 0].copy()
    ball[90:150] = players[90:150, 4]
    ball[35:45] = np.nan
    return Tracking(players, np.full((n, 8), 2.), ball, np.full(n, 2.),
                    np.repeat([0, 1], 4), ("Home", "Away"))


def matrix(tracking, blocks=None):
    blocks = blocks or [(1, 0, len(tracking.ball) - 1)]
    features = extract(tracking, (105.3, 68), blocks, 25, Config())
    return control.feature_matrix(tracking, features, blocks, (105.3, 68), 25)


class ControlTests(unittest.TestCase):
    def example(self):
        rng = np.random.default_rng(3102)
        X = rng.normal(size=(600, len(control.FEATURE_NAMES)))
        y = np.where(X[:, 0] - 0.3 * X[:, 1] > 0, "H", "A")
        return X, y

    def test_fit_and_serialization_are_deterministic(self):
        X, y = self.example()
        model = control.fit_model(X, y)
        self.assertEqual(model, control.fit_model(X, y))
        self.assertGreater(((control.predict_home(model, X) > .5) == (y == "H")).mean(), .96)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            control.save_model(path, model)
            np.testing.assert_array_equal(control.predict_home(model, X),
                                          control.predict_home(control.load_model(path), X))

    def test_team_swap_complements_probabilities(self):
        X, y = self.example()
        model = control.fit_model(X, y)
        np.testing.assert_allclose(control.predict_home(model, X),
                                   1 - control.predict_home(model, -X), atol=2e-16)
        original = invented_tracking()
        changed = copy.deepcopy(original)
        changed.teams = 1 - changed.teams
        np.testing.assert_array_equal(matrix(original), -matrix(changed))

    def test_slot_order_and_pitch_reflection(self):
        original = invented_tracking()
        changed = copy.deepcopy(original)
        order = np.array([7, 2, 6, 3, 0, 4, 1, 5])
        changed.players = changed.players[:, order]
        changed.player_speed = changed.player_speed[:, order]
        changed.teams = changed.teams[order]
        np.testing.assert_array_equal(matrix(original), matrix(changed))
        changed = copy.deepcopy(original)
        changed.players = 1 - changed.players
        changed.ball = 1 - changed.ball
        np.testing.assert_allclose(matrix(original), matrix(changed), atol=2e-12)

    def test_halves_and_missing_observations_are_isolated(self):
        original = invented_tracking()
        changed = copy.deepcopy(original)
        changed.ball[100:] = np.nan
        changed.players[100:] = np.nan
        blocks = [(1, 0, 89), (2, 100, 199)]
        np.testing.assert_array_equal(matrix(original, blocks)[:100], matrix(changed, blocks)[:100])
        changed.ball[:] = np.nan
        changed.players[:] = np.nan
        np.testing.assert_array_equal(matrix(changed), np.zeros((200, len(control.FEATURE_NAMES))))

    def test_unknown_dead_and_zero_weight_rows_do_not_change_fit(self):
        X, y = self.example()
        model = control.fit_model(X, y)
        larger = np.vstack((X, np.full((3, X.shape[1]), 1e6)))
        labels = np.r_[y, ["D", "X", "H"]]
        other = control.fit_model(larger, labels, sample_weight=np.r_[np.ones(len(X) + 2), 0])
        np.testing.assert_array_equal(model["coefficients"], other["coefficients"])
        np.testing.assert_array_equal(model["scale"], other["scale"])

    def test_malformed_artifacts_and_parameters_rejected(self):
        X, y = self.example()
        model = control.fit_model(X, y)
        for key, value in (("intercept", .01), ("scale", [0] * len(control.FEATURE_NAMES)),
                           ("coefficients", [float("nan")] * len(control.FEATURE_NAMES)),
                           ("mean", [0] * len(control.FEATURE_NAMES)), ("feature_names", [])):
            with self.assertRaises(ValueError):
                control.validate_model(dict(model, **{key: value}))
        for l2 in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                control.fit_model(X, y, l2=l2)
        with self.assertRaises(ValueError):
            control.fit_model(X, np.full(len(X), "D"))


if __name__ == "__main__":
    unittest.main()
