import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from possession import robust_control as robust, control, learned
from possession.config import Config
from possession.features import extract
from possession.tags import build
from test_control import invented_tracking


class RobustControlTests(unittest.TestCase):
    def example(self):
        rng = np.random.default_rng(920)
        x = rng.normal(size=(300, len(robust.FEATURE_NAMES)))
        return x, np.where(x[:, 0] + x[:, -1] > 0, "H", "A")

    def test_fit_symmetry_determinism_and_roundtrip(self):
        x, y = self.example()
        model = robust.fit_model(x, y)
        self.assertEqual(model, robust.fit_model(x, y))
        np.testing.assert_allclose(robust.predict_home(model, x), 1-robust.predict_home(model, -x), atol=2e-16)
        self.assertGreater(((robust.predict_home(model, x) > .5) == (y == "H")).mean(), .97)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            robust.save_model(path, model)
            np.testing.assert_array_equal(robust.predict_home(model, x), robust.predict_home(robust.load_model(path), x))

    def test_excluded_labels_and_features_cannot_affect_fit(self):
        x, y = self.example()
        mask = np.arange(x.shape[1]) < 52
        one = robust.fit_model(x, y, feature_mask=mask)
        changed = x.copy(); changed[:, 52:] = 1e12
        two = robust.fit_model(np.vstack((changed, np.ones((3, x.shape[1])) * 1e9)),
            np.r_[y, ["D", "X", "H"]], sample_weight=np.r_[np.ones(len(y)+2), 0], feature_mask=mask)
        np.testing.assert_array_equal(one["coefficients"], two["coefficients"])
        np.testing.assert_array_equal(one["scale"], two["scale"])
        np.testing.assert_array_equal(np.asarray(one["coefficients"])[52:], 0)
        with self.assertRaises(ValueError): robust.fit_model(x, y, feature_mask=np.ones(x.shape[1]))
        invalid = copy.deepcopy(one); invalid["coefficients"][-1] = 1
        with self.assertRaises(ValueError): robust.validate_model(invalid)

    def test_feature_prefix_and_production_fusion(self):
        tracking = invented_tracking(); blocks = [(1, 0, 199)]; pitch = (105, 68)
        config = Config(); features = extract(tracking, pitch, blocks, 25, config)
        matrix = robust.feature_matrix(tracking, features, blocks, pitch, 25)
        np.testing.assert_array_equal(matrix[:, :52], control.feature_matrix(tracking, features, blocks, pitch, 25))
        changed = copy.deepcopy(tracking); changed.teams = 1-changed.teams
        swapped = extract(changed, pitch, blocks, 25, config)
        np.testing.assert_array_equal(matrix, -robust.feature_matrix(changed, swapped, blocks, pitch, 25))
        root = Path(__file__).resolve().parents[1]
        model = learned.load_model(root / "models/baselines/v1.2.json")
        tags = build([], features, tracking.team_names, 200, 25, blocks, config)
        dead_x = learned.feature_matrix(tracking, features, blocks, pitch, 25)
        old = learned.learned_probabilities(features, tags, dead_x, model, matrix[:, :52])
        dead = learned.predict_dead(dead_x, model)
        home = np.clip(control.predict_home(model["control_model"], matrix[:, :52]), 1e-8, 1-1e-8)
        home = learned._sigmoid(np.clip(np.log(home)-np.log1p(-home)+tags.team_log_odds, -8, 8))
        np.testing.assert_array_equal(old, np.column_stack(((1-dead)*home, (1-dead)*(1-home), dead)))
        x, y = self.example(); model["robust_control_model"] = robust.fit_model(x, y)
        production = learned.infer_learned(tracking, [], blocks, pitch, 25, config, model)
        expected = learned.fuse_control_probabilities(dead, robust.predict_home(model["robust_control_model"], matrix), tags.team_log_odds)
        np.testing.assert_array_equal(production[1], expected)
        invalid = copy.deepcopy(model); invalid["robust_control_model"]["intercept"] = 1
        with self.assertRaises(ValueError): learned.validate_model(invalid)
