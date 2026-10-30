"""Production checks for joint emissions and local H/A transition penalties."""
import copy
from pathlib import Path
import unittest
import numpy as np
from possession import boundary, learned, robust_control, temporal
from possession.config import Config
from possession.decoder import decode
from possession.features import extract
from possession.tags import build
from test_joint_temporal import invented_model as invented_joint
from test_boundary import invented_model as invented_boundary
from test_control import invented_tracking

ROOT = Path(__file__).resolve().parents[1]


class JointIntegrationTests(unittest.TestCase):
    def test_production_joint_boundary_matches_shared_emissions_and_half_decoding(self):
        model = learned.load_model(ROOT / "models/baselines/v1.4.json")
        model["joint_model"] = invented_joint(); model["boundary_model"] = invented_boundary()
        tracking = invented_tracking(); blocks = [(1, 0, 89), (2, 100, 199)]; pitch = (105.3, 68)
        config = learned.resolve_config(Config(), model)
        actual = learned.infer_learned(tracking, [], blocks, pitch, 25, Config(), model)
        features = extract(tracking, pitch, blocks, 25, config)
        tags = build([], features, tracking.team_names, 200, 25, blocks, config)
        matrix = robust_control.feature_matrix(tracking, features, blocks, pitch, 25)
        dead_matrix = learned.feature_matrix(tracking, features, blocks, pitch, 25)
        head = model["robust_control_model"]
        base = (matrix / np.asarray(head["scale"])) @ np.asarray(head["coefficients"])
        home = temporal.predict_home(base, matrix, blocks, 25, model["temporal_model"], model["expert_gate"])
        dead = learned.predict_dead(dead_matrix, model)
        expected = learned.joint_probabilities(dead, home, matrix, dead_matrix, blocks, 25,
                                               model["joint_model"], tags.team_log_odds)
        np.testing.assert_array_equal(actual[1], expected)
        costs = boundary.switch_costs(model["boundary_model"], matrix, expected, blocks, 25, config.team_switch_cost)
        labels = np.repeat("X", 200)
        for _, start, last in blocks:
            labels[start:last + 1] = np.array(list("HAD"))[decode(expected[start:last + 1], 25, config,
                                                                  team_switch_costs=costs[start:last + 1])]
        np.testing.assert_array_equal(actual[0], labels)
        np.testing.assert_array_equal(actual[0][90:100], "X")
        self.assertEqual(actual[4], [])
        # Production tag fusion remains active after the joint residual.
        tags.team_log_odds[:] = np.linspace(-2, 2, 200)
        tagged = learned.learned_probabilities(features, tags, dead_matrix, model, matrix, blocks, 25)
        np.testing.assert_array_equal(tagged, learned.joint_probabilities(
            dead, home, matrix, dead_matrix, blocks, 25, model["joint_model"], tags.team_log_odds))
        self.assertGreater(np.max(np.abs(tagged[:, 0] - expected[:, 0])), .01)

    def test_optional_heads_do_not_change_legacy_emissions(self):
        model = learned.load_model(ROOT / "models/baselines/v1.4.json")
        tracking = invented_tracking(); blocks = [(1, 0, 199)]; pitch = (105.3, 68)
        config = learned.resolve_config(Config(), model)
        actual = learned.infer_learned(tracking, [], blocks, pitch, 25, config, model)
        matrix = robust_control.feature_matrix(tracking, actual[2], blocks, pitch, 25)
        dead_matrix = learned.feature_matrix(tracking, actual[2], blocks, pitch, 25)
        head = model["robust_control_model"]
        home = temporal.predict_home((matrix / np.asarray(head["scale"])) @ np.asarray(head["coefficients"]),
                                     matrix, blocks, 25, model["temporal_model"], model["expert_gate"])
        expected = learned.fuse_control_probabilities(learned.predict_dead(dead_matrix, model), home, actual[3].team_log_odds)
        np.testing.assert_array_equal(actual[1], expected)
        np.testing.assert_array_equal(actual[0], np.array(list("HAD"))[decode(expected, 25, config)])

    def test_constant_switch_vector_preserves_default_decoder_exactly(self):
        rng = np.random.default_rng(19); config = Config()
        for n in (1, 7, 200):
            p = rng.uniform(.01, 1, (n, 3)); p /= p.sum(axis=1)[:, None]
            a = decode(p, 25, config)
            b = decode(p, 25, config, team_switch_costs=np.full(n, config.team_switch_cost))
            np.testing.assert_array_equal(a, b)

    def test_only_supported_local_costs_recover_short_internal_possession(self):
        p = np.tile([.8, .19, .01], (70, 1)); p[30:40] = [.39, .60, .01]
        costs = np.full(70, 4.); costs[[30, 40]] = .1
        before = decode(p, 25, Config())
        after = decode(p, 25, Config(), team_switch_costs=costs)
        np.testing.assert_array_equal(before, 0)
        np.testing.assert_array_equal(np.flatnonzero(after == 1), np.arange(30, 40))
        # Discount at the wrong frames is not a global reduction in smoothing.
        wrong = np.full(70, 4.); wrong[[10, 60]] = .1
        np.testing.assert_array_equal(decode(p, 25, Config(), wrong), before)

    def test_local_team_costs_do_not_modify_dead_restart_transitions(self):
        p = np.tile([.99, .000001, .009999], (100, 1)); p[30:60] = [.009999, .000001, .99]
        original = decode(p, 25, Config())
        discounted = decode(p, 25, Config(), np.zeros(100))
        np.testing.assert_array_equal(original, discounted)
        self.assertTrue((original[35:55] == 2).all())

    def test_invalid_cost_vectors_and_incompatible_model_modes_are_rejected(self):
        p = np.tile([.8, .19, .01], (20, 1))
        for cost in (np.ones(19), np.ones((20, 1)), np.full(20, np.nan), np.full(20, np.inf), np.full(20, -.01)):
            with self.assertRaises(ValueError): decode(p, 25, Config(), cost)
        model = learned.load_model(ROOT / "models/baselines/v1.4.json")
        model["boundary_model"] = invented_boundary()
        invalid = copy.deepcopy(model); invalid["decoder_mode"] = "separate"
        with self.assertRaises(ValueError): learned.validate_model(invalid)
        with self.assertRaises(ValueError): learned.infer_learned(
            invented_tracking(), [], [(1, 0, 199)], (105.3, 68), 25, Config(), model, mode="separate")
        invalid = copy.deepcopy(model); invalid.pop("robust_control_model")
        with self.assertRaises(ValueError): learned.validate_model(invalid)


if __name__ == "__main__":
    unittest.main()
