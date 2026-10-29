"""Independent v2 checks. All observations and targets here are invented."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from possession.config import Config
from possession.features import extract
from possession.io import Tracking
from possession import control
from possession import learned
from possession.tags import build
from possession.pipeline import infer, predict
from games import GAME, GAMES, asset_paths
from possession.io import sha256
import tools.train_control as campaign_v2


BLOCKS = [(1, 0, 139), (2, 160, 319)]
PITCH = (105.3, 68.)


def observations():
    rng = np.random.default_rng(78155)
    n = 320
    players = np.empty((n, 8, 2))
    players[:, :4] = [.3, .4]
    players[:, 4:] = [.7, .6]
    players += rng.normal(0, .004, players.shape)
    ball = players[:, 0].copy()
    ball[85:110] = players[85:110, 4]
    ball[200:] = players[200:, 5]
    ball[30:36] = np.nan
    ball[240:265] = np.nan
    return Tracking(players, rng.uniform(.5, 4., (n, 8)), ball,
                    np.full(n, 2.), np.repeat([0, 1], 4), ("Home", "Away"))


def matrix(data, pitch=PITCH):
    features = extract(data, pitch, BLOCKS, 25, Config())
    return control.feature_matrix(data, features, BLOCKS, pitch, 25)


class ControlObservationIntegrityTests(unittest.TestCase):
    def test_team_rename_and_reorder_negates_every_signed_feature(self):
        data = observations()
        swapped = copy.deepcopy(data)
        order = np.array([6, 2, 7, 0, 4, 3, 1, 5])
        swapped.players = swapped.players[:, order]
        swapped.player_speed = swapped.player_speed[:, order]
        swapped.teams = 1 - swapped.teams[order]
        swapped.team_names = ("Different away name", "Different home name")
        np.testing.assert_allclose(matrix(swapped), -matrix(data), atol=1e-12, rtol=1e-12)

    def test_within_team_identity_switches_each_frame_do_not_change_features(self):
        data = observations()
        changed = copy.deepcopy(data)
        rng = np.random.default_rng(91367)
        for frame in range(len(data.ball)):
            order = np.r_[rng.permutation(4), rng.permutation(4) + 4]
            changed.players[frame] = data.players[frame, order]
            changed.player_speed[frame] = data.player_speed[frame, order]
        np.testing.assert_allclose(matrix(changed), matrix(data), atol=1e-12, rtol=1e-12)

    def test_reflecting_pitch_preserves_control_evidence(self):
        data = observations()
        reflected = copy.deepcopy(data)
        reflected.players = 1 - reflected.players
        reflected.ball = 1 - reflected.ball
        np.testing.assert_allclose(matrix(reflected), matrix(data), atol=1e-10, rtol=1e-10)

    def test_later_half_and_unplayed_gap_cannot_change_earlier_half(self):
        data = observations()
        changed = copy.deepcopy(data)
        changed.ball[140:] = np.nan
        changed.players[140:] = np.nan
        changed.player_speed[140:] = np.nan
        changed.ball_speed[140:] = np.nan
        np.testing.assert_array_equal(matrix(data)[:140], matrix(changed)[:140])

    def test_metres_are_preserved_when_normalized_coordinates_change(self):
        data = observations()
        changed = copy.deepcopy(data)
        alternate_pitch = (120., 80.)
        factor = np.array(PITCH) / alternate_pitch
        changed.players *= factor
        changed.ball *= factor
        np.testing.assert_allclose(matrix(data), matrix(changed, alternate_pitch), atol=1e-10, rtol=1e-10)

    def test_all_missing_observations_give_finite_neutral_evidence(self):
        data = observations()
        for attribute in ("players", "player_speed", "ball", "ball_speed"):
            getattr(data, attribute)[:] = np.nan
        result = matrix(data)
        self.assertEqual(result.shape, (320, len(control.FEATURE_NAMES)))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_extraction_does_not_mutate_observations(self):
        data = observations()
        before = copy.deepcopy(data)
        matrix(data)
        for attribute in ("players", "player_speed", "ball", "ball_speed", "teams"):
            np.testing.assert_array_equal(getattr(data, attribute), getattr(before, attribute))


class ControlTrainingIntegrityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(43051)
        self.x = rng.normal(size=(150, len(control.FEATURE_NAMES)))
        self.labels = np.where(self.x[:, 0] + .4 * self.x[:, 1] > -.7, "H", "A")
        self.query = rng.normal(size=(31, len(control.FEATURE_NAMES)))

    def test_dead_unknown_and_zero_weight_targets_do_not_influence_fit(self):
        baseline = control.fit_model(self.x, self.labels)
        added = np.full((15, self.x.shape[1]), 1e6)
        labels = np.r_[self.labels, np.repeat("D", 5), np.repeat("X", 5), np.repeat("H", 5)]
        weights = np.r_[np.ones(len(self.x) + 10), np.zeros(5)]
        model = control.fit_model(np.vstack((self.x, added)), labels, sample_weight=weights)
        np.testing.assert_allclose(control.predict_home(baseline, self.query),
                                   control.predict_home(model, self.query), atol=1e-12, rtol=1e-12)

    def test_team_swap_symmetry_holds_despite_unbalanced_training_labels(self):
        model = control.fit_model(self.x, self.labels)
        p = control.predict_home(model, self.query)
        swapped = control.predict_home(model, -self.query)
        np.testing.assert_allclose(p + swapped, np.ones(len(p)), atol=1e-14, rtol=0)
        neutral = control.predict_home(model, np.zeros((1, len(control.FEATURE_NAMES))))
        np.testing.assert_array_equal(neutral, [.5])

    def test_swapping_training_teams_preserves_fitted_predictor(self):
        model = control.fit_model(self.x, self.labels)
        other = control.fit_model(-self.x, np.where(self.labels == "H", "A", "H"))
        np.testing.assert_allclose(control.predict_home(model, self.query),
                                   control.predict_home(other, self.query), atol=1e-12, rtol=1e-12)

    def test_inference_is_batch_independent_and_does_not_mutate_weights(self):
        model = control.fit_model(self.x, self.labels)
        before = json.dumps(model, sort_keys=True)
        p = control.predict_home(model, self.query)
        augmented = np.vstack((self.query, np.full((2, self.x.shape[1]), -1e6)))
        np.testing.assert_allclose(p, control.predict_home(model, augmented)[:-2], atol=1e-12, rtol=1e-12)
        self.assertEqual(before, json.dumps(model, sort_keys=True))

    def test_serialization_and_repeat_training_are_deterministic(self):
        model = control.fit_model(self.x, self.labels)
        self.assertEqual(model, control.fit_model(self.x, self.labels))
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.json"
            second = Path(temp) / "second.json"
            control.save_model(first, model)
            restored = control.load_model(first)
            control.save_model(second, restored)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            np.testing.assert_array_equal(control.predict_home(model, self.query),
                                          control.predict_home(restored, self.query))

    def test_non_control_targets_cannot_manufacture_a_model(self):
        for labels in (np.full(len(self.x), "D"), np.full(len(self.x), "X"),
                       np.where(np.arange(len(self.x)) % 2, "D", "X")):
            with self.assertRaises(ValueError):
                control.fit_model(self.x, labels)
        with self.assertRaises(ValueError):
            control.fit_model(self.x, self.labels, sample_weight=np.zeros(len(self.x)))

    def test_malformed_schema_and_nonfinite_weights_fail_explicitly(self):
        model = control.fit_model(self.x, self.labels)
        variants = [("feature_names", []), ("kind", "other"),
                    ("scale", [0.] * len(control.FEATURE_NAMES)),
                    ("coefficients", [float("nan")] * len(control.FEATURE_NAMES))]
        for key, value in variants:
            changed = copy.deepcopy(model)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_model(changed)


class CombinedModelIntegrityTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(70911)
        dead_x = rng.normal(size=(130, len(learned.FEATURE_NAMES)))
        result = learned.fit_model(dead_x, np.where(dead_x[:, 0] > .5, "D", "H"))
        team_x = rng.normal(size=(130, len(control.FEATURE_NAMES)))
        result["control_model"] = control.fit_model(team_x, np.where(team_x[:, 0] > 0, "H", "A"))
        result["decoder_settings"] = {"team_switch_cost": 1.0, "min_team_seconds": .2}
        return result

    def test_nested_model_preserves_home_away_probability_symmetry(self):
        data = observations()
        swapped = copy.deepcopy(data)
        swapped.teams = 1 - data.teams
        swapped.team_names = data.team_names[::-1]
        model = self.model()
        labels, probability, *_ = infer(data, [], BLOCKS, PITCH, 25, Config(), model)
        other_labels, other_probability, *_ = infer(swapped, [], BLOCKS, PITCH, 25, Config(), model)
        np.testing.assert_allclose(probability[:, [1, 0, 2]], other_probability, atol=1e-12, rtol=1e-12)
        self.assertTrue((labels[140:160] == "X").all())
        self.assertTrue((other_labels[140:160] == "X").all())

    def test_combined_inference_cannot_read_across_halves(self):
        data = observations()
        changed = copy.deepcopy(data)
        changed.ball[140:] = np.nan
        changed.players[140:] = np.nan
        changed.player_speed[140:] = np.nan
        changed.ball_speed[140:] = np.nan
        model = self.model()
        first = infer(data, [], BLOCKS, PITCH, 25, Config(), model)
        second = infer(changed, [], BLOCKS, PITCH, 25, Config(), model)
        np.testing.assert_array_equal(first[0][:140], second[0][:140])
        np.testing.assert_array_equal(first[1][:140], second[1][:140])

    def test_missing_or_malformed_control_matrix_does_not_silently_fallback(self):
        data = observations()
        features = extract(data, PITCH, BLOCKS, 25, Config())
        tags = build([], features, data.team_names, len(data.ball), 25, BLOCKS, Config())
        dead_matrix = learned.feature_matrix(data, features, BLOCKS, PITCH, 25)
        model = self.model()
        for control_matrix in (None, np.zeros((5, len(control.FEATURE_NAMES))),
                               np.zeros((len(data.ball), 2))):
            with self.assertRaises(ValueError):
                learned.learned_probabilities(features, tags, dead_matrix, model, control_matrix)

    def test_decoder_overrides_are_validated_and_do_not_mutate_input_config(self):
        config = Config()
        model = self.model()
        resolved = learned.resolve_config(config, model)
        self.assertEqual(resolved.team_switch_cost, 1.)
        self.assertEqual(resolved.min_team_seconds, .2)
        self.assertEqual(config.team_switch_cost, 4.)
        for settings in ({"stop_cost": 1.}, {"team_switch_cost": 0.},
                         {"min_team_seconds": float("nan")}, {"team_switch_cost": True}):
            changed = copy.deepcopy(model)
            changed["decoder_settings"] = settings
            with self.assertRaises(ValueError):
                learned.validate_model(changed)

    def test_legacy_model_emissions_remain_exactly_compatible(self):
        data = observations()
        features = extract(data, PITCH, BLOCKS, 25, Config())
        tags = build([], features, data.team_names, len(data.ball), 25, BLOCKS, Config())
        dead_matrix = learned.feature_matrix(data, features, BLOCKS, PITCH, 25)
        model = self.model()
        del model["control_model"]
        del model["decoder_settings"]
        dead = learned.predict_dead(dead_matrix, model)
        home = 1 / (1 + np.exp(-np.clip(features.team_log_odds + tags.team_log_odds, -8, 8)))
        expected = np.column_stack(((1 - dead) * home, (1 - dead) * (1 - home), dead))
        np.testing.assert_array_equal(expected, learned.learned_probabilities(features, tags, dead_matrix, model))

    def test_run_snapshots_nested_weights_effective_settings_and_reproduces_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {key: Path(value) for key, value in asset_paths(data_dir=root / "data").items()}
            paths["atd_meta"].parent.mkdir(parents=True)
            paths["atd_meta"].write_text(
                '<EPTS><Teams><Team id="h"><Name>Home</Name></Team>'
                '<Team id="a"><Name>Away</Name></Team></Teams><Players>'
                '<Player teamId="h"/><Player teamId="a"/></Players></EPTS>', encoding="utf-8")
            paths["atd_raw"].write_text("".join(
                f"{i + 1}:0.3,0.5,2;0.7,0.5,2:{.302 if i < 40 else .698},0.5,2\n"
                for i in range(80)), encoding="utf-8")
            model = self.model()
            learned.save_model(root / "model.json", model)
            spec = dict(GAMES[GAME], n_video_frames=80)
            with patch.dict(GAMES, {GAME: spec}), patch("possession.pipeline.half_video_blocks", return_value=[(1, 0, 79)]):
                first, _ = predict(root / "data", root / "first", tracking_only=True,
                                   model_path=root / "model.json", progress=lambda _: None)
                second, _ = predict(root / "data", root / "second", tracking_only=True,
                                    model_path=root / "first/model.json", config_path=root / "first/config.json",
                                    progress=lambda _: None)
            actual_config = json.loads((root / "first/config.json").read_text(encoding="utf-8"))
            self.assertEqual(actual_config["team_switch_cost"], 1.)
            self.assertEqual(actual_config["min_team_seconds"], .2)
            self.assertEqual(first["config_sha256"], sha256(root / "first/config.json"))
            self.assertEqual(first["learned_model"]["sha256"], sha256(root / "first/model.json"))
            self.assertEqual(json.loads((root / "first/model.json").read_text(encoding="utf-8")), model)
            self.assertEqual(first["output"]["sha256"], second["output"]["sha256"])
            self.assertEqual((root / "first/predictions.csv").read_bytes(), (root / "second/predictions.csv").read_bytes())


class ControlCampaignIntegrityTests(unittest.TestCase):
    def cells(self, candidate, scores=None, matches=None):
        matches = campaign_v2.SPLITS["validation"] if matches is None else matches
        scores = [.4] * (len(matches) * 2) if scores is None else scores
        return [{"match": match, "regime": regime, "candidate": candidate,
                 "spellF1": scores[index * 2 + side], "macroF1": .7}
                for index, match in enumerate(matches) for side, regime in enumerate(campaign_v2.REGIMES)]

    def candidates(self):
        return [campaign_v2.BASELINE, "decoder-cost-2", "decoder-cost-1"] + [
            f"control-{weighting}-l2-{l2:g}-cost-{cost:g}"
            for weighting in campaign_v2.WEIGHTINGS for l2 in campaign_v2.L2_VALUES
            for cost in campaign_v2.SWITCH_COSTS]

    def test_fresh_final_matches_are_separate_from_all_exposed_matches(self):
        train, validation, final = [set(campaign_v2.SPLITS[k]) for k in ("train", "validation", "final")]
        self.assertEqual(final, {"J03WPY", "J03WQQ", "J03WR9"})
        self.assertIn("J03WMX", validation)
        self.assertFalse(train & validation or train & final or validation & final)
        self.assertNotIn("BAR-ATH", train | validation | final)
        self.assertEqual(len(self.candidates()), 15)

    def test_cell_regression_rejects_a_large_positive_mean(self):
        baseline = self.cells(campaign_v2.BASELINE)
        fragile = self.cells("fragile", [.8, .8, .8, .37])
        gate = campaign_v2.compare_cells(fragile, baseline)
        self.assertGreater(gate["mean_spellF1_gain"], .02)
        self.assertFalse(gate["passed"])
        self.assertTrue(campaign_v2.compare_cells(self.cells("robust", [.43] * 4), baseline)["passed"])
        self.assertFalse(campaign_v2.compare_cells(self.cells("small", [.41] * 4), baseline)["passed"])

    def test_duplicate_missing_nonfinite_and_wrong_regime_cells_fail(self):
        baseline = self.cells(campaign_v2.BASELINE)
        valid = self.cells("candidate", [.45] * 4)
        malformed = [valid + [valid[0]], valid[:-1], valid[::2]]
        for field, value in (("spellF1", float("nan")), ("macroF1", float("inf")),
                             ("spellF1", 1.1), ("regime", "unknown")):
            changed = copy.deepcopy(valid)
            changed[0][field] = value
            malformed.append(changed)
        for rows in malformed:
            with self.assertRaises(ValueError):
                campaign_v2.compare_cells(rows, baseline)

    def test_selection_gates_before_ranking_and_retains_incumbent_when_needed(self):
        names = self.candidates()
        records = []
        for name in names:
            scores = [.4] * 4
            if name == names[-1]:
                scores = [.8, .8, .8, .37]
            elif name == names[-2]:
                scores = [.45] * 4
            records.extend(self.cells(name, scores))
        selected = campaign_v2.select_candidate(records)
        self.assertEqual(selected["selected"], names[-2])
        self.assertNotIn(names[-1], selected["eligible_ranking"])
        fallback = campaign_v2.select_candidate([row for name in names for row in self.cells(name)])
        self.assertEqual(fallback["selected"], campaign_v2.BASELINE)

    def test_selection_requires_complete_predeclared_cohort(self):
        names = self.candidates()
        complete = [row for name in names for row in self.cells(name)]
        malformed = [
            [row for row in complete if row["match"] == campaign_v2.SPLITS["validation"][0]],
            [row for row in complete if row["candidate"] != names[-1]],
            complete + self.cells("undeclared-candidate"),
        ]
        for rows in malformed:
            with self.assertRaises(ValueError):
                campaign_v2.select_candidate(rows)

    def test_final_reads_only_frozen_winner_after_validation(self):
        winner = self.candidates()[-1]
        calls = []
        rng = np.random.default_rng(19014)
        invented_matrix = rng.normal(size=(40, len(control.FEATURE_NAMES)))
        invented_reference = np.where(invented_matrix[:, 0] > 0, "H", "A")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "campaign"
            args = SimpleNamespace(out=output, data_dir=Path(temp) / "data", workers=1,
                                   baseline=campaign_v2.ROOT / "models/baselines/v1.1.json",
                                   config=campaign_v2.ROOT / "configs/baselines/v1.1.json")

            def fake_training_read(directory, match):
                self.assertIn(match, campaign_v2.SPLITS["train"])
                self.assertTrue((output / "protocol.json").is_file())
                self.assertFalse((output / "selection-frozen-before-final.json").exists())
                calls.append(("train", match))
                return None, invented_reference, [(1, 0, 39)], PITCH, {}, {}

            def fake_evaluate(pool, out, match, directory, models, config, stage, bootstrap=0):
                calls.append((stage, match))
                frozen_path = out / "selection-frozen-before-final.json"
                if stage == "validation":
                    self.assertEqual(set(models), set(self.candidates()))
                    self.assertFalse(frozen_path.exists())
                else:
                    self.assertEqual(stage, "final")
                    self.assertTrue(frozen_path.is_file())
                    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                    self.assertEqual(frozen["selected"], winner)
                    self.assertEqual(set(models), {campaign_v2.BASELINE, winner})
                    self.assertEqual(frozen["selected_model_sha256"], sha256(out / "models" / f"{winner}.json"))
                    self.assertTrue(all(("validation", name) in calls for name in campaign_v2.SPLITS["validation"]))
                return [row for name in models for row in self.cells(name, [.45 if name == winner else .4] * 2,
                                                                      matches=[match])], {}

            with patch.object(campaign_v2, "read_external", side_effect=fake_training_read), \
                 patch.object(campaign_v2, "prepared", return_value=(None, None, None, invented_matrix)), \
                 patch.object(campaign_v2, "evaluate_match", side_effect=fake_evaluate), \
                 patch.object(campaign_v2, "ProcessPoolExecutor"), contextlib.redirect_stdout(io.StringIO()):
                campaign_v2.run(args)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["promoted"])
            self.assertEqual(summary["release_candidate"], winner)
            self.assertEqual([match for stage, match in calls if stage == "final"], campaign_v2.SPLITS["final"])
            self.assertFalse(summary["final_set_retuning"])

    def test_campaign_joint_decoder_equals_production_on_synthetic_observations(self):
        data = observations()
        model = CombinedModelIntegrityTests().model()
        config = learned.resolve_config(Config(), model)
        features = extract(data, PITCH, BLOCKS, 25, config)
        tags = build([], features, data.team_names, len(data.ball), 25, BLOCKS, config)
        dead_matrix = learned.feature_matrix(data, features, BLOCKS, PITCH, 25)
        control_matrix = control.feature_matrix(data, features, BLOCKS, PITCH, 25)
        expected = learned.decode_learned(features, tags, dead_matrix, BLOCKS, 25,
                                          config, model, control_matrix=control_matrix)[0]
        probability = learned.learned_probabilities(features, tags, dead_matrix, model, control_matrix)
        reference = np.where(np.arange(len(data.ball)) < 150, "H", "A")
        reference[140:160] = "X"
        usable = [True] * len(BLOCKS)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(campaign_v2, "measure", return_value={}), patch.object(campaign_v2, "write_json"):
                _, actual = campaign_v2.score_job(Path(temp), "invented", "candidate", "clean",
                                                 reference, probability, BLOCKS, config, usable, 0)
                _, changed_targets = campaign_v2.score_job(Path(temp), "invented", "candidate", "clean",
                                                          np.full(len(reference), "D"), probability,
                                                          BLOCKS, config, usable, 0)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, changed_targets)


if __name__ == "__main__":
    unittest.main()
