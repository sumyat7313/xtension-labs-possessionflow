"""Independent ML integrity tests using invented observations and labels only."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from possession.config import Config
from possession.features import extract
from possession.io import Tracking
from possession.learned import (
    FEATURE_NAMES, feature_matrix, fit_model, load_model, predict_dead, save_model, decode_learned,
)
from possession.tags import build
from tools.train_stoppages import (
    SPLITS, MIN_MEAN_GAIN, MAX_REGIME_DROP, corrupt_tracking, promotion_gate, select_candidate,
    paired_spell_bootstrap,
)
import tools.train_stoppages as campaign


def invented_tracking():
    n = 260
    frame = np.arange(n)
    players = np.empty((n, 6, 2), dtype=np.float64)
    players[:, :3] = [.25, .35]
    players[:, 3:] = [.75, .65]
    players[:, :, 0] += np.sin(frame[:, None] / 25) * .01
    players[:, :, 1] += np.cos(frame[:, None] / 30) * .01
    ball = players[:, 0].copy()
    ball[150:] = players[150:, 3]
    ball[30:44] = np.nan
    ball[180:200] = np.nan
    speed = np.tile((1.5 + .5 * np.sin(frame / 20))[:, None], (1, 6))
    return Tracking(players, speed, ball, np.full(n, 2.),
                    np.array([0, 0, 0, 1, 1, 1]), ("Home", "Away"))


def make_matrix(tracking):
    blocks = [(1, 0, 119), (2, 140, 259)]
    features = extract(tracking, (105.3, 68), blocks, 25, Config())
    return feature_matrix(tracking, features, blocks, (105.3, 68), 25)


class TrainingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(51240)
        self.x = self.rng.normal(size=(96, len(FEATURE_NAMES)))
        self.y = np.where(self.x[:, 0] + .3 * self.x[:, 1] > 0, "D", "H")
        self.query = self.rng.normal(size=(19, len(FEATURE_NAMES)))

    def test_unknown_reference_rows_do_not_change_fit_or_scaling(self):
        base = fit_model(self.x, self.y, l2=.01)
        # Extreme values would materially change the scaler if X rows leaked in.
        extra = np.full((12, self.x.shape[1]), 100000.)
        extended = fit_model(np.vstack((self.x, extra)), np.r_[self.y, np.repeat("X", 12)], l2=.01)
        np.testing.assert_allclose(predict_dead(self.query, base), predict_dead(self.query, extended),
                                   atol=1e-12, rtol=1e-12)

    def test_zero_weight_rows_do_not_change_fit_or_scaling(self):
        base = fit_model(self.x, self.y, l2=.01)
        extra = np.full((10, self.x.shape[1]), -100000.)
        extended = fit_model(np.vstack((self.x, extra)), np.r_[self.y, np.repeat("D", 10)],
                             l2=.01, sample_weight=np.r_[np.ones(len(self.x)), np.zeros(10)])
        np.testing.assert_allclose(predict_dead(self.query, base), predict_dead(self.query, extended),
                                   atol=1e-12, rtol=1e-12)

    def test_inference_does_not_refit_scaler_or_mutate_artifact(self):
        model = fit_model(self.x, self.y, l2=.01)
        before = json.dumps(model, sort_keys=True)
        ordinary = predict_dead(self.query, model)
        # An unrelated extreme test row must not change another row's prediction.
        together = predict_dead(np.vstack((self.query, np.full((1, self.x.shape[1]), 1e6))), model)
        np.testing.assert_allclose(ordinary, together[:-1], atol=1e-12, rtol=1e-12)
        self.assertEqual(before, json.dumps(model, sort_keys=True))

    def test_artifact_round_trip_reproduces_predictions(self):
        model = fit_model(self.x, self.y, l2=.01, metadata={"training_matches": ["invented"]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(path, model)
            first_bytes = path.read_bytes()
            loaded = load_model(path)
            np.testing.assert_array_equal(predict_dead(self.query, model), predict_dead(self.query, loaded))
            save_model(path, loaded)
            self.assertEqual(first_bytes, path.read_bytes())

    def test_repeated_fit_is_deterministic(self):
        first = fit_model(self.x, self.y, l2=.01)
        second = fit_model(self.x, self.y, l2=.01)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_unlabelled_rows_cannot_manufacture_a_training_class(self):
        with self.assertRaises(ValueError):
            fit_model(self.x, np.repeat("X", len(self.x)), l2=.01)
        with self.assertRaises(ValueError):
            fit_model(self.x, np.where(np.arange(len(self.x)) % 2, "X", "H"), l2=.01)

    def test_features_are_invariant_to_team_and_slot_renaming(self):
        original = invented_tracking()
        changed = copy.deepcopy(original)
        order = np.array([5, 2, 4, 0, 1, 3])
        changed.players = changed.players[:, order]
        changed.player_speed = changed.player_speed[:, order]
        changed.teams = 1 - changed.teams[order]
        changed.team_names = ("Arbitrary visitors", "Arbitrary hosts")
        np.testing.assert_allclose(make_matrix(original), make_matrix(changed), atol=1e-12, rtol=1e-12)

    def test_future_half_cannot_change_previous_half_features(self):
        original = invented_tracking()
        changed = copy.deepcopy(original)
        changed.ball[140:] = np.nan
        changed.players[140:] = np.nan
        changed.player_speed[140:] = np.nan
        np.testing.assert_array_equal(make_matrix(original)[:120], make_matrix(changed)[:120])

    def test_all_missing_observations_produce_finite_features(self):
        tracking = invented_tracking()
        tracking.ball[:] = np.nan
        tracking.ball_speed[:] = np.nan
        tracking.players[:] = np.nan
        tracking.player_speed[:] = np.nan
        matrix = make_matrix(tracking)
        self.assertEqual(matrix.shape, (260, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(matrix).all())

    def test_separate_dead_path_does_not_depend_on_team_preference(self):
        data = invented_tracking()
        blocks = [(1, 0, 119), (2, 140, 259)]
        features = extract(data, (105.3, 68), blocks, 25, Config())
        tags = build([], features, data.team_names, len(data.ball), 25, blocks, Config())
        matrix = make_matrix(data)
        targets = np.where(np.arange(len(matrix)) % 80 < 30, "D", "H")
        model = fit_model(matrix, targets)
        first, _, _ = decode_learned(features, tags, matrix, blocks, 25, Config(), model, mode="separate")
        changed = copy.deepcopy(features)
        changed.team_log_odds[:] = np.where(np.arange(len(matrix)) % 20 < 10, -8., 8.)
        second, _, _ = decode_learned(changed, tags, matrix, blocks, 25, Config(), model, mode="separate")
        np.testing.assert_array_equal(first == "D", second == "D")
        self.assertTrue((first[120:140] == "X").all())
        self.assertTrue((second[120:140] == "X").all())


class CampaignIntegrityTests(unittest.TestCase):
    def test_splits_are_whole_distinct_external_matches(self):
        sets = [set(SPLITS[name]) for name in ("train", "validation", "final")]
        self.assertTrue(all(sets))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertNotIn("BAR-ATH", set.union(*sets))
        self.assertEqual(SPLITS["final"], ["J03WMX"])

    def test_corruption_is_seeded_input_only_and_preserves_original(self):
        original = invented_tracking()
        original_copy = copy.deepcopy(original)
        blocks = [(1, 0, 119), (2, 140, 259)]
        first = corrupt_tracking(original, blocks, (105.3, 68), 25, 123)
        second = corrupt_tracking(original, blocks, (105.3, 68), 25, 123)
        different = corrupt_tracking(original, blocks, (105.3, 68), 25, 987)
        for field in ("players", "player_speed", "ball", "ball_speed", "teams"):
            np.testing.assert_array_equal(getattr(original, field), getattr(original_copy, field))
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        for field in ("players", "player_speed", "ball", "ball_speed"):
            np.testing.assert_array_equal(getattr(first, field)[120:140], getattr(original, field)[120:140])
            self.assertTrue(np.isnan(getattr(first, field))[np.isnan(getattr(original, field))].all())
        self.assertFalse(np.array_equal(first.players, different.players, equal_nan=True))
        self.assertEqual(first.team_names, original.team_names)

    def test_promotion_gate_requires_gain_and_both_regimes(self):
        self.assertEqual(MIN_MEAN_GAIN, .02)
        self.assertEqual(MAX_REGIME_DROP, .02)
        baseline = {"clean": {"spellF1": .4}, "degraded": {"spellF1": .3}}
        good = {"clean": {"spellF1": .45}, "degraded": {"spellF1": .33}}
        average_only = {"clean": {"spellF1": .6}, "degraded": {"spellF1": .25}}
        too_small = {"clean": {"spellF1": .41}, "degraded": {"spellF1": .31}}
        self.assertTrue(promotion_gate(good, baseline)["passed"])
        self.assertFalse(promotion_gate(average_only, baseline)["passed"])
        self.assertFalse(promotion_gate(too_small, baseline)["passed"])

    def test_selection_rejects_incomplete_duplicate_or_nonfinite_evidence(self):
        base = [{"candidate": "baseline", "regime": regime, "spellF1": .3, "macroF1": .5}
                for regime in ("clean", "degraded")]
        self.assertEqual(select_candidate(base)["selected"], "baseline")
        for bad in (base[:1], base + [base[0]],
                    base + [{"candidate": "broken", "regime": "clean", "spellF1": float("nan"), "macroF1": .5}]):
            with self.assertRaises(ValueError):
                select_candidate(bad)

    def test_strongest_eligible_candidate_survives_rejected_higher_mean(self):
        records = []
        for candidate, clean, degraded in (("baseline", .4, .3), ("robust", .46, .35),
                                           ("fragile", .6, .25)):
            records.extend({"candidate": candidate, "regime": regime, "spellF1": score, "macroF1": .6}
                           for regime, score in (("clean", clean), ("degraded", degraded)))
        result = select_candidate(records)
        self.assertEqual(result["selected"], "robust")

    def test_paired_bootstrap_zero_for_identical_and_symmetric_under_swap(self):
        reference = np.array(list("HHHHDDDAAAAADDDHHHHHDAAAAADDDHHHHH"))
        imperfect = reference.copy()
        imperfect[4:7] = "H"
        imperfect[20:25] = "H"
        identical = paired_spell_bootstrap(reference, imperfect, imperfect, draws=100)
        self.assertEqual(identical["mean_delta"], 0.)
        self.assertEqual(identical["percentiles_5_95"], [0., 0.])
        forward = paired_spell_bootstrap(reference, imperfect, reference, draws=100)
        reverse = paired_spell_bootstrap(reference, reference, imperfect, draws=100)
        self.assertGreater(forward["mean_delta"], 0)
        self.assertAlmostEqual(forward["mean_delta"], -reverse["mean_delta"])
        np.testing.assert_allclose(forward["percentiles_5_95"],
                                   -np.asarray(reverse["percentiles_5_95"])[::-1], atol=1e-12)

    def test_campaign_reads_final_only_after_frozen_selection(self):
        # Exercise actual training/campaign orchestration with invented fixtures;
        # synthetic scores isolate execution order rather than model quality.
        config_path = Path(campaign.ROOT) / "configs/default.json"
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "campaign"
            args = SimpleNamespace(out=out, data_dir=root / "data", config=config_path)

            def fake_read(directory, match):
                calls.append(match)
                if match in SPLITS["final"]:
                    frozen = json.loads((out / "selection-frozen-before-final.json").read_text())
                    self.assertNotEqual(frozen["selected"], "baseline")
                    self.assertTrue(frozen["selected_model_sha256"])
                    self.assertTrue(frozen["validation_records"])
                    self.assertTrue(all(row["match"] in SPLITS["validation"]
                                        for row in frozen["validation_records"]))
                data = invented_tracking()
                reference = np.where(np.arange(len(data.ball)) % 80 < 30, "D", "H")
                reference[120:140] = "X"
                return data, reference, [(1, 0, 119), (2, 140, 259)], (105.3, 68), {}, {"invented.xml": {"sha256": "fixture", "bytes": 1}}

            def fake_measure(destination, match, candidate, regime, reference, labels, bootstrap=0):
                return {"match": match, "candidate": candidate, "regime": regime,
                        "spellF1": .3 if candidate == "baseline" else .5,
                        "macroF1": .5 if candidate == "baseline" else .6}

            with patch.object(campaign, "ROOT", root), \
                 patch.object(campaign, "read_external", side_effect=fake_read), \
                 patch.object(campaign, "measure", side_effect=fake_measure), \
                 patch.object(campaign, "source_hashes", return_value={"fixture": "unchanged"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                campaign.run(args)
            self.assertEqual(calls, SPLITS["train"] + SPLITS["validation"] + SPLITS["final"])
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["competition_public_labels_used"])
            self.assertFalse(summary["final_set_retuning"])
            self.assertEqual(set(summary["training_matches"]), set(SPLITS["train"]))


if __name__ == "__main__":
    unittest.main()
