"""Fixed v4 protocol checks; no PyTorch or match downloads are required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession import robust_control
from tools.train_multimodal import (BASELINE, CANDIDATES, REGIMES, PRIMARY_REGIMES,
                                    FOLDS, CATALOG, select, partition, sequences,
                                    base_logits, verify_incumbent_scores)


def records(gains=None):
    gains = gains or {}
    return [{"candidate": candidate, "match": match, "regime": regime,
             "spellF1": .4 + gains.get(candidate, 0), "macroF1": .8,
             "dead_recall": .94,
             "control_diagnostics": {"transitions": {"recall": .3},
                                      "spell_recall_by_duration": {"<2s": {"recall": .1}}}}
            for candidate in (BASELINE,) + CANDIDATES for fold in FOLDS for match in fold for regime in REGIMES]


def zero_head():
    count = len(robust_control.FEATURE_NAMES)
    return {"schema_version": 1, "kind": "symmetric_logistic_control_v3",
            "feature_names": list(robust_control.FEATURE_NAMES), "feature_mask": [True] * count,
            "scale": [1.] * count, "coefficients": [0.] * count, "intercept": 0.}


class MultimodalCampaignTests(unittest.TestCase):
    def test_baseline_fallback_and_fixed_candidate_winner(self):
        self.assertEqual(select(records())["selected"], BASELINE)
        chosen = select(records({"calibrated-control": .02, "temporal": .04, "gated-temporal": .03}))
        self.assertEqual(chosen["selected"], "temporal")
        self.assertTrue(chosen["gates"]["temporal"]["passed"])
        self.assertFalse(chosen["independent_audit_used_for_selection"])

    def test_complete_twenty_cells_required_for_every_candidate(self):
        rows = records()
        for broken in (rows[:-1], rows + [rows[0]], [dict(rows[0], regime="unregistered")] + rows[1:]):
            with self.assertRaises(ValueError):
                select(broken)
        wrong = copy.deepcopy(rows)
        wrong[0]["candidate"] = "invented"
        with self.assertRaises(ValueError):
            select(wrong)

    def test_calibrated_condition_cannot_win_without_old_condition_gain(self):
        rows = records()
        for row in rows:
            if row["candidate"] == "temporal" and row["regime"] == "calibrated":
                row["spellF1"] = .9
        self.assertEqual(select(rows)["selected"], BASELINE)

    def test_each_twenty_cell_nonregression_limit_is_enforced(self):
        rows = records({"temporal": .05})
        for metric, value in (("spellF1", .379), ("dead_recall", .929)):
            broken = copy.deepcopy(rows)
            cell = next(row for row in broken if row["candidate"] == "temporal" and row["regime"] == "calibrated")
            cell[metric] = value
            self.assertEqual(select(broken)["selected"], BASELINE)

    def test_transition_and_short_spell_regressions_block_promotion(self):
        rows = records({"temporal": .05})
        for diagnostic in ("transition", "short"):
            broken = copy.deepcopy(rows)
            cell = next(row for row in broken if row["candidate"] == "temporal")
            if diagnostic == "transition":
                cell["control_diagnostics"]["transitions"]["recall"] -= .01
            else:
                cell["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"] -= .01
            self.assertEqual(select(broken)["selected"], BASELINE)

    def test_invalid_or_unknown_diagnostics_cannot_silently_pass(self):
        for value in (np.nan, np.inf, None, -1., 1.1, True):
            rows = records({"temporal": .05})
            rows[0]["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select(rows)

    def test_selection_is_order_independent_and_uses_primary_tiebreak(self):
        rows = records({"temporal": .05, "gated-temporal": .05})
        for row in rows:
            if row["candidate"] == "gated-temporal" and row["regime"] in PRIMARY_REGIMES:
                row["macroF1"] += .01
        expected = select(rows)
        np.random.default_rng(418).shuffle(rows)
        self.assertEqual(select(rows), expected)
        self.assertEqual(expected["selected"], "gated-temporal")

    def test_outer_and_inner_partitions_have_no_shared_evaluation_match(self):
        scored = {match for fold in FOLDS for match in fold}
        self.assertTrue({"J03WOH", "J03WOY"}.isdisjoint(scored))
        for held in FOLDS:
            train, inner, inner_train = partition(held)
            self.assertEqual(set(train), set(CATALOG) - set(held))
            self.assertTrue(set(held).isdisjoint(inner_train))
            self.assertNotIn(inner, held)
            self.assertNotIn(inner, inner_train)
            self.assertEqual(inner, sorted(train)[0])
            self.assertEqual(set(inner_train) | {inner}, set(train))

    def test_half_local_five_hz_training_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "cache" / "train-match"
            folder.mkdir(parents=True)
            n, columns = 33, len(robust_control.FEATURE_NAMES)
            matrix = np.repeat(np.arange(n)[:, None], columns, axis=1).astype(float)
            labels = np.array(list("HADXX") * 7)[:n]
            weight = np.isin(labels, ["H", "A"]).astype(float)
            np.save(folder / "reference.npy", labels)
            np.save(folder / "weights.npy", weight)
            (folder / "metadata.json").write_text(json.dumps({"blocks": [[1, 2, 13], [2, 19, 32]]}))
            for regime in REGIMES:
                np.save(folder / f"{regime}-features.npy", matrix)
            result = sequences(root, ["train-match"], zero_head())
            self.assertEqual(len(result), len(REGIMES) * 2)
            for item in result:
                expected = np.array([2, 7, 12]) if item["half"] == 1 else np.array([19, 24, 29])
                np.testing.assert_array_equal(item["X"][:, 0], expected)
                np.testing.assert_array_equal(item["y"], labels[expected])
                np.testing.assert_array_equal(item["w"], weight[expected])
                np.testing.assert_array_equal(item["base"], np.zeros(len(expected)))
                self.assertEqual(item["match"], "train-match")

    def test_base_logit_avoids_probability_clipping_roundtrip(self):
        head = zero_head()
        head["coefficients"][0] = 1.
        matrix = np.zeros((2, len(robust_control.FEATURE_NAMES)))
        matrix[:, 0] = [100., -100.]
        np.testing.assert_array_equal(base_logits(head, matrix), [100., -100.])

    def test_incumbent_verification_checks_predictions_not_just_scores(self):
        historical, current = [], []
        for fold in FOLDS:
            for match in fold:
                for regime in PRIMARY_REGIMES:
                    row = {"candidate": "camera-context", "match": match, "regime": regime,
                           "spellF1": .4, "macroF1": .8, "dead_recall": .9, "dead_precision": .8,
                           "prediction_sha256": "same-prediction-bytes"}
                    historical.append(row)
                    current.append(dict(row, candidate=BASELINE))
        verify_incumbent_scores(current, {"records": historical})
        current[0]["prediction_sha256"] = "different-bytes-same-rounded-score"
        with self.assertRaises(ValueError):
            verify_incumbent_scores(current, {"records": historical})


if __name__ == "__main__":
    unittest.main()
