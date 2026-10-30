"""Selection and training-weight checks using invented match scores and labels."""
import copy
import unittest
import numpy as np
from tools.train_joint_campaign import BASELINE, CANDIDATES, FOLDS, REGIMES, joint_weights, select


def invented_records():
    records = []
    gains = {BASELINE: 0., "joint-state": .02, "boundary": .015, "joint-boundary": .03}
    for candidate in (BASELINE, *CANDIDATES):
        for fold in FOLDS:
            for match in fold:
                for regime in REGIMES:
                    records.append({"candidate": candidate, "match": match, "regime": regime,
                                    "spellF1": .5 + gains[candidate], "macroF1": .8,
                                    "dead_recall": .9, "dead_precision": .85,
                                    "control_diagnostics": {"transitions": {"recall": .4},
                                        "spell_recall_by_duration": {"<2s": {"recall": .2}}}})
    return records


class JointCampaignTests(unittest.TestCase):
    def test_complete_predeclared_cells_select_best_eligible_candidate(self):
        result = select(invented_records())
        self.assertEqual(result["selected"], "joint-boundary")
        self.assertEqual(len(result["gates"]["joint-boundary"]["cell_gains"]), 20)
        self.assertFalse(result["fresh_audit_used_for_selection"])
        self.assertAlmostEqual(result["means"]["joint-boundary"]["primary_spellF1"], .53)

    def test_missing_duplicate_unknown_and_nonfinite_cells_fail_closed(self):
        records = invented_records()
        for invalid in (records[:-1], records + [copy.deepcopy(records[0])]):
            with self.assertRaises(ValueError): select(invalid)
        for key, value in (("candidate", "unregistered"), ("regime", "unknown"), ("spellF1", np.nan),
                           ("macroF1", True), ("dead_precision", 1.01), ("dead_recall", -.01)):
            invalid = copy.deepcopy(records); invalid[0][key] = value
            with self.assertRaises(ValueError): select(invalid)
        invalid = copy.deepcopy(records)
        invalid[0]["control_diagnostics"]["transitions"]["recall"] = np.inf
        with self.assertRaises(ValueError): select(invalid)

    def test_calibrated_cell_regression_cannot_hide_behind_primary_mean(self):
        records = invented_records()
        row = next(row for row in records if row["candidate"] == "joint-boundary" and row["regime"] == "calibrated")
        row["spellF1"] = .47
        result = select(records)
        self.assertGreater(result["gates"]["joint-boundary"]["primary_mean_spellF1_gain"], .02)
        self.assertFalse(result["gates"]["joint-boundary"]["passed"])
        self.assertEqual(result["selected"], "joint-state")

    def test_dead_and_short_possession_regressions_block_promotion(self):
        for metric in ("dead_recall", "dead_precision", "macroF1", "transition", "short"):
            records = invented_records()
            row = next(row for row in records if row["candidate"] == "joint-boundary" and row["regime"] == "clean")
            if metric == "dead_recall": row[metric] = .88
            elif metric in ("dead_precision", "macroF1"): row[metric] -= .01
            elif metric == "transition": row["control_diagnostics"]["transitions"]["recall"] -= .01
            else: row["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"] -= .01
            result = select(records)
            self.assertFalse(result["gates"]["joint-boundary"]["passed"], metric)
            self.assertEqual(result["selected"], "joint-state", metric)

    def test_baseline_fallback_and_deterministic_candidate_tie(self):
        records = invented_records()
        for row in records: row["spellF1"] = .5
        self.assertEqual(select(records)["selected"], BASELINE)
        for row in records:
            if row["candidate"] != BASELINE: row["spellF1"] = .52
        self.assertEqual(select(records)["selected"], "boundary")
        self.assertEqual(select(list(reversed(records))), select(records))

    def test_dead_weights_positive_unknown_zero_and_transitions_bounded(self):
        labels = np.array(list("H" * 30 + "D" * 30 + "A" * 30 + "X" * 10))
        weights = joint_weights(labels, [(1, 0, 89)])
        self.assertTrue((weights[:90] > 0).all())
        np.testing.assert_array_equal(weights[90:], 0)
        np.testing.assert_allclose(weights[41:49], 1)
        self.assertTrue((weights[20:41] >= 1.5).all())
        self.assertTrue((weights[50:71] >= 1.5).all())
        self.assertTrue((weights <= 3).all())
        # Known labels outside a graded block still have zero loss weight.
        outside = labels.copy(); outside[90:] = "D"
        np.testing.assert_array_equal(joint_weights(outside, [(1, 0, 89)])[90:], 0)

    def test_half_edges_and_unknown_gaps_are_not_transition_targets(self):
        labels = np.array(list("H" * 30 + "A" * 30))
        np.testing.assert_allclose(joint_weights(labels, [(1, 0, 29), (2, 30, 59)]), 1)
        within = joint_weights(labels, [(1, 0, 59)])
        self.assertTrue((within[20:41] >= 1.5).all())
        unknown = np.array(list("H" * 20 + "X" * 4 + "A" * 20 + "D" * 20))
        weights = joint_weights(unknown, [(1, 0, len(unknown) - 1)])
        np.testing.assert_array_equal(weights[20:24], 0)
        np.testing.assert_allclose(weights[:20], 1)
        np.testing.assert_allclose(weights[24:34], 1)


if __name__ == "__main__":
    unittest.main()
