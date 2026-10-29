"""Synthetic checks of external-only turnover diagnostics and training weights."""

import unittest

import numpy as np

from tools.control_diagnostics import diagnostic, training_spell_weights


class ControlDiagnosticsTests(unittest.TestCase):
    def test_turnovers_match_once_and_report_timing(self):
        gt = np.array(list("HHHHAAAAAHHHHHAAAA"))
        pred = np.array(list("HHHAAAAAHHHHHAAAAA"))
        result = diagnostic(gt, pred, fps=2)["transitions"]
        self.assertEqual((result["n_gt"], result["n_pred"], result["n_matched"]), (3, 3, 3))
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["median_absolute_timing_error_seconds"], .5)
        crowded = diagnostic(list("HHHHHHHAAAAAA"), list("HHAAHHAAHHHAA"), fps=25)["transitions"]
        self.assertEqual(crowded["n_gt"], 1)
        self.assertEqual(crowded["n_pred"], 5)
        self.assertEqual(crowded["n_matched"], 1)
        self.assertEqual(crowded["n_spurious"], 4)
        self.assertEqual(crowded["precision"], .2)

    def test_does_not_match_across_even_short_unknown_reference_gap(self):
        result = diagnostic(list("HAAXHHH"), list("AAAXHAA"), fps=25)["transitions"]
        self.assertEqual((result["n_gt"], result["n_pred"]), (1, 1))
        self.assertEqual(result["n_matched"], 0)

    def test_dead_and_unknown_are_not_direct_turnovers(self):
        result = diagnostic(list("HDAHXAH"), list("HXAHDAH"), fps=25)["transitions"]
        self.assertEqual(result["n_gt"], 2)
        self.assertEqual(result["n_pred"], 2)
        # An opposite-direction turnover cannot satisfy a true transition.
        other = diagnostic(list("HHHAAA"), list("AAAHHH"), fps=25)["transitions"]
        self.assertEqual(other["n_matched"], 0)
        gap = diagnostic(list("HHHAAA"), list("HHXAAA"), fps=25)["transitions"]
        self.assertEqual(gap["n_pred"], 0)

    def test_duration_strata_use_official_spell_matching(self):
        gt = np.array(list("H" * 1 + "D" + "A" * 2 + "D" + "H" * 5 + "D" + "A" * 15))
        result = diagnostic(gt, gt, fps=1)
        self.assertEqual(result["n_gt_spells"], 4)
        for values in result["spell_recall_by_duration"].values():
            self.assertEqual(values, {"n_gt": 1, "n_matched": 1, "recall": 1})
        # Official temporal metrics bridge abstention within a reference-known run.
        bridged = diagnostic(list("HHHHH"), list("HHXHH"), fps=1)
        self.assertEqual(bridged["spell_recall_by_duration"]["5-15s"]["recall"], 1)

    def test_weights_are_bounded_normalized_and_training_reference_only(self):
        gt = np.array(list("H" + "X" + "A" * 4 + "D" + "H" * 100))
        before = gt.copy()
        weights = training_spell_weights(gt, fps=25)
        np.testing.assert_array_equal(gt, before)
        np.testing.assert_array_equal(weights, training_spell_weights(gt, fps=25))
        live = np.isin(gt, ["H", "A"])
        self.assertAlmostEqual(float(weights[live].mean()), 1, places=14)
        self.assertGreaterEqual(float(weights[live].min()), .5)
        self.assertLessEqual(float(weights[live].max()), 3)
        self.assertTrue((weights[~live] == 0).all())
        self.assertGreater(weights[0], weights[-1])
        self.assertTrue((weights[2:6] == weights[2]).all())

    def test_unknown_gap_splits_training_spells(self):
        weights = training_spell_weights(list("HXHHHH"), fps=1)
        self.assertGreater(weights[0], weights[2])
        self.assertEqual(weights[1], 0)
        np.testing.assert_array_equal(training_spell_weights(list("DDXX")), np.zeros(4))

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(diagnostic([], [])["n_gt_spells"], 0)
        self.assertIsNone(diagnostic([], [])["transitions"]["median_absolute_timing_error_seconds"])
        for bad_fps in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                diagnostic(["H"], ["H"], fps=bad_fps)
        for a, b in ((["H"], []), (["?"], ["H"]), ([["H"]], ["H"]), ([4], [0])):
            with self.assertRaises(ValueError):
                diagnostic(a, b)
        np.testing.assert_array_equal(training_spell_weights([0, 0, 3, 1]),
                                      training_spell_weights(list("HHXA")))


if __name__ == "__main__":
    unittest.main()
