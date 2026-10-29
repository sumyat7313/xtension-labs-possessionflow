import copy
import unittest
from tools.train_robust import select, FOLDS, CANDIDATES, BASELINE, REGIMES
from tools.verify_release import same_evidence


class RobustCampaignTests(unittest.TestCase):
    def records(self):
        return [{"candidate": name, "match": match, "regime": regime,
                 "spellF1": .4 if name == BASELINE else .44, "macroF1": .8, "dead_recall": .9,
                 "control_diagnostics": {"transitions": {"recall": .3},
                    "spell_recall_by_duration": {"<2s": {"recall": .1}}}}
                for name in (BASELINE, *CANDIDATES) for fold in FOLDS for match in fold for regime in REGIMES]

    def test_complete_evidence_and_duplicate_detection(self):
        rows = self.records()
        self.assertNotEqual(select(rows)["selected"], BASELINE)
        for bad in (rows[:-1], rows+[rows[0]], [dict(rows[0], candidate="unknown")]+rows[1:]):
            with self.assertRaises(ValueError): select(bad)
        rows[0]["spellF1"] = float("nan")
        with self.assertRaises(ValueError): select(rows)

    def test_regressions_and_fallback(self):
        original = self.records()
        for key in ("spell", "dead", "transition", "short"):
            rows = copy.deepcopy(original)
            for row in rows:
                if row["candidate"] == BASELINE: continue
                if key == "spell": row["spellF1"] = .39
                if key == "dead": row["dead_recall"] = .88
                if key == "transition": row["control_diagnostics"]["transitions"]["recall"] = .2
                if key == "short": row["control_diagnostics"]["spell_recall_by_duration"]["<2s"]["recall"] = .09
            self.assertEqual(select(rows)["selected"], BASELINE)

    def test_no_dead_head_training_matches_are_scored(self):
        scored = [match for fold in FOLDS for match in fold]
        self.assertEqual(len(scored), len(set(scored)))
        self.assertFalse(set(scored) & {"J03WOH", "J03WOY"})

    def test_verifier_accepts_only_numerical_roundoff(self):
        self.assertTrue(same_evidence({"gain": .07587155229937127}, {"gain": .0758715522993713}))
        self.assertFalse(same_evidence({"gain": .075}, {"gain": .076}))
        self.assertFalse(same_evidence({"gain": float("nan")}, {"gain": float("nan")}))

    def test_verifier_preserves_decisions_and_structure_exactly(self):
        self.assertFalse(same_evidence({"passed": True}, {"passed": 1}))
        self.assertFalse(same_evidence({"selected": "camera-context"}, {"selected": "context-control"}))
        self.assertFalse(same_evidence(["a", "b"], ["b", "a"]))
        self.assertFalse(same_evidence({"a": 1}, {"a": 1, "b": 2}))
