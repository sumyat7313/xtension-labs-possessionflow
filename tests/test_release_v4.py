"""Release gates must not promote untested candidates or rewrite failed audits."""
import copy
import hashlib
import json
import unittest

from tools.train_multimodal import BASELINE, CANDIDATES, FOLDS, REGIMES, select
from tools.verify_release import V13_MODEL_SHA256, verify_v4_decision, verify_paired_audit, protocol_matches_hash


CANDIDATE_SHA256 = "a" * 64


def evidence(gain=0):
    rows = [{"candidate": candidate, "match": match, "regime": regime,
             "spellF1": .4 + (gain if candidate == "temporal" else 0), "macroF1": .8,
             "dead_recall": .94, "control_diagnostics": {
                 "transitions": {"recall": .3}, "spell_recall_by_duration": {"<2s": {"recall": .1}}}}
            for candidate in (BASELINE, *CANDIDATES) for fold in FOLDS for match in fold for regime in REGIMES]
    selected = select(rows)
    eligible = selected["selected"] != BASELINE
    return {"records": rows, "selection": selected, "development_candidate_selected": eligible,
            "competition_labels_used": False, "independent_test_claimed": False,
            "incumbent_historical_predictions_reproduced": True,
            "protocol": {"independent_audit_used_for_selection": False},
            "release_candidate_model_sha256": CANDIDATE_SHA256 if eligible else V13_MODEL_SHA256,
            "release_decision": {"model_sha256": V13_MODEL_SHA256, "promoted": False,
                                 "independent_audit_status": "not-run-no-eligible-candidate", "reason": "Fixed gate."}}


def audit(delta=0):
    score = {"metric": "sparse_pass_origin_team_accuracy", "n": 100,
             "accuracy": .9, "balanced_team_recall": .9,
             "predicted_live_coverage": .96, "predicted_dead_fraction": .04,
             "predicted_unknown_fraction": 0, "dense_possession_truth": False, "spellF1": None}
    baseline, candidate = copy.deepcopy(score), copy.deepcopy(score)
    candidate["accuracy"] += delta
    candidate["balanced_team_recall"] += delta
    scores = {"baseline": baseline, "candidate": candidate}
    deltas = {metric: candidate[metric] - baseline[metric] for metric in (
        "accuracy", "balanced_team_recall", "predicted_live_coverage", "predicted_dead_fraction", "predicted_unknown_fraction")}
    raw = {"protocol": {"game": 2, "role": "reserved_final_sparse_audit",
                        "no_parameter_tuning_in_this_command": True,
                        "artifacts": {"baseline": {"model_sha256": V13_MODEL_SHA256},
                                      "candidate": {"model_sha256": CANDIDATE_SHA256}}},
           "reference_events_passed_to_inference": False, "reference_read_count": 1,
           "dense_possession_truth": False, "scores": scores, "candidate_minus_baseline": deltas}
    return {"policy": {"dataset": "Metrica sample Game 2", "metric": "sparse_pass_origin_team_accuracy",
                       "maximum_accuracy_drop": .02, "maximum_balanced_recall_drop": .02},
            "baseline_model_sha256": V13_MODEL_SHA256, "candidate_model_sha256": CANDIDATE_SHA256,
            "scores": copy.deepcopy(scores), "candidate_minus_baseline": copy.deepcopy(deltas),
            "passed": delta >= -.02, "audit": raw}


class ReleaseV4Tests(unittest.TestCase):
    def test_no_eligible_candidate_keeps_exact_baseline_and_audit_unread(self):
        value = evidence()
        verify_v4_decision(value, V13_MODEL_SHA256)
        with self.assertRaises(ValueError):
            verify_v4_decision(value, V13_MODEL_SHA256, audit())
        value["release_decision"]["model_sha256"] = CANDIDATE_SHA256
        with self.assertRaises(ValueError):
            verify_v4_decision(value, CANDIDATE_SHA256)

    def test_selected_candidate_cannot_ship_without_audit(self):
        with self.assertRaises(ValueError):
            verify_v4_decision(evidence(.04), V13_MODEL_SHA256)

    def test_passing_audit_promotes_only_the_selected_bytes(self):
        value = evidence(.04)
        value["release_decision"].update(promoted=True, model_sha256=CANDIDATE_SHA256, independent_audit_status="passed")
        verify_v4_decision(value, CANDIDATE_SHA256, audit(.01))
        wrong = audit(.01)
        wrong["candidate_model_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            verify_v4_decision(value, CANDIDATE_SHA256, wrong)

    def test_failed_audit_forces_baseline_without_runner_up(self):
        value = evidence(.04)
        value["release_decision"]["independent_audit_status"] = "failed-retained-baseline"
        verify_v4_decision(value, V13_MODEL_SHA256, audit(-.03))
        value["release_decision"].update(promoted=True, model_sha256=CANDIDATE_SHA256, independent_audit_status="passed")
        with self.assertRaises(ValueError):
            verify_v4_decision(value, CANDIDATE_SHA256, audit(-.03))

    def test_relaxed_threshold_or_changed_gate_is_rejected(self):
        value = audit(-.03)
        value["policy"]["maximum_accuracy_drop"] = .05
        with self.assertRaises(ValueError):
            verify_paired_audit(value, CANDIDATE_SHA256)
        value = audit(-.03)
        value["passed"] = True
        with self.assertRaises(ValueError):
            verify_paired_audit(value, CANDIDATE_SHA256)

    def test_scores_deltas_and_reference_separation_are_checked(self):
        for mutate in (
            lambda v: v["scores"]["candidate"].update(accuracy=.99),
            lambda v: v["audit"]["candidate_minus_baseline"].update(accuracy=.1),
            lambda v: v["audit"].update(reference_events_passed_to_inference=True),
            lambda v: v["audit"].update(reference_read_count=2),
            lambda v: v["audit"]["scores"]["candidate"].update(spellF1=.9),
        ):
            value = audit()
            mutate(value)
            with self.assertRaises(ValueError):
                verify_paired_audit(value, CANDIDATE_SHA256)

    def test_recomputed_selection_cannot_be_overridden(self):
        value = evidence(.04)
        value["selection"]["selected"] = "calibrated-control"
        with self.assertRaises(ValueError):
            verify_v4_decision(value, V13_MODEL_SHA256)

    def test_embedded_protocol_hash_is_platform_independent_but_content_sensitive(self):
        value = {"x": 1, "y": False}
        serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        for text in (serialized, serialized.replace("\n", "\r\n")):
            digest = hashlib.sha256(text.encode()).hexdigest()
            self.assertTrue(protocol_matches_hash(value, digest))
            self.assertFalse(protocol_matches_hash(dict(value, x=2), digest))
