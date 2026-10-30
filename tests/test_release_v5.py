"""Fresh evidence must not rescue a failed development gate or claim false truth."""
import copy
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.train_joint_campaign import BASELINE, CANDIDATES, FOLDS, REGIMES, CONFIG_SHA256, select
from tools.fresh_benchmark import POLICY, SPLIT, decide
from tools.verify_release import V14_MODEL_SHA256, verify_v5_decision, verify_fresh_audit, verify_archived_sources

CANDIDATE = "a" * 64


def evidence(gain=0):
    rows = [{"candidate": candidate, "match": match, "regime": regime,
             "spellF1": .5 + (gain if candidate == "joint-state" else 0), "macroF1": .8,
             "dead_recall": .94, "dead_precision": .9, "control_diagnostics": {
                 "transitions": {"recall": .3}, "spell_recall_by_duration": {"<2s": {"recall": .1}}}}
            for candidate in (BASELINE, *CANDIDATES) for fold in FOLDS for match in fold for regime in REGIMES]
    selection = select(rows)
    eligible = selection["selected"] != BASELINE
    return {"records": rows, "selection": selection, "development_candidate_selected": eligible,
            "competition_labels_used": False, "independent_test_claimed": False,
            "incumbent_historical_predictions_reproduced": True,
            "protocol": {"fresh_audit_used_for_selection": False, "config_sha256": CONFIG_SHA256},
            "release_candidate_model_sha256": CANDIDATE,
            "release_decision": {"model_sha256": V14_MODEL_SHA256, "promoted": False,
                                 "independent_audit_status": "failed-development-retained-baseline", "reason": "Fixed gates."}}


def audit(metrica_delta=0, silver_deltas=(0, 0)):
    scores, references, counts = {}, {}, {}
    for match, delta in zip(("metrica3", "skillcorner1886347", "skillcorner1899585"), (metrica_delta, *silver_deltas)):
        silver = match != "metrica3"
        quality = "provider_silver" if silver else "provider_sparse_events"
        score = {"metric": "phase_present_team_accuracy" if silver else "sparse_pass_origin_team_accuracy",
                 "reference_quality": quality, "n": 100, "accuracy": .8, "balanced_team_recall": .8,
                 "predicted_live_coverage": .96, "predicted_dead_fraction": .04, "predicted_unknown_fraction": 0,
                 "dense_possession_truth": False, "dense_had_truth": False, "spellF1": None}
        candidate = dict(score, accuracy=.8+delta, balanced_team_recall=.8+delta)
        scores[match] = {"baseline": score, "candidate": candidate}
        references[match] = {"sha256": "b" * 64, "quality": quality}
        counts[match] = {"scored_frames": 100}
        if silver: counts[match].update(gap_frames_are_unknown=True, dead_reference_frames=0)
    split_bytes = (json.dumps(SPLIT, sort_keys=True, indent=2)+"\n").encode()
    raw = {"protocol": {"selection_frozen_before_reference_access": True, "no_training_or_retuning": True,
                        "dense_human_had_validation": False, "split": copy.deepcopy(SPLIT),
                        "split_sha256": hashlib.sha256(split_bytes).hexdigest(), "policy": copy.deepcopy(POLICY),
                        "artifacts": {"baseline": {"model_sha256": V14_MODEL_SHA256, "config_sha256": CONFIG_SHA256},
                                      "candidate": {"model_sha256": CANDIDATE, "config_sha256": CONFIG_SHA256}}},
           "scores": scores, "references": references, "reference_counts": counts,
           "gate": decide(scores), "reference_fields_used_for_inference": False,
           "dense_human_had_validation": False, "reference_read_count": 3}
    return {"policy": copy.deepcopy(POLICY), "passed": raw["gate"]["passed"],
            "split_file_base64": base64.b64encode(split_bytes).decode("ascii"),
            "baseline_model_sha256": V14_MODEL_SHA256, "candidate_model_sha256": CANDIDATE, "audit": raw}


class ReleaseV5Tests(unittest.TestCase):
    def test_original_noncanonical_split_bytes_are_authenticated(self):
        value = audit()
        reordered = dict(reversed(list(SPLIT.items())))
        original = ("\ufeff" + json.dumps(reordered, indent=4).replace("\n", "\r\n") + "\r\n").encode("utf-8")
        value["split_file_base64"] = base64.b64encode(original).decode("ascii")
        value["audit"]["protocol"]["split_sha256"] = hashlib.sha256(original).hexdigest()
        self.assertTrue(verify_fresh_audit(value, CANDIDATE, CONFIG_SHA256))

    def test_changed_split_bytes_hash_or_contents_are_rejected(self):
        value = audit()
        value["split_file_base64"] = base64.b64encode(b"{}\r\n").decode("ascii")
        with self.assertRaises(ValueError): verify_fresh_audit(value, CANDIDATE, CONFIG_SHA256)
        value["audit"]["protocol"]["split_sha256"] = hashlib.sha256(b"{}\r\n").hexdigest()
        with self.assertRaises(ValueError): verify_fresh_audit(value, CANDIDATE, CONFIG_SHA256)
        value = audit(); value["split_file_base64"] = "not base64!"
        with self.assertRaises(ValueError): verify_fresh_audit(value, CANDIDATE, CONFIG_SHA256)

    def test_fresh_audit_is_required_even_when_development_gate_fails(self):
        value = evidence()
        with self.assertRaises(ValueError): verify_v5_decision(value, V14_MODEL_SHA256)
        verify_v5_decision(value, V14_MODEL_SHA256, audit(.01, (.01, .01)))
        value["release_decision"].update(promoted=True, model_sha256=CANDIDATE, independent_audit_status="passed")
        with self.assertRaises(ValueError): verify_v5_decision(value, CANDIDATE, audit(.01, (.01, .01)))

    def test_promotion_requires_both_gates_and_exact_candidate(self):
        value = evidence(.04)
        value["release_decision"].update(promoted=True, model_sha256=CANDIDATE, independent_audit_status="passed")
        verify_v5_decision(value, CANDIDATE, audit(.01, (.01, .01)))
        other = audit(.01, (.01, .01)); other["candidate_model_sha256"] = "c"*64
        with self.assertRaises(ValueError): verify_v5_decision(value, CANDIDATE, other)

    def test_any_metrica_regression_forces_exact_baseline(self):
        value = evidence(.04)
        value["release_decision"]["independent_audit_status"] = "failed-retained-baseline"
        verify_v5_decision(value, V14_MODEL_SHA256, audit(-.00001, (.05, .05)))
        value["release_decision"].update(promoted=True, model_sha256=CANDIDATE, independent_audit_status="passed")
        with self.assertRaises(ValueError): verify_v5_decision(value, CANDIDATE, audit(-.00001, (.05, .05)))

    def test_silver_mean_and_single_match_regression_rules_both_apply(self):
        self.assertFalse(verify_fresh_audit(audit(.01, (-.01, 0)), CANDIDATE, CONFIG_SHA256))
        self.assertFalse(verify_fresh_audit(audit(.01, (-.03, .05)), CANDIDATE, CONFIG_SHA256))
        self.assertTrue(verify_fresh_audit(audit(.01, (-.01, .02)), CANDIDATE, CONFIG_SHA256))

    def test_reference_quality_gap_semantics_and_frozen_split_are_enforced(self):
        mutations = (
            lambda v: v["audit"].update(reference_fields_used_for_inference=True),
            lambda v: v["audit"].update(reference_read_count=4),
            lambda v: v["audit"]["protocol"].update(no_training_or_retuning=False),
            lambda v: v["audit"]["reference_counts"]["skillcorner1886347"].update(dead_reference_frames=100),
            lambda v: v["audit"]["reference_counts"]["skillcorner1886347"].update(gap_frames_are_unknown=False),
            lambda v: v["audit"]["scores"]["skillcorner1886347"]["candidate"].update(reference_quality="human_sparse_events"),
            lambda v: v["audit"]["scores"]["metrica3"]["candidate"].update(spellF1=.9),
            lambda v: v["audit"]["protocol"]["split"].update(final_sparse_metrica=[2]),
        )
        for mutate in mutations:
            value = audit(); mutate(value)
            with self.assertRaises(ValueError): verify_fresh_audit(value, CANDIDATE, CONFIG_SHA256)

    def test_scores_policy_and_decisions_cannot_be_rewritten(self):
        mutations = (
            lambda v: v["policy"].update(metrica_accuracy_minimum_delta=-.02),
            lambda v: v["audit"]["gate"].update(passed=False),
            lambda v: v["audit"]["scores"]["metrica3"]["candidate"].update(accuracy=float("nan")),
            lambda v: v["audit"]["scores"]["metrica3"]["candidate"].update(n=101),
            lambda v: v["audit"]["protocol"]["artifacts"]["candidate"].update(config_sha256="f"*64),
        )
        for mutate in mutations:
            value = audit(); mutate(value)
            with self.assertRaises(ValueError): verify_fresh_audit(value, CANDIDATE, CONFIG_SHA256)

    def test_selection_audit_winner_and_required_cells_cannot_be_changed(self):
        value = evidence(.04)
        value["selection"]["audit_candidate"] = "boundary"
        with self.assertRaises(ValueError): verify_v5_decision(value, V14_MODEL_SHA256, audit())
        rows = evidence()["records"]
        with self.assertRaises(ValueError): select(rows[:-1])
        with self.assertRaises(ValueError): select(rows+[rows[0]])
        rows[0]["dead_precision"] = True
        with self.assertRaises(ValueError): select(rows)

    def test_dead_precision_regression_disqualifies_high_spell_f1(self):
        value = evidence(.04)
        for row in value["records"]:
            if row["candidate"] == "joint-state": row["dead_precision"] -= .001
        result = select(value["records"])
        self.assertEqual(result["selected"], BASELINE)
        self.assertEqual(result["audit_candidate"], "joint-state")

    def test_historical_sources_are_verified_against_the_pinned_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/"old.zip"
            with zipfile.ZipFile(archive,"w") as z: z.writestr("possession/old.py",b"old source")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            source = {"possession/old.py": hashlib.sha256(b"old source").hexdigest()}
            verify_archived_sources(source, archive, digest)
            with self.assertRaises(ValueError): verify_archived_sources(source, archive, "f"*64)
            source["possession/old.py"] = hashlib.sha256(b"new source").hexdigest()
            with self.assertRaises(ValueError): verify_archived_sources(source, archive, digest)


if __name__ == "__main__": unittest.main()
