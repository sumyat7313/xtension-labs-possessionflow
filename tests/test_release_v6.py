"""V6 release evidence must preserve both gates and dense-reference scope."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from tools.train_temporal_v6 import BASELINE, CANDIDATES, FOLDS, REGIMES, GATE, CONFIG_SHA256, select
from tools.fresh_benchmark_v6 import SPLIT, POLICY, WCSF, GEOMETRY_TOLERANCE_METRES, dense_score, decide
from tools.verify_release import (V15_MODEL_SHA256, V15_SOURCE_SHA256, verify_v6_decision,
    verify_fresh_audit_v6, verify_archived_sources, protocol_matches_hash, verify_v6_candidate_artifact)

CANDIDATE = "a"*64


def evidence(gain=0.):
    rows=[]
    for name in (BASELINE,*CANDIDATES):
        d=gain if name == "stoppage-loss" else 0.
        for fold in FOLDS:
            for match in fold:
                for regime in REGIMES:
                    rows.append(dict(candidate=name,match=match,regime=regime,spellF1=.5+d,macroF1=.8,
                        dead_recall=.94,dead_precision=.9,temporal_diagnostics={
                            "dead_segments":{"f1":.8},"dead_onsets":{"f1":.8},"dead_offsets":{"f1":.8},
                            "short_possessions":{"recall":.2+d}},
                        duration_precision_recall={"bins":{k:{"n_pred":10,"n_matched_pred":5}
                            for k in ("<0.4s","0.4-1s","1-2s")}}))
    selection=select(rows)
    return dict(records=rows,selection=selection,development_candidate_selected=selection["selected"]!=BASELINE,
        competition_labels_used=False,independent_test_claimed=False,incumbent_historical_predictions_reproduced=True,
        protocol=dict(fresh_audit_used_for_selection=False,config_sha256=CONFIG_SHA256,
            baseline_sha256=V15_MODEL_SHA256,archive_sha256=V15_SOURCE_SHA256,gate=copy.deepcopy(GATE)),
        release_candidate_model_sha256=CANDIDATE,
        release_decision=dict(model_sha256=V15_MODEL_SHA256,promoted=False,
            independent_audit_status="failed-development-retained-baseline",reason="Fixed gates."))


def refresh_protocol_hash(value):
    raw=(json.dumps(value["audit"]["protocol"],sort_keys=True,indent=2)+"\n").encode()
    value["audit_protocol_sha256"]=hashlib.sha256(raw).hexdigest()


def audit(improve=True,silver_deltas=(0.,0.)):
    reference=np.array(list("H"*30+"D"*30+"A"*10+"H"*30+"D"*30+"A"*10))
    baseline=reference.copy();baseline[baseline=="A"]="H"
    candidate=reference.copy() if improve else baseline.copy()
    blocks=[(1,0,len(reference)-1)]
    dense={k:dense_score(reference,p,blocks,10) for k,p in (("baseline",baseline),("candidate",candidate))}
    scores={"wcsf2023":dense};counts={"wcsf2023":dict(scored_frames=len(reference),
        class_counts={c:int(sum(reference==c)) for c in "HADX"},annotation_method="undocumented_provider_flags")}
    references={"wcsf2023":{n:"b"*64 for n in WCSF}}
    for match,delta in zip(SPLIT["final_silver_matches"],silver_deltas):
        name=f"skillcorner{match}"
        value=dict(metric="phase_present_team_accuracy",reference_quality="provider_silver",n=100,
            accuracy=.8,balanced_team_recall=.8,predicted_live_coverage=.96,predicted_dead_fraction=.04,
            predicted_unknown_fraction=0.,dense_possession_truth=False,dense_had_truth=False,spellF1=None)
        scores[name]={"baseline":value,"candidate":dict(value,accuracy=.8+delta)}
        counts[name]=dict(scored_frames=100,gap_frames_are_unknown=True,dead_reference_frames=0)
        references[name]=dict(sha256="c"*64,quality="provider_silver")
    split_bytes=(json.dumps(SPLIT,sort_keys=True,indent=2)+"\n").encode()
    protocol=dict(schema_version=1,selection_frozen_before_reference_access=True,no_training_or_retuning=True,
        dense_human_had_validation=False,geometry_tolerance_metres=GEOMETRY_TOLERANCE_METRES,
        split=copy.deepcopy(SPLIT),split_sha256=hashlib.sha256(split_bytes).hexdigest(),policy=copy.deepcopy(POLICY),
        artifacts={"baseline":dict(model_sha256=V15_MODEL_SHA256,config_sha256=CONFIG_SHA256),
            "candidate":dict(model_sha256=CANDIDATE,config_sha256=CONFIG_SHA256)})
    raw=dict(protocol=protocol,scores=scores,references=references,reference_counts=counts,gate=decide(scores),
        prediction_sha256={f"{name}-{which}.csv":"d"*64 for name in scores for which in ("baseline","candidate")},
        reference_fields_used_for_inference=False,dense_human_had_validation=False,reference_match_read_count=3)
    value=dict(policy=copy.deepcopy(POLICY),passed=raw["gate"]["passed"],split_file_base64=base64.b64encode(split_bytes).decode(),
        baseline_model_sha256=V15_MODEL_SHA256,candidate_model_sha256=CANDIDATE,audit=raw)
    refresh_protocol_hash(value)
    return value


class ReleaseV6Tests(unittest.TestCase):
    def test_one_fresh_experiment_required_even_when_development_fails(self):
        e=evidence()
        with self.assertRaises(ValueError):verify_v6_decision(e,V15_MODEL_SHA256)
        verify_v6_decision(e,V15_MODEL_SHA256,audit())
        e["release_decision"].update(promoted=True,model_sha256=CANDIDATE,independent_audit_status="passed")
        with self.assertRaises(ValueError):verify_v6_decision(e,CANDIDATE,audit())

    def test_promotion_requires_both_gates_and_exact_candidate(self):
        e=evidence(.04)
        e["release_decision"].update(promoted=True,model_sha256=CANDIDATE,independent_audit_status="passed")
        verify_v6_decision(e,CANDIDATE,audit())
        a=audit();a["candidate_model_sha256"]="f"*64
        with self.assertRaises(ValueError):verify_v6_decision(e,CANDIDATE,a)
        with self.assertRaises(ValueError):verify_v6_decision(e,"f"*64,audit())

    def test_failed_fresh_weakness_gate_requires_exact_baseline(self):
        e=evidence(.04);e["release_decision"]["independent_audit_status"]="failed-retained-baseline"
        self.assertFalse(verify_fresh_audit_v6(audit(False),CANDIDATE,CONFIG_SHA256))
        verify_v6_decision(e,V15_MODEL_SHA256,audit(False))
        e["release_decision"].update(promoted=True,model_sha256=CANDIDATE,independent_audit_status="passed")
        with self.assertRaises(ValueError):verify_v6_decision(e,CANDIDATE,audit(False))

    def test_silver_mean_and_individual_regression_limits(self):
        self.assertFalse(verify_fresh_audit_v6(audit(silver_deltas=(-.01,0)),CANDIDATE,CONFIG_SHA256))
        self.assertFalse(verify_fresh_audit_v6(audit(silver_deltas=(-.03,.04)),CANDIDATE,CONFIG_SHA256))
        self.assertTrue(verify_fresh_audit_v6(audit(silver_deltas=(-.01,.02)),CANDIDATE,CONFIG_SHA256))

    def test_frozen_protocol_and_original_split_bytes_are_authenticated(self):
        a=audit();a["audit_protocol_sha256"]="f"*64
        with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)
        a=audit();original=("\ufeff"+json.dumps(dict(reversed(list(SPLIT.items()))),indent=4).replace("\n","\r\n")+"\r\n").encode()
        a["split_file_base64"]=base64.b64encode(original).decode()
        a["audit"]["protocol"]["split_sha256"]=hashlib.sha256(original).hexdigest();refresh_protocol_hash(a)
        self.assertTrue(verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256))
        a["split_file_base64"]=base64.b64encode(b"{}").decode()
        with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)
        a["audit"]["protocol"]["split_sha256"]=hashlib.sha256(b"{}").hexdigest();refresh_protocol_hash(a)
        with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)

    def test_missing_predictions_references_and_unpaired_support_rejected(self):
        mutations=[lambda a:a["audit"]["prediction_sha256"].pop(next(iter(a["audit"]["prediction_sha256"]))),
            lambda a:a["audit"]["references"].pop("wcsf2023"),
            lambda a:a["audit"]["scores"]["wcsf2023"]["candidate"].update(n=100),
            lambda a:a["audit"]["scores"]["wcsf2023"]["candidate"]["temporal"]["short_possessions"].update(n_gt=999)]
        for mutate in mutations:
            a=audit();mutate(a)
            with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)

    def test_no_human_gold_or_silver_dead_truth_claim_allowed(self):
        silver=f"skillcorner{SPLIT['final_silver_matches'][0]}"
        mutations=[lambda a:a["audit"].update(dense_human_had_validation=True),
            lambda a:a["audit"].update(reference_fields_used_for_inference=True),
            lambda a:a["audit"].update(reference_match_read_count=4),
            lambda a:a["audit"]["scores"]["wcsf2023"]["candidate"].update(dense_human_had_truth=True),
            lambda a:a["audit"]["reference_counts"]["wcsf2023"].update(annotation_method="human gold"),
            lambda a:a["audit"]["reference_counts"][silver].update(dead_reference_frames=100),
            lambda a:a["audit"]["reference_counts"][silver].update(gap_frames_are_unknown=False)]
        for mutate in mutations:
            a=audit();mutate(a)
            with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)

    def test_scores_cannot_disagree_with_event_counts_or_gate(self):
        mutations=[lambda a:a["audit"]["gate"].update(passed=False),
            lambda a:a["audit"]["scores"]["wcsf2023"]["candidate"].update(dead_precision=float("nan")),
            lambda a:a["audit"]["scores"]["wcsf2023"]["candidate"]["temporal"]["dead_onsets"].update(f1=.1),
            lambda a:a["audit"]["scores"]["wcsf2023"]["candidate"]["temporal"]["dead_offsets"].update(n_matched=999)]
        for mutate in mutations:
            a=audit();mutate(a)
            with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)

    def test_changed_policy_artifacts_and_baseline_are_rejected(self):
        a=audit();a["policy"]["dense_maximum_drop"]=-.1
        with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)
        a=audit();a["audit"]["protocol"]["artifacts"]["candidate"]["config_sha256"]="f"*64;refresh_protocol_hash(a)
        with self.assertRaises(ValueError):verify_fresh_audit_v6(a,CANDIDATE,CONFIG_SHA256)
        e=evidence();e["protocol"]["baseline_sha256"]="f"*64
        with self.assertRaises(ValueError):verify_v6_decision(e,V15_MODEL_SHA256,audit())

    def test_changed_winner_missing_cells_and_baseline_disguised_as_experiment_rejected(self):
        e=evidence(.04);e["selection"]["audit_candidate"]="transition"
        with self.assertRaises(ValueError):verify_v6_decision(e,V15_MODEL_SHA256,audit())
        e=evidence();e["records"].pop()
        with self.assertRaises(ValueError):verify_v6_decision(e,V15_MODEL_SHA256,audit())
        e=evidence();e["release_candidate_model_sha256"]=V15_MODEL_SHA256
        with self.assertRaises(ValueError):verify_v6_decision(e,V15_MODEL_SHA256,audit())

    def test_stored_experiment_must_match_hash_and_selected_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"candidate.json";p.write_text(json.dumps({"candidate_id":"stoppage-loss"}))
            e=evidence(.04);e["release_candidate_model_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
            verify_v6_candidate_artifact(e,p)
            e["selection"]["audit_candidate"]="transition"
            with self.assertRaises(ValueError):verify_v6_candidate_artifact(e,p)
            e["selection"]["audit_candidate"]="stoppage-loss"
            p.write_text(json.dumps({"candidate_id":"stoppage-loss","altered":True}))
            with self.assertRaises(ValueError):verify_v6_candidate_artifact(e,p)
            with self.assertRaises(FileNotFoundError):verify_v6_candidate_artifact(e,Path(folder)/"missing.json")

    def test_pinned_historical_source_archive_rejects_modified_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"historical.zip"
            with zipfile.ZipFile(p,"w") as z:z.writestr("possession/old.py",b"old model inference")
            h=hashlib.sha256(p.read_bytes()).hexdigest();sources={"possession/old.py":hashlib.sha256(b"old model inference").hexdigest()}
            verify_archived_sources(sources,p,h)
            with self.assertRaises(ValueError):verify_archived_sources(sources,p,V15_SOURCE_SHA256)
            sources["possession/old.py"]="f"*64
            with self.assertRaises(ValueError):verify_archived_sources(sources,p,h)


if __name__ == "__main__": unittest.main()
