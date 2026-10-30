"""Verify committed release bytes without loading restricted match assets."""

import hashlib
import base64
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V13_MODEL_SHA256 = "f4feb096face6a2a3eb1214079a5f34ea8b495147478b7a92b6b224a42dc0573"
V13_SOURCE_SHA256 = "6bba5e359ec0c425aba15a764bf8c7e76c42ee804004de6eca311e2af7d2decb"
V14_MODEL_SHA256 = "7f72b108801a9c6029a2543ceaccd2ef9f27947c256ab59265df47c316688653"
V14_SOURCE_SHA256 = "04c5341978660a52fc32148221508ddb3db291921ff19d8fb1bfbf3e5cafeeb9"
V15_MODEL_SHA256 = "3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0"
V15_SOURCE_SHA256 = "9ff7b8103355d8fdf3160112f78019fc01b001e34fb438a581d23193ad0964fb"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_matches_hash(protocol, expected):
    """Authenticate embedded write_json evidence across Windows/Linux checkouts."""
    value = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return expected in {hashlib.sha256(value.encode()).hexdigest(),
                       hashlib.sha256(value.replace("\n", "\r\n").encode()).hexdigest()}


def verify_archived_sources(source_hashes, archive_path, expected_archive_sha256):
    """Authenticate historical source bytes without comparing them to new code."""
    if sha256(archive_path) != expected_archive_sha256:
        raise ValueError("Historical source archive hash changed")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Historical source map is empty")
    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in source_hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Historical source differs: {name}")


def verify_paired_audit(wrapper, candidate_sha256):
    """Check the fixed sparse-audit rule without claiming dense possession truth."""
    policy, audit = wrapper["policy"], wrapper["audit"]
    if (policy["dataset"] != "Metrica sample Game 2"
            or policy["metric"] != "sparse_pass_origin_team_accuracy"
            or policy["maximum_accuracy_drop"] != .02
            or policy["maximum_balanced_recall_drop"] != .02):
        raise ValueError("Independent audit policy changed")
    if wrapper["baseline_model_sha256"] != V13_MODEL_SHA256 or wrapper["candidate_model_sha256"] != candidate_sha256:
        raise ValueError("Independent audit does not compare the frozen models")
    protocol = audit["protocol"]
    if (protocol["game"] != 2 or protocol["role"] != "reserved_final_sparse_audit"
            or protocol["no_parameter_tuning_in_this_command"] is not True
            or audit["reference_events_passed_to_inference"] is not False
            or audit["reference_read_count"] != 1 or audit["dense_possession_truth"] is not False):
        raise ValueError("Independent audit separation changed")
    for name, expected in (("baseline", V13_MODEL_SHA256), ("candidate", candidate_sha256)):
        if protocol["artifacts"][name]["model_sha256"] != expected:
            raise ValueError("Independent audit artifact hash changed")
    if not same_evidence(wrapper["scores"], audit["scores"]) or not same_evidence(wrapper["candidate_minus_baseline"], audit["candidate_minus_baseline"]):
        raise ValueError("Independent audit summary differs from its underlying evidence")
    scores = audit["scores"]
    counts = []
    for value in scores.values():
        if (value["metric"] != "sparse_pass_origin_team_accuracy"
                or value["dense_possession_truth"] is not False or value["spellF1"] is not None
                or type(value["n"]) is not int or value["n"] <= 0):
            raise ValueError("Independent audit metric is not a valid sparse audit")
        counts.append(value["n"])
    if len(counts) != 2 or counts[0] != counts[1]:
        raise ValueError("Independent audit models must score the same reference points")
    deltas = {}
    for metric in ("accuracy", "balanced_team_recall", "predicted_live_coverage", "predicted_dead_fraction", "predicted_unknown_fraction"):
        values = [scores[name][metric] for name in ("baseline", "candidate")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Independent audit scores must be finite probabilities")
        deltas[metric] = values[1] - values[0]
    if not same_evidence(deltas, audit["candidate_minus_baseline"]):
        raise ValueError("Independent audit deltas do not match the scores")
    passed = deltas["accuracy"] >= -.02 and deltas["balanced_team_recall"] >= -.02
    if type(wrapper["passed"]) is not bool or wrapper["passed"] != passed:
        raise ValueError("Independent audit gate decision changed")
    return passed


def verify_v4_decision(evidence, release_model_sha256, audit=None):
    """Fail closed on promotion, fallback and one-shot audit inconsistencies."""
    from tools.train_multimodal import BASELINE, select
    recomputed = select(evidence["records"])
    if not same_evidence(recomputed, evidence["selection"]):
        raise ValueError("Multimodal selection recomputation changed")
    selected = recomputed["selected"] != BASELINE
    if evidence["development_candidate_selected"] is not selected:
        raise ValueError("Multimodal development selection flag changed")
    if (evidence["competition_labels_used"] is not False or evidence["independent_test_claimed"] is not False
            or evidence["incumbent_historical_predictions_reproduced"] is not True
            or evidence["protocol"]["independent_audit_used_for_selection"] is not False):
        raise ValueError("Multimodal evaluation separation changed")
    decision = evidence["release_decision"]
    if (type(decision["promoted"]) is not bool or decision["model_sha256"] != release_model_sha256
            or not isinstance(decision["reason"], str) or not decision["reason"].strip()):
        raise ValueError("Invalid multimodal release decision")
    candidate_sha256 = evidence["release_candidate_model_sha256"]
    if not selected:
        if (audit is not None or decision["promoted"] or candidate_sha256 != V13_MODEL_SHA256
                or release_model_sha256 != V13_MODEL_SHA256
                or decision["independent_audit_status"] != "not-run-no-eligible-candidate"):
            raise ValueError("No eligible candidate must retain v1.3 and leave Game 2 unused")
        return recomputed
    if audit is None:
        raise ValueError("An eligible candidate requires the frozen paired audit before release")
    passed = verify_paired_audit(audit, candidate_sha256)
    expected_status = "passed" if passed else "failed-retained-baseline"
    expected_model = candidate_sha256 if passed else V13_MODEL_SHA256
    if (decision["promoted"] != passed or decision["independent_audit_status"] != expected_status
            or release_model_sha256 != expected_model):
        raise ValueError("Multimodal promotion/fallback does not follow the paired audit")
    return recomputed


def verify_fresh_audit(wrapper, candidate_sha256, config_sha256):
    """Require fresh sparse and provider-silver results without relabeling gaps."""
    from tools.fresh_benchmark import SPLIT, decide
    policy, audit = wrapper["policy"], wrapper["audit"]
    fixed = {"metrica_accuracy_minimum_delta": 0., "metrica_balanced_recall_minimum_delta": 0.,
             "silver_mean_accuracy_minimum_delta": 0., "silver_maximum_match_accuracy_drop": .02,
             "no_dense_human_or_dead_ball_validation": True}
    if any(isinstance(policy.get(key), bool) for key in fixed if key != "no_dense_human_or_dead_ball_validation"):
        raise ValueError("Fresh audit policy uses invalid threshold types")
    if any(policy.get(key) != value for key, value in fixed.items()) or policy["no_dense_human_or_dead_ball_validation"] is not True:
        raise ValueError("Fresh audit policy changed")
    if wrapper["baseline_model_sha256"] != V14_MODEL_SHA256 or wrapper["candidate_model_sha256"] != candidate_sha256:
        raise ValueError("Fresh audit compares different model bytes")
    protocol = audit["protocol"]
    if (protocol["selection_frozen_before_reference_access"] is not True
            or protocol["no_training_or_retuning"] is not True
            or protocol["dense_human_had_validation"] is not False
            or audit["reference_fields_used_for_inference"] is not False
            or audit["dense_human_had_validation"] is not False
            or audit["reference_read_count"] != 3):
        raise ValueError("Fresh audit separation or scope changed")
    encoded_split = wrapper["split_file_base64"]
    if not isinstance(encoded_split, str):
        raise ValueError("Fresh split evidence must contain the original encoded bytes")
    split_bytes = base64.b64decode(encoded_split, validate=True)
    if (protocol["split"] != SPLIT
            or hashlib.sha256(split_bytes).hexdigest() != protocol["split_sha256"]
            or json.loads(split_bytes.decode("utf-8-sig")) != SPLIT):
        raise ValueError("Fresh match split changed")
    if not same_evidence(policy, protocol["policy"]):
        raise ValueError("Fresh audit policy differs from frozen protocol")
    for name, expected in (("baseline", V14_MODEL_SHA256), ("candidate", candidate_sha256)):
        artifact = protocol["artifacts"][name]
        if artifact["model_sha256"] != expected or artifact["config_sha256"] != config_sha256:
            raise ValueError("Fresh audit frozen model/configuration differs")
    expected_matches = {"metrica3", "skillcorner1886347", "skillcorner1899585"}
    if set(audit["scores"]) != expected_matches or set(audit["references"]) != expected_matches:
        raise ValueError("Fresh audit must include the three fixed matches")
    for match, scores in audit["scores"].items():
        silver = match != "metrica3"
        quality = "provider_silver" if silver else "provider_sparse_events"
        metric = "phase_present_team_accuracy" if silver else "sparse_pass_origin_team_accuracy"
        if set(scores) != {"baseline", "candidate"} or audit["references"][match]["quality"] != quality:
            raise ValueError("Fresh audit reference provenance differs")
        counts = audit["reference_counts"][match]
        if silver and (counts["gap_frames_are_unknown"] is not True or counts["dead_reference_frames"] != 0):
            raise ValueError("Silver phase gaps cannot become dead-ball truth")
        for value in scores.values():
            if (value["reference_quality"] != quality or value["metric"] != metric
                    or value["dense_had_truth"] is not False or value["dense_possession_truth"] is not False
                    or value["spellF1"] is not None or type(value["n"]) is not int or value["n"] <= 0
                    or value["n"] != counts["scored_frames"]):
                raise ValueError("Fresh audit scores misrepresent the reference scope")
            for name in ("accuracy", "balanced_team_recall", "predicted_live_coverage", "predicted_dead_fraction", "predicted_unknown_fraction"):
                number = value[name]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
                    raise ValueError("Invalid fresh audit score")
    recomputed = decide(audit["scores"])
    if not same_evidence(recomputed, audit["gate"]) or not same_evidence(policy, recomputed["policy"]):
        raise ValueError("Fresh audit gate differs from its scores")
    if type(wrapper["passed"]) is not bool or wrapper["passed"] != recomputed["passed"]:
        raise ValueError("Fresh audit gate decision changed")
    return recomputed["passed"]


def verify_v5_decision(evidence, release_model_sha256, audit=None):
    """Audit one frozen experimental winner; require BOTH gates for promotion."""
    from tools.train_joint_campaign import BASELINE, CONFIG_SHA256, select
    recomputed = select(evidence["records"])
    if not same_evidence(recomputed, evidence["selection"]):
        raise ValueError("Joint/boundary selection recomputation changed")
    selected = recomputed["selected"] != BASELINE
    if (evidence["development_candidate_selected"] is not selected
            or evidence["competition_labels_used"] is not False
            or evidence["independent_test_claimed"] is not False
            or evidence["incumbent_historical_predictions_reproduced"] is not True
            or evidence["protocol"]["fresh_audit_used_for_selection"] is not False):
        raise ValueError("Joint/boundary evaluation separation changed")
    if evidence["protocol"]["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Joint/boundary fixed configuration changed")
    decision = evidence["release_decision"]
    if (type(decision["promoted"]) is not bool or decision["model_sha256"] != release_model_sha256
            or not isinstance(decision["reason"], str) or not decision["reason"].strip()):
        raise ValueError("Invalid joint/boundary release decision")
    candidate = evidence["release_candidate_model_sha256"]
    if recomputed["audit_candidate"] == BASELINE or candidate == V14_MODEL_SHA256:
        raise ValueError("V5 must audit the single preselected experimental winner")
    if audit is None: raise ValueError("V5 requires the fixed fresh-match audit, including rejected development candidates")
    passed = verify_fresh_audit(audit, candidate, CONFIG_SHA256)
    promoted = selected and passed
    status = "failed-development-retained-baseline" if not selected else ("passed" if passed else "failed-retained-baseline")
    if (decision["promoted"] != promoted or decision["independent_audit_status"] != status
            or release_model_sha256 != (candidate if promoted else V14_MODEL_SHA256)):
        raise ValueError("Joint/boundary release does not follow the fresh audit")
    return recomputed


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _valid_rate(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def _verify_event_rates(value):
    counts = [value[k] for k in ("n_gt", "n_pred", "n_matched")]
    if any(type(n) is not int or n < 0 for n in counts) or counts[2] > min(counts[:2]):
        raise ValueError("Invalid temporal event counts")
    gt, pred, hits = counts
    expected = {"precision": hits/pred if pred else 0., "recall": hits/gt if gt else 0.,
                "f1": 2*hits/(gt+pred) if gt+pred else 0.}
    if any(not _valid_rate(value[k]) or not same_evidence(value[k], v) for k, v in expected.items()):
        raise ValueError("Temporal event rates disagree with counts")


def verify_fresh_audit_v6(wrapper, candidate_sha256, config_sha256):
    """Authenticate the one-shot dense-provider/silver audit and recompute gates."""
    from tools.fresh_benchmark_v6 import SPLIT, POLICY, WCSF, GEOMETRY_TOLERANCE_METRES, decide
    audit, policy = wrapper["audit"], wrapper["policy"]
    protocol = audit["protocol"]
    if not same_evidence(policy, POLICY) or not same_evidence(policy, protocol["policy"]):
        raise ValueError("V6 fresh policy differs from the fixed protocol")
    if (wrapper["baseline_model_sha256"] != V15_MODEL_SHA256
            or wrapper["candidate_model_sha256"] != candidate_sha256 or not _valid_digest(candidate_sha256)):
        raise ValueError("V6 fresh audit compares different model bytes")
    if (protocol["schema_version"] != 1 or protocol["selection_frozen_before_reference_access"] is not True
            or protocol["no_training_or_retuning"] is not True or protocol["dense_human_had_validation"] is not False
            or audit["reference_fields_used_for_inference"] is not False or audit["dense_human_had_validation"] is not False
            or type(audit["reference_match_read_count"]) is not int or audit["reference_match_read_count"] != 3
            or protocol["geometry_tolerance_metres"] != GEOMETRY_TOLERANCE_METRES):
        raise ValueError("V6 fresh audit separation or scope changed")
    if not protocol_matches_hash(protocol, wrapper["audit_protocol_sha256"]):
        raise ValueError("V6 frozen fresh protocol hash changed")
    encoded = wrapper["split_file_base64"]
    if not isinstance(encoded, str): raise ValueError("V6 split requires exact original bytes")
    split_bytes = base64.b64decode(encoded, validate=True)
    if (protocol["split"] != SPLIT or json.loads(split_bytes.decode("utf-8-sig")) != SPLIT
            or hashlib.sha256(split_bytes).hexdigest() != protocol["split_sha256"]):
        raise ValueError("V6 fresh match split changed")
    if set(protocol["artifacts"]) != {"baseline", "candidate"}:
        raise ValueError("V6 audit must freeze exactly two models")
    for name, expected in (("baseline", V15_MODEL_SHA256), ("candidate", candidate_sha256)):
        artifact = protocol["artifacts"][name]
        if artifact["model_sha256"] != expected or artifact["config_sha256"] != config_sha256:
            raise ValueError("V6 fresh frozen model/configuration differs")
    matches = {"wcsf2023", *[f"skillcorner{m}" for m in SPLIT["final_silver_matches"]]}
    if any(set(audit[k]) != matches for k in ("scores", "references", "reference_counts")):
        raise ValueError("V6 audit requires all three reserved matches")
    predictions = {f"{m}-{which}.csv" for m in matches for which in ("baseline", "candidate")}
    if (set(audit["prediction_sha256"]) != predictions
            or any(not _valid_digest(v) for v in audit["prediction_sha256"].values())):
        raise ValueError("V6 audit requires hashes for all six frozen predictions")
    dense_counts = audit["reference_counts"]["wcsf2023"]
    if (set(dense_counts["class_counts"]) != set("HADX")
            or any(type(v) is not int or v < 0 for v in dense_counts["class_counts"].values())
            or dense_counts["annotation_method"] != "undocumented_provider_flags"
            or dense_counts["scored_frames"] != sum(dense_counts["class_counts"][c] for c in "HAD")):
        raise ValueError("V6 dense reference support or provenance differs")
    if (set(audit["references"]["wcsf2023"]) != set(WCSF)
            or any(not _valid_digest(v) for v in audit["references"]["wcsf2023"].values())):
        raise ValueError("V6 dense reference hashes missing")
    dense = audit["scores"]["wcsf2023"]
    if set(dense) != {"baseline", "candidate"}: raise ValueError("V6 dense scores are not paired")
    for value in dense.values():
        if (value["reference_quality"] != "dense_provider_flags_annotation_unverified"
                or value["dense_human_had_truth"] is not False or value["fps"] != 10
                or type(value["n"]) is not int or value["n"] <= 0 or value["n"] != dense_counts["scored_frames"]
                or value["class_counts"] != {c:dense_counts["class_counts"][c] for c in "HAD"}):
            raise ValueError("V6 dense score misrepresents its reference support")
        for metric in ("spellF1", "macroF1", "dead_precision", "dead_recall", "dead_f1"):
            if not _valid_rate(value[metric]): raise ValueError("Invalid V6 dense score")
        temporal = value["temporal"]
        for key in ("dead_segments", "dead_onsets", "dead_offsets"):
            _verify_event_rates(temporal[key])
            if temporal[key]["n_gt"] != dense["baseline"]["temporal"][key]["n_gt"]:
                raise ValueError("V6 dense models scored different reference events")
        short = temporal["short_possessions"]
        if (type(short["n_gt"]) is not int or type(short["n_matched"]) is not int
                or not 0 <= short["n_matched"] <= short["n_gt"]
                or short["n_gt"] != dense["baseline"]["temporal"]["short_possessions"]["n_gt"]
                or not same_evidence(short["recall"], short["n_matched"]/short["n_gt"] if short["n_gt"] else None)):
            raise ValueError("V6 short-possession support/rate mismatch")
    for match in matches - {"wcsf2023"}:
        pair, counts, ref = audit["scores"][match], audit["reference_counts"][match], audit["references"][match]
        if (set(pair) != {"baseline", "candidate"} or ref["quality"] != "provider_silver"
                or not _valid_digest(ref["sha256"]) or counts["gap_frames_are_unknown"] is not True
                or counts["dead_reference_frames"] != 0):
            raise ValueError("V6 silver reference gaps cannot become dead truth")
        for value in pair.values():
            if (value["reference_quality"] != "provider_silver" or value["metric"] != "phase_present_team_accuracy"
                    or value["dense_had_truth"] is not False or value["dense_possession_truth"] is not False
                    or value["spellF1"] is not None or type(value["n"]) is not int or value["n"] <= 0
                    or value["n"] != counts["scored_frames"]):
                raise ValueError("V6 silver score misrepresents its reference scope")
            for key in ("accuracy", "balanced_team_recall", "predicted_live_coverage", "predicted_dead_fraction", "predicted_unknown_fraction"):
                if not _valid_rate(value[key]): raise ValueError("Invalid V6 silver score")
    recomputed = decide(audit["scores"])
    if (not same_evidence(recomputed, audit["gate"]) or type(wrapper["passed"]) is not bool
            or wrapper["passed"] != recomputed["passed"]):
        raise ValueError("V6 fresh gate differs from its scores")
    return recomputed["passed"]


def verify_v6_candidate_artifact(evidence, path):
    """Keep the exact audited experiment available even when release falls back."""
    if sha256(path) != evidence["release_candidate_model_sha256"]:
        raise ValueError("V6 stored experimental model hash differs from its audit")
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("candidate_id") != evidence["selection"]["audit_candidate"]:
        raise ValueError("V6 stored experiment is not the development-selected audit candidate")


def verify_v6_decision(evidence, release_model_sha256, audit=None):
    """Both gates are mandatory; failed development still requires one audit."""
    from tools.train_temporal_v6 import BASELINE, CONFIG_SHA256, GATE, select
    recomputed = select(evidence["records"])
    if not same_evidence(recomputed, evidence["selection"]):
        raise ValueError("V6 temporal selection recomputation changed")
    selected = recomputed["selected"] != BASELINE
    if (evidence["development_candidate_selected"] is not selected
            or evidence["competition_labels_used"] is not False or evidence["independent_test_claimed"] is not False
            or evidence["incumbent_historical_predictions_reproduced"] is not True
            or evidence["protocol"]["fresh_audit_used_for_selection"] is not False):
        raise ValueError("V6 temporal evaluation separation changed")
    if (evidence["protocol"]["config_sha256"] != CONFIG_SHA256
            or evidence["protocol"]["baseline_sha256"] != V15_MODEL_SHA256
            or evidence["protocol"]["archive_sha256"] != V15_SOURCE_SHA256
            or not same_evidence(evidence["protocol"]["gate"], GATE)):
        raise ValueError("V6 fixed baseline, configuration or gate changed")
    decision = evidence["release_decision"]
    if (type(decision["promoted"]) is not bool or decision["model_sha256"] != release_model_sha256
            or not isinstance(decision["reason"], str) or not decision["reason"].strip()):
        raise ValueError("Invalid V6 release decision")
    candidate = evidence["release_candidate_model_sha256"]
    if recomputed["audit_candidate"] == BASELINE or candidate == V15_MODEL_SHA256 or not _valid_digest(candidate):
        raise ValueError("V6 must audit one preselected experimental winner")
    if audit is None: raise ValueError("V6 requires a fresh audit even if development rejected its experiment")
    passed = verify_fresh_audit_v6(audit, candidate, CONFIG_SHA256)
    promoted = selected and passed
    status = "failed-development-retained-baseline" if not selected else ("passed" if passed else "failed-retained-baseline")
    if (decision["promoted"] != promoted or decision["independent_audit_status"] != status
            or release_model_sha256 != (candidate if promoted else V15_MODEL_SHA256)):
        raise ValueError("V6 promotion/fallback does not follow both gates")
    return recomputed


def same_evidence(actual, expected):
    """Preserve decisions exactly; tolerate only floating summation roundoff.

    The frozen v3 selector reduces some diagnostic means in set iteration order.
    Python hash randomization changes the last few floating-point bits, not the
    model, ranking or gate decision. Do not alter the frozen scientific source.
    """
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_evidence(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(same_evidence(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def main():
    manifest = json.loads((ROOT / "submission/manifest.json").read_text(encoding="utf-8"))
    checks = {
        "submission/Xtension Labs.csv": manifest["output"]["sha256"],
        "models/submission.json": manifest["learned_model"]["sha256"],
        "submission/config.json": manifest["config_sha256"],
        "configs/default.json": manifest["config_sha256"],
    }
    failures = []
    for name, expected in checks.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            failures.append(name)
        print(f"{'OK' if actual == expected else 'FAIL'}  {name}")

    digest = hashlib.sha256()
    for path in [ROOT / "games.py", *sorted((ROOT / "possession").glob("*.py"))]:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != manifest["code_sha256"]:
        failures.append("inference source")
    print(f"{'FAIL' if 'inference source' in failures else 'OK'}  inference source")
    selection_path = ROOT / "models/control-selection.json"
    robust_path = ROOT / "models/robust-selection.json"
    multimodal_path = ROOT / "models/multimodal-selection.json"
    joint_path = ROOT / "models/joint-selection.json"
    temporal_path = ROOT / "models/temporal-selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        historical_model = ROOT / "models/baselines/v1.2.json"
        expected_current = hashlib.sha256(historical_model.read_bytes()).hexdigest() if robust_path.exists() else manifest["learned_model"]["sha256"]
        if selection["release_model_sha256"] != expected_current:
            failures.append("selected release model")
        if robust_path.exists():
            archive = ROOT / "models/baselines/v1.2-source.zip"
            if hashlib.sha256(archive.read_bytes()).hexdigest() != "f42507eba5601776046c33a2f9df8d46384ea732a19b4e019d2ec35bad23786d":
                failures.append("historical v1.2 source archive")
            with zipfile.ZipFile(archive) as source:
                for name, expected in selection["source_hashes"].items():
                    if hashlib.sha256(source.read(name)).hexdigest() != expected:
                        failures.append(f"historical campaign source: {name}")
        else:
            for name, expected in selection["source_hashes"].items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                    failures.append(f"campaign source: {name}")
        selected = selection["release_candidate"]
        if selected != "release-1.1":
            if not selection["promoted"] or not selection["final_gate"]["passed"]:
                failures.append("promotion evidence")
            if selection["final_set_retuning"] or selection["competition_public_labels_used"]:
                failures.append("evaluation separation")
        print(f"{'FAIL' if len(failures) else 'OK'}  campaign and release provenance")
    if robust_path.exists():
        from tools.train_robust import select
        evidence = json.loads(robust_path.read_text(encoding="utf-8"))
        recomputed = select(evidence["records"])
        if not same_evidence(recomputed, evidence["selection"]):
            failures.append("robust selection recomputation")
        robust_release_sha256 = sha256(ROOT / "models/baselines/v1.3.json") if multimodal_path.exists() else manifest["learned_model"]["sha256"]
        if evidence["release_model_sha256"] != robust_release_sha256:
            failures.append("robust release model")
        if evidence["competition_labels_used"] or evidence["independent_test_claimed"]:
            failures.append("robust evaluation separation")
        if multimodal_path.exists():
            archive = ROOT / "models/baselines/v1.3-source.zip"
            if sha256(archive) != V13_SOURCE_SHA256 or robust_release_sha256 != V13_MODEL_SHA256:
                failures.append("historical v1.3 source/model archive")
            with zipfile.ZipFile(archive) as source:
                for name, expected in evidence["source_hashes"].items():
                    if hashlib.sha256(source.read(name)).hexdigest() != expected:
                        failures.append(f"historical robust campaign source: {name}")
            # The old selection code itself must still match before trusting its recomputation.
            if sha256(ROOT / "tools/train_robust.py") != evidence["source_hashes"]["tools/train_robust.py"]:
                failures.append("historical robust selector changed")
        else:
            for name, expected in evidence["source_hashes"].items():
                if sha256(ROOT / name) != expected:
                    failures.append(f"robust campaign source: {name}")
        promoted = recomputed["selected"] != "release-1.2"
        if evidence["promoted"] != promoted:
            failures.append("robust promotion flag")
        if not promoted and evidence["release_model_sha256"] != evidence["protocol"]["baseline_sha256"]:
            failures.append("baseline fallback bytes")
        print(f"{'FAIL' if failures else 'OK'}  robust development CV and release provenance")
    if multimodal_path.exists():
        evidence = json.loads(multimodal_path.read_text(encoding="utf-8"))
        audit_path = ROOT / "models/independent-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
        try:
            if not robust_path.exists():
                raise ValueError("Historical v1.3 selection evidence is missing")
            v4_release_sha = sha256(ROOT / "models/baselines/v1.4.json") if joint_path.exists() else manifest["learned_model"]["sha256"]
            verify_v4_decision(evidence, v4_release_sha, audit)
            if joint_path.exists():
                if v4_release_sha != V14_MODEL_SHA256:
                    raise ValueError("Historical v1.4 model changed")
                if evidence["source_hashes"] != evidence["protocol"]["source_hashes"]:
                    raise ValueError("Historical v4 source maps disagree")
                verify_archived_sources(evidence["source_hashes"], ROOT / "models/baselines/v1.4-source.zip", V14_SOURCE_SHA256)
                if sha256(ROOT / "tools/train_multimodal.py") != evidence["source_hashes"]["tools/train_multimodal.py"]:
                    raise ValueError("Historical multimodal selector changed")
            else:
                from tools.train_multimodal import scientific_hashes
                current_sources = scientific_hashes()
                if current_sources != evidence["source_hashes"] or current_sources != evidence["protocol"]["source_hashes"]:
                    raise ValueError("Multimodal scientific source hashes changed")
            if not protocol_matches_hash(evidence["protocol"], evidence["protocol_sha256"]):
                raise ValueError("Multimodal embedded protocol hash changed")
            if evidence["protocol"]["baseline_sha256"] != V13_MODEL_SHA256:
                raise ValueError("Multimodal baseline changed")
            if audit is not None:
                if joint_path.exists():
                    verify_archived_sources(audit["audit"]["protocol"]["sources"], ROOT / "models/baselines/v1.4-source.zip", V14_SOURCE_SHA256)
                else:
                    sources = {p.relative_to(ROOT).as_posix(): sha256(p)
                               for p in [ROOT / "tools/independent_benchmark.py", *sorted((ROOT / "possession").glob("*.py"))]}
                    if sources != audit["audit"]["protocol"]["sources"]:
                        raise ValueError("Paired-audit inference source hashes changed")
        except (ValueError, KeyError, TypeError) as error:
            failures.append(f"multimodal release evidence: {error}")
        print(f"{'FAIL' if failures else 'OK'}  multimodal development CV and release decision")
    if joint_path.exists():
        from tools.train_joint_campaign import scientific_hashes
        evidence = json.loads(joint_path.read_text(encoding="utf-8"))
        audit_path = ROOT / "models/fresh-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
        try:
            if not multimodal_path.exists(): raise ValueError("Historical v1.4 selection evidence is missing")
            v5_release_sha = sha256(ROOT / "models/baselines/v1.5.json") if temporal_path.exists() else manifest["learned_model"]["sha256"]
            verify_v5_decision(evidence, v5_release_sha, audit)
            if temporal_path.exists():
                if v5_release_sha != V15_MODEL_SHA256: raise ValueError("Historical v1.5 model changed")
                if evidence["source_hashes"] != evidence["protocol"]["source_hashes"]:
                    raise ValueError("Historical v5 source maps disagree")
                verify_archived_sources(evidence["source_hashes"], ROOT / "models/baselines/v1.5-source.zip", V15_SOURCE_SHA256)
                if sha256(ROOT / "tools/train_joint_campaign.py") != evidence["source_hashes"]["tools/train_joint_campaign.py"]:
                    raise ValueError("Historical v5 selector changed")
            else:
                current_sources = scientific_hashes()
                if current_sources != evidence["source_hashes"] or current_sources != evidence["protocol"]["source_hashes"]:
                    raise ValueError("Joint/boundary scientific source hashes changed")
            if not protocol_matches_hash(evidence["protocol"], evidence["protocol_sha256"]):
                raise ValueError("Joint/boundary embedded protocol hash changed")
            if evidence["protocol"]["baseline_sha256"] != V14_MODEL_SHA256:
                raise ValueError("Joint/boundary baseline changed")
            if audit is not None:
                sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [
                    ROOT / "tools/fresh_benchmark.py", ROOT / "tools/independent_benchmark.py",
                    *sorted((ROOT / "possession").glob("*.py"))]}
                if temporal_path.exists():
                    verify_archived_sources(audit["audit"]["protocol"]["sources"], ROOT / "models/baselines/v1.5-source.zip", V15_SOURCE_SHA256)
                elif sources != audit["audit"]["protocol"]["sources"]:
                    raise ValueError("Fresh-audit inference source hashes changed")
                if manifest["config_sha256"] != audit["audit"]["protocol"]["artifacts"]["baseline"]["config_sha256"]:
                    raise ValueError("Fresh audit configuration differs from release")
        except (ValueError, KeyError, TypeError) as error:
            failures.append(f"joint/boundary release evidence: {error}")
        print(f"{'FAIL' if failures else 'OK'}  joint/boundary development CV and fresh audit decision")
    if temporal_path.exists():
        from tools.train_temporal_v6 import scientific_hashes, CONFIG_SHA256
        evidence = json.loads(temporal_path.read_text(encoding="utf-8"))
        audit_path = ROOT / "models/fresh-audit-v6.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
        try:
            if not joint_path.exists(): raise ValueError("Historical v1.5 selection evidence is missing")
            verify_v6_decision(evidence, manifest["learned_model"]["sha256"], audit)
            verify_v6_candidate_artifact(evidence, ROOT / "models/experiments/v6-candidate.json")
            sources = scientific_hashes()
            if sources != evidence["source_hashes"] or sources != evidence["protocol"]["source_hashes"]:
                raise ValueError("V6 temporal scientific sources changed")
            if not protocol_matches_hash(evidence["protocol"], evidence["protocol_sha256"]):
                raise ValueError("V6 embedded campaign protocol hash changed")
            if sha256(ROOT / "models/baselines/v1.5.json") != V15_MODEL_SHA256:
                raise ValueError("V6 historical baseline model changed")
            if sha256(ROOT / "configs/baselines/v1.5.json") != CONFIG_SHA256 or manifest["config_sha256"] != CONFIG_SHA256:
                raise ValueError("V6 historical/release configuration changed")
            names = ("tools/fresh_benchmark_v6.py", "tools/fresh_benchmark.py", "tools/independent_benchmark.py",
                     "tools/temporal_diagnostics_v6.py", "tools/control_diagnostics.py", "evaluation/possession_metrics.py")
            fresh_sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [
                *[ROOT / n for n in names], *sorted((ROOT / "possession").glob("*.py"))]}
            if fresh_sources != audit["audit"]["protocol"]["sources"]:
                raise ValueError("V6 fresh audit scientific sources changed")
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
            failures.append(f"v6 temporal release evidence: {error}")
        print(f"{'FAIL' if failures else 'OK'}  v6 temporal development and fresh audit decision")
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}", file=sys.stderr)
        print("Release bytes changed; regenerate the matching artifacts and provenance.", file=sys.stderr)
        return 1
    print("Frozen release verified. Match data was not required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
