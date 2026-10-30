"""Verify committed release bytes without loading restricted match assets."""

import hashlib
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_matches_hash(protocol, expected):
    """Authenticate embedded write_json evidence across Windows/Linux checkouts."""
    value = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return expected in {hashlib.sha256(value.encode()).hexdigest(),
                       hashlib.sha256(value.replace("\n", "\r\n").encode()).hexdigest()}


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
        from tools.train_multimodal import scientific_hashes
        evidence = json.loads(multimodal_path.read_text(encoding="utf-8"))
        audit_path = ROOT / "models/independent-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
        try:
            if not robust_path.exists():
                raise ValueError("Historical v1.3 selection evidence is missing")
            verify_v4_decision(evidence, manifest["learned_model"]["sha256"], audit)
            current_sources = scientific_hashes()
            if current_sources != evidence["source_hashes"] or current_sources != evidence["protocol"]["source_hashes"]:
                raise ValueError("Multimodal scientific source hashes changed")
            if not protocol_matches_hash(evidence["protocol"], evidence["protocol_sha256"]):
                raise ValueError("Multimodal embedded protocol hash changed")
            if evidence["protocol"]["baseline_sha256"] != V13_MODEL_SHA256:
                raise ValueError("Multimodal baseline changed")
            if audit is not None:
                sources = {p.relative_to(ROOT).as_posix(): sha256(p)
                           for p in [ROOT / "tools/independent_benchmark.py", *sorted((ROOT / "possession").glob("*.py"))]}
                if sources != audit["audit"]["protocol"]["sources"]:
                    raise ValueError("Paired-audit inference source hashes changed")
        except (ValueError, KeyError, TypeError) as error:
            failures.append(f"multimodal release evidence: {error}")
        print(f"{'FAIL' if failures else 'OK'}  multimodal development CV and release decision")
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}", file=sys.stderr)
        print("Release bytes changed; regenerate the matching artifacts and provenance.", file=sys.stderr)
        return 1
    print("Frozen release verified. Match data was not required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
