"""Verify committed release bytes without loading restricted match assets."""

import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


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
        from train_robust import select
        evidence = json.loads(robust_path.read_text(encoding="utf-8"))
        recomputed = select(evidence["records"])
        if not same_evidence(recomputed, evidence["selection"]):
            failures.append("robust selection recomputation")
        if evidence["release_model_sha256"] != manifest["learned_model"]["sha256"]:
            failures.append("robust release model")
        if evidence["competition_labels_used"] or evidence["independent_test_claimed"]:
            failures.append("robust evaluation separation")
        for name, expected in evidence["source_hashes"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                failures.append(f"robust campaign source: {name}")
        promoted = recomputed["selected"] != "release-1.2"
        if evidence["promoted"] != promoted:
            failures.append("robust promotion flag")
        if not promoted and evidence["release_model_sha256"] != evidence["protocol"]["baseline_sha256"]:
            failures.append("baseline fallback bytes")
        print(f"{'FAIL' if failures else 'OK'}  robust development CV and release provenance")
    if failures:
        print("Release bytes changed; regenerate the matching artifacts and provenance.", file=sys.stderr)
        return 1
    print("Frozen release verified. Match data was not required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
