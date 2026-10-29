"""Verify committed release bytes without loading restricted match assets."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


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
    if failures:
        print("Release bytes changed; regenerate the matching artifacts and provenance.", file=sys.stderr)
        return 1
    print("Frozen release verified. Match data was not required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
