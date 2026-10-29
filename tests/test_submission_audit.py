"""Submission packaging must distinguish local artifacts from an actual entry."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from games import GAME, GAMES
from possession.bundle import make_bundle
from possession.io import sha256, write_submission
from possession.pipeline import code_digest


class SubmissionAuditTests(unittest.TestCase):
    def test_source_package_contains_audit_and_keeps_submission_tasks_open(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)
            archive = make_bundle(dest)
            status = json.loads((dest / "bundle-status.json").read_text())
            self.assertFalse(status["local_deliverables_ready"])
            self.assertFalse(status["submission_ready"])
            self.assertIn("Run on actual match assets", status["remaining"])
            self.assertTrue(any("public repository" in item for item in status["remaining"]))
            self.assertTrue(any("submission form" in item for item in status["remaining"]))
            with zipfile.ZipFile(archive) as handle:
                names = set(handle.namelist())
                self.assertTrue({"run-challenge.ps1", "docs/verification.json",
                                 "docs/COMPETITIVE_AUDIT.md"}.issubset(names))
                self.assertFalse(any(name.startswith(("data/", "runs/", ".venv/")) for name in names))
                self.assertNotIn(".python-environment", names)

    def test_validated_local_package_is_not_reported_as_submitted(self):
        # Invented eight-frame fixture exercises packaging only, with no match data.
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            run.mkdir()
            labels = np.array(list("XHHDAADX"))
            write_submission(run / "predictions.csv", labels)
            (run / "config.json").write_text("{}", encoding="utf-8")
            manifest = {"synthetic": False, "code_sha256": code_digest(),
                        "config_sha256": sha256(run / "config.json"),
                        "output": {"sha256": sha256(run / "predictions.csv")}}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            dest = Path(folder) / "package"
            spec = dict(GAMES[GAME], n_video_frames=len(labels))
            with patch.dict(GAMES, {GAME: spec}):
                make_bundle(dest, team_name="test-team", run_dir=run)
            status = json.loads((dest / "bundle-status.json").read_text())
            self.assertTrue(status["local_deliverables_ready"])
            self.assertTrue(status["includes_submission_csv"])
            self.assertFalse(status["submission_ready"])
            self.assertNotIn("Run on actual match assets", status["remaining"])
            self.assertTrue(any("registered team name" in item for item in status["remaining"]))
            self.assertTrue(any("public repository" in item for item in status["remaining"]))
            self.assertTrue(any("submission form" in item for item in status["remaining"]))
            self.assertEqual((dest / "test-team.csv").read_bytes(), (run / "predictions.csv").read_bytes())

    def test_named_team_model_is_preserved_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            run.mkdir()
            write_submission(run / "predictions.csv", np.array(list("HHDA")))
            (run / "config.json").write_text("{}", encoding="utf-8")
            (run / "model.json").write_text('{"fixture": true}', encoding="utf-8")
            manifest = {"synthetic": False, "code_sha256": code_digest(),
                        "config_sha256": sha256(run / "config.json"),
                        "output": {"sha256": sha256(run / "predictions.csv")},
                        "learned_model": {"sha256": sha256(run / "model.json")}}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            spec = dict(GAMES[GAME], n_video_frames=4)
            with patch.dict(GAMES, {GAME: spec}):
                archive = make_bundle(Path(folder) / "package", "Xtension Labs", run)
                with zipfile.ZipFile(archive) as z:
                    self.assertEqual(z.read("models/submission.json"), (run / "model.json").read_bytes())
                self.assertTrue((archive.parent / "Xtension Labs.csv").is_file())
                (run / "model.json").write_text('{"fixture": false}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Trained model changed"):
                    make_bundle(Path(folder) / "changed", "Xtension Labs", run)


if __name__ == "__main__":
    unittest.main()
