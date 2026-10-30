"""Synthetic adapter/alignment tests; no reserved real event labels are read."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import independent_benchmark as bench


class IndependentBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def events(self, rows):
        path = self.folder / "events.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Type", "Team", "Start Frame", "Period"])
            writer.writerows(rows)
        return path

    def tracking(self, wrong_away_time=False):
        for side in ("Home", "Away"):
            path = self.folder / bench.filename(1, f"RawTrackingData_{side}_Team.csv")
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([side]); writer.writerow([1])
                writer.writerow(["Period", "Frame", "Time [s]", "X1", "Y1", "Ball X", "Ball Y"])
                for row in range(12):
                    half = 1 if row < 6 else 2
                    time = row / 25 + (1 if wrong_away_time and side == "Away" else 0)
                    x = .1 + row * .001 if half == 1 else .9 + row * .001
                    writer.writerow([half, 101 + row, time, x, .4, .5, .5])

    def test_exact_frame_lookup_and_no_interval_filling(self):
        path = self.events([("PASS", "Home", 102, 1), ("PASS", "Away", 110, 2),
                            ("SHOT", "Home", 103, 1)])
        refs, counts = bench.point_references(path, np.arange(101, 113), np.repeat([1, 2], 6))
        self.assertEqual(refs.tolist(), list("XHXXXXXXXAXX"))
        self.assertEqual(counts["scored_frames"], 2)
        self.assertEqual(counts["other_events_excluded"], 1)

    def test_conflicting_point_teams_excluded_not_arbitrarily_chosen(self):
        path = self.events([("PASS", "Home", 101, 1), ("PASS", "Away", 101, 1),
                            ("PASS", "Home", 102, 1), ("PASS", "Home", 102, 1)])
        refs, counts = bench.point_references(path, np.array([101, 102]), np.array([1, 1]))
        self.assertEqual(refs.tolist(), ["X", "H"])
        self.assertEqual(counts["conflicting_frames"], 1)
        self.assertEqual(counts["duplicate_same_team"], 1)

    def test_dead_unknown_are_errors_and_conditional_metric_discloses_coverage(self):
        scores = bench.score_points(list("HHAAX"), list("HDAXH"))
        self.assertEqual(scores["n"], 4)
        self.assertEqual(scores["accuracy"], .5)
        self.assertEqual(scores["predicted_live_coverage"], .5)
        self.assertEqual(scores["conditional_live_team_accuracy"], 1)
        self.assertIsNone(scores["spellF1"])
        none = bench.score_points(list("HA"), list("DX"))
        self.assertIsNone(none["conditional_live_team_accuracy"])

    def test_observation_adapter_never_reads_event_file_or_crosses_half(self):
        self.tracking()
        (self.folder / bench.filename(1, "RawEventsData.csv")).write_text("intentionally not a CSV")
        with patch.object(bench, "verify_file") as verify:
            tracking, blocks, pitch, alignment = bench.read_tracking(self.folder, 1)
        self.assertEqual(verify.call_count, 2)
        self.assertTrue(all("TrackingData" in str(call.args[0]) for call in verify.call_args_list))
        self.assertEqual(blocks, [(1, 0, 5), (2, 6, 11)])
        self.assertEqual(pitch, (105., 68.))
        self.assertEqual(tracking.players.shape, (12, 2, 2))
        np.testing.assert_array_equal(alignment["frame_ids"], np.arange(101, 113))
        self.assertTrue(np.isnan(tracking.player_speed[4:8]).all())
        np.testing.assert_allclose(tracking.player_speed[[2, 3, 8, 9]], 2.625)

    def test_timeline_and_event_period_errors_fail_closed(self):
        self.tracking(wrong_away_time=True)
        with patch.object(bench, "verify_file"), self.assertRaises(ValueError):
            bench.read_tracking(self.folder, 1)
        path = self.events([("PASS", "Home", 101, 2)])
        with self.assertRaises(ValueError):
            bench.point_references(path, np.array([101]), np.array([1]))

    def test_split_is_immutable_and_written_before_download(self):
        path = bench.freeze_split(self.folder)
        value = json.loads(path.read_text())
        self.assertEqual(value["reserved_final_sparse_audit"], [2])
        self.assertEqual(path, bench.freeze_split(self.folder))
        value["reserved_final_sparse_audit"] = [1]
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            bench.freeze_split(self.folder)

    def test_git_blob_verification_rejects_corruption(self):
        path = self.folder / "sample.csv"
        data = b"a,b\n1,2\n"
        path.write_bytes(data)
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        bench.verify_file(path, len(data), digest)
        path.write_bytes(data.replace(b"1", b"3"))
        with self.assertRaises(ValueError):
            bench.verify_file(path, len(data), digest)

    def paired_fixture(self):
        candidate, baseline, config = [self.folder / name for name in ("candidate.json", "baseline.json", "config.json")]
        candidate.write_text(json.dumps({"name": "candidate"}))
        baseline.write_text(json.dumps({"name": "baseline"}))
        config.write_text("{}")
        event_path = self.events([("PASS", "Home", 101, 1), ("PASS", "Away", 102, 1)])
        event_path.replace(self.folder / bench.filename(2, "RawEventsData.csv"))
        observations = (object(), [(1, 0, 2)], bench.PITCH,
                        {"frame_ids": np.array([101, 102, 103]), "periods": np.array([1, 1, 1])})
        return candidate, baseline, config, observations

    def test_pair_freezes_both_saves_both_then_reads_references_once(self):
        candidate, baseline, config, observations = self.paired_fixture()
        out = self.folder / "paired-audit"
        original_point_references = bench.point_references
        order = []

        def infer(tracking, events, blocks, pitch, fps, settings, model):
            self.assertEqual(events, [])
            protocol = json.loads((out / "frozen-before-reference-access.json").read_text())
            self.assertEqual(protocol["artifacts"]["baseline"]["model_sha256"], bench.sha256(baseline))
            self.assertEqual(protocol["artifacts"]["candidate"]["model_sha256"], bench.sha256(candidate))
            order.append(model["name"])
            labels = "HAD" if model["name"] == "candidate" else "ADD"
            return np.array(list(labels)), None, None, None, []

        def references(*args):
            self.assertTrue((out / "candidate-predictions.csv").exists())
            self.assertTrue((out / "baseline-predictions.csv").exists())
            self.assertTrue((self.folder / "game2-reference-access.json").exists())
            order.append("references")
            return original_point_references(*args)

        with (patch.object(bench, "read_tracking", return_value=observations),
              patch.object(bench, "verify_file"),
              patch.object(bench, "load_config", return_value=object()),
              patch("possession.learned.load_model", side_effect=lambda p: json.loads(p.read_text())),
              patch("possession.pipeline.infer", side_effect=infer),
              patch("builtins.print"),
              patch.object(bench, "point_references", side_effect=references) as reference_reader):
            result = bench.evaluate_pair(self.folder, 2, candidate, baseline, config, out)
        self.assertEqual(order, ["baseline", "candidate", "references"])
        self.assertEqual(reference_reader.call_count, 1)
        self.assertEqual(result["scores"]["candidate"]["accuracy"], 1)
        self.assertEqual(result["scores"]["baseline"]["accuracy"], 0)
        self.assertEqual(result["candidate_minus_baseline"]["accuracy"], 1)
        with self.assertRaisesRegex(ValueError, "already been accessed"):
            bench.evaluate_pair(self.folder, 2, candidate, baseline, config, self.folder / "retry")

    def test_pair_artifact_change_fails_before_consuming_reserved_references(self):
        candidate, baseline, config, observations = self.paired_fixture()

        def mutate(*args):
            candidate.write_text(json.dumps({"name": "candidate", "changed": True}))
            return np.array(list("HAD")), None, None, None, []

        with (patch.object(bench, "read_tracking", return_value=observations),
              patch.object(bench, "verify_file"),
              patch.object(bench, "load_config", return_value=object()),
              patch("possession.learned.load_model", side_effect=lambda p: json.loads(p.read_text())),
              patch("possession.pipeline.infer", side_effect=mutate),
              patch.object(bench, "point_references") as reference_reader):
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                bench.evaluate_pair(self.folder, 2, candidate, baseline, config, self.folder / "changed")
        self.assertEqual(reference_reader.call_count, 0)
        self.assertFalse((self.folder / "game2-reference-access.json").exists())


if __name__ == "__main__":
    unittest.main()
