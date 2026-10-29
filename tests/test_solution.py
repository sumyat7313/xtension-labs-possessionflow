"""Engineering tests use invented data only, never the public match labels."""

from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.bundle import make_bundle, source_files
from possession.config import Config, load_config
from possession.decoder import decode, observation_probabilities
from possession.features import clean_ball, extract, runs
from possession.io import Tracking, read_events, read_tracking, sha256, write_submission
from possession.pipeline import infer, validate_csv
from possession.pipeline import predict
from possession.tags import build, team_index


def tracking(n=150):
    players = np.tile(np.array([[[0.3, 0.5], [0.7, 0.5]]], dtype=np.float32), (n, 1, 1))
    ball = np.tile([0.302, 0.5], (n, 1)).astype(np.float32)
    return Tracking(players, np.full((n, 2), 2, dtype=np.float32), ball,
                    np.full(n, 2, dtype=np.float32), np.array([0, 1]), ("Barcelona", "Athletic Club"))


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.meta = self.root / "meta.xml"
        self.meta.write_text('<EPTS xmlns="test"><Teams><Team id="h"><Name>Home</Name></Team>'
                             '<Team id="a"><Name>Away</Name></Team></Teams><Players>'
                             '<Player teamId="h"/><Player teamId="a"/></Players></EPTS>')
        self.raw = self.root / "raw.txt"

    def tearDown(self):
        self.temp.cleanup()

    def line(self, index):
        return f"{index}:0.3,0.5,2;0.7,0.5,2:0.302,0.5,2\n"

    def test_alignment_is_one_based(self):
        self.raw.write_text(self.line(1) + self.line(2))
        result = read_tracking(self.meta, self.raw, 2)
        self.assertEqual(result.players.shape, (2, 2, 2))
        self.assertEqual(result.teams.tolist(), [0, 1])
        self.assertEqual(result.team_names, ("Home", "Away"))

    def test_duplicate_shifted_and_truncated_ids_rejected(self):
        for value in [self.line(1) + self.line(1), self.line(0) + self.line(1), self.line(1)]:
            self.raw.write_text(value)
            with self.assertRaises(ValueError):
                read_tracking(self.meta, self.raw, 2)

    def test_malformed_slot_and_unknown_team_rejected(self):
        self.raw.write_text("1:0.3,0.5;2,0.7,0.5,2:0.3,0.5,2\n")
        with self.assertRaises(ValueError):
            read_tracking(self.meta, self.raw, 1)
        self.meta.write_text(self.meta.read_text().replace('teamId="a"', 'teamId="unknown"'))
        with self.assertRaises(ValueError):
            read_tracking(self.meta, self.raw, 1)

    def test_tag_timebase_and_case(self):
        xml = self.root / "events.xml"
        xml.write_text('<file><instance><start>1.0</start><end>2.0</end><code>BUILD UP</code>'
                       '<label><group>Team</group><text>Barcelona</text></label></instance></file>')
        event = read_events(xml, 100, 25)[0]
        self.assertEqual((event["start"], event["end"]), (25, 50))
        self.assertEqual(event["tags"]["team"], "Barcelona")


class FeatureTests(unittest.TestCase):
    def test_gap_interpolation_is_short_bounded_and_half_local(self):
        raw = np.tile([0.5, 0.5], (40, 1))
        raw[3:7] = np.nan
        raw[12:23] = np.nan
        ball, seen, weight, rejected, filled = clean_ball(raw, (105.3, 68), [(1, 0, 19), (2, 20, 39)], 25, Config())
        self.assertTrue(np.isfinite(ball[3:7]).all())
        self.assertTrue(np.isnan(ball[12:23]).all())
        self.assertFalse(seen[3:7].any())
        self.assertEqual(filled, 4)
        self.assertTrue((weight[3:7] < 1).all())

    def test_isolated_spike_is_rejected(self):
        raw = np.tile([0.5, 0.5], (20, 1))
        raw[10] = [0.9, 0.9]
        _, seen, _, rejected, filled = clean_ball(raw, (105.3, 68), [(1, 0, 19)], 25, Config())
        self.assertEqual(rejected, 1)
        self.assertFalse(seen[10])
        self.assertEqual(filled, 1)

    def test_distances_use_rectangular_pitch_metres(self):
        data = tracking(30)
        data.ball[:] = [0.5, 0.5]
        data.players[:, 0] = [0.51, 0.5]
        data.players[:, 1] = [0.5, 0.514]
        feat = extract(data, (105.3, 68), [(1, 0, 29)], 25, Config())
        self.assertLess(feat.team_log_odds[10], 0)

    def test_missing_ball_does_not_alone_mean_dead(self):
        data = tracking(100)
        data.ball[30:70] = np.nan
        feat = extract(data, (105.3, 68), [(1, 0, 99)], 25, Config())
        tags = build([], feat, data.team_names, 100, 25, [(1, 0, 99)], Config())
        probabilities = observation_probabilities(feat, tags, Config())
        self.assertLess(probabilities[50, 2], probabilities[50, 0])
        labels, *_ = infer(data, [], [(1, 0, 99)], (105.3, 68), 25, Config())
        self.assertTrue((labels == "H").all())

    def test_defensive_tag_is_inverted_and_na_is_neutral(self):
        data = tracking(100)
        feat = extract(data, (105.3, 68), [(1, 0, 99)], 25, Config())
        events = [{"start": 10, "end": 90, "code": "defending in middle third", "tags": {"team": "Barcelona"}}]
        evidence = build(events, feat, data.team_names, 100, 25, [(1, 0, 99)], Config())
        self.assertLess(evidence.team_log_odds[50], 0)
        events[0]["tags"]["team"] = "N/A"
        evidence = build(events, feat, data.team_names, 100, 25, [(1, 0, 99)], Config())
        self.assertEqual(evidence.team_log_odds[50], 0)
        self.assertEqual(team_index("Atletic Club", data.team_names), 1)

    def test_restart_is_inferred_from_motion_not_interval_edges(self):
        data = tracking(100)
        feat = extract(data, (105.3, 68), [(1, 0, 99)], 25, Config())
        feat.ball_speed[:50] = 0
        feat.ball_speed[50:] = 4
        feat.stationary[:50] = 1
        feat.stationary[50:] = 0
        events = [{"start": 20, "end": 75, "code": "set pieces", "tags": {"team": "Barcelona"}}]
        evidence = build(events, feat, data.team_names, 100, 25, [(1, 0, 99)], Config())
        self.assertEqual(evidence.restart_candidates, 1)
        first_live = np.flatnonzero(evidence.restart_live)[0]
        self.assertGreater(first_live, 40)
        self.assertLess(first_live, 55)
        self.assertGreater(evidence.dead[30], 0)

    def test_halves_reset_and_outside_frames_are_x(self):
        data = tracking(120)
        data.ball[80:] = [0.698, 0.5]
        labels, *_ = infer(data, [], [(1, 10, 39), (2, 80, 109)], (105.3, 68), 25, Config())
        self.assertTrue((labels[10:40] == "H").all())
        self.assertTrue((labels[80:110] == "A").all())
        self.assertTrue((labels[40:80] == "X").all())

    def test_no_evidence_abstains(self):
        data = tracking(40)
        data.ball[:] = np.nan
        labels, _, _, _, warnings = infer(data, [], [(1, 0, 39)], (105.3, 68), 25, Config())
        self.assertTrue((labels == "X").all())
        self.assertTrue(warnings)


class DecoderTests(unittest.TestCase):
    def test_short_deflection_does_not_switch_possession(self):
        probabilities = np.tile([0.97, 0.02, 0.01], (200, 1))
        probabilities[80:82] = [0.02, 0.97, 0.01]
        self.assertTrue((decode(probabilities, 25, Config()) == 0).all())

    def test_sustained_turnover_and_stoppage_are_kept(self):
        probabilities = np.vstack([np.tile(p, (100, 1)) for p in [[.98, .01, .01], [.01, .98, .01], [.01, .01, .98]]])
        expected = np.repeat([0, 1, 2], 100)
        np.testing.assert_array_equal(decode(probabilities, 25, Config()), expected)

    def test_matches_bruteforce_minimum_cost(self):
        rng = np.random.default_rng(11)
        for min_seconds in (0.2, 0.4):
            config = replace(Config(), min_team_seconds=min_seconds, min_dead_seconds=min_seconds,
                             team_switch_cost=0.3, stop_cost=0.2, restart_cost=0.4)
            probabilities = rng.dirichlet([1, 1, 1], size=7)
            result = decode(probabilities, 5, config)
            costs = np.array([[0, .3, .2], [.3, 0, .2], [.4, .4, 0]])
            def energy(sequence):
                return sum(-np.log(probabilities[t, state]) for t, state in enumerate(sequence)) + sum(
                    costs[a, b] for a, b in zip(sequence[:-1], sequence[1:]))
            def valid(sequence):
                edges = np.r_[0, np.flatnonzero(np.diff(sequence)) + 1, len(sequence)]
                return all(b - a >= round(min_seconds * 5) for a, b in list(zip(edges[:-1], edges[1:]))[1:-1])
            optimum = min(energy(s) for s in itertools.product(range(3), repeat=7) if valid(s))
            self.assertAlmostEqual(energy(result), optimum, places=9)


class OutputTests(unittest.TestCase):
    def test_complete_pipeline_writes_json_and_reproduces_csv(self):
        from games import GAME, GAMES
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            tactical = data / "tactical"
            tactical.mkdir(parents=True)
            (data / ".synthetic-fixture.json").write_text('{"synthetic": true}')
            (tactical / "BAR-ATH_tactical_atd.xml").write_text(
                '<EPTS><Team id="h"><Name>Barcelona</Name></Team>'
                '<Team id="a"><Name>Athletic Club</Name></Team>'
                '<Player teamId="h"/><Player teamId="a"/></EPTS>')
            with (tactical / "BAR-ATH_tactical_atd.txt").open("w") as handle:
                for frame in range(80):
                    ball = "NaN,NaN,NaN" if 20 <= frame < 24 else "0.302,0.5,2"
                    handle.write(f"{frame+1}:0.3,0.5,2;0.7,0.5,2:{ball}\n")
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                '<file><instance><start>0</start><end>3.2</end><code>BUILD UP</code>'
                '<label><group>Team</group><text>Barcelona</text></label></instance></file>')
            spec = dict(GAMES[GAME], n_video_frames=80)
            with patch.dict(GAMES, {GAME: spec}), patch("possession.pipeline.half_video_blocks", return_value=[(1, 0, 79)]):
                first, report = predict(data, root / "first", allow_synthetic=True, progress=lambda _: None)
                second, _ = predict(data, root / "second", allow_synthetic=True, progress=lambda _: None)
            self.assertEqual(first["output"]["sha256"], second["output"]["sha256"])
            self.assertEqual(report["interpolated_frames"], 4)
            self.assertTrue(first["synthetic"])
            self.assertFalse((root / "first" / ".prediction-run.json").exists())
            self.assertEqual(json.loads((root / "first" / "manifest.json").read_text())["output"]["rows"], 80)

    def test_csv_is_reproducible_and_official_validator_accepts_it(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.csv", Path(folder) / "b.csv"
            labels = np.array(list("XHHHDAAX"))
            write_submission(a, labels)
            write_submission(b, labels)
            self.assertEqual(sha256(a), sha256(b))
            self.assertEqual(validate_csv(a, 8), 8)
            a.write_text(a.read_text().replace("0,X", "1,X"))
            with self.assertRaises(ValueError):
                validate_csv(a, 8)

    def test_synthetic_submission_packaging_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "manifest.json").write_text(json.dumps({"synthetic": True}))
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                make_bundle(run / "export", run_dir=run)

    def test_config_rejects_unknown_nonfinite_and_nonphysical_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for value in [{"typo": 1}, {"min_team_seconds": -1}, {"team_switch_cost": float("nan")}, {"interpolated_weight": 2}]:
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_bundle_allowlist_excludes_restricted_assets_and_local_environment(self):
        files = [p.as_posix() for p in source_files()]
        self.assertFalse(any("/data/" in p or "/runs/" in p or ".python-environment" in p for p in files))
        self.assertTrue(any(p.endswith("TECHNICAL_REPORT.md") for p in files))


if __name__ == "__main__":
    unittest.main()
