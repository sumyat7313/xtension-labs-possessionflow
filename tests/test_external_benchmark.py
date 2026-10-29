"""Invented Sportec fixtures independently check the external benchmark adapter.

No competition assets or real external-match observations are loaded here.
Provider conventions are specified in Bassek et al. (2025), Scientific Data 12,
195, page 5, and Kloppy's Sportec tracking and metadata deserializers.
"""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession.config import Config
from possession.pipeline import infer
from tools.external_benchmark import read_match


class ExternalBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.meta = self.root / "metadata.xml"
        # Intentionally reverse XML team order: roles, not order, define home.
        self.meta.write_text(
            '<root><General MatchId="DFL-MAT-J03WOH"/>'
            '<Environment PitchX="100" PitchY="60"/><Teams>'
            '<Team Role="guest" TeamName="Visitors"><Players>'
            '<Player PersonId="away"/></Players></Team>'
            '<Team Role="home" TeamName="Hosts"><Players>'
            '<Player PersonId="home"/></Players></Team></Teams></root>',
            encoding="utf-8")
        self.raw = self.root / "tracking.xml"

    def tearDown(self):
        self.temp.cleanup()

    def write_tracking(self, reference=None, frame_seconds=.04):
        reference = reference or (lambda half, index: ("1", "1"))
        pieces = ["<root>"]
        for half, section, start in [(1, "firstHalf", 1000), (2, "secondHalf", 9000)]:
            for person, team, x, y, speed in [
                ("away", "AWAY", -30, 20, 18),
                ("ball", "BALL", 9.8, -6, 36),
                ("home", "HOME", 10, -6, 18),
            ]:
                pieces.append(f'<FrameSet MatchId="DFL-MAT-J03WOH" '
                              f'GameSection="{section}" PersonId="{person}" TeamId="{team}">')
                for index in range(80):
                    if person == "ball" and index == 10:
                        continue  # A missing observation must not compress the timeline.
                    status, owner = reference(half, index)
                    stamp = (datetime(2022, 1, 1) + timedelta(seconds=index * frame_seconds)).isoformat()
                    pieces.append(f'<Frame N="{start + index}" T="{stamp}" X="{x}" Y="{y}" '
                                  f'S="{speed}" BallStatus="{status}" BallPossession="{owner}"/>')
                pieces.append("</FrameSet>")
        pieces.append("</root>")
        self.raw.write_text("".join(pieces), encoding="utf-8")

    def test_role_mapping_units_and_gaps_keep_frame_identity(self):
        codes = {0: ("0", "2"), 1: ("1", "2"), 2: ("1", "0"), 3: ("2", "1")}
        self.write_tracking(lambda half, index: codes.get(index, ("1", "1")))
        tracking, labels, blocks, pitch, alignment = read_match(self.meta, self.raw, "J03WOH")
        self.assertEqual(pitch, (100, 60))
        self.assertEqual(blocks, [(1, 0, 79), (2, 81, 160)])
        self.assertEqual(alignment["fps"], 25)
        self.assertEqual(alignment["frame_ranges"], {"1": [1000, 1079, 0], "2": [9000, 9079, 81]})
        self.assertEqual(tracking.team_names, ("Hosts", "Visitors"))
        np.testing.assert_array_equal(tracking.teams, [0, 1])
        np.testing.assert_allclose(tracking.players[0, 0], [.6, .4])
        np.testing.assert_allclose(tracking.players[0, 1], [.2, 5 / 6])
        np.testing.assert_allclose(tracking.ball[0], [.598, .4])
        np.testing.assert_allclose(tracking.player_speed[0], [5, 5])
        self.assertEqual(tracking.ball_speed[0], 10)
        self.assertEqual(list(labels[:5]), ["D", "A", "X", "X", "H"])
        self.assertTrue(np.isnan(tracking.ball[10]).all())
        self.assertTrue(np.isfinite(tracking.players[10]).all())
        self.assertTrue(np.isnan(tracking.players[80]).all())
        self.assertEqual(labels[80], "X")
        self.assertTrue(np.isnan(tracking.ball[91]).all())

    def test_reference_mutation_does_not_change_observations_blocks_or_predictions(self):
        self.write_tracking()
        first, first_labels, blocks, pitch, alignment = read_match(self.meta, self.raw, "J03WOH")
        # Flip live/dead and team labels, including each half's boundary frames.
        self.write_tracking(lambda half, index: ("0", "2") if index % 2 else ("1", "2"))
        second, second_labels, other_blocks, other_pitch, other_alignment = read_match(self.meta, self.raw, "J03WOH")
        self.assertFalse(np.array_equal(first_labels, second_labels))
        self.assertEqual(blocks, other_blocks)
        self.assertEqual(pitch, other_pitch)
        self.assertEqual(alignment["frame_ranges"], other_alignment["frame_ranges"])
        for attribute in ("players", "player_speed", "ball", "ball_speed", "teams"):
            np.testing.assert_array_equal(getattr(first, attribute), getattr(second, attribute))
        prediction = infer(first, [], blocks, pitch, 25, Config())[0]
        other_prediction = infer(second, [], other_blocks, other_pitch, 25, Config())[0]
        np.testing.assert_array_equal(prediction, other_prediction)

    def test_non_25hz_input_is_rejected(self):
        self.write_tracking(frame_seconds=.05)
        with self.assertRaisesRegex(ValueError, "25 Hz"):
            read_match(self.meta, self.raw, "J03WOH")


if __name__ == "__main__":
    unittest.main()
