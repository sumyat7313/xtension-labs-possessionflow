"""Characterize synthetic observation sensitivity; never estimate match accuracy.

Run from the repository root:
    python tools/audit_observations.py --output runs/observation-audit.json

This tool reads no match assets or ground-truth labels and changes no parameters.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from possession.config import Config
from possession.io import Tracking, write_json
from possession.pipeline import code_digest, infer


def scenario(ball_speed, player_speed, active_phase=False, missing_middle=False):
    """Hold artificial geometry fixed while changing named observations."""
    n, fps = 500, 25
    teams = np.repeat([0, 1], 11)
    players = np.empty((n, 22, 2), dtype=np.float32)
    players[:, :11] = [.30, .50]
    players[:, 11:] = [.70, .50]
    ball = np.tile([.302, .50], (n, 1)).astype(np.float32)
    if missing_middle:
        ball[100:400] = np.nan
    data = Tracking(
        players, np.full((n, 22), player_speed, dtype=np.float32), ball,
        np.full(n, ball_speed, dtype=np.float32), teams, ("Home", "Away"),
    )
    events = ([{"start": 0, "end": n, "code": "build up", "tags": {"team": "Home"}}]
              if active_phase else [])
    labels, probabilities, features, _, _ = infer(
        data, events, [(1, 0, n - 1)], (105.3, 68), fps, Config(),
    )
    speed = features.ball_speed[250]
    return {
        "reported_ball_speed_mps": ball_speed,
        "reported_player_speed_mps": player_speed,
        "active_home_phase": active_phase,
        "missing_middle": missing_middle,
        "frame_count": n,
        "label_counts": {label: int((labels == label).sum()) for label in "HADX"},
        "middle_observation_support_HAD": probabilities[250].round(6).tolist(),
        "middle_extracted_ball_speed_mps": float(speed) if np.isfinite(speed) else None,
        "middle_stationary": round(float(features.stationary[250]), 6),
        "middle_still_players": round(float(features.still_players[250]), 6),
    }


def audit():
    scenarios = [
        scenario(ball_speed, player_speed, phase)
        for ball_speed, player_speed in [(0, 0), (0, .2), (0, .4), (.6, .4)]
        for phase in [False, True]
    ]
    scenarios += [scenario(0, speed, missing_middle=True) for speed in [0, .2, .4]]
    return {
        "purpose": "Observation sensitivity characterization; not accuracy or model selection",
        "uses_match_assets": False,
        "uses_ground_truth": False,
        "parameters_changed": False,
        "code_sha256": code_digest(),
        "configuration": Config().as_dict(),
        "caveats": [
            "Stationary geometry does not determine the referee's live/dead decision.",
            "These scenarios have no ground-truth labels and cannot be scored for accuracy.",
            "Fixed duplicate player positions and independently set speed fields are deliberate artificial inputs to isolate evidence contributions, not physically complete trajectories.",
            "Reported speed is averaged with geometric speed by feature extraction; extracted speed is recorded separately.",
            "Heuristic support values are not calibrated probabilities of correctness.",
        ],
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/observation-audit.json"))
    args = parser.parse_args()
    report = audit()
    write_json(args.output, report)
    print(f"Wrote {len(report['scenarios'])} unlabelled synthetic scenarios to {args.output.resolve()}")


if __name__ == "__main__":
    main()
