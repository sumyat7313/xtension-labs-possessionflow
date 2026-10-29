"""Synthetic engineering fixture. Never a proxy for measured match performance."""

from pathlib import Path
import math
import numpy as np

from games import GAME, GAMES, half_video_blocks
from .io import write_json


def create_fixture(destination):
    root = Path(destination)
    marker = root / ".synthetic-fixture.json"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError("Refusing to put synthetic files in an existing data directory")
    tactical = root / "tactical"
    tactical.mkdir(parents=True, exist_ok=True)
    (tactical / "BAR-ATH_tactical_atd.xml").write_text(
        '<EPTS><Teams><Team id="h"><Name>Barcelona</Name></Team>'
        '<Team id="a"><Name>Athletic Club</Name></Team></Teams><Players>'
        '<Player id="1" teamId="h"/><Player id="2" teamId="h"/>'
        '<Player id="3" teamId="a"/><Player id="4" teamId="a"/>'
        '</Players></EPTS>\n', encoding="utf-8")
    n, fps = GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]
    truth = np.full(n, "X", dtype="<U1")
    events = []
    for _, first, last in half_video_blocks(GAME):
        for start in range(first, last + 1, 1000):
            end = min(last + 1, start + 1000)
            for lo, hi, label in [(0, 400, "H"), (400, 500, "D"), (500, 900, "A"), (900, 1000, "D")]:
                a, b = start + lo, min(end, start + hi)
                if b <= a:
                    continue
                truth[a:b] = label
                if label in "HA":
                    side = "Barcelona" if label == "H" else "Athletic Club"
                    events.append((a, b, "BUILD UP", side))
    with (tactical / "BAR-ATH_tactical_atd.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for frame in range(n):
            label = truth[frame]
            y = 0.5 + 0.025 * math.sin(frame / 25)
            players = f"0.30,{y:.6f},2;0.20,0.3,2;0.70,{y:.6f},2;0.8,0.7,2"
            if label == "X":
                players = ";".join(["NaN,NaN,NaN"] * 4)
                ball = "NaN,NaN,NaN"
            elif label == "D":
                ball = "0.5,1.03,0"
            else:
                x = 0.302 if label == "H" else 0.698
                ball = f"{x},{y:.6f},2"
                # Brief gaps and a short deflection test continuity without real labels.
                if frame % 73 < 6:
                    ball = "NaN,NaN,NaN"
                elif frame % 139 < 2:
                    ball = f"{1-x},{y:.6f},30"
            handle.write(f"{frame + 1}:{players}:{ball}\n")
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    document = Element("file")
    collection = SubElement(document, "ALL_INSTANCES")
    for a, b, code, team in events:
        event = SubElement(collection, "instance")
        SubElement(event, "start").text = str(a / fps)
        SubElement(event, "end").text = str(b / fps)
        SubElement(event, "code").text = code
        label = SubElement(event, "label")
        SubElement(label, "group").text = "Team"
        SubElement(label, "text").text = team
    ElementTree(document).write(tactical / "BAR-ATH_tactical_smart_tagging.xml", encoding="utf-8", xml_declaration=True)
    write_json(marker, {"synthetic": True, "purpose": "Engineering tests only; unrelated to the match", "frames": n})
    return root
