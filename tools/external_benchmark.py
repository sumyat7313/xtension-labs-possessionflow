"""Independent other-match benchmark; label fields never enter inference.

IDSSE/Sportec data: Bassek et al. 2025, CC BY 4.0,
https://doi.org/10.6084/m9.figshare.28196177 .
Only the two declared OTHER matches are accepted. This does not read the
competition match, its public labels, or human event labels for that match.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.config import load_config
from possession.io import Tracking, sha256, write_json, write_submission
from possession.pipeline import infer, code_digest
from evaluation.possession_metrics import evaluate, label_codes

SOURCE = "https://doi.org/10.6084/m9.figshare.28196177"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
# Splits declared before looking at any benchmark result. Never split adjacent
# frames at random: the entire second match is held out from development.
MATCHES = {
    "J03WOH": {"split": "development", "metadata": (51643478, 11745, "b9710781d400dd812f62b0a6e3d7bce8"),
                "tracking": (51643520, 348945715, "3efff25a86a61768e7d253b4bba137a3")},
    "J03WN1": {"split": "held_out", "metadata": (51643472, 11722, "37b0f85588c0acfb995fe633c7ac2e0d"),
                "tracking": (51643517, 391722803, "bacb2f609c1d1b0bfc906f6b9d4109f2")},
}
PERIODS = {"firstHalf": 1, "secondHalf": 2}


def digest(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_file(directory, match, kind, download=False):
    ident, size, expected = MATCHES[match][kind]
    path = directory / f"{match}_{kind}.xml"
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {path}; run with --download")
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".download")
        print(f"Downloading public {match} {kind}: {size:,} bytes", flush=True)
        with urllib.request.urlopen(f"https://ndownloader.figshare.com/files/{ident}", timeout=60) as r, tmp.open("wb") as f:
            while block := r.read(1024 * 1024):
                f.write(block)
        if tmp.stat().st_size != size or digest(tmp) != expected:
            raise ValueError(f"Dataset checksum mismatch: {tmp}")
        tmp.replace(path)
    if path.stat().st_size != size or digest(path) != expected:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    return path


def iter_frames(path):
    """Yield a frame with its FrameSet context; bounded-memory XML reader."""
    context = None
    for event, node in ET.iterparse(path, events=("start", "end")):
        tag = node.tag.rsplit("}", 1)[-1]
        if event == "start" and tag == "FrameSet":
            context = dict(node.attrib)
        elif event == "end" and tag == "Frame":
            if context is not None:
                yield context, dict(node.attrib)
            node.clear()
        elif event == "end" and tag == "FrameSet":
            context = None
            node.clear()


def read_metadata(path, match):
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    if general is None or general.get("MatchId") != f"DFL-MAT-{match}":
        raise ValueError("External match identity differs from selected match")
    environment = root.find(".//Environment")
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    teams = root.findall(".//Teams/Team")
    by_role = {t.get("Role"): t for t in teams}
    if set(by_role) != {"home", "guest"}:
        raise ValueError("Expected explicit home and guest metadata roles")
    slots, sides, names = {}, [], []
    for side, role in enumerate(("home", "guest")):
        team = by_role[role]
        names.append(team.get("TeamName"))
        for player in team.findall("Players/Player"):
            ident = player.get("PersonId")
            if not ident or ident in slots:
                raise ValueError("Duplicate or absent player metadata identifier")
            slots[ident] = len(sides)
            sides.append(side)
    return pitch, slots, np.asarray(sides, dtype=np.int8), tuple(names)


def read_match(metadata, raw, match):
    """Separate observation adapter and reference labels, with explicit units.

    DFL positions are metres around pitch centre; S is km/h. Only X/Y/S are
    observations. BallStatus and BallPossession populate the scoring array only.
    BallStatus=1 means alive; BallPossession=1 home, 2 away. Unknown codes remain
    ungraded X instead of silently becoming away. See Kloppy Sportec serializer.
    """
    pitch, slots, sides, names = read_metadata(metadata, match)
    ball_rows = {1: {}, 2: {}}
    for context, row in iter_frames(raw):
        if context.get("MatchId") != f"DFL-MAT-{match}":
            raise ValueError("Tracking frame set has a different match identity")
        period = PERIODS.get(context.get("GameSection"))
        if period and context.get("TeamId") == "BALL":
            frame = int(row["N"])
            if frame in ball_rows[period]:
                raise ValueError("Duplicate ball frame")
            ball_rows[period][frame] = row
    ranges, blocks, offset = {}, [], 0
    for period in (1, 2):
        ids = sorted(ball_rows[period])
        if not ids:
            raise ValueError("Both complete half FrameSets are required")
        # Include one X separator between halves for the official temporal scorer.
        ranges[period] = (ids[0], ids[-1], offset)
        blocks.append((period, offset, offset + ids[-1] - ids[0]))
        offset = blocks[-1][2] + 2
        first = ball_rows[period][ids[0]]
        second = ball_rows[period][ids[1]]
        seconds = (datetime.fromisoformat(second["T"]) - datetime.fromisoformat(first["T"])).total_seconds()
        if abs(seconds / (ids[1] - ids[0]) - 1 / 25) > 1e-6:
            raise ValueError("Tracking is not on the required 25 Hz timeline")
    n, count = offset - 1, len(sides)
    players = np.full((n, count, 2), np.nan, dtype=np.float32)
    player_speed = np.full((n, count), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    reference = np.full(n, "X", dtype="<U1")
    label_counts = Counter()

    def xy(row):
        return [float(row.get("X", "nan")) / pitch[0] + .5,
                float(row.get("Y", "nan")) / pitch[1] + .5]

    for period, rows in ball_rows.items():
        start, last, base = ranges[period]
        for fid, row in rows.items():
            i = base + fid - start
            ball[i] = xy(row)
            ball_speed[i] = float(row.get("S", "nan")) / 3.6
            status, possession = row.get("BallStatus"), row.get("BallPossession")
            label_counts[(status, possession)] += 1
            if status == "0":
                reference[i] = "D"
            elif status == "1" and possession in ("1", "2"):
                reference[i] = "H" if possession == "1" else "A"
    # Drop dictionaries holding label-bearing XML attributes before inference.
    del ball_rows
    for context, row in iter_frames(raw):
        period = PERIODS.get(context.get("GameSection"))
        slot = slots.get(context.get("PersonId"))
        if period in ranges and slot is not None:
            first, last, base = ranges[period]
            fid = int(row["N"])
            if first <= fid <= last:
                i = base + fid - first
                if np.isfinite(players[i, slot]).any():
                    raise ValueError("Duplicate player observation")
                players[i, slot] = xy(row)
                player_speed[i, slot] = float(row.get("S", "nan")) / 3.6
    tracking = Tracking(players, player_speed, ball, ball_speed, sides, names)
    metadata = {"frame_ranges": {str(k): list(v) for k, v in ranges.items()},
                "half_blocks": blocks, "fps": 25, "pitch_metres": pitch,
                "reference_code_counts": {f"status={s},possession={p}": c for (s,p),c in sorted(label_counts.items(), key=lambda kv: str(kv[0]))},
                "unknown_reference_frames": int(np.sum(reference == "X")) - 1,
                "unit_conversion": "centre-origin metres to normalized; S km/h divided by 3.6"}
    return tracking, reference, blocks, pitch, metadata


def self_test():
    """Synthetic schema/unit/label-isolation check, not match-specific tuning."""
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        meta = d / "meta.xml"
        meta.write_text('<root><General MatchId="DFL-MAT-J03WOH"/><Environment PitchX="100" PitchY="60"/><Teams><Team Role="home" TeamName="Home"><Players><Player PersonId="h"/></Players></Team><Team Role="guest" TeamName="Away"><Players><Player PersonId="a"/></Players></Team></Teams></root>')
        def xml(status, owner):
            parts = ['<root>']
            for half in PERIODS:
                for obj, team in [('h','HOME'),('a','AWAY'),('ball','BALL')]:
                    parts.append(f'<FrameSet MatchId="DFL-MAT-J03WOH" GameSection="{half}" PersonId="{obj}" TeamId="{team}">')
                    for i in range(2):
                        parts.append(f'<Frame N="{100+i}" T="2022-01-01T00:00:00.{i*40:03d}+00:00" X="10" Y="-6" S="36" BallStatus="{status}" BallPossession="{owner}"/>')
                    parts.append('</FrameSet>')
            return ''.join(parts)+'</root>'
        raw=d/'raw.xml'; raw.write_text(xml('1','1'))
        t,g,b,p,m=read_match(meta,raw,'J03WOH')
        assert b==[(1,0,1),(2,3,4)] and list(g)==['H','H','X','H','H']
        assert np.allclose(t.ball[0],[.6,.4]) and t.ball_speed[0]==10
        assert t.player_speed[0,0]==10 and np.allclose(t.players[0,0],[.6,.4])
        raw.write_text(xml('0','2')); t2,g2,*_=read_match(meta,raw,'J03WOH')
        for key in ('players','player_speed','ball','ball_speed','teams'):
            assert np.array_equal(getattr(t,key),getattr(t2,key),equal_nan=True),key
        assert list(g2)==['D','D','X','D','D']
        raw.write_text(xml('1','0')); _,g3,*_=read_match(meta,raw,'J03WOH')
        assert np.all(g3=='X')
    print('PASS: coordinate/speed conversion, separate halves, label isolation, unknown labels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/external/idsse')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/external-benchmark')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/default.json')
    parser.add_argument('--matches', nargs='+', choices=tuple(MATCHES), default=list(MATCHES))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'protocol.json').exists():
        raise ValueError('Choose a new output folder; benchmark records are immutable')
    frozen_code = code_digest()
    protocol = {'dataset': SOURCE, 'license': LICENSE,
                'attribution': 'Bassek, Rein, Weber and Memmert (2025), An integrated dataset of spatiotemporal and event data in elite soccer',
                'splits': {k: v['split'] for k,v in MATCHES.items()}, 'selected_matches': args.matches,
                'config': config.as_dict(), 'config_sha256': sha256(args.config),
                'inference_code_sha256': frozen_code, 'adapter_sha256': sha256(__file__),
                'official_metrics_sha256': sha256(ROOT/'evaluation/possession_metrics.py'),
                'model_selection': 'none; fixed pre-existing configuration',
                'features': ['player X,Y,S','ball X,Y,S','team membership','pitch dimensions','half metadata'],
                'reference_only': ['BallStatus','BallPossession'],
                'tagging': 'disabled; no automatic Smart Tagging counterpart in external data',
                'evaluation': 'whole halves; official spellF1 and macroF1; no public competition labels',
                'limitations': ['Different provider and corrected tracking quality', 'Possession semantics may differ',
                                'Two matches cannot establish challenge ranking', 'Held-out match ceases to be untouched after this report']}
    write_json(args.out/'protocol.json', protocol)
    summary=[]
    for match in args.matches:
        meta=get_file(args.data_dir,match,'metadata',args.download)
        raw=get_file(args.data_dir,match,'tracking',args.download)
        print(f'Parsing {match} locally with separate reference labels...',flush=True)
        tracking, reference, blocks, pitch, alignment=read_match(meta,raw,match)
        print(f'Inferring both complete halves of {match} with fixed configuration...',flush=True)
        labels, _, _, _, warnings = infer(tracking, [], blocks, pitch, 25, config)
        if code_digest()!=frozen_code:
            raise ValueError('Inference source changed during benchmark; repeat from frozen source')
        destination=args.out/match;destination.mkdir()
        write_submission(destination/'predictions.csv',labels)
        # Reference is persisted only in gitignored local run folder, never features.
        write_submission(destination/'reference.csv',reference)
        result=evaluate(label_codes(reference),label_codes(labels),bootstrap_n=1000)
        write_json(destination/'metrics.json',result)
        write_json(destination/'alignment.json',alignment)
        write_json(destination/'manifest.json',{'match':match,'split':MATCHES[match]['split'],
                   'source':SOURCE,'license':LICENSE,'protocol_sha256':sha256(args.out/'protocol.json'),
                   'inputs':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in (meta,raw)},
                   'prediction_sha256':sha256(destination/'predictions.csv'),'warnings':warnings})
        row={'match':match,'split':MATCHES[match]['split'],'frames':len(labels),
             'spellF1':result['spell_f1_mean'],'macroF1':result['macro_f1'],
             'spellF1_bootstrap_5_95':[result['spell_bootstrap_low'],result['spell_bootstrap_high']]}
        summary.append(row); print(json.dumps(row),flush=True)
        del tracking, reference, labels
    write_json(args.out/'summary.json',{'runs':summary,'model_selection_performed':False,'challenge_ranking_estimate':False})


if __name__ == '__main__':
    main()
