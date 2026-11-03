"""Raw-visible PFF adapter; event-derived smoothing never enters observations."""
from __future__ import annotations

import bz2
import json
import math
import struct
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from possession.io import Tracking, sha256

TARGET_FPS = 25
_PLAYER = struct.Struct('<qqdd')
OBSERVATION_KEYS = ('gameRefId', 'frameNum', 'period', 'videoTimeMs',
                    'homePlayers', 'awayPlayers', 'balls')
POLICY = {
    'fps': TARGET_FPS, 'geometry': 'raw-centred-metres',
    'visibility': 'VISIBLE only, for players and balls',
    'ball_smoothing': 'forbidden: derived from event-labelled carrier',
    'resampling': 'nearest native observation within half a native frame; no bridging missing frames',
    'duplicate_frames': 'collapse only identical whitelisted observations',
    'reference_quality': 'provider event intervals; not independently reviewed human gold',
}


def _pack_players(players):
    """Lossless compact transient records; retained coordinates stay float64.

    Signed zero is numerically identical in the prior tuple representation and
    normalization, so canonicalize it to preserve duplicate-frame equality.
    """
    return b''.join(_PLAYER.pack(side, jersey, x + 0., y + 0.)
                    for side, jersey, x, y in players)


def _players(record):
    return _PLAYER.iter_unpack(record['players'])


def _integer(value, name):
    if isinstance(value, str) and value and value.lstrip('-').isdigit() and str(int(value)) == value:
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError('Invalid '+name)
    return int(value)


def _xy(point):
    if not isinstance(point, dict) or point.get('visibility') != 'VISIBLE':
        return None
    x, y = point.get('x'), point.get('y')
    if x is None or y is None:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (x, y)):
        raise ValueError('Invalid raw coordinate')
    return (float(x), float(y))


def project_observation(frame):
    """Explicit whitelist discards event objects and all Smoothed fields."""
    period = frame.get('period')
    if period is None:
        return None
    period = _integer(period, 'period')
    if period not in (1, 2, 3, 4):
        return None
    number = _integer(frame.get('frameNum'), 'frameNum')
    if number < 0:
        raise ValueError('Negative frame index')
    players = []
    for side, key in enumerate(('homePlayers', 'awayPlayers')):
        seen = set()
        values = frame.get(key) or []
        if not isinstance(values, list):
            raise ValueError('Raw player field must be a list')
        for point in values:
            xy = _xy(point)
            if xy is None:
                continue
            jersey = _integer(point.get('jerseyNum'), 'jerseyNum')
            if jersey < 0:
                continue
            if jersey in seen:
                raise ValueError('Repeated visible jersey in one team/frame')
            seen.add(jersey)
            players.append((side, jersey, *xy))
    balls = frame.get('balls') or []
    if not isinstance(balls, list):
        raise ValueError('Raw balls field must be a list')
    visible = [p for ball in balls if (p := _xy(ball)) is not None]
    # Multiple detections cannot be disambiguated using event-labelled carrier.
    ball = visible[0] if len(visible) == 1 else None
    milliseconds = frame.get('videoTimeMs')
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)) or not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError('Invalid videoTimeMs')
    game = frame.get('gameRefId')
    return {'period': period, 'frame': number, 'video_seconds': milliseconds / 1000., 'players': sorted(players), 'ball': ball,
            'game_id': None if game is None else _integer(game, 'gameRefId')}


def _metadata(document):
    if isinstance(document, list):
        if len(document) != 1:
            raise ValueError('Expected exactly one match metadata record')
        document = document[0]
    if not isinstance(document, dict):
        raise ValueError('Invalid match metadata')
    fps = float(document['fps'])
    if not np.isfinite(fps) or not 20 <= fps <= 60:
        raise ValueError('Unexpected native FPS')
    pitch = document['stadium']['pitches'][0]
    dimensions = (float(pitch['length']), float(pitch['width']))
    if not 90 <= dimensions[0] <= 120 or not 45 <= dimensions[1] <= 90:
        raise ValueError('Invalid pitch dimensions')
    ident = _integer(document['id'], 'metadata game id')
    home, away = document['homeTeam'], document['awayTeam']
    if home['id'] == away['id']:
        raise ValueError('Match needs two distinct teams')
    return document, ident, fps, dimensions, (str(home['name']), str(away['name']))


def observations(metadata, frames):
    """Return Tracking, blocks, pitch, alignment; no labels are reconstructed."""
    meta, ident, native_fps, pitch, names = _metadata(metadata)
    records = {}; duplicates = 0; previous = -1; max_clock_error = 0.; header_seen = False
    for raw in frames:
        frame = project_observation(raw)
        if frame is None:
            continue
        number = frame['frame']
        declared = frame.pop('game_id')
        if (declared is not None and declared != ident) or number < previous:
            raise ValueError('Mismatched match or unsorted native frames')
        header_seen |= declared == ident
        frame['players'] = _pack_players(frame['players'])
        previous = number
        clock_error = abs(frame['video_seconds'] - number / native_fps)
        max_clock_error = max(max_clock_error, clock_error)
        if clock_error > 1. / native_fps:
            raise ValueError('Native frame and recorded video clocks disagree')
        if number in records:
            if records[number] != frame:
                raise ValueError('Duplicated frame has conflicting raw geometry')
            duplicates += 1
        else:
            records[number] = frame
    if not records or not header_seen:
        raise ValueError('No valid tracking periods or matching file identity')
    periods = sorted({r['period'] for r in records.values()})
    slots = sorted({(v[0], v[1]) for r in records.values() for v in _players(r)})
    if not slots or {s[0] for s in slots} != {0, 1}:
        raise ValueError('Both teams need visible observations')
    lookup = {slot: i for i, slot in enumerate(slots)}
    parts = []; ballparts = []; source = []; timestamps = []; periodids = []; blocks = []; offset = 0
    seen_intervals = []
    for period in periods:
        if parts:
            parts.append(np.full((1, len(slots), 2), np.nan, dtype=np.float32))
            ballparts.append(np.full((1, 2), np.nan, dtype=np.float32))
            source.append(-1); timestamps.append(None); periodids.append(0); offset += 1
        keys = np.array([n for n, r in records.items() if r['period'] == period], dtype=np.int64)
        if np.any(np.diff(keys) <= 0):
            raise ValueError('Period native frames must increase')
        if seen_intervals and keys[0] <= seen_intervals[-1][1]:
            raise ValueError('Periods overlap or interleave')
        seen_intervals.append((int(keys[0]), int(keys[-1])))
        times = np.array([records[int(k)]['video_seconds'] for k in keys])
        if np.any(np.diff(times) <= 0):
            raise ValueError('Recorded video clock must strictly increase')
        # Both raw observations and current event times use the recorded video clock.
        grid = np.arange(np.ceil(times[0] * TARGET_FPS), np.floor(times[-1] * TARGET_FPS) + 1) / TARGET_FPS
        if not len(grid) or len(grid) > TARGET_FPS * 80 * 60:
            raise ValueError('Invalid period grid length')
        right = np.searchsorted(times, grid); right = np.minimum(right, len(times) - 1)
        left = np.maximum(right - 1, 0)
        chosen = np.where(np.abs(times[left] - grid) <= np.abs(times[right] - grid), left, right)
        valid = np.abs(times[chosen] - grid) <= (.5 / native_fps + 1e-8)
        native = np.where(valid, keys[chosen], -1)
        players = np.full((len(grid), len(slots), 2), np.nan, dtype=np.float32)
        ball = np.full((len(grid), 2), np.nan, dtype=np.float32)
        for i in np.flatnonzero(valid):
            r = records[int(native[i])]
            for side, jersey, x, y in _players(r):
                players[i, lookup[side, jersey]] = (x / pitch[0] + .5, .5 - y / pitch[1])
            if r['ball'] is not None:
                x, y = r['ball']; ball[i] = (x / pitch[0] + .5, .5 - y / pitch[1])
        parts.append(players); ballparts.append(ball); source.extend(native.tolist()); timestamps.extend(grid.tolist())
        periodids.extend([period] * len(grid)); blocks.append((period, offset, offset + len(grid) - 1)); offset += len(grid)
    players, ball = np.concatenate(parts), np.concatenate(ballparts)
    player_speed = np.full(players.shape[:2], np.nan, dtype=np.float32)
    ball_speed = np.full(len(ball), np.nan, dtype=np.float32)
    for _, a, b in blocks:
        player_speed[a + 1:b + 1] = np.linalg.norm(np.diff(players[a:b + 1], axis=0) * pitch, axis=2) * TARGET_FPS
        ball_speed[a + 1:b + 1] = np.linalg.norm(np.diff(ball[a:b + 1], axis=0) * pitch, axis=1) * TARGET_FPS
    tracking = Tracking(players, player_speed, ball, ball_speed, np.array([s[0] for s in slots]), names)
    alignment = {'match': str(ident), 'fps': TARGET_FPS, 'native_fps': native_fps,
                 'source_frames': source, 'video_seconds': timestamps, 'periods': periodids,
                 'policy': POLICY, 'duplicates_collapsed': duplicates,
                 'missing_source_grid_frames': int(sum(n < 0 and period > 0 for n, period in zip(source, periodids))),
                 'synthetic_period_separators': len(periods)-1,
                 'visible_ball_fraction': float(np.isfinite(ball).all(axis=1).mean()),
                 'maximum_native_video_clock_error_seconds': max_clock_error,
                 'reference_fields_used_for_inference': False}
    return tracking, blocks, pitch, alignment


def _rows(path):
    with bz2.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_observations(metadata_path, tracking_path):
    result = observations(json.loads(Path(metadata_path).read_text(encoding='utf-8-sig')), _rows(tracking_path))
    result[-1]['source_sha256'] = {'metadata': sha256(metadata_path), 'tracking': sha256(tracking_path)}
    return result


def references(events, alignment):
    """Reconstruct explicit provider silver states; never passed to inference.

    Current event-file time fields are video seconds, not match-clock seconds.
    Missing-video/initial-non-event annotations reset certainty; absent events alone do
    not imply dead play. Possession is retained between explicit control events.
    """
    if not isinstance(events, list) or alignment.get('reference_fields_used_for_inference') is not False:
        raise ValueError('Expected event list and observation-only alignment')
    n = len(alignment['source_frames'])
    if len(alignment['video_seconds']) != n or len(alignment['periods']) != n:
        raise ValueError('Invalid reference timeline')
    by_id = {}; duplicate_events = 0
    for event in events:
        if _integer(event['gameId'], 'event game id') != int(alignment['match']):
            raise ValueError('Reference belongs to another match')
        game = event['gameEvents']
        if not isinstance(game, dict):
            raise ValueError('Missing game event definition')
        period = _integer(game['period'], 'event period')
        if period not in (1, 2, 3, 4):
            continue
        start = event.get('startTime')
        if isinstance(start, bool) or not isinstance(start, (float, int)) or not math.isfinite(start) or start < 0:
            raise ValueError('Invalid event video start time')
        end = event.get('endTime')
        if end is not None and (isinstance(end, bool) or not isinstance(end, (float, int)) or not math.isfinite(end) or end < start):
            raise ValueError('Invalid event video end time')
        owner = game.get('homeTeam')
        if owner is not None and type(owner) is not bool:
            raise ValueError('Event ownership must be Boolean or unknown')
        uncertain = False
        # Subordinate possessionEvents (including their nonEvent flags) are never
        # used as ownership supervision; only explicit game-event states are used.
        for key in ('videoMissing', 'initialNonEvent'):
            value = game.get(key, False)
            # Native training files contain explicit nulls on some FOUL rows.
            # Null does not establish a false flag: reset reference certainty
            # until another explicit known state, even for an ignored event.
            if value is None:
                uncertain = True
            elif type(value) is not bool:
                raise ValueError('Invalid event uncertainty flag')
            else:
                uncertain |= value
        kind = game.get('gameEventType')
        if not isinstance(kind, str):
            raise ValueError('Invalid game event kind')
        kind = kind.replace(' ', '').upper()
        event_id = _integer(event['gameEventId'], 'game event id')
        # Native files reuse an OTB identifier for a distinct later FOUL.
        # Event kind is part of identity; conflicting same-kind copies still fail.
        identity = (event_id, kind)
        value = (period, float(start), None if end is None else float(end), kind, owner, uncertain)
        if identity in by_id:
            if by_id[identity] != value:
                raise ValueError('Conflicting copies of one game event')
            duplicate_events += 1
        else:
            by_id[identity] = value
    states = np.full(n, 'X', dtype='<U1')
    live = {'OTB', 'FIRSTKICKOFF', 'SECONDKICKOFF', 'THIRDKICKOFF', 'FOURTHKICKOFF'}
    ignored = {'G', 'ON', 'OFF', 'SUB', 'FOUL'}
    native = np.asarray(alignment['source_frames']); periods = np.asarray(alignment['periods'])
    conflicts = 0; masked = np.zeros(n, bool)
    for period in sorted(set(periods) - {0}):
        selected = np.flatnonzero(periods == period)
        times = np.array([alignment['video_seconds'][i] for i in selected], dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid within-period reference clock')
        changes = {}
        for ep, start, end, kind, owner, uncertain in by_id.values():
            if ep != period:
                continue
            if uncertain or kind == 'VID':
                label = 'X'
                stop = start if end is None else end
                masked[selected[(times >= start) & (times <= stop)]] = True
            elif kind in live:
                label = 'X' if owner is None else 'H' if owner else 'A'
            elif kind in ('OUT', 'END'):
                label = 'D'
            elif kind in ignored:
                continue
            else:
                # New event semantics cannot silently extend previous certainty.
                label = 'X'
            index = int(np.searchsorted(times, start, side='left'))
            if index == len(times):
                continue
            changes.setdefault(index, set()).add(label)
        state = 'X'
        for j, index in enumerate(selected):
            if j and (native[index] < 0 or native[selected[j-1]] < 0):
                state = 'X'
            if j in changes:
                possible = changes[j]
                if len(possible) != 1:
                    state = 'X'; conflicts += 1
                else:
                    state = next(iter(possible))
            states[index] = state if native[index] >= 0 else 'X'
    states[masked] = 'X'
    counts = {'unique_game_events': len(by_id), 'duplicate_event_rows_collapsed': duplicate_events,
              'reused_game_event_ids': len(by_id) - len({key[0] for key in by_id}),
              'uncertain_game_events': sum(v[5] or v[3] == 'VID' for v in by_id.values()),
              'conflicting_state_grid_frames': conflicts, 'explicit_uncertainty_masked_frames': int(masked.sum()),
              'class_counts': {c: int(np.sum(states == c)) for c in 'HADX'},
              'reference_quality': 'provider_event_derived_silver', 'dense_human_had_truth': False,
              'promotion_eligible_by_labels_alone': False}
    return states, counts


def read_references(event_path, alignment):
    states, counts = references(json.loads(Path(event_path).read_text(encoding='utf-8-sig')), alignment)
    counts['event_sha256'] = sha256(event_path)
    return states, counts
