"""Training-only aggregate diagnostics for false stoppage intervals.

This utility cannot select a development/final fixture. It records no raw labels
in its report and makes no correction or deployment decision.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from possession.learned import load_model
from tools import pff_campaign, pff_training


def boundary_regions(y, blocks, radius=50):
    """Known live frames within radius of a direct live/dead transition."""
    y = np.asarray(y)
    before = np.zeros(len(y), bool)
    after = before.copy()
    for _, a, b in blocks:
        for i in np.flatnonzero((y[a:b] != y[a + 1:b + 1])) + a + 1:
            if y[i] == 'D' and y[i - 1] in 'HA':
                lo = i - 1
                while lo >= a and i - lo <= radius and y[lo] in 'HA':
                    before[lo] = True
                    lo -= 1
            if y[i - 1] == 'D' and y[i] in 'HA':
                hi = i
                while hi <= b and hi - i < radius and y[hi] in 'HA':
                    after[hi] = True
                    hi += 1
    return before, after


def segments(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def summarize(y, baseline, labels, probabilities, z, blocks):
    y, baseline, labels = map(np.asarray, (y, baseline, labels))
    if y.shape != baseline.shape or y.shape != labels.shape or probabilities.shape != (len(y), 3):
        raise ValueError('Mismatched diagnostic inputs')
    if not np.isin(y, list('HADX')).all():
        raise ValueError('Invalid reference labels')
    live = np.isin(y, list('HA'))
    dead = y == 'D'
    fp = live & (labels == 'D')
    added = fp & (baseline != 'D')
    removed = live & (baseline == 'D') & (labels != 'D')
    recovered = dead & (baseline != 'D') & (labels == 'D')
    before, after = boundary_regions(y, blocks)
    conditions = {
        'raw_ball_observed': z[:, 0] > .5,
        'ball_missing_support': z[:, 18] > .5,
        'outside_support': z[:, 6] > .5,
        'moving_ball_observed': (z[:, 0] > .5) & (z[:, 3] > .5) & (z[:, 2] >= np.log(3) / np.log(46)),
        'observed_contact_1p5m': (z[:, 0] > .5) & (z[:, 5] > .5) & (z[:, 4] < 1.5 / 20),
        'players_median_above_2mps': z[:, 11] > 2 / 13,
        'pre_onset_within_2s': before,
        'post_restart_within_2s': after,
        'far_from_direct_boundary': ~(before | after),
    }
    result = {'known_live': int(live.sum()), 'known_dead': int(dead.sum()),
              'baseline_false_dead': int((live & (baseline == 'D')).sum()),
              'candidate_false_dead': int(fp.sum()), 'removed_false_dead': int(removed.sum()),
              'candidate_true_dead': int((dead & (labels == 'D')).sum()),
              'baseline_true_dead': int((dead & (baseline == 'D')).sum())}
    for name, mask in [('added_false_dead', added), ('recovered_true_dead', recovered), ('all_false_dead', fp)]:
        result[name] = {'frames': int(mask.sum()), 'conditions': {k: int((mask & v).sum()) for k, v in conditions.items()},
                        'dead_probability_below': {str(v): int((mask & (probabilities[:, 2] < v)).sum()) for v in (.5, .6, .7, .8, .9, .95)}}
    duration = {'under_0p5s': 0, '0p5_to_2s': 0, '2_to_5s': 0, 'at_least_5s': 0}
    isolated = overlaps = 0
    for _, a, b in blocks:
        for lo, hi in segments(fp[a:b + 1]):
            length = int(hi - lo)
            key = 'under_0p5s' if length < 12.5 else '0p5_to_2s' if length < 50 else '2_to_5s' if length < 125 else 'at_least_5s'
            duration[key] += length
        for lo, hi in segments(labels[a:b + 1] == 'D'):
            if fp[a + lo:a + hi].any():
                if dead[a + lo:a + hi].any():
                    overlaps += 1
                else:
                    isolated += 1
    result['false_dead_frames_by_contiguous_error_duration'] = duration
    result['predicted_dead_segments_with_false_frames'] = {'overlap_reference_dead': overlaps, 'no_reference_dead_overlap': isolated}
    return result


def add_tree(total, value):
    for key, item in value.items():
        if isinstance(item, dict):
            add_tree(total.setdefault(key, {}), item)
        else:
            total[key] = total.get(key, 0) + item


def run(run_path, output):
    protocol = pff_campaign.protocol()
    model = load_model(run_path / 'candidate.json')
    records = []
    total = {}
    for mid in protocol['split']['train']:
        if mid in protocol['split']['development'] or mid in protocol['split']['final']:
            raise ValueError('Training partition is not exclusive')
        item = pff_campaign.item_at(run_path, mid)
        labels, probabilities = pff_training.predict_cache(item, model)
        y, _ = pff_campaign.reference(run_path, mid)
        baseline = np.load(run_path / 'cache' / str(mid) / 'baseline.npy', allow_pickle=False)
        summary = summarize(y, baseline, labels, probabilities, item['z'], item['blocks'])
        add_tree(total, summary)
        records.append({'match_id': mid, **summary})
        report = {'schema_version': 1, 'split': 'train', 'held_out_references_read': False,
                  'reference_quality': 'provider event-derived silver; not independent dense truth',
                  'candidate_sha256': sha256(run_path / 'candidate.json'),
                  'protocol_sha256': protocol['protocol_sha256'], 'diagnostic_source_sha256': sha256(Path(__file__)),
                  'completed_matches': len(records), 'expected_matches': len(protocol['split']['train']),
                  'total': total, 'records': records}
        write_json(output, report)
        print(json.dumps({'match': mid, 'completed': len(records), 'false_dead': summary['candidate_false_dead'], 'added_false_dead': summary['added_false_dead']['frames']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/pff-campaign-v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pff-precision-v4/training-diagnostics.json')
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
