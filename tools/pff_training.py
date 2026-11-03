"""Isolated PFF joint-head research; never mutates the deployed artifact."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from possession import robust_control, temporal
from possession.config import load_config
from possession.decoder import decode
from possession.features import extract
from possession.io import sha256
from possession.learned import (
    feature_matrix, joint_probabilities, load_model, predict_dead, resolve_config,
    validate_model,
)
from tools.control_diagnostics import training_spell_weights

BASELINE_SHA256 = '3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0'
POLICY = {
    'candidate': 'pff-mixed-joint', 'epochs': 8, 'seed': 20261101,
    'sample_hz': 5, 'upstream_heads': 'frozen-v1.5',
    'decoder': 'frozen-v1.5', 'threshold_search': False,
    'weighting': 'capped-short-spell plus boundary emphasis; equal sampled mass per fixture',
    'reference_quality': 'provider event-derived silver until independently verified',
    'automatic_promotion': False,
}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return np.log(p) - np.log1p(-p)


def _blocks(blocks, n, fps):
    if isinstance(fps, bool) or not np.isfinite(fps) or fps < 5 or not np.isclose(fps / 5, round(fps / 5)):
        raise ValueError('Use a fixed observation-only timeline with FPS divisible by five')
    active = np.zeros(n, bool)
    for block in blocks:
        if len(block) != 3:
            raise ValueError('Invalid period block')
        _, a, b = block
        if type(a) not in (int, np.int64, np.int32) or type(b) not in (int, np.int64, np.int32) or not 0 <= a <= b < n or active[a:b + 1].any():
            raise ValueError('Overlapping or invalid period blocks')
        active[a:b + 1] = True
    if not active.any():
        raise ValueError('No active observation period')
    return active, round(fps / 5)


def baseline_model(path=ROOT / 'models/submission.json'):
    if sha256(path) != BASELINE_SHA256:
        raise ValueError('PFF comparison requires the exact frozen v1.5 model')
    return load_model(path)


def prepare_cache(tracking, blocks, pitch, fps, model=None, config=None):
    """Observation-only cache. There is deliberately no event/reference argument."""
    model = baseline_model() if model is None else validate_model(model)
    config = resolve_config(load_config(ROOT / 'configs/baselines/v1.5.json') if config is None else config, model)
    _blocks(blocks, len(tracking.ball), fps)
    f = extract(tracking, pitch, blocks, fps, config)
    x = robust_control.feature_matrix(tracking, f, blocks, pitch, fps)
    z = feature_matrix(tracking, f, blocks, pitch, fps)
    head = model['robust_control_model']
    raw_home = (x / np.asarray(head['scale'])) @ np.asarray(head['coefficients'])
    home = temporal.predict_home(raw_home, x, blocks, fps, model['temporal_model'], model['expert_gate'])
    dead = predict_dead(z, model)
    item = {
        'x': x, 'z': z, 'base_home': home, 'base_dead': dead,
        'blocks': [list(v) for v in blocks], 'fps': float(fps),
        'usable': [bool(f.reliability[a:b + 1].sum() > 0) for _, a, b in blocks],
        'reference_fields_used_for_inference': False,
    }
    labels, probabilities = predict_cache(item, model, config)
    return item, labels, probabilities


def predict_cache(item, model, config=None):
    """Reuse frozen observation features for an exactly paired comparison."""
    validate_model(model)
    config = resolve_config(load_config(ROOT / 'configs/baselines/v1.5.json') if config is None else config, model)
    x, z = np.asarray(item['x']), np.asarray(item['z'])
    blocks, fps = item['blocks'], item['fps']
    _blocks(blocks, len(x), fps)
    if item.get('reference_fields_used_for_inference') is not False:
        raise ValueError('Observation-only cache declaration is required')
    for key in ('base_home', 'base_dead'):
        p = np.asarray(item[key])
        if p.shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid upstream probabilities')
    if len(item['usable']) != len(blocks) or any(type(v) is not bool for v in item['usable']):
        raise ValueError('Each period needs an observation support declaration')
    p = joint_probabilities(item['base_dead'], item['base_home'], x, z, blocks, fps,
                            model['joint_model'], np.zeros(len(x)))
    labels = np.full(len(x), 'X', dtype='<U1')
    for (_, a, b), usable in zip(blocks, item['usable']):
        if usable:
            labels[a:b + 1] = np.array(list('HAD'))[decode(p[a:b + 1], fps, config)]
    return labels, p


def build_sequences(items, *, training_matches, held_out_matches):
    """References enter only here, after whole-fixture membership validation."""
    train, held = set(training_matches), set(held_out_matches)
    if not train or train & held or len(train) != len(training_matches) or len(held) != len(held_out_matches):
        raise ValueError('Training and held-out fixtures must be unique and disjoint')
    seen = set()
    sequences, support = [], {}
    for item in items:
        match = item['match']
        if match not in train or match in held or match in seen:
            raise ValueError('Unexpected, repeated or held-out training fixture')
        seen.add(match)
        x, z, y = np.asarray(item['x']), np.asarray(item['z']), np.asarray(item['y'])
        if x.shape != (len(y), 98) or z.shape != (len(y), 58) or not np.isfinite(x).all() or not np.isfinite(z).all() or not np.isin(y, list('HADX')).all():
            raise ValueError('Invalid training features/references')
        active, stride = _blocks(item['blocks'], len(y), item['fps'])
        if item.get('reference_fields_used_for_inference') is not False:
            raise ValueError('References cannot generate model inputs')
        weights = np.zeros(len(y))
        for _, a, b in item['blocks']:
            local = y[a:b + 1]
            w = training_spell_weights(local, item['fps'])
            w[local == 'D'] = 1.
            radius = round(.4 * item['fps'])
            for j in np.flatnonzero((local[1:] != local[:-1]) & (local[1:] != 'X') & (local[:-1] != 'X')) + 1:
                lo, hi = max(0, j - radius), min(len(w), j + radius + 1)
                w[lo:hi] = np.maximum(w[lo:hi], 1.5)
            w[local == 'X'] = 0
            weights[a:b + 1] = w
        indices = [np.arange(a, b + 1, stride) for _, a, b in item['blocks']]
        mass = sum(weights[index].sum() for index in indices)
        if mass <= 0:
            raise ValueError('Training fixture has no supported sampled labels')
        weights *= 10000. / mass
        for key in ('base_home', 'base_dead'):
            p = np.asarray(item[key])
            if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('Invalid upstream training probabilities')
        support[match] = {'class_counts': {c: int(np.sum(active & (y == c))) for c in 'HADX'},
                          'sampled_weight': float(sum(weights[i].sum() for i in indices))}
        for index in indices:
            sequences.append({'X': x[index], 'Z': z[index], 'y': y[index], 'w': weights[index],
                              'base_home': _logit(item['base_home'])[index],
                              'base_dead': _logit(item['base_dead'])[index]})
    if seen != train:
        raise ValueError('Every frozen training fixture is required')
    return sequences, support


def fit_candidate(items, *, training_matches, held_out_matches, protocol_sha256, baseline=None):
    from tools import train_joint
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64 or any(c not in '0123456789abcdef' for c in protocol_sha256):
        raise ValueError('A frozen protocol SHA-256 is required')
    frozen = baseline_model()
    baseline = frozen if baseline is None else validate_model(baseline)
    if baseline != frozen:
        raise ValueError('PFF training must retain exact v1.5 upstream weights')
    sequences, support = build_sequences(items, training_matches=training_matches, held_out_matches=held_out_matches)
    meta = {'protocol_sha256': protocol_sha256, 'training_matches': sorted(training_matches),
            'held_out_matches': sorted(held_out_matches), 'policy': POLICY, 'support': support,
            'base_model_sha256': BASELINE_SHA256, 'reference_fields_used_for_inference': False}
    head = train_joint.fit(sequences, seed=POLICY['seed'], epochs=POLICY['epochs'], metadata=meta)
    candidate = copy.deepcopy(baseline)
    candidate['joint_model'] = head
    candidate['metadata'] = dict(candidate.get('metadata', {}), pff_research=meta)
    return validate_model(candidate)
