"""Metadata-only PFF reservation and fail-closed paired promotion checks.

No provider file, label or production model is read by this module.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

BASELINE_SHA256 = '3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0'
SEED = 'xtension-pff-v1-2026-11-01'
EXPECTED_IDS = tuple(range(3812, 3860)) + tuple(range(10502, 10518))
METRICS = ('spell', 'macro', 'short_precision', 'short_recall', 'dead_segment',
           'dead_boundary', 'dead_precision', 'dead_recall')
EVIDENCE = ('dense_truth_validated', 'independent_observations', 'identity_checked',
            'usage_permitted', 'development_passed', 'observations_only',
            'predictions_frozen_before_references', 'adequate_label_support')
POLICY = {
    'minimum_spell_gain': .005, 'minimum_targeted_gain': .01,
    'minimum_macro_gain': -.001, 'maximum_cell_spell_drop': .01,
    'maximum_cell_dead_recall_drop': .005, 'minimum_test_matches': 12,
    'minimum_reference_dead_segments': 100, 'minimum_reference_short_spells': 100,
    'bootstrap_seed': 20261101, 'bootstrap_resamples': 20000,
    'bootstrap_quantiles': [.05, .95],
    'reference_requirement': 'validated dense H/A/D with uncertainty gaps preserved',
    'inference_conditions': ['native'],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def build_protocol(metadata_rows, metadata_hashes, identity_audit_sha256):
    """Reserve all64 fixtures using only six whitelisted metadata fields.

    Caller must validate the provenance of metadata and audit declarations.
    Returned order is independent of input iteration order and label values.
    """
    fields = {'match_id', 'date', 'home_team', 'away_team', 'stage', 'competition'}
    rows = []
    for raw in metadata_rows:
        if set(raw) != fields:
            raise ValueError('Metadata reservation accepts exactly six identity fields')
        row = dict(raw)
        if type(row['match_id']) is not int or row['match_id'] not in EXPECTED_IDS:
            raise ValueError('Unexpected PFF fixture ID')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in fields - {'match_id'}):
            raise ValueError('Missing identity field')
        if row['competition'] != 'FIFA World Cup 2022':
            raise ValueError('Unexpected competition')
        day = date.fromisoformat(row['date'])
        if not date(2022, 11, 20) <= day <= date(2022, 12, 18):
            raise ValueError('Fixture is outside the tournament')
        expected_stage = 'group' if row['match_id'] < 10000 else 'knockout'
        if row['stage'] != expected_stage or row['home_team'].casefold() == row['away_team'].casefold():
            raise ValueError('Invalid stage or teams')
        rows.append(row)
    ids = [r['match_id'] for r in rows]
    if len(ids) != 64 or set(ids) != set(EXPECTED_IDS):
        raise ValueError('Reservation requires all64 unique fixtures; revise protocol explicitly if incomplete')
    keys = [(r['date'], *sorted([r['home_team'].strip().casefold(), r['away_team'].strip().casefold()])) for r in rows]
    if len(set(keys)) != 64:
        raise ValueError('Duplicate underlying fixture under different provider IDs')
    hashes = {str(k): v for k, v in metadata_hashes.items()}
    if set(hashes) != {str(i) for i in EXPECTED_IDS} or not all(_sha(v) for v in hashes.values()):
        raise ValueError('Every metadata file requires an authenticated SHA256')
    if not _sha(identity_audit_sha256):
        raise ValueError('Identity audit SHA256 required')
    order = lambda i: hashlib.sha256(f'{SEED}:{i}'.encode()).hexdigest()
    groups = sorted(range(3812, 3860), key=order)
    knockout = sorted(range(10502, 10518), key=order)
    split = {'train': sorted(groups[18:] + knockout[6:]),
             'development': sorted(groups[:9] + knockout[:3]),
             'final': sorted(groups[9:18] + knockout[3:6])}
    teams = {part: sorted({r[k] for r in rows if r['match_id'] in selected
                          for k in ('home_team', 'away_team')}) for part, selected in split.items()}
    result = {'schema_version': 1, 'seed': SEED, 'split': split, 'pilot_train_match': 3812,
              'metadata': sorted(rows, key=lambda r: r['match_id']), 'metadata_sha256': hashes,
              'identity_audit_sha256': identity_audit_sha256, 'baseline_sha256': BASELINE_SHA256,
              'policy': POLICY, 'conditions': ['native'], 'teams': teams,
              'team_overlap_train_final': sorted(set(teams['train']) & set(teams['final'])),
              'labels_used_for_split': False, 'freshness_unit': 'whole fixture including extra time'}
    return dict(result, protocol_sha256=digest(result))


def verify_protocol(protocol):
    rebuilt = build_protocol(protocol['metadata'], protocol['metadata_sha256'], protocol['identity_audit_sha256'])
    if protocol != rebuilt:
        raise ValueError('PFF reservation differs from fixed metadata-only procedure')
    return protocol


def freeze_protocol(path, protocol):
    """Idempotent same-byte reservation; a changed split cannot overwrite it."""
    verify_protocol(protocol)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n'
    try:
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_text(encoding='utf-8') != content:
            raise ValueError('Existing PFF reservation may not be replaced') from None


def _comparison(records, protocol, partition):
    verify_protocol(protocol)
    rows = {}
    for row in records:
        mid = row['match_id']
        if type(mid) is not int or mid in rows or mid not in protocol['split'][partition]:
            raise ValueError('Duplicate or wrong-partition PFF score')
        if row.get('condition') != 'native' or row.get('baseline_sha256') != BASELINE_SHA256:
            raise ValueError('Unexpected comparison condition or baseline')
        if not _sha(row.get('candidate_sha256')):
            raise ValueError('Candidate artifact identity required')
        for model in ('baseline', 'candidate'):
            values = row[model]
            if set(values) != set(METRICS) or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
                raise ValueError('Every metric must be defined and bounded')
        for name in ('reference_dead_segments', 'reference_short_spells'):
            if type(row.get(name)) is not int or row[name] < 1:
                raise ValueError('Each test fixture requires explicit nonzero reference support')
        rows[mid] = row
    if set(rows) != set(protocol['split'][partition]):
        raise ValueError('All reserved test fixtures must be scored; no selective omission')
    if len({r['candidate_sha256'] for r in rows.values()}) != 1:
        raise ValueError('Only one frozen candidate may be evaluated')
    differences = {k: np.array([rows[m]['candidate'][k] - rows[m]['baseline'][k] for m in sorted(rows)]) for k in METRICS}
    means = {k: float(v.mean()) for k, v in differences.items()}
    rng = np.random.default_rng(POLICY['bootstrap_seed'])
    indices = rng.integers(0, len(rows), size=(POLICY['bootstrap_resamples'], len(rows)))
    lower, upper = np.quantile(differences['spell'][indices].mean(axis=1), POLICY['bootstrap_quantiles'])
    support = {k: sum(r[k] for r in rows.values()) for k in ('reference_dead_segments', 'reference_short_spells')}
    checks = {
        'spell_gain': means['spell'] >= .005,
        'no_target_regression': all(means[k] >= 0 for k in ('short_precision', 'short_recall', 'dead_segment', 'dead_boundary', 'dead_precision')),
        'targeted_gain': max(means['short_precision'], means['dead_boundary']) >= .01,
        'macro_guard': means['macro'] >= -.001,
        'worst_fixture_spell': float(differences['spell'].min()) >= -.01,
        'worst_fixture_dead_recall': float(differences['dead_recall'].min()) >= -.005,
        'positive_bootstrap_lower': bool(lower > 0),
        'match_support': len(rows) >= POLICY['minimum_test_matches'],
        'event_support': support['reference_dead_segments'] >= POLICY['minimum_reference_dead_segments'] and support['reference_short_spells'] >= POLICY['minimum_reference_short_spells'],
    }
    return {'numeric_gate_passed': all(checks.values()), 'checks': checks, 'mean_gains': means,
            'paired_match_bootstrap_5_95': [float(lower), float(upper)],
            'reference_support': support, 'n_matches': len(rows),
            'candidate_sha256': next(iter(rows.values()))['candidate_sha256'],
            'baseline_sha256': BASELINE_SHA256, 'protocol_sha256': protocol['protocol_sha256']}


def development_gate(records, protocol):
    """Same numeric requirements on exactly12 development fixtures; never promote."""
    result = _comparison(records, protocol, 'development')
    return dict(result, eligible_for_final_audit=result['numeric_gate_passed'],
                promoted=False, reference_scope='development; provenance must be assessed separately')


def promotion_gate(records, protocol, evidence):
    """Compare one frozen candidate on all final fixtures and require evidence.

    Sparse metrics or inferred dense truth cannot authorize promotion. Evidence
    flags are declarations that the caller must independently substantiate.
    """
    result = _comparison(records, protocol, 'final')
    result['checks']['evidence'] = all(evidence.get(k) is True for k in EVIDENCE)
    return dict(result, promoted=all(result['checks'].values()))
