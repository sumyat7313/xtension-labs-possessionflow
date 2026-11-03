"""Fixed development selection and guarded final audit for PFF precision.

These checks consume aggregate records, never provider files. They cannot
certify human review, usage rights or the provenance of caller declarations.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from tools import pff_validation as original

PARENT_SHA256 = '0ab133743c593466d8ae92ea94937e8c6bf22510f6b64053e3e69c7b467dcf3f'
PENALTIES = (0., .125, .25, .375, .5, .75, 1., 1.25)
RETAIN_METRICS = ('spell', 'macro', 'short_precision', 'short_recall',
                  'dead_segment', 'dead_boundary', 'dead_recall')
POLICY = {
    'schema_version': 1,
    'minimum_retained_positive_gain_fraction': .9,
    'minimum_dead_precision_gain': .002,
    'retained_metrics': list(RETAIN_METRICS),
    'dead_logit_penalties': list(PENALTIES),
    'selection': 'passing candidates: highest mean spell gain, then dead-boundary gain, then declared grid order',
    'development_scope': 'previously consumed development; tuning only, not a fresh confirmation',
    'final_scope': 'one selected candidate; complete paired cohort predictions before references',
    'automatic_replacement': False,
}


def _paired(records, prior_records, reservation, partition):
    current = original._comparison(records, reservation, partition)
    prior = original._comparison(prior_records, reservation, partition)
    if prior['candidate_sha256'] != PARENT_SHA256:
        raise ValueError('Retention requires the exact frozen PFF parent')
    before = {r['match_id']: r for r in prior_records}
    for row in records:
        old = before[row['match_id']]
        if (row['baseline'] != old['baseline']
            or row['reference_dead_segments'] != old['reference_dead_segments']
            or row['reference_short_spells'] != old['reference_short_spells']):
            raise ValueError('Paired comparison baseline or reference support differs')
    floors = {k: POLICY['minimum_retained_positive_gain_fraction'] * max(0., prior['mean_gains'][k])
              for k in RETAIN_METRICS}
    # Tiny arithmetic tolerance only for the retention product; it cannot turn
    # a negative gain or a failed original gate into a passing comparison.
    retention = {k: current['mean_gains'][k] + 1e-12 >= floors[k] for k in RETAIN_METRICS}
    checks = dict(current['checks'],
                  retain_parent_gains=all(retention.values()),
                  dead_precision_margin=current['mean_gains']['dead_precision'] >= POLICY['minimum_dead_precision_gain'])
    return dict(current, checks=checks, numeric_gate_passed=all(checks.values()),
                parent_candidate_sha256=PARENT_SHA256, parent_mean_gains=prior['mean_gains'],
                retention_floors=floors, retention_checks=retention)


def development_gate(records, prior_records, reservation):
    value = _paired(records, prior_records, reservation, 'development')
    return dict(value, eligible_for_final_audit=value['numeric_gate_passed'],
                promoted=False, reference_scope=POLICY['development_scope'])


def build_protocol(reservation, prior_records, candidates, source_hashes):
    original.verify_protocol(reservation)
    prior = original.development_gate(prior_records, reservation)
    if prior['candidate_sha256'] != PARENT_SHA256:
        raise ValueError('Wrong PFF parent study')
    if not isinstance(candidates, list) or len(candidates) != len(PENALTIES):
        raise ValueError('Every fixed grid member must be declared before scoring')
    ids, hashes = set(), set()
    for position, row in enumerate(candidates):
        if set(row) != {'candidate_id', 'artifact_sha256', 'parameters'}:
            raise ValueError('Candidate declaration fields differ')
        ident = row['candidate_id']
        if not isinstance(ident, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', ident) or ident in ids:
            raise ValueError('Candidate IDs must be unique safe names')
        if not original._sha(row['artifact_sha256']) or row['artifact_sha256'] in hashes:
            raise ValueError('Every declared candidate requires a distinct artifact SHA256')
        params = row['parameters']
        if not isinstance(params, dict) or set(params) != {'dead_logit_penalty'}:
            raise ValueError('Only the declared dead-logit penalty may vary')
        penalty = params['dead_logit_penalty']
        if type(penalty) not in (int, float) or penalty != PENALTIES[position]:
            raise ValueError('Candidate grid values or order changed')
        ids.add(ident); hashes.add(row['artifact_sha256'])
    if (not isinstance(source_hashes, dict) or not source_hashes
        or any(not isinstance(k, str) or not k or not original._sha(v) for k, v in source_hashes.items())):
        raise ValueError('Scientific source hashes are required')
    value = {
        'schema_version': 1, 'reservation_sha256': reservation['protocol_sha256'],
        'parent_candidate_sha256': PARENT_SHA256,
        'prior_development_records_sha256': original.digest(prior_records),
        'policy': POLICY, 'candidates': candidates, 'source_hashes': source_hashes,
        'baseline_sha256': original.BASELINE_SHA256,
    }
    # Round-trip removes aliases to mutable caller objects.
    value = json.loads(json.dumps(value, allow_nan=False))
    return dict(value, protocol_sha256=original.digest(value))


def verify_protocol(study, reservation, prior_records):
    rebuilt = build_protocol(reservation, prior_records, study['candidates'], study['source_hashes'])
    if study != rebuilt:
        raise ValueError('Precision study protocol changed')
    return study


def freeze_protocol(path, study, reservation, prior_records):
    verify_protocol(study, reservation, prior_records)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(study, sort_keys=True, indent=2, allow_nan=False) + '\n'
    try:
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_text(encoding='utf-8') != content:
            raise ValueError('Existing precision protocol cannot be replaced') from None


def select_candidate(study, candidate_records, prior_records, reservation):
    verify_protocol(study, reservation, prior_records)
    expected = [r['candidate_id'] for r in study['candidates']]
    if not isinstance(candidate_records, dict) or set(candidate_records) != set(expected):
        raise ValueError('Selection requires the complete declared grid; no omitted or extra candidates')
    gates, passing = {}, []
    for rank, declared in enumerate(study['candidates']):
        ident = declared['candidate_id']
        gate = development_gate(candidate_records[ident], prior_records, reservation)
        if gate['candidate_sha256'] != declared['artifact_sha256']:
            raise ValueError('Candidate records do not match the declared artifact')
        if rank == 0:
            prior_by_id = {r['match_id']: r for r in prior_records}
            if any(r['candidate'] != prior_by_id[r['match_id']]['candidate'] for r in candidate_records[ident]):
                raise ValueError('Zero-penalty control must reproduce the frozen parent metrics')
        gates[ident] = gate
        if gate['eligible_for_final_audit']:
            passing.append((ident, rank, gate))
    passing.sort(key=lambda row: (-row[2]['mean_gains']['spell'], -row[2]['mean_gains']['dead_boundary'], row[1]))
    winner = passing[0] if passing else None
    return {
        'study_protocol_sha256': study['protocol_sha256'],
        'selected_candidate_id': None if winner is None else winner[0],
        'selected_candidate_sha256': None if winner is None else winner[2]['candidate_sha256'],
        'eligible_for_final_audit': winner is not None, 'promoted': False,
        'candidate_gates': gates, 'n_evaluated_candidates': len(expected),
        'development_scope': POLICY['development_scope'],
    }


def require_final_candidate(study, candidate_records, prior_records, reservation, candidate_sha256):
    """Recompute selection before preparing or opening any final-fixture data."""
    selection = select_candidate(study, candidate_records, prior_records, reservation)
    if not selection['eligible_for_final_audit'] or selection['selected_candidate_sha256'] != candidate_sha256:
        raise ValueError('Final audit requires exactly the passing development winner')
    return selection


def final_gate(records, prior_parent_records, reservation, evidence, selection):
    """Audit one selected candidate; caller first recomputes final access guard.

    Evidence declarations require independent substantiation. This function does
    not write artifacts or replace production even if every requirement passes.
    """
    value = _paired(records, prior_parent_records, reservation, 'final')
    if (selection.get('eligible_for_final_audit') is not True
        or selection.get('promoted') is not False
        or selection.get('selected_candidate_sha256') != value['candidate_sha256']):
        raise ValueError('Final records differ from the selected development winner')
    checks = dict(value['checks'], evidence=all(evidence.get(k) is True for k in original.EVIDENCE))
    return dict(value, checks=checks, promoted=all(checks.values()),
                reference_scope='one locked final cohort; evidence conditions remain required')
