"""Second, bounded PFF development family; original precision gates unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import re

from tools import pff_validation as original
from tools import pff_precision_validation as precision

PARENT_SHA256 = precision.PARENT_SHA256
BLENDS = (0., .125, .25, .5, .75, 1.)
POLICY = {
    'schema_version': 1,
    'family': 'training-only contextual dead-logit residual',
    'blends': list(BLENDS),
    'gates': {k: precision.POLICY[k] for k in ('minimum_retained_positive_gain_fraction',
               'minimum_dead_precision_gain', 'retained_metrics')},
    'development_scope': 'second declared family on previously consumed development; tuning only',
    'selection': 'passing candidates: highest mean spell gain, then dead-boundary gain, then declared grid order',
    'training_scope': 'exact original40 PFF training fixtures; no development or final references',
    'comparison_parent': 'original frozen PFF mixed candidate, not the best global-penalty result',
    'automatic_replacement': False,
}

development_gate = precision.development_gate
final_gate = precision.final_gate


def build_protocol(reservation, prior_records, candidates, source_hashes, *,
                   head_sha256, training_matches, fit_policy):
    from tools.pff_context_model import POLICY as expected_fit_policy
    original.verify_protocol(reservation)
    prior = original.development_gate(prior_records, reservation)
    if prior['candidate_sha256'] != PARENT_SHA256:
        raise ValueError('Context retention requires the original frozen PFF parent')
    if not original._sha(head_sha256):
        raise ValueError('Fitted context head SHA256 is required')
    if (not isinstance(training_matches, list)
        or any(type(mid) is not int for mid in training_matches)
        or training_matches != sorted(reservation['split']['train'])):
        raise ValueError('Context head must declare every original training fixture exactly once')
    if fit_policy != expected_fit_policy:
        raise ValueError('The complete fixed context fit policy must be declared unchanged')
    # Reject NaN/Infinity and unserializable declarations before freezing.
    json.dumps(fit_policy, allow_nan=False)
    if not isinstance(candidates, list) or len(candidates) != len(BLENDS):
        raise ValueError('The complete fixed context blend grid is required')
    ids, hashes = set(), set()
    for position, row in enumerate(candidates):
        if set(row) != {'candidate_id', 'artifact_sha256', 'parameters'}:
            raise ValueError('Context candidate fields differ')
        ident = row['candidate_id']
        if not isinstance(ident, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', ident) or ident in ids:
            raise ValueError('Context candidate IDs must be unique safe names')
        if not original._sha(row['artifact_sha256']) or row['artifact_sha256'] in hashes:
            raise ValueError('Every blend requires a distinct artifact identity')
        params = row['parameters']
        if not isinstance(params, dict) or set(params) != {'blend'}:
            raise ValueError('Only the declared blend may vary')
        blend = params['blend']
        if type(blend) not in (float, int) or blend != BLENDS[position]:
            raise ValueError('Context blend values or order changed')
        ids.add(ident); hashes.add(row['artifact_sha256'])
    if (not isinstance(source_hashes, dict) or not source_hashes
        or any(not isinstance(k, str) or not k or not original._sha(v) for k, v in source_hashes.items())):
        raise ValueError('Scientific source hashes are required')
    value = {'schema_version': 1, 'reservation_sha256': reservation['protocol_sha256'],
             'baseline_sha256': original.BASELINE_SHA256, 'parent_candidate_sha256': PARENT_SHA256,
             'prior_development_records_sha256': original.digest(prior_records),
             'head_sha256': head_sha256, 'training_matches': training_matches, 'fit_policy': fit_policy,
             'policy': POLICY, 'candidates': candidates, 'source_hashes': source_hashes}
    value = json.loads(json.dumps(value, allow_nan=False))
    return dict(value, protocol_sha256=original.digest(value))


def verify_protocol(study, reservation, prior_records):
    rebuilt = build_protocol(reservation, prior_records, study['candidates'], study['source_hashes'],
                             head_sha256=study['head_sha256'], training_matches=study['training_matches'],
                             fit_policy=study['fit_policy'])
    if study != rebuilt:
        raise ValueError('Context study protocol changed')
    return study


def freeze_protocol(path, study, reservation, prior_records):
    verify_protocol(study, reservation, prior_records)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(study, sort_keys=True, indent=2, allow_nan=False) + '\n'
    try:
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
    except FileExistsError:
        if path.read_text(encoding='utf-8') != text:
            raise ValueError('Existing context protocol cannot be replaced') from None


def select_candidate(study, candidate_records, prior_records, reservation):
    verify_protocol(study, reservation, prior_records)
    expected = [d['candidate_id'] for d in study['candidates']]
    if not isinstance(candidate_records, dict) or set(candidate_records) != set(expected):
        raise ValueError('Selection requires all six declared blends and no extra candidates')
    gates, passing = {}, []
    old = {r['match_id']: r for r in prior_records}
    for rank, declared in enumerate(study['candidates']):
        ident = declared['candidate_id']; records = candidate_records[ident]
        gate = development_gate(records, prior_records, reservation)
        if gate['candidate_sha256'] != declared['artifact_sha256']:
            raise ValueError('Context scores differ from the declared correction artifact')
        if rank == 0 and any(row['candidate'] != old[row['match_id']]['candidate'] for row in records):
            raise ValueError('Zero blend must reproduce the original parent metrics')
        gates[ident] = gate
        if gate['eligible_for_final_audit']:
            passing.append((ident, rank, gate))
    passing.sort(key=lambda row: (-row[2]['mean_gains']['spell'], -row[2]['mean_gains']['dead_boundary'], row[1]))
    winner = passing[0] if passing else None
    return {'study_protocol_sha256': study['protocol_sha256'],
            'selected_candidate_id': None if winner is None else winner[0],
            'selected_candidate_sha256': None if winner is None else winner[2]['candidate_sha256'],
            'eligible_for_final_audit': winner is not None, 'promoted': False,
            'candidate_gates': gates, 'n_evaluated_candidates': len(expected),
            'development_scope': POLICY['development_scope']}


def require_final_candidate(study, candidate_records, prior_records, reservation, candidate_sha256):
    selection = select_candidate(study, candidate_records, prior_records, reservation)
    if not selection['eligible_for_final_audit'] or selection['selected_candidate_sha256'] != candidate_sha256:
        raise ValueError('Final audit requires exactly the passing context winner')
    return selection
