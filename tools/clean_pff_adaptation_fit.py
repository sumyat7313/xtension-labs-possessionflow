"""Pure, bounded inner-training kernel for the fixed clean PFF A/B family.

There is deliberately no filesystem loader, CLI, scorer, or refit command here.
The future runner must authenticate cache bytes, connected lineage, source hashes
and consumed attempts before supplying a lazy record factory. Content identity of
the clean parent is checked here too; caller-declared SHA strings are not proof.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from possession import learned, robust_control
from tools import clean_pff_adaptation_protocol as protocol
from tools import train_joint
from tools.control_diagnostics import training_spell_weights


PARENT_CONTENT_SHA256 = 'bd25daae95757541d84130226b7a7f34327911a3919dd8ace96f3e84a80d7f09'
PARENT_FILE_SHA256 = '7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47'
DEAD_PARAMETERS = ('mean', 'scale', 'coefficients', 'intercept')
RECORD_KEYS = {'match', 'x', 'z', 'y', 'blocks', 'fps', 'reference_fields_used_for_inference'}


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_parent(model):
    """Exact clean model contents, not merely compatible feature dimensions."""
    if content_digest(model) != PARENT_CONTENT_SHA256:
        raise ValueError('Only the authenticated clean parent content is accepted')
    learned.validate_model(model)
    if set(model) != {'candidate_id', 'coefficients', 'decoder_mode', 'feature_names',
                     'intercept', 'joint_model', 'kind', 'mean', 'metadata',
                     'robust_control_model', 'scale', 'schema_version'}:
        raise ValueError('Unexpected clean parent components')
    return model


def validate_fixture_ids(fixture_ids, phase='inner'):
    """Validate the complete canonical roster before calling any record loader."""
    if phase != 'inner':
        raise ValueError('Winner refit requires a future verified-selection adapter')
    expected = tuple(protocol.fit_ids('inner'))
    if (not isinstance(fixture_ids, (tuple, list)) or
            any(type(v) is not str for v in fixture_ids) or
            len(fixture_ids) != len(set(fixture_ids)) or tuple(fixture_ids) != expected):
        raise ValueError('Exact ordered inner-fitting roster is required; held-out fixtures forbidden')
    return expected


def _blocks(value, n):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError('A nonempty period list is required')
    result, previous, periods = [], -1, set()
    for block in value:
        if not isinstance(block, (tuple, list)) or len(block) != 3:
            raise ValueError('Invalid period block')
        period, a, b = block
        if (type(period) is not int or period in periods or
                type(a) is not int or type(b) is not int or not previous < a <= b < n):
            raise ValueError('Periods must be unique, ordered, disjoint and in range')
        periods.add(period); previous = b; result.append((period, a, b))
    return result


def sample_record(record, requested_match, parameters):
    """Materialize only one approved fixture; reject upstream arrays before access."""
    if requested_match not in protocol.fit_ids('inner'):
        raise ValueError('Only inner-fitting records are accepted')
    if not isinstance(record, dict) or record.get('match') != requested_match:
        raise ValueError('Record identity must match the requested fitting fixture')
    if set(record) != RECORD_KEYS:
        raise ValueError('Only observation matrices, training references and period metadata are accepted')
    if record['reference_fields_used_for_inference'] is not False or type(record['fps']) not in (int, float) or record['fps'] != 25:
        raise ValueError('Require observation-only features on the fixed 25 Hz timeline')
    x, z, y = np.asarray(record['x']), np.asarray(record['z']), np.asarray(record['y'])
    if (x.ndim != 2 or x.shape[1] != len(robust_control.FEATURE_NAMES) or
            z.shape != (len(x), len(learned.FEATURE_NAMES)) or y.shape != (len(x),) or
            not np.isfinite(x).all() or not np.isfinite(z).all() or
            not np.isin(y, list('HADX')).all()):
        raise ValueError('Invalid fitting observation matrices or reference labels')
    blocks = _blocks(record['blocks'], len(x))
    weights = np.zeros(len(y), dtype=float)
    stride = int(record['fps'] / parameters['sample_hz'])
    radius = round(parameters['boundary_radius_seconds'] * record['fps'])
    indices = []
    for _, a, b in blocks:
        local = y[a:b+1]
        w = training_spell_weights(local, record['fps'])
        w[local == 'D'] = 1.
        transitions = np.flatnonzero((local[1:] != local[:-1]) &
                                    (local[1:] != 'X') & (local[:-1] != 'X')) + 1
        for j in transitions:
            lo, hi = max(0, j-radius), min(len(w), j+radius+1)
            w[lo:hi] = np.maximum(w[lo:hi], parameters['boundary_weight'])
        w[local == 'X'] = 0.
        weights[a:b+1] = w
        indices.append(np.arange(a, b+1, stride))
    mass = float(sum(weights[ix].sum() for ix in indices))
    if mass <= 0:
        raise ValueError('Every fitting fixture needs positive sampled known-label mass')
    weights *= parameters['fixture_mass'] / mass
    sequences = [dict(X=np.array(x[ix], dtype=float, copy=True),
                      Z=np.array(z[ix], dtype=float, copy=True),
                      y=np.array(y[ix], copy=True), w=np.array(weights[ix], copy=True))
                 for ix in indices]
    support = dict(periods=len(sequences), sampled_rows=sum(len(s['y']) for s in sequences),
                   class_counts={c: sum(int(np.sum(s['y'] == c)) for s in sequences) for c in 'HADX'},
                   sampled_weight=float(sum(s['w'].sum() for s in sequences)))
    return sequences, support


def _logit(probability):
    value = np.asarray(probability, dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Invalid recomputed upstream probabilities')
    value = np.clip(value, 1e-8, 1-1e-8)
    return np.log(value) - np.log1p(-value)


def fit_candidate(candidate, record_factory, *, fixture_ids, parent_model,
                  protocol_sha256, phase='inner'):
    """Fit exactly A or B on the inner roster, returning local weights and aggregates.

    B trains its D scaler/head on these same fitting rows and then uses its own
    in-sample upstream outputs for joint training. Neither candidate uses inner
    validation rows, cached base probabilities, or the discarded old joint head.
    The supplied protocol digest must identify the exact static scientific core;
    it is not a future execution-manifest digest or an authorization receipt.
    """
    roster = validate_fixture_ids(fixture_ids, phase)
    if candidate not in ('A', 'B') or type(candidate) is not str:
        raise ValueError('Only the fixed A/B family is accepted')
    if type(protocol_sha256) is not str or protocol_sha256 != protocol.fixed_protocol_sha256():
        raise ValueError('The exact frozen scientific-core protocol digest is required')
    if not callable(record_factory):
        raise ValueError('A lazy authenticated record factory is required')
    validate_parent(parent_model)
    saved_parent = content_digest(parent_model)
    model = copy.deepcopy(parent_model)
    model.pop('joint_model')
    parameters = protocol.recipe()
    sequences, support = [], {}
    for match in roster:
        sampled, counts = sample_record(record_factory(match), match, parameters)
        sequences.extend(sampled); support[match] = counts
    if not all(sum(support[m]['class_counts'][c] for m in roster) > 0 for c in 'HAD'):
        raise ValueError('The fixed fitting cohort requires sampled H, A and D support')
    metadata = dict(candidate=candidate, phase=phase, protocol_sha256=protocol_sha256,
                    training_matches=list(roster), clean_parent_file_sha256=PARENT_FILE_SHA256,
                    clean_parent_content_sha256=PARENT_CONTENT_SHA256,
                    excluded_pff_matches=[f'pff{mid}' for mid in protocol.EXCLUDED_INNER_IDS],
                    fitting_data='PFF inner32 plus seven clean Sportec fixtures',
                    use_scope='local analytical research; delivery rights unresolved',
                    fitted_weights_must_remain_local=True,
                    frozen_control_content_sha256=content_digest(parent_model['robust_control_model']),
                    parent_dead_parameters_content_sha256=content_digest({k: parent_model[k] for k in DEAD_PARAMETERS}),
                    discarded_joint_content_sha256=content_digest(parent_model['joint_model']),
                    upstream_stacking='in-sample-fit-only', old_joint_used=False,
                    external_only=True, final_accessed=False, promoted=False,
                    submission_eligible=False)
    if candidate == 'B':
        dead_fit = learned.fit_model(np.concatenate([s['Z'] for s in sequences]),
                                    np.concatenate([s['y'] for s in sequences]),
                                    l2=parameters['dead_l2'],
                                    sample_weight=np.concatenate([s['w'] for s in sequences]),
                                    max_iter=parameters['dead_max_iter'], metadata=metadata)
        for key in DEAD_PARAMETERS:
            model[key] = copy.deepcopy(dead_fit[key])
        model['training'] = copy.deepcopy(dead_fit['training'])
    for sequence in sequences:
        sequence['base_home'] = _logit(robust_control.predict_home(model['robust_control_model'], sequence['X']))
        sequence['base_dead'] = _logit(learned.predict_dead(sequence['Z'], model))
    model['joint_model'] = train_joint.fit(sequences, seed=parameters['seed'],
                                         epochs=parameters['epochs'], metadata=metadata)
    model['candidate_id'] = f'clean-pff-adaptation-{candidate}-inner'
    model['metadata'] = dict(adaptation=metadata, frozen_parent_metadata=copy.deepcopy(parent_model['metadata']))
    learned.validate_model(model)
    if (content_digest(parent_model) != saved_parent or
            model['robust_control_model'] != parent_model['robust_control_model'] or
            model['decoder_mode'] != parent_model['decoder_mode'] or
            (candidate == 'A' and any(model[k] != parent_model[k] for k in DEAD_PARAMETERS))):
        raise ValueError('Fixed parent heads or decoder changed')
    report = dict(metadata, recipe=parameters, fitting_fixtures=len(roster),
                  fixture_support=support, new_joint_training=copy.deepcopy(model['joint_model'].get('training', {})),
                  new_dead_training=copy.deepcopy(model.get('training')) if candidate == 'B' else None,
                  artifact_delivery_rights_verified=False, cache_bytes_authenticated_by_kernel=False,
                  development_accessed=False, refit_authorized=False)
    return model, report
