"""Pure training-only A/B protocol and inner selection; no provider I/O or fits.

Receipts authenticate internal consistency, not actual artifact bytes or chronology.
A future campaign must independently verify execution before authorizing a refit.
No development/final scoring or promotion interface is provided here.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import PurePosixPath

from tools import pff_validation as original

CLEAN_SHA256 = '7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47'
CONFIG_SHA256 = '2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b'
RESERVATION_SHA256 = '4475cf6cf95de433a49a1c03ab93838be96f05059eee15a09a73e1673557e980'
DIAGNOSIS_SHA256 = '2142e61b6f0331c29addc7050fb351624ea5660f50b0144538d8a4f57e6d37a2'
BASELINE_RECORDS_SHA256 = '4a8f9ba80057e0e7eaef82d721308b16d6109d852e36f48f657e487d0bbba188'
SPLIT_SEED = 'xtension-clean-pff-adaptation-v1:inner'
TRAIN_IDS = (3812, 3817, 3819, 3820, 3821, 3822, 3825, 3826, 3827, 3828, 3829,
             3832, 3833, 3834, 3835, 3836, 3838, 3839, 3842, 3843, 3845, 3846,
             3847, 3849, 3851, 3853, 3855, 3856, 3857, 3859, 10503, 10504, 10505,
             10507, 10508, 10509, 10512, 10513, 10515, 10516)
SPORTEC_IDS = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')
RANKED_IDS = tuple(sorted(TRAIN_IDS, key=lambda mid: hashlib.sha256(f'{SPLIT_SEED}:{mid}'.encode()).hexdigest()))
INNER_VALIDATION_IDS = tuple(sorted(RANKED_IDS[:8]))
INNER_TRAIN_IDS = tuple(sorted(RANKED_IDS[8:]))
DEVELOPMENT_IDS = (3813, 3816, 3818, 3823, 3831, 3844, 3850, 3854, 3858, 10502, 10514, 10517)
FINAL_IDS = (3814, 3815, 3824, 3830, 3837, 3840, 3841, 3848, 3852, 10506, 10510, 10511)
EXCLUDED_INNER_IDS = tuple(sorted((*INNER_VALIDATION_IDS, *DEVELOPMENT_IDS, *FINAL_IDS)))
CANDIDATES = ('A', 'B')
METRICS = tuple(original.METRICS)
PROTECTED = ('spell', 'short_precision', 'dead_precision', 'dead_recall')
SUPPORT_FIELDS = ('known_frames', 'reference_live', 'reference_dead', 'reference_dead_segments',
                  'reference_short_spells', 'reference_short_frames')
FLAGS = dict(execution_evidence_verified=False, campaign_ready=False, refit_authorized=False,
             development_access_authorized=False, final_access_authorized=False,
             promoted=False, submission_eligible=False)


def recipe():
    return dict(schema_version=1, seed=20261101, epochs=8, dead_l2=.01, dead_max_iter=80,
                fixture_mass=10000., sample_hz=5, boundary_radius_seconds=.4, boundary_weight=1.5,
                upstream_stacking='in-sample-fit-only',
                short_spell_weighting='training_spell_weights: capped inverse-square-root .5..3',
                dead_frame_weight=1., unknown_frame_weight=0.,
                sportec_regimes=['clean'], sportec_required=True,
                clean_control_sha256=CLEAN_SHA256, frozen_config_sha256=CONFIG_SHA256,
                joint_architecture='equivariant_joint_temporal_residual',
                joint_optimizer=dict(name='AdamW', learning_rate=.002, weight_decay=.01),
                new_scaling='fitting fixtures only; weighted existing scaler implementation',
                loss_weight_search=False, decoder_search=False, cross_fitted_upstream=False)


def fit_ids(stage='inner'):
    if stage not in ('inner', 'refit'):
        raise ValueError('Only declared inner fitting or future winner refit rosters exist')
    pff = INNER_TRAIN_IDS if stage == 'inner' else TRAIN_IDS
    return tuple(sorted((*[f'pff{mid}' for mid in pff], *SPORTEC_IDS)))


def inner_gate():
    return dict(mean_nonregression=list(PROTECTED), strictly_positive_mean_spell_gain=True,
                maximum_fixture_spell_drop=.01, maximum_fixture_dead_recall_drop=.005,
                minimum_fixtures=8, minimum_dead_segments_per_fixture=1,
                minimum_short_spells_per_fixture=1, minimum_total_dead_segments=100,
                minimum_total_short_spells=100,
                ranking=['mean_spell', 'mean_short_precision', 'mean_dead_precision', 'mean_dead_recall', 'A_before_B'],
                tolerance=0., bootstrap_or_fresh_significance_claim=False)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    _need(original._sha(value), 'Expected a lowercase SHA-256 identity')
    return value


def _integer(value, minimum=0):
    _need(type(value) is int and value >= minimum, 'Expected a bounded native integer, not bool')
    return value


def _metrics(value):
    _need(isinstance(value, dict) and set(value) == set(METRICS), 'Exact eight-metric record required')
    for number in value.values():
        _need(type(number) in (int, float) and math.isfinite(number) and 0 <= number <= 1,
              'Metric must be finite in [0,1], not bool')
    return value


def _support(value):
    _need(isinstance(value, dict) and set(value) == set(SUPPORT_FIELDS), 'Reference support schema differs')
    for count in value.values():
        _integer(count)
    _need(value['known_frames'] == value['reference_live'] + value['reference_dead'], 'Known-frame support does not reconcile')
    _need(value['reference_live'] > 0 and value['reference_dead'] > 0, 'Both live and stopped support are required')
    _need(1 <= value['reference_short_spells'] <= value['reference_short_frames'] <= value['reference_live'],
          'Short-spell support does not reconcile')
    _need(1 <= value['reference_dead_segments'] <= value['reference_dead'], 'Stoppage segment support does not reconcile')
    return value


def _sources(value):
    required = {'tools/clean_pff_adaptation_protocol.py', 'tools/clean_pff_adaptation_fit.py'}
    _need(isinstance(value, dict) and required <= set(value),
          'Protocol and fitting-kernel source identities are required')
    for name, identity in value.items():
        _need(isinstance(name, str) and '\\' not in name and ':' not in name and
              not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
              'Source names must be repository-relative paths')
        _hash(identity)
    return value


def comparator_records(diagnosis):
    _need(original.digest(diagnosis) == DIAGNOSIS_SHA256, 'Completed training diagnosis identity differs')
    rows = []
    for record in diagnosis['records']:
        if record['match_id'] not in INNER_VALIDATION_IDS:
            continue
        detail = next(value for value in record['details'] if value['candidate'] == 'full')
        counts = record['diagnostics']['models']['full']
        rows.append(dict(match_id=record['match_id'], condition='native', baseline_sha256=CLEAN_SHA256,
            baseline_prediction_sha256=detail['prediction_sha256'],
            reference_sha256=record['reference_support']['event_sha256'], metrics=record['metrics']['full'],
            reference_support=dict(known_frames=counts['known_frames'], reference_live=counts['reference_live'],
                reference_dead=counts['reference_dead'], reference_dead_segments=detail['temporal_diagnostics']['dead_segments']['n_gt'],
                reference_short_spells=record['diagnostics']['short_false_dead']['n_gt'],
                reference_short_frames=record['diagnostics']['short_false_dead']['reference_frames'])))
    rows.sort(key=lambda row: row['match_id'])
    _need(original.digest(rows) == BASELINE_RECORDS_SHA256, 'Frozen clean comparator records differ')
    return copy.deepcopy(rows)


def core_protocol():
    return dict(schema_version=1, family='clean-pff-adaptation-v1', reservation_sha256=RESERVATION_SHA256,
                diagnosis_sha256=DIAGNOSIS_SHA256, comparator_sha256=CLEAN_SHA256, config_sha256=CONFIG_SHA256,
                split_seed=SPLIT_SEED, split_rule='SHA256(seed:decimal_match_id), ascending digest; first eight inner validation',
                train_ids=list(TRAIN_IDS), inner_train_ids=list(INNER_TRAIN_IDS), inner_validation_ids=list(INNER_VALIDATION_IDS),
                split_ranked_ids=list(RANKED_IDS), sportec_ids=list(SPORTEC_IDS),
                labels_used_for_split=False, inner_freshness='Previously exposed training-development fixtures',
                fitting_roster=list(fit_ids()), future_winner_refit_roster=list(fit_ids('refit')),
                candidates={'A': {'upstream_dead': 'frozen-clean', 'joint': 'new-fit'},
                            'B': {'upstream_dead': 'refit-inner', 'joint': 'new-fit'}},
                recipe=recipe(), inner_gate=inner_gate(), baseline_records_sha256=BASELINE_RECORDS_SHA256,
                original_development_policy=copy.deepcopy(original.POLICY),
                additional_development_precision_margin=.002, parent_gain_retention=.9,
                stages='Inner fit/score only; future runner must authenticate execution and authorize any winner refit',
                **FLAGS)


def fixed_protocol_sha256():
    """Static scientific recipe identity; actual execution sources bind separately."""
    return original.digest(core_protocol())


def _body(source_hashes, baseline_records):
    return dict(core_protocol(), core_protocol_sha256=fixed_protocol_sha256(),
                source_hashes=copy.deepcopy(_sources(source_hashes)), baseline_records=copy.deepcopy(baseline_records))


def build_protocol(reservation, diagnosis_summary, source_hashes):
    original.verify_protocol(reservation)
    _need(reservation['protocol_sha256'] == RESERVATION_SHA256 and tuple(reservation['split']['train']) == TRAIN_IDS,
          'Original training reservation differs')
    value = _body(source_hashes, comparator_records(diagnosis_summary))
    return dict(value, protocol_sha256=original.digest(value))


def verify_protocol(protocol):
    _need(isinstance(protocol, dict), 'Protocol must be a mapping')
    baseline = protocol.get('baseline_records')
    _need(original.digest(baseline) == BASELINE_RECORDS_SHA256, 'Frozen comparator scores or support changed')
    expected = _body(protocol.get('source_hashes'), baseline)
    expected['protocol_sha256'] = original.digest(expected)
    _need(protocol == expected, 'Fixed A/B recipe, split, gates or protocol identity changed')
    # JSON distinguishes bool from int but Python container equality does not.
    _need(original.digest(protocol) == original.digest(expected), 'Protocol contains noncanonical types')
    return protocol


def validate_candidate_receipt(receipt, candidate, protocol):
    verify_protocol(protocol)
    _need(candidate in CANDIDATES and isinstance(receipt, dict), 'Only A/B candidate receipts are allowed')
    _hash(receipt.get('model_sha256'))
    _need(receipt['model_sha256'] != CLEAN_SHA256, 'New fitted artifact cannot claim the comparator identity')
    expected = dict(candidate_id=candidate, model_sha256=receipt['model_sha256'],
        protocol_sha256=protocol['protocol_sha256'], core_protocol_sha256=fixed_protocol_sha256(), phase='inner',
        parent_sha256=CLEAN_SHA256, config_sha256=CONFIG_SHA256, recipe_sha256=original.digest(recipe()),
        fitting_fixture_ids=list(fit_ids()), normalization_fixture_ids=list(fit_ids()),
        new_dead_fit_fixture_ids=list(fit_ids()) if candidate == 'B' else [],
        upstream_dead=protocol['candidates'][candidate]['upstream_dead'], upstream_stacking='in-sample-fit-only')
    _need(receipt == expected and original.digest(receipt) == original.digest(expected),
          'Candidate receipt violates fixed inner-only fitting provenance')
    return receipt


def _freeze(freeze, protocol, candidates):
    _need(isinstance(freeze, dict), 'Complete prediction freeze required')
    expected_keys = {'schema_version', 'protocol_sha256', 'model_sha256', 'fit_receipt_sha256',
                     'records', 'references_accessed', 'all_predictions_complete'}
    _need(set(freeze) == expected_keys and type(freeze['schema_version']) is int and freeze['schema_version'] == 1,
          'Prediction-freeze schema differs')
    _need(freeze['protocol_sha256'] == protocol['protocol_sha256'] and freeze['references_accessed'] is False and
          freeze['all_predictions_complete'] is True, 'Complete predictions must be declared before references')
    _need(freeze['model_sha256'] == {cid: candidates[cid]['model_sha256'] for cid in CANDIDATES}, 'Frozen model identity differs')
    _need(freeze['fit_receipt_sha256'] == {cid: original.digest(candidates[cid]) for cid in CANDIDATES}, 'Fit receipt binding differs')
    expected = {(cid, mid) for cid in CANDIDATES for mid in INNER_VALIDATION_IDS}
    pairs = {}
    _need(isinstance(freeze['records'], list), 'Prediction record list required')
    for row in freeze['records']:
        _need(isinstance(row, dict) and set(row) == {'candidate_id', 'match_id', 'prediction_sha256'}, 'Prediction record schema differs')
        _integer(row['match_id'])
        pair = (row['candidate_id'], row['match_id'])
        _need(pair in expected and pair not in pairs, 'Duplicate, missing or forbidden-cohort prediction')
        pairs[pair] = _hash(row['prediction_sha256'])
    _need(set(pairs) == expected, 'Both complete eight-match prediction cohorts are required')
    return pairs


def select_candidate(records, protocol, *, candidates, prediction_freeze, score_receipt):
    """Select a numerically qualified inner winner, without authorizing execution."""
    verify_protocol(protocol)
    _need(isinstance(candidates, dict) and set(candidates) == set(CANDIDATES), 'Exactly two fitted candidate receipts required')
    for cid in CANDIDATES:
        validate_candidate_receipt(candidates[cid], cid, protocol)
    _need(len({value['model_sha256'] for value in candidates.values()}) == 2, 'A/B fitted artifacts must have distinct identities')
    predicted = _freeze(prediction_freeze, protocol, candidates)
    _need(isinstance(records, list), 'Paired score record list required')
    expected_receipt = dict(schema_version=1, protocol_sha256=protocol['protocol_sha256'],
        prediction_freeze_sha256=original.digest(prediction_freeze), records_sha256=original.digest(records),
        scored_match_ids=list(INNER_VALIDATION_IDS), condition='native',
        reference_access_started_after_prediction_freeze=True, both_candidates_scored=True,
        development_accessed=False, final_accessed=False)
    _need(score_receipt == expected_receipt and original.digest(score_receipt) == original.digest(expected_receipt),
          'Score records, reference-access declaration or frozen prediction receipt differ')
    baseline = {row['match_id']: row for row in protocol['baseline_records']}
    groups = {cid: {} for cid in CANDIDATES}
    fields = {'candidate_id', 'match_id', 'condition', 'protocol_sha256', 'model_sha256', 'fit_receipt_sha256',
              'prediction_sha256', 'baseline_sha256', 'baseline_prediction_sha256', 'reference_sha256',
              'reference_support', 'metrics', 'short_diagnostics'}
    for row in records:
        _need(isinstance(row, dict) and set(row) == fields, 'Paired score schema differs')
        mid = _integer(row['match_id'])
        cid = row['candidate_id']
        _need(cid in CANDIDATES and mid in INNER_VALIDATION_IDS and mid not in groups[cid], 'Duplicate or forbidden-cohort score')
        base = baseline[mid]
        expected_ids = dict(condition='native', protocol_sha256=protocol['protocol_sha256'],
            model_sha256=candidates[cid]['model_sha256'], fit_receipt_sha256=original.digest(candidates[cid]),
            prediction_sha256=predicted[cid, mid], baseline_sha256=CLEAN_SHA256,
            baseline_prediction_sha256=base['baseline_prediction_sha256'], reference_sha256=base['reference_sha256'])
        _need(all(row[k] == value for k, value in expected_ids.items()), 'Score provenance or frozen artifact identity differs')
        _metrics(row['metrics'])
        _support(row['reference_support'])
        _need(row['reference_support'] == base['reference_support'], 'Reference support differs between paired models')
        short = row['short_diagnostics']
        _need(isinstance(short, dict) and set(short) == {'predicted_short_spells', 'matched_predicted_short_spells',
              'short_spells_with_dead', 'short_spells_erased_by_dead', 'short_false_dead_frames'}, 'Short diagnostic counts required')
        for value in short.values():
            _integer(value)
        _need(short['matched_predicted_short_spells'] <= short['predicted_short_spells'] and
              short['short_spells_erased_by_dead'] <= short['short_spells_with_dead'] <= base['reference_support']['reference_short_spells'] and
              short['short_spells_with_dead'] <= short['short_false_dead_frames'] <= base['reference_support']['reference_short_frames'] and
              ((short['short_spells_with_dead'] == 0) == (short['short_false_dead_frames'] == 0)), 'Short diagnostic counts do not reconcile')
        derived_precision = (short['matched_predicted_short_spells'] / short['predicted_short_spells']
                             if short['predicted_short_spells'] else 0.)
        _need(math.isclose(row['metrics']['short_precision'], derived_precision, rel_tol=0, abs_tol=1e-12),
              'Short precision differs from the reported matched/predicted counts')
        groups[cid][mid] = row
    _need(all(set(rows) == set(INNER_VALIDATION_IDS) for rows in groups.values()), 'Both full eight-match score cohorts are required')
    _need(sum(row['reference_support']['reference_dead_segments'] for row in baseline.values()) >= 100 and
          sum(row['reference_support']['reference_short_spells'] for row in baseline.values()) >= 100, 'Insufficient reference segment support')
    results = {}
    for cid, rows in groups.items():
        gains = {metric: math.fsum(rows[mid]['metrics'][metric] - baseline[mid]['metrics'][metric]
                                  for mid in INNER_VALIDATION_IDS) / 8 for metric in METRICS}
        means = {metric: math.fsum(row['metrics'][metric] for row in rows.values()) / 8 for metric in METRICS}
        worst_spell = min(rows[mid]['metrics']['spell'] - baseline[mid]['metrics']['spell'] for mid in INNER_VALIDATION_IDS)
        worst_dead = min(rows[mid]['metrics']['dead_recall'] - baseline[mid]['metrics']['dead_recall'] for mid in INNER_VALIDATION_IDS)
        checks = {f'nonregression_{metric}': gains[metric] >= 0 for metric in PROTECTED}
        checks.update(positive_mean_spell=gains['spell'] > 0,
                      worst_fixture_spell=all(rows[mid]['metrics']['spell'] >= baseline[mid]['metrics']['spell'] - .01
                                             for mid in INNER_VALIDATION_IDS),
                      worst_fixture_dead_recall=all(rows[mid]['metrics']['dead_recall'] >= baseline[mid]['metrics']['dead_recall'] - .005
                                                   for mid in INNER_VALIDATION_IDS))
        results[cid] = dict(passed=all(checks.values()), checks=checks, mean_metrics=means, mean_gains=gains,
                           worst_spell_gain=worst_spell, worst_dead_recall_gain=worst_dead,
                           short_diagnostic_totals={key: sum(row['short_diagnostics'][key] for row in rows.values())
                                                    for key in next(iter(rows.values()))['short_diagnostics']})
        totals = results[cid]['short_diagnostic_totals']
        totals['unmatched_predicted_short_spells'] = totals['predicted_short_spells'] - totals['matched_predicted_short_spells']
    ranking = sorted(CANDIDATES, key=lambda cid: (*[-results[cid]['mean_metrics'][metric] for metric in PROTECTED], cid))
    selected = next((cid for cid in ranking if results[cid]['passed']), None)
    return dict(schema_version=1, protocol_sha256=protocol['protocol_sha256'],
                records_sha256=original.digest(records), prediction_freeze_sha256=original.digest(prediction_freeze),
                score_receipt_sha256=original.digest(score_receipt), candidates=results, ranking=ranking,
                selected_candidate_id=selected, inner_numerical_passed=selected is not None,
                interpretation='Training-development selection on declared receipts; execution verification remains mandatory',
                **FLAGS)
