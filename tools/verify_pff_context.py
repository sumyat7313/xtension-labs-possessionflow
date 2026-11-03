"""Authenticate aggregate PFF context research without publishing weights."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from tools import pff_context_campaign as campaign, pff_context_validation as validation
from tools import pff_validation as original, verify_pff_research as parent_verifier


def _scalar_context():
    from tools.verify_pff_precision import verify_bundle as verify_scalar
    path = ROOT / 'models/pff-precision-research.json'
    bundle = read(path)
    result = verify_scalar(bundle)
    if result['development_gate_passed'] or result['final_evaluated']:
        raise ValueError('Context follow-up requires the reported failed scalar grid with final untouched')
    return sha256(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _parent_context():
    path = ROOT / 'models/pff-research.json'
    parent = read(path)
    parent_verifier.verify_bundle(parent)
    if parent['fit']['candidate_sha256'] != validation.PARENT_SHA256:
        raise ValueError('Published parent identity differs')
    return parent, sha256(path)


def _verify_training(bundle, parent, reservation, study):
    plan, fit, access = bundle['training_plan'], bundle['fit'], bundle['fit_access']
    expected_plan = {'schema_version': 1, 'source_hashes': study['source_hashes'],
                     'fit_policy': campaign.correction.POLICY, 'training_matches': reservation['split']['train'],
                     'held_out_matches': reservation['split']['development'] + reservation['split']['final'],
                     'reservation_sha256': reservation['protocol_sha256'], 'blend_grid': list(validation.BLENDS),
                     'parent_sha256': validation.PARENT_SHA256, 'previous_scalar_research_sha256': _scalar_context(),
                     'adaptive_followup': True, 'development_previously_consumed': True}
    expected_plan['protocol_sha256'] = original.digest(expected_plan)
    if plan != expected_plan:
        raise ValueError('Pre-fit plan, exact training membership or previous-study provenance differs')
    if set(fit) != {'head_sha256', 'training_plan_sha256', 'fit_access_sha256', 'head_metadata', 'training',
                    'training_reference_support', 'training_probability_manifests', 'training_probability_complete',
                    'held_out_references_read'}:
        raise ValueError('Fit evidence fields differ; fitted weights may not be included')
    if (fit['head_sha256'] != study['head_sha256']
        or fit['training_plan_sha256'] not in parent_verifier._serialized_hashes(plan)
        or fit['fit_access_sha256'] not in parent_verifier._serialized_hashes(access, False)
        or fit['held_out_references_read'] is not False
        or fit['training_reference_support'] != parent['fit']['training_reference_support']):
        raise ValueError('Fitted head, training references or fit-access provenance differs')
    complete = fit['training_probability_complete']
    expected_access = {'training_plan_sha256': fit['training_plan_sha256'],
                       'training_probabilities_sha256': access['training_probabilities_sha256'],
                       'held_out_references_read': False}
    if access != expected_access or access['training_probabilities_sha256'] not in parent_verifier._serialized_hashes(complete):
        raise ValueError('Exclusive fit access is not bound to complete pre-fit inputs')
    training_ids = plan['training_matches']; preparation_sha = study['source_hashes']['tools/pff_context_prepare.py']
    if (complete['schema_version'] != 1 or complete['parent_candidate_sha256'] != validation.PARENT_SHA256
        or complete['protocol_sha256'] != reservation['protocol_sha256'] or complete['references_read'] is not False
        or complete['preparation_source_sha256'] != preparation_sha
        or [r['match_id'] for r in complete['records']] != training_ids
        or [r['match_id'] for r in fit['training_probability_manifests']] != training_ids):
        raise ValueError('Training probability preparation must cover exactly the frozen forty fixtures')
    parents = {r['match_id']: r for r in parent['cache_manifests']}
    for record, manifest in zip(complete['records'], fit['training_probability_manifests']):
        mid = record['match_id']; old = parents[mid]; identity = manifest['identity']
        expected_identity = {'parent_candidate_sha256': validation.PARENT_SHA256,
                             'source_cache_manifest_sha256': identity['source_cache_manifest_sha256'],
                             'source_cache_identity': old['identity'], 'protocol_sha256': reservation['protocol_sha256'],
                             'preparation_source_sha256': preparation_sha}
        if (manifest['schema_version'] != 1 or identity != expected_identity
            or identity['source_cache_manifest_sha256'] not in parent_verifier._serialized_hashes(old)
            or record['manifest_sha256'] not in parent_verifier._serialized_hashes(manifest)
            or manifest['rows'] != old['rows'] or record['rows'] != old['rows'] or manifest['fps'] != 25
            or manifest['columns'] != ['H', 'A', 'D'] or manifest['references_read'] is not False
            or manifest['reference_fields_used_for_inference'] is not False
            or not original._sha(record['probability_sha256'])
            or manifest['files'] != {'parent_probabilities.npy': record['probability_sha256']}):
            raise ValueError('Training probability cache provenance differs')
    metadata = fit['head_metadata']
    expected_metadata = {'training_matches': training_ids, 'held_out_matches': sorted(plan['held_out_matches']),
                         'protocol_sha256': plan['protocol_sha256'], 'policy': campaign.correction.POLICY,
                         'support': metadata['support'], 'parent_sha256': validation.PARENT_SHA256,
                         'reference_fields_used_for_inference': False}
    if metadata != expected_metadata or set(metadata['support']) != {str(i) for i in training_ids}:
        raise ValueError('Context head training membership, plan or reference isolation differs')
    sampled_rows = 0
    for mid in training_ids:
        support = metadata['support'][str(mid)]; counts = support['sampled_class_counts']
        if (set(counts) != set('HAD') or any(type(n) is not int or n < 0 for n in counts.values())
            or type(support['sampled_rows']) is not int or support['sampled_rows'] <= 0
            or support['sampled_rows'] != sum(counts.values())
            or support['sampled_fixture_mass_before_global_normalization'] != 1.
            or any(counts[c] > fit['training_reference_support'][str(mid)]['class_counts'][c] for c in counts)):
            raise ValueError('Context sampled training support is inconsistent')
        sampled_rows += support['sampled_rows']
    training = fit['training']
    if (training['epochs'] != campaign.correction.POLICY['epochs'] or training['seed'] != campaign.correction.POLICY['seed']
        or training['sampled_rows'] != sampled_rows or training['cpu_threads'] != 1
        or training['conditional_home_away_trained'] is not False
        or len(training['losses']) != training['epochs']
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in training['losses'])
        or not isinstance(training['torch_version'], str) or not training['torch_version']):
        raise ValueError('Context fitting diagnostics differ from the fixed policy')


def _verify_local_training(bundle, out):
    fit = bundle['fit']; head_path = out / 'context-head.json'
    if sha256(head_path) != fit['head_sha256']:
        raise ValueError('Local context fitted weights changed')
    head = campaign.correction.validate_head(read(head_path))
    if head['metadata'] != fit['head_metadata'] or head['training'] != fit['training']:
        raise ValueError('Local context head provenance differs')
    if (sha256(out / 'training-plan.json') != fit['training_plan_sha256']
        or sha256(out / 'fit-access.json') != fit['fit_access_sha256']
        or read(out / 'training-probabilities/complete.json') != fit['training_probability_complete']
        or sha256(out / 'training-probabilities/complete.json') != bundle['fit_access']['training_probabilities_sha256']):
        raise ValueError('Local pre-fit plan or probability completion bytes differ')
    parent = read(ROOT / 'models/pff-research.json')
    old_manifests = {m['match_id']: m for m in parent['cache_manifests']}
    for record, manifest in zip(fit['training_probability_complete']['records'], fit['training_probability_manifests']):
        mid = record['match_id']; folder = out / 'training-probabilities' / str(mid)
        if (sha256(folder / 'manifest.json') != record['manifest_sha256']
            or read(folder / 'manifest.json') != manifest
            or sha256(folder / 'parent_probabilities.npy') != record['probability_sha256']):
            raise ValueError('Local training probability bytes changed')
        cache = campaign.PARENT / 'cache' / str(mid)
        if sha256(cache / 'complete.json') != manifest['identity']['source_cache_manifest_sha256']:
            raise ValueError('Local training source manifest changed')
        for name, expected in old_manifests[mid]['files'].items():
            if sha256(cache / name) != expected:
                raise ValueError('Local training source observation bytes changed')


def bundle_from_run(out):
    out = Path(out); study = read(out / 'protocol.json')
    bundle = {'schema_version': 1, 'protocol': study,
              'parent_research_sha256': sha256(ROOT / 'models/pff-research.json'),
              'corrections': {r['candidate_id']: read(out / 'corrections' / (r['candidate_id'] + '.json'))
                              for r in study['candidates']},
              'final_preparation_manifests': [read(p) for p in sorted((out / 'final-observations/cache').glob('*/complete.json'))],
              'final_artifacts_present': any((out / name).exists() for name in (
                  'final', 'final-observations', 'final-reference-access.json',
                  'final-predictions-before-references.json', 'final-summary.json')),
              'weights_included': False, 'promoted': False,
              'retained_model_sha256': original.BASELINE_SHA256,
              'training_plan': read(out / 'training-plan.json'), 'fit': read(out / 'fit.json'),
              'fit_access': read(out / 'fit-access.json')}
    for part in ('development', 'final'):
        summary = out / (part + '-summary.json')
        bundle[part] = None if not summary.exists() else {
            'summary': read(summary),
            'prediction_freeze': read(out / (part + '-predictions-before-references.json')),
            'reference_access': read(out / (part + '-reference-access.json'))}
    return bundle


def _metrics(detail):
    values = campaign.prior.metrics(detail)
    values['short_recall'] = values.pop('short')
    return values


def _details(summary, mids, declarations):
    names = {'baseline', 'parent', *(r['candidate_id'] for r in declarations)}
    expected = {(str(mid), name) for mid in mids for name in names}
    details = {}
    for d in summary['details']:
        key = d['match'], d['candidate']
        if key not in expected or key in details or d['regime'] != 'native' or not original._sha(d['prediction_sha256']):
            raise ValueError('Detailed diagnostics have duplicate, missing or foreign identities')
        details[key] = d
    if set(details) != expected:
        raise ValueError('Every model needs diagnostic details on the entire cohort')
    groups = {'parent': summary['parent_records'], **summary['records']}
    hashes = {'parent': validation.PARENT_SHA256, **{r['candidate_id']: r['artifact_sha256'] for r in declarations}}
    if set(groups) != names - {'baseline'}:
        raise ValueError('Metric record candidate coverage differs')
    for name, records in groups.items():
        if [r['match_id'] for r in records] != mids:
            raise ValueError('Metric records must cover the exact ordered cohort')
        for row in records:
            if row['candidate_sha256'] != hashes[name] or row['baseline_sha256'] != original.BASELINE_SHA256:
                raise ValueError('Metric record model identity differs')
            for side, detail_name in (('baseline', 'baseline'), ('candidate', name)):
                detail = details[str(row['match_id']), detail_name]
                if _metrics(detail) != row[side]:
                    raise ValueError('Metric records differ from saved diagnostics')
                for field, key in (('reference_dead_segments', 'dead_segments'), ('reference_short_spells', 'short_possessions')):
                    if row[field] != detail['temporal_diagnostics'][key]['n_gt']:
                        raise ValueError('Diagnostic reference support differs')


def verify_bundle(bundle):
    parent, parent_hash = _parent_context()
    reservation = parent['protocol']; old = parent['development']['summary']['records']
    study = validation.verify_protocol(bundle['protocol'], reservation, old)
    if (bundle['schema_version'] != 1 or bundle['parent_research_sha256'] != parent_hash
        or bundle['weights_included'] is not False or bundle['promoted'] is not False
        or bundle['retained_model_sha256'] != original.BASELINE_SHA256
        or sha256(ROOT / 'models/submission.json') != original.BASELINE_SHA256
        or study['source_hashes'] != campaign.sources()):
        raise ValueError('Context research source, parent or deployment identity differs')
    if set(bundle['corrections']) != {r['candidate_id'] for r in study['candidates']}:
        raise ValueError('Correction artifact coverage differs')
    for declared in study['candidates']:
        artifact = bundle['corrections'][declared['candidate_id']]
        expected = {'schema_version': 1, 'parent_sha256': validation.PARENT_SHA256,
                    'head_sha256': study['head_sha256'], 'parameters': declared['parameters'], 'policy': campaign.correction.POLICY}
        if artifact != expected or declared['artifact_sha256'] not in parent_verifier._serialized_hashes(artifact):
            raise ValueError('Correction artifact differs from frozen declaration')
    _verify_training(bundle, parent, reservation, study)
    if bundle['development'] is None:
        raise ValueError('Completed development selection is required')
    if bundle['final_artifacts_present'] is not (bundle['final'] is not None):
        raise ValueError('Partial or unreported final activity cannot be omitted')
    prior_manifests = {r['match_id']: r for r in parent['cache_manifests']}
    parent_predictions = {r['match_id']: r for r in parent['development']['prediction_freeze']['records']}
    final_manifests = {}
    for manifest in bundle['final_preparation_manifests']:
        mid = manifest['match_id']
        if mid in final_manifests or mid not in reservation['split']['final']:
            raise ValueError('Duplicate or foreign final preparation')
        identity = manifest['identity']
        if (identity['sources'] != campaign.prior.sources()
            or identity['metadata'] != reservation['metadata_sha256'][str(mid)]
            or not original._sha(identity['events']) or not original._sha(identity['tracking'])
            or set(manifest['files']) != parent_verifier.CACHE_FILES
            or not all(original._sha(h) for h in manifest['files'].values())
            or type(manifest['rows']) is not int or manifest['rows'] <= 0):
            raise ValueError('Final observation provenance differs')
        final_manifests[mid] = manifest
    if set(final_manifests) != (set(reservation['split']['final']) if bundle['final'] else set()):
        raise ValueError('Unreported or incomplete final preparation')
    selection = None
    for part in ('development', 'final'):
        block = bundle[part]
        if block is None:
            continue
        if part == 'final' and (selection is None or not selection['eligible_for_final_audit']):
            raise ValueError('Final data accessed after failed development selection')
        summary, frozen, marker = block['summary'], block['prediction_freeze'], block['reference_access']
        declarations = study['candidates'] if part == 'development' else [d for d in study['candidates'] if d['candidate_id'] == selection['selected_candidate_id']]
        mids = sorted(reservation['split'][part])
        if (summary['part'] != part or summary['protocol_sha256'] != study['protocol_sha256']
            or summary['source_hashes'] != study['source_hashes'] or summary['promoted'] is not False
            or summary['retained_model_sha256'] != original.BASELINE_SHA256
            or frozen['part'] != part or frozen['candidates'] != declarations
            or frozen['protocol_sha256'] != study['protocol_sha256'] or frozen['source_hashes'] != study['source_hashes']
            or frozen['references_accessed_in_this_study'] is not False
            or frozen['development_previously_consumed'] is not (part == 'development')
            or [r['match_id'] for r in frozen['records']] != mids
            or summary['prediction_freeze_sha256'] not in parent_verifier._serialized_hashes(frozen)):
            raise ValueError('Frozen cohort prediction or summary provenance differs')
        expected_marker = {'part': part, 'prediction_freeze_sha256': summary['prediction_freeze_sha256'],
                           'protocol_sha256': study['protocol_sha256'], 'source_hashes': study['source_hashes'],
                           'development_previously_consumed': part == 'development'}
        if marker != expected_marker or summary['reference_access_sha256'] not in parent_verifier._serialized_hashes(marker, False):
            raise ValueError('Exclusive reference-access record differs')
        evidence = {'dense_truth_validated': False, 'independent_observations': True, 'identity_checked': True,
                    'usage_permitted': False, 'development_passed': part == 'final', 'observations_only': True,
                    'predictions_frozen_before_references': True, 'adequate_label_support': False}
        if summary['evidence'] != evidence:
            raise ValueError('Silver-reference status or usage rights overstated')
        names = {'baseline', 'parent', *(r['candidate_id'] for r in declarations)}
        if set(summary['reference_support']) != {str(mid) for mid in mids}:
            raise ValueError('Reference support cohort differs')
        for row in frozen['records']:
            mid = row['match_id']; manifest = row['observation_manifest']
            expected_manifest = prior_manifests[mid] if part == 'development' else final_manifests[mid]
            hashes = row['prediction_sha256']
            if (manifest != expected_manifest or set(hashes) != names
                or not all(original._sha(h) for h in hashes.values())
                or hashes['baseline'] != manifest['files']['baseline.npy']):
                raise ValueError('Prediction or observation manifest identity differs')
            if part == 'development':
                if (hashes['parent'] != parent_predictions[mid]['candidate_prediction_sha256']
                    or hashes[study['candidates'][0]['candidate_id']] != hashes['parent']):
                    raise ValueError('Zero correction or parent predictions changed')
            support = summary['reference_support'][str(mid)]
            parent_verifier._support(support)
            if support['event_sha256'] != manifest['identity']['events'] or sum(support['class_counts'].values()) != manifest['rows']:
                raise ValueError('Reference support or source differs')
        _details(summary, mids, declarations)
        if part == 'development':
            if summary['parent_records'] != old:
                raise ValueError('Development parent metrics changed')
            selection = validation.select_candidate(study, summary['records'], old, reservation)
            if summary['selection'] != selection:
                raise ValueError('Development selection does not reproduce')
        else:
            chosen = selection['selected_candidate_id']
            gate = validation.final_gate(summary['records'][chosen], summary['parent_records'], reservation, evidence, selection)
            if summary['gate'] != gate or gate['promoted'] is not False:
                raise ValueError('Final gate differs or cannot authorize publication')
    return {'development_gate_passed': selection['eligible_for_final_audit'],
            'selected_candidate_id': selection['selected_candidate_id'],
            'final_evaluated': bundle['final'] is not None, 'promoted': False,
            'aggregate_gates_recomputed': True, 'local_artifact_bytes_verified': False}


def verify_local(bundle, out):
    result = verify_bundle(bundle); out = Path(out)
    if bundle_from_run(out) != bundle or sha256(campaign.PARENT / 'candidate.json') != validation.PARENT_SHA256:
        raise ValueError('Local evidence or parent candidate bytes differ')
    _verify_local_training(bundle, out)
    for declared in bundle['protocol']['candidates']:
        if sha256(out / 'corrections' / (declared['candidate_id'] + '.json')) != declared['artifact_sha256']:
            raise ValueError('Local correction artifact changed')
    for part in ('development', 'final'):
        block = bundle[part]
        if block is None:
            continue
        for row in block['prediction_freeze']['records']:
            mid = row['match_id']; cache = campaign.observation_root(out, part) / 'cache' / str(mid)
            for name, expected in row['observation_manifest']['files'].items():
                if sha256(cache / name) != expected:
                    raise ValueError('Local observation cache bytes changed')
            for name, expected in row['prediction_sha256'].items():
                if sha256(out / part / str(mid) / (name + '.npy')) != expected:
                    raise ValueError('Local prediction bytes changed')
        for detail in block['summary']['details']:
            path = out / part / detail['match'] / detail['candidate'] / 'predictions.csv'
            if sha256(path) != detail['prediction_sha256']:
                raise ValueError('Scored prediction CSV bytes changed')
    return dict(result, local_artifact_bytes_verified=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT / 'models/pff-context-research.json')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--write', type=Path)
    args = parser.parse_args()
    if args.write and not args.run:
        parser.error('--write requires --run')
    bundle = bundle_from_run(args.run) if args.run else read(args.evidence)
    result = verify_local(bundle, args.run) if args.run else verify_bundle(bundle)
    if args.write:
        write_json(args.write, bundle)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
