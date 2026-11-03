"""Authenticate aggregate PFF precision research without publishing weights."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from tools import pff_precision_campaign as campaign, pff_precision_validation as validation
from tools import pff_validation as original, verify_pff_research as parent_verifier


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _parent_context():
    path = ROOT / 'models/pff-research.json'
    parent = read(path)
    parent_verifier.verify_bundle(parent)
    if parent['fit']['candidate_sha256'] != validation.PARENT_SHA256:
        raise ValueError('Published parent identity differs')
    return parent, sha256(path)


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
              'retained_model_sha256': original.BASELINE_SHA256}
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
        raise ValueError('Precision research source, parent or deployment identity differs')
    if set(bundle['corrections']) != {r['candidate_id'] for r in study['candidates']}:
        raise ValueError('Correction artifact coverage differs')
    for declared in study['candidates']:
        artifact = bundle['corrections'][declared['candidate_id']]
        expected = {'schema_version': 1, 'parent_sha256': validation.PARENT_SHA256,
                    'parameters': declared['parameters'], 'policy': campaign.correction.POLICY}
        if artifact != expected or declared['artifact_sha256'] not in parent_verifier._serialized_hashes(artifact):
            raise ValueError('Correction artifact differs from frozen declaration')
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
    parser.add_argument('--evidence', type=Path, default=ROOT / 'models/pff-precision-research.json')
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
