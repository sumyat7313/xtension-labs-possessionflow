"""Verify published aggregate PFF evidence; optionally authenticate local bytes.

This verifier derives metric records from detailed diagnostics and recomputes
gates, not predictions or reference labels. It reads no
provider event records and never changes the deployed model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from tools import pff_campaign as campaign, pff_training, pff_adapter, pff_validation as validation
from tools.train_temporal_v6 import CONFIG_SHA256
from tools import pff_migrate_preparation as migration

CACHE_FILES = {'x.npy', 'z.npy', 'base_home.npy', 'base_dead.npy', 'baseline.npy', 'item.json', 'alignment.json'}


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _serialized_hashes(value, terminal_newline=True):
    # Published evidence preserves file hashes from Windows. Accept canonical
    # JSON encoded with LF or CRLF so verification also runs on Linux.
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + ('\n' if terminal_newline else '')
    return {hashlib.sha256(text.encode()).hexdigest(), hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest()}


def _support(value):
    if (value['reference_quality'] != 'provider_event_derived_silver'
        or value['dense_human_had_truth'] is not False
        or value['promotion_eligible_by_labels_alone'] is not False):
        raise ValueError('PFF reference quality was overstated')
    counts = value['class_counts']
    if set(counts) != set('HADX') or any(type(n) is not int or n < 0 for n in counts.values()) or not sum(counts.values()):
        raise ValueError('Invalid reference class support')


def _verify_details(summary, match_ids, candidate):
    """Bind one pair of detailed diagnostics to each declared metric record."""
    expected = {(str(mid), which) for mid in match_ids for which in ('baseline', 'candidate')}
    details = {}
    for detail in summary['details']:
        key = (detail['match'], detail['candidate'])
        if key not in expected or key in details or detail['regime'] != 'native':
            raise ValueError('Duplicate, missing, or wrong-cohort detailed diagnostics')
        details[key] = detail
    if set(details) != expected:
        raise ValueError('Detailed diagnostics must cover both models on every fixture')
    for row in summary['records']:
        if row['candidate_sha256'] != candidate or row['candidate_sha256'] != summary['candidate_sha256']:
            raise ValueError('Metric record candidate differs from fitted model')
        for which in ('baseline', 'candidate'):
            detail = details[str(row['match_id']), which]
            values = campaign.metrics(detail)
            values['short_recall'] = values.pop('short')
            if values != row[which]:
                raise ValueError('Metric record differs from detailed diagnostics')
            diagnostics = detail['temporal_diagnostics']
            for field, section in (('reference_dead_segments', 'dead_segments'),
                                   ('reference_short_spells', 'short_possessions')):
                if diagnostics[section]['n_gt'] != row[field]:
                    raise ValueError('Metric reference support differs from detailed diagnostics')


def bundle_from_run(out):
    out = Path(out)
    bundle = {'schema_version': 1, 'protocol': _load(ROOT / 'models/pff-protocol.json'),
              'identity_audit': _load(ROOT / 'models/pff-identity-audit.json'),
              'run_protocol': _load(out / 'run-protocol.json'), 'fit': _load(out / 'fit.json'),
              'cache_manifests': [_load(p) for p in sorted((out / 'cache').glob('*/complete.json'))],
              'weights_included': False, 'promoted': False,
              'retained_model_sha256': validation.BASELINE_SHA256}
    proof = out / 'preparation-migration.json'
    if proof.exists():
        bundle['preparation_migration'] = _load(proof)
        bundle['preparation_migration_completion'] = _load(out / 'preparation-migration-complete.json')
    for part in ('development', 'final'):
        path = out / (part + '-summary.json')
        if not path.exists():
            bundle[part] = None
        else:
            bundle[part] = {'summary': _load(path),
                            'prediction_freeze': _load(out / (part + '-predictions-before-references.json')),
                            'reference_access': _load(out / (part + '-reference-access.json'))}
    return bundle


def verify_bundle(bundle):
    if sha256(ROOT / 'configs/baselines/v1.5.json') != CONFIG_SHA256:
        raise ValueError('Frozen v1.5 decoder config changed')
    if (bundle['schema_version'] != 1 or bundle['weights_included'] is not False
        or bundle['promoted'] is not False or bundle['retained_model_sha256'] != validation.BASELINE_SHA256):
        raise ValueError('PFF research must retain v1.5 and exclude fitted weights')
    p = validation.verify_protocol(bundle['protocol'])
    if p != _load(ROOT / 'models/pff-protocol.json'):
        raise ValueError('Published reservation differs')
    if p['identity_audit_sha256'] not in _serialized_hashes(bundle['identity_audit']):
        raise ValueError('Identity audit differs')
    sources = campaign.sources()
    migration_proof = bundle.get('preparation_migration')
    if migration_proof is not None:
        migration.verify_proof(migration_proof, sources, p,
            (ROOT / migration.ADAPTER).read_bytes().decode('utf-8'))
        completion = bundle['preparation_migration_completion']
        if (completion['proof_sha256'] not in migration.serialized_hashes(migration_proof)
            or completion['copied_matches'] != len(migration_proof['match_ids'])
            or completion['copied_artifacts'] != len(migration_proof['match_ids']) * len(CACHE_FILES)
            or completion['original_run_unchanged'] is not True or completion['labels_read'] is not False
            or completion['predictions_recomputed'] is not False):
            raise ValueError('Preparation migration completion differs')
    elif bundle.get('preparation_migration_completion') is not None:
        raise ValueError('Preparation migration proof missing')
    run = bundle['run_protocol']
    if (run['split_protocol_sha256'] != p['protocol_sha256'] or run['source_hashes'] != sources
        or run['training_policy'] != pff_training.POLICY or run['observation_policy'] != pff_adapter.POLICY
        or run['raw_data_and_fitted_weights_local_only'] is not True or run['automatic_promotion'] is not False):
        raise ValueError('Frozen PFF policies or sources differ')
    fit = bundle['fit']
    expected_train = ['pff' + str(i) for i in p['split']['train']] + ['sportec' + i for i in sorted(campaign.CATALOG)]
    candidate = fit['candidate_sha256']
    if (not validation._sha(candidate) or fit['training_matches'] != expected_train
        or fit['source_hashes'] != sources or fit['held_out_references_used'] is not False
        or set(fit['training_reference_support']) != {str(i) for i in p['split']['train']}):
        raise ValueError('PFF fit provenance or training membership differs')
    manifests = {}
    inherited_ids = set()
    for value in bundle['cache_manifests']:
        mid = value['match_id']
        if type(mid) is not int or mid in manifests or mid not in validation.EXPECTED_IDS:
            raise ValueError('Duplicate or unexpected PFF cache manifest')
        identity = value['identity']
        if (identity['sources'] != sources or identity['metadata'] != p['metadata_sha256'][str(mid)]
            or not validation._sha(identity['tracking']) or not validation._sha(identity['events'])
            or set(value['files']) != CACHE_FILES or not all(validation._sha(h) for h in value['files'].values())):
            raise ValueError('PFF cache identity differs')
        if type(value['rows']) is not int or value['rows'] < 1:
            raise ValueError('PFF cache rows invalid')
        if 'inherited_preparation' in value:
            if migration_proof is None:
                raise ValueError('Inherited preparation requires its original code proof')
            migration.verify_inheritance(value, migration_proof, sources, p)
            inherited_ids.add(mid)
        manifests[mid] = value
    if migration_proof is not None and inherited_ids != set(migration_proof['match_ids']):
        raise ValueError('Inherited preparation cohort differs from migration proof')
    needed = set(p['split']['train']) | set(p['split']['development'])
    if bundle['final'] is not None:
        needed.update(p['split']['final'])
    if set(manifests) != needed:
        raise ValueError('Prepared cache set differs from declared reference access')
    for mid in p['split']['train']:
        support = fit['training_reference_support'][str(mid)]
        _support(support)
        if support['event_sha256'] != manifests[mid]['identity']['events'] or sum(support['class_counts'].values()) != manifests[mid]['rows']:
            raise ValueError('Training reference source/support differs')
    if bundle['development'] is None:
        raise ValueError('Completed development evaluation is required')
    verified = {}
    for part in ('development', 'final'):
        block = bundle[part]
        if block is None:
            continue
        if part == 'final' and not verified['development']['eligible_for_final_audit']:
            raise ValueError('Final references accessed after failed development')
        summary, frozen, marker = (block[k] for k in ('summary', 'prediction_freeze', 'reference_access'))
        if (summary['part'] != part or summary['source_hashes'] != sources or summary['protocol_sha256'] != p['protocol_sha256']
            or summary['candidate_sha256'] != candidate or summary['promoted'] is not False
            or summary['retained_model_sha256'] != validation.BASELINE_SHA256
            or summary['reference_quality'] != 'provider event-derived silver; no independent dense human validation'):
            raise ValueError('PFF summary identity or reference claims differ')
        expected_evidence = {'dense_truth_validated': False, 'independent_observations': True, 'identity_checked': True,
            'usage_permitted': False, 'development_passed': part == 'final', 'observations_only': True,
            'predictions_frozen_before_references': True, 'adequate_label_support': False}
        if summary['evidence'] != expected_evidence:
            raise ValueError('PFF evidence claims differ from the research scope')
        _verify_details(summary, p['split'][part], candidate)
        gate = (validation.development_gate(summary['records'], p) if part == 'development'
                else validation.promotion_gate(summary['records'], p, summary['evidence']))
        if summary['gate'] != gate or gate['promoted'] is not False:
            raise ValueError('PFF gate does not reproduce')
        if (frozen['part'] != part or frozen['candidate_sha256'] != candidate
            or frozen['baseline_sha256'] != validation.BASELINE_SHA256
            or frozen['split_protocol_sha256'] != p['protocol_sha256'] or frozen['source_hashes'] != sources
            or frozen['references_accessed'] is not False
            or [r['match_id'] for r in frozen['records']] != p['split'][part]
            or summary['prediction_freeze_sha256'] not in _serialized_hashes(frozen)):
            raise ValueError('PFF prediction freeze differs')
        for row in frozen['records']:
            if any(not validation._sha(row[k]) for k in ('candidate_prediction_sha256', 'baseline_prediction_sha256')):
                raise ValueError('Invalid prediction identity')
            if row['baseline_prediction_sha256'] != manifests[row['match_id']]['files']['baseline.npy']:
                raise ValueError('Frozen comparator differs from observation cache')
        if (marker['part'] != part or marker['references_consumed'] is not True
            or marker['source_hashes'] != sources or marker['candidate_sha256'] != candidate
            or marker['protocol_sha256'] != p['protocol_sha256']
            or marker['prediction_freeze_sha256'] != summary['prediction_freeze_sha256']
            or summary['reference_access_sha256'] not in _serialized_hashes(marker, False)):
            raise ValueError('Exclusive reference-access evidence differs')
        if set(summary['reference_support']) != {str(i) for i in p['split'][part]}:
            raise ValueError('Reference support does not cover exact evaluation cohort')
        for mid in p['split'][part]:
            support = summary['reference_support'][str(mid)]
            _support(support)
            if support['event_sha256'] != manifests[mid]['identity']['events'] or sum(support['class_counts'].values()) != manifests[mid]['rows']:
                raise ValueError('Evaluated reference source/support differs')
        verified[part] = gate
    if sha256(ROOT / 'models/submission.json') != validation.BASELINE_SHA256:
        raise ValueError('Deployed v1.5 changed')
    return {'development_gate_passed': verified['development']['eligible_for_final_audit'],
            'final_evaluated': 'final' in verified, 'promoted': False,
            'local_model_bytes_verified': False, 'aggregate_gates_recomputed': True}


def verify_local(bundle, out):
    """Authenticate cached/prediction/model bytes without reading event records."""
    result = verify_bundle(bundle)
    out = Path(out)
    if bundle_from_run(out) != bundle:
        raise ValueError('Local evidence differs from published bundle')
    if sha256(out / 'candidate.json') != bundle['fit']['candidate_sha256']:
        raise ValueError('Local candidate bytes changed')
    candidate = _load(out / 'candidate.json')
    baseline = pff_training.baseline_model()
    from possession.learned import validate_model
    validate_model(candidate)
    for key in baseline:
        if key not in ('joint_model', 'metadata') and candidate.get(key) != baseline[key]:
            raise ValueError('Experimental fit changed frozen upstream fields')
    meta = candidate['metadata']['pff_research']
    if (meta['training_matches'] != sorted(bundle['fit']['training_matches'])
        or meta['held_out_matches'] != sorted('pff' + str(i) for part in ('development', 'final') for i in bundle['protocol']['split'][part])
        or meta['protocol_sha256'] != bundle['protocol']['protocol_sha256'] or meta['policy'] != pff_training.POLICY):
        raise ValueError('Local fitted model provenance differs')
    for manifest in bundle['cache_manifests']:
        folder = out / 'cache' / str(manifest['match_id'])
        for name, expected in manifest['files'].items():
            if sha256(folder / name) != expected:
                raise ValueError('Local cache bytes changed')
    for part in ('development', 'final'):
        block = bundle[part]
        if block is not None:
            for row in block['prediction_freeze']['records']:
                for which in ('baseline', 'candidate'):
                    if sha256(out / part / str(row['match_id']) / (which + '.npy')) != row[which + '_prediction_sha256']:
                        raise ValueError('Local prediction bytes changed')
    return dict(result, local_model_bytes_verified=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT / 'models/pff-research.json')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--write', type=Path, help='Write a verified aggregate bundle from --run')
    args = parser.parse_args()
    if args.write and not args.run:
        parser.error('--write requires --run')
    bundle = bundle_from_run(args.run) if args.run else _load(args.evidence)
    result = verify_local(bundle, args.run) if args.run else verify_bundle(bundle)
    if args.write:
        write_json(args.write, bundle)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
