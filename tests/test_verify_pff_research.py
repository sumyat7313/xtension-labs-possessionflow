import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from tools import verify_pff_research as verifier, pff_validation as validation


def evidence_bundle():
    p = verifier._load(verifier.ROOT / 'models/pff-protocol.json')
    sources = {'test': 'a' * 64}
    candidate = 'c' * 64
    def support():
        return {'reference_quality': 'provider_event_derived_silver', 'dense_human_had_truth': False,
                'promotion_eligible_by_labels_alone': False, 'class_counts': dict.fromkeys('HAD', 100) | {'X': 0},
                'event_sha256': 'd' * 64}
    manifests = [{'match_id': i, 'identity': {'sources': sources, 'metadata': p['metadata_sha256'][str(i)],
                    'tracking': 'e' * 64, 'events': 'd' * 64}, 'rows': 300,
                  'files': dict.fromkeys(verifier.CACHE_FILES, 'f' * 64)}
                 for part in ('train', 'development') for i in p['split'][part]]
    rows = [{'match_id': i, 'condition': 'native', 'baseline_sha256': validation.BASELINE_SHA256,
             'candidate_sha256': candidate, 'baseline': dict.fromkeys(validation.METRICS, .7),
             'candidate': dict.fromkeys(validation.METRICS, .7),
             'reference_dead_segments': 100, 'reference_short_spells': 100} for i in p['split']['development']]
    details = [{'match': str(i), 'candidate': which, 'regime': 'native',
                'spellF1': .7, 'macroF1': .7, 'dead_recall': .7, 'dead_precision': .7,
                'temporal_diagnostics': {'dead_segments': {'f1': .7, 'n_gt': 100},
                    'dead_onsets': {'f1': .7}, 'dead_offsets': {'f1': .7},
                    'short_possessions': {'recall': .7, 'n_gt': 100}},
                'duration_precision_recall': {'bins': {k: {'n_pred': 10, 'n_matched_pred': 7}
                    for k in ('<0.4s', '0.4-1s', '1-2s')}}}
               for i in p['split']['development'] for which in ('baseline', 'candidate')]
    frozen = {'part': 'development', 'candidate_sha256': candidate, 'baseline_sha256': validation.BASELINE_SHA256,
              'split_protocol_sha256': p['protocol_sha256'], 'source_hashes': sources, 'references_accessed': False,
              'records': [{'match_id': i, 'candidate_prediction_sha256': 'b' * 64, 'baseline_prediction_sha256': 'f' * 64}
                          for i in p['split']['development']]}
    freeze_hash = min(verifier._serialized_hashes(frozen))
    marker = {'part': 'development', 'references_consumed': True, 'prediction_freeze_sha256': freeze_hash,
              'candidate_sha256': candidate, 'protocol_sha256': p['protocol_sha256'], 'source_hashes': sources}
    summary = {'part': 'development', 'source_hashes': sources, 'protocol_sha256': p['protocol_sha256'],
               'candidate_sha256': candidate, 'promoted': False, 'retained_model_sha256': validation.BASELINE_SHA256,
               'reference_quality': 'provider event-derived silver; no independent dense human validation',
               'evidence': {'dense_truth_validated': False, 'independent_observations': True, 'identity_checked': True,
                  'usage_permitted': False, 'development_passed': False, 'observations_only': True,
                  'predictions_frozen_before_references': True, 'adequate_label_support': False},
               'records': rows, 'details': details, 'gate': validation.development_gate(rows, p),
               'prediction_freeze_sha256': freeze_hash, 'reference_access_sha256': min(verifier._serialized_hashes(marker, False)),
               'reference_support': {str(i): support() for i in p['split']['development']}}
    return {'schema_version': 1, 'protocol': p,
            'identity_audit': verifier._load(verifier.ROOT / 'models/pff-identity-audit.json'),
            'run_protocol': {'split_protocol_sha256': p['protocol_sha256'], 'source_hashes': sources,
                'training_policy': verifier.pff_training.POLICY, 'observation_policy': verifier.pff_adapter.POLICY,
                'raw_data_and_fitted_weights_local_only': True, 'automatic_promotion': False},
            'fit': {'candidate_sha256': candidate,
                'training_matches': ['pff' + str(i) for i in p['split']['train']] + ['sportec' + i for i in sorted(verifier.campaign.CATALOG)],
                'source_hashes': sources, 'held_out_references_used': False,
                'training_reference_support': {str(i): support() for i in p['split']['train']}},
            'cache_manifests': manifests, 'weights_included': False, 'promoted': False,
            'retained_model_sha256': validation.BASELINE_SHA256,
            'development': {'summary': summary, 'prediction_freeze': frozen, 'reference_access': marker}, 'final': None}


def attach_migration(bundle):
    migration = verifier.migration
    source = (verifier.ROOT / migration.ADAPTER).read_bytes().decode('utf-8')
    # A whitespace-only adapter revision is sufficient to exercise portable
    # provenance; separate migration tests exercise semantic body-only changes.
    old_source = source + '\n'
    current = bundle['run_protocol']['source_hashes']
    current[migration.ADAPTER] = hashlib.sha256(source.encode()).hexdigest()
    old_sources = dict(current, **{migration.ADAPTER: hashlib.sha256(old_source.encode()).hexdigest()})
    old_run = copy.deepcopy(bundle['run_protocol'])
    old_run['source_hashes'] = old_sources
    proof = {'schema_version': 1, 'method': 'copy authenticated observation artifacts',
        'original_run_protocol': old_run, 'original_run_protocol_sha256': min(migration.serialized_hashes(old_run)),
        'original_adapter_source': old_source, 'new_source_hashes': current,
        'observation_ast_sha256': migration.observation_ast_digest(source),
        'decoder_config_sha256': verifier.CONFIG_SHA256,
        'match_ids': sorted(bundle['protocol']['split']['train'] + bundle['protocol']['split']['development']),
        'candidate_or_fit_existed': False, 'held_out_access_existed': False}
    for manifest in bundle['cache_manifests']:
        original = copy.deepcopy(manifest)
        original['identity']['sources'] = old_sources
        manifest['inherited_preparation'] = {'original_manifest': original,
            'original_manifest_sha256': min(migration.serialized_hashes(original)),
            'original_run_protocol_sha256': proof['original_run_protocol_sha256'],
            'observation_ast_sha256': proof['observation_ast_sha256'],
            'old_adapter_sha256': old_sources[migration.ADAPTER], 'new_adapter_sha256': current[migration.ADAPTER]}
    bundle['preparation_migration'] = proof
    bundle['preparation_migration_completion'] = {'proof_sha256': min(migration.serialized_hashes(proof)),
        'copied_matches': 52, 'copied_artifacts': 364, 'original_run_unchanged': True,
        'labels_read': False, 'predictions_recomputed': False}
    block = bundle['development']
    freeze_hash = min(verifier._serialized_hashes(block['prediction_freeze']))
    block['summary']['prediction_freeze_sha256'] = freeze_hash
    block['reference_access']['prediction_freeze_sha256'] = freeze_hash
    block['summary']['reference_access_sha256'] = min(verifier._serialized_hashes(block['reference_access'], False))
    return current


class VerifyPFFResearchTests(unittest.TestCase):
    def setUp(self):
        self.sources = patch.object(verifier.campaign, 'sources', return_value={'test': 'a' * 64})
        self.sources.start()
        self.addCleanup(self.sources.stop)

    def test_complete_research_bundle_retains_baseline(self):
        result = verifier.verify_bundle(evidence_bundle())
        self.assertFalse(result['promoted'])
        self.assertFalse(result['development_gate_passed'])
        self.assertFalse(result['final_evaluated'])
        self.assertFalse(result['local_model_bytes_verified'])

    def test_aggregate_gate_tamper_detected(self):
        bundle = evidence_bundle()
        bundle['development']['summary']['gate']['eligible_for_final_audit'] = True
        with self.assertRaises(ValueError):
            verifier.verify_bundle(bundle)

    def test_decoder_config_drift_rejected(self):
        bundle = evidence_bundle()
        original = verifier.sha256
        def changed_config(path):
            return '0' * 64 if path == verifier.ROOT / 'configs/baselines/v1.5.json' else original(path)
        with patch.object(verifier, 'sha256', side_effect=changed_config):
            with self.assertRaisesRegex(ValueError, 'decoder config changed'):
                verifier.verify_bundle(bundle)

    def test_consistent_foreign_candidate_records_rejected(self):
        bundle = evidence_bundle()
        summary = bundle['development']['summary']
        for row in summary['records']:
            row['candidate_sha256'] = '9' * 64
        summary['gate'] = validation.development_gate(summary['records'], bundle['protocol'])
        with self.assertRaisesRegex(ValueError, 'candidate differs from fitted model'):
            verifier.verify_bundle(bundle)

    def test_recomputed_gate_cannot_hide_metric_tampering(self):
        for which in ('baseline', 'candidate'):
            for metric in validation.METRICS:
                with self.subTest(which=which, metric=metric):
                    bundle = evidence_bundle()
                    summary = bundle['development']['summary']
                    summary['records'][0][which][metric] = .75
                    summary['gate'] = validation.development_gate(summary['records'], bundle['protocol'])
                    with self.assertRaisesRegex(ValueError, 'differs from detailed diagnostics'):
                        verifier.verify_bundle(bundle)

    def test_detail_metric_tampering_rejected(self):
        bundle = evidence_bundle()
        bundle['development']['summary']['details'][0]['duration_precision_recall']['bins']['<0.4s']['n_matched_pred'] = 8
        with self.assertRaisesRegex(ValueError, 'differs from detailed diagnostics'):
            verifier.verify_bundle(bundle)

    def test_details_require_exact_model_fixture_pairs(self):
        for kind in ('missing', 'duplicate', 'foreign', 'regime', 'model'):
            with self.subTest(kind=kind):
                bundle = evidence_bundle()
                details = bundle['development']['summary']['details']
                if kind == 'missing':
                    details.pop()
                elif kind == 'duplicate':
                    details[-1] = copy.deepcopy(details[0])
                elif kind == 'foreign':
                    details[0]['match'] = str(bundle['protocol']['split']['final'][0])
                elif kind == 'regime':
                    details[0]['regime'] = 'degraded'
                else:
                    details[0]['candidate'] = 'other'
                with self.assertRaises(ValueError):
                    verifier.verify_bundle(bundle)

    def test_both_model_details_must_share_reference_support(self):
        for which in (0, 1):
            for section in ('dead_segments', 'short_possessions'):
                with self.subTest(which=which, section=section):
                    bundle = evidence_bundle()
                    bundle['development']['summary']['details'][which]['temporal_diagnostics'][section]['n_gt'] += 1
                    with self.assertRaisesRegex(ValueError, 'reference support differs'):
                        verifier.verify_bundle(bundle)

    def test_detail_order_is_irrelevant(self):
        bundle = evidence_bundle()
        bundle['development']['summary']['details'].reverse()
        self.assertFalse(verifier.verify_bundle(bundle)['promoted'])

    def test_authenticated_preparation_migration_is_accepted(self):
        bundle = evidence_bundle()
        sources = attach_migration(bundle)
        with patch.object(verifier.campaign, 'sources', return_value=sources):
            self.assertFalse(verifier.verify_bundle(bundle)['promoted'])

    def test_inherited_preparation_requires_complete_portable_proof(self):
        for kind in ('missing_proof', 'missing_inheritance', 'false_completion', 'changed_original'):
            with self.subTest(kind=kind):
                bundle = evidence_bundle()
                sources = attach_migration(bundle)
                if kind == 'missing_proof':
                    del bundle['preparation_migration']
                    del bundle['preparation_migration_completion']
                elif kind == 'missing_inheritance':
                    del bundle['cache_manifests'][0]['inherited_preparation']
                elif kind == 'false_completion':
                    bundle['preparation_migration_completion']['labels_read'] = True
                else:
                    inherited = bundle['cache_manifests'][0]['inherited_preparation']
                    inherited['original_manifest']['files']['x.npy'] = '0' * 64
                    inherited['original_manifest_sha256'] = min(verifier.migration.serialized_hashes(inherited['original_manifest']))
                with patch.object(verifier.campaign, 'sources', return_value=sources):
                    with self.assertRaises(ValueError):
                        verifier.verify_bundle(bundle)

    def test_migration_cannot_relabel_observation_source_changes(self):
        bundle = evidence_bundle()
        sources = attach_migration(bundle)
        proof = bundle['preparation_migration']
        proof['original_adapter_source'] += '\nMIGRATION_INFERENCE_CHANGE = True\n'
        old_sources = proof['original_run_protocol']['source_hashes']
        old_sources[verifier.migration.ADAPTER] = hashlib.sha256(proof['original_adapter_source'].encode()).hexdigest()
        proof['original_run_protocol_sha256'] = min(verifier.migration.serialized_hashes(proof['original_run_protocol']))
        with patch.object(verifier.campaign, 'sources', return_value=sources):
            with self.assertRaisesRegex(ValueError, 'outside reference parsing changed'):
                verifier.verify_bundle(bundle)

    def test_training_leakage_or_source_drift_rejected(self):
        for kind in ('membership', 'source', 'used'):
            bundle = evidence_bundle()
            if kind == 'membership':
                bundle['fit']['training_matches'][0] = 'pff' + str(bundle['protocol']['split']['final'][0])
            elif kind == 'source':
                bundle['fit']['source_hashes'] = {'test': 'b' * 64}
            else:
                bundle['fit']['held_out_references_used'] = True
            with self.assertRaises(ValueError):
                verifier.verify_bundle(bundle)

    def test_no_human_or_rights_claim_can_be_added(self):
        for key in ('dense_truth_validated', 'usage_permitted', 'adequate_label_support'):
            bundle = evidence_bundle()
            bundle['development']['summary']['evidence'][key] = True
            with self.assertRaises(ValueError):
                verifier.verify_bundle(bundle)

    def test_prediction_and_access_hashes_authenticated(self):
        for section in ('prediction_freeze', 'reference_access'):
            bundle = evidence_bundle()
            bundle['development'][section]['candidate_sha256'] = 'b' * 64
            with self.assertRaises(ValueError):
                verifier.verify_bundle(bundle)

    def test_support_matches_frozen_event_source_and_rows(self):
        for kind in ('source', 'count', 'quality'):
            bundle = evidence_bundle()
            record = next(iter(bundle['development']['summary']['reference_support'].values()))
            if kind == 'source':
                record['event_sha256'] = 'b' * 64
            elif kind == 'count':
                record['class_counts']['H'] += 1
            else:
                record['dense_human_had_truth'] = True
            with self.assertRaises(ValueError):
                verifier.verify_bundle(bundle)

    def test_unreported_final_preparation_rejected(self):
        bundle = evidence_bundle()
        extra = copy.deepcopy(bundle['cache_manifests'][0])
        extra['match_id'] = bundle['protocol']['split']['final'][0]
        extra['identity']['metadata'] = bundle['protocol']['metadata_sha256'][str(extra['match_id'])]
        bundle['cache_manifests'].append(extra)
        with self.assertRaises(ValueError):
            verifier.verify_bundle(bundle)

    def test_publication_cannot_include_weights_or_claim_promotion(self):
        for key in ('weights_included', 'promoted'):
            bundle = evidence_bundle()
            bundle[key] = True
            with self.assertRaises(ValueError):
                verifier.verify_bundle(bundle)


if __name__ == '__main__':
    unittest.main()
