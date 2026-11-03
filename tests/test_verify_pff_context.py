import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import verify_pff_context as verifier, pff_context_validation as validation
from tools import pff_validation as original
from test_verify_pff_precision import evidence_bundle as scalar_fixture, make_block

SOURCES = {'test.py': 'e' * 64, 'tools/pff_context_prepare.py': 'd' * 64}


def fixture(passing=False, final=False):
    scalar, parent = scalar_fixture(passing=passing, final=final)
    reservation = parent['protocol']; old = parent['development']['summary']['records']
    plan = {'schema_version': 1, 'source_hashes': SOURCES, 'fit_policy': verifier.campaign.correction.POLICY,
            'training_matches': reservation['split']['train'],
            'held_out_matches': reservation['split']['development'] + reservation['split']['final'],
            'reservation_sha256': reservation['protocol_sha256'], 'blend_grid': list(validation.BLENDS),
            'parent_sha256': validation.PARENT_SHA256, 'previous_scalar_research_sha256': 'c' * 64,
            'adaptive_followup': True, 'development_previously_consumed': True}
    plan['protocol_sha256'] = original.digest(plan)
    manifests = []; records = []
    old_manifests = {m['match_id']: m for m in parent['cache_manifests']}
    for mid in plan['training_matches']:
        old_manifest = old_manifests[mid]
        identity = {'parent_candidate_sha256': validation.PARENT_SHA256,
                    'source_cache_manifest_sha256': min(verifier.parent_verifier._serialized_hashes(old_manifest)),
                    'source_cache_identity': old_manifest['identity'],
                    'protocol_sha256': reservation['protocol_sha256'],
                    'preparation_source_sha256': SOURCES['tools/pff_context_prepare.py']}
        manifest = {'schema_version': 1, 'match_id': mid, 'rows': old_manifest['rows'], 'fps': 25.,
                    'columns': ['H', 'A', 'D'], 'identity': identity, 'references_read': False,
                    'reference_fields_used_for_inference': False, 'files': {'parent_probabilities.npy': '1' * 64}}
        manifests.append(manifest)
        records.append({'match_id': mid, 'manifest_sha256': min(verifier.parent_verifier._serialized_hashes(manifest)),
                        'probability_sha256': '1' * 64, 'rows': old_manifest['rows']})
    complete = {'schema_version': 1, 'parent_candidate_sha256': validation.PARENT_SHA256,
                'protocol_sha256': reservation['protocol_sha256'], 'references_read': False,
                'preparation_source_sha256': SOURCES['tools/pff_context_prepare.py'], 'records': records}
    fit_access = {'training_plan_sha256': min(verifier.parent_verifier._serialized_hashes(plan)),
                  'training_probabilities_sha256': min(verifier.parent_verifier._serialized_hashes(complete)),
                  'held_out_references_read': False}
    metadata = {'training_matches': plan['training_matches'], 'held_out_matches': sorted(plan['held_out_matches']),
                'protocol_sha256': plan['protocol_sha256'], 'policy': verifier.campaign.correction.POLICY,
                'parent_sha256': validation.PARENT_SHA256, 'reference_fields_used_for_inference': False,
                'support': {str(mid): {'sampled_rows': 3, 'sampled_class_counts': dict.fromkeys('HAD', 1),
                                     'sampled_fixture_mass_before_global_normalization': 1.} for mid in plan['training_matches']}}
    fit = {'head_sha256': '9' * 64, 'training_plan_sha256': fit_access['training_plan_sha256'],
           'fit_access_sha256': min(verifier.parent_verifier._serialized_hashes(fit_access, False)),
           'head_metadata': metadata,
           'training': {'epochs': 4, 'seed': 20261103, 'sampled_rows': 120, 'cpu_threads': 1,
                        'torch_version': 'synthetic-test-only', 'conditional_home_away_trained': False, 'losses': [.5] * 4},
           'training_reference_support': copy.deepcopy(parent['fit']['training_reference_support']),
           'training_probability_manifests': manifests, 'training_probability_complete': complete,
           'held_out_references_read': False}
    declarations = []; corrections = {}; rename = {}
    for i, blend in enumerate(validation.BLENDS):
        ident = f'context-{i:02d}'; rename[f'penalty-{i:02d}'] = ident
        artifact = {'schema_version': 1, 'parent_sha256': validation.PARENT_SHA256, 'head_sha256': fit['head_sha256'],
                    'parameters': {'blend': blend}, 'policy': verifier.campaign.correction.POLICY}
        corrections[ident] = artifact
        declarations.append({'candidate_id': ident, 'artifact_sha256': min(verifier.parent_verifier._serialized_hashes(artifact)),
                             'parameters': artifact['parameters']})
    study = validation.build_protocol(reservation, old, declarations, SOURCES, head_sha256=fit['head_sha256'],
                                       training_matches=plan['training_matches'], fit_policy=verifier.campaign.correction.POLICY)
    bundle = {'schema_version': 1, 'protocol': study, 'parent_research_sha256': 'f' * 64,
              'corrections': corrections, 'training_plan': plan, 'fit': fit, 'fit_access': fit_access,
              'final_preparation_manifests': scalar['final_preparation_manifests'], 'final_artifacts_present': final,
              'weights_included': False, 'promoted': False, 'retained_model_sha256': original.BASELINE_SHA256}
    for part in ('development', 'final'):
        block = scalar[part]
        if block is None:
            bundle[part] = None; continue
        summary = copy.deepcopy(block['summary']); summary['source_hashes'] = SOURCES; summary['protocol_sha256'] = study['protocol_sha256']
        selected_declarations = declarations if part == 'development' else [declarations[1]]
        summary['records'] = {rename[name]: rows for name, rows in summary['records'].items() if name in rename}
        for declared in selected_declarations:
            for row in summary['records'][declared['candidate_id']]: row['candidate_sha256'] = declared['artifact_sha256']
        summary['details'] = [d for d in summary['details'] if d['candidate'] in rename or d['candidate'] in ('baseline', 'parent')]
        for detail in summary['details']: detail['candidate'] = rename.get(detail['candidate'], detail['candidate'])
        frozen_rows = copy.deepcopy(block['prediction_freeze']['records'])
        for row in frozen_rows:
            row['prediction_sha256'] = {rename.get(name, name): value for name, value in row['prediction_sha256'].items()
                                        if name in rename or name in ('baseline', 'parent')}
        if part == 'development':
            summary['selection'] = validation.select_candidate(study, summary['records'], old, reservation)
        elif passing:
            selection = bundle['development']['summary']['selection']
            summary['gate'] = validation.final_gate(summary['records'][selection['selected_candidate_id']],
                                                    summary['parent_records'], reservation, summary['evidence'], selection)
        bundle[part] = make_block(summary, frozen_rows, selected_declarations, study)
    return bundle, parent


class ContextEvidenceTests(unittest.TestCase):
    def verify(self, bundle, parent):
        with patch.object(verifier, '_parent_context', return_value=(parent, 'f' * 64)), \
             patch.object(verifier, '_scalar_context', return_value='c' * 64), \
             patch.object(verifier.campaign, 'sources', return_value=SOURCES):
            return verifier.verify_bundle(bundle)

    def test_complete_context_evidence_retains_baseline(self):
        bundle, parent = fixture()
        result = self.verify(bundle, parent)
        self.assertFalse(result['development_gate_passed']); self.assertFalse(result['final_evaluated']); self.assertFalse(result['promoted'])

    def test_synthetic_passing_context_final_still_requires_evidence(self):
        bundle, parent = fixture(passing=True, final=True)
        result = self.verify(bundle, parent)
        self.assertTrue(result['development_gate_passed']); self.assertTrue(result['final_evaluated']); self.assertFalse(result['promoted'])

    def test_training_plan_history_and_membership_rejected(self):
        pristine, parent = fixture()
        for key, value in (('adaptive_followup', False), ('previous_scalar_research_sha256', 'a' * 64),
                           ('training_matches', pristine['training_plan']['training_matches'][:-1]),
                           ('development_previously_consumed', False)):
            with self.subTest(field=key):
                bundle = copy.deepcopy(pristine); bundle['training_plan'][key] = value
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_fit_metadata_cannot_claim_a_different_plan_or_training_set(self):
        pristine, parent = fixture()
        for key, value in (('protocol_sha256', 'a' * 64), ('parent_sha256', 'b' * 64),
                           ('reference_fields_used_for_inference', True), ('training_matches', [3814])):
            with self.subTest(field=key):
                bundle = copy.deepcopy(pristine); bundle['fit']['head_metadata'][key] = value
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_weight_publication_and_held_out_fit_claims_rejected(self):
        pristine, parent = fixture()
        for kind in ('weights', 'flag', 'head'):
            with self.subTest(kind=kind):
                bundle = copy.deepcopy(pristine)
                if kind == 'weights': bundle['fit']['input_weight'] = [[1.]]
                elif kind == 'flag': bundle['fit']['held_out_references_read'] = True
                else: bundle['fit']['head_sha256'] = 'a' * 64
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_training_probability_cohort_and_observation_provenance_rejected(self):
        pristine, parent = fixture()
        for kind in ('missing', 'reference', 'source', 'columns'):
            with self.subTest(kind=kind):
                bundle = copy.deepcopy(pristine); manifests = bundle['fit']['training_probability_manifests']
                if kind == 'missing': manifests.pop()
                elif kind == 'reference': manifests[0]['references_read'] = True
                elif kind == 'source': manifests[0]['identity']['source_cache_identity']['tracking'] = 'a' * 64
                else: manifests[0]['columns'] = ['D', 'H', 'A']
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_fit_access_marker_and_reference_support_rejected(self):
        pristine, parent = fixture()
        bundle = copy.deepcopy(pristine); bundle['fit_access']['held_out_references_read'] = True
        with self.assertRaises(ValueError): self.verify(bundle, parent)
        bundle = copy.deepcopy(pristine)
        mid = next(iter(bundle['fit']['training_reference_support']))
        bundle['fit']['training_reference_support'][mid]['class_counts']['D'] += 1
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_training_sample_counts_and_fixed_epochs_rejected(self):
        pristine, parent = fixture()
        for kind in ('samples', 'class', 'losses', 'epochs', 'conditional'):
            with self.subTest(kind=kind):
                bundle = copy.deepcopy(pristine); fit = bundle['fit']; mid = next(iter(fit['head_metadata']['support']))
                if kind == 'samples': fit['training']['sampled_rows'] += 1
                elif kind == 'class': fit['head_metadata']['support'][mid]['sampled_class_counts']['D'] += 1
                elif kind == 'losses': fit['training']['losses'][0] = float('nan')
                elif kind == 'epochs': fit['training']['epochs'] = 99
                else: fit['training']['conditional_home_away_trained'] = True
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_every_correction_binds_same_head(self):
        bundle, parent = fixture(); bundle['corrections']['context-01']['head_sha256'] = 'a' * 64
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_final_activity_after_failed_context_is_rejected(self):
        bundle, parent = fixture(final=True)
        with self.assertRaises(ValueError): self.verify(bundle, parent)
        bundle, parent = fixture(); bundle['final_artifacts_present'] = True
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_context_details_and_selection_cannot_be_tampered(self):
        pristine, parent = fixture()
        bundle = copy.deepcopy(pristine); bundle['development']['summary']['selection']['eligible_for_final_audit'] = True
        with self.assertRaises(ValueError): self.verify(bundle, parent)
        bundle = copy.deepcopy(pristine); bundle['development']['summary']['details'][0]['spellF1'] += .001
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_local_head_and_training_probability_bytes(self):
        bundle, parent = fixture(); out = Path('unused').resolve(); fit = bundle['fit']
        head = {'metadata': fit['head_metadata'], 'training': fit['training']}
        reads = {out / 'context-head.json': head, verifier.ROOT / 'models/pff-research.json': parent,
                 out / 'training-probabilities/complete.json': fit['training_probability_complete']}
        hashes = {out / 'context-head.json': fit['head_sha256'], out / 'training-plan.json': fit['training_plan_sha256'],
                  out / 'fit-access.json': fit['fit_access_sha256'],
                  out / 'training-probabilities/complete.json': bundle['fit_access']['training_probabilities_sha256']}
        old = {m['match_id']: m for m in parent['cache_manifests']}
        for record, manifest in zip(fit['training_probability_complete']['records'], fit['training_probability_manifests']):
            mid = record['match_id']; folder = out / 'training-probabilities' / str(mid)
            reads[folder / 'manifest.json'] = manifest; hashes[folder / 'manifest.json'] = record['manifest_sha256']
            hashes[folder / 'parent_probabilities.npy'] = record['probability_sha256']
            cache = verifier.campaign.PARENT / 'cache' / str(mid)
            hashes[cache / 'complete.json'] = manifest['identity']['source_cache_manifest_sha256']
            for name, value in old[mid]['files'].items(): hashes[cache / name] = value
        with patch.object(verifier, 'read', side_effect=lambda path: reads[Path(path)]), \
             patch.object(verifier.campaign.correction, 'validate_head', side_effect=lambda value: value):
            with patch.object(verifier, 'sha256', side_effect=lambda path: hashes[Path(path)]):
                verifier._verify_local_training(bundle, out)
            for bad in (out / 'context-head.json', next(p for p in hashes if p.name == 'parent_probabilities.npy'),
                        next(p for p in hashes if p.name == 'x.npy')):
                with self.subTest(path=bad.name), patch.object(verifier, 'sha256', side_effect=lambda path: 'a'*64 if Path(path) == bad else hashes[Path(path)]):
                    with self.assertRaises(ValueError): verifier._verify_local_training(bundle, out)


if __name__ == '__main__':
    unittest.main()
