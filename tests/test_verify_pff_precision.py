import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import verify_pff_precision as verifier
from tools import pff_precision_validation as validation
from tools import pff_validation as original


def evidence_bundle(passing=False, final=False):
    """Only published aggregate evidence is used; no provider files are read."""
    parent = verifier.read(verifier.ROOT / 'models/pff-research.json')
    reservation = parent['protocol']; old = parent['development']['summary']['records']
    corrections, declarations = {}, []
    for i, penalty in enumerate(validation.PENALTIES):
        ident = f'penalty-{i:02d}'
        artifact = {'schema_version': 1, 'parent_sha256': validation.PARENT_SHA256,
                    'parameters': {'dead_logit_penalty': penalty}, 'policy': verifier.campaign.correction.POLICY}
        corrections[ident] = artifact
        declarations.append({'candidate_id': ident,
                             'artifact_sha256': min(verifier.parent_verifier._serialized_hashes(artifact)),
                             'parameters': artifact['parameters']})
    study = validation.build_protocol(reservation, old, declarations, {'test.py': 'e' * 64})
    manifests = {r['match_id']: r for r in parent['cache_manifests']}
    predictions = {r['match_id']: r for r in parent['development']['prediction_freeze']['records']}
    old_details = {(r['match'], r['candidate']): r for r in parent['development']['summary']['details']}
    records = {d['candidate_id']: [] for d in declarations}; details = []; frozen_rows = []
    for old_row in old:
        mid = old_row['match_id']
        for name, old_name in (('baseline', 'baseline'), ('parent', 'candidate')):
            detail = copy.deepcopy(old_details[str(mid), old_name]); detail['candidate'] = name; details.append(detail)
        for i, declared in enumerate(declarations):
            row = copy.deepcopy(old_row); row['candidate_sha256'] = declared['artifact_sha256']
            detail = copy.deepcopy(old_details[str(mid), 'candidate']); detail['candidate'] = declared['candidate_id']
            if passing and i:
                row['candidate']['dead_precision'] = row['baseline']['dead_precision'] + .003
                detail['dead_precision'] = row['candidate']['dead_precision']
            records[declared['candidate_id']].append(row); details.append(detail)
        hashes = {d['candidate_id']: predictions[mid]['candidate_prediction_sha256'] for d in declarations}
        hashes.update(parent=predictions[mid]['candidate_prediction_sha256'], baseline=predictions[mid]['baseline_prediction_sha256'])
        frozen_rows.append({'match_id': mid, 'prediction_sha256': hashes, 'observation_manifest': copy.deepcopy(manifests[mid])})
    summary = {'part': 'development', 'records': records, 'parent_records': copy.deepcopy(old), 'details': details,
               'reference_support': copy.deepcopy(parent['development']['summary']['reference_support']),
               'source_hashes': study['source_hashes'], 'protocol_sha256': study['protocol_sha256'],
               'evidence': copy.deepcopy(parent['development']['summary']['evidence']),
               'promoted': False, 'retained_model_sha256': original.BASELINE_SHA256,
               'selection': validation.select_candidate(study, records, old, reservation)}
    block = make_block(summary, frozen_rows, declarations, study)
    result = {'schema_version': 1, 'protocol': study, 'parent_research_sha256': 'f' * 64,
              'corrections': corrections, 'final_preparation_manifests': [], 'weights_included': False,
              'final_artifacts_present': final,
              'promoted': False, 'retained_model_sha256': original.BASELINE_SHA256,
              'development': block, 'final': None}
    if final:
        selected = summary['selection']; ident = selected['selected_candidate_id']
        if ident is None: ident = declarations[1]['candidate_id']
        declared = next(d for d in declarations if d['candidate_id'] == ident)
        source_mid = old[0]['match_id']; p_records, c_records, fdetails, frows, supports = [], [], [], [], {}
        for mid in reservation['split']['final']:
            manifest = copy.deepcopy(manifests[source_mid]); manifest.pop('inherited_preparation', None)
            manifest['match_id'] = mid; manifest['identity']['metadata'] = reservation['metadata_sha256'][str(mid)]
            manifest['identity']['sources'] = verifier.campaign.prior.sources()
            result['final_preparation_manifests'].append(manifest)
            p_row = copy.deepcopy(old[0]); p_row['match_id'] = mid; p_records.append(p_row)
            c_row = copy.deepcopy(p_row); c_row['candidate_sha256'] = declared['artifact_sha256']
            c_row['candidate']['dead_precision'] = c_row['baseline']['dead_precision'] + .003; c_records.append(c_row)
            for name, old_name in (('baseline', 'baseline'), ('parent', 'candidate'), (ident, 'candidate')):
                detail = copy.deepcopy(old_details[str(source_mid), old_name]); detail['match'] = str(mid); detail['candidate'] = name
                if name == ident: detail['dead_precision'] = c_row['candidate']['dead_precision']
                fdetails.append(detail)
            frows.append({'match_id': mid, 'prediction_sha256': {'baseline': manifest['files']['baseline.npy'], 'parent': '1' * 64, ident: '2' * 64},
                          'observation_manifest': manifest})
            supports[str(mid)] = copy.deepcopy(parent['development']['summary']['reference_support'][str(source_mid)])
        fsummary = {'part': 'final', 'records': {ident: c_records}, 'parent_records': p_records, 'details': fdetails,
                    'reference_support': supports, 'source_hashes': study['source_hashes'],
                    'protocol_sha256': study['protocol_sha256'], 'evidence': dict(summary['evidence'], development_passed=True),
                    'promoted': False, 'retained_model_sha256': original.BASELINE_SHA256}
        if passing:
            fsummary['gate'] = validation.final_gate(c_records, p_records, reservation, fsummary['evidence'], selected)
        else:
            fsummary['gate'] = {'promoted': False}
        result['final'] = make_block(fsummary, frows, [declared], study)
    return result, parent


def make_block(summary, rows, declarations, study):
    part = summary['part']
    frozen = {'part': part, 'records': rows, 'candidates': declarations,
              'protocol_sha256': study['protocol_sha256'], 'source_hashes': study['source_hashes'],
              'references_accessed_in_this_study': False, 'development_previously_consumed': part == 'development'}
    freeze_hash = min(verifier.parent_verifier._serialized_hashes(frozen))
    marker = {'part': part, 'prediction_freeze_sha256': freeze_hash,
              'protocol_sha256': study['protocol_sha256'], 'source_hashes': study['source_hashes'],
              'development_previously_consumed': part == 'development'}
    summary['prediction_freeze_sha256'] = freeze_hash
    summary['reference_access_sha256'] = min(verifier.parent_verifier._serialized_hashes(marker, False))
    return {'summary': summary, 'prediction_freeze': frozen, 'reference_access': marker}


class PrecisionEvidenceTests(unittest.TestCase):
    def verify(self, bundle, parent):
        with patch.object(verifier, '_parent_context', return_value=(parent, 'f' * 64)), \
             patch.object(verifier.campaign, 'sources', return_value={'test.py': 'e' * 64}):
            return verifier.verify_bundle(bundle)

    def test_failed_grid_retains_submission_and_final(self):
        bundle, parent = evidence_bundle()
        result = self.verify(bundle, parent)
        self.assertFalse(result['development_gate_passed']); self.assertFalse(result['final_evaluated'])
        self.assertFalse(result['promoted']); self.assertFalse(result['local_artifact_bytes_verified'])

    def test_passing_grid_and_final_still_require_truth_and_rights(self):
        bundle, parent = evidence_bundle(passing=True, final=True)
        result = self.verify(bundle, parent)
        self.assertTrue(result['development_gate_passed']); self.assertTrue(result['final_evaluated'])
        self.assertFalse(result['promoted'])

    def test_final_cannot_follow_failed_grid(self):
        bundle, parent = evidence_bundle(final=True)
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_invented_deployment_and_rights_rejected(self):
        original_bundle, parent = evidence_bundle()
        for field in ('weights_included', 'promoted', 'dense_truth_validated', 'usage_permitted', 'adequate_label_support'):
            with self.subTest(field=field):
                bundle = copy.deepcopy(original_bundle)
                if field in bundle: bundle[field] = True
                else: bundle['development']['summary']['evidence'][field] = True
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_artifact_or_source_drift_rejected(self):
        original_bundle, parent = evidence_bundle()
        for kind in ('artifact', 'source', 'parent'):
            with self.subTest(kind=kind):
                bundle = copy.deepcopy(original_bundle)
                if kind == 'artifact': bundle['corrections']['penalty-01']['parameters']['dead_logit_penalty'] = .2
                elif kind == 'source': bundle['protocol']['source_hashes']['test.py'] = 'a' * 64
                else: bundle['parent_research_sha256'] = 'a' * 64
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_selection_tamper_rejected(self):
        bundle, parent = evidence_bundle()
        bundle['development']['summary']['selection']['eligible_for_final_audit'] = True
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_details_bind_every_model_metric(self):
        original_bundle, parent = evidence_bundle()
        for name in ('baseline', 'parent', 'penalty-01'):
            for field in ('spellF1', 'macroF1', 'dead_precision', 'dead_recall'):
                with self.subTest(model=name, field=field):
                    bundle = copy.deepcopy(original_bundle)
                    detail = next(d for d in bundle['development']['summary']['details'] if d['candidate'] == name)
                    detail[field] -= .001
                    with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_duplicate_missing_or_foreign_details_rejected(self):
        original_bundle, parent = evidence_bundle()
        for kind in ('duplicate', 'missing', 'foreign'):
            with self.subTest(kind=kind):
                bundle = copy.deepcopy(original_bundle); details = bundle['development']['summary']['details']
                if kind == 'duplicate': details.append(copy.deepcopy(details[0]))
                elif kind == 'missing': details.pop()
                else: details[0]['candidate'] = 'unreported'
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_record_identity_and_support_rejected(self):
        original_bundle, parent = evidence_bundle()
        for key, value in (('candidate_sha256', 'a' * 64), ('reference_dead_segments', 1), ('reference_short_spells', 1)):
            with self.subTest(field=key):
                bundle = copy.deepcopy(original_bundle)
                bundle['development']['summary']['records']['penalty-01'][0][key] = value
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_prediction_and_access_records_bound(self):
        original_bundle, parent = evidence_bundle()
        for field in ('references_accessed_in_this_study', 'development_previously_consumed'):
            with self.subTest(field=field):
                bundle = copy.deepcopy(original_bundle)
                bundle['development']['prediction_freeze'][field] = not bundle['development']['prediction_freeze'][field]
                with self.assertRaises(ValueError): self.verify(bundle, parent)
        bundle = copy.deepcopy(original_bundle); bundle['development']['reference_access']['prediction_freeze_sha256'] = 'a' * 64
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_parent_prediction_or_zero_identity_cannot_be_rehashed(self):
        original_bundle, parent = evidence_bundle()
        for name in ('parent', 'penalty-00'):
            with self.subTest(name=name):
                bundle = copy.deepcopy(original_bundle); block = bundle['development']
                block['prediction_freeze']['records'][0]['prediction_sha256'][name] = 'a' * 64
                bundle['development'] = make_block(block['summary'], block['prediction_freeze']['records'], block['prediction_freeze']['candidates'], bundle['protocol'])
                with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_unreported_final_preparation_rejected(self):
        bundle, parent = evidence_bundle(passing=True, final=True)
        bundle['final'] = None
        with self.assertRaises(ValueError): self.verify(bundle, parent)

    def test_local_candidate_bytes_are_authenticated(self):
        bundle, parent = evidence_bundle()
        with patch.object(verifier, 'verify_bundle', return_value={}), \
             patch.object(verifier, 'bundle_from_run', return_value=bundle), \
             patch.object(verifier, 'sha256', return_value='a' * 64):
            with self.assertRaises(ValueError): verifier.verify_local(bundle, Path('unused'))

    def test_local_correction_cache_prediction_and_csv_bytes(self):
        bundle, parent = evidence_bundle(); out = Path('unused').resolve()
        expected = {verifier.campaign.PARENT / 'candidate.json': validation.PARENT_SHA256}
        for declared in bundle['protocol']['candidates']:
            expected[out / 'corrections' / (declared['candidate_id'] + '.json')] = declared['artifact_sha256']
        for row in bundle['development']['prediction_freeze']['records']:
            mid = row['match_id']; cache = verifier.campaign.observation_root(out, 'development') / 'cache' / str(mid)
            for name, value in row['observation_manifest']['files'].items(): expected[cache / name] = value
            for name, value in row['prediction_sha256'].items(): expected[out / 'development' / str(mid) / (name + '.npy')] = value
        for detail in bundle['development']['summary']['details']:
            expected[out / 'development' / detail['match'] / detail['candidate'] / 'predictions.csv'] = detail['prediction_sha256']
        with patch.object(verifier, 'verify_bundle', return_value={}), \
             patch.object(verifier, 'bundle_from_run', return_value=bundle):
            with patch.object(verifier, 'sha256', side_effect=lambda path: expected[Path(path)]):
                self.assertTrue(verifier.verify_local(bundle, out)['local_artifact_bytes_verified'])
            targets = [next(path for path in expected if 'corrections' in path.parts),
                       next(path for path in expected if path.name == 'x.npy'),
                       next(path for path in expected if path.name == 'parent.npy'),
                       next(path for path in expected if path.name == 'predictions.csv')]
            for bad in targets:
                with self.subTest(path=bad.name), patch.object(verifier, 'sha256', side_effect=lambda path: 'a'*64 if Path(path) == bad else expected[Path(path)]):
                    with self.assertRaises(ValueError): verifier.verify_local(bundle, out)

    def test_unreported_final_access_without_cache_is_rejected(self):
        bundle, parent = evidence_bundle()
        bundle['final_artifacts_present'] = True
        with self.assertRaises(ValueError): self.verify(bundle, parent)


if __name__ == '__main__':
    unittest.main()
