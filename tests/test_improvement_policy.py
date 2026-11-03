import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools import improvement_policy as policy


def lineage(candidate='c' * 64):
    dataset = 'd' * 64
    return {'schema_version': 1, 'candidate_sha256': candidate,
            'comparator_sha256': policy.original.BASELINE_SHA256,
            'fit_inputs': [dataset], 'artifacts': [
                {'sha256': candidate, 'kind': 'model', 'scope': 'external',
                 'identifier': 'external-rebuild', 'parents': [dataset]},
                {'sha256': dataset, 'kind': 'dataset', 'scope': 'external',
                 'identifier': 'sportec', 'parents': []}]}


class LineageTests(unittest.TestCase):
    def test_valid_graph_does_not_certify_history(self):
        result = policy.audit_lineage(lineage())
        self.assertTrue(result['declared_lineage_valid'])
        self.assertFalse(result['provenance_semantically_certified'])

    def test_challenge_ancestors_rejected(self):
        for scope in ('challenge', 'unknown'):
            with self.subTest(scope=scope):
                value = lineage(); value['artifacts'][1]['scope'] = scope
                with self.assertRaises(ValueError):
                    policy.audit_lineage(value)

    def test_known_contaminated_model_cannot_be_relabelled(self):
        for sha in policy.PROHIBITED_FIT_MODELS:
            with self.subTest(sha=sha), self.assertRaises(ValueError):
                policy.audit_lineage(lineage(sha))

    def test_target_identifiers_rejected(self):
        for name in ('BAR-ATH observations', 'target-profile', 'v1.4 checkpoint'):
            value = lineage(); value['artifacts'][1]['identifier'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                policy.audit_lineage(value)

    def test_missing_parent_cycle_and_unrelated_artifact(self):
        for mode in ('missing', 'cycle', 'unrelated'):
            value = lineage()
            if mode == 'missing':
                value['artifacts'][0]['parents'] = ['e' * 64]
            elif mode == 'cycle':
                value['artifacts'][1]['parents'] = ['c' * 64]
            else:
                value['artifacts'].append({'sha256': 'e' * 64, 'kind': 'code', 'scope': 'synthetic', 'identifier': 'extra', 'parents': []})
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                policy.audit_lineage(value)

    def test_parentless_model_not_allowed(self):
        value = lineage(); value['artifacts'][0]['parents'] = []
        with self.assertRaises(ValueError):
            policy.audit_lineage(value)

    def test_pinned_external_root_model_allowed(self):
        value = lineage()
        value['artifacts'][0]['parents'].append(policy.APPROVED_ROOT_MODEL)
        value['artifacts'].append({'sha256': policy.APPROVED_ROOT_MODEL, 'kind': 'model', 'scope': 'external', 'identifier': 'audited-v1.2-dead-head', 'parents': []})
        self.assertTrue(policy.audit_lineage(value)['declared_lineage_valid'])

    def test_fit_input_omission_and_duplicate_rejected(self):
        for inputs in ([], ['e' * 64], ['d' * 64, 'd' * 64], ['c' * 64]):
            value = lineage(); value['fit_inputs'] = inputs
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                policy.audit_lineage(value)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = self.root / 'review.md'
        self.document.write_text('Synthetic evidence fixture only.', encoding='utf-8')
        self.sha = hashlib.sha256(self.document.read_bytes()).hexdigest()

    def receipts(self, dataset='sportec'):
        return [{'dataset_id': dataset, 'purpose': purpose, 'finding': 'supported',
                 'reviewer': 'Test reviewer', 'source_url': 'https://example.org/terms',
                 'path': 'review.md', 'sha256': self.sha} for purpose in policy.RECEIPT_PURPOSES]

    def test_missing_evidence_is_not_passed_by_boolean(self):
        result = policy.inspect_receipts([], ['sportec'], self.root)
        self.assertFalse(result['complete_support_claims'])
        self.assertEqual(len(result['missing_or_unresolved']), 3)
        with self.assertRaises(ValueError):
            policy.inspect_receipts([{'usage_permitted': True}], ['sportec'], self.root)

    def test_hashed_claim_not_semantic_certification(self):
        result = policy.inspect_receipts(self.receipts(), ['sportec'], self.root)
        self.assertTrue(result['complete_support_claims'])
        self.assertFalse(result['semantic_certification'])

    def test_unresolved_and_prohibited_block_readiness(self):
        for finding in ('unresolved', 'prohibited'):
            receipts = self.receipts(); receipts[0]['finding'] = finding
            self.assertFalse(policy.inspect_receipts(receipts, ['sportec'], self.root)['complete_support_claims'])

    def test_changed_receipt_bytes_rejected(self):
        self.document.write_text('Changed after receipt was recorded.', encoding='utf-8')
        with self.assertRaises(ValueError):
            policy.inspect_receipts(self.receipts(), ['sportec'], self.root)

    def test_duplicate_receipt_and_escape_rejected(self):
        receipts = self.receipts()
        with self.assertRaises(ValueError):
            policy.inspect_receipts(receipts + [receipts[0]], ['sportec'], self.root)
        receipts[0]['path'] = '../review.md'
        with self.assertRaises(ValueError):
            policy.inspect_receipts(receipts, ['sportec'], self.root)

    def test_model_binding_is_exact(self):
        model = self.root / 'model.json'; model.write_text('{}', encoding='utf-8')
        sha = hashlib.sha256(model.read_bytes()).hexdigest()
        ancestry = policy.audit_lineage(lineage(sha))
        self.assertTrue(policy.authenticate_models(ancestry, {sha: 'model.json'}, self.root)['model_bytes_authenticated'])
        model.write_text('{"changed":true}', encoding='utf-8')
        with self.assertRaises(ValueError):
            policy.authenticate_models(ancestry, {sha: 'model.json'}, self.root)
        with self.assertRaises(ValueError):
            policy.authenticate_models(ancestry, {}, self.root)


class DevelopmentTests(unittest.TestCase):
    setUp = EvidenceTests.setUp
    receipts = EvidenceTests.receipts

    def setup_screen(self):
        bundle = json.loads((policy.ROOT / 'models/pff-research.json').read_text(encoding='utf-8'))
        parent = bundle['development']['summary']['records']
        reservation = bundle['protocol']
        model = self.root / 'model.json'; model.write_text('{"synthetic_test":true}', encoding='utf-8')
        sha = hashlib.sha256(model.read_bytes()).hexdigest()
        records = copy.deepcopy(parent)
        for row in records:
            row['candidate_sha256'] = sha
            row['candidate']['dead_precision'] = min(1., row['baseline']['dead_precision'] + .01)
        return records, parent, reservation, lineage(sha), sha

    def screen(self, records, parent, reservation, manifest, sha, receipts=None):
        return policy.screen_development(records, parent, reservation, manifest,
                                         [] if receipts is None else receipts, self.root,
                                         model_paths={sha: 'model.json'}, artifact_root=self.root)

    def test_numeric_pass_does_not_unlock_final_or_promote(self):
        args = self.setup_screen()
        result = self.screen(*args)
        self.assertTrue(result['numeric_gate']['numeric_gate_passed'])
        self.assertFalse(result['ready_for_independent_final_review'])
        self.assertFalse(result['final_access_authorized'])
        self.assertFalse(result['promoted'])

    def test_even_supported_receipts_require_independent_review(self):
        args = self.setup_screen()
        receipts = self.receipts() + self.receipts(policy.EVALUATION_DATASET)
        result = self.screen(*args, receipts=receipts)
        self.assertTrue(result['ready_for_independent_final_review'])
        self.assertFalse(result['final_access_authorized'])
        self.assertFalse(result['promoted'])

    def test_evaluation_dataset_needs_evidence_even_when_not_used_for_fit(self):
        result = self.screen(*self.setup_screen(), receipts=self.receipts())
        missing = result['evidence_receipts']['missing_or_unresolved']
        self.assertEqual({row['dataset_id'] for row in missing}, {policy.EVALUATION_DATASET})

    def test_partial_cohort_nonfinite_and_artifact_mismatch_rejected(self):
        for mode in ('partial', 'nonfinite', 'candidate'):
            args = list(self.setup_screen())
            if mode == 'partial':
                args[0].pop()
            elif mode == 'nonfinite':
                args[0][0]['candidate']['spell'] = float('nan')
            else:
                for row in args[0]:
                    row['candidate_sha256'] = 'e' * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.screen(*args)

    def test_original_gain_floors_cannot_be_redefined(self):
        args = list(self.setup_screen())
        args[1] = copy.deepcopy(args[1]); args[1][0]['candidate']['short_precision'] = 0
        with self.assertRaises(ValueError):
            self.screen(*args)

    def test_reservation_cannot_be_replaced(self):
        args = list(self.setup_screen())
        args[2] = copy.deepcopy(args[2]); args[2]['protocol_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            self.screen(*args)

    def test_precision_margin_not_relaxed(self):
        args = list(self.setup_screen())
        for row in args[0]:
            row['candidate']['dead_precision'] = row['baseline']['dead_precision'] + .001
        result = self.screen(*args)
        self.assertFalse(result['numeric_gate']['checks']['dead_precision_margin'])


if __name__ == '__main__':
    unittest.main()
