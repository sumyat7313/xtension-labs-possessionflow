"""Independent synthetic execution-boundary checks; no provider records or fits."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import clean_pff_adaptation_inputs as inputs
from tools import clean_pff_adaptation_protocol as protocol
from tools import clean_pff_adaptation_stages as stages


class InputBoundaryIntegrityTests(unittest.TestCase):
    def test_heldout_factory_call_rejects_before_any_authentication_io(self):
        factory = inputs._InnerFactory({})
        ids = (*protocol.INNER_VALIDATION_IDS, *protocol.DEVELOPMENT_IDS, *protocol.FINAL_IDS)
        with patch.object(inputs, '_sources') as sources, \
             patch.object(inputs, '_pff_record') as pff, \
             patch.object(inputs, '_sportec_record') as sportec:
            for mid in ids:
                with self.subTest(mid=mid), self.assertRaises(ValueError):
                    factory(f'pff{mid}')
            sources.assert_not_called()
            pff.assert_not_called()
            sportec.assert_not_called()

    def test_core_identity_mismatch_rejects_before_evidence_or_parent_reads(self):
        with patch.object(inputs, '_evidence') as evidence:
            with self.assertRaises(ValueError):
                inputs.make_inner_factory(fixture_ids=protocol.fit_ids(), protocol_sha256='a'*64)
            evidence.assert_not_called()

    def test_array_modified_during_load_is_rejected_after_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            file = root/'synthetic.npy'
            np.save(file, np.arange(12, dtype=np.float64))
            expected = hashlib.sha256(file.read_bytes()).hexdigest()
            real_load = np.load
            def altered(stream, **kwargs):
                result = real_load(stream, **kwargs)
                with file.open('ab') as writer:
                    writer.write(b'changed after parse')
                return result
            with patch.object(inputs, 'ROOT', root), patch.object(inputs.np, 'load', side_effect=altered):
                with self.assertRaisesRegex(ValueError, 'changed during materialization'):
                    inputs._array('synthetic.npy', expected)

    def test_json_parser_uses_exact_authenticated_bytes_once(self):
        with patch.object(inputs, '_bytes', return_value=b'{"synthetic":true}') as checked:
            self.assertEqual(inputs._json('synthetic.json', 'a'*64), {'synthetic': True})
            checked.assert_called_once_with('synthetic.json', 'a'*64)


class LifecycleBoundaryIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root/'runs').mkdir()
        self.out = self.root/'runs'/stages.OUT_NAME
        self.description = {'descriptor_sha256': 'a'*64, 'synthetic': True}
        fake_descriptor = patch.object(stages, 'descriptor', side_effect=lambda root: copy.deepcopy(self.description))
        fake_descriptor.start()
        self.addCleanup(fake_descriptor.stop)
        stages.initialize(self.out, self.root)

    @staticmethod
    def receipt():
        return {'artifact_sha256': {'synthetic': 'b'*64}, 'execution_verified': False}

    def events(self):
        return [json.loads(line) for line in (self.out/'ledger.jsonl').read_text().splitlines()]

    def test_source_descriptor_change_during_attempt_consumes_failure(self):
        with self.assertRaisesRegex(ValueError, 'descriptor/source/runtime changed'):
            with stages.attempt(self.out, 'initialize', self.root) as handle:
                handle.complete(self.receipt())
                self.description['synthetic'] = False
        self.assertEqual([event['state'] for event in self.events()], ['started', 'failed'])
        self.description['synthetic'] = True
        state = stages.status(self.out, self.root)
        self.assertTrue(state['blocked'])
        with self.assertRaises(ValueError):
            handle.complete(self.receipt())
        with self.assertRaises(ValueError):
            with stages.attempt(self.out, 'initialize', self.root):
                self.fail('Consumed attempt must not yield')

    def test_manifest_mutation_failure_retains_original_descriptor_identity(self):
        with self.assertRaisesRegex(ValueError, 'Manifest changed'):
            with stages.attempt(self.out, 'initialize', self.root) as handle:
                handle.complete(self.receipt())
                changed = dict(self.description, descriptor_sha256='c'*64)
                (self.out/'manifest.json').write_text(json.dumps(changed))
        self.assertEqual([event['state'] for event in self.events()], ['started', 'failed'])
        self.assertTrue(all(event['descriptor_sha256']=='a'*64 for event in self.events()))

    def test_marker_before_started_append_crash_stays_consumed(self):
        marker = stages._marker_value(0, self.description, stages.ZERO)
        stages._exclusive(stages._marker(self.out, 0), marker)
        self.assertTrue(stages.status(self.out, self.root)['blocked'])
        with self.assertRaises(ValueError):
            stages.initialize(self.out, self.root)
        with self.assertRaises(ValueError):
            with stages.attempt(self.out, 'initialize', self.root):
                self.fail('Orphan marker must not yield')

    def test_keyboard_interrupt_consumes_attempt_and_closes_handle(self):
        with self.assertRaises(KeyboardInterrupt):
            with stages.attempt(self.out, 'initialize', self.root) as handle:
                raise KeyboardInterrupt()
        self.assertEqual(self.events()[-1]['detail'], {'exception_type': 'KeyboardInterrupt'})
        self.assertTrue(stages.status(self.out, self.root)['blocked'])
        with self.assertRaises(ValueError):
            handle.complete(self.receipt())

    def test_declared_success_never_grants_execution_or_refit(self):
        with stages.attempt(self.out, 'initialize', self.root) as handle:
            handle.complete(self.receipt())
        state = stages.status(self.out, self.root)
        self.assertEqual(state['next_stage'], 'fit-A')
        for key in protocol.FLAGS:
            self.assertIs(state[key], False)
        self.assertIs(state['resource_limits_enforced'], False)


if __name__ == '__main__':
    unittest.main()
