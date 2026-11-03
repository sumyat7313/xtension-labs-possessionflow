import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import development_harness as driver


class DevelopmentHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tools').mkdir()
        (self.root / 'possession').mkdir()
        for name in driver.SOURCE_ENTRIES:
            (self.root / name).write_text('from possession import features\n', encoding='utf-8')
        (self.root / 'possession/features.py').write_text('# synthetic feature implementation\n')
        refit = self.root / driver.REBUILD / 'refit'
        refit.mkdir(parents=True)
        for name in ('candidate.json', 'config.json', 'lineage.json', 'model-bindings.json', 'complete.json'):
            (refit / name).write_text(json.dumps({'synthetic': name}) + '\n')
        (refit.parent / 'protocol.json').write_text('{"synthetic": true}\n')
        self.model_hash = driver.storage.file_hash(refit / 'candidate.json')
        active = patch.object(driver, 'CANDIDATE_SHA256', self.model_hash)
        active.start()
        self.addCleanup(active.stop)
        self.out = self.root / 'runs/development'
        driver.initialize(self.out, self.root)

    def snapshot(self):
        return driver.status(self.out, self.root)

    def execute_stage(self, command, root, log, seconds):
        Path(log).write_text('synthetic bounded execution\n')
        output = Path(command[command.index('--out') + 1])
        output.mkdir(exist_ok=True)
        protocol = output / 'protocol.json'
        if not protocol.exists():
            body = {'synthetic': 'fixed protocol'}
            body['protocol_sha256'] = driver.storage.digest(body)
            driver.storage.atomic_json(protocol, body)
        index = int(Path(log).stem.split('-')[-1]) - 1
        final = index == len(driver.STAGES) - 1
        summary = dict(schema_version=1, candidate_sha256=self.model_hash,
            protocol_sha256=json.loads(protocol.read_text())['protocol_sha256'],
            protocol_file_sha256=driver.storage.file_hash(protocol),
            status='complete' if final else 'in_progress', stage_completed=driver.STAGES[index],
            next_stage=None if final else driver.STAGES[index + 1],
            external_only=True, final_accessed=False, promoted=False,
            submission_eligible=False, final_access_authorized=False,
            development_gate_passed=False)
        driver.storage.atomic_json(output / 'summary.json', summary)
        return 0

    def run_stage(self, **kwargs):
        return driver.run(self.out, root=self.root, **kwargs)

    def state(self, result):
        return result['trials'][driver.TRIAL]['state']

    def test_all_stages_advance_once_and_completed_does_not_rerun(self):
        with patch.object(driver.storage, 'execute', side_effect=self.execute_stage) as execute:
            for index in range(len(driver.STAGES)):
                result = self.run_stage()
                self.assertEqual(self.state(result), 'completed' if index == len(driver.STAGES) - 1 else 'queued')
            self.run_stage()
        self.assertEqual(execute.call_count, 15)
        self.assertEqual(result['trials'][driver.TRIAL]['attempts'], 15)
        self.assertFalse(result['final_accessed'])
        self.assertFalse(result['promoted'])
        self.assertFalse(result['submission_eligible'])

    def test_command_has_only_fixed_module_output_and_next_stage(self):
        trial = self.snapshot()['trials'][driver.TRIAL]
        command = driver.authenticate(self.out, trial['recipe_sha256'], self.root)
        self.assertEqual(command[1:4], ['-m', 'tools.external_pff_development', 'run'])
        self.assertEqual(command[4:], ['--out', str((self.out / 'output').resolve()), '--stage', 'next'])
        self.assertNotIn('--fit', command)
        self.assertNotIn('--recipe', command)
        self.assertNotIn('--split', command)

    def test_source_tampering_blocks_before_launch(self):
        (self.root / 'possession/features.py').write_text('# changed dependency\n')
        with patch.object(driver.storage, 'execute') as execute:
            result = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(result), 'blocked')

    def test_verifier_is_part_of_frozen_source_contract(self):
        (self.root / 'tools/verify_external_pff_development.py').write_text('# modified verifier\n')
        with patch.object(driver.storage, 'execute') as execute:
            result = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(result), 'blocked')

    def test_model_bytes_changed_before_execution_block(self):
        (self.root / driver.REBUILD / 'refit/candidate.json').write_text('{"changed": true}\n')
        with patch.object(driver.storage, 'execute') as execute:
            result = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(result), 'blocked')

    def test_model_bytes_changed_during_execution_fail(self):
        def changed(*args):
            self.execute_stage(*args)
            (self.root / driver.REBUILD / 'refit/candidate.json').write_text('{"changed": true}\n')
            return 0
        with patch.object(driver.storage, 'execute', side_effect=changed):
            result = self.run_stage()
        self.assertEqual(self.state(result), 'failed')

    def test_recipe_tampering_blocks(self):
        path = self.out / 'recipe.json'
        value = json.loads(path.read_text())
        value['development_matches'] = [3814]
        driver.storage.atomic_json(path, value)
        with patch.object(driver.storage, 'execute') as execute:
            result = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(result), 'blocked')

    def test_completed_summary_tampering_blocks_continuation(self):
        with patch.object(driver.storage, 'execute', side_effect=self.execute_stage):
            self.run_stage()
        (self.out / 'output/summary.json').write_text('{}')
        with patch.object(driver.storage, 'execute') as execute:
            result = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(result), 'blocked')

    def test_protocol_tampering_blocks_continuation(self):
        with patch.object(driver.storage, 'execute', side_effect=self.execute_stage):
            self.run_stage()
        (self.out / 'output/protocol.json').write_text('{"changed": true}')
        with patch.object(driver.storage, 'execute') as execute:
            result = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(result), 'blocked')

    def test_protocol_cannot_be_replaced_mid_stage_even_if_summary_updated(self):
        with patch.object(driver.storage, 'execute', side_effect=self.execute_stage):
            self.run_stage()
        def changed(*args):
            self.execute_stage(*args)
            protocol = self.out / 'output/protocol.json'
            replacement = {'replaced': True}
            replacement['protocol_sha256'] = driver.storage.digest(replacement)
            driver.storage.atomic_json(protocol, replacement)
            path = self.out / 'output/summary.json'
            value = json.loads(path.read_text())
            value['protocol_sha256'] = replacement['protocol_sha256']
            value['protocol_file_sha256'] = driver.storage.file_hash(protocol)
            driver.storage.atomic_json(path, value)
            return 0
        with patch.object(driver.storage, 'execute', side_effect=changed):
            result = self.run_stage()
        self.assertEqual(self.state(result), 'failed')

    def test_interrupted_attempt_blocks_entire_loop_and_never_retries(self):
        events = driver.storage.read_ledger(self.out)
        identity = events[0]['recipe_sha256']
        driver.storage.append(self.out, events, driver.TRIAL, 'running', identity, {'pid': 123})
        with patch.object(driver.storage, 'execute') as execute:
            first = self.run_stage()
            second = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(first), 'blocked')
        self.assertEqual(self.state(second), 'blocked')
        self.assertTrue(second['trials'][driver.TRIAL]['detail']['requires_process_review'])

    def test_duplicate_trial_rejected(self):
        events = driver.storage.read_ledger(self.out)
        driver.storage.append(self.out, events, 'other', 'queued', 'a' * 64, {'action': 'run'})
        with patch.object(driver.storage, 'execute') as execute, self.assertRaisesRegex(ValueError, 'single fixed'):
            self.run_stage()
        execute.assert_not_called()

    def test_existing_output_or_stale_log_prevents_fresh_attempt(self):
        (self.out / 'console-0001.log').write_text('previous unrecorded attempt\n')
        with patch.object(driver.storage, 'execute') as execute:
            result = self.run_stage()
        execute.assert_not_called()
        self.assertEqual(self.state(result), 'blocked')

    def test_lock_excludes_runner_but_status_is_readable(self):
        with driver.storage.locked(self.out):
            with self.assertRaisesRegex(RuntimeError, 'holds the lock'):
                self.run_stage()
            self.assertTrue(self.snapshot()['busy'])

    def test_failed_and_timed_out_attempts_stay_consumed(self):
        with patch.object(driver.storage, 'execute', side_effect=subprocess.TimeoutExpired('synthetic', 1)) as execute:
            result = self.run_stage(max_seconds=1)
            self.run_stage()
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(self.state(result), 'failed')
        self.assertLessEqual(execute.call_args.args[-1], 1)

    def test_summary_cannot_authorize_final_access_or_skip_stages(self):
        def forbidden(*args):
            self.execute_stage(*args)
            path = self.out / 'output/summary.json'
            value = json.loads(path.read_text())
            value['final_access_authorized'] = True
            value['stage_completed'] = 'score'
            driver.storage.atomic_json(path, value)
            return 0
        with patch.object(driver.storage, 'execute', side_effect=forbidden):
            result = self.run_stage()
        self.assertEqual(self.state(result), 'failed')

    def test_bad_budget_or_existing_harness_rejected(self):
        for budget in (0, -1, True, 2701):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.run_stage(max_seconds=budget)
        with self.assertRaises(FileExistsError):
            driver.initialize(self.out, self.root)

    def test_truncated_ledger_is_preserved_and_fails_closed(self):
        path = self.out / 'ledger.jsonl'
        with path.open('ab') as stream:
            stream.write(b'{"partial":')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Incomplete ledger'):
            self.run_stage()
        self.assertEqual(path.read_bytes(), before)

    def test_stale_checkpoint_reconstructed_from_ledger(self):
        (self.out / 'checkpoint.json').write_text('{}')
        self.assertEqual(self.state(self.snapshot()), 'queued')

    def test_unrelated_source_addition_does_not_change_import_closure(self):
        (self.root / 'tools/unrelated_experiment.py').write_text('# unrelated\n')
        trial = self.snapshot()['trials'][driver.TRIAL]
        driver.authenticate(self.out, trial['recipe_sha256'], self.root)


if __name__ == '__main__':
    unittest.main()
