import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import experiment_harness as harness


class ExperimentHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('tools', 'possession', 'evaluation', 'data/external/idsse'):
            (self.root / name).mkdir(parents=True)
        (self.root / 'tools/external_rebuild.py').write_text('from possession import model\n', encoding='utf-8')
        (self.root / 'possession/model.py').write_text('# synthetic dependency\n', encoding='utf-8')
        self.out = self.root / 'runs/harness'
        harness.initialize(self.out, self.root)
        self.spec = dict(schema_version=1, trial_id='first', action='run', parameters=dict(
            schema_version=1, seed=10527, epochs=8, control_l2=.001, transition_weight=1.5))

    def enqueue(self, spec=None):
        return harness.enqueue(self.out, spec or self.spec, self.root)

    def result(self, command, root, log, seconds, status='complete'):
        Path(log).write_text('synthetic run only\n', encoding='utf-8')
        output = Path(command[command.index('--out') + 1])
        output.mkdir(exist_ok=True)
        stage = 'next' if (output / 'summary.json').is_file() else 'synthetic'
        harness.atomic_json(output / 'summary.json', dict(status=status, stage_completed=stage,
            next_stage='next' if status == 'in_progress' else None, external_only=True,
            final_accessed=False, promoted=False, submission_eligible=False, development_gate_passed=False))
        return 0

    def run_trial(self, **kwargs):
        return harness.run(self.out, root=self.root, **kwargs)

    def test_success_is_not_promotion_and_never_reruns(self):
        self.enqueue()
        with patch.object(harness, 'execute', side_effect=self.result) as execute:
            first = self.run_trial()
            second = self.run_trial()
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(first['trials']['first']['state'], 'completed')
        self.assertEqual(second['trials']['first']['attempts'], 1)
        self.assertFalse(first['promoted'])
        self.assertFalse(first['submission_eligible'])
        self.assertFalse(first['final_accessed'])

    def test_recipe_freezes_before_execute_and_command_is_allowlisted(self):
        identity = self.enqueue()
        recipe, command, _ = harness.authenticate_trial(self.out, 'first', identity, self.root)
        self.assertEqual(command[1:4], ['-m', 'tools.external_rebuild', 'run'])
        self.assertEqual(command[-2:], ['--stage', 'next'])
        self.assertEqual(recipe['data_directory'], 'data/external/idsse')
        self.assertEqual(harness.read_ledger(self.out)[0]['state'], 'queued')
        self.assertIn('possession/model.py', recipe['source_hashes'])

    def test_frozen_recipe_tampering_blocks_before_execute(self):
        self.enqueue()
        path = self.out / 'trials/first/recipe.json'
        data = json.loads(path.read_text())
        data['module'] = 'os'
        harness.atomic_json(path, data)
        with patch.object(harness, 'execute') as execute:
            result = self.run_trial()
        execute.assert_not_called()
        self.assertEqual(result['trials']['first']['state'], 'blocked')

    def test_parameter_tampering_blocks(self):
        self.enqueue()
        harness.atomic_json(self.out / 'trials/first/parameters.json', {'epochs': 12})
        with patch.object(harness, 'execute') as execute:
            result = self.run_trial()
        execute.assert_not_called()
        self.assertEqual(result['trials']['first']['state'], 'blocked')

    def test_imported_source_drift_blocks(self):
        self.enqueue()
        (self.root / 'possession/model.py').write_text('# changed dependency\n')
        with patch.object(harness, 'execute') as execute:
            result = self.run_trial()
        execute.assert_not_called()
        self.assertEqual(result['trials']['first']['state'], 'blocked')

    def test_edit_during_run_fails_even_after_successful_child(self):
        self.enqueue()
        def edited(*args):
            result = self.result(*args)
            (self.root / 'possession/model.py').write_text('# changed while fitting\n')
            return result
        with patch.object(harness, 'execute', side_effect=edited):
            result = self.run_trial()
        self.assertEqual(result['trials']['first']['state'], 'failed')

    def test_failure_persists_and_cannot_reuse_trial_id(self):
        self.enqueue()
        with patch.object(harness, 'execute', return_value=2) as execute:
            result = self.run_trial()
            self.run_trial()
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result['trials']['first']['state'], 'failed')
        with self.assertRaisesRegex(ValueError, 'cannot be reused'):
            self.enqueue()
        self.assertEqual([e['state'] for e in harness.read_ledger(self.out)], ['queued', 'running', 'failed'])

    def test_crashed_running_attempt_blocks_without_retry(self):
        identity = self.enqueue()
        events = harness.read_ledger(self.out)
        harness.append(self.out, events, 'first', 'running', identity, {'pid': 123})
        with patch.object(harness, 'execute') as execute:
            result = self.run_trial()
        execute.assert_not_called()
        self.assertEqual(result['trials']['first']['state'], 'blocked')
        self.assertIn('Interrupted', result['trials']['first']['detail']['reason'])

    def test_interrupted_attempt_blocks_other_queued_trials_forever(self):
        identity = self.enqueue()
        self.enqueue(dict(self.spec, trial_id='second'))
        events = harness.read_ledger(self.out)
        harness.append(self.out, events, 'first', 'running', identity, {'pid': 123})
        with patch.object(harness, 'execute') as execute:
            first = self.run_trial(max_trials=2)
            second = self.run_trial(max_trials=2)
        execute.assert_not_called()
        self.assertEqual(first['trials']['first']['state'], 'blocked')
        self.assertEqual(second['trials']['second']['state'], 'queued')
        self.assertIn('orphan-process', second['blocked_reason'])

    def test_unrelated_experimental_source_does_not_invalidate_trial(self):
        identity = self.enqueue()
        (self.root / 'tools/unrelated_experiment.py').write_text('# unrelated\n')
        harness.authenticate_trial(self.out, 'first', identity, self.root)

    def test_transitive_relative_import_is_frozen(self):
        (self.root / 'possession/model.py').write_text('from . import features\n')
        (self.root / 'possession/features.py').write_text('# second dependency\n')
        identity = self.enqueue()
        recipe, _, _ = harness.authenticate_trial(self.out, 'first', identity, self.root)
        self.assertIn('possession/features.py', recipe['source_hashes'])
        (self.root / 'possession/features.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'Scientific source changed'):
            harness.authenticate_trial(self.out, 'first', identity, self.root)

    def test_stale_checkpoint_recovers_from_authoritative_ledger(self):
        self.enqueue()
        (self.out / 'checkpoint.json').write_text('{}')
        result = harness.status(self.out, self.root)
        self.assertEqual(result['trials']['first']['state'], 'queued')

    def test_partial_ledger_fails_closed_without_truncation(self):
        self.enqueue()
        path = self.out / 'ledger.jsonl'
        with path.open('ab') as stream:
            stream.write(b'{"incomplete":')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Incomplete ledger'):
            self.run_trial()
        self.assertEqual(before, path.read_bytes())

    def test_ledger_tampering_detected(self):
        self.enqueue()
        path = self.out / 'ledger.jsonl'
        row = json.loads(path.read_text())
        row['state'] = 'completed'
        path.write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.run_trial()

    def test_lock_excludes_second_runner_but_status_returns_snapshot(self):
        self.enqueue()
        with harness.locked(self.out):
            with self.assertRaisesRegex(RuntimeError, 'holds the lock'):
                self.run_trial()
            self.assertTrue(harness.status(self.out, self.root)['busy'])
        self.assertFalse(harness.status(self.out, self.root)['busy'])

    def test_iteration_budget_leaves_other_trials_queued(self):
        self.enqueue()
        second = copy.deepcopy(self.spec)
        second['trial_id'] = 'second'
        self.enqueue(second)
        with patch.object(harness, 'execute', side_effect=self.result) as execute:
            result = self.run_trial(max_trials=1)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result['trials']['second']['state'], 'queued')

    def test_timeout_is_consumed_and_bounded(self):
        self.enqueue()
        with patch.object(harness, 'execute', side_effect=subprocess.TimeoutExpired('synthetic', 1)) as execute:
            result = self.run_trial(max_seconds=2, trial_seconds=100)
        self.assertLessEqual(execute.call_args.args[-1], 2)
        self.assertEqual(result['trials']['first']['state'], 'failed')

    def test_bad_budgets_rejected(self):
        for params in ({'max_trials': 0}, {'max_trials': True}, {'max_seconds': 86401}, {'trial_seconds': -1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.run_trial(**params)

    def test_forbidden_actions_modules_paths_and_shell_arguments_rejected(self):
        for action in ('final', 'prepare-final', 'promote', 'submit', 'run;whoami'):
            value = copy.deepcopy(self.spec)
            value['action'] = action
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.enqueue(value)
        for key, value in (('module', 'os'), ('args', ['--final']), ('command', 'python -c pass')):
            bad = dict(self.spec, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.enqueue(bad)
        for trial_id in ('../escape', 'x/y', 'X', 'x;whoami', ''):
            bad = dict(self.spec, trial_id=trial_id)
            with self.subTest(trial_id=trial_id), self.assertRaises(ValueError):
                self.enqueue(bad)

    def test_unknown_parameter_or_nonfinite_rejected(self):
        for params in (dict(self.spec['parameters'], profile='target'),
                       dict(self.spec['parameters'], epochs=True),
                       dict(self.spec['parameters'], seed=-1),
                       dict(self.spec['parameters'], control_l2=float('nan')),
                       dict(self.spec['parameters'], transition_weight=100)):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.enqueue(dict(self.spec, parameters=params))

    def test_output_outside_runs_rejected_and_existing_run_not_overwritten(self):
        with self.assertRaises(ValueError):
            harness.initialize(self.root / 'models/harness', self.root)
        with self.assertRaises(FileExistsError):
            harness.initialize(self.out, self.root)

    def test_existing_unexplained_output_blocks(self):
        self.enqueue()
        (self.out / 'trials/first/output').mkdir()
        with patch.object(harness, 'execute') as execute:
            result = self.run_trial()
        execute.assert_not_called()
        self.assertEqual(result['trials']['first']['state'], 'blocked')

    def test_missing_or_forbidden_result_cannot_complete(self):
        self.enqueue()
        def bad_result(*args):
            self.result(*args)
            path = self.out / 'trials/first/output/summary.json'
            data = json.loads(path.read_text())
            data['final_accessed'] = True
            harness.atomic_json(path, data)
            return 0
        with patch.object(harness, 'execute', side_effect=bad_result):
            result = self.run_trial()
        self.assertEqual(result['trials']['first']['state'], 'failed')

    def test_successful_stage_continues_without_replaying_consumed_work(self):
        self.enqueue()
        def stage(*args):
            return self.result(*args, status='in_progress')
        with patch.object(harness, 'execute', side_effect=stage):
            first = self.run_trial()
        self.assertEqual(first['trials']['first']['state'], 'queued')
        with patch.object(harness, 'execute', side_effect=self.result) as execute:
            second = self.run_trial()
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(second['trials']['first']['state'], 'completed')
        self.assertEqual(second['trials']['first']['attempts'], 2)
        self.assertTrue((self.out / 'trials/first/console-0001.log').is_file())
        self.assertTrue((self.out / 'trials/first/console-0002.log').is_file())

    def test_stage_summary_mutation_blocks_continuation(self):
        self.enqueue()
        with patch.object(harness, 'execute', side_effect=lambda *args: self.result(*args, status='in_progress')):
            self.run_trial()
        (self.out / 'trials/first/output/summary.json').write_text('{}')
        with patch.object(harness, 'execute') as execute:
            result = self.run_trial()
        execute.assert_not_called()
        self.assertEqual(result['trials']['first']['state'], 'blocked')

    def test_stage_cannot_repeat_or_skip_authenticated_next_stage(self):
        self.enqueue()
        with patch.object(harness, 'execute', side_effect=lambda *args: self.result(*args, status='in_progress')):
            self.run_trial()
        def repeated(*args):
            self.result(*args)
            path = self.out / 'trials/first/output/summary.json'
            value = json.loads(path.read_text())
            value['stage_completed'] = 'synthetic'
            harness.atomic_json(path, value)
            return 0
        with patch.object(harness, 'execute', side_effect=repeated):
            result = self.run_trial()
        self.assertEqual(result['trials']['first']['state'], 'failed')
        self.assertIn('repeats', result['trials']['first']['detail']['reason'])

    def test_sibling_registered_dataset_is_selected_without_copy(self):
        nested_root = self.root / 'outputs/project'
        nested_root.mkdir(parents=True)
        sibling = nested_root.parent / 'more-than-a-hack-2027/data/external/idsse'
        sibling.mkdir(parents=True)
        relative, actual = harness.data_directory(nested_root)
        self.assertEqual(relative, '../more-than-a-hack-2027/data/external/idsse')
        self.assertEqual(actual, sibling.resolve())
        with self.assertRaisesRegex(ValueError, 'registered'):
            harness.data_directory(nested_root, '../somewhere/data')

    def test_subprocess_uses_argument_list_no_shell_and_single_thread_limits(self):
        with patch.object(harness.subprocess, 'Popen') as process:
            process.return_value.wait.return_value = 0
            code = harness.execute(['python', '-m', 'tools.external_rebuild', 'plan'], self.root,
                                   self.root / 'child.log', 10)
        self.assertEqual(code, 0)
        kwargs = process.call_args.kwargs
        self.assertFalse(kwargs['shell'])
        self.assertEqual(kwargs['env']['OPENBLAS_NUM_THREADS'], '1')
        self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')

    def test_timeout_requests_process_tree_cleanup_before_propagating(self):
        with patch.object(harness.subprocess, 'Popen') as process, patch.object(harness, 'stop_process_tree') as stop:
            process.return_value.wait.side_effect = subprocess.TimeoutExpired('synthetic', .1)
            with self.assertRaises(subprocess.TimeoutExpired):
                harness.execute(['python'], self.root, self.root / 'timeout.log', .1)
        stop.assert_called_once_with(process.return_value)

    def test_completed_stage_does_not_authorize_replaying_failed_stage(self):
        self.enqueue()
        with patch.object(harness, 'execute', side_effect=lambda *args: self.result(*args, status='in_progress')):
            self.run_trial()
        with patch.object(harness, 'execute', return_value=1):
            self.run_trial()
        with patch.object(harness, 'execute') as execute:
            result = self.run_trial()
        execute.assert_not_called()
        self.assertEqual(result['trials']['first']['state'], 'failed')

    def test_preflight_does_not_need_training_summary(self):
        self.enqueue(dict(self.spec, action='preflight'))
        def preflight(command, root, log, seconds):
            Path(log).write_text('synthetic preflight\n')
            return 0
        with patch.object(harness, 'execute', side_effect=preflight):
            result = self.run_trial()
        self.assertEqual(result['trials']['first']['state'], 'completed')


if __name__ == '__main__':
    unittest.main()
