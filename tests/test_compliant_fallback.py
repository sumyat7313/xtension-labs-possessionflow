"""Synthetic-only fallback packaging and inference-boundary checks."""
import contextlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import build_compliant_fallback as builder


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.code = builder.inference_sources()
        self.model = b'{"synthetic_model_fixture":true}\n'
        self.config = b'{"synthetic_config_fixture":true}\n'
        for name, raw in self.code.items():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for name in ('requirements-solution.txt', 'LICENSE', builder.REPORT):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((builder.ROOT/name).read_bytes())
        self.evidence = dict(model_bytes=self.model, config_bytes=self.config,
            lineage_bytes=b'{"synthetic_lineage_fixture":true}\n',
            scientific_sources={name:builder._sha(raw) for name,raw in self.code.items()},
            lineage_sha256='a'*64, source_bundle_sha256='b'*64,
            declared_lineage_valid=True, actual_model_ancestry_authenticated=True,
            provenance_semantically_certified=False, configuration_origin='synthetic fixture')
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(builder,'ROOT',self.root))
        self.stack.enter_context(patch.object(builder,'CANDIDATE_SHA',builder._sha(self.model)))
        self.stack.enter_context(patch.object(builder,'CONFIG_SHA',builder._sha(self.config)))
        self.stack.enter_context(patch.object(builder,'authenticate_clean',return_value=self.evidence))

    def prepare(self, suffix='one'):
        out = self.root/'runs'/('compliant-fallback-'+suffix)
        builder.prepare(out)
        return out

    def runner(self, out):
        # Exec avoids importlib producing a bytecode cache inside the sealed source.
        space = {'__name__':'synthetic_test_runner', '__file__':str(out/'source/run_fallback.py')}
        exec(compile((out/'source/run_fallback.py').read_bytes(),'run_fallback.py','exec'),space)
        return space

    def test_exact_source_closure_excludes_training_and_old_entrypoints(self):
        self.assertEqual(set(self.code),builder.INFERENCE_FILES)
        for forbidden in ('possession/pipeline.py','possession/bundle.py','possession/__main__.py'):
            self.assertNotIn(forbidden,self.code)
        self.assertFalse(any(name.startswith(('research/','tools/','data/','models/')) for name in self.code))

    def test_archive_and_runner_reproducible_across_output_paths(self):
        one,two = self.prepare('one'),self.prepare('two')
        self.assertEqual((one/'Xtension Labs_source.zip').read_bytes(),(two/'Xtension Labs_source.zip').read_bytes())
        self.assertEqual(builder.verify_package(one)['source_files'],len(builder.EXPORT_FILES)+1)
        self.assertFalse(builder.verify_package(one)['prediction_generated'])
        self.assertFalse(builder.verify_package(one)['submission_ready'])

    def test_attempt_cannot_be_overwritten_or_written_outside_allowed_directory(self):
        out = self.prepare()
        with self.assertRaises(ValueError): builder.prepare(out)
        with self.assertRaises(ValueError): builder.prepare(self.root/'submission')
        with self.assertRaises(ValueError): builder.prepare(self.root/'runs/nested/compliant-fallback-test')

    def test_tampered_fixed_model_is_rejected(self):
        out=self.prepare()
        (out/'source/models/fallback.json').write_text('{}')
        with self.assertRaises(ValueError):builder.verify_package(out)
        with self.assertRaises(ValueError):self.runner(out)['verify_source']()

    def test_unlisted_source_file_rejected_by_both_verifiers(self):
        out=self.prepare()
        (out/'source/extra.txt').write_text('synthetic')
        with self.assertRaises(ValueError):builder.verify_package(out)
        with self.assertRaises(ValueError):self.runner(out)['verify_source']()

    def test_runtime_git_directory_allowed_but_export_rejects_it(self):
        out=self.prepare()
        (out/'source/.git').mkdir()
        (out/'source/.git/config').write_text('synthetic Git metadata')
        self.runner(out)['verify_source']()
        with self.assertRaises(ValueError):builder.verify_package(out)

    def test_runtime_csv_contract_is_exact(self):
        runner=self.runner(self.prepare())
        # Format tests use invented labels; no challenge reference is read.
        path=self.root/'synthetic.csv'
        with path.open('w',encoding='ascii',newline='') as stream:
            stream.write('frame,label\n')
            for i in range(builder.ROWS):stream.write(f'{i},H\n')
        self.assertEqual(runner['validate'](path),builder.ROWS)
        for content in ('frame,label,extra\n0,H,x\n','frame,label\n0,h\n',
                        'frame,label\n0, H\n','frame,label\n00,H\n','frame,label\n1,H\n'):
            path.write_text(content)
            with self.subTest(content=content):
                with self.assertRaises(ValueError):runner['validate'](path)

    def test_inference_rejects_source_output_and_synthetic_data(self):
        out=self.prepare()
        runner=self.runner(out)
        data=self.root/'invented-data';data.mkdir()
        (data/'.synthetic-fixture.json').write_text('{}')
        with self.assertRaises(ValueError):runner['predict'](data,out/'source/predictions')
        with self.assertRaises(ValueError):runner['predict'](data,self.root/'predictions')
        self.assertFalse((self.root/'predictions').exists())

    def synthetic_inference(self, mutate=False):
        import numpy as np
        import games
        import possession.io
        import possession.config
        import possession.learned
        out=self.prepare()
        runner=self.runner(out)
        data=self.root/'invented-data';data.mkdir()
        inputs={key:data/(key+'.fixture') for key in ('atd_meta','atd_raw','smart_tagging')}
        for path in inputs.values():path.write_bytes(b'invented byte fixture; not tracking data')
        labels=np.full(builder.ROWS,'H',dtype='<U1');labels[1::4]='A';labels[2::4]='D';labels[3::4]='X'
        def infer(*args):
            if mutate:inputs['atd_raw'].write_bytes(b'changed synthetic bytes')
            return labels,None,None,None,[]
        with contextlib.ExitStack() as mocks:
            mocks.enter_context(patch.object(games,'asset_paths',return_value=inputs))
            read_tracking=mocks.enter_context(patch.object(possession.io,'read_tracking',return_value=object()))
            read_events=mocks.enter_context(patch.object(possession.io,'read_events',return_value=object()))
            mocks.enter_context(patch.object(possession.config,'load_config',return_value={}))
            mocks.enter_context(patch.object(possession.learned,'load_model',return_value={}))
            inference=mocks.enter_context(patch.object(possession.learned,'infer_learned',side_effect=infer))
            delivery=self.root/'reproduced'
            if mutate:
                with self.assertRaises(ValueError):runner['predict'](data,delivery)
                self.assertTrue((delivery/'started.json').exists())
                self.assertFalse((delivery/'receipt.json').exists())
            else:
                result=runner['predict'](data,delivery)
                self.assertEqual(result['rows'],builder.ROWS)
                self.assertFalse(result['references_accessed'])
                self.assertFalse(result['final_pff_accessed'])
                self.assertFalse(result['submission_ready'])
                self.assertEqual(set(p.name for p in delivery.iterdir()),{'started.json','receipt.json','Xtension Labs.csv'})
                self.assertEqual(runner['validate'](delivery/'Xtension Labs.csv'),builder.ROWS)
                with self.assertRaises(ValueError):runner['predict'](data,delivery)
                self.assertEqual(inference.call_count,1)
            read_tracking.assert_called_once()
            read_events.assert_called_once()
        self.assertFalse((out/'source/Xtension Labs.csv').exists())
        builder.verify_package(out)

    def test_synthetic_inference_writes_only_delivery_and_aggregate_receipt(self):
        self.synthetic_inference()

    def test_mutated_input_consumes_attempt_without_success_receipt(self):
        self.synthetic_inference(mutate=True)

    def test_owned_child_is_killed_and_reaped_on_timeout(self):
        from tools import owned_python_launch as owned
        out=self.prepare()
        process=Mock()
        process.wait.side_effect=[TimeoutError('synthetic timeout'),0]
        process.poll.return_value=None
        job=Mock()
        with patch.object(builder.subprocess,'Popen',return_value=process) as launch, \
             patch.object(owned,'contain_and_resume',return_value=job), \
             patch.object(owned,'windows_creationflags',return_value=123):
            with self.assertRaises(TimeoutError):builder.predict(out,self.root/'invented-data')
        job.close.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count,2)
        self.assertEqual(launch.call_args.kwargs['creationflags'],123)
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            self.assertEqual(launch.call_args.kwargs['env'][name],'1')


if __name__=='__main__':unittest.main()
