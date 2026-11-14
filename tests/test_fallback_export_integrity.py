"""Independent source-export review using invented bytes, never match records."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import build_compliant_fallback as export


class ExportIntegrityReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        original = export.ROOT
        self.sources = export.inference_sources(original)
        for name, raw in self.sources.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for name in (export.REPORT, 'requirements-solution.txt', 'LICENSE',
                     'models/external-rebuild-001-research.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((original / name).read_bytes())
        # Deliberately invented model bytes: authenticate_clean is mocked below.
        # This tests the export boundary, not scientific model authentication.
        model = b'{"synthetic_review_fixture":true}\n'
        config = b'{}\n'
        self.evidence = dict(
            model_bytes=model, config_bytes=config, lineage_bytes=b'{}\n',
            scientific_sources={name: export._sha(raw) for name, raw in self.sources.items()},
            lineage_sha256=export._sha(b'{}\n'), source_bundle_sha256='0' * 64,
            declared_lineage_valid=True, actual_model_ancestry_authenticated=True,
            provenance_semantically_certified=False, configuration_origin='synthetic test only')
        for name, value in [('ROOT', self.root), ('CANDIDATE_SHA', export._sha(model)),
                            ('CONFIG_SHA', export._sha(config))]:
            context = patch.object(export, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(export, 'authenticate_clean', return_value=self.evidence)
        context.start()
        self.addCleanup(context.stop)
        self.destination = self.root / 'runs' / 'compliant-fallback-independent-review'
        export.prepare(self.destination)

    def _self_consistent_extra_file(self, name):
        """Rehash all mutable receipts as an accidental broad packager could."""
        source = self.destination / 'source'
        raw = b'SYNTHETIC FORBIDDEN CONTENT; NOT REAL DATA\n'
        extra = source / name
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(raw)
        manifest_path = source / 'source-manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['files'][name] = export._sha(raw)
        manifest_path.write_bytes(export._canonical(manifest) + b'\n')
        archive_path = self.destination / f'{export.TEAM}_source.zip'
        with zipfile.ZipFile(archive_path, 'w') as archive:
            for path in sorted(source.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(source).as_posix())
        receipt_path = self.destination / 'package.json'
        receipt = json.loads(receipt_path.read_bytes())
        receipt['source_manifest_sha256'] = export._sha(manifest_path.read_bytes())
        receipt['source_archive_sha256'] = export._sha(archive_path.read_bytes())
        receipt['source_files'] = len(manifest['files']) + 1
        receipt_path.write_bytes(export._canonical(receipt) + b'\n')

    def test_clean_synthetic_export_has_no_data_history_or_prediction_paths(self):
        receipt = export.verify_package(self.destination)
        self.assertFalse(receipt['submission_ready'])
        with zipfile.ZipFile(self.destination / f'{export.TEAM}_source.zip') as archive:
            names = archive.namelist()
        for name in names:
            self.assertNotIn('.git', Path(name).parts)
            self.assertNotIn('gt', Path(name).parts)
            self.assertFalse(name.endswith('.csv'))
        self.assertEqual([name for name in names if name.startswith('models/')],
                         ['models/fallback.json'])

    def test_rehashed_extra_model_is_not_approved_by_mutable_receipts(self):
        self._self_consistent_extra_file('models/pff-restricted.json')
        with self.assertRaises(ValueError):
            export.verify_package(self.destination)

    def test_rehashed_frame_records_are_not_approved_by_mutable_receipts(self):
        self._self_consistent_extra_file('data/predictions.csv')
        with self.assertRaises(ValueError):
            export.verify_package(self.destination)

    def test_rehashed_changed_input_reader_is_not_approved_by_mutable_receipts(self):
        self._self_consistent_extra_file('possession/io.py')
        with self.assertRaises(ValueError):
            export.verify_package(self.destination)

    def test_git_add_and_clone_preserve_all_export_bytes_with_autocrlf(self):
        git = shutil.which('git')
        self.assertIsNotNone(git, 'Git is required for publication-byte verification')
        source = self.destination / 'source'
        expected = {p.relative_to(source).as_posix(): p.read_bytes()
                    for p in source.rglob('*') if p.is_file()}
        self.assertTrue(any(b'\r\n' in raw for name, raw in expected.items()
                            if name.endswith('.py')), 'Exercise original CRLF source bytes')
        self.assertEqual(expected['.gitattributes'], b'* -text\n')
        environment = dict(os.environ, GIT_CONFIG_NOSYSTEM='1',
                           GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')
        # Ignore any parent process's alternate repository/index and injected config.
        for key in list(environment):
            if key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_CONFIG_PARAMETERS') \
                    or key.startswith(('GIT_CONFIG_COUNT', 'GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')):
                environment.pop(key)
        empty = self.root / 'empty-git-template-and-hooks'
        empty.mkdir()

        def command(directory, *arguments, autocrlf='true'):
            result = subprocess.run(
                [git, '-c', f'core.autocrlf={autocrlf}', '-c', 'core.safecrlf=false',
                 '-c', 'core.longpaths=true',
                 '-c', f'init.templateDir={empty}', '-c', f'core.hooksPath={empty}',
                 '-c', 'commit.gpgSign=false', '-c', 'user.name=Synthetic Export Review',
                 '-c', 'user.email=export-review@example.invalid', *arguments],
                cwd=directory, env=environment, check=False, capture_output=True, text=True,
                timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr[:2000])
            return result

        command(source, 'init', '--quiet')
        command(source, 'add', '--all')
        command(source, 'commit', '--quiet', '-m', 'Synthetic byte-preservation fixture')
        # The strict Windows-style path is the primary regression; input/false
        # also cover common Linux/macOS checkout configurations.
        for autocrlf in ('true', 'input', 'false'):
            with self.subTest(checkout_autocrlf=autocrlf):
                clone = self.root / ('clone-autocrlf-' + autocrlf)
                command(self.root, 'clone', '--quiet', '--local', '--no-hardlinks',
                        source.relative_to(self.root).as_posix(), clone.name,
                        autocrlf=autocrlf)
                actual = {p.relative_to(clone).as_posix(): p.read_bytes()
                          for p in clone.rglob('*')
                          if p.is_file() and p.relative_to(clone).parts[0] != '.git'}
                self.assertEqual(set(actual), set(expected))
                for name, raw in expected.items():
                    self.assertEqual(actual[name], raw, name)
                scope = {'__name__': 'synthetic_export_review',
                         '__file__': str(clone / 'run_fallback.py')}
                exec(compile(actual['run_fallback.py'], 'run_fallback.py', 'exec'), scope)
                scope['verify_source']()


if __name__ == '__main__':
    unittest.main()
