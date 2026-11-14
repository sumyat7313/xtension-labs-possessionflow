"""Independent source-export review using invented bytes, never match records."""
import json
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
