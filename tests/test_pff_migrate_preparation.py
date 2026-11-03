import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from possession.io import sha256, write_json
from tools import pff_migrate_preparation as migration


OLD_SOURCE = 'import math\ndef observations():\n    return 1\ndef references(events):\n    return events\n'
NEW_SOURCE = OLD_SOURCE.replace('return events', 'return [] if events is None else events')


class PreparationMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old, self.new = self.root / 'old', self.root / 'new'
        self.old.mkdir()
        self.protocol = migration._load(migration.ROOT / 'models/pff-protocol.json')
        self.old_sources = {migration.ADAPTER: hashlib.sha256(OLD_SOURCE.encode()).hexdigest(), 'other.py': 'a' * 64}
        self.new_sources = dict(self.old_sources, **{migration.ADAPTER: hashlib.sha256(NEW_SOURCE.encode()).hexdigest()})
        self.old_run = {'split_protocol_sha256': self.protocol['protocol_sha256'], 'source_hashes': self.old_sources}
        write_json(self.old / 'run-protocol.json', self.old_run)
        (self.old / 'pff_adapter-frozen.py').write_bytes(OLD_SOURCE.encode())
        (self.root / 'tools').mkdir()
        (self.root / migration.ADAPTER).write_bytes(NEW_SOURCE.encode())
        (self.root / 'configs/baselines').mkdir(parents=True)
        (self.root / 'configs/baselines/v1.5.json').write_bytes((migration.ROOT / 'configs/baselines/v1.5.json').read_bytes())
        self.ids = sorted(self.protocol['split']['train'] + self.protocol['split']['development'])
        for mid in self.ids:
            folder = self.old / 'cache' / str(mid)
            folder.mkdir(parents=True)
            for name in migration.CACHE_FILES:
                (folder / name).write_bytes((str(mid) + ':' + name).encode())
            write_json(folder / 'complete.json', {'match_id': mid, 'rows': 1,
                'identity': {'sources': self.old_sources, 'metadata': self.protocol['metadata_sha256'][str(mid)],
                             'events': 'e' * 64, 'tracking': 'f' * 64},
                'files': {name: sha256(folder / name) for name in migration.CACHE_FILES}})
        patches = [patch.object(migration, 'ROOT', self.root),
                   patch.object(migration.campaign, 'protocol', return_value=self.protocol),
                   patch.object(migration.campaign, 'sources', return_value=self.new_sources),
                   patch.object(migration.campaign, 'freeze', side_effect=self.freeze)]
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def freeze(self, folder, protocol):
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'run-protocol.json', {'source_hashes': self.new_sources, 'split_protocol_sha256': protocol['protocol_sha256']})

    def test_copy_authenticates_all52_and_preserves_originals(self):
        before = {str(p.relative_to(self.old)): sha256(p) for p in self.old.rglob('*') if p.is_file()}
        proof = migration.migrate(self.old, self.new)
        self.assertEqual(len(proof['match_ids']), 52)
        self.assertEqual(before, {str(p.relative_to(self.old)): sha256(p) for p in self.old.rglob('*') if p.is_file()})
        for mid in self.ids:
            manifest = migration._load(self.new / 'cache' / str(mid) / 'complete.json')
            migration.verify_inheritance(manifest, proof, self.new_sources, self.protocol)
            for name, expected in manifest['files'].items():
                self.assertEqual(sha256(self.new / 'cache' / str(mid) / name), expected)
        completion = migration._load(self.new / 'preparation-migration-complete.json')
        self.assertEqual(completion['copied_artifacts'], 364)
        self.assertFalse(completion['labels_read'])

    def test_late_cache_tamper_fails_before_destination_created(self):
        (self.old / 'cache' / str(self.ids[-1]) / 'x.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            migration.migrate(self.old, self.new)
        self.assertFalse(self.new.exists())

    def test_any_fit_or_heldout_artifact_blocks_reuse(self):
        for name in ('fit.json', 'candidate.json', 'development-reference-access.json', 'final-summary.json',
                     'development-predictions-before-references.json', 'final'):
            with self.subTest(name=name):
                path = self.old / name
                path.write_bytes(b'{}')
                with self.assertRaisesRegex(ValueError, 'no fit, candidate'):
                    migration.migrate(self.old, self.new)
                path.unlink()
        self.assertFalse(self.new.exists())

    def test_other_scientific_source_change_blocks_reuse(self):
        self.new_sources['other.py'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Only the reference adapter'):
            migration.migrate(self.old, self.new)
        self.assertFalse(self.new.exists())

    def test_ast_guard_covers_imports_observations_and_reference_signature(self):
        for changed in (NEW_SOURCE.replace('import math', 'import json'),
                        NEW_SOURCE.replace('return 1', 'return 2'),
                        NEW_SOURCE.replace('references(events)', 'references(events=None)')):
            with self.subTest(source=changed):
                (self.root / migration.ADAPTER).write_bytes(changed.encode())
                self.new_sources[migration.ADAPTER] = hashlib.sha256(changed.encode()).hexdigest()
                with self.assertRaisesRegex(ValueError, 'outside reference parsing changed'):
                    migration.migrate(self.old, self.new)
        self.assertFalse(self.new.exists())

    def test_original_source_snapshot_must_match_frozen_hash(self):
        (self.old / 'pff_adapter-frozen.py').write_bytes(NEW_SOURCE.encode())
        with self.assertRaisesRegex(ValueError, 'Original adapter source hash'):
            migration.migrate(self.old, self.new)

    def test_inheritance_rejects_byte_or_provenance_rewriting(self):
        proof = migration.migrate(self.old, self.new)
        original = migration._load(self.new / 'cache' / str(self.ids[0]) / 'complete.json')
        for kind in ('file', 'old_hash', 'ast', 'source', 'metadata'):
            with self.subTest(kind=kind):
                manifest = copy.deepcopy(original)
                if kind == 'file':
                    manifest['files']['x.npy'] = '0' * 64
                elif kind == 'old_hash':
                    manifest['inherited_preparation']['original_manifest_sha256'] = '0' * 64
                elif kind == 'ast':
                    manifest['inherited_preparation']['observation_ast_sha256'] = '0' * 64
                elif kind == 'source':
                    manifest['inherited_preparation']['new_adapter_sha256'] = '0' * 64
                else:
                    manifest['identity']['events'] = '0' * 64
                with self.assertRaises(ValueError):
                    migration.verify_inheritance(manifest, proof, self.new_sources, self.protocol)

    def test_missing_or_extra_cohort_prevents_reuse(self):
        (self.old / 'cache' / str(self.protocol['split']['final'][0])).mkdir()
        with self.assertRaisesRegex(ValueError, 'exactly all training and development'):
            migration.migrate(self.old, self.new)

    def test_existing_output_is_never_overwritten(self):
        self.new.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh, separate'):
            migration.migrate(self.old, self.new)


if __name__ == '__main__':
    unittest.main()
