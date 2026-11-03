import bz2
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import pff_inventory as inventory


def protocol():
    ids = list(inventory.MATCH_IDS)
    return {'labels_used_for_split': False, 'protocol_sha256': 'a' * 64,
            'metadata': [{'match_id': i} for i in ids],
            'split': {'train': ids[:40], 'development': ids[40:52], 'final': ids[52:]}}


class PFFInventoryTests(unittest.TestCase):
    def test_protocol_rejects_overlap_missing_foreign_and_label_based_partition(self):
        for change in ('overlap', 'missing', 'foreign', 'labels'):
            value = protocol()
            if change == 'overlap': value['split']['final'][0] = value['split']['train'][0]
            elif change == 'missing': value['metadata'].pop()
            elif change == 'foreign': value['metadata'][0]['match_id'] = 99999
            else: value['labels_used_for_split'] = True
            with self.assertRaises(ValueError): inventory.expected_files(value)

    def test_full_inventory_is_relative_and_does_not_parse_match_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'data'; config = base / 'protocol.json'
            config.write_text(json.dumps(protocol()), encoding='utf-8')
            payload = b'not parsed as JSON\n' * 100
            for relative in inventory.expected_files(protocol()):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                if relative.endswith('.bz2'): value = bz2.compress(payload)
                elif relative.endswith('.pdf'): value = b'%PDF-1.7\nunit-test signature only'
                elif relative.endswith('.json'): value = b'[intentionally not parsed: held-out labels]'
                else: value = b'header\nvalue\n'
                path.write_bytes(value)
            result = inventory.audit(root, config, workers=2)
            self.assertTrue(result['complete']); self.assertEqual(result['valid_files'], 260)
            self.assertEqual(result['tracking_archives_verified'], 64)
            self.assertEqual(result['tracking_decompressed_bytes'], len(payload) * 64)
            self.assertFalse(result['heldout_labels_inspected'])
            self.assertNotIn(str(base), json.dumps(result))
            entry = next(x for x in result['files'] if x['path'].endswith('.jsonl.bz2'))
            self.assertEqual(entry['decompressed_sha256'], hashlib.sha256(payload).hexdigest())
            (root / inventory.OPTIONAL_FILES[0]).write_bytes(b'PK\x03\x04test')
            self.assertEqual(inventory.audit(root, config)['valid_files'], 261)
            (root / 'Event Data/3812.json').unlink()
            (root / 'unfinished.part').write_bytes(b'incomplete')
            changed = inventory.audit(root, config)
            self.assertFalse(changed['complete'])
            self.assertEqual(changed['missing_files'], ['Event Data/3812.json'])
            self.assertEqual(changed['unexpected_files'], ['unfinished.part'])

    def test_streaming_accepts_concatenated_bzip2_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.bz2'; raw = bz2.compress(b'a' * 5000000) + bz2.compress(b'b' * 100)
            path.write_bytes(raw); result = inventory.scan_bzip2(path)
            self.assertEqual(result['streams'], 2)
            self.assertEqual(result['decompressed_bytes'], 5000100)
            self.assertEqual(result['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertTrue(result['crc_verified'])

    def test_truncated_corrupt_empty_and_trailing_archive_bytes_fail(self):
        good = bz2.compress(b'original tracking rows\n' * 100)
        bad_crc = bytearray(good); bad_crc[-4] ^= 0x10
        for payload in (good[:-1], bytes(bad_crc), b'<html>sign in</html>', good + b'junk', good + b'B', bz2.compress(b'')):
            with self.subTest(payload_length=len(payload)), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'a.bz2'; path.write_bytes(payload)
                with self.assertRaises((ValueError, OSError, EOFError)):
                    inventory.scan_bzip2(path)

    def test_concatenated_stream_headers_can_cross_read_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.bz2'
            path.write_bytes(bz2.compress(b'first') + bz2.compress(b'second'))
            with patch.object(inventory, 'CHUNK', 3):
                result = inventory.scan_bzip2(path)
            self.assertEqual(result['streams'], 2)
            self.assertEqual(result['decompressed_sha256'], hashlib.sha256(b'firstsecond').hexdigest())

    def test_signatures_reject_wrong_payload_without_label_semantics(self):
        for name, payload in [('x.json', b'<html>denied</html>'), ('x.pdf', b'not pdf'),
                              ('x.csv', b' <html>wrong</html>'), ('x.docx', b'notzip'), ('x.json', b'')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name; path.write_bytes(payload)
                with self.assertRaises(ValueError): inventory.scan_regular(path)

    def test_file_change_during_scan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'x.json').write_bytes(b'[]')
            with patch.object(inventory, '_snapshot', side_effect=[(2, 1), (2, 2)]):
                result = inventory.audit_file(root, 'x.json')
            self.assertEqual(result['status'], 'invalid')
            self.assertIn('changed', result['error'])

    def test_path_escape_and_missing_file_fail_without_machine_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'; root.mkdir()
            (root.parent / 'outside.json').write_bytes(b'[]')
            result = inventory.audit_file(root, '../outside.json')
            self.assertEqual(result['status'], 'invalid')
            self.assertNotIn(tmp, json.dumps(result))
            self.assertEqual(inventory.audit_file(root, 'missing.json')['status'], 'missing')


if __name__ == '__main__':
    unittest.main()
