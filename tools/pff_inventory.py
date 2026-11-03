"""Audit delivered PFF bytes without parsing any match labels or observations."""
from __future__ import annotations

import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATCH_IDS = tuple(range(3812, 3860)) + tuple(range(10502, 10518))
FOLDERS = ('Metadata', 'Rosters', 'Event Data', 'Tracking Data')
SUPPORT_FILES = ('competitions.csv', 'players.csv',
                 'Event Data/PFF FC Event Data Specification v2.5.pdf',
                 'Tracking Data/PFF FC Tracking Data Specification v2.2.pdf')
OPTIONAL_FILES = ('PFF FC Change Log.docx',)
CHUNK = 1024 * 1024


def expected_files(protocol):
    """Use fixture metadata and split identities only, never match label files."""
    if not isinstance(protocol, dict) or protocol.get('labels_used_for_split') is not False:
        raise ValueError('Protocol must declare label-independent partitioning')
    metadata = protocol.get('metadata')
    if not isinstance(metadata, list):
        raise ValueError('Protocol metadata missing')
    ids = [item.get('match_id') for item in metadata if isinstance(item, dict)]
    if (len(ids) != len(MATCH_IDS) or any(type(i) is not int for i in ids)
            or len(set(ids)) != len(ids) or set(ids) != set(MATCH_IDS)):
        raise ValueError('Protocol must identify all 64 current-release fixtures')
    split = protocol.get('split')
    if not isinstance(split, dict) or set(split) != {'train', 'development', 'final'}:
        raise ValueError('Protocol needs train/development/final partition')
    groups = list(split.values())
    if any(not isinstance(g, list) or not g or any(type(i) is not int for i in g) for g in groups):
        raise ValueError('Invalid partition identities')
    partition = [i for group in groups for i in group]
    if len(partition) != len(ids) or set(partition) != set(ids):
        raise ValueError('Partition must include every fixture exactly once')
    paths = [f'{folder}/{i}' + ('.jsonl.bz2' if folder == 'Tracking Data' else '.json')
             for folder in FOLDERS for i in sorted(ids)]
    return sorted(paths + list(SUPPORT_FILES))


def _snapshot(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def scan_bzip2(path):
    """Validate all concatenated BZip2 streams and CRCs; reject trailing junk.

    max_length bounds expansion memory. Decompressed bytes are hashed and
    discarded; no JSON decoder or field access is involved.
    """
    compressed = hashlib.sha256(); expanded = hashlib.sha256()
    compressed_size = 0; expanded_size = 0; streams = 0
    decoder = None; pending = b''
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(CHUNK)
            if chunk:
                compressed.update(chunk); compressed_size += len(chunk)
                pending += chunk
            while pending or decoder is not None and not decoder.needs_input:
                if decoder is None:
                    if len(pending) < 4:
                        break
                    if pending[:3] != b'BZh' or pending[3:4] not in b'123456789':
                        raise ValueError('Invalid BZip2 header or trailing bytes')
                    decoder = bz2.BZ2Decompressor()
                output = decoder.decompress(pending, max_length=CHUNK)
                pending = b''
                if output:
                    expanded.update(output); expanded_size += len(output)
                if decoder.eof:
                    streams += 1
                    pending = decoder.unused_data
                    decoder = None
            if not chunk:
                if pending or decoder is not None or streams == 0:
                    raise ValueError('Truncated BZip2 stream or trailing bytes')
                break
    if expanded_size == 0:
        raise ValueError('Empty tracking archive')
    return {'bytes': compressed_size, 'sha256': compressed.hexdigest(),
            'compression': 'bzip2', 'streams': streams, 'decompressed_bytes': expanded_size,
            'decompressed_sha256': expanded.hexdigest(), 'decompression_to_eof': True,
            'crc_verified': True, 'semantic_content_read': False}


def scan_regular(path):
    digest = hashlib.sha256(); size = 0; prefix = b''
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b''):
            digest.update(chunk); size += len(chunk)
            if len(prefix) < 4096:
                prefix += chunk[:4096 - len(prefix)]
    if size == 0:
        raise ValueError('Empty file')
    trimmed = prefix.removeprefix(b'\xef\xbb\xbf').lstrip()
    suffix = path.suffix.lower()
    if suffix == '.json' and trimmed[:1] not in (b'[', b'{'):
        raise ValueError('JSON container signature missing')
    if suffix == '.pdf' and not prefix.startswith(b'%PDF-'):
        raise ValueError('PDF signature missing')
    if suffix == '.docx' and not prefix.startswith(b'PK\x03\x04'):
        raise ValueError('DOCX ZIP signature missing')
    if suffix == '.csv' and trimmed[:1] == b'<':
        raise ValueError('Unexpected HTML in CSV file')
    return {'bytes': size, 'sha256': digest.hexdigest(),
            'signature_checked': suffix.lstrip('.'), 'semantic_content_read': False,
            'full_syntax_checked': False}


def audit_file(data_root, relative):
    path = data_root / relative
    result = {'path': relative}
    try:
        if path.is_symlink() or not path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError('Path escapes data directory')
        if not path.is_file():
            return dict(result, status='missing')
        before = _snapshot(path)
        content = scan_bzip2(path) if relative.endswith('.bz2') else scan_regular(path)
        if before != _snapshot(path) or before[0] != content['bytes']:
            raise ValueError('File changed during inventory')
        return dict(result, status='ok', **content)
    except (OSError, EOFError, ValueError) as exc:
        # Do not include OS exception strings: they can expose machine paths.
        message = str(exc) if type(exc) is ValueError else type(exc).__name__
        return dict(result, status='invalid', error=message)


def audit(data_root, protocol_path, *, workers=1, progress=None):
    data_root, protocol_path = Path(data_root), Path(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode('utf-8-sig'))
    required = expected_files(protocol)
    optional = [p for p in OPTIONAL_FILES if (data_root / p).exists()]
    paths = sorted(required + optional)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Workers must be 1 through 8')
    found = sorted(p.relative_to(data_root).as_posix() for p in data_root.rglob('*') if p.is_file())
    unexpected = sorted(set(found) - set(required) - set(OPTIONAL_FILES))
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_file, data_root, p): p for p in paths}
        for future in as_completed(futures):
            entry = future.result(); entries.append(entry)
            if progress is not None:
                progress(entry)
    entries.sort(key=lambda item: item['path'])
    good = [item for item in entries if item['status'] == 'ok']
    valid_tracking = [item for item in good if item['path'].endswith('.jsonl.bz2')]
    missing = [item['path'] for item in entries if item['status'] == 'missing']
    invalid = [item['path'] for item in entries if item['status'] == 'invalid']
    body = {'schema_version': 1, 'dataset': 'PFF FC World Cup 2022 current release',
            'inventory_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_file_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
            'protocol_sha256': protocol.get('protocol_sha256'),
            'expected_fixtures': len(MATCH_IDS), 'required_files': len(required),
            'optional_files_present': optional, 'inventoried_files': len(entries),
            'valid_files': len(good), 'compressed_or_regular_bytes': sum(item['bytes'] for item in good),
            'tracking_archives_verified': len(valid_tracking),
            'tracking_decompressed_bytes': sum(item['decompressed_bytes'] for item in valid_tracking),
            'missing_files': missing, 'invalid_files': invalid, 'unexpected_files': unexpected,
            'complete': not missing and not invalid and not unexpected,
            'raw_match_semantics_read': False, 'heldout_labels_inspected': False,
            'json_validation': 'signature only; full syntax and semantic checks deferred to authorized split loading',
            'hash_scope': 'locally computed identity; not a provider-signed checksum or authenticity proof',
            'files': entries}
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/pff')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'models/pff-protocol.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/pff-access/inventory.json')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    def progress(entry):
        print(json.dumps({'path': entry['path'], 'status': entry['status']}), flush=True)
    result = audit(args.data, args.protocol, workers=args.workers, progress=progress)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('complete', 'valid_files', 'tracking_archives_verified',
                                           'compressed_or_regular_bytes', 'tracking_decompressed_bytes')}))
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
