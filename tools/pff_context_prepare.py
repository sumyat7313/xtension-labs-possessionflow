"""Authenticated training-only parent emissions for isolated D-context fitting."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from possession.learned import load_model
from tools import pff_campaign, pff_training

PARENT_SHA256 = '0ab133743c593466d8ae92ea94937e8c6bf22510f6b64053e3e69c7b467dcf3f'


def validate_probabilities(value, rows):
    if (value.shape != (rows, 3) or value.dtype != np.float64 or not np.isfinite(value).all()
        or np.any((value < 0) | (value > 1)) or not np.allclose(value.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Invalid full-rate parent H/A/D probabilities')


def prepare_match(source, output, mid, protocol, model):
    if (type(mid) is not int or mid not in protocol['split']['train']
        or mid in protocol['split']['development'] or mid in protocol['split']['final']):
        raise ValueError('Context preparation permits only frozen training fixtures')
    source, output = Path(source), Path(output)
    if sha256(source / 'candidate.json') != PARENT_SHA256:
        raise ValueError('Unexpected parent candidate')
    original = source / 'cache' / str(mid) / 'complete.json'
    identity = {'parent_candidate_sha256': PARENT_SHA256,
                'source_cache_manifest_sha256': sha256(original),
                'source_cache_identity': json.loads(original.read_text())['identity'],
                'protocol_sha256': protocol['protocol_sha256'],
                'preparation_source_sha256': sha256(Path(__file__))}
    # Authenticates the complete source cache even when reusing parent emissions.
    item = pff_campaign.item_at(source, mid)
    folder = output / str(mid)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest['schema_version'] != 1 or manifest['identity'] != identity or manifest['match_id'] != mid
            or manifest['rows'] != len(item['z']) or manifest['fps'] != item['fps'] or manifest['columns'] != ['H', 'A', 'D']
            or manifest['references_read'] is not False
            or manifest['reference_fields_used_for_inference'] is not False
            or manifest['files'] != {'parent_probabilities.npy': sha256(folder / 'parent_probabilities.npy')}):
            raise ValueError('Parent probability cache identity changed')
        probabilities = np.load(folder / 'parent_probabilities.npy', mmap_mode='r', allow_pickle=False)
        validate_probabilities(probabilities, len(item['z']))
        return manifest
    _, probabilities = pff_training.predict_cache(item, model)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    validate_probabilities(probabilities, len(item['z']))
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / 'parent_probabilities.pending.npy'
    np.save(pending, probabilities, allow_pickle=False)
    os.replace(pending, folder / 'parent_probabilities.npy')
    manifest = {'schema_version': 1, 'match_id': mid, 'rows': len(probabilities), 'fps': item['fps'],
                'columns': ['H', 'A', 'D'], 'identity': identity,
                'reference_fields_used_for_inference': False, 'references_read': False,
                'files': {'parent_probabilities.npy': sha256(folder / 'parent_probabilities.npy')}}
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/pff-campaign-v3')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/pff-context-v1/training-probabilities')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    protocol = pff_campaign.protocol()
    if sha256(source / 'candidate.json') != PARENT_SHA256:
        raise ValueError('Parent candidate changed')
    model = load_model(source / 'candidate.json')
    records = []
    for mid in protocol['split']['train']:
        manifest = prepare_match(source, output, mid, protocol, model)
        records.append({'match_id': mid, 'manifest_sha256': sha256(output / str(mid) / 'manifest.json'),
                        'probability_sha256': manifest['files']['parent_probabilities.npy'], 'rows': manifest['rows']})
        print(json.dumps({'prepared': mid, 'count': len(records)}), flush=True)
    write_json(output / 'complete.json', {'schema_version': 1, 'parent_candidate_sha256': PARENT_SHA256,
               'protocol_sha256': protocol['protocol_sha256'], 'references_read': False,
               'preparation_source_sha256': sha256(Path(__file__)), 'records': records})


if __name__ == '__main__':
    main()
