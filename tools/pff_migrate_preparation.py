"""Reuse authenticated observations after a training-only reference parser fix.

This is not part of the frozen scientific pipeline. No labels are opened, no
predictions are recomputed, and the original run is never modified.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.io import sha256, write_json
from tools import pff_campaign as campaign
from tools.train_temporal_v6 import CONFIG_SHA256

ADAPTER = 'tools/pff_adapter.py'
CACHE_FILES = {'x.npy', 'z.npy', 'base_home.npy', 'base_dead.npy', 'baseline.npy', 'item.json', 'alignment.json'}


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def observation_ast_digest(source):
    """Hash every executable AST node except the one top-level references()."""
    tree = ast.parse(source)
    refs = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'references']
    if len(refs) != 1 or isinstance(refs[0], ast.AsyncFunctionDef):
        raise ValueError('Expected exactly one synchronous references function')
    # Function decorators/defaults can execute during import, so they must not
    # change even when the reference function body changes.
    reference = refs[0]
    reference.body = [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode('utf-8')).hexdigest()


def serialized_hashes(value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    return {hashlib.sha256(text.encode()).hexdigest(), hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest()}


def verify_proof(proof, current_sources, protocol, current_adapter_source):
    """Portable check of the code-only proof; does not inspect provider data."""
    if (proof['schema_version'] != 1 or proof['method'] != 'copy authenticated observation artifacts'
        or proof['candidate_or_fit_existed'] is not False or proof['held_out_access_existed'] is not False):
        raise ValueError('Invalid preparation migration scope')
    old = proof['original_run_protocol']
    if proof['original_run_protocol_sha256'] not in serialized_hashes(old):
        raise ValueError('Original frozen protocol hash differs')
    if old['split_protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('Preparation reservation changed')
    prior = old['source_hashes']
    if set(prior) != set(current_sources) or {key for key in prior if prior[key] != current_sources[key]} != {ADAPTER}:
        raise ValueError('Only the reference adapter may change during preparation reuse')
    if proof['new_source_hashes'] != current_sources:
        raise ValueError('Migration target sources differ')
    old_source = proof['original_adapter_source']
    if hashlib.sha256(old_source.encode('utf-8')).hexdigest() != prior[ADAPTER]:
        raise ValueError('Original adapter source hash differs')
    if hashlib.sha256(current_adapter_source.encode('utf-8')).hexdigest() != current_sources[ADAPTER]:
        raise ValueError('Current adapter source hash differs')
    digest = observation_ast_digest(old_source)
    if observation_ast_digest(current_adapter_source) != digest or proof['observation_ast_sha256'] != digest:
        raise ValueError('Executable code outside reference parsing changed')
    expected = sorted(protocol['split']['train'] + protocol['split']['development'])
    if proof['match_ids'] != expected or proof['decoder_config_sha256'] != CONFIG_SHA256:
        raise ValueError('Migration cohort or decoder config differs')
    return prior


def verify_inheritance(manifest, proof, current_sources, protocol):
    inherited = manifest['inherited_preparation']
    original = inherited['original_manifest']
    if inherited['original_manifest_sha256'] not in serialized_hashes(original):
        raise ValueError('Inherited original manifest hash differs')
    if (inherited['original_run_protocol_sha256'] != proof['original_run_protocol_sha256']
        or inherited['observation_ast_sha256'] != proof['observation_ast_sha256']
        or inherited['old_adapter_sha256'] != proof['original_run_protocol']['source_hashes'][ADAPTER]
        or inherited['new_adapter_sha256'] != current_sources[ADAPTER]):
        raise ValueError('Inherited preparation proof identity differs')
    if ('inherited_preparation' in original or original['identity']['sources'] != proof['original_run_protocol']['source_hashes']
        or original['match_id'] not in proof['match_ids']
        or original['identity']['metadata'] != protocol['metadata_sha256'][str(original['match_id'])]
        or set(original['files']) != CACHE_FILES):
        raise ValueError('Invalid original preparation manifest')
    expected = copy.deepcopy(original)
    expected['identity']['sources'] = current_sources
    expected['inherited_preparation'] = inherited
    if manifest != expected:
        raise ValueError('Inherited observation artifacts or metadata changed')


def _assert_unscored(old):
    forbidden = ['candidate.json', 'fit.json', 'development', 'final']
    if any((old / name).exists() for name in forbidden) or any(old.glob('*-reference-access.json')) or any(old.glob('*-summary.json')) or any(old.glob('*-predictions-before-references.json')):
        raise ValueError('Preparation reuse requires no fit, candidate, or held-out access/predictions')


def migrate(old, new):
    old, new = Path(old).resolve(), Path(new).resolve()
    if new.exists() or new.is_relative_to(old) or old.is_relative_to(new):
        raise ValueError('Choose an entirely fresh, separate output directory')
    _assert_unscored(old)
    protocol = campaign.protocol()
    current = campaign.sources()
    old_protocol = _load(old / 'run-protocol.json')
    old_source = (old / 'pff_adapter-frozen.py').read_bytes().decode('utf-8')
    new_source = (ROOT / ADAPTER).read_bytes().decode('utf-8')
    if sha256(ROOT / 'configs/baselines/v1.5.json') != CONFIG_SHA256:
        raise ValueError('Decoder config changed')
    proof = {'schema_version': 1, 'method': 'copy authenticated observation artifacts',
             'reason': 'Training-only reference parser correction before candidate fit or held-out labels',
             'original_run_protocol': old_protocol, 'original_run_protocol_sha256': sha256(old / 'run-protocol.json'),
             'original_adapter_source': old_source, 'new_source_hashes': current,
             'observation_ast_sha256': observation_ast_digest(old_source),
             'decoder_config_sha256': CONFIG_SHA256,
             'match_ids': sorted(protocol['split']['train'] + protocol['split']['development']),
             'candidate_or_fit_existed': False, 'held_out_access_existed': False}
    verify_proof(proof, current, protocol, new_source)
    if {p.name for p in (old / 'cache').iterdir()} != {str(i) for i in proof['match_ids']}:
        raise ValueError('Original cache cohort must be exactly all training and development fixtures')
    originals = {}
    # Preflight the entire cohort before creating the new run.
    for mid in proof['match_ids']:
        folder = old / 'cache' / str(mid)
        path = folder / 'complete.json'
        original = _load(path)
        if (original['match_id'] != mid or original['identity']['sources'] != old_protocol['source_hashes']
            or original['identity']['metadata'] != protocol['metadata_sha256'][str(mid)]
            or set(original['files']) != CACHE_FILES or 'inherited_preparation' in original):
            raise ValueError('Original cache identity differs')
        if any(sha256(folder / name) != value for name, value in original['files'].items()):
            raise ValueError('Original observation artifact changed')
        originals[mid] = (original, sha256(path))
    _assert_unscored(old)
    campaign.freeze(new, protocol)
    write_json(new / 'preparation-migration.json', proof)
    for mid, (original, manifest_hash) in originals.items():
        src, dest = old / 'cache' / str(mid), new / 'cache' / str(mid)
        dest.mkdir(parents=True)
        for name, expected in original['files'].items():
            shutil.copyfile(src / name, dest / name)
            if sha256(dest / name) != expected:
                raise ValueError('Copied observation artifact differs')
        manifest = copy.deepcopy(original)
        manifest['identity']['sources'] = current
        manifest['inherited_preparation'] = {
            'original_manifest': original, 'original_manifest_sha256': manifest_hash,
            'original_run_protocol_sha256': proof['original_run_protocol_sha256'],
            'observation_ast_sha256': proof['observation_ast_sha256'],
            'old_adapter_sha256': old_protocol['source_hashes'][ADAPTER], 'new_adapter_sha256': current[ADAPTER]}
        verify_inheritance(manifest, proof, current, protocol)
        write_json(dest / 'complete.json', manifest)
    _assert_unscored(old)
    campaign.freeze(new, protocol)
    write_json(new / 'preparation-migration-complete.json', {
        'proof_sha256': sha256(new / 'preparation-migration.json'), 'copied_matches': len(originals),
        'copied_artifacts': len(originals) * len(CACHE_FILES), 'original_run_unchanged': True,
        'labels_read': False, 'predictions_recomputed': False})
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-run', type=Path, required=True)
    parser.add_argument('--to-run', type=Path, required=True)
    args = parser.parse_args()
    proof = migrate(args.from_run, args.to_run)
    print(json.dumps({'copied_matches': len(proof['match_ids']), 'observation_ast_sha256': proof['observation_ast_sha256']}))


if __name__ == '__main__':
    main()
