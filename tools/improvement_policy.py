"""Aggregate-only screening for the continuing external-data research campaign.

This module never opens match records, authorizes final access, or promotes a
model. Hashes establish artifact identity, not the truth of a licence or lineage
declaration. Semantic review remains separate and explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pff_precision_validation as precision
from tools import pff_validation as original

APPROVED_ROOT_MODEL = '115c192cd4aa71be837e1c4bd2137d91add6dcea4b006a809b13232e62f43c0b'
RESERVATION_SHA256 = '4475cf6cf95de433a49a1c03ab93838be96f05059eee15a09a73e1673557e980'
ORIGINAL_PARENT_RECORDS_SHA256 = '502bfef18b1d693392d9b22d5e22bb0d560eebab178586be6886a78be0d7a39a'
EVALUATION_DATASET = 'pff-world-cup-2022'
PROHIBITED_FIT_MODELS = frozenset({
    '7f72b108801a9c6029a2543ceaccd2ef9f27947c256ab59265df47c316688653',
    original.BASELINE_SHA256, precision.PARENT_SHA256,
    '41e4f239be88fa69b9ed4dac899a55e4dfaabc74858b932df407cff1847279de',
})
RECEIPT_PURPOSES = ('local_research', 'competition_submission', 'dense_reference_validation')
POLICY = {
    'schema_version': 1,
    'minimum_dead_precision_gain': .002,
    'minimum_retained_positive_gain_fraction': .9,
    'minimum_development_matches': 12,
    'original_gates_required': True,
    'automatic_promotion': False,
    'automatic_final_access': False,
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _inside(root, relative):
    _need(isinstance(relative, str) and relative, 'Artifact path is required')
    root = Path(root).resolve()
    target = (root / relative).resolve()
    _need(target.is_relative_to(root) and target.is_file(), 'Artifact must be an existing file inside its evidence root')
    return target


def audit_lineage(manifest):
    """Validate a complete declared fit DAG; do not certify its historical truth.

    The v1.2 dead head is the only permitted parentless learned artifact. Its
    exact identity is tied to the existing external WOH/WOY training audit.
    Comparator v1.5 must stay outside the fitted ancestry.
    """
    fields = {'schema_version', 'candidate_sha256', 'comparator_sha256', 'fit_inputs', 'artifacts'}
    _need(isinstance(manifest, dict) and set(manifest) == fields, 'Lineage schema differs')
    _need(type(manifest['schema_version']) is int and manifest['schema_version'] == 1, 'Unknown lineage version')
    candidate = manifest['candidate_sha256']
    _need(original._sha(candidate), 'Candidate SHA256 is required')
    _need(manifest['comparator_sha256'] == original.BASELINE_SHA256, 'Historical comparator must stay v1.5')
    artifacts = manifest['artifacts']
    _need(isinstance(artifacts, list) and artifacts, 'Complete fit ancestry is required')
    nodes = {}
    for node in artifacts:
        _need(isinstance(node, dict) and set(node) == {'sha256', 'kind', 'scope', 'identifier', 'parents'}, 'Artifact schema differs')
        sha = node['sha256']
        _need(original._sha(sha) and sha not in nodes, 'Duplicate or invalid artifact hash')
        _need(node['kind'] in ('dataset', 'model', 'configuration', 'code'), 'Unknown artifact kind')
        _need(node['scope'] in ('external', 'synthetic'), 'Challenge or unknown provenance cannot enter fitting')
        _need(isinstance(node['identifier'], str) and node['identifier'].strip(), 'Artifact identifier required')
        identifier = node['identifier'].casefold()
        _need(not any(token in identifier for token in ('bar-ath', 'calibrated-camera', 'target-profile', 'challenge-derived', 'v1.4', 'v1.5')), 'Target-derived training artifact is prohibited')
        _need(sha not in PROHIBITED_FIT_MODELS, 'Known target-derived model cannot enter fitting')
        parents = node['parents']
        _need(isinstance(parents, list) and all(original._sha(p) for p in parents) and len(set(parents)) == len(parents), 'Parent hashes must be unique')
        if node['kind'] == 'model' and not parents:
            _need(sha == APPROVED_ROOT_MODEL, 'Unaudited parentless model is prohibited')
        nodes[sha] = node
    _need(candidate in nodes and nodes[candidate]['kind'] == 'model', 'Candidate model missing from ancestry')
    visited, active = set(), set()

    def visit(sha):
        _need(sha in nodes, 'Missing ancestor declaration')
        _need(sha not in active, 'Cyclic fitted ancestry')
        if sha in visited:
            return
        active.add(sha)
        for parent in nodes[sha]['parents']:
            visit(parent)
        active.remove(sha)
        visited.add(sha)

    visit(candidate)
    _need(visited == set(nodes), 'Unrelated artifacts cannot be mixed into fit ancestry')
    inputs = manifest['fit_inputs']
    _need(isinstance(inputs, list) and inputs and all(original._sha(p) for p in inputs)
          and len(set(inputs)) == len(inputs) and set(inputs) <= visited - {candidate}, 'Fit inputs must be declared candidate ancestors')
    datasets = [node for node in artifacts if node['kind'] == 'dataset']
    _need(datasets, 'External or synthetic training dataset ancestry is required')
    _need(all(node['sha256'] in inputs for node in datasets), 'Every training dataset must be explicitly listed as a fit input')
    return {
        'declared_lineage_valid': True, 'lineage_sha256': original.digest(manifest),
        'candidate_sha256': candidate, 'comparator_sha256': original.BASELINE_SHA256,
        'model_sha256s': sorted(n['sha256'] for n in artifacts if n['kind'] == 'model'),
        'external_dataset_ids': sorted({n['identifier'] for n in datasets if n['scope'] == 'external'}),
        'provenance_semantically_certified': False,
        'limitation': 'A complete declared graph is not proof of unrecorded historical data use.',
    }


def authenticate_models(lineage, model_paths, artifact_root):
    """Bind all declared learned artifacts to actual local bytes, not match data."""
    _need(isinstance(model_paths, dict) and set(model_paths) == set(lineage['model_sha256s']), 'Every learned ancestor requires a local model binding')
    for sha, relative in model_paths.items():
        _need(_file_hash(_inside(artifact_root, relative)) == sha, 'Learned artifact bytes differ from lineage')
    return {'model_bytes_authenticated': True, 'n_models': len(model_paths)}


def inspect_receipts(receipts, dataset_ids, evidence_root):
    """Check review-document identity and coverage, never infer rights from a flag.

    A caller's finding of 'supported' is retained as a claim. It is not upgraded
    to a legal or reference-quality certification by this function.
    """
    _need(isinstance(receipts, list), 'Evidence receipts must be a list')
    found = {}
    for receipt in receipts:
        expected = {'dataset_id', 'purpose', 'finding', 'reviewer', 'source_url', 'path', 'sha256'}
        _need(isinstance(receipt, dict) and set(receipt) == expected, 'Receipt schema differs')
        key = (receipt['dataset_id'], receipt['purpose'])
        _need(receipt['dataset_id'] in dataset_ids and receipt['purpose'] in RECEIPT_PURPOSES and key not in found, 'Unknown or duplicate evidence receipt')
        _need(receipt['finding'] in ('supported', 'unresolved', 'prohibited'), 'Evidence finding must be explicit')
        _need(isinstance(receipt['reviewer'], str) and receipt['reviewer'].strip(), 'Named evidence reviewer required')
        _need(isinstance(receipt['source_url'], str) and receipt['source_url'].startswith('https://'), 'Primary source URL required')
        _need(original._sha(receipt['sha256']), 'Evidence-document hash required')
        path = _inside(evidence_root, receipt['path'])
        _need(path.suffix.casefold() in ('.md', '.txt', '.json'), 'Use a review document, never match records or video')
        _need(_file_hash(path) == receipt['sha256'], 'Evidence document has changed')
        found[key] = receipt['finding']
    missing = [{'dataset_id': dataset, 'purpose': purpose}
               for dataset in sorted(dataset_ids) for purpose in RECEIPT_PURPOSES
               if found.get((dataset, purpose)) != 'supported']
    return {'receipt_bytes_authenticated': True, 'missing_or_unresolved': missing,
            'complete_support_claims': not missing, 'semantic_certification': False,
            'limitation': 'Hashes authenticate review documents, not their correctness or applicable rights.'}


def screen_development(records, original_parent_records, reservation, lineage,
                       receipts, evidence_root, *, model_paths, artifact_root):
    """Recompute unchanged numerical gates and screen evidence for further review.

    This is deliberately not an authorization token. The separate run verifier
    must also authenticate prediction freezes, reference-access records, the
    full bounded candidate grid and one selected artifact before any final run.
    """
    ancestry = audit_lineage(lineage)
    bindings = authenticate_models(ancestry, model_paths, artifact_root)
    _need(reservation.get('protocol_sha256') == RESERVATION_SHA256, 'Original PFF fixture reservation must not change')
    _need(original.digest(original_parent_records) == ORIGINAL_PARENT_RECORDS_SHA256, 'Retention must use the authenticated original PFF development results')
    gate = precision.development_gate(records, original_parent_records, reservation)
    _need(gate['candidate_sha256'] == ancestry['candidate_sha256'], 'Scores and lineage identify different candidates')
    required_datasets = sorted(set(ancestry['external_dataset_ids']) | {EVALUATION_DATASET})
    evidence = inspect_receipts(receipts, required_datasets, evidence_root)
    return {
        'policy': dict(POLICY), 'policy_sha256': original.digest(POLICY),
        'numeric_gate': gate, 'lineage': ancestry, 'artifact_bindings': bindings,
        'evidence_receipts': evidence,
        'ready_for_independent_final_review': gate['numeric_gate_passed'] and evidence['complete_support_claims'],
        'final_access_authorized': False, 'promoted': False,
        'required_before_final_access': [
            'Independent substantive review of dataset rights, reference quality and full lineage.',
            'Run-level verification of frozen protocol, complete bounded selection and all predictions before references.',
            'Exactly one eligible candidate; final cohort remains one-shot and cannot become development data.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True, help='Aggregate-only JSON with records, parent_records, reservation, lineage, receipts, model_paths')
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    _need(set(request) == {'records', 'parent_records', 'reservation', 'lineage', 'receipts', 'model_paths'}, 'Screening request fields differ')
    result = screen_development(request['records'], request['parent_records'], request['reservation'],
                                request['lineage'], request['receipts'], args.evidence_root,
                                model_paths=request['model_paths'], artifact_root=args.artifact_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'numeric_gate_passed': result['numeric_gate']['numeric_gate_passed'],
                      'ready_for_independent_final_review': result['ready_for_independent_final_review'],
                      'final_access_authorized': False, 'promoted': False}))


if __name__ == '__main__':
    main()
