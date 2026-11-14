"""Prepare an isolated clean-Sportec fallback; never replace or submit v1.5.

Only the exact completed external rebuild is accepted. Source export happens
before inference and contains neither challenge records nor a prediction CSV.
All actions are local; this tool cannot create a repository or submit an entry.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REBUILD = 'runs/continuous-improvement/trials/external-rebuild-001/output'
CANDIDATE_SHA = '7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47'
CONFIG_SHA = '2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b'
BASE_SHA = '115c192cd4aa71be837e1c4bd2137d91add6dcea4b006a809b13232e62f43c0b'
BUNDLE_SHA = '9c4150f3ba9cdbb59df3270fa2ca4e667d6eca76898b40a0debf5be39d2867c0'
PINS = {
    'refit/candidate.json': CANDIDATE_SHA,
    'refit/config.json': CONFIG_SHA,
    'refit/lineage.json': '3d31772a8821e869b8569563ab43fcca3b7b2afe11159072ebfd76c1620f0e8c',
    'protocol.json': '0cd152ae7105f4561b5fa6f97f23fc71f1dd2afde49fe32301fb7d57dca6f31f',
    'summary.json': 'c605d16d3550b36db281c95ecee964ffe8e97538d096f9dbef6c6e4b8d5cdb34',
}
REPORT = 'docs/Xtension_Labs_Technical_Report.md'
TEAM = 'Xtension Labs'
ROWS = 199319
ROOTS = ('games.py', 'possession/learned.py', 'possession/config.py',
         'possession/io.py', 'evaluation/possession_metrics.py')
INFERENCE_FILES = frozenset(('evaluation/possession_metrics.py', 'games.py',
    'possession/__init__.py', 'possession/boundary.py', 'possession/changes.py',
    'possession/config.py', 'possession/contact_evidence.py', 'possession/control.py',
    'possession/decoder.py', 'possession/features.py', 'possession/io.py',
    'possession/joint_temporal.py', 'possession/learned.py', 'possession/missing_ball.py',
    'possession/robust_control.py', 'possession/tags.py', 'possession/temporal.py',
    'possession/transition.py', 'possession/typed_boundary.py'))
EXPORT_FILES = INFERENCE_FILES | frozenset(('run_fallback.py', 'models/fallback.json',
    'config.json', 'lineage.json', 'requirements.txt', 'LICENSE', 'Xtension Labs.md',
    'provenance.json', 'ATTRIBUTION.md', 'README.md', '.gitattributes'))
GIT_ATTRIBUTES = b'* -text\n'
GAMES_SHA = '473bf0502a28d93ff49fe5c20be4e546e30a9d8fa4112ab50b1ecdaf44233064'
REQUIREMENTS_SHA = '695cb1c7d3bbfbce4f665574f7287d2bde445511e5d096a1a79d9a42376f9d4c'


def _need(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _no_links(path):
    path = Path(os.path.abspath(path))
    _need(path.resolve() == path, 'Linked paths are forbidden')
    for part in (path, *path.parents):
        _need(not part.is_symlink() and not (hasattr(part, 'is_junction') and part.is_junction()),
              'Linked paths are forbidden')
    return path


def _relative(root, relative):
    _need(type(relative) is str and relative and '\\' not in relative and ':' not in relative
          and not PurePosixPath(relative).is_absolute() and '..' not in PurePosixPath(relative).parts,
          'Only plain relative artifact paths are permitted')
    path = _no_links(Path(root) / relative)
    _need(path.is_relative_to(_no_links(root)), 'Artifact escapes its root')
    return path


def _checked(root, relative, expected):
    raw = _relative(root, relative).read_bytes()
    _need(_sha(raw) == expected, 'Pinned artifact bytes changed: ' + Path(relative).name)
    return raw


def _exclusive(path, raw):
    path = _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _json(path, value):
    _exclusive(path, _canonical(value) + b'\n')


def destination(out):
    path = _no_links(out)
    _need(path.parent in (_no_links(ROOT / 'runs'), _no_links(ROOT / 'dist'))
          and path.name.startswith('compliant-fallback-'),
          'Use a new runs/compliant-fallback-* or dist/compliant-fallback-* directory')
    return path


def authenticate_clean():
    """Authenticate aggregates and learned bytes locally; never open match records."""
    from tools import external_rebuild as rebuild
    from tools import improvement_policy as policy
    bundle = json.loads(_checked(ROOT, 'models/external-rebuild-001-research.json', BUNDLE_SHA))
    values = {name: _checked(ROOT, f'{REBUILD}/{name}', identity) for name, identity in PINS.items()}
    protocol = json.loads(values['protocol.json'])
    lineage = json.loads(values['refit/lineage.json'])
    summary = json.loads(values['summary.json'])
    _need(protocol == bundle['protocol'] and lineage == bundle['lineage'], 'Published clean evidence differs')
    _need(protocol == rebuild.plan(protocol['recipe']) | {'inputs': protocol['inputs']},
          'Clean training recipe, parent, configuration or scientific source changed')
    _need(summary['status'] == 'complete' and summary['next_stage'] is None
          and summary['release_candidate_model_sha256'] == CANDIDATE_SHA
          and summary['external_only'] is True and summary['final_accessed'] is False,
          'Expected completed external-only rebuild')
    audit = policy.audit_lineage(lineage)
    _need(audit['candidate_sha256'] == CANDIDATE_SHA
          and audit['external_dataset_ids'] == ['sportec-idsse-28196177'],
          'Fallback must have only the reviewed external Sportec dataset ancestry')
    policy.authenticate_models(audit, summary['model_paths'], ROOT)
    # This configuration predates target-camera fitting. Its hash also appears
    # in later releases because it was retained; that coincidence is not proof
    # of its origin. Keep the explicit historical source identity and limitation.
    _checked(ROOT, 'configs/baselines/v1.2.json', CONFIG_SHA)
    _checked(ROOT, 'models/baselines/v1.2.json', BASE_SHA)
    from possession.learned import validate_model
    model = validate_model(json.loads(values['refit/candidate.json']))
    _need(model.get('candidate_id') == 'external-joint'
          and 'robust_control_model' in model and 'joint_model' in model
          and not any(k in model for k in ('control_model', 'temporal_model', 'expert_gate',
                                           'boundary_model', 'transition_model', 'typed_boundary_model')),
          'Unexpected fallback model components')
    return dict(model_bytes=values['refit/candidate.json'], config_bytes=values['refit/config.json'],
                lineage_bytes=values['refit/lineage.json'], scientific_sources=protocol['source_hashes'],
                lineage_sha256=PINS['refit/lineage.json'], source_bundle_sha256=BUNDLE_SHA,
                declared_lineage_valid=True, actual_model_ancestry_authenticated=True,
                provenance_semantically_certified=False,
                configuration_origin='Pinned pre-camera-fit v1.2 configuration; historical lineage review remains a substantive claim')


def inference_sources(root=ROOT):
    """Complete static local import closure from inference roots, including lazy imports."""
    root = _no_links(root)
    pending = [_relative(root, n) for n in ROOTS]
    seen = set()
    def add(module):
        parts = module.split('.')
        if not all(p.isidentifier() for p in parts):
            return
        for i in range(1, len(parts) + 1):
            package = root.joinpath(*parts[:i]) / '__init__.py'
            if package.is_file() and package not in seen:
                pending.append(package)
        path = root.joinpath(*parts).with_suffix('.py')
        if path.is_file() and path not in seen:
            pending.append(path)
    while pending:
        path = _no_links(pending.pop())
        if path in seen:
            continue
        relative = path.relative_to(root).as_posix()
        _need(relative == 'games.py' or relative.startswith('possession/')
              or relative in ('evaluation/__init__.py', 'evaluation/possession_metrics.py'),
              'Unexpected local dependency in public inference closure')
        seen.add(path)
        package = list(path.relative_to(root).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                if node.module:
                    base.extend(node.module.split('.'))
                if base:
                    add('.'.join(base))
                    for alias in node.names:
                        if alias.name != '*':
                            add('.'.join([*base, alias.name]))
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(seen)}


RUNNER = r'''"""Fixed clean external fallback. No training, scoring or publication command."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
MODEL_SHA = MODEL_ID_LITERAL
CONFIG_SHA = CONFIG_ID_LITERAL
EXPECTED_FILES = EXPECTED_FILES_LITERAL
SCIENTIFIC_SHA = SCIENTIFIC_SHA_LITERAL
N = 199319

def need(value, message):
    if not value: raise ValueError(message)

def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

def no_links(path):
    path = Path(os.path.abspath(path))
    need(path.resolve() == path, 'Linked paths are forbidden')
    for part in (path, *path.parents):
        need(not part.is_symlink() and not (hasattr(part,'is_junction') and part.is_junction()), 'Linked paths are forbidden')
    return path

def verify_source():
    value = json.loads(no_links(ROOT/'source-manifest.json').read_bytes())
    need(value['model_sha256'] == MODEL_SHA and value['config_sha256'] == CONFIG_SHA, 'Wrong fixed fallback identity')
    need(hash_file(no_links(ROOT/'models/fallback.json')) == MODEL_SHA and hash_file(no_links(ROOT/'config.json')) == CONFIG_SHA, 'Fallback model/config bytes changed')
    need(set(value['files']) == EXPECTED_FILES, 'Unreviewed source file inventory')
    need(all(value['files'].get(name) == identity for name,identity in SCIENTIFIC_SHA.items()), 'Unreviewed inference source bytes')
    # Git metadata is permitted only at runtime, never in the source archive.
    if (ROOT/'.git').exists(): need(no_links(ROOT/'.git').is_dir(), 'Unsafe Git metadata')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and p.relative_to(ROOT).parts[0] != '.git'}
    need(actual == set(value['files']) | {'source-manifest.json'}, 'Public source file inventory differs')
    for relative, expected in value['files'].items():
        need(type(relative) is str and '\\' not in relative and ':' not in relative and '..' not in Path(relative).parts, 'Invalid source path')
        path = no_links(ROOT/relative)
        need(path.is_relative_to(ROOT) and hash_file(path) == expected, 'Source bytes changed')
    return value

def validate(path):
    with Path(path).open('r', encoding='ascii', newline='') as stream:
        rows = csv.reader(stream)
        need(next(rows, None) == ['frame','label'], 'CSV header must be exactly frame,label')
        count = 0
        for i, row in enumerate(rows):
            need(len(row) == 2 and row[0] == str(i) and row[1] in ('H','A','D','X'), 'Invalid exact CSV contract')
            count += 1
        need(count == N, 'Wrong CSV row count')
    from evaluation.possession_metrics import read_label_csv, validate_submission
    table = read_label_csv(path)
    need(validate_submission(table, N) is None, 'CSV violates the official frame/label contract')
    return N

def predict(data_dir, output):
    manifest = verify_source()
    import numpy as np
    need(sys.version_info[:2] == (3,12) and np.__version__ == '2.5.4', 'Use pinned CPython 3.12 / NumPy 2.5.4 environment')
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from possession.io import read_tracking, read_events, write_submission
    from possession.config import load_config
    from possession.learned import load_model, infer_learned
    out = no_links(output)
    need(not out.is_relative_to(ROOT) and not ROOT.is_relative_to(out), 'Prediction outputs must stay outside the public source directory')
    need(not out.exists(), 'A prediction output directory cannot be overwritten')
    data = no_links(data_dir)
    need(not (data/'.synthetic-fixture.json').exists(), 'Synthetic data cannot produce a competition fallback')
    need(not out.is_relative_to(data) and not data.is_relative_to(out), 'Output and input directories must be separate')
    paths = asset_paths(data_dir=str(data))
    keys = ('atd_meta','atd_raw','smart_tagging')
    inputs = {key:no_links(paths[key]) for key in keys}
    need(all(path.is_relative_to(data) and path.is_file() for path in inputs.values()), 'Missing fixed tactical assets')
    # Consume this output before reading records. Failure never silently retries.
    out.mkdir(parents=True)
    (out/'started.json').write_text(json.dumps({'mode':'frozen-fallback-inference','references_accessed':False})+'\n')
    identities = {key:hash_file(path) for key,path in inputs.items()}
    model = load_model(ROOT/'models/fallback.json')
    config = load_config(ROOT/'config.json')
    spec = GAMES[GAME]
    need(spec['n_video_frames'] == N and spec['fps'] == 25, 'Published challenge timeline changed')
    tracking = read_tracking(inputs['atd_meta'],inputs['atd_raw'],N)
    events = read_events(inputs['smart_tagging'],N,25)
    labels, probabilities, features, tags, warnings = infer_learned(tracking,events,half_video_blocks(GAME),spec['pitch'],25,config,model)
    need(labels.shape == (N,) and np.isin(labels,list('HADX')).all(), 'Invalid inference result')
    csv_path = out/'Xtension Labs.csv'
    write_submission(csv_path,labels)
    validate(csv_path)
    need(identities == {key:hash_file(path) for key,path in inputs.items()}, 'Input bytes changed during inference')
    need(verify_source() == manifest, 'Source changed during inference')
    # Only this local receipt contains challenge input hashes; never add it to
    # the public source export. No frame records/probabilities are written.
    receipt = dict(mode='clean-sportec-fallback',rows=N,csv_sha256=hash_file(csv_path),
        model_sha256=MODEL_SHA,config_sha256=CONFIG_SHA,input_sha256=identities,
        source_manifest_sha256=hash_file(ROOT/'source-manifest.json'),
        label_counts={label:int(np.sum(labels==label)) for label in 'HADX'},
        references_accessed=False,video_accessed=False,trained_or_tuned_on_challenge=False,
        stronger_than_v15_established=False,final_pff_accessed=False,
        submission_ready=False,publication_or_submission_performed=False,
        runtime={'python':sys.version.split()[0],'numpy':np.__version__})
    with (out/'receipt.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    return {key:receipt[key] for key in ('mode','rows','csv_sha256','model_sha256','references_accessed','final_pff_accessed','submission_ready')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--out')
    parser.add_argument('--validate')
    args = parser.parse_args()
    if args.validate:
        need(args.data_dir is None and args.out is None, 'Validation and inference options cannot be mixed')
        verify_source()
        result = {'rows':validate(args.validate),'format_valid':True,'accuracy_measured':False}
    else:
        need(args.data_dir and args.out, 'Provide --data-dir and --out')
        result = predict(args.data_dir,args.out)
    print(json.dumps(result,sort_keys=True))

if __name__ == '__main__':
    try: main()
    except Exception as error:
        # Never print provider records, arrays, weights or parser content.
        print(json.dumps({'status':'failed','exception_type':type(error).__name__}),file=sys.stderr)
        raise SystemExit(2)
'''


def runner_bytes():
    scientific = {name:_sha(raw) for name,raw in inference_sources(ROOT).items()}
    scientific['.gitattributes'] = _sha(GIT_ATTRIBUTES)
    return (RUNNER.replace('MODEL_ID_LITERAL', repr(CANDIDATE_SHA))
            .replace('CONFIG_ID_LITERAL', repr(CONFIG_SHA))
            .replace('EXPECTED_FILES_LITERAL', 'set(' + repr(tuple(sorted(EXPORT_FILES))) + ')')
            .replace('SCIENTIFIC_SHA_LITERAL', repr(dict(sorted(scientific.items()))))).encode()


def _archive(path, files):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, raw)


def prepare(out):
    dest = destination(out)
    _need(not dest.exists(), 'Fallback preparation output is already consumed')
    evidence = authenticate_clean()
    files = inference_sources(ROOT)
    _need(set(files) == INFERENCE_FILES, 'Inference closure differs from reviewed inventory')
    _checked(ROOT, 'games.py', GAMES_SHA)
    _checked(ROOT, 'requirements-solution.txt', REQUIREMENTS_SHA)
    for name, raw in files.items():
        if name in evidence['scientific_sources']:
            _need(_sha(raw) == evidence['scientific_sources'][name], 'Frozen inference source changed')
    report = _relative(ROOT, REPORT).read_bytes()
    _need(report.strip(), 'Technical report is required before packaging')
    files.update({
        '.gitattributes': GIT_ATTRIBUTES,
        'run_fallback.py': runner_bytes(),
        'models/fallback.json': evidence['model_bytes'],
        'config.json': evidence['config_bytes'],
        'lineage.json': evidence['lineage_bytes'],
        'requirements.txt': _relative(ROOT, 'requirements-solution.txt').read_bytes(),
        'LICENSE': _relative(ROOT, 'LICENSE').read_bytes(),
        f'{TEAM}.md': report,
    })
    provenance = {key:evidence[key] for key in ('lineage_sha256','source_bundle_sha256',
        'declared_lineage_valid','actual_model_ancestry_authenticated','provenance_semantically_certified','configuration_origin')}
    provenance.update(model_sha256=CANDIDATE_SHA,config_sha256=CONFIG_SHA,
        training_dataset='Sportec/IDSSE 28196177',training_fixture_count=7,
        training_data_license='CC BY 4.0',pff_fitted_components=False,
        target_camera_profile_used=False,improvement_gates_passed=False,
        eligibility_review_required=True,organiser_eligibility_certified=False,
        promoted=False,submission_ready=False)
    files['provenance.json'] = _canonical(provenance)+b'\n'
    files['ATTRIBUTION.md'] = (
        '# Attribution and data rights\n\n'
        'Sportec/IDSSE data: Bassek, Rein, Weber and Memmert (2025), '
        'https://doi.org/10.6084/m9.figshare.28196177, licensed CC BY 4.0 '
        '(https://creativecommons.org/licenses/by/4.0/). The included model is '
        'a learned transformation trained on seven external fixtures with synthetic '
        'observation corruption. It includes an earlier live/dead classifier trained '
        'on two of those fixtures. Provider references are not independently '
        'adjudicated possession truth. No provider records are included.\n\n'
        'Original challenge starter/evaluation code: Metrica Sports, under the '
        'included MIT license. Xtension Labs supplies the inference implementation '
        'and this packaging wrapper. Match assets remain separately licensed, '
        'local and non-redistributable. No PFF-fitted weights are included.\n').encode()
    files['README.md'] = (
        '# Xtension Labs — clean external fallback\n\n'
        'This is a fallback candidate, not a claimed improvement over v1.5. '
        'See `Xtension Labs.md` and `provenance.json`. No private match data or '
        'possession output is included in this source export.\n\n'
        'Use CPython 3.12 and install `python -m pip install -r requirements.txt`.\n\n'
        'Run `python run_fallback.py --data-dir PATH_TO_AUTHORIZED_DATA --out ../reproduced`. '
        'The data directory must contain the supplied tactical folder. '
        'Outputs must be outside this source directory. The output is '
        '`../reproduced/Xtension Labs.csv` plus a local aggregate receipt.\n\n'
        'Validate with `python run_fallback.py --validate "../reproduced/Xtension Labs.csv"`. '
        'Validation checks format, not accuracy. Use a new output directory for '
        'each independent reproduction; compare CSV SHA-256 values. Do not '
        'commit prediction outputs, receipts, data, video or caches into a public repository.\n\n'
        'No training occurs during inference. The model, configuration and '
        'inference sources are pinned by SHA-256. No public repository or '
        'competition submission is created by this package.\n').encode()
    source_manifest = dict(schema_version=1,model_sha256=CANDIDATE_SHA,config_sha256=CONFIG_SHA,
                           files={name:_sha(raw) for name,raw in sorted(files.items())})
    files['source-manifest.json'] = _canonical(source_manifest)+b'\n'
    dest.mkdir(parents=True)
    _json(dest/'preparation-started.json', dict(mode='clean-fallback-source-export',dataset_records_read=False))
    for name, raw in files.items():
        _exclusive(_relative(dest/'source',name),raw)
    archive = dest/f'{TEAM}_source.zip'
    _archive(archive,files)
    _exclusive(dest/f'{TEAM}.md',report)
    receipt = dict(schema_version=1,mode='clean-sportec-fallback',model_sha256=CANDIDATE_SHA,
        config_sha256=CONFIG_SHA,source_archive_sha256=_sha(archive.read_bytes()),
        source_manifest_sha256=_sha(files['source-manifest.json']),source_files=len(files),
        technical_document_sha256=_sha(report),prediction_generated=False,
        contains_challenge_records=False,contains_pff_weights=False,
        private_repository_unchanged=True,public_repository_created=False,
        stronger_than_v15_established=False,final_pff_accessed=False,submission_ready=False)
    _json(dest/'package.json',receipt)
    return receipt


def verify_package(out):
    dest = destination(out)
    evidence = authenticate_clean()
    receipt = json.loads(_no_links(dest/'package.json').read_bytes())
    _need(receipt['model_sha256']==CANDIDATE_SHA and receipt['config_sha256']==CONFIG_SHA,
          'Wrong fallback package identity')
    manifest_raw=_checked(dest,'source/source-manifest.json',receipt['source_manifest_sha256'])
    manifest=json.loads(manifest_raw)
    expected=manifest['files']
    _need(set(expected)==EXPORT_FILES and manifest['model_sha256']==CANDIDATE_SHA
          and manifest['config_sha256']==CONFIG_SHA, 'Unreviewed manifest inventory or identity')
    _checked(dest/'source','models/fallback.json',CANDIDATE_SHA)
    _checked(dest/'source','config.json',CONFIG_SHA)
    _need(_relative(dest/'source','lineage.json').read_bytes()==evidence['lineage_bytes'], 'Unreviewed model lineage')
    for name in INFERENCE_FILES:
        identity=GAMES_SHA if name=='games.py' else evidence['scientific_sources'][name]
        _checked(dest/'source',name,identity)
    _checked(dest/'source','requirements.txt',REQUIREMENTS_SHA)
    _checked(dest/'source','.gitattributes',_sha(GIT_ATTRIBUTES))
    _need(_relative(dest/'source','run_fallback.py').read_bytes()==runner_bytes(), 'Unreviewed inference runner')
    _need({p.relative_to(dest/'source').as_posix() for p in (dest/'source').rglob('*') if p.is_file()}
          ==set(expected)|{'source-manifest.json'},'Source export contains unexpected files')
    for name,identity in expected.items():_checked(dest/'source',name,identity)
    archive_raw=_checked(dest,f'{TEAM}_source.zip',receipt['source_archive_sha256'])
    import io
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        _need(len(archive.namelist())==len(expected)+1 and set(archive.namelist())==set(expected)|{'source-manifest.json'},
              'Archive inventory differs from reviewed source')
        for name,identity in expected.items():_need(_sha(archive.read(name))==identity,'Archive bytes differ')
        _need(archive.read('source-manifest.json')==manifest_raw,'Archive manifest differs')
    _checked(dest,f'{TEAM}.md',receipt['technical_document_sha256'])
    return receipt


def predict(out,data_dir):
    dest=destination(out)
    receipt=verify_package(dest)
    _need(not (dest/'delivery').exists(),'Fallback inference attempt is already consumed')
    command=[sys.executable,'-B',str(dest/'source/run_fallback.py'),'--data-dir',str(_no_links(data_dir)),
             '--out',str(dest/'delivery')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    from tools import owned_python_launch as owned
    with (dest/'inference-console.log').open('xb') as stream:
        process=subprocess.Popen(command,cwd=dest/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,
                                 creationflags=owned.windows_creationflags())
        job=None
        try:
            job=owned.contain_and_resume(process)
            returncode=process.wait(timeout=1800)
        finally:
            try:
                if job is not None: job.close()
            finally:
                if process.poll() is None: process.kill()
                process.wait(timeout=15)
    _need(returncode==0,'Fallback inference failed; preserve its local attempt')
    _need(verify_package(dest)==receipt,'Source export changed during inference')
    generated=json.loads(_no_links(dest/'delivery/receipt.json').read_bytes())
    _need(generated['model_sha256']==CANDIDATE_SHA and generated['config_sha256']==CONFIG_SHA
          and generated['rows']==ROWS and generated['references_accessed'] is False
          and generated['final_pff_accessed'] is False,'Fallback inference receipt differs')
    _checked(dest,'delivery/Xtension Labs.csv',generated['csv_sha256'])
    return {key:generated[key] for key in ('mode','rows','csv_sha256','model_sha256','references_accessed','final_pff_accessed','submission_ready')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify','predict'))
    parser.add_argument('--out',required=True)
    parser.add_argument('--data-dir')
    args=parser.parse_args(argv)
    if args.action=='predict':
        _need(args.data_dir is not None,'Prediction requires authorized local data directory')
        result=predict(args.out,args.data_dir)
    else:
        _need(args.data_dir is None,'Data paths are not accepted for source preparation or verification')
        result={'prepare':prepare,'verify':verify_package}[args.action](args.out)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='failed',exception_type=type(error).__name__)),file=sys.stderr)
        raise SystemExit(2)
