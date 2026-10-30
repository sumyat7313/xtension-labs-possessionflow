"""Explicit v7 source/evidence additions; raw data and local runs stay excluded."""
from pathlib import Path
import argparse,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.package_temporal_v6 import package as previous_package
from possession.io import sha256,write_json

EXTRAS=('tools/package_boundary_v7.py','tools/train_boundary_campaign_v7.py','tools/train_typed_boundary_v7.py',
        'tools/contact_restart_v7.py','tools/fresh_benchmark_v7.py','models/baselines/v1.5.1-source.zip',
        'models/boundary-selection-v7.json','models/fresh-audit-v7.json','models/contact-audit-v7.json',
        'models/experiments/v7-candidate.json','docs/BOUNDARY_CAMPAIGN_V7.md','docs/TYPED_BOUNDARY_V7.md',
        'docs/CONTACT_RESTART_V7.md','docs/FRESH_DATA_V7.md')


def package(destination,run,team='Xtension Labs'):
    missing=[n for n in EXTRAS if not (ROOT/n).is_file()]
    if missing:raise ValueError('Missing v7 evidence: '+', '.join(missing))
    archive=previous_package(destination,run,team)
    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        if set(z.namelist())&set(EXTRAS):raise ValueError('Duplicate v7 source entry')
        for name in sorted(EXTRAS):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,(ROOT/name).read_bytes())
    path=Path(destination)/'bundle-status.json';status=json.loads(path.read_text())
    status.update(source_sha256=sha256(archive),research_evidence='v7 typed boundaries, geometric cue reliability and limited fresh phase audit')
    status['publication']='Private repository retained under likalight at the user\'s request; organizer source-access requirements remain unresolved.'
    status['remaining']=['Confirm registration and team eligibility for Xtension Labs',
        'Resolve organizer source-access requirements while retaining repository privacy',
        'Upload the validated CSV, technical document and approved source reference to the official submission form']
    write_json(path,status);return archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--team',default='Xtension Labs')
    a=p.parse_args();print(package(a.out,a.run,a.team))
