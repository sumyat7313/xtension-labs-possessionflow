"""Package v6 research evidence alongside the frozen inference bundle.

The inference package's original source allowlist remains frozen by the campaign.
This release wrapper explicitly adds new research/provenance files; no recursive
raw-data collection or local-run inclusion is allowed.
"""
from pathlib import Path
import argparse,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from possession.bundle import make_bundle
from possession.io import sha256,write_json

EXTRAS=('tools/package_temporal_v6.py','tools/train_temporal_v6.py','tools/train_stoppage_v6.py',
        'tools/train_transition_v6.py','tools/temporal_diagnostics_v6.py','tools/fresh_benchmark_v6.py',
        'models/baselines/v1.5.json','models/baselines/v1.5-source.zip','configs/baselines/v1.5.json',
        'models/temporal-selection.json','models/fresh-audit-v6.json','models/experiments/v6-candidate.json','docs/TEMPORAL_CAMPAIGN_V6.md',
        'docs/FRESH_DATA_V6.md','docs/STOPPAGE_RESEARCH_v6.md','docs/SHORT_SPELL_RESEARCH_v6.md','docs/NEXT_EXPERIMENTS.md',
        'submission/Xtension Labs.csv','submission/Xtension Labs.md','submission/manifest.json',
        'submission/config.json','submission/verification.json')


def package(destination,run,team='Xtension Labs'):
    missing=[name for name in EXTRAS if not (ROOT/name).is_file()]
    if missing:raise ValueError('Missing v6 research evidence: '+', '.join(missing))
    archive=make_bundle(destination,team,run)
    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        if set(z.namelist())&set(EXTRAS):raise ValueError('Duplicate v6 source entry')
        for name in sorted(EXTRAS):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,(ROOT/name).read_bytes())
    status_path=Path(destination)/'bundle-status.json'
    status=json.loads(status_path.read_text());status['source_sha256']=sha256(archive)
    status['research_evidence']='v6 fixed development campaign and one-shot fresh audit'
    write_json(status_path,status)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--team',default='Xtension Labs')
    a=p.parse_args();print(package(a.out,a.run,a.team))
