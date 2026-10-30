"""Reconstruct input-only dead features and authenticate the exposed v4 cache."""
from pathlib import Path
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from possession.features import extract
from possession import robust_control
from possession.learned import feature_matrix,predict_dead,load_model
from possession.config import load_config
from possession.io import sha256,write_json
from tools.train_control import read_external,CATALOG,SEEDS
from tools.train_stoppages import corrupt_tracking
from tools.camera_augmentation import corrupt_camera_tracking
from tools.calibrated_camera import corrupt_calibrated_tracking,validate_profile
REGIMES=('clean','degraded','camera','calibrated')

def prepare_match(match,data_dir,previous,out,profile_path):
    tracking,y,blocks,pitch,alignment,inputs=read_external(data_dir,match)
    old=previous/'cache'/match; dest=out/match;dest.mkdir(parents=True)
    meta=json.loads((old/'metadata.json').read_text(encoding='utf-8'))
    if not np.array_equal(y,np.load(old/'reference.npy',allow_pickle=False)) or meta['inputs']!=inputs or meta['blocks']!=[list(b) for b in blocks]:
        raise ValueError('Raw references or input provenance differ from v4')
    profile=validate_profile(json.loads(profile_path.read_text(encoding='utf-8')))
    config=load_config(ROOT/'configs/baselines/v1.4.json')
    baseline=load_model(ROOT/'models/baselines/v1.4.json')
    proof={'inputs':inputs,'profile_sha256':sha256(profile_path),'files':{}}
    for regime in REGIMES:
        observed=(tracking if regime=='clean' else corrupt_tracking(tracking,blocks,pitch,25,SEEDS[match]) if regime=='degraded'
                  else corrupt_camera_tracking(tracking,blocks,pitch,25,SEEDS[match]+91001) if regime=='camera'
                  else corrupt_calibrated_tracking(tracking,blocks,pitch,25,SEEDS[match]+192071,profile))
        f=extract(observed,pitch,blocks,25,config)
        x=robust_control.feature_matrix(observed,f,blocks,pitch,25)
        cached=np.load(old/f'{regime}-features.npy',mmap_mode='r',allow_pickle=False)
        if not np.array_equal(x,cached):raise ValueError(f'{match}/{regime} observation reconstruction differs')
        z=feature_matrix(observed,f,blocks,pitch,25)
        dead=predict_dead(z,baseline)
        expected=(np.load(old/'calibrated-dead.npy',allow_pickle=False) if regime=='calibrated' else np.load(old/f'{regime}-baseline.npy',allow_pickle=False)[:,2])
        if not np.array_equal(dead,expected):raise ValueError('Frozen dead predictions differ')
        path=dest/f'{regime}-dead-features.npy';np.save(path,z)
        proof['files'][regime]={'control_sha256':sha256(old/f'{regime}-features.npy'),'dead_features_sha256':sha256(path),'rows':len(y)}
        print(f'{match}/{regime}: exact observations verified, {len(y):,} frames',flush=True)
        del x,cached,z,dead,observed,f
    proof['reference_sha256']=sha256(old/'reference.npy');proof['metadata_sha256']=sha256(old/'metadata.json')
    write_json(dest/'provenance.json',proof)
    return match,proof

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--previous',type=Path,default=ROOT/'runs/multimodal-campaign-v4')
    p.add_argument('--out',type=Path,default=ROOT/'runs/joint-cache-v5')
    p.add_argument('--profile',type=Path,default=ROOT/'runs/calibrated-camera/profile.json')
    p.add_argument('--workers',type=int,default=2,choices=(1,2))
    args=p.parse_args()
    if args.out.exists():raise ValueError('Use a new cache directory')
    args.out.mkdir(parents=True)
    proof={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(prepare_match,m,args.data_dir,args.previous,args.out,args.profile) for m in sorted(CATALOG)]
        for job in as_completed(jobs):
            match,value=job.result();proof[match]=value;write_json(args.out/'provenance.json',proof)
    print('PASS: all seven exposed-match observation caches verified',flush=True)
