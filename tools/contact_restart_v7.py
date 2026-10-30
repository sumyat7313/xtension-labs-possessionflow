"""Fixed development-only alignment audit for geometric contact/restart proxies.

No claim of physical contact precision: H/A/D labels do not annotate all touches.
The audit never activates a hard event rule or an independent video channel.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.contact_evidence import feature_matrix,FEATURE_NAMES,POLICY
from possession.io import sha256,write_json
from evaluation import possession_metrics as pm

MATCHES=('J03WN1','J03WMX','J03WPY','J03WQQ','J03WR9')
REGIMES=('clean','degraded','camera','calibrated')
AUDIT_POLICY={'strength_threshold':.5,'minimum_peak_separation_seconds':.5,
    'event_tolerance_seconds':1.,'restart_alignment_precision_required':.9,
    'minimum_restart_candidates_per_regime':20,
    'independent_contact_annotation_available':False,'hard_event_activation_allowed':False}


def peaks(values,blocks,fps):
    chosen=[];gap=max(1,int(np.ceil(AUDIT_POLICY['minimum_peak_separation_seconds']*fps)))
    values=np.asarray(values)
    for _,a,b in blocks:
        candidates=np.flatnonzero(values[a:b+1]>=AUDIT_POLICY['strength_threshold'])+a
        remaining=set(candidates.tolist())
        for i in sorted(candidates,key=lambda j:(-values[j],j)):
            if int(i) not in remaining:continue
            chosen.append(int(i));remaining.difference_update(range(int(i)-gap,int(i)+gap+1))
    return sorted(chosen)


def align(reference,predicted,blocks,fps,kind):
    gt=pm.label_codes(reference);n_gt=n_pred=hits=0;timing=[]
    for _,a,b in blocks:
        for left,right in pm.valid_runs(gt[a:b+1]):
            run=(a+left,a+right)
            events=[t for t in pm.transitions(gt,[run]) if (
                t[1]==pm.D and t[2] in (pm.H,pm.A) if kind=='restart' else t[1] in (pm.H,pm.A) and t[2] in (pm.H,pm.A))]
            estimates=[(i,0,1) for i in predicted if run[0]<i<run[1]]
            matched,_,_=pm.match_transitions(events,estimates,fps,typed='none')
            n_gt+=len(events);n_pred+=len(estimates);hits+=len(matched)
            timing.extend(abs(a[0]-b[0])/fps for a,b in matched)
    return {'n_reference_boundaries':n_gt,'n_candidates':n_pred,'n_matched':hits,
        'boundary_alignment_precision':hits/n_pred if n_pred else None,
        'boundary_recall':hits/n_gt if n_gt else None,
        'mean_matched_absolute_error_seconds':float(np.mean(timing)) if timing else None,
        'physical_contact_precision':None,'reference_is_independent_contact_truth':False}


def run(args):
    if args.out.exists():raise ValueError('Choose a new immutable audit output')
    args.out.mkdir(parents=True)
    sources={n:sha256(ROOT/n) for n in ('possession/contact_evidence.py','tools/contact_restart_v7.py','evaluation/possession_metrics.py')}
    protocol={'matches':MATCHES,'regimes':REGIMES,'feature_names':FEATURE_NAMES,'cue_policy':POLICY,
        'audit_policy':AUDIT_POLICY,'source_hashes':sources,'all_matches_previously_exposed':True,
        'reference_scope':'Explicit Sportec possession/state references annotate boundaries, not all physical contacts.',
        'order':'Save every observation-only cue candidate before loading references; no threshold tuning.'}
    write_json(args.out/'protocol.json',protocol);predictions={};blocks_by_match={}
    for match in MATCHES:
        old=args.previous/'cache'/match;blocks=json.loads((old/'metadata.json').read_text())['blocks'];blocks_by_match[match]=blocks
        for regime in REGIMES:
            x=np.load(old/f'{regime}-features.npy',mmap_mode='r',allow_pickle=False)
            z=np.load(args.cache/match/f'{regime}-dead-features.npy',mmap_mode='r',allow_pickle=False)
            cues=feature_matrix(x,z,blocks,25)
            predicted={name:peaks(cues[:,column],blocks,25) for name,column in (('contact',1),('restart',3))}
            predictions[match,regime]=predicted
            write_json(args.out/f'{match}-{regime}-candidates.json',predicted)
    rows=[]
    for match in MATCHES:
        reference=np.load(args.previous/'cache'/match/'reference.npy',allow_pickle=False)
        for regime in REGIMES:
            rows.append({'match':match,'regime':regime,**{name:align(reference,indices,blocks_by_match[match],25,name)
                for name,indices in predictions[match,regime].items()}})
    totals={}
    for regime in REGIMES:
        selected=[r['restart'] for r in rows if r['regime']==regime]
        n=sum(r['n_candidates'] for r in selected);hits=sum(r['n_matched'] for r in selected)
        totals[regime]={'n_candidates':n,'n_matched':hits,'alignment_precision':hits/n if n else None,
            'passed_alignment_requirement':n>=20 and hits/n>=.9}
    if sources!={name:sha256(ROOT/name) for name in sources}:raise ValueError('Audit source changed')
    result={'protocol':protocol,'protocol_sha256':sha256(args.out/'protocol.json'),'records':rows,
        'restart_by_regime':totals,'hard_event_activation_allowed':False,
        'independent_video_channel_enabled':False,'physical_contact_precision_established':False}
    write_json(args.out/'summary.json',result);print(json.dumps(totals,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,default=ROOT/'runs/multimodal-campaign-v4')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/joint-cache-v5')
    p.add_argument('--out',type=Path,default=ROOT/'runs/contact-v7-development-audit')
    run(p.parse_args())
