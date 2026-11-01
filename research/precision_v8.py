"""Conservative learned removal of unsupported short H/A islands.

The score is a segment-quality estimate, not a calibrated physical-contact
probability. No reference labels enter inference and no D/X label is changed.
"""
import json
import numpy as np
from possession.boundary import _blocks

POLICY={'maximum_island_seconds':2.,'context_seconds':.4,
        'merge_score_threshold':.8,'minimum_observed_fraction':.5,
        'maximum_signed_nearest_mean':.1,'one_pass':True,
        'preserve_dead_and_unknown':True,'calibration_claimed':False}
FEATURE_NAMES=('duration_over_two','posterior_margin_mean','posterior_margin_p10',
    'posterior_margin_sd','posterior_positive_fraction','dead_probability_mean',
    'nearest_margin_mean','nearest_margin_max','near_3m_mean','contact_control_mean',
    'contact_control_max','uncontested_control_mean','observed_fraction',
    'reliability_mean','player_coverage_mean','flanking_posterior_margin_mean',
    'nearest_distance_mean','integrated_posterior_margin')


def _validate(labels,p,x,z,blocks,fps):
    labels=np.asarray(labels);p,x,z=np.asarray(p,float),np.asarray(x,float),np.asarray(z,float)
    if labels.ndim!=1 or not np.isin(labels,list('HADX')).all():raise ValueError('Invalid precision labels')
    n=len(labels)
    if p.shape!=(n,3) or x.shape!=(n,98) or z.shape!=(n,58):raise ValueError('Invalid precision feature shape')
    if not all(np.isfinite(v).all() for v in (p,x,z)) or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid precision observations/probabilities')
    if isinstance(fps,(bool,np.bool_)) or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid precision FPS')
    sections,_=_blocks(blocks,n)
    return labels,p,x,z,sections


def segments(labels,p,x,z,blocks,fps):
    """Return complete local A-H-A/H-A-H islands and team-even features.

    Window availability and observation coverage are separate: the entire .4s
    context on both sides must exist inside its same-label flanking spell; each
    of left/island/right requires at least 50% directly observed ball frames.
    """
    labels,p,x,z,sections=_validate(labels,p,x,z,blocks,fps)
    rows=[];matrix=[];radius=max(1,int(np.ceil(POLICY['context_seconds']*fps)))
    live=p[:,0]+p[:,1];margin=np.divide(p[:,0]-p[:,1],live,out=np.zeros(len(p)),where=live>1e-8)
    for start,end in sections:
        edges=np.r_[start,np.flatnonzero(labels[start+1:end+1]!=labels[start:end])+start+1,end+1]
        for j in range(1,len(edges)-2):
            a,b=int(edges[j]),int(edges[j+1]);left,right=int(edges[j-1]),int(edges[j+2])
            team=labels[a];other=labels[a-1]
            if (team not in 'HA' or other not in 'HA' or team==other or labels[b]!=other
                    or (b-a)/fps>=POLICY['maximum_island_seconds']
                    or a-left<radius or right-b<radius):continue
            lo,hi=a-radius,b+radius;sign=1 if team=='H' else -1
            center=slice(a,b);side=np.r_[np.arange(lo,a),np.arange(b,hi)]
            observed=[float(np.mean(z[u:v,0]>=1-1e-8)) for u,v in ((lo,a),(a,b),(b,hi))]
            if min(observed)<POLICY['minimum_observed_fraction']:continue
            m=margin[center]*sign;geometry=x[center,0]*sign;contact=x[center,11]*sign;duration=(b-a)/fps
            feature=[duration/2,float(m.mean()),float(np.quantile(m,.1)),float(m.std()),float((m>0).mean()),
                float(p[center,2].mean()),float(geometry.mean()),float(geometry.max()),float((x[center,3]*sign).mean()),
                float(contact.mean()),float(contact.max()),float((x[center,52]*sign).mean()),observed[1],
                float(z[center,1].mean()),float(z[center,9].mean()),float((margin[side]*sign).mean()),
                float(z[center,4].mean()),float(m.mean()*duration)]
            rows.append({'start':a,'end':b,'team':str(team),'replacement':str(other),'context_start':lo,'context_end':hi})
            matrix.append(feature)
    return rows,np.asarray(matrix,float).reshape(-1,len(FEATURE_NAMES))


def validate_model(model):
    if not isinstance(model,dict) or model.get('schema_version')!=1 or model.get('kind')!='short_island_quality_v8':
        raise ValueError('Unsupported precision model')
    if model.get('feature_names')!=list(FEATURE_NAMES) or model.get('policy')!=POLICY:raise ValueError('Precision schema/policy differs')
    for key,expected in POLICY.items():
        actual=model['policy'][key]
        if (type(expected) is bool and actual is not expected) or (type(expected) is not bool and isinstance(actual,bool)):
            raise ValueError('Precision policy types differ')
    for key in ('mean','scale','coefficients'):
        v=np.asarray(model.get(key),float)
        if v.shape!=(len(FEATURE_NAMES),) or not np.isfinite(v).all() or (key=='scale' and (v<=0).any()):
            raise ValueError('Invalid precision '+key)
    intercept=model.get('intercept')
    if isinstance(intercept,bool) or not isinstance(intercept,(int,float)) or not np.isfinite(intercept):raise ValueError('Invalid precision intercept')
    json.dumps(model,allow_nan=False);return model


def predict(model,matrix):
    validate_model(model);matrix=np.asarray(matrix,float)
    if matrix.ndim!=2 or matrix.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(matrix).all():raise ValueError('Invalid precision matrix')
    value=np.clip((matrix-np.asarray(model['mean']))/np.asarray(model['scale']),-8,8)@np.asarray(model['coefficients'])+model['intercept']
    return 1/(1+np.exp(-np.clip(value,-35,35)))


def refine(model,labels,p,x,z,blocks,fps,*,return_diagnostics=False):
    rows,matrix=segments(labels,p,x,z,blocks,fps);scores=predict(model,matrix)
    output=np.asarray(labels).copy();accepted=[]
    # One simultaneous proposal pass. Conflicting islands cannot both consume
    # each other's flanking evidence. Highest score wins, then earliest frame.
    for i in sorted(range(len(rows)),key=lambda k:(-scores[k],rows[k]['start'])):
        row=rows[i]
        if scores[i]<POLICY['merge_score_threshold'] or matrix[i,6]>POLICY['maximum_signed_nearest_mean']:continue
        if any(row['context_start']<v['context_end'] and v['context_start']<row['context_end'] for v in accepted):continue
        output[row['start']:row['end']]=row['replacement']
        accepted.append(dict(row,merge_score=float(scores[i])))
    if return_diagnostics:return output,{'eligible_islands':len(rows),'merged_islands':len(accepted),'accepted':sorted(accepted,key=lambda r:r['start'])}
    return output
