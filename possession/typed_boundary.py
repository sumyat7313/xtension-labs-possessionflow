"""Explicit typed boundary evidence with bounded, half-local transition costs.

Four mutually exclusive neighborhood classes share observation features. These
probabilities describe boundary neighborhoods, not per-frame event hazards or
externally calibrated confidence. Inference never accepts reference labels.
"""
import json
import numpy as np
from . import boundary

CLASSES=('none','team_change','stop','restart')
DEAD_NAMES=('ball_observed','ball_reliability','ball_log_speed','ball_speed_known',
    'nearest_distance','nearest_known','outside_support','boundary_distance','boundary_known',
    'player_coverage','player_p25_speed','player_median_speed','player_p75_speed','player_mean_speed',
    'players_below_1mps','players_below_2mps','players_below_3mps','stationary_support','missing_support')
RADII=(.12,1.)
FEATURE_NAMES=tuple('team_'+n for n in boundary.FEATURE_NAMES)+tuple('state_'+n for n in DEAD_NAMES)+tuple(
    f'{signal}_{kind}_{seconds:g}s' for seconds in RADII for signal in ('dead_probability','ball_log_speed','stationary_support')
    for kind in ('past','future','difference'))
POLICY={'target_radius_seconds':.12,'maximum_cost_adjustment':.25,'evidence_odds_scale':4.,
        'team_context_seconds':.12,'minimum_team_margin':.1,'dead_context_seconds':1.,
        'minimum_dead_context_change':.05,'minimum_complete_context':True}


def feature_names(contact=False):
    if type(contact) is not bool:raise ValueError('Contact feature option must be boolean')
    if contact:
        from .contact_evidence import FEATURE_NAMES as CONTACT_NAMES
        return FEATURE_NAMES+tuple('contact_'+n for n in CONTACT_NAMES)
    return FEATURE_NAMES


def _context(values,sections,fps,seconds):
    radius=max(1,round(seconds*fps));past=np.zeros_like(values);future=np.zeros_like(values);complete=np.zeros(len(values),bool)
    for a,b in sections:
        v=values[a:b+1];n=len(v);i=np.arange(n);left=np.maximum(0,i-radius);right=np.minimum(n,i+radius)
        prefix=np.vstack((np.zeros((1,v.shape[1])),np.cumsum(v,axis=0)))
        lp,rp=i-left,right-i
        past[a:b+1]=np.divide(prefix[i]-prefix[left],lp[:,None],out=np.zeros_like(v),where=lp[:,None]>0)
        future[a:b+1]=(prefix[right]-prefix[i])/rp[:,None]
        complete[a:b+1]=(lp==radius)&(rp==radius)
    return past,future,complete


def feature_matrix(control_matrix,dead_matrix,probabilities,blocks,fps,contact=False):
    x,z,p=np.asarray(control_matrix,float),np.asarray(dead_matrix,float),np.asarray(probabilities,float)
    base=boundary.feature_matrix(x,p,blocks,fps)
    if z.shape!=(len(x),58) or not np.isfinite(z).all():raise ValueError('Typed boundary dead observations must be finite n by 58')
    sections,active=boundary._blocks(blocks,len(x));columns=[base,z[:,:19]]
    signals=np.column_stack((p[:,2],z[:,2],z[:,17]))
    for seconds in RADII:
        past,future,_=_context(signals,sections,fps,seconds)
        columns.append(np.column_stack([v for i in range(3) for v in (past[:,i],future[:,i],future[:,i]-past[:,i])]))
    if contact:
        from .contact_evidence import feature_matrix as contact_matrix
        columns.append(contact_matrix(x,z,blocks,fps))
    result=np.column_stack(columns);result[~active]=0
    if result.shape!=(len(x),len(feature_names(contact))) or not np.isfinite(result).all():raise ValueError('Invalid typed boundary feature matrix')
    return result


def validate_model(model):
    if not isinstance(model,dict) or model.get('kind')!='typed_boundary_softmax' or model.get('schema_version')!=1:
        raise ValueError('Unsupported typed-boundary model')
    contact=model.get('contact_features');names=feature_names(contact)
    if model.get('feature_names')!=list(names) or model.get('classes')!=list(CLASSES):raise ValueError('Typed boundary feature/class schema differs')
    policy=model.get('policy')
    if not isinstance(policy,dict) or set(policy)!=set(POLICY):raise ValueError('Typed boundary policy differs')
    for key,value in POLICY.items():
        actual=policy[key]
        if type(value) is bool:
            if actual is not value:raise ValueError('Typed boundary policy differs')
        elif isinstance(actual,bool) or not isinstance(actual,(int,float)) or not np.isfinite(actual) or actual!=value:
            raise ValueError('Typed boundary policy differs')
    for key,shape in [('mean',(len(names),)),('scale',(len(names),)),('coefficients',(len(names),3)),('intercepts',(3,)),('priors',(4,))]:
        v=np.asarray(model.get(key),float)
        if v.shape!=shape or not np.isfinite(v).all():raise ValueError('Invalid typed boundary '+key)
        if key in ('scale','priors') and (v<=0).any():raise ValueError('Typed boundary '+key+' must be positive')
    if not np.isclose(np.sum(model['priors']),1.,rtol=0,atol=1e-10):raise ValueError('Typed boundary priors must sum to one')
    json.dumps(model,allow_nan=False);return model


def predict_from_features(model,matrix):
    validate_model(model);x=np.asarray(matrix,float)
    if x.ndim!=2 or x.shape[1]!=len(model['feature_names']) or not np.isfinite(x).all():raise ValueError('Invalid typed boundary observations')
    z=np.clip((x-np.asarray(model['mean']))/np.asarray(model['scale']),-8,8)
    eta=z@np.asarray(model['coefficients'])+np.asarray(model['intercepts'])
    logits=np.column_stack((np.zeros(len(x)),eta));logits-=logits.max(axis=1,keepdims=True)
    p=np.exp(logits);return p/p.sum(axis=1,keepdims=True)


def evidence(model,control_matrix,dead_matrix,probabilities,blocks,fps,config):
    matrix=feature_matrix(control_matrix,dead_matrix,probabilities,blocks,fps,model['contact_features'])
    q=predict_from_features(model,matrix);p=np.asarray(probabilities,float);sections,active=boundary._blocks(blocks,len(p))
    prior=np.asarray(model['priors']);relative=np.log(np.maximum(q[:,1:],1e-15))-np.log(np.maximum(q[:,:1],1e-15))-np.log(prior[1:]/prior[0])
    relative[np.abs(relative)<1e-12]=0
    strength=np.tanh(relative/np.log(POLICY['evidence_odds_scale']))
    live=1-p[:,2];margin=np.divide(p[:,0]-p[:,1],live,out=np.zeros(len(p)),where=live>1e-8)
    before,after,team_complete=_context(margin[:,None],sections,fps,POLICY['team_context_seconds'])
    m=POLICY['minimum_team_margin'];team=((before[:,0]>=m)&(after[:,0]<=-m))|((before[:,0]<=-m)&(after[:,0]>=m))
    before,after,dead_complete=_context(p[:,2:3],sections,fps,POLICY['dead_context_seconds'])
    change=after[:,0]-before[:,0];threshold=POLICY['minimum_dead_context_change']
    gates=np.column_stack((team&team_complete,(change>=threshold)&dead_complete,(change<=-threshold)&dead_complete))
    complete=np.column_stack((team_complete,dead_complete,dead_complete))
    strength=np.where((strength<=0)|gates,strength,0);strength[~complete]=0;strength[~active]=0
    for start,_ in sections:strength[start]=0
    bases=np.asarray([config.team_switch_cost,config.stop_cost,config.restart_cost],float)
    if not np.isfinite(bases).all() or (bases<0).any():raise ValueError('Invalid base transition costs')
    costs=bases*(1-POLICY['maximum_cost_adjustment']*strength)
    return dict(zip(('team_switch_costs','stop_costs','restart_costs'),costs.T))
