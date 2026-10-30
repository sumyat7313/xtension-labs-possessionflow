"""Observed-only contact/restart geometry, not verified touches or video evidence.

All cached features are H/A-even and depend only on documented observation
columns. Complete observed windows are required; interpolation never counts as
contact support. A learned outer-fold evaluation must decide their usefulness.
"""
import numpy as np
from .boundary import _blocks

FEATURE_NAMES = ('observed_context','uncontested_contact','uncontested_deceleration',
    'stationary_launch','boundary_stationary_launch','outside_return',
    'player_motion_launch','stationary_near_ball')
POLICY = {'contact_context_seconds':.2,'stationary_context_seconds':.6,
    'launch_context_seconds':.2,'maximum_near_distance_metres':2.,
    'minimum_team_distance_margin_metres':1.,'maximum_stationary_speed_mps':.8,
    'minimum_launch_speed_mps':3.,'maximum_ball_speed_mps':45.,
    'independent_video_evidence':False,'calibrated_contact_probability':False}


def _validate(x,z,blocks,fps):
    x,z=np.asarray(x,float),np.asarray(z,float)
    if x.ndim!=2 or x.shape[1]!=98 or z.shape!=(len(x),58) or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Contact features require finite n by98 control and n by58 dead observations')
    if isinstance(fps,(bool,np.bool_)) or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid FPS')
    sections,active=_blocks(blocks,len(x))
    return x,z,sections,active


def _window(values,sections,left,right):
    """Inclusive fixed window mean and full-context mask, never truncated."""
    values=np.asarray(values,float);out=np.zeros(len(values));full=np.zeros(len(values),bool)
    for a,b in sections:
        n=b-a+1;idx=np.arange(n);lo=idx+left;hi=idx+right+1
        valid=(lo>=0)&(hi<=n)
        if not valid.any():continue
        prefix=np.r_[0.,np.cumsum(values[a:b+1])]
        out[a+idx[valid]]=(prefix[hi[valid]]-prefix[lo[valid]])/(right-left+1)
        full[a+idx[valid]]=True
    return out,full


def feature_matrix(control_matrix,dead_matrix,blocks,fps):
    """Eight bounded even proxy features from existing documented caches.

    Z[0,1,3,4,5,7,9,11] are observed, reliability, speed-known,
    clipped-nearest-distance/20, nearest-known, edge-distance/5, player-coverage,
    player-median-speed/13. Z[2] encodes log1p(speed)/log46 and Z[6] is smoothed
    outside support. X[0] is reliability-weighted tanh(team-distance-margin/3),
    X[11] weighted signed contact control, X[56] weighted signed deceleration.
    No hidden native distance or ball position is reconstructed from saturation.
    """
    x,z,sections,active=_validate(control_matrix,dead_matrix,blocks,fps)
    r=max(1,int(np.ceil(.2*fps)));past=max(1,int(np.ceil(.6*fps)))
    observed=(z[:,0]>=1-1e-8)&(z[:,1]>=1-1e-8)&(z[:,3]>=1-1e-8)
    seen,full=_window(observed,sections,-r,r);context=full&(seen>=1-1e-8)&active
    long_seen,long_full=_window(observed,sections,-past,r)
    restart_context=long_full&(long_seen>=1-1e-8)&active
    speed=np.expm1(np.clip(z[:,2],0,1)*np.log(46))
    near=np.clip(1-z[:,4]*20/POLICY['maximum_near_distance_metres'],0,1)
    unique=(np.abs(x[:,0])>=np.tanh(1/3))&(z[:,5]>=1-1e-8)&(z[:,9]>=.5)
    support=context&unique
    contact=np.clip(np.abs(x[:,11]),0,1)*near*support
    deceleration=np.clip(np.abs(x[:,56]),0,1)*near*support
    before,_=_window(speed,sections,-past,-1)
    after,_=_window(speed,sections,0,r)
    slow=np.clip(1-before/.8,0,1);launch=np.clip((after-3)/7,0,1)
    restart=restart_context*unique*near*slow*launch
    edge=np.clip(1-z[:,7]*5,0,1)
    outside_before,_=_window(np.clip(z[:,6],0,1),sections,-past,-1)
    outside_after,_=_window(np.clip(z[:,6],0,1),sections,0,r)
    return_support=np.clip(outside_before-outside_after,0,1)*restart_context*near*unique*launch
    player_before,_=_window(z[:,11]*13,sections,-past,-1)
    player_after,_=_window(z[:,11]*13,sections,0,r)
    motion=restart*np.clip((player_after-player_before)/2,0,1)
    result=np.column_stack((context.astype(float),contact,deceleration,restart,restart*edge,
                            return_support,motion,restart_context*unique*near*slow))
    result[~active]=0
    if result.shape!=(len(x),len(FEATURE_NAMES)) or not np.isfinite(result).all():raise ValueError('Invalid contact features')
    return np.clip(result,0,1)


def extract_native(tracking,blocks,pitch,fps):
    """Conservative native-geometry candidates with explicit abstention.

    Reads raw XY only, never supplied speed, event/state fields or player IDs.
    A velocity change near an unambiguous player is a geometric contact proxy,
    not proof of a touch. Stationary launch is not proof that play was stopped.
    """
    ball=np.asarray(tracking.ball,float);players=np.asarray(tracking.players,float);teams=np.asarray(tracking.teams)
    pitch=np.asarray(pitch,float)
    if pitch.shape!=(2,) or not np.isfinite(pitch).all() or (pitch<=0).any():raise ValueError('Invalid pitch')
    if isinstance(fps,(bool,np.bool_)) or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid FPS')
    if ball.ndim!=2 or ball.shape[1]!=2 or players.ndim!=3 or players.shape[0]!=len(ball) or players.shape[2]!=2:
        raise ValueError('Invalid native geometry shape')
    if teams.shape!=(players.shape[1],) or not np.isin(teams,[0,1]).all() or not all((teams==t).any() for t in (0,1)):
        raise ValueError('Native geometry needs both teams')
    sections,active=_blocks(blocks,len(ball));n=len(ball)
    good=np.isfinite(ball).all(axis=1)&(ball>=-.15).all(axis=1)&(ball<=1.15).all(axis=1)&active
    b=ball*pitch;r=max(1,int(np.ceil(.12*fps)));past=max(1,int(np.ceil(.6*fps)))
    velocity_before=np.zeros((n,2));velocity_after=np.zeros((n,2));steps=np.full(n,np.inf)
    for a,last in sections:
        if last-a>=1:steps[a+1:last+1]=np.linalg.norm(np.diff(b[a:last+1],axis=0),axis=1)*fps
        if last-a>=2*r:
            ii=np.arange(a+r,last-r+1)
            velocity_before[ii]=(b[ii]-b[ii-r])*fps/r
            velocity_after[ii]=(b[ii+r]-b[ii])*fps/r
    # One implausible or missing raw step blocks the entire context window.
    physical=good&np.isfinite(steps)&(steps<=45)
    seen,full=_window(physical,sections,-r,r);supported=full&(seen>=1-1e-8)
    long_seen,long_full=_window(physical,sections,-past,r)
    long_supported=long_full&(long_seen>=1-1e-8)
    near=np.full((n,2),np.inf)
    for a in range(0,n,4096):
        stop=min(n,a+4096);p=players[a:stop]
        valid=np.isfinite(p).all(axis=2)&(p>=-.2).all(axis=2)&(p<=1.2).all(axis=2)
        distances=np.linalg.norm(p*pitch-b[a:stop,None],axis=2)
        distances[~valid|~np.isfinite(distances)]=np.inf
        for t in (0,1):near[a:stop,t]=np.min(distances[:,teams==t],axis=1)
    finite=np.isfinite(near).all(axis=1);closest=np.min(near,axis=1)
    margin=np.zeros(n);margin[finite]=near[finite,1]-near[finite,0]
    unique=finite&(closest<=1.5)&(np.abs(margin)>=1)
    proximity=np.clip(1-closest/1.5,0,1)
    change=np.linalg.norm(velocity_after-velocity_before,axis=1)
    contact=np.clip((np.nan_to_num(change,nan=0)-3)/7,0,1)*proximity*supported*unique
    incoming,_=_window(np.where(np.isfinite(steps),steps,100),sections,-past,-1)
    outgoing=np.linalg.norm(velocity_after,axis=1)
    restart=np.clip(1-incoming/.8,0,1)*np.clip((np.nan_to_num(outgoing,nan=0)-3)/7,0,1)*proximity*long_supported*unique
    signed=np.sign(margin)
    return {'contact_strength':contact,'restart_strength':restart,'signed_team_support':signed*contact,
            'observed_support':supported&unique,'restart_observed_support':long_supported&unique,
            'verified_contact':False,'verified_restart':False,'independent_video_evidence':False}
