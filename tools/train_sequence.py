"""Optional CPU PyTorch training; exported inference is NumPy-only."""
import numpy as np
from possession.temporal import FEATURE_NAMES, DILATIONS, HIDDEN, GATE_NAMES, validate_model, validate_gate


def fit(sequences, *, seed=9207, epochs=8, metadata=None):
    """Sequences have X/y/w/base at 5Hz, each wholly contained in one half.

    The caller must exclude held-out matches. Scaling uses positive-weight H/A
    training rows only. D/X centers have zero loss, while observations may remain
    temporal context. Chunk halos prevent artificial internal chunk boundaries.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    if not sequences or not isinstance(epochs,int) or epochs < 1: raise ValueError("Need training sequences and positive epochs")
    total = 0.; square = np.zeros(len(FEATURE_NAMES)); rows = 0
    for item in sequences:
        x,y,w = np.asarray(item['X']),np.asarray(item['y']),np.asarray(item['w'])
        if x.shape != (len(y),len(FEATURE_NAMES)) or w.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any():
            raise ValueError("Invalid temporal training sequence")
        if not np.isin(y,['H','A','D','X']).all(): raise ValueError("Unknown training label")
        if np.asarray(item['base']).shape != y.shape or not np.isfinite(item['base']).all():
            raise ValueError("Invalid base training logits")
        good = np.isin(y,['H','A']) & (w>0)
        square += np.sum(x[good]**2*w[good,None],axis=0); total += w[good].sum(); rows += int(good.sum())
    if total <= 0: raise ValueError("No live training rows")
    scale = np.sqrt(square/total); scale[scale<1e-8]=1
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(len(FEATURE_NAMES),HIDDEN,bias=False)
            self.convs = nn.ModuleList([nn.Conv1d(HIDDEN,HIDDEN,3,padding=d,dilation=d,bias=False) for d in DILATIONS])
            self.output = nn.Linear(HIDDEN,1,bias=False)
            nn.init.zeros_(self.output.weight)
        def forward(self,x,valid):
            h = torch.tanh(self.input(x)).transpose(1,2)*valid[:,None,:]
            for conv in self.convs: h = torch.tanh(h+conv(h))*valid[:,None,:]
            return self.output(h.transpose(1,2)).squeeze(-1)
    net=Net(); optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
    halo=sum(DILATIONS); width=128; prepared=[]; windows=[]
    for i,item in enumerate(sequences):
        n=len(item['y']); y=np.asarray(item['y']); w=np.asarray(item['w'])*np.isin(y,['H','A'])
        x=np.clip(np.asarray(item['X'])/scale,-8,8).astype(np.float32)
        prepared.append((x,(y=='H').astype(np.float32),w.astype(np.float32),np.asarray(item['base'],dtype=np.float32)))
        windows.extend((i,start) for start in range(0,n,width) if w[start:start+width].sum()>0)
    rng=np.random.default_rng(seed); losses=[]
    for epoch in range(epochs):
        order=rng.permutation(len(windows)); total_loss=0.; total_weight=0.
        for at in range(0,len(order),16):
            batch=[windows[k] for k in order[at:at+16]]; size=len(batch)
            bx=np.zeros((size,width+2*halo,len(scale)),dtype=np.float32)
            bm=np.zeros((size,width+2*halo),dtype=np.float32)
            by=np.zeros((size,width),dtype=np.float32); bw=by.copy(); bb=by.copy()
            for j,(i,start) in enumerate(batch):
                x,y,w,base=prepared[i]; stop=min(start+width,len(x)); left=max(0,start-halo); right=min(len(x),start+width+halo)
                bx[j,left-start+halo:right-start+halo]=x[left:right]
                bm[j,left-start+halo:right-start+halo]=1
                by[j,:stop-start]=y[start:stop]; bw[j,:stop-start]=w[start:stop]; bb[j,:stop-start]=base[start:stop]
            optimizer.zero_grad(set_to_none=True)
            logit=net(torch.from_numpy(bx),torch.from_numpy(bm))[:,halo:halo+width]+torch.from_numpy(bb)
            weight=torch.from_numpy(bw); loss=(F.binary_cross_entropy_with_logits(logit,torch.from_numpy(by),reduction='none')*weight).sum()/weight.sum()
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.); optimizer.step()
            total_loss += float(loss.detach())*float(weight.sum()); total_weight += float(weight.sum())
        losses.append(total_loss/total_weight)
        print(f"  temporal epoch {epoch+1}/{epochs}: weighted loss {losses[-1]:.6f}",flush=True)
    model={'schema_version':1,'kind':'odd_temporal_residual','feature_names':list(FEATURE_NAMES),'sample_hz':5,
        'dilations':list(DILATIONS),'scale':scale.tolist(),'input_weight':net.input.weight.detach().numpy().tolist(),
        'output_weight':net.output.weight.detach().numpy().tolist(),
        **{f'conv_{i}':layer.weight.detach().numpy().tolist() for i,layer in enumerate(net.convs)},
        'training':{'seed':seed,'epochs':epochs,'positive_weight_live_rows':rows,'losses':losses,'torch_version':torch.__version__,
                    'optimizer':'AdamW lr=.003 weight_decay=.01 clip_grad_norm=5','cpu_threads':1},
        'metadata':metadata or {}}
    return validate_model(model)


def fit_gate(matrix, base, expert, labels, weights, metadata=None):
    """Learn a bounded mixture on an inner held-out match, never outer score data."""
    from scipy.optimize import minimize
    from possession.temporal import gate_features
    x=gate_features(base,expert,matrix); y=np.asarray(labels); w=np.asarray(weights,dtype=float)
    if y.shape != (len(x),) or w.shape != y.shape or not np.isin(y,['H','A','D','X']).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Invalid gate labels or weights')
    use=np.isin(y,['H','A'])&(w>0)
    if not use.any(): raise ValueError('No live gate calibration rows')
    x=x[use]; target=(y[use]=='H').astype(float); w=w[use]/w[use].sum()
    sigmoid=lambda z:1/(1+np.exp(-np.clip(z,-35,35)))
    p=sigmoid(np.asarray(base)[use]); q=sigmoid(np.asarray(expert)[use])
    def objective(c):
        g=sigmoid(x@c); mix=np.clip(p+g*(q-p),1e-7,1-1e-7)
        loss=-np.sum(w*(target*np.log(mix)+(1-target)*np.log1p(-mix)))+.005*(c@c)
        grad=x.T@(w*(mix-target)/(mix*(1-mix))*(q-p)*g*(1-g))+.01*c
        return loss,grad
    result=minimize(objective,np.zeros(len(GATE_NAMES)),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12,'gtol':1e-8})
    if not result.success or not np.isfinite(result.x).all(): raise ValueError('Gate fitting failed: '+result.message)
    return validate_gate({'kind':'reliability_expert_gate','feature_names':list(GATE_NAMES),'coefficients':result.x.tolist(),
        'training':{'rows':int(use.sum()),'objective':float(result.fun),'inner_holdout':True},'metadata':metadata or {}})
