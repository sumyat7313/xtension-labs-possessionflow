"""Training-only PFF context residual with NumPy-only observation inference.

The frozen parent supplies D log-odds. This small head learns a bounded additive
correction; references enter only fit(). It never changes conditional H:A odds.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from possession.config import load_config
from possession.decoder import decode
from possession.learned import FEATURE_NAMES as DEAD_FEATURE_NAMES, resolve_config, validate_model as validate_parent
from tools import pff_precision_model as precision

PARENT_SHA256 = '0ab133743c593466d8ae92ea94937e8c6bf22510f6b64053e3e69c7b467dcf3f'
FEATURE_NAMES = tuple(DEAD_FEATURE_NAMES) + ('parent_dead_logit', 'parent_absolute_team_confidence')
POLICY = {'schema_version': 1, 'kind': 'pff_context_dead_residual', 'hidden': 32,
          'residual_limit': 2., 'epochs': 4, 'seed': 20261103, 'sample_hz': 5,
          'batch_size': 2048, 'learning_rate': .001, 'weight_decay': .01,
          'standardized_clip': 8., 'parent_logit_clip': 1e-6,
          'weighting': 'equal total sampled weight per training fixture; unknown samples excluded',
          'objective': 'weighted D BCE with frozen parent logit offset',
          'conditional_home_away_odds': 'preserved', 'reference_fields_used_for_inference': False,
          'automatic_promotion': False}


def _logit(dead):
    dead = np.clip(dead, POLICY['parent_logit_clip'], 1-POLICY['parent_logit_clip'])
    return np.log(dead)-np.log1p(-dead)


def features(z, probabilities):
    z, p = np.asarray(z, dtype=float), precision._probabilities(probabilities)
    if z.shape != (len(p), len(DEAD_FEATURE_NAMES)) or not np.isfinite(z).all():
        raise ValueError('Expected finite n by 58 observation features')
    live = p[:, 0]+p[:, 1]
    confidence = np.divide(np.abs(p[:, 0]-p[:, 1]), live, out=np.zeros(len(p)), where=live > 0)
    return np.column_stack((z, _logit(p[:, 2]), confidence))


def validate_head(head):
    if (not isinstance(head, dict) or head.get('schema_version') != 1
        or head.get('kind') != POLICY['kind'] or head.get('policy') != POLICY
        or head.get('feature_names') != list(FEATURE_NAMES)):
        raise ValueError('Unexpected context residual schema or policy')
    shapes = {'mean': (60,), 'scale': (60,), 'input_weight': (32, 60), 'input_bias': (32,),
              'output_weight': (32,), 'output_bias': ()}
    for key, shape in shapes.items():
        value = np.asarray(head.get(key), dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid context parameter '+key)
    if np.any(np.asarray(head['scale']) <= 0):
        raise ValueError('Context scales must be positive')
    meta = head.get('metadata', {})
    if meta.get('reference_fields_used_for_inference') is not False or meta.get('parent_sha256') != PARENT_SHA256:
        raise ValueError('Context parent/reference declaration differs')
    json.dumps(head, allow_nan=False)
    return head


def normalized(feature_matrix, head):
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 60 or not np.isfinite(matrix).all():
        raise ValueError('Invalid context feature matrix')
    return np.clip((matrix-np.asarray(head['mean']))/np.asarray(head['scale']),
                   -POLICY['standardized_clip'], POLICY['standardized_clip'])


def forward(feature_matrix, head):
    validate_head(head)
    x = normalized(feature_matrix, head)
    hidden = np.tanh(x @ np.asarray(head['input_weight']).T + np.asarray(head['input_bias']))
    value = hidden @ np.asarray(head['output_weight']) + head['output_bias']
    limit = POLICY['residual_limit']
    return limit*np.tanh(value/limit)


def residuals(item, parent_probabilities, head):
    p = precision._probabilities(parent_probabilities)
    precision._periods(item, len(p))
    z = np.asarray(item['z'])
    if z.shape != (len(p), 58):
        raise ValueError('Context observation shape differs')
    validate_head(head)
    result = np.empty(len(p))
    for start in range(0, len(p), 16384):
        stop = min(start+16384, len(p))
        result[start:stop] = forward(features(z[start:stop], p[start:stop]), head)
    return result


def adjust_probabilities(probabilities, residual, blend):
    p = precision._probabilities(probabilities)
    if (isinstance(blend, (bool, np.bool_)) or not isinstance(blend, (int, float, np.integer, np.floating))
        or not np.isfinite(blend) or not 0 <= blend <= 1):
        raise ValueError('Context blend must be a finite scalar between zero and one')
    residual = np.asarray(residual, dtype=float)
    if residual.shape != (len(p),) or not np.isfinite(residual).all() or np.any(np.abs(residual) > POLICY['residual_limit']+1e-12):
        raise ValueError('Context residual must be finite and bounded per frame')
    q = p.copy()
    if blend == 0:
        return q
    q[:, 2] *= np.exp(blend*residual)
    q /= q.sum(axis=1, keepdims=True)
    return q


def predict_cache(item, parent_model, head, blend=1., parent_probabilities=None, config=None):
    validate_parent(parent_model)
    p = precision.probabilities_from_cache(item, parent_model) if parent_probabilities is None else precision._probabilities(parent_probabilities)
    q = adjust_probabilities(p, residuals(item, p, head), blend)
    blocks, fps, usable = precision._periods(item, len(q))
    config = resolve_config(load_config(ROOT / 'configs/baselines/v1.5.json') if config is None else config, parent_model)
    labels = np.full(len(q), 'X', dtype='<U1')
    for (_, a, b), supported in zip(blocks, usable):
        if supported:
            labels[a:b+1] = np.array(list('HAD'))[decode(q[a:b+1], fps, config)]
    return labels, q


def training_samples(items, training_matches, held_out_matches):
    train, held = set(training_matches), set(held_out_matches)
    if (not train or len(train) != len(training_matches) or len(held) != len(held_out_matches)
        or train & held or any(type(mid) is not int for mid in train | held)):
        raise ValueError('Require unique disjoint integer training and held-out fixture IDs')
    chunks, targets, offsets, weights, support, seen = [], [], [], [], {}, set()
    total = 0.; sums = np.zeros(60); squares = np.zeros(60)
    for item in items:
        mid = item['match']
        if type(mid) is not int or mid not in train or mid in held or mid in seen:
            raise ValueError('Unexpected, duplicate or held-out training fixture')
        seen.add(mid)
        y = np.asarray(item['y'])
        p = precision._probabilities(item['probabilities'])
        if y.shape != (len(p),) or not np.isin(y, list('HADX')).all():
            raise ValueError('Invalid training references')
        blocks, fps, usable = precision._periods(item, len(y))
        stride = round(fps/POLICY['sample_hz'])
        indices = np.concatenate([np.arange(a, b+1, stride) for (_, a, b), good in zip(blocks, usable) if good]) if any(usable) else np.array([], dtype=int)
        indices = indices[y[indices] != 'X']
        if not len(indices):
            raise ValueError('Training fixture has no supported sampled labels')
        z = np.asarray(item['z'])
        if z.shape != (len(y), 58) or not np.isfinite(z).all():
            raise ValueError('Invalid training observation features')
        values = features(z[indices], p[indices])
        w = np.full(len(indices), 1./len(indices))
        sums += np.sum(values*w[:, None], axis=0)
        squares += np.sum(values**2*w[:, None], axis=0); total += w.sum()
        chunks.append(values.astype(np.float32)); targets.append((y[indices] == 'D').astype(np.float32))
        offsets.append(_logit(p[indices, 2]).astype(np.float32)); weights.append(w.astype(np.float32))
        support[str(mid)] = {'sampled_rows': len(indices), 'sampled_class_counts': {c: int(np.sum(y[indices] == c)) for c in 'HAD'},
                            'sampled_fixture_mass_before_global_normalization': 1.}
    if seen != train:
        raise ValueError('Every declared training fixture is required')
    x, y, offset, w = map(np.concatenate, (chunks, targets, offsets, weights))
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError('Context training needs both live and dead observations')
    mean = sums/total; scale = np.sqrt(np.maximum(0., squares/total-mean**2)); scale[scale < 1e-6] = 1.
    w *= len(w)/len(train)
    for start in range(0, len(x), 16384):
        x[start:start+16384] = np.clip((x[start:start+16384]-mean)/scale,
                                      -POLICY['standardized_clip'], POLICY['standardized_clip'])
    return x, y, offset, w, mean, scale, support


def make_network():
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(60, 32)
            self.output = nn.Linear(32, 1)
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        def forward(self, x):
            value = self.output(torch.tanh(self.input(x))).squeeze(-1)
            return POLICY['residual_limit']*torch.tanh(value/POLICY['residual_limit'])
    return Network()


def export_head(net, mean, scale, metadata, training):
    value = lambda tensor: tensor.detach().cpu().numpy().tolist()
    return validate_head({'schema_version': 1, 'kind': POLICY['kind'], 'policy': copy.deepcopy(POLICY),
                          'feature_names': list(FEATURE_NAMES), 'mean': np.asarray(mean).tolist(), 'scale': np.asarray(scale).tolist(),
                          'input_weight': value(net.input.weight), 'input_bias': value(net.input.bias),
                          'output_weight': value(net.output.weight[0]), 'output_bias': value(net.output.bias[0]),
                          'metadata': copy.deepcopy(metadata), 'training': copy.deepcopy(training)})


def fit(items, *, training_matches, held_out_matches, protocol_sha256):
    if (not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64
        or any(c not in '0123456789abcdef' for c in protocol_sha256)):
        raise ValueError('A frozen training protocol SHA256 is required')
    import torch
    from torch.nn import functional as F
    x, y, offset, w, mean, scale, support = training_samples(items, training_matches, held_out_matches)
    torch.set_num_threads(1); torch.manual_seed(POLICY['seed']); torch.use_deterministic_algorithms(True)
    net = make_network()
    optimizer = torch.optim.AdamW(net.parameters(), lr=POLICY['learning_rate'], weight_decay=POLICY['weight_decay'])
    rng = np.random.default_rng(POLICY['seed']); losses = []
    for epoch in range(POLICY['epochs']):
        order = rng.permutation(len(x)); weighted_loss = 0.; mass = 0.
        for start in range(0, len(order), POLICY['batch_size']):
            index = order[start:start+POLICY['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            residual = net(torch.from_numpy(x[index]))
            values = F.binary_cross_entropy_with_logits(torch.from_numpy(offset[index])+residual,
                                                        torch.from_numpy(y[index]), reduction='none')
            weight = torch.from_numpy(w[index]); loss = (values*weight).mean()
            if not torch.isfinite(loss):
                raise ValueError('Context fit produced a nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            weighted_loss += float((values.detach()*weight).sum()); mass += float(weight.sum())
        losses.append(weighted_loss/mass)
        print(f'context epoch {epoch+1}/{POLICY["epochs"]}: weighted D loss {losses[-1]:.6f}', flush=True)
    metadata = {'training_matches': sorted(training_matches), 'held_out_matches': sorted(held_out_matches),
                'protocol_sha256': protocol_sha256, 'policy': copy.deepcopy(POLICY), 'support': support,
                'parent_sha256': PARENT_SHA256, 'reference_fields_used_for_inference': False}
    training = {'losses': losses, 'epochs': POLICY['epochs'], 'seed': POLICY['seed'], 'sampled_rows': len(x),
                'cpu_threads': 1, 'torch_version': torch.__version__, 'conditional_home_away_trained': False}
    return export_head(net, mean, scale, metadata, training)
