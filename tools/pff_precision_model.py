"""Observation-only, bounded dead-odds correction for isolated PFF research.

This module never fits on references, writes fitted models, or changes v1.5.
The campaign must freeze its finite penalty grid before development selection.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from possession.config import load_config
from possession.decoder import decode
from possession.learned import joint_probabilities, resolve_config, validate_model
from tools.pff_training import _blocks


POLICY = {
    'schema_version': 1,
    'kind': 'bounded_dead_log_odds_penalty',
    'maximum_penalty': 2.,
    'conditional_home_away_odds': 'preserved',
    'decoder': 'unchanged-v1.5',
    'reference_fields_used_for_inference': False,
    'automatic_promotion': False,
}


def _probabilities(values):
    p = np.asarray(values, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or not len(p)
        or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
        or not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-10)):
        raise ValueError('Expected finite normalized nonnegative n by 3 H/A/D probabilities')
    return p


def _penalty(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value) or not 0 <= value <= POLICY['maximum_penalty']):
        raise ValueError('Dead log-odds penalty must be a finite scalar between zero and two')
    return float(value)


def _periods(item, n):
    if item.get('reference_fields_used_for_inference') is not False:
        raise ValueError('Observation-only cache declaration is required')
    blocks, fps = item['blocks'], item['fps']
    _blocks(blocks, n, fps)
    usable = item['usable']
    if len(usable) != len(blocks) or any(type(value) is not bool for value in usable):
        raise ValueError('Each period requires a Boolean observation-support declaration')
    return blocks, fps, usable


def probabilities_from_cache(item, model):
    """Compute the frozen joint output once; never reads event/reference fields."""
    validate_model(model)
    x, z = np.asarray(item['x']), np.asarray(item['z'])
    if (x.ndim != 2 or x.shape[1] != 98 or z.shape != (len(x), 58)
        or not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError('Expected finite observation feature matrices')
    blocks, fps, _ = _periods(item, len(x))
    for key in ('base_home', 'base_dead'):
        value = np.asarray(item[key])
        if value.shape != (len(x),) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError('Invalid frozen upstream probabilities')
    result = joint_probabilities(item['base_dead'], item['base_home'], x, z, blocks, fps,
                                 model['joint_model'], np.zeros(len(x)))
    _probabilities(result)
    return result


def adjust_probabilities(probabilities, penalty):
    """Subtract penalty from D log-odds while preserving H:A conditional odds.

    Multiplying D mass then normalizing is the stable equivalent of a binary
    logit shift. It preserves exact pure-class endpoints; if all live mass is
    zero, no team ownership is invented. Zero penalty is a byte-exact copy.
    """
    p, penalty = _probabilities(probabilities), _penalty(penalty)
    result = p.copy()
    if penalty == 0:
        return result
    result[:, 2] *= np.exp(-penalty)
    result /= result.sum(axis=1, keepdims=True)
    return result


def decode_probabilities(probabilities, item, model, penalty=0., config=None):
    """Decode a cached output with the same periods, support and v1.5 settings."""
    validate_model(model)
    p = adjust_probabilities(probabilities, penalty)
    blocks, fps, usable = _periods(item, len(p))
    config = resolve_config(load_config(ROOT / 'configs/baselines/v1.5.json') if config is None else config, model)
    labels = np.full(len(p), 'X', dtype='<U1')
    for (_, a, b), supported in zip(blocks, usable):
        if supported:
            labels[a:b + 1] = np.array(list('HAD'))[decode(p[a:b + 1], fps, config)]
    return labels, p


def predict_cache(item, model, penalty=0., config=None):
    """Convenience path; a sweep should reuse probabilities_from_cache output."""
    return decode_probabilities(probabilities_from_cache(item, model), item, model, penalty, config)
