"""Compose frozen context evidence with a bounded global D-logit penalty.

No model is fitted here. The caller freezes the permitted blend/penalty pairs
before development selection. Conditional H:A odds and decoding stay fixed.
"""
from tools import pff_context_model as context, pff_precision_model as precision


POLICY = {'schema_version': 1, 'kind': 'pff_context_then_dead_penalty',
          'parameters': ['context_blend', 'dead_logit_penalty'],
          'order': ['frozen_context_D_residual', 'global_D_logit_penalty', 'unchanged_v1.5_decoder'],
          'conditional_home_away_odds': 'preserved',
          'reference_fields_used_for_inference': False, 'fit_performed': False,
          'automatic_promotion': False}


def probabilities_from_cache(item, parent_model):
    return precision.probabilities_from_cache(item, parent_model)


def residuals(item, probabilities, head):
    return context.residuals(item, probabilities, head)


def decode_probabilities(probabilities, item, parent_model, head, blend,
                         dead_logit_penalty, precomputed_residual=None):
    context.validate_head(head)
    # Validation of supplied residual shape/bounds is performed by the frozen
    # context operation, whether computed here or supplied from a cached pass.
    value = residuals(item, probabilities, head) if precomputed_residual is None else precomputed_residual
    adjusted = context.adjust_probabilities(probabilities, value, blend)
    adjusted = precision.adjust_probabilities(adjusted, dead_logit_penalty)
    return precision.decode_probabilities(adjusted, item, parent_model, penalty=0.)
