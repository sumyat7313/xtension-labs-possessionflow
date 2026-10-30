# V7 explicit typed-boundary experiment

This is a fixed candidate design, not a claimed improvement. The parent campaign
must compare it with original v1.5 out-of-fold predictions and then audit one
frozen eligible candidate on unused sources. No challenge reference or fresh-audit
label enters this model's training.

## Why explicit boundaries

V6's duration-weighted D objective increased frame recall but harmed D-segment
precision. Its H/A discount improved short recall but could introduce extra
segments. V7 therefore learns boundary categories directly and includes explicit
no-boundary evidence that can increase transition costs, as well as reducing them
when a transition is supported.

The design follows the separation of frame classification and boundary detection
motivated by [ASRF](https://arxiv.org/abs/2007.06866). This implementation is a small
multinomial model on tracking-derived features, not a reproduction of ASRF's video
network. Published results on other action datasets do not establish gains here.

## Shared features and jointly fitted classes

A four-class softmax jointly predicts `none`, `team_change`, `stop`, and `restart`.
The three event classes respectively mean direct H↔A, live→D and D→live. All
classes share 90 native-frame observation features: the existing 53 H/A-symmetric
boundary features, 19 raw unsigned live/dead observations, and 18 past/future/
directional-change features for D probability, ball speed and stationarity at
0.12-second and 1-second scales. Context is offline and strictly half-local.

An explicit ablation appends eight contact/restart proxy features from
`possession.contact_evidence`. The model records `contact_features` and exact
feature names, so missing or different evidence cannot be silently substituted.
Those proxies derive from existing tracking observations; they are not independent
video-verified touches. Changing the team ordering leaves both feature variants
and all three cost vectors unchanged.

## Targets and training

Targets mark ±0.12-second neighborhoods around observed typed transitions. The
nearest edge owns overlapping neighborhoods; exact ties go to the earlier edge.
One frame cannot be simultaneously a stop and a restart. No target crosses an X
gap or a half boundary, and first frames and frames following X are excluded.

All positive-neighborhood frames are retained at the native rate. Negatives are
sampled every five frames independently in each half. Their weights are the
half's original negative count divided by sampled negative count, preserving
exact original class mass despite sampling. If a half's regular stride misses
all negatives, its first negative is retained. This is not an event-frequency or
per-second hazard model; priors describe the fixed neighborhood target.

Training-only weighted means/scales standardize the shared features. The four
classes receive equal total loss mass in a deterministic L-BFGS-B softmax fit,
with L2=0.01, maximum 300 iterations, ftol=1e-11 and gtol=1e-7. Correcting each
event intercept by log(prior_event/prior_none) restores the empirical training
prior. This analytical correction is not a claim of external calibration.
H/A-only spell weights are rejected, since they would incorrectly zero D frames.

## Bounded transition costs and segment consistency

For each event class, compute the log event-versus-none odds relative to its
training prior odds. Map this evidence through `tanh(log_odds/log(4))`, then use
`base_cost * (1 - 0.25*strength)`. Costs therefore stay within 75–125% of baseline;
at the training prior they are exactly unchanged. Negative evidence raises costs
to discourage false segmentation. Positive discounts require complete supporting
context: opposite H/A margins of at least 0.1 across 0.12-second windows, or a
directionally correct D-probability change of at least 0.05 across 1-second windows.
Incomplete edge context and inactive gaps keep baseline costs.

No minimum duration is shortened in this experiment. Existing duration constraints,
three mutually exclusive boundary targets, explicit no-boundary competition and
bounded penalties provide segment-consistency safeguards. They are not a separate
differentiable segment-IoU loss and do not guarantee fewer false segments.

## APIs and validation

`typed_boundary.feature_matrix(X98,Z58,probabilities,blocks,fps,contact=False)`
returns 90 features, or 98 with contact proxies. `train_typed_boundary_v7.fit`
accepts items with `X`, `y`, `blocks`, `fps` and keyword `contact`. Its output is
portable JSON. `typed_boundary.evidence(model,X98,Z58,probabilities,blocks,fps,config)`
returns `team_switch_costs`, `stop_costs`, and `restart_costs` for the root decoder.

Eleven tests cover typed targets, unknown/half barriers, overlap exclusivity,
prior-neutral costs, direction-gated discounts, bounded suppression, team symmetry,
half isolation, optional-contact schema, deterministic multiclass fitting, exact
class priors, a constant-feature optimizer returning empirical priors, excluded unknown rows and invalid inputs. Full development and
fresh-match evaluation remain required before deployment.
