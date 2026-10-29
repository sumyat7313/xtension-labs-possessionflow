# Independent review of control-model improvements

Review started on October 29, 2026 before v2 campaign outcomes. This reviewer owns
the independent integrity tests and this report, not the control implementation,
candidate training, or selection. All reviewer test observations and labels are
invented; no competition records or public labels are opened by these tests.

## Why this change is worth testing

Release 1.1 improved dead-ball recall substantially but predicted only 258/256
possession spells on its clean/degraded external final match, against 511 reference
spells. Its team-control probabilities still came from a fixed nearest-player
heuristic, while a switch cost of 4 at five effective observations per second
strongly discourages short team-control changes. This motivates learning control
evidence and testing a bounded set of transition costs.

These observations do not establish that lowering the switch cost is beneficial:
fragmenting long possessions can also reduce the official spell-F1 score. The
candidate campaign must decide using external validation matches.

## Scientific constraints

The v1 final match J03WMX is now an exposed historical benchmark. It may be used
as a v2 development/validation match, but must never again be described as an
untouched v2 test. Whole matches, rather than randomly shuffled adjacent frames,
must define the train/validation/test separation.

The required comparator is the frozen **release 1.1 learned model**, not only the
weaker original heuristic. The predeclared candidate set must be fixed before
validation outcomes. A selected winner must be frozen, including its configuration
and weight hashes, before loading final-test reference outcomes. Failed final
gates must retain the incumbent without trying runner-up candidates on the same
final data.

Recommended promotion criteria are a mean spell-F1 gain of at least 0.02 over the
incumbent and no more than 0.02 regression in **any final match × regime cell**.
Clean/degraded versions of one match are correlated views, not two independent
matches. Publish every cell and the macro-F1 and precision/recall tradeoffs, even
when a candidate passes.

The implemented v2 protocol uses J03WOH/J03WOY for training, J03WN1 plus exposed
J03WMX for validation, and all three remaining fresh matches J03WPY/J03WQQ/J03WR9
for final testing. Its 15 declared candidates comprise the incumbent, two lower
switch-cost ablations, and 12 control models combining two regularization values,
three switch costs, and uniform or capped duration-balanced training weights.
No post-selection refit occurs. The gate above is enforced across all six final
match/regime cells.

## Independent control integrity checks

The 14 tests in `tests/test_v2_integrity.py` passed on the first completed control
implementation. They verify:

- Swapping home/away membership negates every signed feature, even after track
  reordering and arbitrary team-name changes.
- Within-team track identities may permute independently on every frame without
  changing features.
- Pitch reflection and equivalent physical positions expressed using different
  pitch-normalized coordinates preserve features.
- The second half and unplayed gap cannot alter first-half features.
- Completely missing observations produce finite, neutral control features.
- Feature extraction does not mutate its observations.
- Dead, unknown, and zero-weight rows do not influence training or scaling.
- Team-swap probability symmetry and neutral 0.5 output hold despite imbalanced
  training labels; training on swapped teams preserves the fitted predictor.
- Predictions are independent of unrelated query rows and do not mutate weights.
- Repeated fits and JSON round trips are deterministic.
- Non-control targets alone cannot manufacture a model, and malformed or
  nonfinite model artifacts fail explicitly.

The implementation uses training-only uncentred RMS scaling and a zero-intercept
logistic model. Its signed feature construction and absence of a centering mean
are important: either an arbitrary fitted intercept or ordinary feature centering
would break the promised home/away probability symmetry. The model validator
rejects both forms of symmetry-breaking artifact.

The existing exact duration decoder has independent brute-force optimality tests.
No numerical decoder defect was found in this review. Equivariant probabilities
do not imply unique team labels for perfectly tied observations: deterministic
Viterbi tie-breaking can select one of two equally supported paths.

Six additional independent integration tests also passed, bringing the reviewer
suite to **20 tests**. They verify that the combined model preserves team-swapped
probabilities and half isolation; missing or malformed control features fail
instead of silently choosing the heuristic; decoder overrides are restricted and
validated; original-model emissions remain exactly compatible; and a production
run snapshots the complete nested model and effective decoder configuration and
reproduces an identical CSV from those snapshots.

The external-only duration-weighting utility was also inspected. It derives
weights solely from training H/A spells, separates spells at dead/unknown frames,
and sets non-live weights to zero. Its monotone normalization keeps final weights
within the declared [0.5, 3] caps while making their mean over live frames one.
It is not imported by inference. Short-spell recall and direct-turnover timing
diagnostics supplement the official score; they do not replace it.

Seven campaign-integrity tests bring this independent suite to **27 passing
tests** before real campaign outcomes. They check the fresh split, rejection of
cell regressions despite positive average gains, malformed/duplicate/incomplete
records, the exact declared candidate cohort, eligible-before-ranking selection,
and incumbent fallback. A mocked complete campaign asserts that the protocol is
saved before training references and that all three final calls occur only after
the winner and its hash have been saved, with exactly incumbent and winner models.
The campaign decoder also reproduces production's joint decoder on invented
observations, and changing reference labels cannot change those predictions.

Two defensive issues identified during review were corrected before the run:
the runner now verifies the incumbent's configuration hash as well as its model
hash, and selection rejects a uniformly incomplete candidate cohort rather than
accepting an apparent comparison in which every candidate omitted the same match.

## Research interpretation

The primary study [Automatic event detection in football using tracking
data](https://link.springer.com/article/10.1007/s12283-022-00381-6) distinguishes
proximity from gaining possession: a ball passing close to a player without a
trajectory or speed change need not indicate control. This supports testing
contact/temporal features, while also explaining why a nearest-player method has
an intrinsic limitation. The new control features incorporate a ball-velocity
change signal, but do not observe ball height or prove a purposeful touch.

[Using Temporal Convolutional Networks to estimate ball possession in soccer
games](https://www.sciencedirect.com/science/article/pii/S0957417423002816) studies
sequential possession classification with permutation-invariant aggregation and
both ternary and factorized live/dead plus team predictions. This supports the
architecture choices at a general level; the published study's scores are not
comparators for this challenge, and its underlying data is confidential.

The [Sportec dataset publication](https://www.nature.com/articles/s41597-025-04505-y)
describes corrected multi-camera tracking and seven matches. Those observations
differ materially from partially visible single-camera tactical tracking.
Synthetic corruption is a useful stress test, but cannot establish transfer to
all target-camera errors. Two clean/degraded test matches would be stronger
evidence than one, but still cannot support a claim about winning the hackathon.

## Release review status

Control-unit integrity review passed. Campaign selection, final-test outcomes,
production integration, exact submission reproduction and release provenance
remain separate checks; this paragraph will be updated after those checks are
available. The frozen release 1.1 submission must not be silently replaced by
research candidates that have not passed the release gate.
