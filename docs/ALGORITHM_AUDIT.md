# Independent algorithm audit

This historical audit concerns the original heuristic. Its learned live/dead
recommendation was subsequently implemented, trained and externally tested; see
[the training campaign](TRAINING_CAMPAIGN.md) and [current technical report](TECHNICAL_REPORT.md).

Audit date: October 29, 2026. This review used source code, competition rules, and
invented synthetic observations. It did not inspect restricted match records or
use the public labels or scores for model selection. Production inference and its
default parameters were not changed by this audit.

## Correctness finding

No confirmed production correctness defect was found. Frame alignment follows the
documented one-based ATD convention, defensive tags reverse the tagged side,
missingness is not automatically labelled dead, and half-local decoding prevents
state leakage across halftime. These are engineering findings, not evidence of
competitive accuracy.

An additional exhaustive check compared the decoder against all 1,413 valid
eight-frame paths for each of 20 random observation matrices. Team minimum duration
was one frame and dead minimum duration three frames, with asymmetric stop/restart
costs. Every decoded path attained the brute-force minimum within 1e-10. This
extends the existing short-path test to unequal state durations.

## Reproducible observation audit

```bash
python tools/audit_observations.py --output runs/observation-audit.json
```

The tool generates eleven artificial, unlabelled scenarios, runs the existing
inference path, and records aggregate label counts and observation support. It
does not read match assets, read ground truth, train, or adjust parameters. The
fixed duplicate player positions and independently supplied speeds deliberately
isolate individual observation terms; they are not complete physical trajectories.
Neither live play nor a stoppage can be established from these inputs alone.

For a stationary ball near the home team and 22 stationary player observations:

| Changed observation | D support in middle frame | Decoded labels over 500 frames |
|---|---:|---|
| No tactical phase | 0.598688 | 500 D |
| Add an active home Build Up phase | 0.310026 | 500 H |
| No phase; reported mean player speed becomes 0.2 m/s | 0.480011 | 500 H |
| No phase; reported mean player speed becomes 0.4 m/s | 0.363547 | 500 H |

This demonstrates sensitivity, not errors against known labels. The model has
little margin for noisy low-speed measurements or an automatic phase overlapping
a stoppage. Do not choose new constants to make these cases emit a preferred label.

## Priorities supported by the implementation

1. **Validate and learn live/dead evidence on independent matches.**
   `possession/decoder.py`, `observation_probabilities`, uses fixed weights for
   stationary ball times still-player support, missingness, outside position,
   phase evidence, and motion. The stationary/still term is a product, so moderate
   uncertainty in each factor rapidly weakens it. The synthetic audit quantifies
   that effect. A missed stoppage can merge two same-team spells, directly affecting
   the ranking objective described in `EVALUATION.md`.

2. **Separate certainty about the team from certainty about live play.**
   Current observation probabilities are `(1-D)*H`, `(1-D)*(1-H)`, and `D`.
   Consequently a D value above about one-third can win when H/A is uncertain,
   but must approach one-half when the team is certain. The temporal decoder
   adds duration and transition costs. This is mathematically consistent joint
   decoding, not a software defect, but it makes team certainty affect the live/dead
   threshold. Compare it with a two-stage state model on external validation data.

3. **Model restart evidence through an entire stoppage.**
   `possession/tags.py`, `build`, selects the first stationary-to-moving candidate
   inside a set-piece interval, using a 0.6-second preceding window and 0.4-second
   following window. Ball movement during retrieval or placement can be mistaken
   for a restart; an occluded restart can fail the 60% finite-speed requirement.
   Evaluate multiple candidate boundaries with both prior stoppage context and
   sustained post-restart control, instead of accepting the first speed crossing.

4. **Use robust collective motion rather than an arithmetic mean of track slots.**
   `possession/features.py`, `extract`, averages valid player speed fields and
   scales support by visible slot count. `DATA.md` explicitly says slots can be
   duplicated and change identity. Duplicate slow or fast observations alter the
   mean and visibility support. A spatial deduplication or robust quantile model
   is a candidate, but must be tested under synthetic duplication and on separate
   games before adoption.

5. **Distinguish proximity from control.**
   Team evidence is the nearest-distance margin attenuated by ball speed and
   distance, with phase evidence and temporal costs added later. It does not
   explicitly observe consecutive controlled touches, pass reception, or relative
   ball/team motion. The challenge definition emphasizes control. Add collective
   trajectory and sustained-control features only if they improve held-out spells
   rather than simply making predictions smoother.

6. **Assess missing-ball recovery only after an independent benchmark exists.**
   A whole half with no evidence returns X; shorter low-evidence periods are filled
   through temporal inference. This is reasonable but cannot recover an unseen
   transition. Additional local video detection or aligned broadcast tracking could
   supply evidence, but neither is demonstrated necessary or beneficial yet.

## Preregistered external research design

The following is a proposed experiment, not a completed or validated improvement.
It is specified before inspecting candidate external scores.

### Dataset and split

- Use accessible, licensed matches other than the competition match. An external
  adapter must prove the semantics and alignment of live/dead status and team
  possession, and exclude missing/ambiguous reference rows explicitly.
- Never include external BallStatus, Possession, event outcome, or equivalent
  reference fields in the inference features. Keep them in the scoring/training
  target path only. Record source/version/licence and hashes.
- Split by whole match. Use nested leave-one-match-out validation if the dataset
  is small: the held-out outer match never selects parameters; inner matches do.
  Report each held-out match separately as well as the aggregate.
- Freeze the evaluation protocol, corruption procedure, candidate set, and
  selection rule before running the candidates. The competition game's public
  labels and its aggregate scores remain outside this process.

### Candidate models

- **Reference:** the frozen existing pipeline, with unavailable external tags
  disabled explicitly. Report this domain difference.
- **Candidate A:** fit a regularized binary logistic live/dead model using external
  BallStatus targets and observable geometry/motion/missingness features. Replace
  only the hand-weighted dead-logit model; retain the original team model and joint
  temporal decoder. Use training-only feature standardization and inner-match
  validation for regularization and class weighting. No external label field is
  an input feature.
- **Candidate B:** use the same learned binary live/dead observations with a binary
  temporal decoder. Decode team ownership within each predicted live interval,
  resetting ownership after predicted stoppages. This tests the team/dead coupling
  independently from the learned observation model.

External data may contain corrected, almost complete tracking while competition
inputs are noisy automatic tracks. Apply missingness augmentation **only to inputs**:
whole-ball gaps, isolated detections, partial team visibility, duplicated slots,
and speed noise. Use fixed random seeds and structured contiguous gaps, rather
than only independent frame dropout. Derive corruption ranges from published
input characteristics or training-match observations, never competition labels.
Retain both clean and corrupted held-out results to expose dependence on complete
tracking. Reference target labels are never derived from the corrupted input.

### Decision rule

Use the official spellF1 definition as the primary external outcome, macroF1 as
the first tie-break, and report live/dead confusion, transition behavior, and
runtime as diagnostics. Compare paired per-match differences with uncertainty;
do not present thousands of correlated frames as thousands of independent samples.
Retain the frozen model if the external experiment does not support the new one.
If a candidate is selected, fit it only on the permissible external training set,
freeze its weights/configuration, and regenerate the competition CSV with full
provenance and a clean-source byte-identical reproduction check. Do not choose
between candidates by rescoring the competition public window.

## What this does not establish

Passing engineering checks establishes reproducibility and certain invariants.
It does not establish that all important features are present, that the parameters
generalize, or that the entry can rank first. No hidden-match score, competitor
comparison, or winning claim is available from this audit.
