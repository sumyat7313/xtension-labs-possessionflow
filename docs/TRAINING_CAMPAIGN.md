# Learned stoppage campaign

Protocol recorded October 29, 2026, before any learned-candidate scores. This
experiment uses only accessible other-match labels. The competition match and
its public correctness labels never enter training, model selection, or testing.

## Sources and fixed split

The [IDSSE/Sportec dataset](https://doi.org/10.6084/m9.figshare.28196177), published
by Bassek, Rein, Weber and Memmert (2025), is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its ball status and
possession references describe referee interruptions and the controlling team.
See [the independent benchmark](EXTERNAL_BENCHMARK.md) for provider conventions.

| Role | Matches | Treatment |
|---|---|---|
| Training | J03WOH, J03WOY | Fit scaling and model weights here only |
| Validation | J03WN1 | Select among the declared candidates; its baseline score was already inspected |
| Final test | J03WMX | Untouched until the selected candidate and model checksum are frozen |

Raw XML stays under ignored `data/external/idsse/`. Public Figshare file IDs,
byte sizes and published MD5 checksums are pinned in `tools/train_stoppages.py`.
SHA-256 input provenance is recorded in each campaign. BallStatus and
BallPossession are targets only. They do not determine observation visibility,
feature extraction, temporal blocks, corruption, or state initialization.

## Candidate set and training

The baseline is eligible to win. Six learned candidates combine logistic L2
regularization **0.001, 0.01, 0.1** with **joint** or **separate** decoding.
The first keeps the three-state temporal decoder; the second decodes live/dead
first and team ownership within predicted live intervals. Both use the same
label-free feature implementation and retain the existing team evidence.
Automatic tactical tags are unavailable on these external matches. No learned
dead-ball feature contains a tactical tag or target label.

Training uses each training match twice: its clean inputs and deterministic
degraded inputs. Every fifth frame is sampled, giving 5 Hz training observations.
Unknown targets are excluded before computing scaling or fitting. The final
selected weights are not refit on validation or final-test data.

## Degraded-input regime

Corruption depends only on input shapes, half metadata, pitch size, and the fixed
seed. Reference labels remain unchanged and do not choose corruption locations.

- Alternate visible ball windows of 2–8 seconds with missing windows of 0.4–6 seconds.
- Remove independently selected player slots in 12-second blocks with probability 0.15.
- Add position noise with 0.15 metre Gaussian standard deviation.
- Add speed noise with 0.3 m/s Gaussian standard deviation, clipped nonnegative.
- Use seeds 41021 (J03WOH), 41039 (J03WOY), 51021 (J03WN1), 61021 (J03WMX).

These are declared stress conditions, not an assertion that the competition's
noise follows that exact distribution. Clean and degraded results stay separate.

## Selection and release gate

First filter learned candidates by the eligibility gate: mean spellF1 must exceed
baseline by at least **0.02**, with neither regime regressing by more than **0.02**.
Baseline is always eligible. Rank the eligible candidates by mean **spellF1**
across clean and degraded validation inputs, then mean macroF1, then candidate
identifier. This eligibility-first policy was finalized before fitting or scores;
a brittle highest-mean candidate does not discard a robust runner-up.

Freeze the winner and checksum before reading final-match labels. On the final
match compare only that winner against baseline, in both regimes, and apply the
same fixed gate. Failure retains baseline; do not test runners-up or adjust
parameters after seeing final results. Record paired possession-bootstrap
diagnostics and per-regime metrics, but do not change the gate from those results.

This finite campaign identifies the strongest eligible version among the declared
candidates. Four matches and synthetic corruptions do not establish a hackathon
rank or guarantee transfer from multi-camera tracking to noisy tactical video.

## Reproduce

From the repository root, using the pinned Python environment:

```bash
python tools/train_stoppages.py --download --out runs/stoppage-campaign-v1
```

Use a fresh output directory for each campaign. An existing protocol is never
overwritten. Downloading all four matches requires approximately 1.52 GB; two
matches were already downloaded during the baseline audit.

Outputs include the immutable protocol, six trained JSON model artifacts, full
validation predictions/metrics, the selection frozen before final scoring, final
predictions/metrics, source/input hashes, paired bootstrap diagnostics, and a
summary. `models/selection.json` records whether a learned model passed the gate.
Only a passing learned model is exported to `models/stoppages.json` for the
production submission pipeline. Neither raw external data nor reference CSVs
belong in the submission source archive.

## Annotation provenance and amount

The [dataset paper](https://www.nature.com/articles/s41597-025-04505-y) identifies
DFL as the original data provider. Positions came from TRACAB Gen5 optical
tracking with filtering and smoothing; live operators checked plausibility and
performed final quality control. The BallStatus and BallPossession fields are
provider-supplied references in those files. The paper defines their meanings,
but does not separately specify whether each status/possession value was manually
annotated or algorithmically derived. We do not invent that provenance. The
separately human-annotated event feed was not used. No new hand labels were created.

The two training matches contain 279,750 known reference frames at 25 Hz.
Deterministic 5 Hz sampling gives 55,951 unique time samples. Clean and corrupted
input copies yield 111,902 fitted rows: 52,614 dead and 59,288 live. They reuse
provider targets; augmentation does not create labels. Validation and final
references were used only for their declared evaluation roles, never fitting.

## Results

The fixed campaign completed successfully. The strongest eligible validation
candidate was **L2 0.001 with joint decoding**. Its artifact checksum was frozen
before opening the final match. It also passed the unchanged final release gate;
no runner-up was tested on the final match and no parameters were retuned.

### Validation J03WN1

| Candidate | Clean spellF1 | Degraded spellF1 | Mean spellF1 |
|---|---:|---:|---:|
| baseline | 0.3150 | 0.1610 | 0.2380 |
| l2-0.001_joint | 0.3645 | 0.2254 | 0.2950 |
| l2-0.001_separate | 0.3540 | 0.2232 | 0.2886 |
| l2-0.01_joint | 0.3590 | 0.2210 | 0.2900 |
| l2-0.01_separate | 0.3637 | 0.2197 | 0.2917 |
| l2-0.1_joint | 0.3547 | 0.2189 | 0.2868 |
| l2-0.1_separate | 0.3539 | 0.2224 | 0.2882 |

### Untouched final J03WMX

| Regime | Version | spellF1 | macroF1 | Dead precision | Dead recall |
|---|---|---:|---:|---:|---:|
| clean | baseline | 0.3557 | 0.7199 | 0.9882 | 0.5520 |
| clean | l2-0.001_joint | 0.3938 | 0.8550 | 0.9164 | 0.9663 |
| degraded | baseline | 0.1877 | 0.5969 | 0.9903 | 0.3723 |
| degraded | l2-0.001_joint | 0.2581 | 0.7772 | 0.8591 | 0.9535 |

Final mean spellF1 gain: **0.0543**. Clean gain: **0.0381**; degraded gain: **0.0704**. Both regimes improved.

The missed-stoppage problem is substantially reduced on this independent match,
with a tradeoff: dead-ball precision declines as recall rises. Team-transition
and spell-count errors remain; this is not a perfect possession system.

Paired bootstrap diagnostics resample the same reference possessions for both
versions (1,000 draws, seed 62171), using the official scorer's fixed-prediction-
count bootstrap approximation. The 5th–95th percentile intervals for the spellF1
improvement are:

- clean: 0.0177 to 0.0610.
- degraded: 0.0465 to 0.0944.

These are conditional within-match diagnostics, not match-population confidence
intervals or a guarantee of improvement on the competition video. Cleaner
multi-camera tracking, provider semantics, and unavailable automatic tags remain
transfer limitations. Three dataset matches remain unused.

### Released artifacts and audit trail

- Selected model: `models/stoppages.json`.
- Shareable split, protocol, source/input provenance, and selection evidence:
  `models/selection.json`.
- Full immutable experiment: `runs/stoppage-campaign-v1/` (local, gitignored).
- Validation selection frozen before final: `selection-frozen-before-final.json`
  inside that run.

Selected model SHA-256: `3759734a29a07eafc11723029800d30863d00c0a9e478bea3fcafaf6805c2867`.

Protocol SHA-256: `a68503fb926e439275a365dc0d39c904509c5fecbcc65ac7f11f81a7df869731`.
