# Independent review of the learned-stoppage campaign

This review checks model selection and reproducibility, not a hidden competition
ranking. It was started on October 29, 2026 before candidate training outcomes.
The reviewer owns independent tests and this report, not the production model or
training implementation.

## Rules and scope

The competition permits accessible labelled data from other matches. It prohibits
human labels on the provided match, including use for training or validation, and
limits its public five-minute labels to pipeline correctness checks. This campaign
must not open or score those public labels. Restricted competition observations
remain local; only code and aggregate diagnostics are reviewed.

The Sportec adapter previously passed independent synthetic tests of units, role
mapping, missing-frame preservation, and label isolation. Changing BallStatus and
BallPossession, including boundary rows, leaves observations, half blocks, and
actual predictions unchanged.

## Predeclared experimental separation

- Training matches: J03WOH and J03WOY.
- Validation match: J03WN1. Its baseline result was previously inspected, so it is
  explicitly a development/selection match and not an untouched test.
- Final untouched test: J03WMX. No candidate is fitted or selected with it.
- Candidates: baseline plus logistic live/dead models with regularization
  0.001, 0.01, and 0.1, each using joint or separate temporal decoding.
- Score clean and deterministically degraded observations separately. Corruption
  uses only observation arrays, fixed match-specific seeds and published protocol
  settings, never reference labels or event outcomes.
- Apply the validation gain gate to each learned candidate first, then select
  eligible candidates by mean validation spellF1 across the two regimes, then
  mean macroF1. The baseline is always an eligible fallback.
  The final declared gain gate requires mean spellF1 improvement of at least 0.02
  and no regime more than 0.02 below the baseline. This replaced an initial 0.01
  proposal before any fit or candidate outcome; the immutable campaign protocol
  is the authority for the final settings.
- Evaluate the frozen choice on the final match with the same declared gate. A
  failure retains the original baseline. Do not test alternatives on that match
  or retune the failed candidate using its results.

## Review gates

The completed review must verify: training-only scaling and optimization; unknown
and zero-weight row exclusion; finite model artifacts; batch-independent scoring;
label-free features and corruption; half-local state; stable serialization; exact
match/input/configuration/weight provenance; and untouched-final selection order.

Final campaign outcomes are reviewed below once the frozen execution completes.
The implementation checks establish integrity, not a performance gain.

## Pre-outcome issue corrected

Independent review found that initially ranking candidates and gating only the
top result could discard an otherwise qualifying improvement: a fragile candidate
could have the best average yet fail the degraded regime, causing a fallback to
baseline despite a robust runner-up. Before any training outcomes, selection was
changed to gate each validation candidate and choose the strongest eligible one.
The final-test rule still evaluates only the frozen winner; it does not search
through alternatives after seeing final results. A regression test preserves this
selection behavior.

## Independent implementation tests

All **17** tests in `tests/test_training_integrity.py` pass using invented data:

- Unknown-reference and zero-weight rows cannot change the learned scaler or fit.
- Prediction does not refit normalization on its input batch or mutate the artifact.
- Saving and reloading produces identical predictions and stable JSON bytes;
  repeated fitting is deterministic in the pinned local environment.
- An unknown-only or one-class training set is rejected.
- Learned features are invariant to team/slot renaming, cannot mix halves, and
  stay finite with missing observations.
- In separate decoding, changes to team preference cannot change the dead path.
- Train, validation and final match IDs are disjoint and exclude the competition.
- Corruption is seeded, does not modify original observations, preserves existing
  missingness and leaves frames outside the half blocks unchanged.
- The 0.02 gates reject insufficient average gains and severe regime regressions.
- Selection rejects incomplete, duplicate and nonfinite scores, and chooses a
  robust eligible runner-up over an ineligible highest-average candidate.
- Paired bootstrap differences are zero for identical predictions and negate
  correctly when baseline and candidate are swapped.
- A complete synthetic campaign executes real fitting and orchestration while
  checking that final data cannot be read until the validation selection and
  selected-weight checksum have been written. Only the two training matches
  appear in the fit provenance.

Source review confirms that only `fit_model` accepts targets. It excludes X and
zero-weight rows before weighted standardization and ridge-logistic optimization.
The inference feature function accepts tracking geometry, speeds, visibility,
pitch dimensions and half blocks, with no reference or tag argument. Model loads
validate schema, feature names, coefficient dimensions, finite values and positive
scales. The campaign calls `read_external` on final-test data only after writing
the frozen validation selection and selected-model checksum. Its source hashes are
checked again before release.

The production pipeline snapshots the exact JSON weights used by a run and the bundle
checks that snapshot against its manifest. A learned package includes those exact
bytes as `models/submission.json`. Production reproduction and package checks are
separate release gates after selecting a model.

## Interpretation limits

One untouched final match is a useful guard against selection on validation, but
not broad statistical validation. Clean and synthetically degraded tracking are
two observation regimes of the same match; they are not independent matches.
The possession bootstrap is diagnostic and conditional on that match, not a
confidence interval for hackathon rank. A passed gate supports promotion among
the declared candidates under this protocol; it does not prove the model is best
among all possible methods or that it transfers perfectly to competition data.

## Campaign outcome

**Review passed for promotion of `l2-0.001_joint` under the frozen protocol.**

The independent reviewer recomputed selection from all fourteen validation
records (six learned candidates plus baseline, each in two regimes), verified
that every record belongs to J03WN1, and obtained the recorded winner. Its
validation mean spellF1 gain is 0.056963; the clean and degraded gains are 0.049499
and 0.064427 respectively.

The final summary contains exactly four records: baseline and that frozen winner,
in two regimes, all on J03WMX. Recomputing the gate agrees with its recorded pass:

| Final regime | Baseline spellF1 | Learned spellF1 | Gain | Baseline D recall | Learned D recall | Learned D precision |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.3557 | 0.3938 | +0.0381 | 0.5520 | 0.9663 | 0.9164 |
| Degraded | 0.1877 | 0.2581 | +0.0704 | 0.3723 | 0.9535 | 0.8591 |

The mean gain is 0.054264, above the required 0.02, and neither regime regresses.
Paired possession-bootstrap 5th–95th percentile gain intervals are 0.0177–0.0610
for clean inputs and 0.0465–0.0944 for degraded inputs. These describe uncertainty
within this match, not performance across all matches.

The improved recall has a precision cost: baseline D precision was 0.9882 clean
and 0.9903 degraded. Missed stoppages are substantially reduced on this test;
they are not eliminated or proven fixed on the competition match. Predicted
possession counts remain 258 clean and 256 degraded against 511 reference spells,
so controlled-team transitions and short possessions remain important weaknesses.

Verified provenance includes the immutable protocol hash, frozen selection hash,
unchanged scientific source hashes, selected artifact checksum, release artifact
checksum, training-match membership, and each final prediction CSV hash. No
runner-up was evaluated on the final match, no final-set retuning occurred, and
no competition public labels were used in this campaign. The selected fit used
111,902 sampled training observations across clean and degraded inputs, with
52,614 dead and 59,288 live examples. Its optimizer converged in nine iterations.

Selected and released weight SHA-256:
`3759734a29a07eafc11723029800d30863d00c0a9e478bea3fcafaf6805c2867`.

Rebuilding and byte-identically reproducing the complete competition submission
with these weights is a separate final release check, recorded in
`docs/BUILD_STATUS.md`. External gains alone do not establish that file's hidden
accuracy.
