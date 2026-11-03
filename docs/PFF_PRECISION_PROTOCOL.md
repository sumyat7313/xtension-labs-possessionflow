# PFF false-stoppage refinement protocol

Prepared 2026-11-03 before scoring the refinement grid. The parent experiment improved possession spells and stoppage boundaries but failed the unchanged v1.5 replacement gate because mean stoppage precision fell. This study asks whether a small reduction in the parent's predicted dead-play probability can remove that regression while preserving its gains. It does not reopen the original experiment or retroactively change its failed result.

The frozen parent artifact is `0ab133743c593466d8ae92ea94937e8c6bf22510f6b64053e3e69c7b467dcf3f`. Production remains the original v1.5 model. The exact whole-fixture reservation in [pff-protocol.json](../models/pff-protocol.json) still applies: 40 training, 12 development and 12 locked final matches. **The development fixtures have already been inspected and are now tuning data.** No new development result may be described as a fresh confirmation. Only the final 12 remain an untouched confirmation cohort, until and unless the access gate permits their one-time use.

## Bounded correction

The only varying parameter is a nonnegative dead-logit penalty, in this exact order:

`0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25`.

The correction preserves the conditional home-versus-away probability and the original decoder. The zero-penalty control must reproduce the parent's predictions and per-match metric records. Any lossless probability cache must first demonstrate that identity. Reference labels cannot become inference features. Missing ball observations alone do not establish dead play.

Each small correction artifact is saved and hashed before development scoring. The immutable study protocol binds the ordered candidate IDs, artifact hashes, exact parameter grid, scientific-source hashes, original reservation and previously recorded parent development metrics. All eight declared candidates must have complete paired records on all twelve development fixtures before selection. Missing, extra, duplicated or changed candidates cannot be silently excluded. If a grid candidate fails execution, resolve and disclose the implementation failure rather than selecting only the candidates that completed.

## Development gate and selection

Every candidate must pass **all** original v1.5 development checks, including minimum spell gain, nonregression in the target metrics, worst-fixture spell and dead-recall limits, sufficient reference support and the original paired-bootstrap diagnostic. Baseline metrics and reference counts must match the parent comparison fixture by fixture. In addition:

- Mean stoppage frame precision must exceed v1.5 by at least **0.002**, or **0.2 percentage points**.
- For each of spell F1, macro F1, short-possession precision, short-possession recall, stoppage segment F1, stoppage boundary F1 and stoppage recall, preserve at least **90% of the parent's positive mean gain over v1.5**. A nonpositive parent gain sets a zero floor rather than permitting regression.

Among candidates passing every check, select the highest mean spell-F1 gain over v1.5. Break a tie by higher mean stoppage-boundary F1 gain, then earlier declared grid position. This favors the smaller correction only after the performance tie. If none passes, select nothing and leave the final fixtures unopened. No threshold extension or alternative objective may be introduced after seeing this grid while retaining the same protocol claim.

The fixture-level bootstrap is retained for consistency with the original gate; after development tuning it is a selection diagnostic, **not an unbiased confirmation interval or a correction for searching multiple candidates**. The unchanged final cohort is needed for that subsequent confirmation. Recurring teams and a single tournament still limit generalization claims.

## Locked final audit

Before preparing or reading final fixture records, `require_final_candidate` recomputes selection from the complete development records and verifies the exact selected artifact. The runner must also authenticate the current source files, artifact bytes, saved development diagnostics and immutable protocol against their hashes; the aggregate gate alone cannot certify those bytes.

Freeze predictions for v1.5, the original PFF parent and the single selected refinement on **all twelve final fixtures before opening the first final reference**. All models use identical observations, reference masks and metric definitions. An exclusive reference-access record must be written before reading labels, including attempts that fail partway through scoring. No second winner or further tuning may use this consumed cohort as an untouched final test.

Final numerical checks compare the selected refinement with v1.5 using all original requirements, the same stoppage-precision margin, and the same 90% retention rule relative to the parent's gains on that final cohort. The original PFF parent is a frozen comparison artifact, not another candidate eligible for selection on final results.

Even a numerical pass is insufficient for deployment. The existing evidence requirements still include independently substantiated dense reference quality, observation independence, identity, permitted use, adequate support and prediction-before-reference separation. **Current PFF references are provider-event-derived silver labels, and fitted-model distribution rights remain unverified.** Neither condition is repaired by a favorable metric. The validation helpers do not write or replace production artifacts; retain v1.5 whenever any numerical or evidence condition remains insufficient.

## Implementation and tests

`tools/pff_precision_validation.py` provides immutable protocol construction, exact-grid selection, a recomputed final-access guard and final paired gates. It reads aggregate records only, never provider files. The runner remains responsible for authenticating the evidence and enforcing one-time reference access.

Fifteen focused tests cover precision margin, retention of every target gain, original worst-fixture safeguards, paired baseline/support, wrong parent or partition, grid changes and hidden parameters, immutable protocols, incomplete/foreign candidates, zero-control identity, deterministic ranking, a failed grid keeping final locked, exact winner binding and every final evidence flag. Synthetic test scores do not constitute actual model performance.

`tools/verify_pff_precision.py --run runs/pff-precision-v1 --write models/pff-precision-research.json` authenticates the local correction artifacts, frozen parent, cache arrays, cohort predictions and scored CSV bytes, then writes a verified aggregate evidence bundle. The bundle excludes fitted PFF weights, raw provider records and frame-label arrays. Its correction parameter declarations are original research configuration, not provider data. The default verifier independently derives metric records from the saved diagnostics, recomputes complete-grid selection and any final gate, and validates the parent evidence, source hashes, support, prediction/access bindings, unchanged deployment and explicit reference limitations. It rejects partial or unreported final activity.

Fifteen additional verifier tests cover those provenance and byte-integrity checks, tampering with model or metric identities, false permission/review claims, hidden candidates or final activity, zero-control identity, failed development blocking final, and a synthetic numerical success that still cannot authorize promotion without the required evidence. The portable verifier authenticates aggregate consistency; it does not rerun inference or recreate human annotations.
