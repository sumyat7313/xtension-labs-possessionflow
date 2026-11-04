# Next family: matched external D-head and joint adaptation

Status: original bounded design proposal after the [completed training-only diagnosis](CLEAN_PFF_DIAGNOSIS_RESULTS.md). Its exact split, recipe and inner guards are now implemented in the [protocol and kernel](CLEAN_PFF_ADAPTATION_PROTOCOL.md). **Not an executable frozen campaign; no real fitting or selection has started.** The authenticated execution runner, cache identities, stage bounds and winner-refit adapter remain prerequisites. Do not infer permission to change a recipe after its outcomes are available.

## Rationale and candidate budget

The full clean model reduces false stoppages and D erasure of short possessions relative to its clean base heads. The base D classifier nevertheless makes many confident false-D predictions in PFF observation regimes. Separately, the team correction and coupled decoder introduce many extra unmatched short predictions. Removing D correction or applying a missingness veto is not supported by the diagnosis.

Use exactly one matched family with at most two recipes:

| Candidate | Upstream D head | Joint residual |
| --- | --- | --- |
| A | Retain the authenticated clean external D head | Newly train on the permitted external fitting partition |
| B | Refit on that same permitted external fitting partition | Same architecture, objective and schedule as A, using B's upstream outputs |

Both keep the clean 98-feature H/A control head and the existing decoder configuration fixed. Replace the old joint residual; do not stack it as a teacher or inherit challenge-derived heads. Use authenticated observation features and recompute upstream outputs from the actual chosen clean heads. Keep all PFF-fitted artifacts local. No extra loss-weight grid, decoder search, boundary head, candidate C or post-score modification belongs in this family.

## Required freeze before execution

1. Declare a deterministic **32/8 whole-match subdivision within the original PFF training forty**, including exact IDs and the label-independent assignment rule. All forty already contributed to the component diagnosis; the inner eight are training-development data, not a fresh final test. Do not choose membership from per-match performance.
2. Use only inner-fitting PFF matches for every newly fitted scaler, D classifier, joint residual and fitted weighting rule. If the seven external Sportec fixtures are retained as rehearsal, they belong on the fitting side only, using the authenticated clean external rebuild caches. Their historical exposure must be disclosed. Parent control/head fitting history remains part of the lineage; it must not be misrepresented as newly cross-fitted.
3. Freeze identical architecture, objective, fixture weighting, seed, regularization and epoch count for A and B, apart from the declared upstream-D difference. Select these from established implementation semantics and the training-only design; do not add a tuning grid. Specify how joint training handles fitted upstream outputs, including any cross-fitting, before execution. Never train from v1.5 cached upstream probabilities.
4. Freeze one inner qualification and ranking rule. It must protect spell F1, short precision, D precision and D recall against the fixed clean comparator, with exact numerical checks and deterministic ties stated before fitting. Report predicted short-spell counts, unmatched short predictions, reference-short D erasure and all losses alongside means. This inner rule cannot replace or weaken the original development gates.
5. Freeze complete A/B inner predictions before inner reference scoring. Record both candidates and any failed, interrupted or negative attempt. If neither qualifies, stop this family and retain the failure; do not add settings until one passes.
6. Refit only the declared inner winner on all forty PFF training matches plus the same permitted Sportec rehearsal set. Authenticate its connected clean lineage and actual bytes. Freeze its entire development prediction cohort before one comparison on the already-used development twelve. No runner-up substitution after seeing that result.

The implementation and independent review must determine the exact recipe, split, source hashes, local artifact locations, stage/resource bounds and consumed-attempt handling. Until those are saved, a subsequent continuation should implement and test the protocol rather than launch an ad hoc fit. Run only one heavy fit, at most two data workers and one BLAS/Torch thread per worker.

## Unchanged acceptance and hold

The original development checks, **+0.002 absolute mean stoppage precision over v1.5**, and **at least 90% retention of the original PFF gains** remain mandatory. Neither a training diagnostic gain nor passing the inner selection rule establishes a development win.

The final twelve remain sealed until all existing numerical and substantive review conditions pass. Dataset-use rights and reference quality are separate evidence gates. PFF delivery rights, independent dense-reference validity and the historical v1.5 provenance hold remain unresolved. No automatic promotion, competition submission, provider contact or agreement acceptance is authorized by this plan.
