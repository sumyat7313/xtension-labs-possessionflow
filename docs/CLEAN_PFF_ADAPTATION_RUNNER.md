# Fixed adaptation campaign and winner refit

The new campaign, refit and operator modules connect the [input and lifecycle foundations](CLEAN_PFF_ADAPTATION_EXECUTION.md). The original A/B recipes, 32/8 partition, clean parent and numerical gates are unchanged. This is an engineering implementation; no real campaign initialization, provider-record access or adaptation fitting occurred during its implementation and synthetic testing.

## Inner campaign

`tools/clean_pff_adaptation_campaign.py` owns each lifecycle attempt itself. Its only execution stages are the original 29: initialize, fit A/B, all 16 inner predictions, freeze, eight paired scoring stages and select. It writes exclusive local artifacts, seals actual bytes, and binds each seal to the lifecycle ledger. A failed, partial or interrupted stage cannot be silently retried.

Fitting uses the authenticated inner39 factory and frozen kernel. Inference loads only the inner-eight native observation features and recomputes the selected clean upstream probabilities. Both complete prediction cohorts must be frozen before any inner scoring reference is reconstructed. Scoring retains all eight metrics, short-prediction precision/counts, short-reference stoppage interruption/erasure counts and paired reference support.

`verify_stage` authenticates the actual artifact roster and bytes against its completed lifecycle receipt. `verify_completed` additionally reconstructs authenticated references, recomputes metrics and the exact numerical selector, and checks the saved score/selection artifacts. It returns a verified numerical result with release and final-access flags false. These checks establish internal consistency and enforced execution order, not independently witnessed chronology or a new source of dense ground truth.

## One selected-winner refit

`tools/clean_pff_adaptation_refit.py` obtains the winner from actual completed-inner verification; callers cannot supply a candidate or alternate output directory. A family with no qualifying candidate cannot start refitting. The single refit has its own lock and exclusive consumed-attempt marker. No runner-up or retry path exists.

Only the original PFF40 plus seven clean Sportec fixtures may enter the refit. The new sampling helper reproduces the frozen inner kernel's mathematics on this larger approved roster without changing or monkeypatching its held-out barrier. The recipe, seed, epochs, fixture masses, weighting, model architecture and A/B distinction remain identical. The old joint residual is discarded; the clean H/A head and decoder remain fixed.

The verifier binds actual output bytes to the selected inner result, source closure, runtime versions, model components and exact training settings. All fitted artifacts remain local. A future development-admission step still needs a connected lineage artifact for this newly fitted model; the refit report alone does not authorize development, final evaluation or submission.

## Bounded subprocess operator

`tools/clean_pff_adaptation_operator.py` supplies `plan`, `initialize`, `status` and `run`. Paths, worker entrypoints and stage order are fixed. `run` launches at most one stage. A private worker needs the exact current parent-bound nonce, a held parent lock and a single-use claim. The parent reserves the launch before creating the process; a failed launch or interrupted worker consumes it even if the inner lifecycle did not start.

The outer manifest binds all three new modules and their complete static dependency closure, the preserved scientific snapshot, runtime package versions and the inner descriptor. Source identities are checked again after work. A zero exit code is insufficient: the parent independently verifies actual artifacts and compares the worker receipt before recording success.

The operator runs one worker and one heavy fit at a time. BLAS/OpenMP and Torch intra/inter-op threads are set to one. Each worker has a 1,800-second subprocess timeout. Termination attempts to stop the process tree and always falls back to killing and reaping the owned child handle; a synthetic Windows test exposed the need for this fallback. Cleanup waits are bounded separately. Parent-side source and artifact verification are synchronous and do not have a hard deadline: the 2,700-second wakeup value remains a scheduling target, not a whole-process timeout guarantee.

Before real execution, independently review and freeze the complete operator plan, check existing locks/jobs, authenticate local cache/ancestry compatibility and retain the source snapshot. Do not reinterpret passing synthetic tests as evidence that every real local artifact has already passed preflight. Keep all real outputs, worker logs and fitted models under ignored local run directories.

The final twelve and development twelve have no route in this operator. v1.5 remains unchanged on submission hold. The original development checks, +0.002 absolute stoppage-precision margin, 90% gain-retention requirement, and separate source-rights/label-quality reviews remain mandatory.
