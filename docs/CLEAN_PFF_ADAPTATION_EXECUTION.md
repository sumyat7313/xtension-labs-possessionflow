# Adaptation input authentication and attempt lifecycle

These components extend the [reviewed A/B specification](CLEAN_PFF_ADAPTATION_PROTOCOL.md). They do not change its recipes, split, numerical gates or frozen scientific sources. No real adaptation fit or campaign has been initialized. The complete dispatcher and winner-refit adapter remain required before execution.

## Lazy fitting inputs

`tools/clean_pff_adaptation_inputs.py` exposes `make_inner_factory(fixture_ids=..., protocol_sha256=...)`. It rejects any roster other than the exact ordered PFF inner32 plus seven clean Sportec fixtures, and rejects a different scientific-core digest, before filesystem access. There is no public caller-supplied cache manifest, data root, scoring loader or refit route.

Factory construction authenticates the published protocol and aggregate provenance, clean learned ancestry and actual model bytes. It does not materialize fixture observations or labels. A subsequent request checks the fixture whitelist before source or data access, verifies producer sources, and then authenticates the requested records:

- PFF uses only native observation features `x` and `z`, observation metadata and alignment. Their manifest must match the pinned published manifest. Feature bytes are verified before opening that fitting fixture's hashed event file; references are reconstructed with the existing fixed adapter. Historical upstream probabilities and predictions are never loaded.
- Sportec uses the clean rebuild's pinned preparation-stage seals, source identities, clean features and references. Camera/degraded arrays and old augmentation caches are not loaded.
- NumPy files are hashed through the opened descriptor before and after materialization with pickle disabled. JSON is parsed from the same bytes that were authenticated. Paths reject links, junctions and traversal. Metadata, period barriers, dimensions, finite features and reference classes are checked.

The factory returns data to the local kernel only. Returned records, labels and model weights must remain local. Provenance reports keep PFF delivery rights and independent dense-reference quality unresolved. Authentication proves identity and checked provenance; it does not grant dataset rights or permission to execute a training stage.

## Ordered, consumed attempts

`tools/clean_pff_adaptation_stages.py` supplies an inner-only lifecycle store with 29 fixed stages: initialize, fit A, fit B, all eight A predictions, all eight B predictions, freeze, eight paired scoring stages, then select. It has no refit, development or final stage and no CLI, scheduler or model-training dispatcher.

Its descriptor checks the preserved 24-file scientific snapshot, adds the current static import closure of both new modules, and records runtime package versions. The study path is restricted to `runs/clean-pff-adaptation-inner-v1`. Initialization cannot reset an existing study.

Each attempt holds a kernel lock throughout its context. An exclusive attempt marker is written before the start event. Hash-chained ledger events preserve stage order and original descriptor identity. Exceptions, missing completion receipts, started-only interruption and a crash between marker creation and ledger append consume the attempt. Source or manifest drift prevents completion. A later call cannot silently retry a consumed failure.

Completion receipts contain opaque artifact identities only. The store does not independently read those artifacts or certify their scientific meaning. `recorded_complete` means that bookkeeping completed, not that training, prediction or scoring was verified. Caller data access is explicitly not attested; the store itself accesses no provider records. All campaign, refit, development, final and promotion authorization flags remain false.

The descriptor declares one heavy fit, at most two data workers, one BLAS/Torch thread per worker, 1,800 seconds per stage, one stage per advance and 2,700 seconds per wakeup. These are **declared limits, not enforced subprocess limits** in this component. A future dispatcher must enforce them before any real job runs.

## Remaining execution work

The next bounded implementation must provide and independently test:

1. A fixed stage dispatcher that authenticates the input bridge and consumed attempt, enforces resource/time limits, writes local artifacts exclusively and verifies their actual bytes.
2. Complete A/B inner prediction cohorts frozen before inner scoring-reference access, with independently verified model, prediction and score bindings. Lifecycle receipts alone are insufficient.
3. Selection connected to the frozen scientific selector, including both candidates and failed outcomes, followed by at most one verified winner-refit adapter. The current kernel accepts inner fitting only.
4. A complete reviewed source/cache/stage execution manifest before fitting. No extra candidate, hidden retry, changed split or relaxed gate belongs in this family.

Subsequent development still requires all original checks, a +0.002 absolute stoppage-precision gain and at least 90% retention of the original PFF gains. v1.5 remains unchanged on submission hold; final12 remain untouched. Rights and label-quality reviews remain separate requirements.

## Verification limits

Tests use synthetic temporary files and model stubs, plus source and published aggregate identities. They exercise poisoned caches, forbidden fixtures, source drift, byte mutation during loading, lifecycle interruption, locking and tampered history. No real factory construction, provider-record access or adaptation fit was performed for this engineering milestone. The dated verification report records actual test results and skips; passing tests is not evidence of improved model accuracy.
