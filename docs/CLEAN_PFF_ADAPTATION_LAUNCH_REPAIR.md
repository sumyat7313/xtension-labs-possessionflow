# Adaptation launch failure and versioned repair

The first real adaptation launch failed before training. Local preflight authenticated the clean ancestry and checked 47 observation caches: 32 fitting PFF fixtures, seven clean Sportec fixtures and the eight inner-validation fixtures. Training references were hashed without interpretation; inner-validation references were not opened. The complete 64-source execution manifest was frozen, but the worker rejected its parent-process identity before claiming the initialization stage.

The Windows virtual-environment executable launches an intermediate process. A separate synthetic probe reproduced the difference between the process owned by `Popen` and the actual Python worker. The original worker expected its immediate parent to be the operator; on this installation, its immediate parent is the virtual-environment launcher. Earlier mocked launch-success tests did not exercise this process chain.

The failed operator-v1 launch remains consumed. Its request, failure receipt, console log, manifest and source bytes are retained. The inner lifecycle has zero events and no stage attempt; no fitting, predictions or scoring occurred. No candidate accuracy result follows from either preflight or this failure.

## Repair scope

The repair has a separate operator identity and preserves the original scientific protocol. It must authenticate the parent-owned launch process and the actual worker relationship while retaining the held lock, exact request and nonce, source/runtime checks, single-use claim and independent artifact verification. Simply dropping the parent check or switching to a different Python environment is insufficient.

Adoption is limited to the exact unchanged, empty inner lifecycle associated with the authenticated pre-claim failure. A partial inner attempt, completed stage, extra launch or different failure cannot be treated as this repair. The new operator must retain the old failure as linked evidence; it does not restore an experiment attempt or expand the two-candidate A/B budget.

Real subprocess integration tests must cover the Windows virtual-environment path, as well as invalid ownership and failed launches. Synthetic process tests contain no provider data. The scientific sources, training recipes, held-out barriers, and all precision, recall and gain-retention gates remain unchanged.

Windows process checks use the operating system's parent-process field and queried executable identity. Microsoft documents the parent field in [PROCESSENTRY32W](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32w) and the separate query, synchronization and termination permissions in [Process Security and Access Rights](https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights). Cleanup must also cover the actual worker: terminating only a launcher handle is insufficient if tree termination fails.

The implemented repair starts the Windows launcher suspended, assigns it to a verified JobObject that terminates descendants when its handle closes, then resumes its initial thread. If containment cannot be established, the launcher is stopped before execution. The worker validates an exclusive launch receipt against the live operator, owned process, executable bytes, environment and process creation identities. The original nonce, lock and claim checks remain in force.

## Verified engineering result

The full local suite ran 975 tests in 202.476 seconds: 974 passed and one host-dependent symlink test was skipped. The 46 new tests include 19 launcher checks, 19 operator checks and eight independent integration checks. Actual subprocess tests exercise Windows virtual-environment admission, the sealed initialization artifact, rejected nonces, altered artifacts and receipts, and worker termination both before and after its claim. These tests use synthetic inputs only. [Exact verification and source hashes](clean-pff-adaptation-launch-repair-verification.json).

All 64 sources bound by the failed v1 manifest remain unchanged. The prospective v2 plan adds two modules and links the retained failure; its identity is `5cdceeab6b0b9a600bdcaf1f9a2884c99c93038a498cb6088b8bc3cf6eb2b5e1`. The existing empty inner lifecycle is adopted under predecessor, repair and inner locks; it is not recreated. The first new launch chains to the old failure receipt. Any subsequent failure remains consumed.

The actual v2 operator has not been initialized or run. After reviewing this plan against the local runtime and retained evidence, the next bounded advance is `python -m tools.clean_pff_adaptation_operator_v2 initialize`, followed by exactly one `run` for the initialization stage and artifact verification. Stop at `fit-A`. No retry of the failed v1 launch is permitted. The worker timeout is bounded; parent-side artifact verification still has no hard whole-process deadline.

## Research and delivery limits

This is execution engineering. A repaired launch does not establish model improvement, label validity or delivery rights. The final twelve fixtures remain untouched, and v1.5 remains unchanged on submission hold. PFF-fitted artifacts stay local. Any later refit still requires connected lineage and independent development-admission review before further evaluation.
