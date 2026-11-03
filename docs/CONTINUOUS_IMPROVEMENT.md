# Continuing PossessionFlow development

The objective is a reproducible, rule-compliant improvement over the historical v1.5 comparator. It is not an unlimited threshold search or a claim that a globally best model can be proven. The organiser clarification in [the rules review](INITIAL_CHALLENGE_RULES_REVIEW.md) controls every continuation.

The Codex heartbeat **Improve PossessionFlow with external-only experiments** resumes this chat hourly. Its identifier is `improve-possessionflow-with-external-only-experiments`. Local continuation depends on the computer and Codex being available; scheduling is not a promise of uninterrupted computation. The parent agent coordinates the workers, checks the durable state, and advances only work that is both authorized and unblocked. The user can request that it pause or stop.

## Immediate priority

Preserve the original v1.5 model and submission bytes as a historical comparator, with a submission hold. Rebuild the affected learned components from external data, without the target-derived camera profile, its augmentation caches, or v1.4/v1.5 learned descendants. The earlier PFF context corrections inherit that dependency and cannot remove it merely by improving their scores.

The first reconstruction uses Sportec/DFL data and synthetic camera corruption. The [dataset's current Figshare metadata](https://api.figshare.com/v2/articles/28196177) identifies CC BY 4.0; preserve attribution and provenance. Its [publication record](https://doi.org/10.6084/m9.figshare.28196177) supports dataset identity. Licensing and label validity are separate checks. A declared licence does not establish dense-reference quality or a fresh match split.

The exact v1.2 external dead head is the only permitted parentless learned ancestor in the new lineage screen. This exception is hash-specific and grounded in the prior WOH/WOY training audit; it does not grant permission to reuse arbitrary historical models. Other learned ancestors must have explicit connected parents. See [the rebuild implementation notes](EXTERNAL_REBUILD.md).

## Operating loop

1. Read the current harness status, append-only trial history, rule review, and latest aggregate experiment report. Do not restart the task from conversational memory alone.
2. Check for an active worker or harness lock. Continue or inspect that work before starting another process. Keep heavy CPU training serial unless an explicit resource budget allows more.
3. Audit a concrete hypothesis against training/development failure evidence. Use separate agents for the experimental method, implementation, and independent evaluation review. Agents must exchange exact artifact identities and findings, not assume that another worker validated them.
4. Freeze a bounded trial plan before fitting or scoring. Include source hashes, data identities and splits, candidate recipes, seeds, compute limits, expected outputs, and the acceptance policy. The ledger must retain every planned and attempted candidate, including failures and timeouts.
5. Run external-only training and tests. Each trial writes into a new local run directory. Do not overwrite consumed trial outputs, silently retry a failed attempt, or edit frozen scientific sources underneath a running experiment. A corrected attempt gets a new identity with a recorded reason.
6. Verify artifacts and recompute aggregate metrics. Engineering test success and command completion are not evidence of model improvement. Report negative results as well as gains.
7. Advance the bounded development campaign only when the preceding evidence supports the next hypothesis. At the end of its declared budget, review the whole family. Do not expand it silently, lower thresholds, omit a losing setting, or call reused development matches fresh.
8. Save the next concrete action and any blocker in durable state. A later heartbeat resumes there. If rights or evidence are insufficient, work on independent engineering, documentation or permitted data provenance; do not treat elapsed time as approval or fabricate a successful trial.

The durable harness root is `runs/continuous-improvement/`. It is ignored local state, not a provider-data publication channel. The executable harness restricts commands to the external reconstruction workflow and cannot run a final audit or promote a model. Its success state only means that the declared engineering job completed.

The completed reconstruction remains in that original ledger. Its next experiment uses the separate versioned `tools/development_harness.py` and `runs/external-pff-development-v1/` for the single fixed external candidate's PFF development comparison. When that directory exists, inspect its status and lock before any new job; the old ledger being complete does not mean the development evaluator is idle. Read `runs/continuous-improvement/next-action.json` for the current handoff. Never rerun the completed reconstruction to advance the evaluator.

The subsequent fixed training-only component diagnosis also completed: `tools/clean_pff_training_diagnosis.py`, `runs/clean-pff-training-diagnosis-v1/`, and its bounded operator ledger at `runs/clean-pff-training-diagnosis-v1-operator/`. All 83 stages are consumed. [Results and independent audit](CLEAN_PFF_DIAGNOSIS_RESULTS.md) are published; do not rerun the study. The [next matched adaptation family](CLEAN_PFF_ADAPTATION_PLAN.md) is a design proposal, not a frozen executable recipe. Implement and test its training-only selection protocol before fitting; preserve all earlier source bytes and ledgers.

The harness writes `harness.json`, an authoritative hash-chained `ledger.jsonl`, and a derived `checkpoint.json`. Example operator commands, from the repository root with the project Python environment active:

```powershell
python tools/experiment_harness.py init --out runs/continuous-improvement
python tools/experiment_harness.py enqueue --out runs/continuous-improvement --recipe runs/external-rebuild-001-recipe.json
python tools/experiment_harness.py run --out runs/continuous-improvement --max-trials 1 --max-seconds 2700 --trial-seconds 1800
python tools/experiment_harness.py status --out runs/continuous-improvement
```

The recipe file is local JSON with exactly the declared trial identifier, action and rebuild parameters. For example:

```json
{"schema_version":1,"trial_id":"external-rebuild-001","action":"run","parameters":{"schema_version":1,"seed":10527,"epochs":8,"control_l2":0.001,"transition_weight":1.5}}
```

The runner advances one rebuild stage at a time. A successfully finished intermediate stage remains queued for the next invocation; a crashed or failed attempt does not silently retry. An interrupted worker blocks the harness pending process review, including other queued work; inspect and stop any orphan process before creating a replacement harness. Source hashes for the transitive local import closure, harness, policy and dependency manifests are frozen when the trial is enqueued. Finish review and tests before enqueueing. Changes to bound scientific sources require a new trial identity instead of modifying the sources behind an existing trial.

## Evidence and promotion contract

`tools/improvement_policy.py` consumes aggregate records and provenance documents. It never opens raw match records or calls the final evaluator. It checks:

- A complete connected ancestry graph with no challenge-derived fit inputs, no unknown parent hashes, and no cycles. Known v1.4/v1.5 and subsequent PFF artifacts cannot be renamed into acceptable training parents. v1.5 is permitted only as a separate comparator.
- Actual bytes for every declared learned artifact. A model hash authenticates its identity; it does not establish that its training history was honestly recorded.
- The original PFF 40/12/12 reservation and original PFF parent development records, both pinned by digest. A new study cannot redefine the historical gains it must retain.
- All twelve development matches with the same paired baseline and reference support; complete finite metrics; the original safeguards; at least **0.002** mean stoppage-precision gain over v1.5; and at least **90%** retention of the original PFF candidate's positive mean gains in the seven other metrics.
- Review-document hashes and coverage for local research, competition submission, and dense-reference validity. The PFF evaluation source requires evidence even if it was not used for fitting. Missing, prohibited or unresolved findings block review readiness.

The receipt schema records the dataset, purpose, finding, reviewer, source URL and document hash. A supported finding remains a reviewer's claim: the program does not infer a licence from `true`, certify legal rights, or certify human annotation merely because a file exists. PFF fitted-model submission rights and independent dense reference validation remain unresolved in the current evidence.

The screen always returns `final_access_authorized: false` and `promoted: false`. `ready_for_independent_final_review` is a screening result, not an access token. Before a final run, the coordinating agent must substantively review the evidence and separately verify the frozen complete selection, actual prediction bytes, prediction-before-reference timing, and the identity of exactly one eligible candidate. A final12 runner must independently enforce those checks at each entry point; editing a status JSON must never bypass them.

The final twelve fixtures remain untouched until that process passes. Prepare and freeze the selected candidate and comparator predictions for the entire final cohort before reference access. Use the final cohort once. A failure is a recorded failure, not an invitation to try the runner-up or retune on those fixtures. Replacement additionally requires the final numerical gates, adequate reference evidence, permissible model delivery, and a compliant training lineage. Never confuse a numerically stronger but ineligible model with a submission-ready model.

## Data handling

Do not send the supplied challenge tracking, labels, video, crops, or frame-level records into AI services or agent tool context. Local automatic checks permitted by the clarification may report only the necessary aggregate result. Do not compare predictions against challenge video or make human labels for that game. All fitting and parameter selection use external data.

Keep raw external provider files, reconstructed frame references, restricted documentation and fitted PFF artifacts in ignored local storage. Private GitHub status does not grant redistribution rights. Publish only original code, non-reconstructive aggregate findings, protocols and documentary evidence that the user is entitled to share. Do not send agreements, emails, organiser forms or submissions without separate authorization.

## Honest progress reporting

Report what actually ran, which artifacts were authenticated, what tests passed, which metric and cohort improved, what gates failed, whether the final reservation changed, and what is next. Do not report background workers as active without checking them. Do not claim a win probability or hidden challenge score from external development metrics.

The existing two correction families remain historical development results: eight global penalties and six context blends, with no eligible winner. The completed context suite recorded 580 passing engineering tests. New harness/rebuild tests and runs need their own actual results; the old count is not evidence for new code. The combination correction was not completed or run before the rule review and is not an established improvement.
