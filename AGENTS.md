# Persistent improvement work

The user explicitly authorized long-running, multi-agent implementation and testing for this project. Delegate distinct implementation and independent review tasks, coordinate file ownership, and preserve checkpoints between runs.

Read `docs/INITIAL_CHALLENGE_RULES_REVIEW.md`, `docs/CONTINUOUS_IMPROVEMENT.md` and `docs/IMPROVEMENT_BACKLOG.md` before continuing experiments. Consult the durable harness ledger under ignored `runs/continuous-improvement/`; do not create a second experiment runner if one is active.

## Data and training boundaries

- Never send the supplied challenge inputs, possession labels, video, crops, frame-level observations or predictions to an AI service or expose them in tool output. Agents may inspect code and non-reconstructive aggregate experiment reports. Authorized automatic data processing stays local.
- Never fit, tune, select parameters or check outputs against the challenge video or ground truth. The organiser requires external-only fitting. Do not reuse the target-fitted camera profile, its derived training caches or learned descendants as training ancestors.
- Preserve v1.5 bytes as a historical comparator only. It and its PFF descendants are on submission hold. No automatic deployment, competition submission, provider contact, agreement acceptance, paid compute or restricted-data upload is authorized.
- PFF raw data, reconstructed references, provider documents and fitted weights remain ignored and local. Training/analysis permission does not by itself establish judge-delivery rights.
- Final PFF match IDs are 3814, 3815, 3824, 3830, 3837, 3840, 3841, 3848, 3852, 10506, 10510 and 10511. Do not prepare, predict or read references until a compliant development winner passes the unchanged gates and the access decision is independently reviewed. The final cohort is consumed once, never a tuning loop.

## Execution and evidence

- Run only one heavy fit at a time, at most two data workers, and one BLAS/Torch thread per worker. Keep each scheduled advance bounded and resumable.
- Freeze trial definitions and executable source hashes before execution. Record failed, interrupted and negative trials; never silently reset them or reuse a trial identifier for different work.
- Keep historical scientific files immutable while their published evidence binds their hashes. Prefer new versioned tools; adding to `research/*.py` can invalidate older glob-based evidence.
- Preserve all original numerical gates, the additional +0.002 absolute stoppage precision requirement and at least 90% retention of the original PFF gains in the other seven metrics. Never relax them in response to failures.
- A passing subprocess or engineering test is not a better model. A development win is not fresh-final evidence. Provider-reference agreement is not hidden challenge accuracy. State every limitation.
- Publish only original code and non-reconstructive aggregate results after checking privacy, tests and the existing repository's private visibility. Never stage ignored local artifacts. Do not rewrite historical source archives or Git history to conceal provenance problems.
- Save the next concrete action and blockers before yielding. Report meaningful progress, a verified improvement, or an actionable failure; do not repeat unchanged status each scheduled run.
