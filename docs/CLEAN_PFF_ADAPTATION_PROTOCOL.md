# Clean PFF adaptation: inner protocol and kernel

Status: engineering implementation of the [matched A/B proposal](CLEAN_PFF_ADAPTATION_PLAN.md). No real adaptation fit, inner comparison, winner refit or new development comparison is established by these modules. A numerical selector is **not an authenticated execution runner**. The complete source/cache/stage/lock manifest, consumed-attempt ledger and independently verified winner-refit adapter must exist before real execution.

The reviewed scientific specification and 24-file static source/dependency snapshot are saved in `models/clean-pff-adaptation-protocol.json`. The core digest is `a1271ea8c6b95848d1379f5770fc4c1788c82d2e4472b7123edcbeb4fe66e02f`; the full snapshot digest is `4df9683901ef0eafa08e5344a43870558b09173ac09fc1c3377514ec363cce53`. This snapshot fixes the two recipes and selection rule before outcomes. It does not certify runtime execution, cache rights or campaign readiness.

`tools/clean_pff_adaptation_protocol.py` defines the fixed scientific core and validates aggregate receipts. `tools/clean_pff_adaptation_fit.py` implements only the inner-fitting kernel. It has no file loader, CLI, scoring command or winner-refit API. All model weights returned by that kernel must remain in ignored local storage.

## Exact family and split

There are exactly two candidates. A retains the authenticated clean upstream stoppage head and learns a new joint residual. B refits the upstream stoppage head and learns the same new joint architecture using B's upstream outputs. Both retain the clean 98-feature H/A control head and decoder configuration. The old joint residual is removed, never used as a teacher. Cached historical upstream probabilities are forbidden.

The original PFF training forty are ranked by ascending hexadecimal SHA-256 of `xtension-clean-pff-adaptation-v1:inner:<decimal match ID>`. The first eight are inner validation; the other 32 are inner fitting. Membership uses no labels or scores. All forty were already exposed in the component diagnosis, so this split is training-development evidence, not a fresh test.

| Partition | Exact match IDs |
| --- | --- |
| Inner fitting | 3812, 3817, 3819, 3820, 3822, 3825, 3826, 3827, 3828, 3829, 3832, 3834, 3835, 3838, 3839, 3843, 3845, 3847, 3849, 3851, 3853, 3855, 3856, 3859, 10503, 10504, 10505, 10507, 10509, 10512, 10515, 10516 |
| Inner validation | 3821, 3833, 3836, 3842, 3846, 3857, 10508, 10513 |
| Mandatory clean Sportec rehearsal, fitting only | J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 |

The seven Sportec fixtures and upstream parent have historical training exposure; there is no claim of cross-fitted upstream outputs. New normalization and fit statistics use only the 39 fitting fixtures. The declared future winner-refit roster is the original PFF forty plus these seven Sportec fixtures; declaring that roster does not authorize access or execution. Original development12 and final12 reservations remain unchanged.

## Matched recipe

| Setting | Value |
| --- | --- |
| Seed / joint epochs | 20261101 / 8 |
| Joint architecture | Existing equivariant joint temporal residual |
| Joint optimizer | Existing AdamW, learning rate 0.002, weight decay 0.01 |
| B's new D head | Existing logistic fit, L2 0.01, max iterations 80 |
| Sampling | 5 Hz, independently inside each period of 25 Hz input |
| Fixture mass | Each fitting fixture contributes total sampled weight 10,000 |
| Live spell weighting | Existing inverse-square-root rule, capped 0.5–3 |
| Stoppage / unknown weights | 1 / 0 before fixture normalization |
| Boundary weighting | Existing rule within 0.4 seconds, minimum weight 1.5 before fixture normalization |
| Upstream stacking | In-sample fitting outputs from the actual retained/refitted head |
| Sportec regimes | Clean only, all seven required |
| Search budget | A and B only; no decoder, loss-weight, seed or epoch grid |

The kernel checks the exact canonical clean parent contents, not only a caller's claimed hash. It validates the entire fitting roster before invoking a lazy record factory, checks the returned fixture identity and schema before materializing arrays, copies sampled records and recomputes upstream probabilities. B's D fit and both joint fits use the same sampled fixture-weighted records. Parent H/A parameters, configuration and A's D parameters remain fixed. Sampling and sequence boundaries preserve period separation.

The clean parent file identity is `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`; the configuration identity is `2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b`. The protocol's `fixed_protocol_sha256()` hashes its scientific core. A full protocol additionally binds source names/hashes and the exact comparator aggregate records. These are different identities; a future execution manifest must bind actual transitive source bytes and local cache artifacts separately.

## Inner qualification and ranking

The fixed comparator is the clean full model from the completed training-only diagnosis, extracted for exactly the inner eight. The complete published diagnosis and extracted comparator records are pinned by canonical digest. The selector requires both complete eight-fixture prediction cohorts, distinct A/B model identities, matching fit receipts and complete score records. It rejects missing, duplicate or forbidden-cohort rows and non-finite, Boolean or out-of-range metrics. Reference support must match the paired comparator. Short precision must reconcile with matched/predicted short-spell counts, and interruption/erasure counts must reconcile with available short-reference support.

Each candidate qualifies only when all of the following hold:

- Equal-fixture mean spell F1 gain is strictly positive.
- Equal-fixture mean spell F1, short-possession precision, stoppage precision and stoppage recall do not decrease.
- Every fixture's spell F1 is at least its comparator value minus 0.01.
- Every fixture's stoppage recall is at least its comparator value minus 0.005.
- Every fixture has at least one reference stoppage segment and short spell; the eight fixtures total at least 100 of each.

There is no numerical tolerance allowance. Worst-fixture checks compare the candidate directly against the comparator-minus-limit threshold, avoiding an extra subtraction that can reject an exactly inclusive boundary through binary floating-point rounding. Displayed gains remain candidate-minus-comparator.

Qualifying candidates rank by descending equal-fixture means of spell F1, short precision, stoppage precision, then stoppage recall; exact ties favor A. If neither qualifies, the family stops without a winner. Reports retain all eight metric means/gains, each check, worst-fixture losses and short-fragmentation/erasure counts for both candidates. Passing this inner rule is neither a development win nor a statistical significance claim.

## What receipts do and do not prove

Receipt validation checks consistent declared model identities, fitting/normalization rosters, prediction hashes, score hashes and prediction-before-reference declarations. It does not read actual prediction/model/cache bytes or independently prove chronology. Source maps must include both new modules, but this pure selector does not verify the complete dependency closure or hashes against disk. A self-consistent fabricated receipt can therefore satisfy numerical selection; it cannot authorize execution.

Every selector result retains `execution_evidence_verified`, `campaign_ready`, `refit_authorized`, `development_access_authorized`, `final_access_authorized`, `promoted` and `submission_eligible` as false. A future runner must independently authenticate actual bytes, connected clean lineage, local cache semantics, full source closure, stage order, locks and consumed attempts. It must freeze both A/B prediction cohorts before reading inner scoring references, permit at most one winner refit and preserve the single subsequent development comparison. No runner-up substitution or hidden retry is allowed.

The original development safeguards, +0.002 absolute mean stoppage precision over v1.5 and at least 90% retention of original PFF gains remain unchanged. Rights for fitted PFF artifact delivery and independent dense-reference validity remain unresolved. v1.5 is unchanged on submission hold. The final twelve remain untouched; no automatic promotion or submission is enabled.

## Verification scope

Synthetic tests exercise receipt corruption, held-out exclusion, recipe mutation, complete-cohort requirements, positive/negative selection, tie-breaking, exact guard boundaries and short-count consistency. Separate tests compare the allowed published non-reconstructive diagnosis aggregates to the extracted comparator. Tests use synthetic records or aggregate reports only; they do not train on provider data, open reserved fixtures or establish improved predictive performance. Test counts and execution results belong in the dated engineering checkpoint, not in this method specification.
