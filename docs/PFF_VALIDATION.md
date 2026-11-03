# PFF fresh-fixture reservation and validation

On 2026-11-01, the complete 64-fixture metadata catalog was downloaded and hashed, and [the protocol](../models/pff-protocol.json) was frozen **before opening any PFF event or tracking records**. Only six normalized identity fields enter the split: match ID, date, home/away teams, stage and competition. Native metadata competition/season and stage-week consistency were checked. Original files remain local.

Protocol SHA-256 over canonical content excluding the hash field: `4475cf6cf95de433a49a1c03ab93838be96f05059eee15a09a73e1673557e980`.

The frozen file is an immutable reservation, not a statement that the labels are suitable, training is complete or v1.5 has been beaten. Experimental promotion requires both the fixed metrics and independently substantiated reference/provenance conditions. The active v1.5 artifact remains `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.

## Prior exposure audit

The [identity audit](../models/pff-identity-audit.json) records prior fixture metadata and hashes. No prior consumed external source is a World Cup 2022 fixture:

| Previously consumed source | Exclusion evidence |
| --- | --- |
| Seven Sportec fixtures | Authenticated XML identifies Bundesliga/2. Bundesliga club fixtures in the 2022/23 season. |
| Seven SkillCorner fixtures | Authenticated match metadata identifies A-League club fixtures from November 2024 through January 2025. |
| Metrica sample games 1–3 | The [upstream README](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/README.md) records Game 3's release on 2021-04-15, with the other two samples already present. They predate World Cup 2022. |
| WCSF sample | The [pinned source README](https://github.com/MatKempeGroningen/WCSF_SolutionRoom/blob/1c9ba798f98a9fffb16081a5d3965774d4e5662c/README.md) explicitly describes a professional European league match. Its exact identity remains anonymized, but its competition class excludes a national-team World Cup fixture. |
| Challenge target | Supplied broadcast metadata names Barcelona versus Athletic Club, a club fixture. Competition references were not opened. |

Earlier reports correctly left WCSF's exact identity unresolved. The stronger conclusion here is specific: league-versus-national-tournament metadata resolves nonoverlap with this PFF release; it does not recover the original fixture identity or prove nonoverlap with every other football dataset.

## Exact split

Rank IDs within group and knockout stages using SHA-256 of `xtension-pff-v1-2026-11-01:{match_id}`. For the 48 group fixtures, take the first 9 for development, next 9 for final testing and remaining 30 for training. For the 16 knockout fixtures, take 3, 3 and 10 respectively. No score, label, event count or model prediction affects the ranking. Whole fixtures include both halves and extra time. Teams recur across partitions; this is a fresh-match test, not a claim of generalization to unseen teams or leagues.

| Partition | Group stage | Knockout | Total |
| --- | ---: | ---: | ---: |
| Training | 30 | 10 | 40 |
| Development | 9 | 3 | 12 |
| Locked final | 9 | 3 | 12 |

Training: 3812, 3817, 3819, 3820, 3821, 3822, 3825, 3826, 3827, 3828, 3829, 3832, 3833, 3834, 3835, 3836, 3838, 3839, 3842, 3843, 3845, 3846, 3847, 3849, 3851, 3853, 3855, 3856, 3857, 3859, 10503, 10504, 10505, 10507, 10508, 10509, 10512, 10513, 10515, 10516.

Development: 3813, 3816, 3818, 3823, 3831, 3844, 3850, 3854, 3858, 10502, 10514, 10517.

Locked final: **3814, 3815, 3824, 3830, 3837, 3840, 3841, 3848, 3852, 10506, 10510, 10511**.

Pilot **3812 (Senegal–Netherlands)** is training-only. Its role was recorded locally before schema inspection. `build_protocol` rejects incomplete catalogs or duplicate underlying fixture identities. If missing data prevents evaluation, do not drop a difficult final fixture and keep the same success claim; document an incomplete audit and retain v1.5.

## Promotion requirements

`tools/pff_validation.py` checks one frozen candidate versus exact v1.5 on every reserved native-condition fixture. Both models must use identical observed-input and scoring masks; reference fields, annotation-derived cues and event labels never enter inference. Save and authenticate both prediction sets before separate final reference scoring. The utility verifies a supplied evidence declaration; callers must substantiate it with real source/prediction artifacts. It cannot certify that human review, blinding or rights verification happened.

The fixed numeric gates carry forward the access plan: mean possession-spell F1 gain at least 0.005; no mean regression in short-spell precision/recall or stoppage segment/boundary F1/precision; at least 0.01 gain in short precision or stoppage boundary F1; macro F1 decline at most 0.001; no fixture spell-F1 decline above 0.01 or dead-recall decline above 0.005. Stoppage segment F1 and short-spell metrics use the existing project's one-to-one IoU metrics, with the existing under-two-second definition; boundary F1 uses one-second tolerance. The scoring adapter must preserve periods and unknown reference gaps.

A deterministic paired bootstrap uses 20,000 fixture-level resamples, NumPy RNG seed 20261101, and the 5th/95th percentiles of mean spell-F1 gain. Its lower endpoint must exceed zero. Correlated frames or repeated corruptions are not independent resamples. Recurring teams and one tournament limit the inference; a positive diagnostic interval does not prove out-of-tournament superiority.

All 12 final fixtures must contribute defined metrics, at least one explicit reference stoppage and short spell each, and at least 100 of each across the full test. These are minimum support floors, not a power guarantee. A separate adequate-support declaration must consider coverage, uncertainty and confidence. Dense state/possession truth, independent observations, fixture identity, permitted use, passed development checks and prediction/reference separation must all be established. Sparse positive events, automatic phases of unclear completeness or inferred stoppage gaps cannot meet those conditions.

Final-test failure is final for this reservation. Further tuning consumes the test and requires another untouched test set. No automatic artifact replacement is implemented by these utilities.

## Engineering checks

Fourteen focused tests pass: deterministic/stage-balanced splits, label-field rejection, underlying-fixture duplication, missing metadata hashes, rehashed split tampering, immutable reservations, evidence requirements, stoppage regressions, missing test fixtures, multiple candidates, invalid/sparse metric support, reproducible bootstrap and strict separation of development/final gates. `development_gate(records, protocol)` applies the same numeric criteria to exactly the twelve development fixtures and always returns `promoted=False`; it cannot replace the final audit. No production modules were modified.

Nine campaign tests additionally verify the direct final-evaluation guard, rejection of training-as-test claims, whole-cohort prediction authentication before the first reference read, immutable scientific sources, and exclusive access markers that consume even interrupted scoring attempts. A perfect synthetic period-separated example exercises the real metric adapter. Eighteen aggregate-verifier tests reject tampered gates, leakage, source drift, false human-label/usage-rights claims, changed prediction/access identities, unsupported reference counts, unreported final preparation and deployment/weight-publication claims.

```powershell
python -m unittest discover -s tests -p test_pff_validation.py -v
python -c "import json; from tools.pff_validation import verify_protocol; verify_protocol(json.load(open('models/pff-protocol.json'))); print('PFF reservation verified')"
```

Once the research run completes, `python tools/verify_pff_research.py --run runs/pff-campaign-v3 --write models/pff-research.json` authenticates local model/cache/prediction bytes and writes an aggregate-only evidence bundle. The default `python tools/verify_pff_research.py` then verifies the published bundle without provider data or experimental weights. It verifies the exact v1.5 decoder config and candidate identities, derives metric records from paired diagnostic details, and recomputes metric gates from those aggregates; it does not independently recompute predictions or annotations and does not certify declarations of review. Production release integrity remains covered by the existing release verifier.
