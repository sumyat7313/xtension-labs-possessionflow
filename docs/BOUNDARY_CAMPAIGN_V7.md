# V7 explicit boundaries and contact evidence

Retained the original v1.5 joint-state model and configuration. V7 adds completed research and reproducible evidence; phase-only fresh labels cannot validate stoppage or short-possession improvements or authorize replacement.

Development promotion gate: **fail**. Fresh phase regression gate: **fail**. Fresh stoppage/short-possession validation: **unavailable**. Deployed model: **original v1.5 joint-state**.

## Implemented and evaluated

The explicit boundary learner predicts four classes: no event, team change, live-to-dead and dead-to-live. It uses 90 observation/model features; the contact ablation adds eight observed-only geometric proxies. Team symmetry, half-local context, complete observation gates and invalid-input rejection are explicit. The four-class learner supplies separate bounded switch, stop and restart costs to the duration-aware decoder. These are offline context features, not calibrated physical-event probabilities or a streaming model.

The fixed comparison evaluates the original v1.5 comparator, typed boundaries, and typed boundaries plus geometric proxies in 60 cells: three procedures, five held-out evaluation matches, four conditions. Each experimental outer-fold fit excludes the entire scored fold. Historical v1.5 out-of-fold prediction bytes are reproduced. The all-seven candidate refit is never scored as an out-of-fold model.

## Development results

| Candidate | Spell F1 (primary) | Short recall | Short precision | D segment F1 | D boundary F1 | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-v1.5 | 0.6233 | 0.2784 | 0.3962 | 0.9163 | 0.7979 | incumbent |
| typed-boundary | 0.6222 | 0.2899 | 0.3849 | 0.9134 | 0.7966 | fail |
| typed-contact | 0.6221 | 0.2899 | 0.3846 | 0.9134 | 0.7966 | fail |


Primary spell F1 averages clean/degraded/camera conditions. Other table metrics average all four conditions, including calibrated-camera corruption. D boundary F1 averages one-to-one onset/offset F1 within one second. Short recall covers true H/A spells under two seconds at IoU 0.5; short precision stratifies predicted short spells after global matching.

The frozen development gate requires at least 0.005 primary spell-F1 gain; nondecreasing mean D segment F1, D boundary F1 and short recall; at least 0.01 gain in D boundary F1 or short recall; and fixed precision, macro-F1 and worst-cell regression limits. The exact numeric policy remains in the evidence protocol. Candidates rank by primary spell F1, D boundary F1, short recall, then name. No gate changed after scoring.

Development selection: **baseline-v1.5**. One frozen experimental audit candidate: **typed-boundary**. No runner-up was evaluated after fresh-reference access.

## Fresh phase regression audit

| Fresh phase match | Scored frames | v1.5 accuracy | Experiment accuracy | Change (percentage points) | v1.5 balanced recall | Experiment balanced recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| skillcorner1953632 | 32,739 | 86.90% | 86.83% | -0.076 | 86.59% | 86.52% |
| skillcorner1959846 | 29,798 | 86.30% | 86.29% | -0.010 | 86.30% | 86.29% |


The phase regression gate **failed**. This is a limited possession-phase regression check, not fresh evidence of better stops or short possessions. The policy makes promotion ineligible even if phase agreement improves. All four paired prediction CSVs were sealed before parsing phase references; the split, model/config hashes, source hashes and data identities were frozen first. No fresh references enter training or candidate selection.

All seven Sportec matches are previously exposed development data; five matches are scored under four correlated observation conditions, not twenty independent matches. The two new SkillCorner matches have automatic phase labels, not dense independent H/A/D truth. Missing phases remain unknown, never dead-ball labels. They cannot establish stoppage boundaries, short-possession quality, or physical contacts. These audited sources are now consumed and cannot be reused as untouched final tests. No hidden competition score, human-gold validation, or winning probability is claimed.

## Contact and restart reliability

| Observation regime | Restart candidates | Matched D-to-live boundaries | Alignment precision | Reliability gate |
| --- | ---: | ---: | ---: | --- |
| clean | 39 | 12 | 30.8% | Failed |
| degraded | 0 | 0 | Not estimable | Failed |
| camera | 0 | 0 | Not estimable | Failed |
| calibrated | 20 | 5 | 25.0% | Failed |


The frozen development-only requirement was at least 90% alignment precision with at least 20 accepted candidates per regime. It failed. H/A/D boundary alignment is not physical-touch precision. The native XY extractor and cache adapter are conservative geometric candidate generators, with explicit missing-ball abstention; independent physical-contact accuracy remains unestablished. Hard contact/restart overrides and independent video evidence remain inactive. The prior video detector pilot also remains inactive.

## Verification and evidence

342 engineering tests passed in the release run. This count establishes tested implementation behavior, not football accuracy. Clean-source inference reproduction, package hashes and current CI status are recorded separately in [verification.json](verification.json) and the CI workflow; the renderer does not run or assert those checks itself.

- [Development selection and fixed protocol](../models/boundary-selection-v7.json)
- [Fresh phase audit](../models/fresh-audit-v7.json)
- [Contact reliability evidence](../models/contact-audit-v7.json)
- [Typed-boundary design](TYPED_BOUNDARY_V7.md)
- [Contact limitations](CONTACT_RESTART_V7.md)
- [Source and label provenance](FRESH_DATA_V7.md)

Development protocol SHA-256: `1428aadfe7fa012b6ae056ad8f4b841f22faad0991f07bc09989a21b80dade00`.

Audited candidate SHA-256: `abaa8ec8cf953e86bd3eb081a18a6a7a1a68fc35e232d82ac2b0318797307a52`.

Deployed model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.
