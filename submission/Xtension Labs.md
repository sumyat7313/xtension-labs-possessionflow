# Xtension Labs — PossessionFlow 1.5.2

Retained the original v1.5 joint-state model and configuration. V7 adds completed research and reproducible evidence; phase-only fresh labels cannot validate stoppage or short-possession improvements or authorize replacement.

Development promotion gate: **fail**. Fresh phase regression gate: **fail**. Fresh stoppage/short-possession validation: **unavailable**. Deployed model: **original v1.5 joint-state**.

## Method and release scope

Observation-only tracking geometry feeds camera/missing-ball robust control and half-local joint possession/stoppage learning. Bounded tactical-tag evidence and duration-aware decoding produce H/A/D/X labels while preserving frame order and unknown regions. Inference uses NumPy on CPU. V7 adds explicit boundary learning and confidence-gated geometric research candidates; their presence in source does not activate them in the retained submission model. No reliable independent video/contact channel is claimed.

## Measured results

| Candidate | Spell F1 (primary) | Short recall | Short precision | D segment F1 | D boundary F1 | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-v1.5 | 0.6233 | 0.2784 | 0.3962 | 0.9163 | 0.7979 | incumbent |
| typed-boundary | 0.6222 | 0.2899 | 0.3849 | 0.9134 | 0.7966 | fail |
| typed-contact | 0.6221 | 0.2899 | 0.3846 | 0.9134 | 0.7966 | fail |


Frozen experimental audit candidate: **typed-boundary**.

| Fresh phase match | Scored frames | v1.5 accuracy | Experiment accuracy | Change (percentage points) | v1.5 balanced recall | Experiment balanced recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| skillcorner1953632 | 32,739 | 86.90% | 86.83% | -0.076 | 86.59% | 86.52% |
| skillcorner1959846 | 29,798 | 86.30% | 86.29% | -0.010 | 86.30% | 86.29% |


| Observation regime | Restart candidates | Matched D-to-live boundaries | Alignment precision | Reliability gate |
| --- | ---: | ---: | ---: | --- |
| clean | 39 | 12 | 30.8% | Failed |
| degraded | 0 | 0 | Not estimable | Failed |
| camera | 0 | 0 | Not estimable | Failed |
| calibrated | 20 | 5 | 25.0% | Failed |


All seven Sportec matches are previously exposed development data; five matches are scored under four correlated observation conditions, not twenty independent matches. The two new SkillCorner matches have automatic phase labels, not dense independent H/A/D truth. Missing phases remain unknown, never dead-ball labels. They cannot establish stoppage boundaries, short-possession quality, or physical contacts. These audited sources are now consumed and cannot be reused as untouched final tests. No hidden competition score, human-gold validation, or winning probability is claimed.

## Reproducibility and submission status

342 engineering tests passed. Follow SOLUTION.md for inference and research reproduction. The release includes frozen weights, the original deployment configuration, source, tests, aggregate selection/audit evidence and a team-named submission CSV. Restricted source data, frame-level references and local experiment caches remain excluded. Current clean-source/output verification is recorded in docs/verification.json.

The private repository is under likalight. The organizer's submission form has not been submitted; organizer access and eligibility remain outstanding. See docs/BOUNDARY_CAMPAIGN_V7.md for the complete research outcome and docs/FRESH_DATA_V7.md for label limitations.
