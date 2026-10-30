# Xtension Labs — PossessionFlow 1.5.1

Retained v1.5: experimental combined did not pass both fixed gates (development=False, fresh=False).

Development gate: **fail**. Fresh audit gate: **fail**. Deployed model: **v1.5 joint-state**.

## Method

The pipeline combines observation-only tracking geometry, camera/missing-ball robust control, half-local temporal H/A/D learning, bounded tactical-tag evidence and duration-aware decoding. Inference is offline CPU-only. H/A/D/X labels preserve the required frame order and the known halftime/unknown regions.

V6 implemented stoppage-focused loss weighting and evidence-gated short-possession transitions. Active deployment is **v1.5 joint-state**. Rejected experiments remain in the repository with their measured outcomes; they are not silently enabled. Video recovery and broadcast fusion remain inactive.

## Evidence

| Candidate | Spell F1 (primary) | Short recall | Short precision | D segment F1 | D boundary F1 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| combined | 0.6216 | 0.2970 | 0.3978 | 0.9031 | 0.7947 | fail |
| transition | 0.6203 | 0.2912 | 0.3810 | 0.9163 | 0.7979 | fail |
| stoppage-loss | 0.6239 | 0.2877 | 0.4163 | 0.9031 | 0.7950 | fail |
| baseline-v1.5 | 0.6233 | 0.2784 | 0.3962 | 0.9163 | 0.7979 | incumbent |


| Dense provider metric | v1.5 | Frozen experiment | Change |
| --- | ---: | ---: | ---: |
| Possession spell F1 | 0.7091 | 0.7222 | +0.0132 |
| Frame macro F1 | 0.9428 | 0.9445 | +0.0017 |
| D precision | 0.9889 | 0.9884 | -0.0005 |
| D recall | 0.9693 | 0.9760 | +0.0068 |
| D segment F1, IoU .5 | 0.9032 | 0.8909 | -0.0123 |
| D onset F1, +/-1s | 0.7487 | 0.7701 | +0.0214 |
| D offset F1, +/-1s | 0.8272 | 0.8677 | +0.0405 |
| Short H/A recall, <2s | 0.4750 | 0.4917 | +0.0167 |


| Fresh phase match | Scored frames | v1.5 accuracy | Experiment accuracy |
| --- | ---: | ---: | ---: |
| skillcorner1925299 | 35,640 | 84.26% | 82.82% |
| skillcorner1927964 | 37,111 | 87.21% | 85.49% |


All seven Sportec matches are previously exposed development data. WCSF is an unused, anonymized provider sample with explicit AliveFlag/PossessionFlag references; annotation method and cross-provider fixture independence cannot be verified. Its two halves count as one match. SkillCorner phases are automatic labels and may bridge short disruptions; missing phases remain unknown, never D. The three audited sources are now consumed and cannot be reused as untouched final tests. No human-gold or hidden competition score is claimed.

## Reproducibility

The release contains frozen weights, config, exact frame CSV, source, tests, both selection records and reference provenance. 290 engineering tests pass. Clean-source inference and artifact hashes are verified in verification.json; Linux/Windows CI checks are linked from the release. No challenge reference labels were used for v6 training, selection or inference.

The repository remains private under likalight. The official competition form has not been submitted; organizer source-access and eligibility requirements remain outstanding.

Detailed fixed protocol and results: docs/TEMPORAL_CAMPAIGN_V6.md. Source provenance: docs/FRESH_DATA_V6.md.
