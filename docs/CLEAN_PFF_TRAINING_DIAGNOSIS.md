# Next bounded task: training-only false-stoppage diagnosis

Planning checkpoint after the failed [fixed development comparison](EXTERNAL_PFF_RESULTS.md). No diagnosis or adaptation fit has been run yet. Its purpose is to identify the contributing component before choosing another model family.

Use only the original forty PFF **training** fixtures from the unchanged reservation. Reject development and final IDs before any input read. Authenticate existing observation-feature caches and recompute all upstream probabilities using the clean candidate `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`; never use the v1.5 cached upstream probabilities. Keep period blocks, decoder settings, observation support and conservative references identical.

Freeze three fixed ablations before reference access:

1. Clean control and frozen dead head, with no joint correction.
2. Joint home/away correction retained, joint dead correction disabled.
3. The complete clean candidate.

This is a component diagnosis, not a parameter sweep or a fresh-test claim. It performs no fit. Test that removing only the D correction preserves conditional H:A probabilities, and that the complete candidate reproduces its existing inference path. Preserve all input/source identities, failures and completed stages in a new bounded run; leave earlier scientific modules and evidence unchanged.

Report all forty per-fixture aggregate outcomes: false-D and missed-D counts, precision/recall, conditional team accuracy, and errors grouped by fixed confidence bins and observation availability. Select bin definitions from existing documented feature semantics before scoring; do not inspect labels to choose flattering groups. Ball/player visibility and missing-observation duration must come from observations. If restart distance is reported, it belongs only to reference-side diagnostic grouping and must never become an inference feature. Publish no frame-level examples or provider records.

The development comparison found 37,128 extra false-D frames for 16,794 recovered true-D frames. That motivates the investigation but does not prove which component caused the regression. Use the training ablations to decide whether a new joint residual suffices or whether the dead head also needs external-data refitting.

Only after the diagnosis, predeclare one adaptation family with at most two candidates using forty PFF training and seven Sportec fixtures, clean ancestors, fixed seeds/epochs, training-only normalization and equal fixture weighting. If needed, select a single live-frame error penalty using a whole-match subdivision within the training partition before any candidate development scoring. Report both candidates; do not expand the family after seeing scores.

All original development safeguards, +0.002 stoppage precision and 90% retention against the original PFF gains remain unchanged. The reused development twelve are not fresh, and the final twelve remain locked. PFF-fitted artifacts stay local; delivery rights and dense-reference validity remain separate unresolved evidence gates. No automatic replacement or submission is authorized.
