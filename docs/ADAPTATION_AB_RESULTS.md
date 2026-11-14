# Fixed clean PFF adaptation A/B experiment

Recorded November 14, 2026. **Both fixed candidates passed the inner numerical gates; A won the predeclared ranking.** The complete 29-stage campaign, actual local artifacts and independent aggregate calculations were verified. This establishes an improvement over the clean parent on eight reused training-development fixtures, not a win over v1.5 or a fresh-test result. No candidate has been promoted. v1.5 remains an unchanged historical comparator on submission hold; the final twelve PFF fixtures remain sealed.

## Question and fixed comparison

The preceding clean external rebuild lost stoppage precision and short-possession precision on the already-used development cohort. This experiment tests whether adapting the clean upstream stoppage head, alongside a new joint temporal model, improves the declared training-development comparison without those regressions.

- **A:** retain the clean upstream stoppage head and fit a new joint temporal model.
- **B:** refit the upstream stoppage head and fit a new joint temporal model.

Both candidates retain the same clean home/away head, configuration and decoder. The old joint head is discarded. Neither candidate uses the challenge-fitted camera profile, its augmented caches, or v1.5 as a learned parent. Upstream predictions used in joint fitting come from the fitting cohort itself; this is explicitly in-sample stacking, not cross-fitting.

Each fit uses 32 original PFF training fixtures and seven clean Sportec fixtures. Each fixture receives total weight 10,000. The frozen recipe samples at 5 Hz, uses the existing capped short-spell weighting and 0.4-second boundary weighting, and fits eight joint epochs with seed 20261101. B uses the fixed stoppage-head regularization of 0.01 and at most 80 iterations. No decoder search, added candidate or parameter-grid expansion is permitted.

The comparison cohort comprises the other eight fixtures from the original PFF training reservation. These fixtures have previously been exposed in training-development work. **They are not fresh matches or an untouched test set.** No development12 or final12 fixture is part of this A/B fitting or scoring sequence.

## Selection and safeguards

All sixteen candidate/fixture prediction artifacts must be completed and frozen before either candidate is scored against the inner references. The fixed operator records initialization, the two fits, sixteen predictions, the complete-cohort freeze, eight paired scoring stages and selection in an append-only, hash-chained ledger.

A candidate qualifies only if mean possession-spell F1 improves, mean spell F1/short-possession precision/stoppage precision/stoppage recall do not regress, no fixture loses more than 0.01 spell F1 or 0.005 stoppage recall, and the original reference-support requirements are met. Qualified candidates are ranked by spell F1, short-possession precision, stoppage precision and stoppage recall, with A breaking an exact tie. If neither qualifies, no winner refit is authorized.

An inner winner would still require the one permitted refit, connected and authenticated lineage, and a separately reviewed development route. All original development checks, at least +0.002 absolute stoppage precision over v1.5, and at least 90% retention of the original PFF gains remain unchanged. Passing this inner comparison cannot authorize final access, replacement or submission.

## Execution identity and independent review

The frozen scientific source closure contains 66 files. A pre-execution comparison found that the base Python executable hash had changed since the prior prospective operator plan. Scientific sources, dependencies, split, recipe and gates were unchanged. The previous failed Windows-launch attempt was preserved; no consumed attempt or candidate budget was reset.

Before the new operator manifest was initialized, the current runtime passed 60 launcher, operator, repair-integration and fitting-kernel tests in 47.914 seconds. The new prospective runtime binding was explicitly recorded and independently reviewed. The checked initialize stage matched the earlier authenticated input plan. Heavy fits ran serially with one worker and one BLAS/Torch thread.

| Identity | SHA-256 |
| --- | --- |
| Scientific core | `a1271ea8c6b95848d1379f5770fc4c1788c82d2e4472b7123edcbeb4fe66e02f` |
| Full scientific protocol | `4df9683901ef0eafa08e5344a43870558b09173ac09fc1c3377514ec363cce53` |
| Reviewed execution manifest | `448a47d9d76d498785011d00e27fbbc72a41dc16eeea750299d02a16dd5f88dc` |
| Clean parent | `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47` |
| Configuration | `2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b` |
| A inner model, local only | `68291717233233eca1786e7285e1c635c15053a0ad3970a84a65a2f4867618b6` |
| B inner model, local only | `aadcdb3d13e2f72d04f43f5d7edb4f1f558c195e659af27625a1b710a055b40d` |

Independent fit reviews authenticated the aggregate reports and receipts against completed stage seals, checked the fixed 39-fixture roster, exclusions, recipe, parent and configuration, and verified the recorded single-thread fitting settings. The coordinator separately verified actual local artifacts. Reviewer agents did not open model weights, predictions, references or provider records.

## Complete inner results

The following values are arithmetic means over the same eight inner fixtures. Gains are absolute differences from the clean parent, not percentage changes or comparisons with v1.5. All eight metrics improved for both candidates.

| Metric | Clean parent | A | A gain | B | B gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Possession spell F1 | 0.436570 | 0.575684 | +0.139113 | 0.569207 | +0.132636 |
| Frame macro F1 | 0.853119 | 0.898543 | +0.045424 | 0.897883 | +0.044765 |
| Short-possession precision | 0.089354 | 0.148067 | +0.058713 | 0.163309 | +0.073956 |
| Short-possession recall | 0.116987 | 0.141942 | +0.024955 | 0.141624 | +0.024637 |
| Stoppage segment F1 | 0.690670 | 0.875060 | +0.184390 | 0.872838 | +0.182168 |
| Stoppage boundary F1 | 0.355089 | 0.589091 | +0.234002 | 0.592898 | +0.237809 |
| Stoppage precision | 0.919590 | 0.944086 | +0.024496 | 0.950438 | +0.030848 |
| Stoppage recall | 0.915715 | 0.963799 | +0.048083 | 0.959087 | +0.043372 |

Both passed all seven fixed numerical checks: `nonregression_spell`, `nonregression_short_precision`, `nonregression_dead_precision`, `nonregression_dead_recall`, `positive_mean_spell`, `worst_fixture_spell` and `worst_fixture_dead_recall`. **Failed checks: none for A; none for B.** The original support checks also passed. These are inner checks, not the unchanged development or final gates.

| Worst-fixture safeguard | Allowed minimum gain | A minimum gain | B minimum gain |
| --- | ---: | ---: | ---: |
| Spell F1 | -0.010000 | +0.094597 | +0.092944 |
| Stoppage recall | -0.005000 | +0.032188 | +0.028588 |

A's margins above those allowed minima are 0.104597 and 0.037188; B's are 0.102944 and 0.033588. Both therefore improved even on their weakest fixture for these two safeguards.

The complete short-possession diagnostic totals still show substantial remaining errors:

| Aggregate count over eight fixtures | A | B |
| --- | ---: | ---: |
| Predicted short spells | 530 | 547 |
| Matched predicted short spells | 76 | 90 |
| Unmatched predicted short spells | 454 | 457 |
| Reference short spells containing predicted stoppage | 158 | 142 |
| Reference short spells erased by predicted stoppage | 57 | 50 |
| False stoppage frames within reference short spells | 2,479 | 2,309 |

Mean per-fixture precision in the metric table differs from dividing pooled counts because the former weights each fixture equally. Low short-possession precision and recall remain clear weaknesses despite the gains. These counts are aggregate provider-reference diagnostics, not manual review of challenge output.

A was selected because its spell F1 ranks first under the frozen ordering. B has higher short-possession and stoppage precision, but changing the ranking after observing that tradeoff would invalidate the protocol. No blended third candidate, runner-up refit or further parameter search is authorized by this result.

The coordinator's local verifier authenticated actual artifacts, reconstructed the permitted inner references and recomputed selection in 63.281 seconds. The independent reviewer separately checked the 58 ledger events, stage seals, complete prediction-before-scoring order, all sixteen aggregate score records, means, gains, support, short-precision arithmetic, worst-fixture checks and selected identity. No reviewer agent opened arrays, model weights or provider records.

| Completed evidence | SHA-256 |
| --- | --- |
| Complete prediction freeze | `6f30515df8e1c42b39f62e81ed41468bf6199700680b965c41f76df361674275` |
| Selection | `cac10dba4a03e3839e92cef1dc7b062ec7f4c26b9aa56c36d62e75925739eb0c` |
| Inner ledger tip | `b8352b2d3073275e89700c217c917a7cbc420a6f78a97f03ad24c6bd5cd3838a` |
| Independent aggregate review file | `5d020830dd512b0888419544e942d3427d3563fa51191f145db63ebe3d2611f1` |
| Actual-artifact verification checkpoint | `89eeed5266c2193fd9e61ca016f3c4cbc9518c2695dd1f5fc46d9b56caa13976` |

Complete per-fixture metrics, exact commands, fits and audit receipts remain in the ignored local ledgers. Only non-reconstructive aggregate findings are published here.

## Winner refit and remaining limits

After independent admission, the single fixed A winner refit completed on the original forty PFF training fixtures plus seven clean Sportec fixtures. It retained the recipe, one-heavy-fit limit, one thread per worker and 1,800-second stage bound. The coordinator verified actual artifact bytes; a separate aggregate review checked the exact 47-fixture roster, exclusions, recipe, clean parent, frozen stoppage head, new joint fit and selection binding. The complete bounded operator now records thirty completed stages, including this sole refit, while retaining the earlier failed launch.

The refit model SHA-256 is `0481630d32bd17218c3e897af66a2d9babdace35e5ca33fe112a44730f96eaca`; its aggregate report SHA-256 is `68990945d282277534c975c20c193c5ad14dbdba3534cd9de1d39b95d9691625`. These weights remain local. No second candidate was refitted.

Refitting does not supply new validation scores or authorize development or final access. Connected lineage and a separately reviewed fixed-candidate development route must be admitted before the v1.5 comparison; all original development gates remain in force.

PFF fitted-model delivery rights and independent dense-reference validity remain unresolved. Event-derived provider-reference agreement must not be described as hidden challenge accuracy. PFF records, reconstructed references and fitted weights remain local. The existing repository stays private; this report contains only method descriptions, identities and non-reconstructive aggregate findings.

The separately prepared clean Sportec fallback is an eligibility fallback, not a demonstrated performance improvement over v1.5. Its preparation does not remove the hold on the historical release or authorize a public repository or competition submission.
