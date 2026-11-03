# Clean external candidate: PFF development results

**Do not promote.** The single fixed candidate failed the unchanged development gate. All fifteen stages completed and both historical controls reproduced exactly. The final twelve fixtures remain untouched; the original v1.5 model and CSV are unchanged and remain on submission hold.

## Complete comparison

These are means over the same twelve already-exposed PFF development matches, using unchanged provider-event-derived references and masks. They are not fresh-final results or hidden challenge scores. The candidate was trained on Sportec only; no fitting or calibration occurred in this evaluation.

| Metric | v1.5 | Clean Sportec candidate | Change |
| --- | ---: | ---: | ---: |
| Possession spell F1 | 0.435538 | 0.400247 | -0.035291 |
| Macro F1 | 0.832811 | 0.820819 | -0.011992 |
| Short-possession precision | 0.085490 | 0.074020 | -0.011469 |
| Short-possession recall | 0.106151 | 0.099676 | -0.006475 |
| Stoppage segment F1 | 0.706251 | 0.608086 | -0.098164 |
| Stoppage boundary F1 | 0.339145 | 0.305609 | -0.033537 |
| Stoppage precision | 0.933314 | 0.890371 | -0.042943 |
| Stoppage recall | 0.890216 | 0.911922 | +0.021706 |

All values use the 0–1 scale. Stoppage precision fell **4.2943 percentage points** while stoppage recall rose **2.1706 points**. Spell F1, stoppage segment F1 and stoppage precision declined on all twelve fixtures; recall increased on all twelve. This consistent tradeoff indicates a transfer problem and excess false stoppages, rather than an isolated bad match. The prior Sportec improvement over a new control head did not establish transfer to PFF.

Failed checks: dead_precision_margin, macro_guard, no_target_regression, positive_bootstrap_lower, retain_parent_gains, spell_gain, targeted_gain, worst_fixture_spell. All seven 90%-gain-retention checks failed. No threshold, reference mask, candidate or cohort was changed in response.

An independent count audit of the thirty-six authenticated aggregate metric files found **47,831 → 84,959 false stoppage frames** (+37,128, or 77.6%), versus **704,009 → 720,803 correctly predicted stoppage frames** (+16,794), over the same 1,808,107 graded frames. These pooled counts diagnose the error cost; they are distinct from the equal-match averages used by the gates. They do not identify which architectural component caused the regression.

## Per-fixture outcomes

Changes below are percentage points. Every fixture is reported.

| Fixture | Spell F1 change | Stoppage precision change | Stoppage recall change |
| --- | ---: | ---: | ---: |
| 3813 | -3.148 | -1.750 | +1.588 |
| 3816 | -0.308 | -2.534 | +1.622 |
| 3818 | -1.611 | -4.157 | +2.510 |
| 3823 | -7.568 | -6.319 | +2.215 |
| 3831 | -2.265 | -4.212 | +4.205 |
| 3844 | -4.810 | -2.806 | +1.195 |
| 3850 | -3.642 | -5.161 | +1.934 |
| 3854 | -3.446 | -4.767 | +0.899 |
| 3858 | -2.424 | -5.775 | +2.705 |
| 10502 | -0.556 | -2.631 | +2.462 |
| 10514 | -7.231 | -6.670 | +2.488 |
| 10517 | -5.340 | -4.748 | +2.222 |

## Verification and limitations

**720 engineering tests passed locally** in 127.059 seconds before this run. The separate local verifier authenticated actual model/cache/prediction/scoring artifacts, recomputed gates, bound metrics to detailed diagnostics and confirmed exact historical control/support reproduction. An independent agent reviewed the completed aggregate comparison and observed the completed prediction cohort before reference access. All fifteen consumed stages remain recorded in the hash-chained development ledger.

Candidate SHA-256: `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`.

Protocol canonical digest: `9be74addb409f341f5fcfb9c9b0e49b246600dca333bd857e61e13778582a080`.

[Complete aggregate evidence](../models/external-pff-development-research.json) includes all records and failed checks. Restricted records, frame predictions and fitted artifacts remain local. [Study method and verifier commands](EXTERNAL_PFF_DEVELOPMENT.md).

PFF labels remain provider-event-derived silver references. Reference quality and applicable usage evidence are separate from numerical results. This study cannot authorize final access or submission replacement, including if its numerical result had passed. The historical v1.5 comparator remains on its independent provenance hold.

## Next bounded advance

Keep this negative result and stop this candidate comparison. Before another fit, diagnose the clean model on the original forty PFF **training** matches only, recomputing upstream probabilities from clean heads. Compare the fixed control/dead heads with and without the fixed joint residual, using identical configuration and reference masks. Examine false-stoppage confidence and observation coverage to distinguish upstream and residual errors; reference event distance may be a diagnostic grouping only, never an inference feature. Then predeclare a small external-data adaptation experiment with whole-match training/validation separation inside that training partition. Retain all twelve used development fixtures as development and preserve the twelve final fixtures. No parameter search on the final set, mask changes to recover passing scores, or lowering of the original precision/retention gates is permitted.

Any PFF-fitted artifact stays local until its delivery rights are established. The next experiment has not been implemented or run at this checkpoint.
