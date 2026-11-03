# Training-only false-stoppage diagnosis

The fixed PFF joint-head candidate was inspected on its 40 training fixtures, using the same authenticated observation caches and provider-derived reference reconstruction as the original campaign. This is an in-sample diagnostic, not fresh validation. No development or final reference was read by this utility. The current v1.5 submission was not modified.

“Added false dead” means a known live reference frame classified D by the PFF candidate but not by v1.5. “Recovered true dead” means a reference-D frame correctly classified D by the PFF candidate but missed by v1.5. Unknown references are excluded. These terms describe agreement with carried provider-event references, not independently adjudicated human truth.

Across the 40 matches, the candidate added **75,408 false-dead frames** and removed **58,298** previous false-dead frames. It recovered **200,760** reference-D frames missed by v1.5; the net increase in correctly predicted D frames was **193,518**. Totals are frame-pooled and must not be confused with mean-per-match development metrics.

| Observation condition | Added false-dead frames | Recovered true-dead frames |
| --- | ---: | ---: |
| Missing-ball support | 70.56% | 70.92% |
| Observed ball within 1.5 m of a player | 16.02% | 15.54% |
| Observed ball speed at least 2 m/s | 15.14% | 10.85% |
| Raw ball observed | 30.99% | 30.20% |

Missing-ball support and proximity to a player occur at similar rates in both groups. They do not justify a blanket missing-ball or contact veto. Moving observed-ball evidence covers only a small portion of the errors and also appears in recovered true-D frames.

Emission confidence separates these groups more clearly: **44.10%** of added false-D frames have P(D) below 0.6, versus **20.01%** of recovered true-D frames. At 0.7 the fractions are **59.85%** and **30.62%**. These emission bins are descriptive; the temporal decoder means they are not predicted outcomes of applying a probability threshold.

Of added false-D frames, **55.66%** are outside the attributable two-second windows around direct live/dead reference transitions, **16.86%** are within two seconds after a direct restart, and **28.78%** are within two seconds before an onset. The near-onset and near-restart regions can overlap in short live intervals. Unknown gaps and period boundaries stop attribution, so “far” does not establish an exact causal timing error.

Of 3,005 predicted dead segments containing false-dead frames, 2,465 also overlap reference-D frames and 540 do not. Among all false-dead frames, 81,585 occur in contiguous error runs lasting at least five seconds. A restart-only correction would therefore leave substantial error coverage unexplained.

The first bounded experiment should adjust D confidence conservatively while preserving the joint head's conditional H:A odds and the existing decoder. The separate development protocol must decide whether such calibration improves precision without sacrificing gains; these training counts cannot authorize selection or promotion. No new contact/restart rule is justified solely by these aggregates.

[Machine-readable aggregate evidence](pff-false-stoppage-diagnostics.json) records exact source/model/protocol identities. The full per-training-match aggregate report remains ignored at `runs/pff-precision-v4/training-diagnostics.json`. No raw records, frame labels or fitted weights are included.

Four engineering tests cover unknown gaps, period boundaries, added/removed/recovered error accounting, JSON serialization and exact aggregation. Reproduce with `python tools/pff_false_stoppage_diagnostics.py`; this tool accepts only the frozen training partition.
