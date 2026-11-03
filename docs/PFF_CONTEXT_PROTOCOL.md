# PFF contextual false-stoppage refinement

This is a second, explicitly adaptive development family after the fixed global dead-logit penalty grid failed its strict gate. The previous grid, its failures and its full results remain preserved. This family does not expand that grid or change its thresholds after the fact. The same twelve development matches have already been consumed and tuned against; results on them are development selection, not fresh confirmation.

## Training-only residual

Fit one small contextual residual using exactly the original forty PFF training fixtures. The residual changes only dead-play probability while preserving conditional H:A probability and the existing decoder. No development or final references may enter fitting, feature scaling, example weighting or model normalization. The inherited PFF parent was already trained on forty PFF plus seven Sportec fixtures; this additional head uses only the forty PFF training fixtures.

The fixed head uses the existing 58 dead-play observation features, parent dead log-odds and absolute conditional team confidence. One 32-unit tanh hidden layer produces a bounded ±2 dead-logit residual. Fit for four epochs at 5 Hz with seed 20261103, AdamW learning rate 0.001, weight decay 0.01 and batches of 2,048 samples. Each training fixture has equal total sampled weight; unknown labels are excluded. Scaling uses training samples only. These are fixed engineering choices, not selected from development scores.

The training plan is frozen before fitting and records all scientific-source hashes, the complete model fit policy, the exact training IDs, the forthcoming blend grid, and the prior failed global-grid evidence hash. The fitted model's metadata binds that training plan. The subsequent immutable development protocol binds the fitted-head SHA-256, complete unchanged fit policy, sorted forty training IDs and all correction artifact hashes before new development scores are opened. A hash or membership declaration must be verified against the actual fitted artifact; the aggregate validation module cannot independently certify the fit's history.

The comparison parent remains the original PFF mixed model, SHA-256 `0ab133743c593466d8ae92ea94937e8c6bf22510f6b64053e3e69c7b467dcf3f`, rather than a selected global-penalty result. The production comparator remains v1.5.

## Fixed six-way blend selection

Evaluate the residual at these exact blend strengths, in order:

`0, 0.125, 0.25, 0.5, 0.75, 1.0`.

Zero blend must reproduce the original PFF parent predictions and per-match metrics. All six declared artifacts must complete evaluation on all twelve development fixtures; no result may be omitted to select a favorable subset. The artifact for each blend also binds the same fitted-head hash. No fitted weights or raw provider labels belong in the repository.

The gates are imported directly from the preceding precision study, unchanged:

- Every original v1.5 development requirement, including worst-fixture safeguards and reference support.
- Mean stoppage frame precision at least **0.002 above v1.5**, or **0.2 percentage points**.
- At least **90% of the original PFF parent's positive mean gain over v1.5** in spell F1, macro F1, short-possession precision and recall, stoppage segment and boundary F1, and stoppage recall. A nonpositive parent gain supplies a zero floor.

Among fully passing blends, choose the highest mean spell-F1 gain, then higher mean stoppage-boundary F1 gain, then earlier declared grid order. Development bootstrap intervals remain tuning diagnostics; they are not corrected for selection across either family and do not establish fresh superiority.

## Final access and deployment

The final twelve matches remain locked unless one blend passes every development check. Recompute selection and authenticate all model, protocol, source and development evidence before preparing final observations. Freeze predictions for v1.5, the original PFF parent and the single selected context blend on all twelve final fixtures before opening any final reference. An exclusive access marker consumes a failed or interrupted scoring attempt as well as a completed one. No second candidate may be chosen from final results.

The one final comparison applies the same original gates, precision margin and gain-retention requirements on the complete final cohort. Numerical success still does not establish independent dense truth or permitted distribution of PFF-trained models. Current references are provider-event-derived silver, and fitted-model submission rights remain unverified. Those unresolved evidence conditions require retaining v1.5 even if numerical checks pass. The validation helpers never replace production files.

`tools/pff_context_validation.py` implements the six-blend protocol and selection while reusing the original precision gate functions. Its focused tests cover unchanged gates, head and training-policy binding, exact fixture membership, fixed grid/artifact identities, protocol immutability, zero control, deterministic selection, failed development keeping final locked, and the exact selected-winner guard.

`tools/verify_pff_context.py --run runs/pff-context-v1 --write models/pff-context-research.json` verifies the actual local fitted head, training probability files, source observations and evaluation predictions, then writes aggregate evidence without fitted weights or frame labels. The default verifier independently checks the preceding failed scalar study, pre-fit plan, exclusive fit-access marker, forty-fixture preparation, sampled support, fitted-head provenance, six-blend selection and any final gate. It derives metric records from saved diagnostics and preserves the silver-reference and unresolved-rights declarations. Eight validation tests and twelve verifier tests exercise these controls, including tampered probabilities, source caches, heads, memberships, policies and score records. Synthetic passing tests do not establish real candidate performance.
