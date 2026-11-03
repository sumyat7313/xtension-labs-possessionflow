# PFF context correction results

**Retain v1.5.** No context setting passed all development checks. The final 12 matches remain untouched. Production inference, the model, configuration and 199,319-row submission CSV remain unchanged.

## Two reported development families

The [first study](PFF_PRECISION_RESULTS.md) tested eight fixed global D-logit penalties. It reduced false stoppages, but no setting met both the precision margin and retention of every required gain. That complete failed result was preserved before this adaptive follow-up. No criterion was relaxed.

This second family fitted one small D-only residual model on the original 40 PFF training fixtures. It uses 60 team-invariant inputs: 58 existing observation features, frozen parent D log-odds and absolute conditional home/away confidence. The head has 32 tanh hidden units and a bounded ±2 logit output. It can raise or lower D confidence from context while preserving conditional H:A odds and the original decoder.

The recipe was frozen before fitting: seed 20261103, four epochs, AdamW learning rate .001, batches of 2048, known labels sampled at 5 Hz and equal sampled fixture mass. Normalization is learned from training rows only; unknown and unsupported rows are excluded. The actual fit used 1,164,844 sampled rows. The parent model had already been trained on 40 PFF plus seven Sportec fixtures; only the 40 PFF training fixtures fit this correction. Fitted weights remain local.

Training-only diagnostics and rationale are in [the false-stoppage audit](PFF_FALSE_STOPPAGE_DIAGNOSTICS.md). The full fit plan binds those source files, training identities and the previous failed study. The fitted head hash and six blends (`0, .125, .25, .5, .75, 1`) were frozen before any context development scores. [Context protocol](PFF_CONTEXT_PROTOCOL.md).

## Development selection

These twelve fixtures have already been used for development and tuning. They are not a fresh confirmation cohort. All six blends must be scored before selection. Acceptance keeps the original v1.5 gates, requires at least 0.2 percentage-point higher mean stoppage precision than v1.5, and retains at least 90% of the original PFF candidate's positive mean gains in seven other metrics. Zero blend must exactly reproduce the original parent.

| Blend | Spell F1 | D precision | D recall | D boundary F1 | Short precision | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 0.5317 | 0.9268 | 0.9624 | 0.5442 | 0.1428 | False |
| 0.125 | 0.5314 | 0.9281 | 0.9625 | 0.5481 | 0.1419 | False |
| 0.25 | 0.5328 | 0.9299 | 0.9624 | 0.5452 | 0.1418 | False |
| 0.5 | 0.5337 | 0.9319 | 0.9625 | 0.5561 | 0.1456 | False |
| 0.75 | 0.5335 | 0.9331 | 0.9623 | 0.5583 | 0.1462 | False |
| 1 | 0.5343 | 0.9336 | 0.9618 | 0.5533 | 0.1432 | False |


Failed checks by setting:

- Blend 0: dead_precision_margin, no_target_regression.
- Blend 0.125: dead_precision_margin, no_target_regression, retain_parent_gains.
- Blend 0.25: dead_precision_margin, no_target_regression.
- Blend 0.5: dead_precision_margin, no_target_regression.
- Blend 0.75: dead_precision_margin, no_target_regression.
- Blend 1: dead_precision_margin.

Selected setting: **none**. Development gate passed: **False**. No post-result threshold changes or unreported candidates were used.


## Scope and deployment decision

All metrics measure agreement with carried PFF event-derived references, not independently adjudicated dense labels or the hidden challenge score. Unknown intervals can truncate short spells. Team overlap and one tournament limit generalization; development bootstrap intervals after tuning are diagnostic rather than fresh significance claims. Independently reviewed reference quality and permission to distribute PFF-trained models remain unverified. The numerical and evidence gates are separate.

Final matches evaluated in this study: **0**. No fitted PFF weights, raw tracking records or frame references were published. The private repository contains original code, aggregate results, arbitrary blend settings and hashes.

## Verification

**580 engineering tests passed locally.** Tests cover deterministic fitting, Torch/NumPy parity, team invariance, whole-fixture exclusion, unknown handling, equal fixture weighting, unchanged zero controls, tamper detection and final-access guards. Synthetic test scores are not model-performance evidence.

The research verifier binds the pre-fit plan, training probability caches, exact training membership and supports, fitted head identity, complete development grid and any one-shot final audit. Local verification authenticates the actual model/cache/prediction bytes; portable CI recomputes evidence consistency without restricted data or weights. Original release and earlier research verifiers remain valid.

```powershell
python tools/verify_pff_context.py
python tools/verify_pff_context.py --run runs/pff-context-v1
python tools/verify_pff_precision.py
python tools/verify_pff_research.py
python tools/verify_release.py
```

Reproduction needs the authenticated local PFF parent and observation caches. Run `tools/pff_context_prepare.py` for training-only parent probabilities, choose a fresh output, initialize the immutable training plan, fit once, predict all declared development blends, and score once. Every final entry point independently rechecks passing development evidence. Interrupted scoring stays consumed. Reproduction does not make previously inspected fixtures fresh again.

Study protocol SHA-256: `2bfd883b2bfa0107c591e3363dabd9336287a8d8ba4ba9a5ed189b9eaf7d8951`.

Local fitted head SHA-256: `41e4f239be88fa69b9ed4dac899a55e4dfaabc74858b932df407cff1847279de`.

Retained deployment SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.
