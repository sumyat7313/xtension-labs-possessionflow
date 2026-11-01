# V8 precision and stoppage refinement

The released v1.5 model, configuration, production inference and 199,319-row submission are unchanged. This isolated research study acquired better sparse human annotations and evaluated three refinements. It cannot establish a fresh-match win: no new independently labelled dense H/A/D holdout was available, and no unused phase-only match was consumed as a substitute.

Development selection: **baseline-v1.5**. Frozen experimental candidate: **combined**. Deployment promotion: **no**.

## What changed

The precision head learns whether a predicted short H/A island can safely merge into its flanking team, using posterior consistency and observed geometry. The frozen 0.8 score threshold, observation gates and single-pass conflict handling preserve D/X and avoid duration-only deletion. Human touch/reception conflicts exclude uncertain training examples; a touch is not automatically possession. Unsupported training classes produce an explicit no-op model.

The stoppage head relocates existing stop/restart boundaries within two seconds. It preserves their ordering, number and minimum-duration constraints, uses separate supervised ranking heads and abstains on insufficient observations. Validated sparse human times refine unambiguous training targets only; missing events never become negative or dense labels. The combination applies precision first, then stoppage refinement. Production inference imports neither experimental module.

## Better annotations obtained

The pinned [ELASTIC benchmark](https://github.com/hyunsungkim-ds/elastic) contributes 3,560 accepted corrected event rows on three already exposed Sportec matches, including 282 set-piece restart, 150 ball-out and 2,790 reception/touch points. One inconsistent event was quarantined. Source identities, native half/frame mapping, CC BY 4.0 attribution and normalized annotations are included in [research/annotations/elastic](../research/annotations/elastic/README.md). These are provider-reported median-of-three human timestamps; individual rater uncertainty is unavailable.

In the final all-seven research refit, the stoppage learner used 357 human stop-target groups and 666 human restart-target groups. These counts include repeated observation conditions and are **not distinct independent events**. The precision learner excluded 3 conflicting training examples. See [label provenance](LABELLED_DATA_V8.md), [precision design](PRECISION_V8.md) and [stoppage design](STOPPAGE_V8.md).

## Fixed development comparison

Four procedures were evaluated in 20 cells each: five held-out matches and four correlated observation conditions. Every upstream model and refinement fit excludes its complete outer evaluation fold. The refinement learner's upstream predictions within its training matches are in-sample, which limits calibration claims. Evaluation labels and sparse points are never passed to inference. All 20 incumbent output hashes exactly reproduce the original v1.5 out-of-fold predictions.

| Procedure | Spell F1 | Short precision | Short recall | D segment F1 | D boundary F1 | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| combined | 0.6254 | 0.4006 | 0.2801 | 0.9154 | 0.8018 | fail |
| stoppage | 0.6248 | 0.3989 | 0.2801 | 0.9154 | 0.8018 | fail |
| precision | 0.6239 | 0.3978 | 0.2784 | 0.9163 | 0.7979 | fail |
| baseline-v1.5 | 0.6233 | 0.3962 | 0.2784 | 0.9163 | 0.7979 | incumbent |


Spell F1 averages clean/degraded/camera conditions. Other columns average all four conditions, including calibrated corruption. Short spells are under two seconds; precision uses global one-to-one matching before stratifying predicted durations. Stoppage segment matching uses IoU 0.5; boundary F1 uses a one-second tolerance.

The policy requires at least 0.005 primary spell-F1 gain; nondecreasing short precision/recall and stoppage segment/boundary F1/precision; bounded worst-cell regressions; and at least 0.01 short-precision or D-boundary improvement. Its lower 5th-percentile bootstrap gain must be positive. Resampling enumerates all 3,125 resamples of the five matches, after averaging their correlated conditions. This diagnostic cannot remove prior development exposure or selection optimism; adequate fresh full-match evidence is separately required. No thresholds changed after scoring.

## Corrected human timing diagnostic

| Procedure, clean observations | Restart points within 1s | Ball-out points within 1s |
| --- | ---: | ---: |
| combined | 261/282 (92.55%) | 146/150 (97.33%) |
| stoppage | 261/282 (92.55%) | 146/150 (97.33%) |
| precision | 260/282 (92.20%) | 144/150 (96.00%) |
| baseline-v1.5 | 260/282 (92.20%) | 144/150 (96.00%) |


This table uses only clean observations on the three annotated held-out matches, avoiding repeated-corruption event counts. Restart matching requires the correct team; all matching stays within the half and is one-to-one within one second. Sparse references do not enumerate every true stop: precision, F1, dense possession quality and fresh-match status remain unavailable. Reported timing errors elsewhere refer to the official matcher's chosen alignment, not minimum possible latency.

## Verification and reproduction

395 engineering tests passed. The all-seven experimental refit reproduced identical model bytes. The existing release verifier still authenticates the unchanged v1.5 deployment and prior v1.5.2 release evidence. The additional research verifier checks source/protocol hashes, all fold models, sparse-label scope, annotation identities, candidate bytes and the no-promotion decision.

```powershell
python -m pip install -r requirements-training.txt
python tools/train_refinement_v8.py --workers 2 --out runs/refinement-v8-reproduction
python tools/verify_release.py
python tools/verify_refinement_v8.py
python -m unittest discover -s tests -v
```

Training requires the original checksum-matched Sportec caches and historical out-of-fold models; raw restricted tracking assets remain local. The published sparse annotation JSONs need no Parquet engine. The v8 research is reproduced from this repository, not the earlier v1.5.2 source archive. One preparation run was stopped before any refinement fit or score to fix a future reviewed-label importer's reviewer-list validation; the hypotheses and scoring thresholds did not change.

## What would make a replacement defensible

Use the [data-access brief](DATA_ACCESS_BRIEF_V8.md) to request PFF's paired 64-match World Cup release or another permitted synchronized source. Confirm all-frame observation coverage, actual live/dead and possession annotation semantics, and hackathon-use rights. If dense truth is absent, two blinded reviewers should annotate complete unseen halves and adjudicate disagreements. Reserve entire fixtures before any new tuning. The reviewed-interval importer preserves unknown uncertainty margins and validates provenance declarations; it cannot certify that review actually happened.

Protocol SHA-256: `5294556287c49f0da9eac52164a1815e53e44bf37429b734fd017b3411b1ea4f`.

Experimental artifact SHA-256: `a7dc4c9020dfd40e08efb95ce8b5cfd27375608baeebae82f851bd04cf2f97b8`.

Active model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.
