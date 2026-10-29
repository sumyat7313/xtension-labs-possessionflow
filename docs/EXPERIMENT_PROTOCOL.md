# Competitive evaluation protocol

Historical protocol for the initial frozen-heuristic benchmark. The subsequent
training and promotion protocol and its results are in
[TRAINING_CAMPAIGN.md](TRAINING_CAMPAIGN.md) and `models/selection.json`.

Recorded October 29, 2026, before evaluating the external matches below.

## Purpose

Establish whether the current method generalizes and identify where further work
has evidence behind it. Engineering correctness, external predictive performance,
and a competition ranking are three different claims. None implies the others.

The current competition submission remains the frozen baseline. Its CSV SHA-256
is `e8c36ef12681bb078d2d2d721fca2cc52b60623cba6881e32e864c79da1fb1f5`.
Its fixed configuration SHA-256 is
`8d190bec7b92ae61157f6b41fbc839bc6d400e6f626908d4a6d8bdacc7dd76cf`.
Audit, packaging, and benchmark additions do not justify changing predictions.

## Independent external benchmark

Source: the Sportec/DFL integrated dataset of seven other professional matches,
published at https://doi.org/10.6084/m9.figshare.28196177 under CC BY 4.0.
The adapter must verify its schema and label semantics before reporting scores.

- Development smoke-check match: `J03WOH`.
- Held-out diagnostic match: `J03WN1`.
- Evaluate the existing fixed configuration first, with tracking only. The external
  data does not provide the competition's automatic Smart Tagging input.
- Neither match is the hackathon match. Do not open the hackathon public labels in
  this benchmark, train on them, or select models from their scores.
- BallStatus and BallPossession are reference labels only. Do not use them to
  mask observations, select frame spans, define decoding blocks, recover the ball,
  compute features, or initialize state. Use recording/half metadata for blocks.
- Preserve dead-ball frames. Unknown reference labels stay unknown; do not create
  possession truth from a nearest-player heuristic.
- Apply the official possession metric to adapted labels. Report the mapping,
  excluded/unknown frames, scoring coverage, input hashes, and configuration.
- No parameter fitting or model selection is part of this first benchmark.

External labels can differ from the challenge's definition of controlled team
possession, and external tracking can be cleaner. Inspect documented definitions;
report that domain mismatch rather than treating an external score as the hidden
challenge score. Two matches cannot establish broad generalization.

## Subsequent development gates

1. Use additional external matches for development and retain untouched matches
   for final testing. Once a held-out result informs a change, that match is no
   longer untouched; do not present it as a fresh final test.
2. Compare principled candidates: separately learned live/dead evidence, stoppage
   persistence through ball retrieval, sustained control rather than one nearest
   player, and algorithmic ball recovery from local video.
3. Simulate observation degradation using explicitly recorded random seeds and
   synthetic missingness/noise on other matches. Do not choose corruptions to
   produce a preferred model ranking. Keep clean and degraded results separate.
4. Assess spellF1 first, then frame and transition metrics. Compare changes across
   matches and regimes; do not accept a gain caused by one unusually easy match.
5. Promote a candidate only after documenting independent evidence, all training
   sources and settings, failure cases, runtime, and reproducible model weights.
6. Regenerate the entire competition CSV algorithmically, validate the exact
   packaged file, and reproduce it from a clean source extraction.

No victory probability or ranking can be justified without the private evaluation
and competing entries. The correct deliverable is a stronger, independently tested
system with an honest record of what remains untested.
