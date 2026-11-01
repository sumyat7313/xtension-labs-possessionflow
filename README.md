<p align="center">
  <img src="assets/banner.png" alt="PossessionFlow by Xtension Labs — football possession, frame by frame. More Than A Hack 2027 at Talent Arena. Organizers: Mobile World Capital Barcelona, Barça Innovation Hub and Metrica Sports." width="100%">
</p>

<p align="center">
  <a href="https://github.com/likalight/xtension-labs-possessionflow/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" alt="View GitHub Actions CI"></a>
  <a href="requirements-solution.txt"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="docs/refinement-verification-v8.json"><img src="https://img.shields.io/badge/tests-395%20verified-31b88a" alt="395 tests verified locally"></a>
  <a href="models/submission.json"><img src="https://img.shields.io/badge/inference-CPU%20only-0c7769" alt="CPU only inference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue" alt="MIT licensed code"></a>
  <img src="https://img.shields.io/badge/release-1.5.2-112b36" alt="Release 1.5.2">
</p>

<p align="center">
  <b>From player tracking to possession spells.</b><br>
  A reproducible football analytics pipeline built by <b>Xtension Labs</b> for More Than A Hack 2027.
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#results">Results</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#submission-artifacts">Submission</a> ·
  <a href="#documentation">Documentation</a>
</p>

## What is PossessionFlow?

PossessionFlow assigns H/A/D/X labels to every football video frame using automatic tactical tracking and event tags. Built by **Xtension Labs** for More Than A Hack 2027.

Research release **1.5.2** retains the original v1.5 submission and adds explicit boundary learning, contact-cue reliability checks, and a fresh two-match phase audit. **Retained the original v1.5 joint-state model and configuration. V7 adds completed research and reproducible evidence; phase-only fresh labels cannot validate stoppage or short-possession improvements or authorize replacement.**

## V8 research update

We obtained [corrected human event timestamps](research/annotations/elastic/README.md) and tested isolated precision and stoppage refinements. Development selection: **baseline-v1.5**. The original v1.5 deployment and v1.5.2 submission bundle remain unchanged; sufficient fresh dense validation is unavailable. See [the complete V8 results](docs/REFINEMENT_V8.md) and [the concrete data-access plan](docs/DATA_ACCESS_BRIEF_V8.md). 395 engineering tests passed in this research revision.

## Quick start

```bash
git clone https://github.com/likalight/xtension-labs-possessionflow.git
cd xtension-labs-possessionflow
python -m pip install -r requirements-solution.txt
python -m possession demo
```

Use Python 3.12 in a virtual environment. Follow [SOLUTION.md](SOLUTION.md) for real inference, tests and packaging. Restricted match assets stay local.

## Results

Development promotion gate: **fail**. Fresh phase regression gate: **fail**. Fresh stoppage/short-possession validation: **unavailable**. Deployed model: **original v1.5 joint-state**.

| Candidate | Spell F1 (primary) | Short recall | Short precision | D segment F1 | D boundary F1 | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-v1.5 | 0.6233 | 0.2784 | 0.3962 | 0.9163 | 0.7979 | incumbent |
| typed-boundary | 0.6222 | 0.2899 | 0.3849 | 0.9134 | 0.7966 | fail |
| typed-contact | 0.6221 | 0.2899 | 0.3846 | 0.9134 | 0.7966 | fail |


The development comparison uses complete held-out match folds within previously exposed data. Frozen experimental audit candidate: **typed-boundary**.

| Fresh phase match | Scored frames | v1.5 accuracy | Experiment accuracy | Change (percentage points) | v1.5 balanced recall | Experiment balanced recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| skillcorner1953632 | 32,739 | 86.90% | 86.83% | -0.076 | 86.59% | 86.52% |
| skillcorner1959846 | 29,798 | 86.30% | 86.29% | -0.010 | 86.30% | 86.29% |


Fresh phase accuracy cannot validate stoppage or short-possession gains. The fixed contact reliability audit also failed: only 12 of 39 clean and 5 of 20 calibrated restart candidates aligned; degraded/camera had no candidates. Physical-contact accuracy is unestablished and independent video remains inactive.

All seven Sportec matches are previously exposed development data; five matches are scored under four correlated observation conditions, not twenty independent matches. The two new SkillCorner matches have automatic phase labels, not dense independent H/A/D truth. Missing phases remain unknown, never dead-ball labels. They cannot establish stoppage boundaries, short-possession quality, or physical contacts. These audited sources are now consumed and cannot be reused as untouched final tests. No hidden competition score, human-gold validation, or winning probability is claimed.

## How it works

Tracking observations feed robust control and joint possession/stoppage learning. Bounded tactical evidence and duration-aware decoding produce possession spells. The v7 experimental model learns separate team-change, stop and restart evidence, optionally with eight observed-only geometric cues. [The full report](docs/BOUNDARY_CAMPAIGN_V7.md) distinguishes experiments from the retained deployment and records all fixed gates.

## Submission artifacts

The [v1.5.2 release package](https://github.com/likalight/xtension-labs-possessionflow/releases/tag/v1.5.2) contains the exact team-named CSV, technical report, source and evidence. 342 engineering tests passed; current clean-source verification is recorded in [verification](docs/verification.json). The repository remains **private under likalight**. The organizer's competition form is not submitted; access and eligibility remain outstanding.

## Documentation

- [V7 boundary, fresh phase and reliability results](docs/BOUNDARY_CAMPAIGN_V7.md)
- [Typed-boundary design](docs/TYPED_BOUNDARY_V7.md)
- [Contact/restart limitations](docs/CONTACT_RESTART_V7.md)
- [Fresh dataset provenance](docs/FRESH_DATA_V7.md)
- [Technical report](docs/TECHNICAL_REPORT.md)
- [Next evidence needed](docs/NEXT_EXPERIMENTS.md)
- [Previous v1.5.1 release](docs/releases/v1.5.1/README.md)

Code is MIT licensed. Dataset and organizer assets retain their respective terms and attribution; logos identify challenge organizers and do not imply endorsement.
