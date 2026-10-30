<p align="center">
  <img src="assets/banner.png" alt="PossessionFlow by Xtension Labs — football possession, frame by frame. More Than A Hack 2027 at Talent Arena. Organizers: Mobile World Capital Barcelona, Barça Innovation Hub and Metrica Sports." width="100%">
</p>

<p align="center">
  <a href="https://github.com/likalight/xtension-labs-possessionflow/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" alt="View GitHub Actions CI"></a>
  <a href="requirements-solution.txt"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="docs/verification.json"><img src="https://img.shields.io/badge/tests-290%20verified-31b88a" alt="290 tests verified locally"></a>
  <a href="models/submission.json"><img src="https://img.shields.io/badge/inference-CPU%20only-0c7769" alt="CPU only inference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue" alt="MIT licensed code"></a>
  <img src="https://img.shields.io/badge/release-1.5.1-112b36" alt="Release 1.5.1">
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

Release **1.5.1** includes the complete 199,319-frame submission and the v6 stoppage/short-possession experiments. **Retained v1.5: experimental combined did not pass both fixed gates (development=False, fresh=False).**

## Quick start

```bash
git clone https://github.com/likalight/xtension-labs-possessionflow.git
cd xtension-labs-possessionflow
python -m pip install -r requirements-solution.txt
python -m possession demo
```

Use Python 3.12 in a virtual environment. For real match inference, tests and packaging, follow [SOLUTION.md](SOLUTION.md). Restricted match assets stay local.

## Results

Development gate: **fail**. Fresh audit gate: **fail**. Deployed model: **v1.5 joint-state**.

| Candidate | Spell F1 (primary) | Short recall | Short precision | D segment F1 | D boundary F1 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| combined | 0.6216 | 0.2970 | 0.3978 | 0.9031 | 0.7947 | fail |
| transition | 0.6203 | 0.2912 | 0.3810 | 0.9163 | 0.7979 | fail |
| stoppage-loss | 0.6239 | 0.2877 | 0.4163 | 0.9031 | 0.7950 | fail |
| baseline-v1.5 | 0.6233 | 0.2784 | 0.3962 | 0.9163 | 0.7979 | incumbent |


Development uses whole-match folds on previously exposed Sportec data. It is not a hidden competition score. Short-possession precision is checked as well as recall to limit fragmentation.

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

## How it works

Tracking observations feed robust control and half-local joint possession/stoppage learning. Bounded event evidence and a duration-aware decoder turn frame probabilities into possession spells. The new optional transition module allows shorter team spells only with strong local reversal evidence. See the [v6 report](docs/TEMPORAL_CAMPAIGN_V6.md) for active/rejected candidates and all fixed gates.

## Submission artifacts

Download [release 1.5.1](https://github.com/likalight/xtension-labs-possessionflow/releases/tag/v1.5.1) for the exact team-named CSV, technical report and source bundle. 290 local tests pass; clean-source output reproduction is recorded in [verification](docs/verification.json). The repository stays **private under likalight**. The competition form is not submitted; organizer access and eligibility remain outstanding.

## Documentation

- [V6 experiment and fresh results](docs/TEMPORAL_CAMPAIGN_V6.md)
- [Fresh dataset provenance](docs/FRESH_DATA_V6.md)
- [Stoppage training design](docs/STOPPAGE_RESEARCH_v6.md)
- [Short-possession design](docs/SHORT_SPELL_RESEARCH_v6.md)
- [Next experiments and remaining evidence gaps](docs/NEXT_EXPERIMENTS.md)
- [Technical report](docs/TECHNICAL_REPORT.md)
- [Previous v1.5 results](docs/releases/v1.5/README.md)

Code is MIT licensed. Dataset and organizer assets retain their respective terms and attribution; logos identify the challenge organizers and do not imply endorsement.
