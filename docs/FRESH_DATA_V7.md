# Fresh data and validation scope for v7

The v7 boundary/contact experiment has **no new dense stoppage or short-possession validation source**. The existing submission must remain deployed. A successful phase-only audit cannot promote this candidate. The user chose to use available data and retain the submission when validation is insufficient.

## Reserved fresh audit

SkillCorner matches **1953632 and 1959846** are the next two unused IDs in the pinned public catalog, selected without inspecting their labels. Previously consumed IDs are 1874553, 1886347, 1899585, 1925299 and 1927964. All seven Sportec games, Metrica games 1–3 and the WCSF2023 match are already exposed and cannot become fresh again by changing a split or folder.

Source: [SkillCorner Open Data](https://github.com/SkillCorner/opendata), commit `4340d274572876239c154c90bc507a9b3250a656`. Exact Git-blob identities for metadata and phases, plus Git-LFS SHA-256 and byte sizes for tracking, are embedded in `tools/fresh_benchmark_v7.py`. Downloads verify content identities without parsing labels. Runtime manifests also record SHA-256 for every file.

The [official schema](https://skillcorner.github.io/opendata/#tracking-data-description) documents video frames at **10 Hz**, metric coordinates centered on the pitch, and detection flags. The existing validated adapter uses `x_normalized = .5 + x / pitch_length`, `y_normalized = .5 - y / pitch_width`. Only detected player/ball XY, frame, period and static team/pitch metadata enter inference. Extrapolated positions, embedded possession, dynamic events, playing-time metadata and phase references are excluded. Speed calculations reset at half boundaries. Native frame rate passes directly to inference.

We inspected the complete schema of **already consumed development match 1874553**, not fresh labels: 60,301 rows have `frame`, `timestamp`, `period`, `ball_data`, `possession`, `image_corners_projection`, `player_data`. Ball data contains only `x`, `y`, `z`, `is_detected`; possession contains only `player_id`, `group`. There is no explicit alive/dead or ball-state field. Null possession is **unknown**, not dead. The provider's phase product is automatic, not human ground truth. Although the [phase documentation](https://skillcorner.github.io/opendata/#phases-of-play-file) describes phases during play, absence alone is not sufficient evidence of stoppage: gaps, coverage failures and invalid intervals remain **X**. We do not manufacture D labels from phase gaps.

The fixed phase regression checks require nonnegative mean team accuracy change, nonnegative mean balanced team recall change, and no match accuracy drop greater than one percentage point. These are secondary checks only. `gate.passed` and `promotion_eligible` remain false regardless of phase scores; `phase_regression_passed` reports this limited check separately. Automatic phase intervals cannot establish reliable sub-two-second possession boundaries, contact timing, restart timing or stoppage recall.

Before access, the tool freezes the split, policy, model/config hashes, the local scientific source closure and every input file hash. The source closure is all `possession/*.py` modules plus `tools/fresh_benchmark_v7.py`, `tools/fresh_benchmark.py` and `tools/independent_benchmark.py`; the `scientific_hashes()` function exposes this exact set for release verification. Packaging and unused trainers cannot change inference/scoring and are excluded. A one-shot marker is written before loading tracking JSON that contains ignored possession fields. All four candidate/baseline prediction CSVs are written and hashed before separate phase parsing. Source/artifact/data changes fail closed; interrupted access remains consumed. A new directory does not restore freshness. The audit is run once for the development-selected candidate even when development rejects it; no runner-up is tested on these references. Remaining SkillCorner matches are reserved for future work.

## Search for better labels (30 October 2026)

| Primary source | Useful evidence | Access / suitability result |
|---|---|---|
| [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) | Synchronized player/ball tracking and annotated ball actions, including OUT and THROW IN; official card provides timing and split details. | Both current canonical and deprecated SNGAR releases require access review and a personalized NDA. No request submitted, terms accepted or gated payload downloaded. Event-centered clips also cannot automatically be treated as dense H/A/D truth. |
| [FOOTPASS official repository](https://github.com/JeremieOchin/FOOTPASS) | Player-centric ball-action annotations, tracking and broadcast video; promising contact/restart supervision. | Official instructions require an NDA/password for dataset extraction and video access. Not obtained or bypassed. |
| [SoccerTrack v2 official project](https://atomscott.github.io/SoccerTrack-v2/) | Ten full amateur matches with player tracks and 12 ball-action classes, including OUT, THROW IN and FREE KICK. | Official sharing/access conditions have not been accepted. No gated payload accessed through another mirror. Auxiliary ball trajectories interpolated from action annotations must not be used as independent ball observations to validate those same actions. |
| [BEPRO free data example](https://bepro.ai/free-data-example) | Official full-match on-ball events and 25 Hz player optical tracking sample. | Requires a personal-information request form (including name, email, phone, team and role). No form submitted. Ball coverage and annotation provenance would need checking after authorized access. |
| [SkillCorner Open Data](https://skillcorner.github.io/opendata/) | Immediately available independent broadcast observations with automatic possession phases. | Appropriate only for a limited cross-match phase regression test. No explicit dense dead-ball labels found in the documented/schema-checked tracking product. |

No third-party prediction, interpolated ball coordinate, absent event or missing possession label is relabelled as human ground truth. A stronger future evaluation requires independently annotated stoppage onset/restart times and short possession intervals aligned to raw observed tracking or video, split by whole match, with enough distinct short stoppages to estimate uncertainty. Authorized event data can supplement supervision, but sparse actions must retain unknown intervals rather than be silently expanded into dense labels.

## Reproduction

```powershell
python tools/fresh_benchmark_v7.py download --data-dir runs/fresh-data-v7
python tools/fresh_benchmark_v7.py evaluate-pair --data-dir runs/fresh-data-v7 --model PATH_TO_FROZEN_CANDIDATE --baseline-model PATH_TO_RELEASED_BASELINE --config PATH_TO_FROZEN_CONFIG --baseline-config PATH_TO_BASELINE_CONFIG --out runs/fresh-audit-v7
python -m unittest tests.test_fresh_benchmark_v7
```

The second command consumes this holdout. It must run only after candidate selection and all code changes are frozen. The resulting report is research evidence, not permission to replace the submitted predictions.
