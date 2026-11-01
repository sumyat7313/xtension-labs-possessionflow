# Better labels for v8

We obtained a useful new annotation source: corrected human-timed events from ELASTIC. It improves development supervision, but all three underlying Sportec fixtures were already exposed. It cannot establish a fresh-match win over v1.5.

## Available now

[ELASTIC](https://github.com/hyunsungkim-ds/elastic) publishes corrected event and reception timestamps for J03WMX, J03WN1 and J03WPY. We pinned commit `e245224aad7450961eca40954ae2bc85e1e6e696`, verified Git blob sizes/hashes, and imported the annotated ground-truth tables rather than synchronized model predictions. The data is CC BY 4.0, with attribution to ELASTIC and the original Sportec/DFL dataset. [Attribution and normalized files](../research/annotations/elastic/README.md).

The intake accepted 3,560 of 3,561 source event rows, yielding 282 set-piece restart points, 150 explicit ball-out points and 2,790 reception/touch points. One temporally inconsistent row was quarantined without moving either timestamp. The frame IDs map directly to existing Sportec cache indices using each half's recorded start/end/base offsets.

These are sparse times, not possession durations. A receipt may be a one-touch action; a foul is not necessarily a whistle. Missing events do not imply D, and a point is not expanded to cover the gap until the next point. Actual repeated event IDs, phase boundaries and half separators remain explicit. All three match identities remain development-only even after re-annotation.

Use the annotations only from the outer training matches when fitting a fold. An evaluated match's labels may support a separate timing diagnostic after predictions are frozen, never input features. Evaluate D onset precision and recall separately from restart timing. With only three annotation-bearing fixtures, resampling points cannot manufacture independent matches or reliable confidence bounds on generalization.

## How to get a usable fresh labelled test

The most direct route is a permitted new synchronized match with full observed player/ball tracks and video, plus corrected dense possession and live/dead references. Reserve entire fixtures before model selection. If providers supply only events, request actual restart, whistle/ball-out and controlled-acquisition annotations with annotation methodology and timing uncertainty; sparse events still do not supply dense negative intervals.

| Source | What it adds | Current practical limit |
| --- | --- | --- |
| [PFF FC World Cup 2022 official release](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) | Paired broadcast tracking and events for all 64 fixtures; a substantially larger pool of identifiable unseen matches | Official access form required; no application submitted or mirror used to bypass it. The [Kloppy loader](https://kloppy.pysport.org/user-guide/loading-data/pff/) supports in-play filtering, but full live/dead coverage, ball observability, label provenance and permitted hackathon use must be verified after access. |
| [SoccerNet-GAR official dataset](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) | Paired video/tracking and reviewed actions across many fixtures | Access review/NDA required; event clips alone are insufficient dense H/A/D truth. No gated payload accessed. |
| [FOOTPASS official repository](https://github.com/JeremieOchin/FOOTPASS) | Ball-action annotations with video/tracking | NDA/password access. No application or acceptance submitted. |
| [SoccerTrack v2 official project](https://atomscott.github.io/SoccerTrack-v2/) | Full amateur matches with actions and game-state tracks | Access conditions remain unresolved. Auxiliary ball trajectories derived from annotations cannot independently validate those annotations. |
| [BEPRO official sample](https://bepro.ai/free-data-example) | Full-match player tracking and event sample | Request form needs personal information; not submitted. Independently observed ball and dense dead-state coverage still need verification. |
| [Alfheim official dataset](https://datasets.simula.no/alfheim/) | New-match video and home-team body-sensor positions; limited ball pixel annotations | Home-team sensors omit opponent tracking and goalkeeper in the documented analysis. Not a drop-in full H/A observation dataset or dense stoppage benchmark; requires substantial extraction and review. Non-commercial research terms apply. |
| [Collective states and their transitions](https://doi.org/10.1371/journal.pone.0251970) | Describes manually annotated possession on Alfheim video | Paper's data statement points back to Alfheim; independently released annotation files were not located. Do not represent the paper as an obtained label download. |
| [SkillCorner open data](https://skillcorner.github.io/opendata/) | Further unused broadcast-tracking fixtures | Automatic phases cannot establish dense stoppage/short-possession truth; another phase audit alone cannot authorize promotion. |

The searches did not obtain a new accessible dense H/A/D holdout. Existing Metrica, Sportec and WCSF samples and the seven already audited SkillCorner IDs remain consumed. No fresh label was parsed for this intake. Replacing folders, adding corruption or splitting another half does not restore freshness.

The most concrete next access request is PFF's full 64-match release, retaining all frames rather than loading only live play. [A ready-to-use project brief](DATA_ACCESS_BRIEF_V8.md) describes the required fields and validation plan. Access to paired tracking and events is a promising starting point, not proof that dense controlled-possession labels already exist.

## Build reviewed labels from permitted video

For a newly authorized synchronized match, use two reviewers blinded to all model predictions, then adjudicate disagreements. Follow the challenge's controlled-possession definition: during a loose/contested phase, retain the last controlled team; a deflection alone does not change ownership. Mark the actual cessation of play and actual restart separately. Annotate whole halves, including ordinary live play, rather than only model-proposed difficult clips. Keep uncertain timing margins unknown in research truth; this is annotation uncertainty, not an extra competition class. Record source video hash, exact decoded frame count/rate, half boundaries, reviewer identifiers, convention, and adjudication history.

`reviewed_intervals(document)` provides a fail-closed importer for this future workflow. It requires declared human origin, two distinct reviewers, prediction blinding, adjudication and a video hash; rejects overlap and cross-half intervals; keeps gaps and boundary-uncertainty margins X. It reports that human review is **declared, not independently verified**, and never grants promotion eligibility. A separate evaluation protocol must check real provenance, fixture independence and sufficient event support. Model-generated labels cannot be passed through this importer as human annotations.

## Reproduce intake

The published JSON points need only the solution's NumPy dependency. Rebuilding them from the original small Parquet files additionally needs PyArrow 26.0.0:

```powershell
python -m pip install pyarrow==26.0.0
python tools/data_intake_v8.py download-elastic --data-dir runs/fresh-data-v8/elastic
python tools/data_intake_v8.py compile-elastic --data-dir runs/fresh-data-v8/elastic --cache runs/multimodal-campaign-v4/cache --out runs/elastic-intake-reproduction
python -m unittest discover -s tests -p test_data_intake_v8.py -v
```

The original provenance-matched Sportec cache metadata is required for mapping to this project's timeline. Raw provider assets stay local. Tests cover source corruption, half/index alignment, sparse semantics, malformed chronology, annotation uncertainty and review/provenance restrictions.
