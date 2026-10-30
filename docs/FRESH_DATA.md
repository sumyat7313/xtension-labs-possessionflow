# Fresh-match evaluation for v5

The release audit is now complete. The frozen joint-state candidate improved
team accuracy on all three reserved matches and passed the predeclared gate.
These references have been consumed and are no longer untouched. The split and
reservation descriptions below document the procedure before that final audit;
see [the measured outcomes](JOINT_CAMPAIGN.md) and `models/fresh-audit.json`.

The seven Sportec matches, Metrica Games 1 and 2, and earlier corruption regimes
are already exposed. They are **not** fresh tests. The v5 audit reserves three
previously unused matches and evaluates one frozen candidate against v1.4.
None of these new matches is used for training.

This audit does **not** supply fresh human-verified dense H/A/D ground truth.
It measures team classification on provider event points and automatic phase
intervals. In particular, it cannot establish independent stoppage recall,
short-possession recovery, or the competition's spell F1.

## Sources and split

| Source | Match | Role | Reference quality |
| --- | --- | --- | --- |
| SkillCorner | 1874553 | Adapter development only | Automatic provider phase labels |
| SkillCorner | 1886347 | Reserved final paired audit | Automatic provider phase labels |
| SkillCorner | 1899585 | Reserved final paired audit | Automatic provider phase labels |
| Metrica | Sample Game 3 | Reserved final paired audit | Provider-supplied sparse pass events |

The split was saved before parsing reference contents in
`runs/fresh-data-v5/split-before-reference-access.json`. Only the development
match's phase file has been examined during adapter implementation. Game 3's
observation-only metadata/tracking were checked without opening its event file.
The final audit creates an exclusive consumption marker and cannot be repeated
as an untouched test. After it runs, all three reserved matches are exposed.

### SkillCorner

[Official repository](https://github.com/SkillCorner/opendata), pinned to commit
`4340d274572876239c154c90bc507a9b3250a656`. Credit: **SkillCorner and PySport**.
The repository carries an MIT license and requests attribution. Raw files remain
local and are not redistributed with this project.

Tracking uses native 10 Hz, pitch-centred metre coordinates. The adapter converts
detected player/ball XY observations to normalized coordinates, flipping the
upward-positive source y axis to Metrica's downward-positive normalized axis, and derives speeds
within each half. Extrapolated positions are discarded. A strict field whitelist
excludes embedded possession, phase, ball-state, event, and metadata playing-time
information from model inputs. Embedded possession fields occur in the tracking
JSON but are never accessed by the observation adapter.

The [provider's phase specification](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Phases%20of%20Play/20250216%20-%20Phases%20of%20Play%20CSV%20Specifications.pdf)
describes phases derived from tracking and machine-learning outputs. It also
retains the surrounding phase through some short disruptions. Therefore phase
agreement is a silver-label domain-shift check, not the organizer's possession
definition or independent human truth.

Only explicit phase intervals supply H/A references. End frames are exclusive,
matching the [official SkillCorner tutorial's Step 3](https://skillcorner.github.io/opendata/notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part7_Pitch_Maps_Tracking_by_Phase_Tutorial/),
which uses `IntervalIndex.from_arrays(..., closed="left")`; this was also checked
in the notebook at the pinned repository revision. Gaps remain X/unknown: **absence of a phase
is never converted to a dead-ball label**. Contradictory overlaps are excluded.
Intervals crossing a half or the observation range are excluded entirely; more
than 1% misaligned intervals abort scoring. This fixed rule follows one invalid
halftime-straddling interval found among 423 development intervals. That match
provides 32,722 phase-present frames after exclusion, out of 60,301 total frames.

### Metrica Sample Game 3

[Official repository](https://github.com/metrica-sports/sample-data), pinned to
commit `e706dd506b360d69d9d123d5b8026e7294b13996`. Credit: **Metrica Sports**.
The README invites community analysis and asks for attribution; no standard
license file is supplied. Raw data is not redistributed.

The source documents synchronized tracking/events but does not explicitly
establish the annotation method of this game. We therefore describe its labels
as **provider-supplied sparse events**, without claiming verified human annotation.

Game 3 uses 25 Hz FIFA/EPTS tracking, 143,761 frames and a 105 by 68 metre pitch.
The metadata explicitly declares normalized XY units. Observation-only range
checks confirm coordinates mostly lie within 0–1; small excursions beyond the
pitch are retained rather than clipped. No coordinate rescaling is needed.
Its per-range channel specifications account for substitutions. The adapter maps
those dynamic channels to global player slots, respects both halves and reads no
event file. Event parsing follows the published Metrica JSON schema described by
[the official Kloppy provider implementation](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/event/metrica/json_deserializer.py).

One reference is taken at each unique pass-origin frame for the completed,
incomplete, out and offside pass event types. Team and period must agree with the
tracking metadata. Duplicate same-team points collapse; contradictory points
are excluded. No intervals between events are filled.

## Fixed audit procedure and decision

1. Freeze candidate/baseline model and configuration hashes, source hashes, the
   split, and policy before accessing final data for inference. The candidate is
   the eligible development winner, or the highest-ranked experiment when none
   qualifies. This lets the fresh audit diagnose a failed experiment without
   authorizing its promotion.
2. Save both prediction CSVs for all three matches using the production dispatcher
   and observation-only inputs, with no tactical reference events.
3. Read the three separate event/phase files for scoring only after all six CSVs
   exist. D and X predictions at reference H/A frames count as errors.
4. Require Metrica exact-point accuracy and balanced team recall to be
   nondecreasing. Require equal-match mean SkillCorner phase-present accuracy to
   be nondecreasing, with no individual SkillCorner accuracy drop over 0.02.
5. Promotion requires both development gates and the fresh audit. On failure
   retain v1.4. Do not retune, retrain or try a runner-up using these
   audit references.

The policy's initial `human_*` key names were corrected to `metrica_*` before any
final reference access because manual annotation provenance was not verified.
Numerical gates did not change. Both policy versions and their hashes are
preserved locally in the provenance-correction record. A further pre-access prose
clarification permits auditing the highest-ranked experiment when no experiment
passes development gates; promotion still requires both stages, and numerical
fresh-match gates remain unchanged.

```bash
python tools/fresh_benchmark.py download
# Only after development selection and model/source freeze:
python tools/fresh_benchmark.py evaluate-pair --model runs/joint-campaign-v5/release-candidate-model.json --baseline-model models/baselines/v1.4.json --config runs/joint-campaign-v5/release-candidate-config.json --baseline-config configs/baselines/v1.4.json --out runs/fresh-audit-v5
```

Downloads are bounded and pinned by Git blob identity or LFS SHA-256, with local
SHA-256 manifests. Final-label files may be downloaded and hashed beforehand,
but are not parsed or scored before the frozen audit. The raw files and local
tracking caches stay under ignored `runs/`.

## Validation limits

Ten adapter/protocol tests cover reference-field exclusion, detected-only inputs,
native frame rates, duplicate observations, EPTS substitution channels, exact
pass timing, unknown phase gaps, overlap/misalignment handling, strict audit
gates, reservation guards, and saving all predictions before reference scoring.

SoccerTrack v2 remains a possible additional source, but its access requires
login and acceptance of information-sharing terms; no such consent was given
here. Its event-derived auxiliary ball trajectory also requires careful leakage
handling. Fresh dense human possession/stoppage labels remain unavailable in
the current accessible data, and the audit makes no claim to provide them.
