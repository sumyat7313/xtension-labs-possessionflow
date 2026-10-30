# Contact and restart evidence v7

The implemented channel supplies conservative **geometric proxy features** for a
controlled learned ablation. It does not establish physical touches or reliable
restart events. A fixed development audit failed its reliability requirement, so
hard event overrides and independent video evidence remain inactive.

## Interfaces and observation gates

`possession.contact_evidence.feature_matrix(X, Z, blocks, fps)` returns eight
finite features in `[0, 1]`, using the existing 98 control and 58 dead-ball
observation columns. It accepts no possession reference, provider state, event
table, or target event label. The features are:

| Feature | Meaning |
| --- | --- |
| `observed_context` | Complete observed ball/speed context around the timestamp |
| `uncontested_contact` | Existing velocity-change control proxy near an unambiguous team |
| `uncontested_deceleration` | Existing deceleration control proxy under the same gates |
| `stationary_launch` | Low preceding ball speed followed by a fast departure |
| `boundary_stationary_launch` | Stationary launch weighted by proximity to the pitch edge |
| `outside_return` | Decreasing outside support followed by a fast departure |
| `player_motion_launch` | Stationary launch accompanied by increasing player motion |
| `stationary_near_ball` | Observed stationary ball near an unambiguous team |

Contact context requires every observation in a full +/-0.2-second window.
Restart context requires every observation from -0.6 to +0.2 seconds. The ball
must be marked observed, have full observation reliability, and have known speed.
Interpolation never satisfies these gates. Missing context, gaps, and half edges
abstain. The covariates are unchanged by swapping home and away.

The adapter decodes only documented, invertible speed scaling and unsaturated
near-distance ranges. It never invents native XY, ball height, exact large
distances, or individual player identities from normalized model matrices.
The team-distance margin gate uses the existing reliability-weighted margin as
a conservative support test. Feature magnitudes are **not calibrated confidence
probabilities**. Smoothed outside support remains a geometric covariate.

`extract_native(tracking, blocks, pitch, fps)` is a separate raw-XY candidate
extractor. It ignores supplied ball/player speed fields and requires observed
raw ball coordinates throughout its support, physically plausible consecutive
steps (at most 45 m/s), a player within 1.5 m, and at least a 1 m separation
between nearest opposing-team distances. A local velocity change supplies
contact strength; a stationary-to-moving trajectory supplies restart strength.
It returns explicit observation masks and `verified_contact=False`,
`verified_restart=False`, and `independent_video_evidence=False`. These are
experimental candidates; native extraction is not activated in inference.

## Fixed development reliability audit

`tools/contact_restart_v7.py` writes its policy and scientific source hashes
before computing candidates. It saves candidates for all cells before reading
any references. The five previously exposed Sportec matches are J03WN1,
J03WMX, J03WPY, J03WQQ, and J03WR9, each with clean, degraded, camera, and
calibrated observation regimes. These results are development evidence, not a
fresh test or an independent physical-contact validation.

The frozen candidate threshold is 0.5, peak separation is 0.5 seconds, and
matching is one-to-one within one second and within each half/known reference
span. A restart cue needed at least 20 candidates and at least 90% alignment
precision in every regime. No threshold was tuned after the audit.

| Regime | Restart candidates | Matched D-to-live boundaries | Alignment precision | Requirement |
| --- | ---: | ---: | ---: | --- |
| Clean | 39 | 12 | 30.8% | Failed |
| Degraded | 0 | 0 | Not estimable | Failed |
| Camera | 0 | 0 | Not estimable | Failed |
| Calibrated | 20 | 5 | 25.0% | Failed |

There are 488 reference restart boundaries in each regime. Contact proxies
aligned to direct team-change boundaries at 299/2,271 clean, 238/2,125 degraded,
281/1,851 camera, and 187/1,253 calibrated candidates. These ratios are **not
physical-touch precision**: a touch may correctly occur without changing team
possession, and H/A/D labels do not annotate every touch. Conversely, a geometry
cue close to a label transition does not prove a physical contact.

Local output: `runs/contact-v7-development-audit/`, including the immutable
protocol, all candidate timestamps, source hashes, and per-match results.
Protocol SHA-256:
`4039462b76f9c879f8e51ca4363f956d05a9b9b2e82770f126f31e6ff1864292`.
Raw evidence remains in ignored run artifacts.

The repository retains [models/contact-audit-v7.json](../models/contact-audit-v7.json)
as shareable evidence. Its version-1 schema includes explicit inactive/failed
flags, limitations, original protocol and source hashes, and the complete
underlying aggregate summary for all 20 match/regime cells. Export verified the
protocol digest and all frozen scientific source digests. It includes no raw
coordinates, frame labels, candidate timestamps, images, or video. The original
local summary digest is retained so the export can be checked against its source.

```powershell
python -m unittest tests.test_contact_evidence_v7 -q
python tools/contact_restart_v7.py --out runs/contact-v7-development-reproduction
```

Eleven focused tests pass: bounded finite covariates, team symmetry, complete
observation gates, half isolation, missing/implausible native steps, raw-XY
independence from provider speeds, invalid input rejection, peak selection, and
reference-gap matching. These tests establish implementation behavior, not
real-world physical-contact accuracy.

## Promotion limits and next evidence needed

The optional eight-feature typed-boundary ablation must win the separate frozen
match-level evaluation before any submission change. Even such a win would
support learned usefulness, not verified contact accuracy. The current fixed
audit explicitly rejects using stationary-launch candidates as hard restart
instructions.

The previous video pilots also remain inactive: the stronger detector's accepted
points had a 17.8% error rate beyond 2 m against automatic tracking, which itself
is not independent contact truth. See [the video recovery report](VIDEO_RECOVERY.md).
No new detector was scored against the final evaluation matches in this work.

Reliable physical evidence requires independently reviewed contact/restart
timestamps with uncertainty intervals, explicit missing/occluded-ball labels,
and match-separated validation. An annotation should distinguish a physical
touch, possession acquisition, dead-ball start, and actual restart. Apparent
ground-plane proximity alone cannot establish an airborne-ball touch. Such
annotations must be frozen before scoring a video/contact model; current H/A/D
references cannot substitute for them.
