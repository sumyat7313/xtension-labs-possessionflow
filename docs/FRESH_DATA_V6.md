Audit completed: **failed** for the frozen **combined** experiment. These sources are now consumed. See [results and release decision](TEMPORAL_CAMPAIGN_V6.md).

# V6 reserved fresh-match audit

This audit reserves one new dense provider sample and two new SkillCorner
matches. The v5 Metrica/SkillCorner audit matches, both earlier Metrica CSV games,
and all seven Sportec matches are already exposed and are not fresh tests.
No final-match rows or reference values were read while building this adapter.
Only the WCSF CSV header was inspected after downloading and verifying its ZIPs.
The evaluator consumes the reservation exactly once, after model/source freeze.

## Sources and reference quality

The [University of Groningen / SciSports WCSF sample](https://github.com/MatKempeGroningen/WCSF_SolutionRoom/tree/1c9ba798f98a9fffb16081a5d3965774d4e5662c)
contains an anonymized full match, downsampled and smoothed from optical tracking
to 10 Hz. Its [data guide](https://github.com/MatKempeGroningen/WCSF_SolutionRoom/blob/1c9ba798f98a9fffb16081a5d3965774d4e5662c/README.md)
defines `AliveFlag=1` as live and `0` as dead, and `PossessionFlag` as team
`group_id` (`1` home, `2` away). Coordinates use a 105 by 68 metre pitch and each
team's own goal is at negative x. These are explicit dense provider references;
the guide does not establish human annotation or explain how the flags were
produced. It does not expose the original fixture identity, so independent
cross-provider identity checking is unavailable. No human-gold claim is made.
Credit: University of Groningen, Matthias Kempe, Floris Goes-Smit and SciSports.
The repository requests citation and supplies no standard license file; raw
tracking is retained locally and is not redistributed.

Pinned source commit: `1c9ba798f98a9fffb16081a5d3965774d4e5662c`.
Both halves form **one match**, never two independent samples.

| File | Bytes | Git blob SHA-1 |
| --- | ---: | --- |
| positions_FIRST_HALF.csv.zip | 22,704,565 | d7de2a38f5650150684b1008aa7f4fd28b85d777 |
| positions_SECOND_HALF.csv.zip | 23,746,663 | 66fc3ee2966aaf3392d4a327ba3f49385e9c7032 |

[SkillCorner open data](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656)
supplies matches **1925299 and 1927964**, the two lowest unconsumed match IDs in
the pinned catalog. Its phase references are automatic provider labels. The
existing v5 observation whitelist and documented half-open phase intervals are
reused. Unknown phase gaps never become dead-ball references. These matches
test team agreement in another observation domain, not independent stoppage
truth. Source commit: `4340d274572876239c154c90bc507a9b3250a656`.
Credit: SkillCorner. See [v5 provenance](FRESH_DATA.md) for source specifications.

SoccerTrack v2 and the [SoccerNet/SNGAR tracking release](https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting-Tracking-deprecated)
remain gated. The latter requires login, access review and an NDA. No conditions
were accepted and no gated files were downloaded for this audit.

## Observation-only adaptation

WCSF inference parses only `x`, `y`, `x_ball`, `y_ball`, `group_id`,
`timestamp_ms` and `player_id`. The parser excludes `AliveFlag`,
`PossessionFlag`, `team_game_phase`, and provider-derived speed/acceleration.
Speeds are derived from the reconstructed geometry within each half. Missing
100 ms frames remain missing, and no tracking interpolation is added.

The guide leaves two coordinate details ambiguous: y reflection and whether
ball coordinates are team-frame coordinates or offsets from each player.
Before reading any final rows, the adapter fixes four possible conventions:
the Cartesian product of those two choices. Home x is kept, away x is negated.
The adapter selects a convention only when every available copy of the same
frame's ball agrees within **0.01 metres** after transformation, with at least
two frames containing copies from both teams. Distinct consistent reconstructions
are rejected as ambiguous. Mathematically identical reconstructions may share
the first canonical convention. No possession/state label contributes to this
choice. Selected convention and maximum discrepancy are recorded in the audit.

Half timelines are validated at native 10 Hz and all inference is half-local.
For dense scoring, explicit zero AliveFlags produce D. Live frames with team 1
or 2 produce H/A. Missing/unknown reference values and conflicting copies remain
X; more than 1% conflicting reference frames aborts scoring. No gap, absent ball,
phase label or model output is converted into a dead reference.

## Fixed promotion audit

The split and numeric policy are stored before final reference access. After
development selection, the evaluator freezes both model/configuration hashes,
all inference sources, adapters and metric sources. An exclusive access marker
is written **before opening final tracking records with embedded reference
fields**, even though those fields are excluded from inference. All six paired
prediction CSVs are saved before separate flag/phase scoring.

The v1.5 baseline and exactly one selected experiment are compared. The fixed
fresh gate requires:

- WCSF spell F1, macro F1, D precision and D recall each decline by no more than
  0.005 absolute.
- WCSF D-segment F1 at IoU 0.5 and the mean of D-onset/offset F1 within Â±1 second
  each decline by no more than 0.01.
- WCSF under-two-second H/A spell recall at IoU 0.5 declines by no more than 0.01.
- At least one strict improvement in the dense mean D-boundary F1 or short-spell
  recall. Identical predictions alone cannot pass.
- All H/A/D classes, short possessions, D onsets and D offsets have reference
  support; otherwise the result is inconclusive and fails promotion.
- Equal-match mean SkillCorner team accuracy is nondecreasing, with no individual
  decline greater than 0.02.

All time thresholds use native FPS. The official scorer contributes only
FPS-independent macro/spell/class metrics; custom temporal diagnostics use 10 Hz
explicitly. Unknown separators preserve halftime boundaries for official spell
matching. Native diagnostic blocks avoid crossing halves or reference gaps.
Both development and fresh gates must pass to promote an experiment. No fresh
failure authorizes trying a runner-up, retuning, or using the same references as
untouched again.

```bash
python tools/fresh_benchmark_v6.py download --data-dir runs/fresh-data-v6
# Run only after one development candidate and baseline are frozen:
python tools/fresh_benchmark_v6.py evaluate-pair --data-dir runs/fresh-data-v6 --model runs/temporal-campaign-v6/release-candidate-model.json --baseline-model models/baselines/v1.5.json --config configs/default.json --out runs/fresh-audit-v6
```

The thirteen synthetic adapter/protocol tests cover mirroring and relative-ball
conventions, ambiguity failures, reference exclusion, missing observations,
native-FPS metrics, half boundaries, explicit flags, conflicting labels,
promotion regressions, reservation guards, and prediction-before-reference
ordering. Real final matches are not used to adjust the adapter.

An anonymous provider match and two automatic phase references cannot establish
generalization across leagues or independent human stoppage/short-possession
accuracy. Smoothing and undocumented label generation are material limitations.
Passing this audit is a bounded release check, not a winning-probability claim.
