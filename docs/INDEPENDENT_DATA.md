# Independent-source evaluation

The seven Sportec matches used by earlier campaigns are all exposed development
data. Repeating a split does not make them an untouched final test. The independent
adapter added here measures **team predictions at sparse pass origins** in another
provider's matches. It does not establish dense possession accuracy or spellF1.

## Metrica public sample matches

[Metrica's sample-data repository](https://github.com/metrica-sports/sample-data)
publishes anonymized, synchronized player/ball tracking and events. Games 1 and 2
use normalized coordinates on a 105 by 68 metre pitch. Its README invites community
analytics and requests attribution for public uses; it supplies no standard
LICENSE file. We use local copies for this audit and do not redistribute raw data.
Credit: **Metrica Sports, sample tracking and event data**.

Before parsing either event file, the download command writes
`runs/independent-data/split-before-reference-access.json`: Game 1 is adapter
development and Game 2 is reserved for a frozen-candidate audit. Six CSV downloads
are bounded by published byte lengths and checked against their Git blob SHA-1;
the local manifest also records SHA-256. No videos are downloaded. No events are
parsed by the downloader or observation adapter.

The adapter maps each published frame identifier exactly, checks both team
timelines and their overlapping ball measurements, and derives speed within each
half. It passes no events into model inference. Once predictions have been saved,
the separate scorer reads only PASS actor-team references at their exact start
frames. Duplicate same-team references collapse; contradictory team references
are excluded. No unlabelled intervals are filled.

The principal metric is exact-frame pass-origin team accuracy. H/A errors and D/X
predictions all count as incorrect. Per-team recall, predicted-live coverage, and
conditional accuracy among H/A predictions are reported separately. High
conditional accuracy with low coverage is not success. The audit cannot measure
possession transitions between events, short-spell recall, or stoppage accuracy.
Pass contact timing and the competition's possession conventions may differ.

```powershell
python tools/independent_benchmark.py download
python tools/independent_benchmark.py evaluate --game 1 --out runs/independent-audit-game1
# Only after both the final candidate and baseline artifacts are frozen:
python tools/independent_benchmark.py evaluate-pair --game 2 --model models/submission.json --baseline-model models/baselines/v1.3.json --out runs/independent-audit-game2
```

Each evaluation records model, configuration and inference source hashes **before
opening reference events**. It rejects artifact/source changes during the run.
The paired command freezes both model hashes before inference, saves both
prediction CSVs, and opens the event references only once for shared scoring.
`--baseline-config` optionally supplies a separate baseline configuration; otherwise
both models use `--config` with their own validated decoder settings. Both paths
use the production inference dispatcher, including any supported sequence head.
Game 2 reference consumption is recorded with an exclusive access marker; the
tool rejects another audit against the consumed dataset directory. A failed
artifact/source check before reference access does not consume that reservation.
Game 2 must not be repeatedly used to choose or tune candidates; subsequent use
would make it exposed development data. This split is independent of Sportec,
but the sparse metric is not interchangeable with the organizer's dense metric.

The first Game 1 development audit used the v1.3 `camera-context` artifact
(`f4feb096face6a2a3eb1214079a5f34ea8b495147478b7a92b6b224a42dc0573`).
Among 798 unique pass-origin frames, exact team accuracy was **89.10%**,
balanced team recall **89.09%**, predicted-live coverage **93.36%**, and
conditional-live team accuracy **95.44%**. The model predicted D at **6.64%** of
pass-origin frames and X at none. One same-team duplicate was collapsed; no pass
frame was unmapped or contradictory. These are adapter-development observations,
not a reserved Game 2 result or a dense-possession estimate. The local immutable
audit evidence is `runs/independent-audit-game1-v1.3/summary.json`.

## SoccerTrack v2: verified access and leakage limitations

[The authors' repository](https://github.com/AtomScott/SoccerTrack-v2) identifies
ten university matches and separates MIT source code from CC BY 4.0 data. The
[paper](https://arxiv.org/abs/2508.01802) describes player-linked ball actions and
full-pitch trajectories, which would broaden the domain beyond professional
Sportec matches.

As inspected on 2026-10-29, the [official dataset page](https://huggingface.co/datasets/atomscott/soccertrack-v2)
requires a Hugging Face login and acceptance of conditions, including sharing
account information. No acceptance was made and no gated file was downloaded.
The page's v1.2 release notes also identify its auxiliary ball track as reconstructed
from event anchors and interpolation. **That track is unsuitable as an independent
input for scoring those same event labels:** it already encodes their timing.
A future audit would need player-only inputs or independently recovered video-ball
observations, and would still have sparse action references rather than dense HAD
possession labels.

The official [GSR format documentation](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md)
warns that older examples differ from shipped COCO-style objects and documents
historical timing inconsistencies. Its [BAS format](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-bas.md)
describes event-team annotations, not dense possession. Any adapter must inspect
the actual accepted release and test frame alignment instead of assuming older
examples are authoritative. The current implementation makes no claim to have
loaded or evaluated SoccerTrack v2.

Citation: Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim and Keisuke Fujii,
*SoccerTrack v2: A Full-Pitch Panoramic Video Dataset for Game State Reconstruction
and Ball Action Spotting*, arXiv:2508.01802 (2025; revised 2026).
