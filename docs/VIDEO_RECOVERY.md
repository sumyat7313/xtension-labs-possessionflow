# Local video recovery: experimental and inactive

Two local tactical-video pilots were completed. A classical tracker recovered
too few observations. A stronger pretrained YOLO11n detector accepted 81.1% of
masked validation observations, but 17.8% of those accepted points differed by
more than 2 m from the automatic tracking reference. Neither result establishes
safe improvement in possession predictions. Video recovery remains experimental
and **inactive in submission inference**.

## Implemented method

`possession/video_recovery.py` provides an optional OpenCV tracker operating on
native 1920 × 1080 video frames. An existing automatic ball observation seeds a
13 × 13 appearance template. Subsequent frames are searched locally for compact,
bright components supported by template correlation. The candidate must pass
physical displacement, appearance, score-margin, pitch-boundary and homography
quality gates. It expires 0.8 seconds after its original anchor; accepted frames
do not renew that lifetime. Cuts and unusable homographies reset its state.

This is a local computer-vision candidate generator, not a learned football
detector. It does not estimate airborne ball height. Pitch homography maps the
ground plane and can mislocate an airborne ball. White kit, boots, markings,
occlusion and tiny balls can defeat the detector. Abstention is explicit.

`transform(H, points, inverse=False)` follows the organizer's documented mapping:
normalized pitch coordinates to image pixels. `LocalBallTracker.seed(...)`
initializes from an observed anchor; `recover(...)` returns either a candidate
with confidence or a rejection reason. The tracker never receives the withheld
current or future ball coordinates.

## Reproduce the bounded pilot

Use a separate research environment with NumPy and `opencv-python-headless`.
The completed pilot used OpenCV 5.0.0. Minimal production inference does not
require OpenCV. Its optional tests skip when OpenCV is absent.

```powershell
python tools/recover_video_ball.py `
  --video "C:/Users/tkala/Downloads/BAR-ATH_tactical-001 (1).mp4" `
  --atd "../more-than-a-hack-2027/data/tactical/BAR-ATH_tactical_atd.txt" `
  --homography "../more-than-a-hack-2027/data/tactical/BAR-ATH_tactical_homography.csv" `
  --out runs/video-pilot-reproduction --windows 24 --mask-frames 12
```

Replace paths with authorized local assets. The output directory must be new.
The runner verifies video FPS/frame count against tracking and homography rows,
then writes its settings and scientific source hashes before scoring. No network
request, remote model service or target possession reference is used.

The sampling rule chooses the earliest fully observed 13-frame span after the
first second of each fixed 120-second time block, then samples 24 eligible blocks
evenly across the recording. The first observation is the seed; the next twelve
are hidden from the tracker and evaluated only after prediction. Even time blocks
are development and odd blocks validation. No detector parameter was changed
after either split was scored. Entire windows stay inside a match half.

The reference is existing **automatic ball tracking**, a silver reference rather
than independent ball ground truth. Sampling observed spans is optimistic
relative to genuine missing-ball intervals. Consecutive frames in one window are
correlated, so twelve successes in one window are not twelve independent trials.

## Completed classical pilot, 2026-10-29

| Split | Masked frames | Accepted | Within 2 m of reference | Median accepted error |
| --- | ---: | ---: | ---: | ---: |
| Development | 156 | 12 | 12 | 0.087 m |
| Validation | 132 | 0 | 0 | Not estimable |
| Combined | 288 | 12 | 12 | 0.087 m |

Eight of 24 seeds were unusable; the remaining rejected frames failed visual
evidence gates. Combined acceptance was 4.17%. No accepted point exceeded the
2 m reporting threshold, but **there were no accepted validation observations**.
This is insufficient evidence of reliable recovery and no evidence of improved
possession spell F1. It must not be described as a validated video improvement.

The recording was 199,319 frames at 25 FPS, 1920 × 1080, 5,222,307,334 bytes.
The complete pilot took 7.6 seconds on the local machine. Local artifacts under
`runs/video-pilot-v4/` retain the immutable protocol, aggregate summary, sample
selection, per-frame diagnostics and recovered coordinate array. Frame-level
records and coordinates remain local and are excluded from source publication.

## Stronger pretrained detector and fresh-window pilot

`YoloBallTracker` accepts a locally loaded COCO-pretrained YOLO11n model and
selects its `sports ball` class. Each missing frame is searched in two native
image ROIs, with radii 96 and 192 pixels around the last accepted pitch location.
Both crops are enlarged for inference at 960 pixels. Duplicates across scales
are merged; confidence must be at least 0.4 and the best distinct candidate must
lead by at least 0.1. Physical displacement, pitch bounds, calibration quality,
shot cuts and the original 0.8-second anchor lifetime remain enforced.

The runner verifies the checkpoint SHA-256 before loading it. It disables
Ultralytics network checks, automatic installation and telemetry (`YOLO_OFFLINE`,
`YOLO_AUTOINSTALL`, and `sync=False`), runs on CPU with two Torch threads, and
passes local image arrays directly to the model. No target image or recording
is sent to a hosted inference endpoint. No target fine-tuning was performed.

The checkpoint was obtained from the publisher's
[official release asset](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt).
Its SHA-256 is:

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

The [official model documentation](https://docs.ultralytics.com/models/yolo11/)
describes COCO pretraining and the upstream AGPL-3.0 / Enterprise license options.
These third-party weights and software retain their upstream terms; the
repository's MIT license does not relicense them. The checkpoint is stored only
under excluded `runs/video-assets/`, and is not included in the submission model,
source archive, or core inference requirements. The research runner is optional.

Verified research versions were Ultralytics 8.4.192, Torch 2.15.0+cpu,
Torchvision 0.29.2, OpenCV 5.0.0.93 and NumPy 2.5.4. The selected checkpoint,
detector parameters and dependencies are recorded in the local pilot artifacts.

To reproduce, install the optional research dependencies in a separate
environment, download the official checkpoint to `runs/video-assets/yolo11n.pt`,
and repeat the earlier command with a new output directory plus:

```text
--backend yolo --weights runs/video-assets/yolo11n.pt
--exclude-sampling runs/video-pilot-v4/sampling.json --windows 20
```

All 120-second time blocks used in the first pilot are excluded, including its
previous validation blocks. The second pilot selected 20 additional blocks and
kept all detector parameters fixed before scoring. Its windows and source hashes
are recorded under `runs/video-yolo-pilot-v4/`. There was no time-block overlap.

| Split | Masked frames | Accepted | Within 2 m | Median accepted error | Accepted points beyond 2 m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 108 | 86 | 76 | 0.796 m | 11.6% |
| Validation | 132 | 107 | 88 | 0.691 m | 17.8% |
| Combined | 240 | 193 | 164 | 0.727 m | 15.0% |

The detector accepted observations in all 20 sampled windows. Validation
acceptance was 81.1%, while 66.7% of all masked validation frames had an accepted
point within 2 m. Validation p90 accepted-point error was 2.758 m. The complete
pilot took 79 seconds locally. These are silver-reference discrepancies, not
human-verified detector false positives. Some may reflect automatic-reference
error, ground-plane projection of an airborne ball, or a wrong detection.

The two pilots use different windows; their aggregate rates are not a paired
accuracy comparison. The stronger detector demonstrates practical recovery
coverage, but its remaining error is material near competing players. It is not
automatically injected into tracking, and no possession-F1 benefit is claimed.
These second-pilot windows are now exposed and cannot serve as an untouched test
for a later detector tuned using their results.

A subsequent **development-only** confidence screen checked every distinct
accepted confidence threshold against a 95% within-2m target, requiring at least
24 accepted frames across three time windows. No threshold qualified. Increasing
confidence to 0.8 retained 37 frames across five windows but only 81.1% lay within
2 m; 0.9 retained 14 frames and failed both precision and minimum support. Thus
detector class confidence is not a reliable localization-error calibration here.
The screen used no validation rows, selected no candidate and performed no new
validation. Remaining unused time blocks were left untouched. The full rejected
screen is recorded locally in `confidence-screen.json`; no high-precision claim
or activation follows from it.

## Optional broadcast alignment

`align_tracking_window(reference, other, start, stop, offsets, ...)` estimates a
local offset from dynamic ball observations. It requires at least 40 matched
observations, more than 8 m spatial variation, at least 80% of residuals below
3 m, median residual below 1.5 m, and separation from competing offsets. Invalid
calibration frames are excluded. Static, ambiguous and repeated trajectories
are rejected. It never assumes a global offset across broadcast cuts.

A bounded tracking-only pilot tested three ten-second tactical windows against
the available broadcast tracking and valid homographies. Two lacked sufficient
consistent dynamic support; one was ambiguous. **No alignment was accepted.**
The local diagnostic is `runs/video-pilot-v4/broadcast-alignment-pilot.json`.

Broadcast video is not locally available, so no broadcast video recovery or
independent cut/replay validation was performed. The helper's ambiguity check
can reject repeated trajectories but cannot certify that a uniquely matching
segment is live rather than a replay. Any later integration needs consistent
neighboring-window alignment and shot/replay checks, followed by independent
recovery validation. The present release never merges broadcast candidates.

## Tests and next evidence needed

Nine tests cover homography round trips, invalid calibration, static and replay
ambiguity, synthetic visual recovery, competing candidates, anchor expiry, shot
cuts, optional-detector class/gate behavior, ROI deduplication and input
immutability. They passed in the OpenCV research environment. The detector unit
test uses synthetic outputs; the separate real-video pilot exercised the actual
downloaded model.

A useful next candidate would improve precision using permitted other-match ball
imagery and temporal validation, then test on newly frozen silver windows. A
separate possession benchmark must demonstrate benefit before enabling recovered
observations. These requirements are not satisfied by the current pilots.
