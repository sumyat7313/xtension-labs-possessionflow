# PFF release: terms and label provenance review

Reviewed 2026-11-01. This records source evidence and the scope of our technical work, not a determination of legal rights. No message, access agreement or additional consent was sent or accepted during this review. The subsequent adapter review inspected only the preassigned training pilot 3812. No development/final match labels or predictions were opened by this reviewer.

## What the release supports

The [original PFF FC announcement](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) intentionally supplies a form-based download for analysis and invites users to share findings. The [February 2025 enhanced-release announcement](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) reiterates free access and analytical use, with per-game event, metadata and roster files. Its older generic file-list paragraph still describes the previous consolidated layout; the newer restructuring description and actual delivered inventory should govern parsing.

The completed form and working official delivery folder establish authorized access to the supplied release. The analytical invitation supports downloading, inspecting and privately experimenting on that release. Treating local predictive-model experiments as part of that analytical purpose is a reasonable working interpretation, **not an explicit machine-learning licence clause**. The source does not establish unrestricted redistribution, commercial product use, sublicensing, or permission to distribute fitted artifacts to competition judges. Those broader permissions remain unverified.

## Terms found, and their limits

| Primary source | Verified scope and relevance |
| --- | --- |
| [Gradient Sports terms PDF](https://fc.gradientsports.com/documents/terms-and-conditions.pdf), pages 1–3, sections 1–5 | Names PFF FC, LLC doing business as Gradient Sports. It describes the gradientsports.com website/app, limits website-material uses and includes a restriction on competitive/benchmarking uses of the website/content. It is not labelled a World Cup dataset licence. No evidence inspected here establishes how it is incorporated into the separate historical Drive release or reconciled with the release's express analytical invitation. Do not silently treat either text as overriding the other. |
| [pff.com terms](https://www.pff.com/terms), dated October 12, 2026 | Names Slide Sports Holdings, LLC and defines scope through services linked to those terms. It contains detailed data, AI and personal-use restrictions, including provisions for locally controlled traditional models. This is a different stated entity/service context. No inspected release-page evidence incorporates it into the PFF FC World Cup download. It must not be applied automatically merely because both brands contain “PFF.” |

The direct release is stronger evidence for this particular download and its stated analytical purpose than a generic terms search result. Conversely, free access is not proof of a permissive open-data licence. No dataset-specific licence grant was found in the pages, change log or the two delivered specifications inspected by this reviewer. Both specifications carry a PFFA Acquisition LLC proprietary-information notice on every page. A proprietary notice is not itself a prohibition of the explicitly offered analytical use, but it does not grant redistribution rights. This is a search outcome, not proof that no other agreement exists. The old `fc.pff.com` homepage/documentation endpoint failed to load through web retrieval, so it could not resolve these points.

## Work scope while the remaining rights are unresolved

- Download only through the official delivered route; preserve originals and hashes in ignored local data storage.
- Continue schema/coverage checks and local research experiments, with the inference features separated from reference fields. Keep data, frame labels and fitted PFF artifacts local until their sharing scope is established. This is the bounded analytical-use interpretation above, not a claim that every model use is explicitly permitted.
- The release's invitation to share findings supports original adapter/test code, methodological descriptions, non-reconstructive aggregate research metrics and provenance summaries as analytical findings. This is the stated bounded interpretation of that invitation, not a general redistribution licence. Aggregate accuracy/support counts should describe limitations and must not expose frame-level reference sequences or enable reconstruction of provider records. Do not add raw data, reconstructed labels, provider documentation or fitted PFF weights to the repo merely because it is private.
- Obtain dataset-specific clarification before treating a PFF-trained artifact as cleared for challenge submission or broader release. The prepared [support enquiry](PFF_ACCESS_REQUEST.txt) asks about model training, judge delivery and reporting rights; it remains unsent. A numerical win cannot resolve a rights question.

## Label evidence available before match inspection

The supplied PFF FC Change Log supplied with the download was exported locally to `runs/pff-access/change-log.md`. Its March 2025 entry describes event-frame coordinates changing from smoothed values to raw broadcast tracking. May adds initial-touch events (`possessionEventType` of `IT`), possession sequence identifiers, pitch dimensions and starting direction. June adds attacking direction and player position groups. These are useful schema facts, not evidence that every frame has adjudicated controlled-possession and stoppage truth.

The [Kloppy loader documentation](https://kloppy.pysport.org/user-guide/loading-data/pff/) exposes ball state, owning team and an `only_alive` option. This confirms an interface capable of filtering live play, not completeness of dead frames or the source of those labels. Retain all available frames, then verify actual delivery coverage.

## Delivered specifications inspected

These originals were downloaded from the official supplied folder and read locally using pypdf 6.21.4. Extracted text remains ignored in `runs/pff-access/`; the PDFs remain in ignored data storage. All relevant definition pages were read, and both complete documents were searched for licence/permission, annotation, control, timing and observation terms. This specification review preceded the separate adapter audit on training pilot 3812.

| Local original | Pages / date | SHA-256 |
| --- | --- | --- |
| `data/pff/Tracking Data/PFF FC Tracking Data Specification v2.2.pdf` | 8 / October 2023 | `d6d2ab083f864e749c951a3256f6d3415eba7bd5f67eea6025403eaa59838f49` |
| `data/pff/Event Data/PFF FC Event Data Specification v2.5.pdf` | 43 / June 2025 | `90b34f6b7e0401a0ab4c9f1328ef8e10a8f7237e4461359ee8c292d8fa9e0246` |

### Observation provenance and leakage

Tracking specification pp. 2–4 describes a synchronized combination of PFF events and Sportlogiq broadcast tracking. It explicitly states that `ballsSmoothed` uses event annotations: the ball is placed at the labelled carrier and interpolated between events. **Never use `ballsSmoothed` to recover or independently evaluate those event/possession labels.** Original `balls`, `homePlayers` and `awayPlayers` are available separately. Raw is not synonymous with observed: off-screen players and some airborne-ball positions are context estimates; `visibility` distinguishes `VISIBLE` from `ESTIMATED`. Player smoothing is a Kalman operation, distinct from the event-guided ball smoothing.

The same page states that multiple associated possession events can duplicate tracking frames. Group by native game/period/frame identity, verify duplicate observation agreement and preserve event multiplicity in reference storage only. Do not count repeated annotation rows as additional frames or independent evidence. Contradictory duplicates require a documented rule or quarantine.

Tracking pp. 3, 7–8 define video milliseconds, zero-based video frame number, period-local elapsed seconds and an estimated game clock as different fields. Periods 3–4 are extra time. Coordinates are pitch-centred metres, with x/y directions referenced to the main camera; coordinates are provided during main-camera segments. FPS and pitch dimensions come from metadata. Camera gaps must not become invented stoppages.

### Reference semantics

Tracking p. 5 defines event start/end frames, the controlled player's team, a possession-sequence field and `home_ball` for the controlling team during the sequence. It distinguishes `OUT` from `VID` and other event types. These are promising provider references; the document does not establish blinded independent adjudication or timestamp error bounds.

Event specification pp. 3–8 defines `startTime`, `endTime` and `eventTime` in video-relative seconds. Game clocks are separate. `sequence` denotes uninterrupted team possession; `OTB` denotes an on-ball player possession. `OUT` identifies out-of-play events, with `outType` distinguishing scores, touch and whistle. `videoMissing` says part of an event is cut by broadcast; it is not a dead-state flag. Set-piece types identify corner, drop ball, free kick, goal kick, kickoff, penalty and throw-in, versus open play. Disallowed events have `initialNonEvent` or `nonEvent` flags. Do not turn a foul into a whistle or infer a restart time from a set-piece category alone without checking the associated temporal fields.

The initial-touch type `IT` is used only when another event does not already occur on that initial touch (p. 8). Missing IT records are therefore not negative examples. Event pp. 34–35 describe broadcast tracking coordinates and visible/estimated flags; the delivered change log clarifies that event-frame coordinates changed to raw broadcast tracking in March 2025. This does not change the separate 2023 tracking specification's explicit warning about `ballsSmoothed`.

The specifications improve the basis for provider-reference labels substantially, but do not establish exhaustive frame coverage, per-label human-review procedure, inter-rater agreement or timing uncertainty. Audit these properties on the designated training pilot before deciding which reference intervals can be scored. Keep unknown/ambiguous spans unknown; do not relabel all unlisted frames as D.

## Required technical checks

1. Map the delivered native schema explicitly: timestamps and frame IDs, periods including extra time, pitch units, direction, team IDs, missing observations and provider flags. Do not infer a fixed frame rate from a tutorial.
2. Distinguish observed tracking from event-attached frame snapshots and from interpolated/extrapolated values. Establish whether ball coordinates or state were produced using the same annotations being evaluated. Shared derivation limits independent validation claims.
3. Determine whether live/dead and owning-team fields are actual provider references or loader inferences. Keep them, event types, possession sequences and initial-touch labels out of the inference input whitelist.
4. Verify that a possession sequence means the challenge's controlled-team definition. One touch, a deflection and controlled acquisition are not interchangeable. No-possession values are not automatically dead play. Unlisted sparse events do not create negative labels.
5. Audit native time continuity and coverage of stoppages. Camera cuts, absent ball observations, absent records and unknown state remain distinct. Never call a live-only sample a stoppage benchmark.
6. Reserve full independent fixtures before inspecting their labels; check cross-provider match identity rather than filename alone. Report results as provider-reference agreement unless human annotation and uncertainty are established. Require adequate dense state and boundary evidence before claiming v1.5 was beaten on stoppages and short possessions.

The current v1.5 deployment must remain unchanged if reference quality, independent support, numerical promotion gates or deployment rights are insufficient. Downloading additional matches alone resolves none of those tests.
