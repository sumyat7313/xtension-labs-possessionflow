# Xtension Labs data-access brief

This brief is prepared for data-access enquiries. The PFF dataset access form was submitted on 2026-11-01 using user-provided details, and its supplied Drive folder was opened successfully. This prose brief and the support enquiry have not been sent; no separate dataset agreement has been accepted on the team's behalf.

Updated 2026-11-03: the complete **261-file PFF release covering 64 fixtures is downloaded**. The metadata-only **40 training / 12 development / 12 locked final** split is frozen, and the isolated fixed training/evaluation campaign is complete. v1.5 remains active because the development comparison regressed stoppage precision. The final 12 matches remain untouched. See the [PFF results report](PFF_RESULTS.md).

The [PFF access packet](PFF_ACCESS.md) retains the verified form details and ready-to-send support enquiry. The [terms and label review](PFF_TERMS_AND_LABELS.md), [validation protocol](PFF_VALIDATION.md) and [training/adapter design](PFF_TRAINING.md) document the completed intake checks and remaining promotion requirements. Local analytical research is distinct from unverified rights to redistribute provider data or fitted PFF artifacts; the support enquiry remains unsent.

## Project description

Xtension Labs is developing PossessionFlow for the More Than A Hack 2027 football analytics challenge. The system estimates home possession, away possession, stoppages and unknown periods from player/ball observations. We seek permission to use synchronized tracking and event/video annotations for research, training and independent evaluation in this challenge. The development repository is private.

## Data needed

We need identifiable full fixtures, both halves and all frames including stoppages; player/team identities and pitch coordinates; observed ball positions with missing/extrapolated flags; native timestamps, frame rate, pitch calibration and attacking direction; and synchronized event labels with actual restart, ball-out/whistle and controlled-acquisition timing where available. Please specify annotation methodology and uncertainty, coverage limitations, redistribution rules, derived-model rights and whether hackathon use is permitted. Sparse event records are useful, but dense controlled-possession and live/dead labels would require separate confirmation or independent review.

## Evaluation plan

Complete PFF fixtures were reserved before model selection. The completed experiment excludes all reference fields and event-derived smoothed ball positions from inference, using raw visible observations instead. Its carried event references are provider-silver labels, with uncertainty retained as unknown. The locked final test is accessed only if the fixed development gate passes; it then compares stoppage precision/recall, onset/restart timing, short-possession precision/recall and complete possession-spell accuracy against v1.5. Raw files, reconstructed labels and fitted PFF weights remain local. No model replaces the submission without adequate independent evidence and established sharing rights.

## Primary access routes

- [PFF FC World Cup 2022 release](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data): complete release downloaded; terms, raw-observation semantics and event-reference definitions reviewed; isolated research completed on the frozen fixture split; v1.5 retained after a failed development gate. Independent dense truth and rights to distribute fitted artifacts remain unresolved, so access or a numerical gain alone cannot authorize promotion.
- [BEPRO sample](https://bepro.ai/free-data-example): request a full-match event/tracking sample, confirming ball coverage and annotation provenance.
- [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) or [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS): follow the official reviewed-access process for richer contact/action supervision. Their access agreements require the applicant's own review.

If only synchronized video and observed tracking are available, arrange two reviewers blinded to predictions, adjudicate disagreements, and use the reviewed-interval importer described in [LABELLED_DATA_V8.md](LABELLED_DATA_V8.md). A model's own labels cannot substitute for independent review.
