# Xtension Labs data-access brief

This brief is prepared for data-access enquiries. The PFF dataset access form was submitted on 2026-11-01 using user-provided details, and its supplied Drive folder was opened successfully. This prose brief and the support enquiry have not been sent; no separate dataset agreement has been accepted on the team's behalf.

The [PFF access packet](PFF_ACCESS.md) now includes the verified live form fields, a ready-to-send support enquiry, and the proposed ingestion and fresh-match validation plan.

## Project description

Xtension Labs is developing PossessionFlow for the More Than A Hack 2027 football analytics challenge. The system estimates home possession, away possession, stoppages and unknown periods from player/ball observations. We seek permission to use synchronized tracking and event/video annotations for research, training and independent evaluation in this challenge. The development repository is private.

## Data needed

We need identifiable full fixtures, both halves and all frames including stoppages; player/team identities and pitch coordinates; observed ball positions with missing/extrapolated flags; native timestamps, frame rate, pitch calibration and attacking direction; and synchronized event labels with actual restart, ball-out/whistle and controlled-acquisition timing where available. Please specify annotation methodology and uncertainty, coverage limitations, redistribution rules, derived-model rights and whether hackathon use is permitted. Sparse event records are useful, but dense controlled-possession and live/dead labels would require separate confirmation or independent review.

## Evaluation plan

We will reserve complete unseen fixtures before model selection, exclude all reference fields from inference, and publish only permitted code, derived metrics and models. A final test will assess stoppage precision/recall, onset/restart timing, short-possession precision/recall and complete possession-spell accuracy against the frozen current model. Reference gaps remain unknown. No model replaces the current submission without adequate independent evidence.

## Primary access routes

- [PFF FC World Cup 2022 release](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data): official form completed and delivery folder accessible. Usage rights, live/dead coverage and reliable state-reference provenance still need verification before model training and independent evaluation.
- [BEPRO sample](https://bepro.ai/free-data-example): request a full-match event/tracking sample, confirming ball coverage and annotation provenance.
- [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) or [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS): follow the official reviewed-access process for richer contact/action supervision. Their access agreements require the applicant's own review.

If only synchronized video and observed tracking are available, arrange two reviewers blinded to predictions, adjudicate disagreements, and use the reviewed-interval importer described in [LABELLED_DATA_V8.md](LABELLED_DATA_V8.md). A model's own labels cannot substitute for independent review.
