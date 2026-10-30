# Next experiments after v6

These are research proposals, not measured improvements. Keep the v6 experiment
and its consumed fresh references fixed. A new experiment needs a new protocol
and a separately reserved evaluation source.

## 1. Establish reliable dense labels and boundary timing

The five scored development matches contain only three unique D spells under
two seconds. Repeating corruption conditions does not increase that independent
sample size. Add synchronized footage with independently reviewed H/A/D spans,
especially 2–5-second stops, rapid turnovers, contested touches and ball occlusion.
Use two annotators and adjudicate ambiguous boundaries while blinded to model
predictions. Reserve complete matches, including uninterrupted ordinary play,
for final evaluation; model-selected hard clips alone cannot estimate accuracy.

The new [WCSF provider sample](https://github.com/MatKempeGroningen/WCSF_SolutionRoom/blob/1c9ba798f98a9fffb16081a5d3965774d4e5662c/README.md)
has explicit live/dead and possession flags, but its guide does not establish
human annotation or disclose fixture identity. It cannot replace independently
adjudicated labels, and is already consumed by the v6 audit.

## 2. Learn typed boundaries and discourage false segments

V6's stoppage weighting increased dead-frame recall but created more false or
split stops. Across the 20 repeated match/condition cells, predicted D segments
increased from 2,154 to 2,202 while matched segments decreased from 1,898 to 1,891.
These counts are correlated across four conditions, not 20 independent matches.

A next model should jointly predict H/A/D states and separate H↔A, live→D and
D→live boundary evidence. Add a high-resolution branch for short contacts and a
longer-context branch for stoppages, then train segment consistency and penalize
unnecessary boundaries. Keep inference half-local and preserve team symmetry.
Evaluate onset/offset precision, segment IoU, short-spell precision and recall,
not only dead-frame recall.

This is motivated by [ASRF](https://arxiv.org/abs/2007.06866), which separates
frame classification and boundary prediction to address temporal over-segmentation.
Its reported results concern other action datasets; they are not a numerical
forecast for football tracking or evidence that this proposed adaptation works.

## 3. Add complementary contact and restart observations

Tracking alone can leave tackles, deflections and brief control ambiguous. A
separate video branch could contribute actual ball-contact, line-crossing or
restart evidence, after checking synchronization, camera calibration and detection
precision on development footage. Keep it confidence-gated and require a frozen
paired improvement before enabling it. Previous video pilots remain inactive.

Do not use event-interpolated auxiliary ball trajectories as independent sensor
observations when evaluating event/boundary recovery; this can leak the target.
Do not lower decoder penalties globally to chase short recall: v6's transition
ablation raised short recall while lowering short-prediction precision and overall
spell F1. Additional independent evidence is more promising than another global
threshold reduction, but still needs experimental confirmation.
