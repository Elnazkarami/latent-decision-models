# Next task

**Done:**
- LDM-001 / LDM-002: ds002158 audit, behavioral baselines, computational confidence model.
- LDM-006: ds007474 audit ([`docs/ds007474-audit.md`](../docs/ds007474-audit.md)).
- LDM-007: ds007474 two-step RL benchmark, behavior only ([`docs/ds007474-two-step-model.md`](../docs/ds007474-two-step-model.md)).
- LDM-004 (metadata part): ds002158 fMRI derivative and design audit ([`docs/ds002158-fmri-audit.md`](../docs/ds002158-fmri-audit.md)).
- LDM-003: ds002158 EEG, 19 subjects (+ post hoc follow-ups: within-subject effect robust to the BCG method, cross-subject transfer is not; heartbeat timing does not explain it; the weaker transfer with OBS is explained by its signal loss) ([`docs/ds002158-eeg.md`](../docs/ds002158-eeg.md)). Pre-registered on 17 new subjects: confidence after own decisions decodes post-response (within r = 0.16, Holm p = 0.0009; left-out subjects r = 0.10, Holm p = 0.005), not pre-decision, and mostly overlaps accuracy/RT/evidence.

## Option A: further LDM-007 extensions (no approval needed, lower priority)
The misfit is addressed: hybrid + win-stay/lose-shift is best, including on a separate session.
1. Done: trial-wise arbitration (recovery and plan first). Less model-based weight after high-stakes trials (−0.27, calibrated Holm p = 0.02; replicates in the structural session); none for state uncertainty. Follow-up 3 (points-based model-free learning, plan committed first): the stakes effect reverses sign (+0.56), so it is not separable from the magnitude sensitivity of learning; no arbitration claim. Possible next model: graded magnitude sensitivity (reward = win × magnitude^γ), though arbitration weights recover weakly. Learned pad transitions are low value here (pads of a planet share their win probability).
2. Hierarchical (empirical-Bayes) fitting, to tighten per-subject estimates at 154 trials.

## Option B: LDM-003 follow-ups (no approval needed; data already on external storage)
Done: OBS sensitivity, ECG-only control, exploratory time course, cardiac-timing controls and ICA (see the EEG doc). Heartbeat timing does not explain the effect; ICA finds nothing heartbeat-locked after template subtraction and is inadequate alone. Open:
1. Done: the planted-signal test shows OBS (4 components) removes about half of a slow response-locked signal; OBS with 1 component restores the cross-subject transfer (r = 0.11), as predicted.
2. Done: own-vs-observed cross-condition decoding (plan committed first): no transfer in either direction. Remaining EEG questions need new data (e.g. separating movement from commitment).

## Option C: LDM-004 continued, ds002158 trial-level fMRI (needs explicit approval)
Download published derivative images for a 2-subject pilot (6 runs each, about 38 GB; full set 374 GB). Then:
1. LSS single-trial betas with 6 realignment regressors.
2. Atlas ROI extraction on the MNI grid.
3. A subject-disjoint confidence benchmark using these whole-trial features.

## Open questions for the PI
- **ds002158 fMRI:**
  - Approve the derivative-image pilot (about 38 GB)?
  - Exclude sub-20 from fMRI as well? It has the highest motion, a rejected run-007 derivative and unresolved EEG labels.
  - Accept that fMRI features are whole-trial, not pre-decision?
- **ds007474 licensing:** the OSF project and the authors' code declare no license. Ask the authors to add one before any public derived release?
- **ds007474 fMRI:** there are no subject-level derivatives. Is preprocessing 259 GB of raw BOLD in scope, or does the work stay behavior-only?
- **ds007474 exclusions:** the primary analysis excludes 8 subjects with > 10% missed trials and does not score RTs < 150 ms. Results are unchanged without exclusions. Keep this as the default?
