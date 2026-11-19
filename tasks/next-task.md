# Next task

**Done:**
- LDM-001 / LDM-002: ds002158 audit, behavioral baselines, computational confidence model.
- LDM-006: ds007474 audit ([`docs/ds007474-audit.md`](../docs/ds007474-audit.md)).
- LDM-007: ds007474 two-step RL benchmark, behavior only ([`docs/ds007474-two-step-model.md`](../docs/ds007474-two-step-model.md)).
- LDM-004 (metadata part): ds002158 fMRI derivative and design audit ([`docs/ds002158-fmri-audit.md`](../docs/ds002158-fmri-audit.md)).
- LDM-003: ds002158 EEG, 19 subjects ([`docs/ds002158-eeg.md`](../docs/ds002158-eeg.md)). Pre-registered on 17 new subjects: confidence after own decisions decodes post-response (within r = 0.16, Holm p = 0.0009; left-out subjects r = 0.10, Holm p = 0.005), not pre-decision, and mostly overlaps accuracy/RT/evidence.

## Option A: further LDM-007 extensions (no approval needed, lower priority)
The misfit is addressed: hybrid + win-stay/lose-shift is best, including on a separate session.
1. Learned pad transitions and trial-wise arbitration (the authors' `isPost_*` terms), each with recovery first.
2. Hierarchical (empirical-Bayes) fitting, to tighten per-subject estimates at 154 trials.

## Option B: LDM-003 follow-ups (no approval needed; data already on external storage)
1. Residual BCG: compare ICA or OBS with template subtraction on sub-02 and sub-11 (weakest heartbeat detection).
2. Time-resolved decoding of confidence after own decisions (where after the response the signal appears), as an exploratory, clearly labelled analysis.
3. Pre-register an own-vs-observed contrast before testing it (not part of the confirmatory plan).

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
