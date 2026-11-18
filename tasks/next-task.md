# Next task

**LDM-001 is done** (see `docs/ds002158-audit.md`). **LDM-002 is in progress** (first pass in `docs/ds002158-behavioral-baseline.md`).

## Option A: finish LDM-002 (no approvals needed)
1. Add a hierarchical logistic metacognition model (accuracy ~ confidence × condition, with random effects by subject), mirroring the paper's primary analysis. Use `statsmodels` or `bambi` as an optional extra.
2. Add ordinal/beta-regression alternatives and calibration plots for the confidence target.
3. Add run-held-out (within-subject) evaluation, reported separately from subject-held-out results.
4. Add a computational confidence model (e.g. evidence accumulation with post-decision evidence) with a parameter-recovery test on synthetic data before fitting real data.

## Option B: start LDM-003 (needs explicit approval)
- Download the raw EEG `.eeg/.vhdr/.vmrk` for **sub-02 and sub-04** (about 4.2 GB) to `data/ds002158/`.
- Use MNE (`[eeg]` extra) to read the data and the verified marker map (`latent_decision_models.datasets.ds002158.STIMULUS_MARKERS` / `RESPONSE_MARKERS`).
- Run `align_eeg_markers` before epoching and refuse to proceed unless the status is `verified`.
- Keep pre-response windows (stimulus-locked, before the response marker) separate from post-response windows.

## Open questions for the PI (from the audit)
- sub-20: should it be excluded from EEG work? Recommended: yes, until the events/marker label conflict is resolved.
- Is the continuous [0, 1] confidence target acceptable as the primary endpoint?
- Should ds007474 (LDM-006, ready) be audited in parallel?
