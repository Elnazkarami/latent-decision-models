# Validation

This page records two checks that the repository's results do not depend on how they were produced here:
1. a comparison with what the original authors published from the same data;
2. a clean-room reproduction of the committed reports.

The comparisons and agreement rules below were committed before any new number was computed for them (2026-11-25). Numbers that already existed in the repository are marked as such.

## 1. Comparison with the original publications

### ds002158 vs Pereira et al. (2020), PNAS 117:8382

**Source.** The main text (PMC7165419) was read. Its supplementary material was not, so values that appear only there are not compared.

**Expected differences.** Our pipeline is independent of the authors' and differs in known ways:
- **Trials:** observation trials with an accidental button press were excluded by the authors, but these presses are not recorded in `events.tsv`.
- **Behavioral models:** the authors used mixed models; we use per-subject and population-averaged (GEE) models.
- **EEG:** the authors used zero-phase filtering, ICA and per-sample regression or decoding on independent components. We use causal filtering (which delays activity by up to ~0.1 s), template subtraction, and windowed decoding on channels.

**Agreement rules:**
- **Descriptives:** "agree" if our value lies within the paper's reported ± interval around its value.
- **Tests:** "agree" if the effect has the same direction and falls on the same side of p = 0.05.
- **EEG timing:** qualitative. Results are compared on two points: whether a confidence signal is present in each period, and how the conditions are ordered in time.
- Any disagreement is reported with a candidate explanation, never tuned away.

| # | Paper result | Our comparable quantity | New? |
| --- | --- | --- | --- |
| P1 | 20 participants analysed | subjects in the trial table | no |
| P2 | Trials kept per condition 119 ± 5 (active), 118 ± 5 (observed) | mean valid trials per subject and condition | no (audit: 121.8 / 121.2) |
| P3 | First-order accuracy 71.2% ± 1.0% | mean over subjects of accuracy on trials with a response | yes |
| P4 | Dot difference 13.1 ± 1.7 | mean over subjects of the mean evidence | yes |
| P5 | RT 385 ± 8 ms | mean over subjects of the mean RT, valid active trials | yes |
| P6 | Mean confidence, active vs observed: t(19) = −0.19, p = 0.85 | paired t on per-subject means, valid trials | yes |
| P7 | Confidence predicts accuracy more steeply for active decisions: interaction z = −2.90, p = 0.004 | GEE interaction (existing: −1.73, p = 0.006); paired Wilcoxon on per-subject logistic slopes | GEE no; slopes yes |
| P8 | Type-2 AROC 0.92 ± 0.02 (active) vs 0.90 ± 0.03 (observed), Wilcoxon V = 163, p = 0.03 | per-subject type-2 AUROC, paired Wilcoxon | yes |
| P9 | Active: confidence decreases with RT on correct trials and increases with RT on errors; observed: no confidence effect on RT | per-subject Spearman(RT, confidence) by accuracy, Wilcoxon vs 0 | yes |
| P10 | EEG, active: confidence correlates from 68 ms after the response; decoding peaks ~100 ms and ~320 ms | response-locked time course (`reports/ds002158-eeg-timecourse.json`) | no |
| P11 | EEG, observed: no confidence correlate before 200 ms; decoding peak ~350 ms | same | no |
| P12 | fMRI (EEG-informed GLM; pMFC, IFG, insula, aPFC) | not comparable: trial-level fMRI was closed here ([fMRI pilot](ds002158-fmri-pilot.md)) | – |

### ds007474 vs Ding, Cockburn et al. (2026), Cell Reports 45(6)

The comparison is **not done** because the paper could not be read: the publisher pages refused automated access and the preprint server was rate-limited. It needs a local copy of the paper or the preprint (bioRxiv 10.1101/2025.09.17.676398).

## 2. Clean-room reproduction

To follow after part 1.
