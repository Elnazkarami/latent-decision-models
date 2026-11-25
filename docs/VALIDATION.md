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

### Results (2026-11-25)

Report: [`reports/ds002158-paper-comparison.json`](../reports/ds002158-paper-comparison.json) (`ldm ds002158 compare-paper`). Our ± values are SDs over subjects.

| # | Paper | Ours | Verdict |
| --- | --- | --- | --- |
| P1 | 20 participants | 20 | **agree** |
| P2 | 119 ± 5 / 118 ± 5 trials | 121.9 ± 15.6 / 121.2 ± 16.8 | **agree** (within ±5). The authors also removed observation-period button presses, which are not in `events.tsv`. |
| P3 | accuracy 71.2% ± 1.0 | 72.8% ± 2.3 (all answered trials) | **differs** by 1.6 points. Post hoc explanation: on valid trials (RT 200–500 ms) ours is **71.5%**, inside the paper's interval. The paper evidently computed accuracy after its RT exclusions. |
| P4 | dot difference 13.1 ± 1.7 | 13.2 ± 3.8 | **agree** |
| P5 | RT 385 ± 8 ms | 384.5 ± 19.1 ms | **agree** |
| P6 | active vs observed mean confidence: t(19) = −0.19, p = 0.85 | t(19) = −0.11, p = 0.92 (0.629 vs 0.629) | **agree** |
| P7 | steeper confidence–accuracy relation for active decisions: z = −2.90, p = 0.004 | per-subject slopes steeper for active in 15/20, Wilcoxon V = 189, p = 0.0009; GEE −1.73, p = 0.006 | **agree** |
| P8 | type-2 AROC 0.92 ± 0.02 vs 0.90 ± 0.03, V = 163, p = 0.03 | 0.955 ± 0.038 vs 0.941 ± 0.045, **V = 163, p = 0.030** | **test agrees exactly; levels differ.** Ours is ~0.035–0.04 higher in both conditions. The difference is unexplained: the paper's main text does not say how AROC was computed (e.g. binned confidence, or which trials). |
| P9 | active: confidence falls with RT on correct trials, rises on errors; observed: no confidence effect | active correct r = −0.09 (p = 0.0001), active error r = +0.14 (p = 0.003); observed, pooled r = +0.14 (p < 0.001) | **active agrees; observed differs as pre-specified.** Post hoc explanation: the paper's model includes accuracy, but our pre-specified quantity pooled correct and error trials. Split by accuracy, the observed relation is null (correct −0.03, p = 0.41; error +0.04, p = 0.39), matching the paper. |
| P10 | active EEG: correlates from 68 ms; decoding peaks ~100 and ~320 ms | earliest window with uncorrected p < 0.05 is 0.10–0.20 s (r = 0.07); strongest 0.15–0.30 s (r = 0.13) and 0.35–0.60 s (r = 0.09–0.12) | **partly agrees.** Both show an early and a later post-response component, but ours begins ~50–100 ms later. Our causal filter delays activity by up to ~0.1 s, and the paper's zero-phase filter shifts activity earlier, so onsets cannot be compared more closely. |
| P11 | observed EEG: nothing before 200 ms; peak ~350 ms | nothing after the response before 0.30–0.40 s; strongest 0.45–0.55 s (r = 0.09, p = 0.0003) | **agree** (no early component; a later, delayed signal) |
| P12 | fMRI | – | not comparable |

**Summary:**
- **Agreement:** every inferential result in the paper's main text that we could compare agrees in direction and significance once the paper's model is matched (P6–P9, P11). This includes an exact match of the type-2 AUROC Wilcoxon statistic. The descriptives agree within the paper's intervals once its RT exclusions are applied (P3).
- **Unexplained:** the type-2 AUROC levels are higher in ours.
- **EEG:** onset times are not comparable because the two pipelines filter in opposite directions in time.
- **Pre-specified quantities:** two did not match the paper's definitions (P3, P9). They are reported as pre-specified, with the explanation labelled post hoc.

### ds007474 vs Ding, Cockburn et al. (2026), Cell Reports 45(6)

The comparison is **not done** because the paper could not be read: the publisher pages refused automated access and the preprint server was rate-limited. It needs a local copy of the paper or the preprint (bioRxiv 10.1101/2025.09.17.676398).

## 2. Clean-room reproduction

**Question.** Does a fresh copy of the repository regenerate the committed reports, or do they depend on state on the machine where they were first made? Plan fixed before the run.

**Procedure** ([`reproduce/cleanroom.sh`](../reproduce/cleanroom.sh)):
1. **Clone and environment:** fresh clone at a pinned commit; fresh virtual environment from the system Python 3.12, with packages from `requirements-lock.txt`.
2. **Metadata:** fresh download of the ds002158 metadata (events and sidecars) from OpenNeuro; fresh ds007474 behavior from OSF.
3. **Signal files:** the large files already on external storage (35 GB raw EEG, 38 GB fMRI derivative) are reused only after their size and SHA-256 match the fetch manifests written when they were downloaded. They are not re-downloaded, which would take about 14 h at the measured ~1.5 MB/s.
4. **Pipeline:** everything downstream is recomputed: trial table, behavioral baseline, paper comparison, EEG cleaning and epochs for 19 subjects, confirmatory decoding, cardiac controls, fusion, the fMRI pilot from the images, and the ds007474 primary model comparison.
5. **Check:** `ldm compare-reports` compares every regenerated report with the committed copy. The ds007474 and behavioral-baseline outputs are not committed, so they are compared with the local copies made when the results were first reported.

**Reports compared (12):**
- behavioral baseline;
- paper comparison;
- confirmatory EEG decoding;
- cardiac behavior;
- cardiac-control decoding;
- fusion;
- fMRI QC, pilot and timing, plus the derivative variant's QC and pilot;
- ds007474 primary model comparison.

**Agreement rule:**
- Numbers must match within relative 1e-6 (absolute 1e-9). Strings, booleans and list lengths must match exactly.
- Keys added to a report format after a report was committed are allowed. Every value in the committed report must still match.
- Any difference is reported and explained, never accepted silently.
