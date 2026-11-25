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

**Source.** The publisher's pages refuse automated access. The bioRxiv preprint of the study (10.1101/2025.09.17.676398v1, posted 2025-09-19) was read instead. It has not been confirmed to match the published version. Only the Abstract, Introduction, Results 2.1–2.3 and the end of Methods were read verbatim; the rest of the Methods came as a machine summary. Only verbatim results are compared below.

**Expected differences:**
- **Exclusions:** the authors excluded 10 participants for fMRI motion and analysed 179. Our behavioral analysis excludes participants who missed more than 10% of trials and analyses 181.
- **Models:** the authors compare an "arbitration mixture" model with reliability signals (RPE/SPE) by in-sample BIC. We compare a fixed-weight hybrid, a win-stay/lose-shift extension and condition-based arbitration on held-out runs and sessions.

**Agreement rules:** the same as above.

| # | Paper (verbatim) | Our comparable quantity | New? |
| --- | --- | --- | --- |
| D1 | 179 analysed of 189 | subjects with data; subjects analysed | no |
| D2 | common transition 70% | fraction of common (non-rare) transitions | yes |
| D3 | reward probability 47.19% vs null 43.08%, paired t p < 0.001 | per-subject mean reward probability of the planet reached vs the mean over the two planets; paired t. The paper's exact definition was not in the verbatim text, so this is our reading. | yes |
| D4 | stay regression: previous outcome β = 0.657, p < 0.001; outcome × transition β = −0.676, p < 0.001 (mixed logistic) | logistic GEE of stay on previous win × previous rare, clustered on subject. Compared on direction and significance, because model and coding differ. | yes |
| D5 | the mixture model beats pure MF and MB (BIC, p < 0.001) | held-out log-likelihood, hybrid vs MF and MB | no |
| D6 | the arbitration mixture beats the fixed-weight mixture (BIC) | not comparable: a different arbitration model and criterion (ours: [RL models](ds007474-two-step-model.md)) | – |

### Results (2026-11-25)

Computed with `ldm ds007474 compare-paper`. The output is not committed because the OSF source has no license; the summary numbers are given here. All 189 subjects with data and their valid trials are included. The preprint analysed 179 subjects after fMRI motion exclusions; which subjects is not stated.

| # | Preprint | Ours | Verdict |
| --- | --- | --- | --- |
| D1 | 189 with data; 179 analysed (fMRI motion) | 189 with data; 181 analysed in our RL models (missed-trial rule) | **agree** on the data; the two exclusion rules differ by design |
| D2 | common transition 70% | 71.9% of valid trials | **agree** (the designed probability vs the realised fraction) |
| D3 | 47.19% vs null 43.08%, p < 0.001 | null **43.09%**; planet reached 45.06% (t = 14.1, p < 10⁻³⁰) | **null agrees; achieved value differs as pre-specified** (−2.1 points). Post hoc explanation: under stimulus contingency the reward follows the chosen ship, not the planet. Using the option the reward depends on gives **47.12%** (t = 20.5), matching the preprint. |
| D4 | previous outcome β = 0.657; outcome × transition β = −0.676; both p < 0.001 | reward 0.49, reward × rare −0.53 (±1 coding, population-averaged GEE); both p < 10⁻²⁹ | **agree** (direction and significance). Population-averaged coefficients are expected to be somewhat smaller than mixed-model ones. |
| D5 | mixture beats MF and MB | hybrid beats MF (+1,084 [718, 1,451]) and MB (+1,116 [680, 1,605]) on held-out runs | **agree** |

**Summary:**
- Every comparable verbatim result of the preprint is reproduced from our independent adapter once the reward-relevant option is used (D3, post hoc). This includes the null reward probability to 0.01 percentage points.
- Arbitration (D6) cannot be compared: the two studies define and test different arbitration models.

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

### Results (2026-11-25)

Run at commit 22914e7 on macOS with the system Python 3.12; about 2 h 10 min including the ds007474 retries (environment 10 min, hash checks 17 min, EEG 35 min, fMRI 57 min). Every signal file matched its manifest: 333 EEG files (111 runs, 35 GB) and 24 fMRI files (38 GB), none failing.

| Report | Differences | Explanation |
| --- | --- | --- |
| behavioral baseline, paper comparison | none | |
| EEG: confirmatory decoding, cardiac behavior, cardiac-control decoding | none | |
| fusion | none | includes the PyTorch model, seeded |
| fMRI: QC | 1 (text only) | `beta_units` wording was shortened in commit 385ddec, after the committed copy was written; the HRF is now its own field. No number differs. The committed copy has been replaced by the regenerated one. |
| fMRI: pilot, timing, derivative QC, derivative pilot | none | |
| **ds007474 primary model comparison** | **11 numbers, within optimizer tolerance** | See below |

**ds007474:**
- **Deviation from the plan.** On the day of the run, the OSF API returned HTTP 500 to about 4 of 5 requests, so a fresh download could not complete. The run used the files fetched on 2026-11-19. All 378 (13.4 MB) matched their manifest by size and SHA-256, and those hashes had been checked against OSF's published ones when fetched. That is the same standard as the EEG and fMRI files. The script now accepts such a folder explicitly (`OSF_LOCAL`).
- **Bug found.** The first attempt crashed because copying to an exFAT drive created macOS `._*` metadata files, which the ds007474 loader read as behavior files. The loader now ignores hidden files (regression test added); the EEG loader already did.
- **Trial table.** The rebuilt table has the same content as the original once rows are sorted. Three columns look different, and the models use none of them:
  - `onset_seconds` and `onset_reference`: the original was built without `--timing`;
  - `choice_side`: 897 missing values are written differently.
- **Model comparison.** The 11 differences:
  - the contingency effect: −0.01183889 in the reference vs −0.01183898 regenerated;
  - the time effect: +0.0332389 in both, differing in the seventh significant digit;
  - the related CI bounds and medians;
  - one subject's best model, hybrid vs hybrid + contingency (counts 49/35 → 50/34; better than reference 61 → 60).
- **Cause, found by bisection.**
  - The local reference was produced by commit 6538889. Every commit up to 8a808b9 reproduces it exactly.
  - Commit f6b3a4b (empirical-Bayes fitting) rewrote the prior penalty from `θ·θ / sd²` to `Σ((θ − μ)/sd)²`. With μ = 0 the two are mathematically identical but round differently in the last bit, so the optimizer converges to a point a little further along within its tolerance.
  - Per-subject parameters move by at most 1e-5 (w, α) and 3e-4 (β); held-out log-likelihoods by at most 5e-4.
  - The subject whose best model changed, MBMF021901, has an exact tie between the two hybrid models: held-out log-likelihood −59.661 for both. Rounding breaks the tie.
  - The current code is deterministic: it gives the clean-room output exactly from the original trial table and environment.
- **Effect on reported results:** none. Every published ds007474 number is the same at its printed precision.

**Conclusion.** Everything committed reproduces from a fresh clone and a fresh environment, using signal files and metadata checked by hash. The ds002158 reports match to 1e-6 in every number. The ds007474 primary comparison matches to about 1e-5, with the drift traced to a single commit and to floating-point rounding. Neither the data nor the environment changed anything. The clean room found one real bug, in loading from exFAT/FAT drives, now fixed.

Reproduce (paths are examples):

```bash
STORE=/data/ldm COMMIT=<sha> WORK=/scratch/cleanroom REF_LOCAL=/data/ldm-ref \
    OSF_LOCAL=/data/ldm/ds007474_osf zsh reproduce/cleanroom.sh   # omit OSF_LOCAL to fetch from OSF
```
