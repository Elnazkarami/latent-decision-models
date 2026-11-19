# ds002158 fMRI audit: derivatives and trial-level design (metadata only)

- **Date:** 2026-11-19
- **Snapshot:** OpenNeuro ds002158 **v1.0.2**
- **Data read:** no image data was downloaded.
  - The file listing.
  - The first 4–16 KB (header) of every raw `bold.nii.gz` and derivative `preproc.nii`, via HTTP range requests, capped in code at 64 KB.
  - The 117 realignment-parameter text files (4.3 MB).
  - The already-fetched `events.tsv`.
- **Machine-readable:** [`reports/ds002158-fmri-audit.json`](../reports/ds002158-fmri-audit.json) (inventory, headers, motion) and [`reports/ds002158-fmri-design.json`](../reports/ds002158-fmri-design.json) (design diagnostics).

## Verdict

| Use | Decision | Reason |
| --- | --- | --- |
| Published derivative (`derivatives/fmriprep`) for trial-level GLM / ROI work | **GO, with known preprocessing** | The headers state the pipeline: SPM DARTEL normalisation to MNI-152, 1.5 mm isotropic, 5 mm FWHM smoothing. Volume counts match raw in 117/117 runs, and motion files exist for 117 runs. TR must be supplied (headers store 0). Slice-timing correction is undocumented. |
| **Single-trial (whole-trial) BOLD estimates** | **GO** | Trials are ≥ 9 s apart. Single-trial regressors are nearly orthogonal (LSS VIF ≤ 1.2; neighbouring trials \|r\| ≤ 0.25). |
| **Separating pre-decision from response-locked BOLD** | **NO-GO** | Stimulus and response are 0.4 s apart. Their regressors correlate at r = 0.95–0.98 (VIF 14–52, and 100–2,000 with an HRF time derivative). Any trial-level fMRI estimate is a whole-trial measure, not a pre-decision one. |
| Separating a post-decision (confidence-report) event | **CONDITIONAL** | It is estimable (VIF 1.3–3.4, ≤ 4.1 with derivative), but only under an assumed event time. The confidence-scale onset is not recorded (see unknowns). |
| Raw-data reprocessing (e.g. fMRIPrep) | Not needed for a first pass | Only if slice timing or the DARTEL template choice turns out to matter. |

## Derivative: what it actually is

The `descrip` field of all 118 derivative headers reads **"Smoothed (5x5x5) Dartel normed"**. That resolves hypothesis 2 of the [main audit](ds002158-audit.md).

| Property | Raw `bold.nii.gz` (118) | Derivative `preproc.nii` (118) |
| --- | --- | --- |
| Grid | 108 × 108 × 64 | 121 × 145 × 121 (SPM12 MNI template grid) |
| Voxel size | 1.99 × 1.99 × 2.0 mm | 1.5 mm isotropic |
| Space (sform / qform code) | scanner | **MNI-152** (code 4), origin at voxel (60, 84, 48) |
| TR in header | 1.28 s | **0** (must come from the raw data) |
| Data type | int16 | float32, uncompressed, ≈ 3.2 GB per run |
| Volumes | 373 (108 runs), 375 (sub-02, 6), 371 (sub-04 runs 3–6) | identical to raw in every run with both |

**Other contents:**
- **What the folder holds:** only `*_headmove.txt` (6 realignment parameters per volume: 3 translations in mm, 3 rotations in rad) and copies of `events.tsv`.
- **What it lacks:** no dataset description, brain mask, anatomy, confounds table or provenance file.
- **The README's "FMRI PREPROCESSING" section describes the authors' GLM,** not the preprocessing. It used stick regressors at stimulus onset, separately for active and observed trials, with parametric confidence, RT and evidence modulators, six motion regressors, and AAL parcellation.

## Run inventory and anomalies

| Run | Raw BOLD | Derivative | Behavior | Note |
| --- | --- | --- | --- | --- |
| sub-14 run-005 | `func/rejected/` | `func/rejected/` (no motion file) | not loaded by the adapter | rejected by the authors, consistently |
| **sub-20 run-007** | normal folder | **`func/reject/`** | 48 trials, loaded | **derivative rejected, raw not:** the reason is undocumented. Exclude from fMRI work unless resolved. |
| sub-19 run-001 | present | **missing** | 38 trials | no derivative |
| sub-19 run-006 | **missing** | present | 48 trials | derivative without raw; the volume count can't be cross-checked |

**Scan coverage:** the last stimulus falls 7.8–110 s before the end of the scan (median 15.0 s). The tightest run is sub-09 run-005 (7.8 s), so the last trial's HRF is truncated after its peak.

## Head motion (realignment parameters; FD after Power et al. 2012, 50 mm radius)

- **Mean framewise displacement (FD) per run:** median 0.16 mm, 90th percentile 0.22 mm, max 0.46 mm.
- **Maximum translation / rotation over all runs:** 4.8 mm / 2.7°.
- **Runs with more than 20% of volumes above FD 0.5 mm:**
  - **sub-07 run-004**: 28%; maximum translation 4.8 mm, maximum FD 3.7 mm.
  - **sub-20 run-001**: 29%.
- **sub-20** has the highest mean FD of any subject (0.33 mm averaged over runs; 0.29–0.37 mm per run).

These are summaries only. Scrubbing or exclusion thresholds are a decision for the analysis, not the audit.

## Trial-level design diagnostics

These are computed from event timing alone (`ldm ds002158 fmri-design`):
- **Model:** nilearn design matrices with a Glover HRF and a 1/128 Hz cosine high-pass, at TR 1.28 s.
- **Trials included:** all trials, including those the authors flagged, because they happened.

| Quantity (118 runs) | Min | Median | 95th pct | Max |
| --- | --- | --- | --- | --- |
| Shortest onset gap (s) | 8.99 | 9.00 | 9.00 | 9.04 |
| LSA VIF per trial, median | 1.27 | 1.29 | 1.29 | 1.30 |
| LSS VIF per trial, max | 1.10 | 1.12 | 1.14 | 1.20 |
| \|r\| neighbouring trial regressors, max | 0.24 | 0.24 | 0.24 | 0.25 |
| r(stimulus, response) | 0.945 | 0.965 | 0.972 | 0.975 |
| VIF stimulus / response | 14 / 13 | 27 / 28 | 36 / 41 | 43 / 52 |
| … with HRF time derivative | 101 / 104 | 292 / 295 | 836 / 840 | 2,072 / 2,088 |
| VIF post-decision event | 1.30 | 1.98 | 2.76 | 3.35 |
| … with HRF time derivative | 1.44 | 2.15 | 3.24 | 4.14 |

- **Abbreviations:** LSA is one regressor per trial. LSS is each trial against all other trials of each condition (Mumford et al. 2012). A VIF above about 5–10 means the estimate is dominated by shared variance.
- **Single-trial betas are well conditioned.**
- **The stimulus- and response-locked responses cannot be separated:** the response follows the stimulus by 0.41 s on average.
- **Correlations with the post-decision event change sign across runs** (−0.36 to +0.54). Trials recur every ~9.5 s, so a 2–3 s shift moves the regressor partly out of phase. VIF, not correlation, is the relevant measure there.

**Consequence for the project rule on neural windows:** trial-level fMRI features from this dataset are **whole-trial** measures. They mix evidence accumulation, the response, and at least the start of the confidence report. They must not be described as pre-decision or prospective.

## Unknowns

1. **Slice-timing correction.** The raw sidecar has no `SliceTiming` (multiband 4, 64 slices). The derivative's processing history is not recorded. With TR 1.28 s, the residual timing error is at most about one TR.
2. **Confidence-scale onset.** `events.tsv` records `response_time2` ("time to rate confidence"; the sidecar warns the scale moves automatically) but not when the scale appeared. The post-decision diagnostics assume the rating time is measured from the response. The EEG marker codes `S171`/`S43` at about +1.47 s may be this onset; that is unverified (main audit, hypothesis 3).
3. **Why sub-20 run-007's derivative is in `reject/`** while its raw data and behavior are not.
4. **DARTEL template:** a study-specific template is likely, given DARTEL. Group-space comparability with standard MNI atlases is approximate.

## Decisions for the next step (please review)

1. **Trial-level fMRI features will use the published derivative**, not our own preprocessing. Inputs: one LSS beta per trial, stimulus-locked, from a GLM with the 6 realignment parameters, plus atlas ROIs on the MNI grid.
   - This needs **downloading derivative images: about 3.2 GB per run, 19 GB per subject, 374 GB in total**. That is signal data, which **requires your explicit approval**.
   - Smaller option: a 2-subject pilot, about 38 GB.
   - Alternative: ROI-restricted HTTP range reads, contiguous z-slabs per volume, roughly 15–40% of a run depending on the ROI. This is still signal data, so it also needs approval.
2. **fMRI exclusions:** exclude sub-20 run-007 (rejected derivative). Flag sub-07 run-004 for motion. sub-20 is already excluded from EEG work and has the highest motion; decide whether to exclude it from fMRI too.
3. **Neural features are described as whole-trial measures** (see above). No pre-decision vs response contrast is attempted with fMRI.

## Reproduce

```bash
pip install -e '.[fmri]'                                         # nibabel + nilearn (optional extra)
ldm ds002158 fetch-metadata --dest data/ds002158                  # events (if not already present)
ldm ds002158 trials --bids-root data/ds002158 --out data/ds002158_trials.csv
ldm ds002158 fmri-audit --dest data/ds002158 --out reports/ds002158-fmri-audit.json   # ~2 min; headers + 4.3 MB text
ldm ds002158 fmri-design --trials data/ds002158_trials.csv --audit-table reports/ds002158-fmri-audit.csv
```
