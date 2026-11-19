# latent-decision-models

[![CI](https://github.com/Elnazkarami/latent-decision-models/actions/workflows/ci.yml/badge.svg)](https://github.com/Elnazkarami/latent-decision-models/actions/workflows/ci.yml)

**Research software in active development: not a validated neuroscience analysis.**

A reproducible Python library for trial-aligned modeling of decision confidence, belief updating and latent cognitive variables from behavior, EEG and fMRI. It provides task adapters with strict validation, subject-disjoint evaluation, and audits of public datasets with full provenance.

Python package `latent_decision_models` · CLI `ldm` · repository `latent-decision-models`.

## What you can do with it

- **Load and validate** trial tables with verified column semantics and row-level problem reports. There are adapters for ds002158 (confidence; BIDS events) and ds007474 (two-step task; behavior and event timing from OSF, because the BIDS events files are empty).
- **Check EEG/behavior alignment** from BrainVision marker files. Matching is by timing, never by filename or row order.
- **Evaluate confidence models fairly**, with subject-disjoint cross-validation, within-subject held-out runs, calibration, participant-bootstrap CIs and type-2 AUROC.
- **Fit a computational confidence model** (second-order signal detection; committed vs observed decisions). It includes simulation, parameter and model recovery, and posterior predictive checks.
- **Fit model-free / model-based / hybrid reinforcement learning** to the two-step task. Agents can be replayed on each participant's real task schedule. Recovery studies, held-out-run and held-out-subject comparison, and posterior predictive stay-probability checks are included.
- **Clean EEG recorded inside an MRI scanner and decode from it.** Gradient and heartbeat (BCG) artifacts are removed by template subtraction, with causal filters so that pre-decision features never contain later activity. Trial epochs are locked to verified markers, and decoding uses leave-one-run-out or cross-subject evaluation with within-run permutation tests.
- **Audit fMRI data without downloading images.** It reads NIfTI headers through capped HTTP range requests, builds a raw/derivative run inventory, and summarises head motion. It also computes design diagnostics from event timing alone: single-trial VIF (LSA/LSS) and whether within-trial phases can be separated.
- **Fetch files safely.** It downloads text metadata from pinned OpenNeuro snapshots (refusing signal files), and allow-listed, size-capped, SHA-256-checked files from OSF projects. Signal data (raw EEG) is fetched only by a separate command that needs an explicit size budget.

**Limitations:**
- EEG work so far covers two ds002158 subjects; fMRI signal processing and deep learning are not implemented yet.
- Dynamic (RT-based) and hierarchical confidence models are not implemented yet.
- The two-step RL models use fixed, instructed transitions; trial-wise arbitration models are not implemented yet.
- No population-level neural findings are claimed.

## Install

```bash
python -m pip install -e '.[dev]'      # core + tests/lint
# optional extras: '.[eeg]' (MNE), '.[fmri]' (Nilearn/PyBIDS), '.[deep]' (PyTorch)
pytest -q
```

## Quick start

**Synthetic smoke test** (no network; results are synthetic):

```bash
ldm simulate --out examples/synthetic_trials.csv --subjects 12 --trials-per-subject 35
ldm validate --input examples/synthetic_trials.csv
ldm benchmark --config configs/synthetic_confidence.yaml
```

**Real ds002158 behavioral pipeline.** It downloads about 0.3 MB of text metadata and no imaging data:

```bash
ldm ds002158 fetch-metadata --dest data/ds002158
ldm ds002158 trials --bids-root data/ds002158 --out data/ds002158_trials.csv --issues data/ds002158_issues.csv
ldm confidence-baseline --config configs/ds002158_confidence_baseline.yaml
```

**Computational confidence model.** It runs a synthetic recovery study first, then a per-subject fit and a held-out-run comparison:

```bash
ldm confidence-model recovery --config configs/confidence_model_recovery.yaml
ldm confidence-model compare --config configs/ds002158_confidence_model.yaml
```

**ds007474 two-step task.** It downloads about 13 MB of behavior and timing files from OSF and no MRI data. The timing files are MATLAB tables, so `--timing` needs the optional `[ds007474]` extra:

```bash
ldm ds007474 fetch-behavior --dest data/ds007474_osf
pip install -e '.[ds007474]'
ldm ds007474 trials --osf-root data/ds007474_osf --out data/ds007474_trials.csv --timing
```

**Two-step reinforcement-learning models** (after the two commands above). Recovery runs on synthetic and real task schedules (simulated choices), then the real-data comparison:

```bash
ldm two-step-model recovery --config configs/two_step_recovery_fitted_range.yaml
ldm two-step-model compare --config configs/ds007474_two_step_model.yaml
```

**fMRI metadata audit (ds002158).** It reads image headers via range requests and 4.3 MB of motion text; no image data is downloaded:

```bash
pip install -e '.[fmri]'
ldm ds002158 fmri-audit --dest data/ds002158 --out reports/ds002158-fmri-audit.json
ldm ds002158 fmri-design --trials data/ds002158_trials.csv --audit-table reports/ds002158-fmri-audit.csv
```

**ds002158 EEG (raw signal data, about 2 GB per subject).** Pairing, in-scanner cleaning, epochs, then the pre-specified decoding plan:

```bash
pip install -e '.[eeg]'
ldm ds002158 fetch-eeg --subjects sub-02 sub-04 --dest data/ds002158 --approved-gb 4.2 --dry-run   # size first
ldm ds002158 fetch-eeg --subjects sub-02 sub-04 --dest data/ds002158 --approved-gb 4.2
ldm ds002158 eeg-preprocess --bids-root data/ds002158 --trials data/ds002158_trials.csv \
    --subjects sub-02 sub-04 --out-dir data/derivatives/ldm-eeg --report reports/ds002158-eeg-preprocess.json
ldm ds002158 eeg-decode --config configs/ds002158_eeg_decoding.yaml --out reports/ds002158-eeg-decoding.json
```

The full sample is ~35 GB. Raw data can live on an external drive: point `--dest`, `--bids-root` and `--out-dir` there, or make `data/` a symlink to it (it is git-ignored either way). `fetch-eeg` refuses to start if the download would leave less than `--min-free-gb` (default 20 GB) free on the destination disk.

**Check one run's EEG/behavior alignment.** It needs only that run's `.vhdr` and `.vmrk` text files:

```bash
ldm ds002158 align-eeg --events <..._events.tsv> --vmrk <..._eeg.vmrk>
```

## What's in the package

| Module | Purpose |
| --- | --- |
| `schema` | Generic trial-table validation (unique trial identity, onsets, RT) |
| `datasets.ds002158` | Verified codebook and strict events parser for ds002158 (experimental). Aligns EEG markers by **timing**, never by row order or filename. |
| `datasets.ds007474` | ds007474 adapter: OSF behavior CSV to trial table, OSF-to-BIDS subject mapping, event timing from the `.mat` files cross-checked against the behavior |
| `datasets.ds002158_fmri` | ds002158 fMRI raw/derivative inventory, header audit and realignment-parameter (FD) summary |
| `datasets.ds002158_eeg` | ds002158 raw EEG: run pairing by marker timing, in-scanner cleaning pipeline, stimulus- and response-locked epochs, analysis windows, decoding runner |
| `eeg_clean` | Sliding-window template subtraction (MR gradient, BCG), heartbeat detection (MNE), causal band-pass and decimation |
| `eeg_decoding` | Binned window features; leave-one-run-out and cross-subject decoding (logistic / ridge, regularisation tuned inside training folds), training-fold-only residualisation on covariates, within-run permutation tests, Holm correction |
| `fmri_design` | Design diagnostics from event timing (nilearn): single-trial VIF under LSA/LSS, neighbouring-trial correlation, separability of stimulus / response / post-decision regressors |
| `io.nifti` | NIfTI-1 header summary from a byte prefix (nibabel) |
| `io.osf` | OSF listing and allow-listed, capped, SHA-256-verified downloads |
| `io.openneuro` | Pinned OpenNeuro snapshots: metadata fetch (refuses signal files), header range reads, and budgeted signal downloads (whole request refused above the approved size) |
| `io.brainvision` | Dependency-free `.vhdr`/`.vmrk` marker reader; `read_raw` opens recordings with MNE even when the header names stale data files |
| `confidence_baseline` | Nested confidence models with subject-disjoint CV, within-subject run-held-out CV, calibration, participant bootstrap and type-2 AUROC |
| `confidence_model` | Second-order signal-detection confidence model for committed vs observed decisions. Includes simulation, censored-rating likelihood, parameter and model recovery (per subject and group level), held-out-run comparison and posterior predictive checks |
| `two_step_model` | Model-free / model-based / hybrid RL for the two-step task. Includes schedule replay of simulated agents, MAP fitting, leave-one-run-out and subject-k-fold held-out comparison, parameter and group-level model recovery, an order-balanced contingency contrast and stay-probability checks |
| `metacognition` | Group metacognitive sensitivity: GEE logistic of accuracy on confidence × condition |
| `benchmark`, `synthetic` | Synthetic pipeline smoke test |
| `provenance` | SHA-256 and git-blob checksums |

## Datasets (not bundled)

- **Primary: [OpenNeuro ds002158](https://openneuro.org/datasets/ds002158) v1.0.2** (CC0): simultaneous EEG/fMRI with confidence after committed vs observed decisions, 20 subjects. Pereira et al. (2020) *PNAS* 117:8382, [doi:10.1073/pnas.1918335117](https://doi.org/10.1073/pnas.1918335117). *This data was obtained from the OpenNeuro database.*
- **Secondary: [OpenNeuro ds007474](https://openneuro.org/datasets/ds007474) v1.0.5** (CC0): two-step task fMRI, 189 subjects. Ding, Cockburn et al. (2026) *Cell Reports* 45(6), [doi:10.1016/j.celrep.2026.117454](https://doi.org/10.1016/j.celrep.2026.117454).
  - Behavior and timing live on the linked [OSF project](https://osf.io/ctfzd), which **declares no license**. This repo therefore stores none of its files, and the tests are synthetic.

No participant data are redistributed apart from a minimal CC0 test fixture (one 48-trial events file and marker text; see [`tests/fixtures/ds002158/PROVENANCE.md`](tests/fixtures/ds002158/PROVENANCE.md)). `data/` is git-ignored.

## Findings on ds002158

Details and evidence: [audit](docs/ds002158-audit.md) · [EEG](docs/ds002158-eeg.md) · [fMRI audit](docs/ds002158-fmri-audit.md) · [behavioral baselines](docs/ds002158-behavioral-baseline.md) · [computational model](docs/ds002158-confidence-model.md).

- **Behavior → fMRI timing:** events `onset` = stimulus marker − first scanner-volume trigger, with 0.0 ms residual. Trigger count equals the BOLD volume count.
- **Behavior → EEG:** 22/25 sampled runs are fully verified.
  - **sub-12's EEG run filenames are shifted**, so filename pairing would be wrong.
  - **sub-20's events and EEG markers disagree on stimulus and response side.** This is unresolved, so sub-20 is excluded from EEG work.
- **EEG, two subjects (sub-02, sub-04):**
  - All 12 runs pair with their events by marker timing and verify. The raw headers name the original lab files, so MNE cannot open them as published; `io.brainvision.read_raw` handles this.
  - MR gradient and heartbeat artifacts are still in the raw files. Template subtraction removes them (slice-frequency power down 25- to 1000-fold, heartbeat-locked activity down 80–96%).
  - **Positive control passes:** response side decodes from response-locked and post-response activity (AUC 0.78–0.99 within subject, 0.67–0.94 across subjects) and is at chance before the decision.
  - **Confidence:** no pre-specified test survives correction (best: Holm-adjusted p = 0.18). Accuracy decodes after own responses (AUC 0.64–0.68).
- **fMRI derivative:** it is SPM DARTEL-normalised to MNI-152, 1.5 mm isotropic and 5 mm FWHM smoothed; the image headers say so, although the dataset docs do not.
  - Its headers store TR = 0; use 1.28 s.
  - sub-20 run-007's derivative was moved to a `reject/` folder, but its raw data and behavior were not.
- **Trial-level fMRI:** single-trial responses are estimable (trials ≥ 9 s apart; LSS VIF ≤ 1.2). Pre-decision and response-locked activity are **not** separable (r ≈ 0.96, 0.4 s apart), so fMRI trial features are whole-trial measures.
- **Observed trials** are shuffled replays of the participant's own preceding active block. Their "RT" is the replayed virtual-hand latency, not a participant RT.
- **Behavioral baseline (post-decision, held-out subjects):** error vs null 0.136 vs 0.222, almost entirely from objective accuracy. Predictions are well calibrated (slope 0.99). Per-person models add only a little (MAE 0.010 lower).
- **Metacognition (committed vs observed):** confidence tracks accuracy less steeply for observed decisions. GEE interaction −1.73 [−2.95, −0.51], p = 0.006; robust to excluding sub-20. The direction matches the paper. These are post-hoc, population-averaged estimates, and the design is paired (replayed trials).
- **Computational model (second-order SDT):**
  - Synthetic recovery shows that the decision–confidence correlation ρ is **not identifiable** per subject at about 240 trials.
  - The test of whether observed choices are evaluated as one's own (`agency_blind` vs `normative`) is inconclusive on held-out runs: Δ log-likelihood +19 [−60, +99].
  - In-sample BIC favours `agency_blind`, but recovery shows that criterion is biased in the fitted region.
  - No variant reproduces the active > observed sensitivity gap.
  - Conclusion: no mechanistic claim.

## Findings on ds007474

Details: [audit](docs/ds007474-audit.md) · [JSON](reports/ds007474-audit.json) · [RL models](docs/ds007474-two-step-model.md).

- **Missing BIDS events:** all 378 BIDS `events.tsv` files are header-only. Behavior and timing come from OSF.
  - The OSF files map one-to-one onto the BIDS subjects (189/189).
  - Their event timing agrees exactly with the behavior files.
- **Timing reference:** event onsets are relative to each run's first scanner pulse (TR 1.12 s, 650 volumes).
  - One run (sub-MBMF025002 run-1) stops after 381 volumes, so 30 of its trials have no BOLD.
- **No usable derivatives:** the published derivatives are group-level T maps only. Trial-level fMRI work would need preprocessing of 259 GB of raw BOLD.
- **Sanity check:** stay probabilities show the standard reward × transition pattern. This checks the coding; it is not a finding.
- **Task design fully verified:** every shown outcome is reproduced from the trial schedule stored in the behavior files (0 mismatches in 28,512 trials). Under stimulus contingency the reward follows the chosen ship; under state contingency it follows the planet reached.
- **RL models (behavior, 181 subjects):**
  - On held-out runs, a hybrid of model-free and model-based learning predicts choices better than either alone. Δ log-likelihood vs MF is +1,084 [718, 1,451]; vs MB +1,116 [680, 1,605].
  - The ranking holds on held-out subjects and with no exclusions; learning from points instead of wins leaves the hybrid vs MB difference inconclusive.
  - Recovery over the fitted parameter range shows the group comparison is not biased toward the hybrid.
- **No reliable contingency effect on the MB weight:** it does not reliably increase when model-based control is the rewarded strategy. Order-balanced Δw is −0.012 [−0.047, 0.023] in the scanner session and +0.032 [−0.006, 0.072] in a separate session. A planted effect of 0.15 is recovered in both.
- **Follow-up (exploratory):** adding one-trial win-stay/lose-shift terms fixes the hybrid's underprediction of the reward × transition stay interaction.
  - It improves prediction of held-out runs (+400 [61, 721]) and of a **separate behavioral session** scored with parameters fitted on the scanner session (+318 [194, 454]).
  - Recovery shows this selection makes no false positives.
  - Separate MF/MB learning rates add nothing.

## Principles

- Subject-disjoint evaluation by default. Preprocessing is fit only on training folds.
- No dataset column mapping without verification against the actual files. Facts in docs are labelled as verified or hypothesis.
- Pre-decision, response-locked and post-decision neural windows are kept separate. No prospective claims are made from post-response features.
- Signal data is never downloaded implicitly: the metadata fetcher refuses signal files, and signal downloads need an explicit size budget. Neuroimaging dependencies are optional extras.
- Filters applied before a pre-decision window are causal, so no later activity can leak into it.

Contributor rules are in [`AGENTS.md`](AGENTS.md); the research plan is in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Changelog

- **2026-11-19:** ds002158 EEG (two subjects):
  - `ldm ds002158 fetch-eeg` (budgeted signal download), `eeg-preprocess` (pairing, gradient and BCG removal, causal filtering, epochs) and `eeg-decode` (pre-specified within- and cross-subject decoding).
  - New `eeg_clean`, `eeg_decoding` and `datasets.ds002158_eeg` modules; `io.brainvision.read_raw`; `io.openneuro.fetch_signal_files`.
  - A CI job for the optional `[eeg]` extra.
  - `fetch-eeg --min-free-gb`: downloads never fill the destination disk. Volume-trigger grid tolerance raised to 10 samples after checking the artifact position in the full sample.

- **2026-11-19:** ds002158 fMRI metadata audit:
  - `ldm ds002158 fmri-audit` (header range reads, run inventory, motion) and `ldm ds002158 fmri-design` (design diagnostics).
  - New `io.nifti`, `fmri_design` and `datasets.ds002158_fmri` modules.
  - A CI job for the optional `[fmri]` extra.

- **2026-11-19:** Two-step RL follow-up:
  - Win-stay/lose-shift and two-learning-rate extensions, with recovery over the fitted range.
  - Out-of-session replication on the structural-session trials, including a structural-session contingency contrast with a known-answer check.
  - Fix: the contingency contrast now excludes subjects who saw only one contingency.

- **2026-11-19:** Two-step RL models:
  - MF, MB, hybrid and contingency-specific hybrid models, with replay on real task schedules.
  - Recovery on synthetic schedules, real schedules and the fitted range, with group-level model recovery and a known-answer check of the contingency contrast.
  - Real-data comparison: pre-specified primary analysis plus two sensitivity analyses.
  - The ds007474 adapter now exposes the full trial schedule and checks the reward rules on every file.
  - New `ldm two-step-model recovery|compare` commands.

- **2026-11-19:** Audit of ds007474. Added the ds007474 adapter (OSF behavior and event timing), the OSF fetcher, `ldm ds007474 fetch-behavior|trials`, and the optional `[ds007474]` extra (`mat-io`).

- **2026-11-19:** Computational confidence model:
  - Second-order SDT model with four observed-trial readout variants.
  - Tests that the likelihood normalizes and that known parameters are recovered.
  - Parameter and model recovery at a priori and fitted-region ranges, plus group-level model recovery.
  - Real-data held-out-run comparison and posterior predictive checks, with a sub-20 sensitivity run.
  - New `ldm confidence-model recovery|compare` commands.

- **2026-11-18:** Confidence baselines, second pass: calibration metrics, within-subject run-held-out evaluation, GEE metacognition model (`ldm confidence-baseline` now reports all three); statsmodels added as a core dependency.
- **2026-11-18:** Audit of ds002158; ds002158 adapter, EEG marker alignment, metadata-only OpenNeuro fetcher; first-pass behavioral confidence baselines.
- **2026-11-18:** Initial package: package, synthetic generator, validator, subject-disjoint benchmark, CI.

## License

MIT (see [`LICENSE`](LICENSE)). Dataset licenses apply separately.
