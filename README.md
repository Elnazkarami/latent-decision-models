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
- **Fetch small files safely.** It downloads text metadata from pinned OpenNeuro snapshots (refusing signal files), and allow-listed, size-capped, SHA-256-checked files from OSF projects.

**Limitations:**
- EEG/fMRI signal processing, neural decoding and deep learning are not implemented yet.
- Dynamic (RT-based) and hierarchical confidence models are not implemented yet.
- Reinforcement-learning models for the two-step task are not implemented yet.
- No neural findings are claimed.

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
| `io.osf` | OSF listing and allow-listed, capped, SHA-256-verified downloads |
| `io.openneuro` | Metadata-only access to pinned OpenNeuro snapshots. Refuses signal files and caps file size. |
| `io.brainvision` | Dependency-free `.vhdr`/`.vmrk` marker reader (no signal IO) |
| `confidence_baseline` | Nested confidence models with subject-disjoint CV, within-subject run-held-out CV, calibration, participant bootstrap and type-2 AUROC |
| `confidence_model` | Second-order signal-detection confidence model for committed vs observed decisions. Includes simulation, censored-rating likelihood, parameter and model recovery (per subject and group level), held-out-run comparison and posterior predictive checks |
| `metacognition` | Group metacognitive sensitivity: GEE logistic of accuracy on confidence × condition |
| `benchmark`, `synthetic` | Synthetic pipeline smoke test |
| `provenance` | SHA-256 and git-blob checksums |

## Datasets (not bundled)

- **Primary: [OpenNeuro ds002158](https://openneuro.org/datasets/ds002158) v1.0.2** (CC0): simultaneous EEG/fMRI with confidence after committed vs observed decisions, 20 subjects. Pereira et al. (2020) *PNAS* 117:8382, [doi:10.1073/pnas.1918335117](https://doi.org/10.1073/pnas.1918335117). *This data was obtained from the OpenNeuro database.*
- **Secondary: [OpenNeuro ds007474](https://openneuro.org/datasets/ds007474) v1.0.5** (CC0): two-step task fMRI, 189 subjects. Ding, Cockburn et al. (2026) *Cell Reports* 45(6), [doi:10.1016/j.celrep.2026.117454](https://doi.org/10.1016/j.celrep.2026.117454).
  - Behavior and timing live on the linked [OSF project](https://osf.io/ctfzd), which **declares no license**. This repo therefore stores none of its files, and the tests are synthetic.

No participant data are redistributed apart from a minimal CC0 test fixture (one 48-trial events file and marker text; see [`tests/fixtures/ds002158/PROVENANCE.md`](tests/fixtures/ds002158/PROVENANCE.md)). `data/` is git-ignored.

## Findings on ds002158

Details and evidence: [audit](docs/ds002158-audit.md) · [behavioral baselines](docs/ds002158-behavioral-baseline.md) · [computational model](docs/ds002158-confidence-model.md).

- **Behavior → fMRI timing:** events `onset` = stimulus marker − first scanner-volume trigger, with 0.0 ms residual. Trigger count equals the BOLD volume count.
- **Behavior → EEG:** 22/25 sampled runs are fully verified.
  - **sub-12's EEG run filenames are shifted**, so filename pairing would be wrong.
  - **sub-20's events and EEG markers disagree on stimulus and response side.** This is unresolved, so sub-20 is excluded from EEG work.
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

Details: [audit](docs/ds007474-audit.md) · [JSON](reports/ds007474-audit.json).

- **Missing BIDS events:** all 378 BIDS `events.tsv` files are header-only. Behavior and timing come from OSF.
  - The OSF files map one-to-one onto the BIDS subjects (189/189).
  - Their event timing agrees exactly with the behavior files.
- **Timing reference:** event onsets are relative to each run's first scanner pulse (TR 1.12 s, 650 volumes).
  - One run (sub-MBMF025002 run-1) stops after 381 volumes, so 30 of its trials have no BOLD.
- **No usable derivatives:** the published derivatives are group-level T maps only. Trial-level fMRI work would need preprocessing of 259 GB of raw BOLD.
- **Sanity check:** stay probabilities show the standard reward × transition pattern. This checks the coding; it is not a finding.

## Principles

- Subject-disjoint evaluation by default. Preprocessing is fit only on training folds.
- No dataset column mapping without verification against the actual files. Facts in docs are labelled as verified or hypothesis.
- Pre-decision, response-locked and post-decision neural windows are kept separate. No prospective claims are made from post-response features.
- Signal data is never downloaded implicitly: the fetcher only accepts text metadata. Neuroimaging dependencies are optional extras.

Contributor rules are in [`AGENTS.md`](AGENTS.md); the research plan is in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Changelog

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
