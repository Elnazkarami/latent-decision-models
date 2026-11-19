# latent-decision-models

[![CI](https://github.com/Elnazkarami/latent-decision-models/actions/workflows/ci.yml/badge.svg)](https://github.com/Elnazkarami/latent-decision-models/actions/workflows/ci.yml)

**Research software in active development: not a validated neuroscience analysis.**

A reproducible Python library for trial-aligned modeling of decision confidence, belief updating and latent cognitive variables from behavior, EEG and fMRI. It provides task adapters with strict validation, subject-disjoint evaluation, and audits of public datasets with full provenance.

Python package `latent_decision_models` · CLI `ldm` · repository `latent-decision-models`.

## Status

| Ticket | Scope | Status |
| --- | --- | --- |
| LDM-000 | Package scaffold, synthetic data, CPU smoke tests, CI | ✅ done |
| LDM-001 | Audit of OpenNeuro ds002158 (confidence; EEG + fMRI) | ✅ done: [audit](docs/ds002158-audit.md) · [JSON](reports/ds002158-audit.json) |
| LDM-002 | Behavioral confidence baselines and computational model on ds002158 | ✅ done: [baselines](docs/ds002158-behavioral-baseline.md) · [computational model](docs/ds002158-confidence-model.md) (recovery-checked; model comparison inconclusive) |
| LDM-003 | EEG alignment, QC, classical decoding | ⏸ waiting for approval to download raw EEG (~4.2 GB) |
| LDM-004 | fMRI derivative audit, trial-level GLM/ROI | ready (derivative audit is metadata-only; volume download needs approval) |
| LDM-005 | Multimodal baseline, small fusion model | ⏳ blocked by LDM-003/004 |
| LDM-006 | Audit of ds007474 (two-step RL) | ready |
| LDM-007 | Belief-updating RL benchmark | ⏳ blocked by LDM-006 |
| LDM-008 | Report and tagged release | ⏳ |

The machine-readable backlog is [`tasks/backlog.json`](tasks/backlog.json), and the current work order is [`tasks/next-task.md`](tasks/next-task.md).

**Not implemented yet:** EEG/fMRI signal processing or decoding, deep learning, dynamic (RT-based) or hierarchical confidence models. No neural findings are claimed.

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

**Check one run's EEG/behavior alignment.** It needs only that run's `.vhdr` and `.vmrk` text files:

```bash
ldm ds002158 align-eeg --events <..._events.tsv> --vmrk <..._eeg.vmrk>
```

## What's in the package

| Module | Purpose |
| --- | --- |
| `schema` | Generic trial-table validation (unique trial identity, onsets, RT) |
| `datasets.ds002158` | Verified codebook and strict events parser for ds002158 (experimental). Aligns EEG markers by **timing**, never by row order or filename. |
| `io.openneuro` | Metadata-only access to pinned OpenNeuro snapshots. Refuses signal files and caps file size. |
| `io.brainvision` | Dependency-free `.vhdr`/`.vmrk` marker reader (no signal IO) |
| `confidence_baseline` | Nested confidence models with subject-disjoint CV, within-subject run-held-out CV, calibration, participant bootstrap and type-2 AUROC |
| `confidence_model` | Second-order signal-detection confidence model for committed vs observed decisions. Includes simulation, censored-rating likelihood, parameter and model recovery (per subject and group level), held-out-run comparison and posterior predictive checks |
| `metacognition` | Group metacognitive sensitivity: GEE logistic of accuracy on confidence × condition |
| `benchmark`, `synthetic` | Synthetic pipeline smoke test |
| `provenance` | SHA-256 and git-blob checksums |

## Datasets (not bundled)

- **Primary: [OpenNeuro ds002158](https://openneuro.org/datasets/ds002158) v1.0.2** (CC0): simultaneous EEG/fMRI with confidence after committed vs observed decisions, 20 subjects. Pereira et al. (2020) *PNAS* 117:8382, [doi:10.1073/pnas.1918335117](https://doi.org/10.1073/pnas.1918335117). *This data was obtained from the OpenNeuro database.*
- **Secondary: [OpenNeuro ds007474](https://openneuro.org/datasets/ds007474)**: two-step RL task fMRI. Not yet audited.

No participant data are redistributed apart from a minimal CC0 test fixture (one 48-trial events file and marker text; see [`tests/fixtures/ds002158/PROVENANCE.md`](tests/fixtures/ds002158/PROVENANCE.md)). `data/` is git-ignored.

## Key findings so far (ds002158)

Details and evidence are in [`docs/ds002158-audit.md`](docs/ds002158-audit.md) and [`docs/ds002158-behavioral-baseline.md`](docs/ds002158-behavioral-baseline.md).

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

## Principles

- Subject-disjoint evaluation by default. Preprocessing is fit only on training folds.
- No dataset column mapping without verification against the actual files. Facts in docs are labelled as verified or hypothesis.
- Pre-decision, response-locked and post-decision neural windows are kept separate. No prospective claims are made from post-response features.
- Large or raw neuroimaging downloads need explicit approval. Optional dependencies stay optional.

See [`AGENTS.md`](AGENTS.md) for contributor and agent rules, and [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan.

## Changelog

- **2026-11-19:** LDM-002 computational confidence model, which closes LDM-002:
  - Second-order SDT model with four observed-trial readout variants.
  - Tests that the likelihood normalizes and that known parameters are recovered.
  - Parameter and model recovery at a priori and fitted-region ranges, plus group-level model recovery.
  - Real-data held-out-run comparison and posterior predictive checks, with a sub-20 sensitivity run.
  - New `ldm confidence-model recovery|compare` commands.

- **2026-11-18:** LDM-002 second pass: calibration metrics, within-subject run-held-out evaluation, GEE metacognition model (`ldm confidence-baseline` now reports all three); statsmodels added as a core dependency.
- **2026-11-18:** LDM-001 audit of ds002158; ds002158 adapter, EEG marker alignment, metadata-only OpenNeuro fetcher; first-pass behavioral confidence baselines (LDM-002).
- **2026-11-18:** LDM-000 scaffold: package, synthetic generator, validator, subject-disjoint benchmark, CI.

## License

MIT (see [`LICENSE`](LICENSE)). Dataset licenses apply separately.
