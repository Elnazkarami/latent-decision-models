# Coding Assistant Implementation Brief — latent-decision-models
**Read PROJECT_PLAN.md and DATASET_SELECTION_AND_AUDIT.md first.**

## Role and goal
Act as a research software engineer collaborating with a computational neuroscientist. Build **latent-decision-models**, a modular, pip-installable Python package that links decision-task behavioral trials and computational-model latent variables to EEG/fMRI features. Scientific correctness, transparent uncertainty, and reuse outrank flashy model complexity.

## Task scope for your FIRST implementation iteration
Do not train on real neuroscience data yet. First create a clean repository with:
1. Python `src/` package, `pyproject.toml`, dependencies separated into core/eeg/fmri/deep/dev extras, and a CLI entrypoint `ldm`.
2. Typed `TrialTable` and `TrialID` schema (Pydantic or dataclasses + Pandera; justify choice) allowing nullable optional fields, explicit units and trial-level provenance.
3. `TaskAdapter` protocol: `discover`, `parse_events`, `validate`, `to_trials` with a generic BIDS-events adapter and placeholder ds002158-specific adapter that throws a descriptive error until field mappings are verified.
4. Deterministic synthetic task generator: participant, run, condition, evidence, accuracy, RT, confidence; include condition-specific missingness and edge cases.
5. Baselines using scikit-learn: null predictor and regularized confidence regression/ordinal model as appropriate to *synthetic* target; subject-disjoint split, train-only preprocessing, fold-wise metrics.
6. Config-driven CLI: `ldm simulate --out ...`, `ldm validate --input ...`, `ldm benchmark --config ...`. Prefer simple YAML configs; validate schemas.
7. Unit tests and integration smoke tests; GitHub Actions running install, Ruff, tests; static typing if practical. Provide readable error messages and docs.
8. Dataset audit utility that inspects a local BIDS tree or small metadata files and creates `reports/dataset_audit.json` **without downloading raw EEG/MRI**.
9. README with quick start, example output labeled "synthetic", licenses and dataset citations. `.gitignore` for data, models, raw scans, outputs and secrets. Example configs.

## Proposed repo tree
```
latent-decision-models/
  pyproject.toml
  README.md
  LICENSE
  CITATION.cff
  .github/workflows/ci.yml
  configs/synthetic_confidence.yaml
  src/latent_decision_models/__init__.py
  src/latent_decision_models/cli.py
  src/latent_decision_models/schema.py
  src/latent_decision_models/io/{__init__.py,bids.py,adapters.py}
  src/latent_decision_models/behavior/{__init__.py,baselines.py,synthetic.py}
  src/latent_decision_models/evaluation/{__init__.py,splits.py,metrics.py}
  src/latent_decision_models/reports/{__init__.py,audit.py}
  src/latent_decision_models/eeg/__init__.py
  src/latent_decision_models/fmri/__init__.py
  src/latent_decision_models/models/__init__.py
  tests/
  docs/data_audit.md
  examples/
```
Keep placeholders minimal; do not create sophisticated unused abstractions.

## Design contracts
- `TrialID` uniqueness across `(dataset_id,subject,session,task,run,trial_index)`; session optional but represented consistently. Assert uniqueness, sortedness only where necessary; never assume file row index aligns across modalities.
- `TrialTable` should have explicit confidence scaling metadata and separate `confidence_raw` and modeled label; never infer confidence scale without data documentation.
- All features define their time reference (stimulus/response/feedback), clock/time axis and acquisition source.
- `TaskAdapter` does not silently drop malformed trials. It returns validation records and exclusion reasons.
- Behavior model API should be task-specific. Avoid promising a general DDM/RL implementation until the task design is checked.
- Evaluation API accepts `groups=subject` by default for cross-participant performance and reports per-fold counts; no random-trial shortcut.
- Separate train and test transforms, no global imputation/normalization/feature selection before splitting.
- Use deterministic seeds. Record code version, environment, config, data snapshot, fold assignments, and relevant preprocessing settings.

## Acceptance criteria for iteration 1
A fresh user can run:
```
python -m pip install -e '.[dev]'
latent_decision_models simulate --out examples/synthetic_trials.csv
latent_decision_models validate --input examples/synthetic_trials.csv
latent_decision_models benchmark --config configs/synthetic_confidence.yaml
python -m pytest -q
```
Expected: commands terminate successfully, emit machine-readable validation + fold metrics, tests pass, and no network access or large data download is required. No fabricated neural predictions. Include tests for duplicate TrialID, invalid RT, out-of-range confidence when a scale is explicitly set, missing observations, train/test group overlap, reproducible synthetic output, and nonleaky transforms.

## Audit in parallel (metadata only)
Inspect ds002158 metadata, README, task-main_events.json, sample actual events.tsv and available derivative folder names from canonical OpenNeuro snapshot. Record actual keys, condition coding, presence of choice/RT/confidence variables, EEG and fMRI run matching evidence, and any uncertainties. Do not hardcode hypothetical event names in the adapter. Mark ds002158 adapter `experimental` until mapping tests are grounded in actual files. Do not download the 429GB dataset.

## Scientific/engineering guardrails
- Never claim published performance without actual run logs, split records and metrics.
- Do not claim an EEG or fMRI neural result from synthetic data.
- Avoid leakage through trial order, group effects, postdecisional EEG, nuisance regressors, or downstream confidence-derived features.
- No deep learning until classical baselines work and paired real-data trials are audited.
- Do not confuse "prediction of confidence from after-response signals" with prospective prediction from before-response signals. Name estimands explicitly.
- Never mix datasets participant-wise or infer EEG-fMRI matching from row number.
- Avoid huge dependencies by installing MNE/Nilearn/PyTorch only as optional extras.
- Follow licenses and cite OpenNeuro datasets and original studies.

## Response format from coding assistant
1. Explain repo tree and important design decisions briefly.
2. Implement the files rather than returning pseudocode.
3. Run actual tests and report exactly which passed/failed.
4. Show example synthetic CLI outputs, explicitly labeled synthetic.
5. List factual data-audit findings with source links and remaining uncertainties.
6. Summarize changed files, exact commands and next actionable milestone.

## Future increments (NOT first iteration)
- MNE EEG import, event mapping and artifact-aware preprocessing; EEG baseline.
- Nilearn first-level GLM/ROI extraction and confounds, cautious HRF modeling.
- Compact multimodal network with late/early fusion and missing-modality mask; subject-level ablation metrics.
- ds007474 adapter with verified OSF events, transition/reward semantics; parameter recovery and RL model comparison.
