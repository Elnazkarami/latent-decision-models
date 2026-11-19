# latent-decision-models — Research & Engineering Project Plan
Version: 1.0 | 2026-11-18 | Status: proposed, dataset metadata audit pending

## Executive decision
Build an open-source, pip-installable Python framework that connects task behavior, computational decision-model latent variables, and trial-aligned EEG/fMRI features. Two reference applications: (A) perceptual confidence/metacognition using OpenNeuro ds002158, (B) two-step reinforcement learning and belief updating using OpenNeuro ds007474. Do **not** present as a novel neuroscience finding before testing; prioritize reproducibility, validated methodology, and modular software useful beyond these datasets.

## Goals and success criteria
- Another researcher can use a documented task adapter to load a different BIDS-compatible decision task.
- All analyses run with a configuration file and generate provenance, metrics, and figures.
- Baselines precede neural networks and are compared in honest, group-aware evaluation.
- A small synthetic dataset and complete CPU smoke test work without downloading large neuroimaging data.
- First real-data example reliably loads and validates behavioral trials from ds002158; later paired EEG/fMRI experiments operate on verified alignments.
- v0.1 is an installable, typed, tested library, not a notebook-only collection.

## Scientific questions
A. Does neural information improve trial-level confidence prediction beyond evidence strength, accuracy, condition, and RT? Does joint EEG/fMRI add out-of-sample value over unimodal models, with uncertainty and an explicit null outcome?
B. Can trial-sequence computational RL models recover interpretable learning signals, predict held-out choice, and relate those signals to BOLD responses? Can the framework support different tasks without a rewrite?
Exploratory: compare patterns across tasks only when the target construct and analyses are defined; never equate embeddings from different participants/tasks to a shared neural representation by default.

## Dataset decision
### Primary: OpenNeuro ds002158 (confidence)
- DOI: https://doi.org/10.18112/openneuro.ds002158.v1.0.2
- Browser: https://openneuro.org/datasets/ds002158
- Dataset index: https://github.com/OpenNeuroDatasets/ds002158
- Original study: Pereira et al. (2020), PNAS, https://doi.org/10.1073/pnas.1918335117
- 20 subjects, simultaneous EEG/fMRI, confidence after committed versus observed decisions, BrainVision EEG, T1, fieldmaps, BIDS event files, eegprep and fmriprep derivatives; NEMAR mirror reports ~429 GB complete size. License reported CC0 on NEMAR (verify OpenNeuro snapshot-specific license before use).
- v0.1: **metadata and behavioral events first**; then one small, verified subset of EEG and derivatives. Avoid whole-dataset download.
- **Unverified until audit:** actual event column names/coding, participant/event quality, synchronization markers, whether published derivatives permit required trial-level single-trial analyses, run associations, and usable trial counts.
- Critical experimental factor: own/committed versus observed decisions. Do not automatically treat observed decisions as having an ordinary participant choice RT.

- **Audit corrections (2026-11-18, see `docs/ds002158-audit.md`):**
  - The full snapshot is 460 GB; 374 GB of that is the uncompressed fMRI derivative. Raw EEG + BOLD is about 82 GB.
  - Subject IDs are non-contiguous.
  - The `fmriprep` derivative is not fMRIPrep-style output.
  - `eegprep` is one response-locked file per subject, with no trial mapping.
  - Observed trials are shuffled replays of the participant's own active block.
  - sub-12 EEG run labels are shifted, and sub-20 events and EEG markers disagree on stimulus and response side.

### Secondary: OpenNeuro ds007474 (two-step RL)
- https://openneuro.org/datasets/ds007474
- Dataset repository: https://github.com/OpenNeuroDatasets/ds007474
- Behavioral timing and task data are linked separately from MRI dataset README: https://osf.io/ctfzd/overview?view_only=022584fa16944755982f0d06a713835c
- Author analysis code: https://github.com/Gentu-Ding/2-step-task-MBMF-fMRI-code
- Reported 189 healthy adult participants. Dataset notes identify special reward conditions for five IDs and missing run-1 fieldmaps for one ID; document exclusions/stratification.
- **Unverified until audit:** behavioral file completeness, exact subject ID mapping, response/reward and transition encoding, preprocessing status and permission/terms of external material.
- **Audit corrections (2026-11-19, see `docs/ds007474-audit.md`):**
  - The BIDS events files are empty. Behavior and timing come from OSF.
  - Subject IDs map one-to-one (189/189).
  - Each subject has 259 trials, 105 of them from a structural-scanner session without BOLD.
  - Derivatives are group T maps only.
  - The OSF project and the authors' code have no license.
  - Five subjects had state-contingent reward only; one run (MBMF025002 run-1) is truncated.

## Product boundary
IN v0.1: typed trial schema, BIDS discovery and provenance, dataset-specific behavior adapters, synthetic generator, behavioral confidence baseline, group-aware evaluation, report and tests.
IN v0.2: MNE EEG import/epoching and conventional decoder with artifact-aware tests.
IN v0.3: nilearn fMRI GLM/ROI extraction, careful single-trial availability check, confounds and acquisition timing.
IN v0.4: small regularized PyTorch model and fusion/ablation comparisons, only after sample and synchronization validation.
IN v0.5: ds007474 adapter and model-based/model-free RL reference analysis.
OUT of initial scope: raw MRI reprocessing, universal BIDS conversion, diagnostic claims, population genetics, gigantic transformers, unlicensed data redistribution, causal conclusions from decoding.

## Architecture
Layers:
1. IO: PyBIDS / MNE-BIDS adapters; immutable dataset/version manifest.
2. Common typed trial table and modality provenance; explicit missing-data indicators.
3. Task adapter interface: `parse_events`, `validate_schema`, `derive_trials`, `target_definition`.
4. Behavior: confidence baselines; sequential choice and RL models separately, plus parameter-recovery simulations for fitted models.
5. Imaging: EEG MNE epochs and fMRI Nilearn GLM/ROI or beta maps; no unverified trial correspondence.
6. ML: sklearn baselines and optional PyTorch encoders; fusion with missing modality mask.
7. Evaluation: nested subject-disjoint CV where practical, permutation controls, bootstrap CIs resampled by participant, calibration, ablations.
8. CLI/reports: configurations; file checksums/versions; summary JSON + tables and figures.

## Canonical schema (prototype, finalized after data audit)
TrialID = (dataset_id, subject, session?, task, run, trial_index). Required: unique TrialID, source_file, condition, trial_onset_seconds, valid_trial. Optional with explicit missingness: stimulus/evidence, choice, correct_choice, accuracy, response_time_seconds, confidence_raw, confidence_normalized, feedback, reward, stage, eeg_event_marker, fmri_event_onset_seconds. Task-specific columns go in extension namespace; do not fabricate unavailable fields. Store timing reference frames and coding dictionaries in metadata.
NeuralFeatureSet: TrialID + modality + feature_definition + feature_array / pointer + preprocessing provenance and time window; subject/run match must be verified, never inferred only by row order.

## Evaluation protocol
- Primary generalization question: held-out-subject performance. All transformations/feature selection/hyperparameter selection fit only in training fold.
- Secondary within-person question: temporal/run hold-out, clearly separate from new-subject generalization; avoid adjacent-window leakage.
- Define confidence target based on actual ratings (ordinal/continuous, range verified); choose MAE and rank association for ordinal/continuous, or log loss/Brier for suitably defined classification, plus calibration and per-subject distributions.
- Always compare null (training-mean/majority), task-only, behavior-only, neural-only, and multimodal where available.
- No random trial split across the same person for claims of participant-generalizing performance.
- Reporting units, missing trial exclusions, chance baselines, confidence intervals, and small-n limitations are mandatory.
- Decision to pursue deep fusion is contingent on enough cross-modality matched trials and stable baseline results.

## Milestones / gates
M0 Data feasibility (week 1): inspect BIDS metadata and event JSON/TSVs, count usable runs/trials, cross-subject matches and derivative availability. Deliver DATA_AUDIT.md + machine-readable manifest. GO if a reliable confidence target, timing and trial identity can be established. Otherwise narrow scope or switch dataset with documented rationale.
M1 Foundation (week 2): pyproject, src layout, tests, lint/type-check, CI, CLI skeleton, synthetic dataset. GO if clean install and all CPU tests pass.
M2 Behavior (weeks 3–4): schema adapter, descriptives, one well-specified baseline, group CV, report. GO if reproducible real-data behavior run and synthetic known-answer tests pass.
M3 EEG (weeks 5–6): minimal verified EEG subset; preprocessing and decoding, independent baseline. GO if event alignment/QC clear and no leakage.
M4 MRI (weeks 7–9): GLM/ROI features with confounds, provenance. GO if matched trials and valid designs. **Status 2026-11-19:** metadata audit done (`docs/ds002158-fmri-audit.md`); image download awaiting approval.
M5 Fusion (weeks 10–12): simple feature concat then gated/late fusion, ablations and uncertainty; never privilege deep model if no observed benefit.
M6 Paper-ready (weeks 13–14): reproducible end-to-end documentation, benchmarks, project website if justified, release tag, CITATION.cff; scientific report includes negative findings.
M7 Belief updating follow-up: ds007474 alignment and distinct adapter; implement RL baseline after task contingencies verified. **Status 2026-11-19:** behavior-only RL benchmark done (`docs/ds007474-two-step-model.md`).
Durations are estimates for part-time implementation and not deadlines.

## Deliverable checklist
- PUBLIC: README with features/limitations, quick-start; `src/latent_decision_models`; `tests`; small synthetic `examples`; docs; sample config; CI; CITATION.cff; license; contribution guide.
- PRIVATE/LOCAL: `data/raw`, `data/derivatives`, participant-level intermediate features, large neuroimaging files. Git-ignored.
- REPORTS: audit log, summary metrics with folds and uncertainties, condition-wise analyses, reproducible figures, model card, source citations, data availability instructions.
- Definition of Done for any milestone: code + automated tests + example + documentation + reproducibility command + reviewer-readable evidence; no fabricated results.

## Risks and mitigations
1. N=20 and correlated trials: compact models, hierarchical statistics, participant-level uncertainty; no large deep net claims.
2. BIDS events != paired neural trial features by default: audit clock reference and event markers, show matching QA report.
3. fMRI temporal resolution and BOLD lag: specify HRF/GLM design, no spurious instantaneous EEG-to-fMRI synchronization claims.
4. Observed vs committed condition: separate model definitions and missing RT rules.
5. Data size: subset-first and version pin; no raw data in Git.
6. Recreating existing tools: use MNE, Nilearn, PyBIDS rather than reinvent preprocessing.
7. Dataset-specific task semantics: use adapters, not forced universal column mapping.
8. License/privacy: verify provenance, follow licenses and participant restrictions; do not expose individual raw data.

## Decisions required after first audit
Exact confidence target and scale; sufficient valid trials by condition; choice/RT availability by condition; EEG vs MRI timing mapping; derivative selection; selected starter participant subset; feasible held-out-unit strategy.
