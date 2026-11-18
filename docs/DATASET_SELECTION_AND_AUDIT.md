# latent-decision-models — Dataset Decision & Audit Protocol
Version 1.0, 2026-11-18

## Decision
Use ds002158 as the first end-to-end reference dataset for **confidence**. Keep ds007474 reserved for the belief-updating / RL reference application. Do not mix unmatched subjects or treat different tasks as interchangeable.

## Verified from public descriptions
| Criterion | ds002158 | ds007474 |
| --- | --- | --- |
| Construct | Perceptual confidence, own vs observed decisions | Two-step sequential choice / model-based vs model-free RL |
| Modalities | Simultaneous EEG + fMRI, structural MRI | Task fMRI + structural MRI; behavioral task linked separately |
| Subjects | 20 | 189 |
| Preprocessing | eegprep and fmriprep derivatives reported | Some published second-level T-map derivatives; audit actual individual files |
| Special risks | Small N, large total download, subtle active/observation trial design | External OSF behavioral files, separate participant ID map, reward condition variations |
| Priority | Primary | Secondary |

## Primary sources
- https://openneuro.org/datasets/ds002158
- https://github.com/OpenNeuroDatasets/ds002158
- https://nemar.org/dataset/on002158
- https://doi.org/10.1073/pnas.1918335117
- https://openneuro.org/datasets/ds007474
- https://github.com/OpenNeuroDatasets/ds007474
- https://osf.io/ctfzd/overview?view_only=022584fa16944755982f0d06a713835c
- https://github.com/Gentu-Ding/2-step-task-MBMF-fMRI-code

## Agent task: audit before download
1. Record source URL, dataset snapshot identifier, license and requested data-use restrictions; distinguish the OpenNeuro ds002158 v1.0.2 snapshot from NEMAR mirror metadata.
2. Retrieve or inspect **metadata files only**: README, dataset_description.json, participants.tsv, task-main_events.json, all small event TSV/JSON files, one file tree from each subject, and derivative directory structure. Record source URLs and checksums when available.
3. Inventory participant/condition/run counts and missingness. Extract `actual_column_names`, dtype examples and codebook; do not guess column names.
4. Inspect whether active/observation conditions have confidence, response time, stimulus difference, stimulus onset and separate EEG response markers; document differences.
5. Create a per-trial key audit. Confirm whether it is possible to pair EEG and fMRI trials by run, temporal markers, and protocol. **Never join by row order only.**
6. Check preprocessing derivative naming, what each derivative represents, spatial reference frame, frequency filter, event epoch timing, and whether access is practical.
7. Estimate starter subset size and storage (prefer metadata and 1–2 participants/runs after initial audit). Do not fetch full ~429GB without approval.
8. For ds007474, verify OSF behavioral files are obtainable; confirm participant ID mapping, choice sequence and transition/reward variables, unusual reward-design subjects and fieldmap exception.
9. Write `docs/data_audit.md` with factual findings, unknowns, blockers and a GO/NO-GO recommendation.

## Audit output schema
A JSON file `reports/dataset_audit.json` should contain: dataset_id, source_uri, snapshot_version, license_verified, citation, subjects_total, subjects_with_events, tasks, conditions, per_condition_trial_counts, available_columns, target_candidate, known_missingness, EEG_files, fMRI_files, derivatives, trial_alignment_status (unverified/partial/verified), storage_estimate, audit_date, blockers. Unknown fields should be null, not made up.

## Pre-registered-style analysis decision notes
- Define primary endpoint and contrast **before** seeing performance; log any changes.
- Subject-level cross-validation for generalization; report participant-level uncertainty.
- Condition-stratified evaluation and/or an interaction test; avoid pooling active and observed without a model term.
- Compare behavior-only, neural-only and combined models; simple baseline first.
- Test whether EEG time windows are pre- versus post-response and avoid introducing outcome information into a model purporting to make an earlier prediction.

## Stop conditions
STOP and request scientific review if no reliable trial-level confidence labels exist, modality alignment cannot be established, use terms are unclear, or the proposed neural model would necessarily leak the target from post-outcome features while claiming prospective prediction. Narrow the claim or experiment rather than inventing data.
