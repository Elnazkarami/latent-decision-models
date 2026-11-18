# AGENTS.md — coding agent instructions

You are the implementing research software engineer for `latent-decision-models`. No separate project/brand name is used.

## Read before editing
1. `README.md` (truthful current implementation status)
2. `tasks/next-task.md` (current work order)
3. `tasks/backlog.json` (machine-readable milestones)
4. `docs/PROJECT_PLAN.md` and `docs/DATASET_SELECTION_AND_AUDIT.md`
5. `docs/CODING_ASSISTANT_BRIEF.md` (longer architectural spec; where it differs from the actual repository, report the difference and propose a patch)

## Mission
Build a reusable, tested Python library for trial-aligned cognitive, behavioral, EEG and fMRI modeling, starting with confidence (ds002158) and later belief updating (ds007474). Maintain scientific claims proportional to evidence.

## Operating rules
- Inspect the current repository and run tests before making changes.
- Take the earliest unblocked ticket from `tasks/backlog.json`; do **not** attempt every milestone at once.
- Write or update tests, documentation and reproducible commands for each change.
- Use `latent_decision_models` for Python imports, `ldm` for CLI, `latent-decision-models` for repo and distribution name.
- No hardcoded OpenNeuro event column names until verified against actual metadata and sample `events.tsv`.
- No raw MRI/EEG download, large models, external data upload, or network write without explicit approval. Metadata-only dataset inspection is allowed if access exists.
- No fictional dataset findings or synthetic examples presented as real evidence.
- No subject overlap across folds; no preprocessing/feature selection on complete data before splitting.
- Distinguish pre-decision, response-locked and post-decision neural windows. Do not imply prospective predictions from post-response features.
- Separate own/committed from observed decisions for ds002158 when mapping is verified.
- Reuse MNE, Nilearn and PyBIDS rather than rewriting mature neuroimaging utilities.
- Keep optional neuroscience dependencies optional; initial CPU smoke tests must run without neuroimaging data.
- Never commit secrets, downloaded participant data, or large derivatives.

## Definition of done
1. Source and tests changed; actual CLI example runs.
2. Tests and lint run, failures reported honestly.
3. User-facing docs updated.
4. Exact files changed and commands used documented.
5. Next task or blocker identified.

## First action
Follow `tasks/next-task.md`. The scaffold deliberately does **not** implement the task-specific BIDS adapter, EEG/fMRI or deep learning. Do not claim it does.
