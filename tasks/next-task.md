# Next task

**Done:**
- LDM-001: the audit ([`docs/ds002158-audit.md`](../docs/ds002158-audit.md)).
- LDM-002: [`docs/ds002158-behavioral-baseline.md`](../docs/ds002158-behavioral-baseline.md) and [`docs/ds002158-confidence-model.md`](../docs/ds002158-confidence-model.md).

The computational model's comparison of committed vs observed readouts is **inconclusive**, and its key parameter ρ is not identifiable per subject. Do not build on its per-subject parameters.

## Option A: LDM-006, audit ds007474 (no approval needed; metadata only)
Do the same kind of audit as LDM-001: events columns, behavior on OSF, subject mapping, permissions. This unblocks the belief-updating RL benchmark (LDM-007).

## Option B: LDM-003, EEG (needs explicit approval)
- Download the raw EEG `.eeg/.vhdr/.vmrk` for **sub-02 and sub-04** (about 4.2 GB) to `data/ds002158/`.
- Read it with MNE (`[eeg]` extra).
- Before epoching, run `align_eeg_markers` and refuse to continue unless the status is `verified`.
- Keep pre-response and post-response windows separate.

## Option C: LDM-004, fMRI derivative audit (metadata first)
Settle the derivative's space and smoothing from headers and sidecars before any volume download. Volume downloads (about 3.2 GB per run) need approval.

## Optional LDM-002 follow-ups
- A hierarchical version of the confidence model. Partial pooling may make ρ, or the observed-readout comparison, identifiable at the group level; it must pass group-level recovery first.
- A dynamic (accumulation) model that uses active RTs.
- Mixed-effects or ordinal metacognition variants.

## Open questions for the PI
- sub-20: exclude it from EEG work? Recommended: yes.
- Is the continuous [0, 1] confidence target acceptable as the primary endpoint?
- Approval for the LDM-003 EEG download (about 4.2 GB).
