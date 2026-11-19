# Next task

**Done:**
- LDM-001 / LDM-002: ds002158 audit, behavioral baselines, computational confidence model.
- LDM-006: ds007474 audit ([`docs/ds007474-audit.md`](../docs/ds007474-audit.md)).
- LDM-007: ds007474 two-step RL benchmark, behavior only ([`docs/ds007474-two-step-model.md`](../docs/ds007474-two-step-model.md)).

## Option A: further LDM-007 extensions (no approval needed, lower priority)
The misfit is addressed: hybrid + win-stay/lose-shift is best, including on a separate session.
1. Learned pad transitions and trial-wise arbitration (the authors' `isPost_*` terms), each with recovery first.
2. Hierarchical (empirical-Bayes) fitting, to tighten per-subject estimates at 154 trials.

## Option B: LDM-003, ds002158 EEG (needs explicit approval)
Raw EEG for sub-02 and sub-04, about 4.2 GB.

## Option C: LDM-004, ds002158 fMRI derivative audit (metadata first)

## Open questions for the PI
- **ds007474 licensing:** the OSF project and the authors' code declare no license. Ask the authors to add one before any public derived release?
- **ds007474 fMRI:** there are no subject-level derivatives. Is preprocessing 259 GB of raw BOLD in scope, or does the work stay behavior-only?
- **ds007474 exclusions:** the primary analysis excludes 8 subjects with > 10% missed trials and does not score RTs < 150 ms. Results are unchanged without exclusions. Keep this as the default?
- **ds002158:** exclude sub-20 from EEG work (recommended: yes); is the continuous [0, 1] confidence target acceptable; approval for the LDM-003 download.
