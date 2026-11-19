# Next task

**Done:**
- LDM-001 / LDM-002: ds002158 audit, behavioral baselines, computational confidence model.
- LDM-006: ds007474 audit ([`docs/ds007474-audit.md`](../docs/ds007474-audit.md)).

## Option A: LDM-007, two-step RL benchmark, behavior only (no approval needed)
Use `ldm ds007474 trials` on the OSF behavior already fetched.
1. **Settle the design semantics first:** what `condState`, `condReward` and `isPost_*` mean, from Cockburn et al. (doi:10.31234/osf.io/egq3x).
2. **Fix exclusions before fitting:** decide on RT < 150 ms trials (658 functional) and high-miss subjects.
3. **Models:** model-free / model-based / hybrid RL with the reward-magnitude and contingency structure. Simulate and check parameter recovery before any real-data fit.
4. **Evaluation:** predictive comparison on held-out runs and held-out subjects.
5. **Structural-session trials:** decide whether they enter the fits; this is a separate question.

## Option B: LDM-003, ds002158 EEG (needs explicit approval)
Raw EEG for sub-02 and sub-04, about 4.2 GB.

## Option C: LDM-004, ds002158 fMRI derivative audit (metadata first)

## Open questions for the PI
- **ds007474 licensing:** the OSF project and the authors' code declare no license. Ask the authors to add one before any public derived release?
- **ds007474 fMRI:** there are no subject-level derivatives. Is preprocessing 259 GB of raw BOLD in scope, or does LDM-007 stay behavior-only?
- **ds002158:** exclude sub-20 from EEG work (recommended: yes); is the continuous [0, 1] confidence target acceptable; approval for the LDM-003 download.
