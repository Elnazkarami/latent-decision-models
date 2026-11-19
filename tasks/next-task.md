# Next task

**Done:**
- LDM-001 / LDM-002: ds002158 audit, behavioral baselines, computational confidence model.
- LDM-006: ds007474 audit ([`docs/ds007474-audit.md`](../docs/ds007474-audit.md)).
- LDM-007: ds007474 two-step RL benchmark, behavior only ([`docs/ds007474-two-step-model.md`](../docs/ds007474-two-step-model.md)).

## Option A: LDM-007 follow-ups (no approval needed)
1. **Fix the predictive misfit.** All models underpredict the reward × transition stay interaction. Add separate MF/MB learning rates and MB win-stay/lose-shift terms. Run recovery first, then the held-out comparison.
2. **Out-of-session replication.** Use the 105 structural-session trials per subject, which also have one contingency switch. Fit on the functional session and predict the structural session, and repeat the contingency contrast.
3. **Trial-wise arbitration** (the authors' `isPost_*` terms), only if recovery supports it.

## Option B: LDM-003, ds002158 EEG (needs explicit approval)
Raw EEG for sub-02 and sub-04, about 4.2 GB.

## Option C: LDM-004, ds002158 fMRI derivative audit (metadata first)

## Open questions for the PI
- **ds007474 licensing:** the OSF project and the authors' code declare no license. Ask the authors to add one before any public derived release?
- **ds007474 fMRI:** there are no subject-level derivatives. Is preprocessing 259 GB of raw BOLD in scope, or does the work stay behavior-only?
- **ds007474 exclusions:** the primary analysis excludes 8 subjects with > 10% missed trials and does not score RTs < 150 ms. Results are unchanged without exclusions. Keep this as the default?
- **ds002158:** exclude sub-20 from EEG work (recommended: yes); is the continuous [0, 1] confidence target acceptable; approval for the LDM-003 download.
