# ds007474: model-free / model-based reinforcement learning (behavior only)

- **Date:** 2026-11-19
- **Data:** ds007474 v1.0.5 behavior from OSF [`ctfzd`](https://osf.io/ctfzd), functional-run trials only (2 runs × 77 per subject). See the [audit](ds007474-audit.md).
- **Code:** [`latent_decision_models.two_step_model`](../src/latent_decision_models/two_step_model.py); CLI `ldm two-step-model recovery|compare`.
- **Reports:** recovery JSONs are committed ([synthetic schedules](../reports/two_step_recovery.json), [real schedules](../reports/two_step_recovery_real_schedules.json), [fitted range](../reports/two_step_recovery_fitted_range.json)). Real-data outputs are regenerated locally (`reports/ds007474_two_step_model*.json`, git-ignored).

## Summary

- **The task design is now fully verified.** Every shown outcome is reproduced from the trial schedule stored in the CSV (0 mismatches on 28,512 responded trials). Any agent can therefore be replayed on each subject's real task sequence.
- **Hybrid beats pure model-free and pure model-based learning on held-out data.** On held-out runs, hybrid beats MF by 1,084 log-likelihood units [718, 1,451] and MB by 1,116 [680, 1,605], summed over 181 subjects. The ranking is the same on held-out subjects, with learning from points instead of wins, and with no exclusions. Recovery over the fitted parameter range shows this group-level comparison is not biased toward the hybrid.
- **No evidence that people weight model-based control more when it pays off.** The design switches between reward contingencies where model-based control is correct (state) or useless (stimulus). The order-balanced difference w_state − w_stimulus is −0.003 [−0.039, 0.031]. A planted effect of 0.15 is recovered as 0.11 [0.08, 0.14], so an effect of that size would have been seen.
- **The larger stay interaction under state contingency comes from the schedule, not from arbitration.** Observed: 0.26 vs 0.21. The hybrid with one weight reproduces the same 0.05 difference.
- **All four models miss part of the behavior:** they underpredict the reward × transition interaction (see "Posterior predictive check").

These are descriptive model comparisons on one dataset; they are not new claims about the brain.

## Task facts used by the model (verified on all 189 subjects)

Verified against the authors' task code (`task_code/` in `Gentu-Ding/2-step-task-MBMF-fMRI-code`, read but not copied) and checked on every file:

| Fact | Evidence |
| --- | --- |
| **Ship → planet:** the left ship commonly (70%) reaches planet 1 (pads 1–2) and the right ship planet 2 (pads 3–4). There is no second-stage choice. | Crosstab of choice × common planet: 100% / 100%. |
| **`state2_1`, `state2_2`:** the pad each ship would reach on that trial. | `outcome2 == state2_{Resp1}` on 100% of responded trials. |
| **`rewBinary_k`:** this trial's win draw for pad k. `rewPWin_k` is its probability, shared by the two pads of a planet. | `rewPWin_1 == rewPWin_2`, `rewPWin_3 == rewPWin_4` everywhere. |
| **State contingency:** the win is the draw of the pad reached. | 100% of trials. |
| **Stimulus contingency:** the win is pad 1's draw for the left ship and pad 3's for the right ship, wherever the ship landed. | 100% of trials. |
| **Points shown:** `outcomeMag = round(win × rewMagnitude × 100)`. | 100% of trials. |
| **`condState`:** sets P(common pad \| planet) to 0.5 (code 1, "high" state uncertainty) or 0.9 (code 2). It works in 26-trial blocks. | Code `pTrans([LOW HIGH]) = [0.9 0.5]`. Empirical within-block max pad share: 0.62 vs 0.89. |
| **`condReward`:** sets the magnitude scale to U(0.30, 1.00) (code 1) or U(0.10, 0.19) (code 2). It works in 26-trial blocks offset by 13 trials. | Observed ranges 0.30–1.00 and 0.10–0.19. |
| **`isPost_ContState` / `isPost_StateLow` / `isPost_RewHigh`:** the *previous* trial's condition (state contingency / `condState` = 2 / `condReward` = 1). They are NaN on the first trial of a run. | Equal to the one-trial lag on 100% of trials. |
| **Reward probabilities:** reverse 7 times per session. After each reversal they start near 0/1 and settle at 0.3 vs 0.5 within about 6 trials. | Data and code (`defineRewardProbability.m`). |
| **Contingency:** **switches once, at the run boundary.** Run 1 is 100% one contingency and run 2 98.7–100% the other. 100 subjects start stimulus-contingent and 89 state-contingent. | Data. |

The adapter now enforces these rules on every file (`reward_rule`, `magnitude_rule`, `pad_rule`, `probability_sharing` checks). It also exposes the schedule as columns: `pad_if_left`, `pad_if_right`, `reward_draw_pad1..4`, `reward_probability_planet1/2`, `magnitude_scale` and `pad_transition_probability`.

## Models

All four models share one set of rules:
- **Values** reset at the start of each run, as in the authors' model.
- **Learning:**
  - MF updates the chosen ship: `Q_mf(a) += α (r − Q_mf(a))`.
  - MB updates the planet reached: `V(s) += α (r − V(s))`, and values each ship as `Q_mb(a) = 0.7 V(common planet) + 0.3 V(other)`.
- **Choice:** `P(a) ∝ exp(β (w Q_mb + (1−w) Q_mf) + stick · [a = previous choice])`.

| Model | Free parameters |
| --- | --- |
| `mf` (w = 0) | α, β, stick |
| `mb` (w = 1) | α, β, stick |
| `hybrid` | α, β, w, stick |
| `hybrid_contingency` | α, β, w_stimulus, w_state, stick |

**Fitting:** MAP with an N(0, 2.5²) prior on the unconstrained parameters (logit / log / identity). This matches the scale of the authors' N(0, 6.25) prior. Each fit uses L-BFGS-B with 4 starts.

**Simplifications compared with the authors' 12-parameter model** (`model_getLLE_magMF_binMB_...m`), which has separate MF and MB learning rates, magnitude-based MF and binary MB learning, learned pad transitions, win-stay/lose-switch terms and trial-wise arbitration:
- One learning rate.
- The same reward signal for MF and MB.
- Fixed, instructed transitions.

This is a reference benchmark, not a replication of their model.

## Pre-specified analysis choices (fixed before any real-data fit)

| Choice | Primary | Sensitivity |
| --- | --- | --- |
| Trials | functional runs only (154 per subject) | — |
| Reward signal | win / no win | points shown / 100 |
| Fast responses (RT < 150 ms, 658 trials) | update values (the outcome was shown) but are not scored | all scored |
| Subjects | exclude > 10% missed functional trials (8 subjects; 181 kept) | all 189 |
| No-response trials | no update, not scored; stickiness refers to the last choice made | — |

## Validation before real data

All recovery used the real 189 task schedules, with each subject's real missed trials kept missing, plus synthetic schedules with the same structure. Choices were always simulated.

| Recovery study | Pearson r, true vs fitted | Group-level model recovery (disjoint groups of 10 simulated subjects, summed held-out log-lik) |
| --- | --- | --- |
| Synthetic schedules, a priori ranges | α 0.66–0.87, β 0.71–0.85, w 0.88, w_stim 0.76, w_state 0.82, stick 0.91–0.94 | mf 9/10, mb 10/10, hybrid 6/10 (3 → mf), hybrid_contingency 6/10 |
| Real schedules, a priori ranges | α 0.79–0.90, β 0.66–0.81, w 0.79, w_stim 0.81, w_state 0.79, stick 0.87–0.95 | mf 8/10, mb 10/10, hybrid 5/10 (3 → mf), hybrid_contingency 2/10 |
| **Real schedules, fitted range** (5th–95th percentiles of real fits) | α 0.78–0.84, β 0.69–0.90, w 0.84, w_stim 0.84, w_state 0.82, stick 0.94–0.95 | **mf 10/10, mb 10/10, hybrid 9/10, hybrid_contingency 6/10 (4 → hybrid)** |

Real fits extend beyond the a priori ranges (α up to 0.95, β 0.7–14, stick down to −1.6), so the fitted-range study is the relevant one.

**Per-subject model selection is unreliable** at 154 trials: for example, only 32/100 hybrid subjects are selected as hybrid. Only group-level comparisons are interpreted.

**Leave-one-run-out cannot evaluate `hybrid_contingency`.** Each run is a single contingency, so a model trained on one run has no information about the other weight. Its run-held-out score is driven by the prior; it is reported but not interpreted. The contingency question is answered by the subject-held-out comparison and the contrast below.

**Known-answer check of the contingency contrast** (one simulated subject per real schedule, fitted range):

| Planted w_state − w_stimulus | Estimate [95% CI] | Time effect estimate |
| --- | --- | --- |
| 0 | −0.019 [−0.045, 0.008] | −0.014 [−0.040, 0.011] |
| 0.15 | 0.110 [0.082, 0.137] | −0.009 [−0.037, 0.018] |

The prior shrinks the estimate by about 25%, but it detects the effect, and the time and contingency estimates do not leak into each other.

## Results (primary analysis; 181 subjects, 26,853 scored trials)

**Held-out log-likelihood per trial** (chance = −0.693):

| Model | Leave-one-run-out | Δ vs hybrid, summed [95% subject-bootstrap CI] | Subject-held-out (5-fold, pooled parameters) |
| --- | --- | --- | --- |
| hybrid | **−0.537** | — | **−0.595** |
| mf | −0.577 | −1,084 [−1,451, −718] | −0.613 |
| mb | −0.578 | −1,116 [−1,605, −680] | −0.623 |
| hybrid_contingency | −0.609 (not interpretable, see above) | −1,943 | −0.595 (= hybrid) |

**Robustness of the comparison:**
- Per-subject parameters predict held-out runs better than pooled parameters predict held-out subjects (−0.537 vs −0.595). This reflects large individual differences.
- **Sensitivity analyses:**
  - **Points** (learning from points shown): the same ranking. Hybrid vs MF is −493 [−878, −103]; hybrid vs MB is −237 [−832, +368] and does not exclude 0. Learning from win/no-win predicts choices better than learning from points (hybrid −0.537 vs −0.597 per held-out trial).
  - **No exclusions** (189 subjects): the same as the primary analysis. Hybrid vs MF is −1,114 [−1,489, −762]; hybrid vs MB is −1,210 [−1,702, −747].

**Fitted parameters (hybrid, median [IQR]):**

| Parameter | Median [IQR] |
| --- | --- |
| α | 0.82 [0.65, 0.91] |
| β | 4.2 [2.0, 7.1] |
| w | 0.58 [0.32, 0.81] |
| stick | 0.05 [−0.55, 0.45] |

Learning rates are high: choices depend mainly on the last few outcomes.

**Does the MB weight follow the contingency?** From `hybrid_contingency` per-subject fits, w_state − w_stimulus:

| First contingency | n | Mean | Median | Wilcoxon p |
| --- | --- | --- | --- | --- |
| stimulus first | 94 | +0.021 | +0.015 | 0.39 |
| state first | 87 | −0.028 | −0.001 | 0.50 |
| **Contingency effect (order-balanced)** | 181 | **−0.003 [−0.039, 0.031]** | | |
| Time effect (run 2 − run 1) | | +0.025 [−0.009, 0.058] | | |

The result is the same with points (+0.010 [−0.032, 0.050]) and with no exclusions (+0.001 [−0.031, 0.035]). With this model and these data, there is **no evidence that participants weight model-based control more when it is the correct strategy**. The CI excludes effects around the size the known-answer check recovers.

**Posterior predictive check:** group mean P(stay) from 20 replays per subject of the fitted agent on its own schedule.

| Source | Contingency | Rewarded, common | Rewarded, rare | Unrewarded, common | Unrewarded, rare | Reward effect | Interaction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| observed | state | 0.824 | 0.591 | 0.384 | 0.670 | 0.181 | 0.259 |
| observed | stimulus | 0.823 | 0.635 | 0.382 | 0.617 | 0.229 | 0.211 |
| hybrid | state | 0.800 | 0.603 | 0.436 | 0.605 | 0.181 | 0.183 |
| hybrid | stimulus | 0.791 | 0.659 | 0.425 | 0.559 | 0.233 | 0.133 |
| mf | state | 0.739 | 0.720 | 0.490 | 0.508 | 0.230 | 0.019 |
| mb | state | 0.749 | 0.450 | 0.515 | 0.694 | −0.005 | 0.239 |

- **MF and MB each miss a whole pattern.** MF shows no transition sensitivity; MB shows no reward main effect. Hybrid matches the reward main effect.
- **Hybrid underpredicts the interaction by about 0.08 in both contingencies.** People shift away after an unrewarded common transition more than the model does (0.38 vs 0.43).
- **The observed interaction is larger under state contingency** (0.259 vs 0.211). The hybrid predicts the same difference (0.183 vs 0.133) with one w, so the difference comes from the reward schedule. It is not evidence of contingency-specific arbitration.

## Limitations

- **One shared learning rate, no MB-specific win-stay/lose-shift, and fixed transitions.** The predictive misses above point to these as the next extensions. The authors' model includes them.
- **Contingency and time are confounded by design.** Only the across-subject counterbalancing of order separates them.
- **Trial-wise arbitration is not modelled.** The authors' `isPost_*` sign-flip terms are not tested.
- **Structural-session trials are not used** (105 per subject, a separate session, also with one contingency switch). They are a natural out-of-session replication.
- **Behavior only.** No fMRI derivative supports trial-level neural analysis (see the audit).

## Reproduce

```bash
ldm ds007474 fetch-behavior --dest data/ds007474_osf
ldm ds007474 trials --osf-root data/ds007474_osf --out data/ds007474_trials.csv
ldm two-step-model recovery --config configs/two_step_recovery.yaml                    # ~4 min, 8 cores
ldm two-step-model recovery --config configs/two_step_recovery_real_schedules.yaml     # ~4 min
ldm two-step-model recovery --config configs/two_step_recovery_fitted_range.yaml       # ~6 min
ldm two-step-model compare --config configs/ds007474_two_step_model.yaml               # ~3.5 min
ldm two-step-model compare --config configs/ds007474_two_step_model_points.yaml
ldm two-step-model compare --config configs/ds007474_two_step_model_no_exclusions.yaml
```
