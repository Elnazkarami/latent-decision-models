# ds002158 computational confidence model (LDM-002)

**Behavior only, real data.** OpenNeuro ds002158 v1.0.2 events (CC0); no EEG/fMRI. Recovery studies are on **synthetic** data and are labelled as such.

**Bottom line:** the model reproduces mean confidence on correct and error trials and most of the type-2 sensitivity. However:
- it **cannot** tell whether observed choices are evaluated differently from one's own (that comparison is inconclusive, and biased in the region where the real fits lie);
- the correlation between the decision and the confidence evidence (ρ) is **not identifiable** per subject;
- no variant reproduces the small active > observed difference in metacognitive sensitivity.

No mechanistic claim is made from this model.

## Model

This is a static second-order signal-detection model in the tradition of post-decisional and second-order accounts (e.g. Fleming & Daw 2017, *Psychol Rev* 124:91). It is our own specification, not the model fitted by Pereira et al. (2020). Code: [`confidence_model.py`](../src/latent_decision_models/confidence_model.py).

**Evidence.** Per trial, stimulus side is s = ±1 and evidence strength is μ = k · e. Here e = dots difference ÷ the subject's median dots difference.
- *Why normalize:* the staircase holds accuracy near 71% using very different dot counts across subjects (median 8 to 21).
- *What it uses:* stimulus values only, no outcomes.

**Confidence sample.** x_c = sμ + σ_c z.

**Active trials.**
- A decision sample x_a is correlated with x_c, with corr = ρ.
- The choice is a = sign(x_a).
- Both the choice and the rating are scored.

**Observed trials.**
- The choice is the replay of the participant's own earlier decision, so given s it is independent of the current x_c.
- The model conditions on the choice; only the rating is scored. This avoids counting the replayed choices twice.

**Readout.**
- conf = P(choice correct | x_c, a), computed with a believed correlation ρ̂ between the choice and one's own confidence sample.
- The rating is r* = logistic(gain · logit conf + bias) + N(0, σ_r²), censored at 0 and 1.

**Variants** (they differ only in how observed trials are read out):

| Variant | Observed readout | Free parameters |
| --- | --- | --- |
| `agency_blind` | ρ̂_obs = ρ (treats the observed choice as its own) | k, σ_c, ρ, bias, gain, σ_r |
| `normative` | ρ̂_obs = 0 (ideal for an independent replayed choice) | same 6 |
| `free_rho` | ρ̂_obs free | 7 |
| `obs_noise` | ρ̂_obs = 0, separate σ_c for observed trials | 7 |

**Fitting.**
- Maximum likelihood per subject, with z integrated on a 121-point grid. The likelihood is unchanged to 3 decimals at 2,001 points.
- L-BFGS-B, 3 starts.
- Tests check that the likelihood normalizes to 1 and that choice probabilities equal Φ(asμ).

## Step 1: recovery before any real-data fit (synthetic)

Each simulated subject has 240 trials (half active, half observed), roughly the size of a ds002158 subject. Sources:
- [`reports/confidence_model_recovery.json`](../reports/confidence_model_recovery.json): wide a-priori ranges.
- [`reports/confidence_model_recovery_fitted_range.json`](../reports/confidence_model_recovery_fitted_range.json): ranges covering roughly the 10th–90th percentile of the real fits.

I added the second set after seeing the real fits, because the fitted gain (median 0.25) and σ_r (median 0.05) lay outside the a-priori ranges.

**Parameter recovery** (Pearson r between true and fitted values; 60 simulations per model):

| Parameter | a-priori ranges: `agency_blind` / `normative` / `free_rho` / `obs_noise` | fitted-region ranges |
| --- | --- | --- |
| k | 0.90 / 0.89 / 0.91 / 0.91 | 0.69 / 0.64 / 0.55 / 0.59 |
| σ_c | 0.83 / 0.83 / 0.55 / 0.28 | 0.64 / 0.64 / 0.56 / 0.54 |
| σ_r | 0.86 / 0.92 / 0.94 / 0.94 | 0.93 / 0.92 / 0.92 / 0.93 |
| gain | 0.44 / 0.39 / 0.43 / 0.65 | 0.89 / 0.87 / 0.88 / 0.90 |
| bias | 0.48 / 0.42 / 0.35 / 0.42 | 0.85 / 0.87 / 0.73 / 0.84 |
| **ρ** | 0.42 / **0.08** / 0.26 / **0.07** | 0.83 / **0.09** / **−0.03** / 0.21 |
| ρ_obs (`free_rho`) | 0.48 | 0.82 |
| σ_c,obs (`obs_noise`) | 0.68 | 0.56 |

ρ is not identifiable from choices and ratings in `normative`, `free_rho` or `obs_noise`, where it affects active trials only.
- Constraining gain = 1 and bias = 0, or quadrupling the trials to 960, did not fix it: r ≤ 0.48.
- So ρ seems to have only a weak, non-monotonic effect on what we observe.
- In `agency_blind`, ρ also governs observed trials, which is why it recovers there.

**Per-subject model recovery** (lowest BIC; 30 simulations per generating model):
- `free_rho` is selected in only 1/30 of its own simulations, and `obs_noise` in 5–7/30.
- `agency_blind` and `normative` are confused 6–12 times out of 30.

**Group-level model recovery** for the two parameter-matched variants. Each synthetic group has 20 subjects with independent parameters; the model with the lower summed per-subject BIC is selected. 20 groups per generating model:

| Ranges | `agency_blind` → chose `agency_blind` | `normative` → chose `normative` |
| --- | --- | --- |
| a-priori | 18/20 | 19/20 |
| fitted region | 20/20 | **12/20** (8/20 misclassified as `agency_blind`) |

In the region where the real data lie, group BIC is **biased toward `agency_blind`**.

## Step 2: real data (20 subjects, 4,860 valid trials)

Command: `ldm confidence-model compare --config configs/ds002158_confidence_model.yaml`. Every fit converged.

**Primary comparison: leave-one-run-out held-out log-likelihood within each subject.** Each run is scored with parameters fit on that subject's other runs. Differences are relative to `normative`; 95% CIs come from 2,000 participant bootstraps.

| Model | Held-out LL (total) | Δ vs `normative` [95% CI] | Subjects better than `normative` | Summed BIC |
| --- | --- | --- | --- | --- |
| `normative` | 856.4 | — | — | −1909.8 |
| `agency_blind` | 875.5 | +19.1 [−59.6, +99.1] | 11/20 | **−1999.2** |
| `free_rho` | 883.5 | +27.1 [−25.4, +89.9] | 10/20 | −1888.2 |
| `obs_noise` | 869.7 | +13.3 [−53.5, +86.2] | 6/20 | −1877.2 |

- **Held-out data:** no variant is reliably better than `normative`; every CI includes 0.
- **In-sample BIC:** it favours `agency_blind` by 89. Group recovery in the fitted region shows this criterion is biased toward `agency_blind`, so this is **not** evidence that participants treat observed choices as their own.
- **Excluding sub-20** (19 subjects): the same pattern. Δ `agency_blind` = +19.1 [−59.9, +104.8], 10/19 subjects; BIC still favours `agency_blind`.

**Posterior predictive check.** Group means over subjects; 50 simulated replicas per subject from that subject's fit. Active choices are simulated; observed trials keep the actual replayed choice.

| | Data | `agency_blind` | `normative` | `free_rho` | `obs_noise` |
| --- | --- | --- | --- | --- | --- |
| Mean conf, correct (active / observed) | 0.758 / 0.760 | 0.758 / 0.742 | 0.747 / 0.751 | 0.752 / 0.747 | 0.742 / 0.756 |
| Mean conf, error (active / observed) | 0.307 / 0.305 | 0.330 / 0.300 | 0.315 / 0.329 | 0.326 / 0.319 | 0.321 / 0.322 |
| Type-2 AUROC (active / observed) | 0.955 / 0.941 | 0.941 / 0.938 | 0.936 / 0.933 | 0.937 / 0.934 | 0.931 / 0.935 |
| Active − observed AUROC | **+0.014** | +0.003 | +0.003 | +0.003 | −0.004 |

- All variants slightly underpredict active-trial sensitivity.
- None reproduces the active > observed gap, which the descriptive analyses found: type-2 AUROC, and the GEE interaction in [`ds002158-behavioral-baseline.md`](ds002158-behavioral-baseline.md).

**Fitted parameters** (medians, `normative`): k 0.54, σ_c 0.42, gain 0.27, bias −0.11, σ_r 0.05.
- Only k and σ_r are reasonably recoverable here; σ_c only moderately (r ≈ 0.64 in the fitted region).
- **ρ, and the rest, should not be interpreted per subject.**
- Some subjects reach boundary-like values (gain ≈ 0.002, σ_r ≈ 0.004). These were not investigated further.

## Decisions that affect interpretation (please review)
1. **Static model.** Response time and post-decision timing are not modelled. The active RT window is truncated to 0.2–0.5 s by the authors' exclusions, and observed trials have no participant RT. A dynamic (accumulation) model would be needed to use RT.
2. **Observed choices are conditioned on, not predicted.** They are replays, so scoring them would double-count active choices.
3. **Evidence normalization** by each subject's median dots difference (stimulus-only).
4. **Censored ratings:** exact 0 and 1 ratings (60 trials) are treated as censored, not as point values.
5. **The fitted-region recovery ranges were chosen after seeing the real fits.** This is the purpose of that check, but it is post hoc.
6. **The primary comparison is within-subject held-out runs.** Cognitive-model parameters are subject-specific, so held-out *subjects* would need a hierarchical model; that is not implemented. The comparison is reported as a within-person generalization result, not across people.

## Reproduce
```bash
ldm confidence-model recovery --config configs/confidence_model_recovery.yaml               # ~4 min on 8 cores
ldm confidence-model recovery --config configs/confidence_model_recovery_fitted_range.yaml
ldm confidence-model compare --config configs/ds002158_confidence_model.yaml
ldm confidence-model compare --config configs/ds002158_confidence_model_excl_sub-20.yaml
```
Real-data outputs (`reports/ds002158_confidence_model*.json/.csv`) are git-ignored because they contain per-subject parameters. The recovery reports are synthetic and versioned.
