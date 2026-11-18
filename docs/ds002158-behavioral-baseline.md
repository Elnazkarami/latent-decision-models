# ds002158 behavioral confidence baseline (LDM-002, first pass)

**Real data, behavior only.** Source: OpenNeuro ds002158 v1.0.2 events files (CC0). No EEG or fMRI is used. These are *post-decision* models: they include objective accuracy and response latency, which exist only after the first-order decision. They describe how well behavior accounts for confidence **in held-out participants**. They are not prospective predictions and not neural results.

## Setup
- **Target:** `confidence_raw`, continuous on [0, 1] (scale verified in [`ds002158-audit.md`](ds002158-audit.md)). Valid trials only, as defined by the authors' `-bad` flags. Missing confidence is never imputed.
- **Models**, each a superset of the previous one:
  - `null_train_mean`: the training-fold mean.
  - `task`: evidence (`dots_difference`) + condition.
  - `behavior`: adds accuracy, response latency, and accuracy×condition and latency×condition interactions. For observed trials, latency is the replayed virtual-hand latency the participant saw.
- **Evaluation:** 5-fold GroupKFold by subject. Standardization is fit inside each training fold. Ridge α = 1 is fixed a priori, so there is no tuning and no nested CV is needed. Metrics are computed on out-of-fold predictions. 95% CIs come from 2,000 participant-level bootstrap resamples of per-subject MAE.
- **Metacognitive sensitivity (descriptive):** per-subject type-2 AUROC of confidence for correct vs error, by condition.

## Results

### Primary (20 subjects)

4860 valid trials (2437 active, 2423 observed); GroupKFold(n_splits=5) by subject; scaler fit per train fold; ridge α = 1.0 (fixed).

| Model | MAE, mean over subjects [95% CI] | MAE active / observed | Improvement vs null [95% CI] | Subjects improved | Spearman ρ (pooled OOF) |
|---|---|---|---|---|---|
| `null_train_mean` | 0.222 [0.203, 0.239] | 0.215 / 0.227 | — | — | — |
| `task` | 0.216 [0.199, 0.234] | 0.210 / 0.222 | 0.005 [0.002, 0.009] | 14/20 | 0.16 |
| `behavior` | 0.136 [0.122, 0.152] | 0.129 / 0.140 | 0.085 [0.068, 0.102] | 19/20 | 0.59 |

Type-2 AUROC (confidence discriminating correct vs error, per subject): active 0.955 (SD 0.038), observed 0.941 (SD 0.045); active − observed = 0.014 [95% CI 0.0000, 0.029], higher for active in 16/20 subjects.

Standardized ridge coefficients, `behavior` model fit on all data (descriptive only): `evidence_dots_difference` +0.019, `is_observed` -0.001, `accuracy` +0.201, `response_latency_seconds` +0.006, `accuracy_x_observed` +0.002, `latency_x_observed` -0.000

### Sensitivity: excluding sub-20 (19 subjects)

4583 valid trials (2299 active, 2284 observed); GroupKFold(n_splits=5) by subject; scaler fit per train fold; ridge α = 1.0 (fixed).

| Model | MAE, mean over subjects [95% CI] | MAE active / observed | Improvement vs null [95% CI] | Subjects improved | Spearman ρ (pooled OOF) |
|---|---|---|---|---|---|
| `null_train_mean` | 0.221 [0.202, 0.240] | 0.214 / 0.228 | — | — | — |
| `task` | 0.215 [0.196, 0.234] | 0.207 / 0.222 | 0.006 [0.002, 0.010] | 15/19 | 0.16 |
| `behavior` | 0.137 [0.122, 0.152] | 0.129 / 0.141 | 0.084 [0.064, 0.102] | 18/19 | 0.57 |

Type-2 AUROC (confidence discriminating correct vs error, per subject): active 0.954 (SD 0.038), observed 0.939 (SD 0.046); active − observed = 0.014 [95% CI -0.0006, 0.029], higher for active in 15/19 subjects.

Standardized ridge coefficients, `behavior` model fit on all data (descriptive only): `evidence_dots_difference` +0.020, `is_observed` -0.002, `accuracy` +0.197, `response_latency_seconds` +0.008, `accuracy_x_observed` +0.005, `latency_x_observed` -0.002

## Interpretation (proportional to the evidence)
- Evidence strength and condition alone barely beat the null model: MAE improves by about 0.005. That is expected, because the staircase holds accuracy near 71% and so compresses the range of evidence.
- Adding behavior cuts MAE by about 0.085 (about 38%) in held-out participants, consistently across 19/20 subjects. **Objective accuracy carries almost all of this.** Confidence tracking accuracy is metacognitive sensitivity itself, so this baseline describes behavior; it is not a mechanistic model.
- Type-2 AUROC is very high in both conditions. It is slightly higher for active than observed decisions, but the CI touches 0 and crosses it once sub-20 is excluded. **Read this as directionally consistent with the paper's reported active > observed effect, not as an independent confirmation.** These AUROCs are not directly comparable to the preprint's (0.92 vs 0.90): the trial sets differ (the observation button-press exclusions are not in events.tsv) and so do the estimation procedures.
- Excluding sub-20 changes nothing material.
- Limitations:
  - N = 20, with observed trials that replay the active trials, so the two conditions are not independent samples.
  - The staircase makes evidence depend on recent accuracy.
  - The bootstrap CIs are descriptive and not corrected for the several comparisons shown.

## Additional analyses (LDM-002, second pass)

### Calibration of held-out-subject predictions
OLS of observed on predicted confidence, pooled out-of-fold (ideal: slope 1, intercept 0).

| Model | Slope (all 20) | Intercept (all 20) | Slope (excl. sub-20) | Intercept (excl. sub-20) |
|---|---|---|---|---|
| `task` | 0.755 | 0.153 | 0.817 | 0.113 |
| `behavior` | 0.991 | 0.005 | 0.987 | 0.006 |

### Within-subject, run-held-out evaluation (separate estimand)
These are per-subject models trained on that subject's other runs (leave-one-run-out). They measure how well behavior explains confidence *once a person's own scale use is learned*. Don't compare them with the held-out-subject numbers as if they were the same claim.

| Model | MAE, mean over subjects [95% CI] | Improvement vs within-subject null [95% CI] | Gain over held-out-subject model [95% CI] | Subjects better within |
|---|---|---|---|---|
| `null_train_mean` | 0.222 [0.203, 0.240] | — | — | — |
| `task` | 0.213 [0.196, 0.229] | 0.009 [0.006, 0.014] | 0.004 [-0.001, 0.008] | 13/20 |
| `behavior` | 0.127 [0.113, 0.141] | 0.095 [0.078, 0.112] | 0.010 [0.005, 0.015] | 14/20 |

### Group metacognitive sensitivity (GEE)
`accuracy ~ confidence × observed`, logistic GEE clustered on subject, exchangeable working correlation, robust SEs. Confidence is on its native 0–1 scale.

| Term | All 20: estimate [95% CI], p | Excl. sub-20: estimate [95% CI], p |
|---|---|---|
| `Intercept` | -5.97 [-7.16, -4.79], p = 4.59e-23 | -6.04 [-7.31, -4.76], p = 1.51e-20 |
| `confidence` | 12.62 [10.46, 14.78], p = 2.07e-30 | 12.84 [10.51, 15.16], p = 2.5e-27 |
| `observed` | 1.04 [0.32, 1.76], p = 0.00448 | 1.16 [0.43, 1.90], p = 0.00195 |
| `confidence:observed` | -1.73 [-2.95, -0.51], p = 0.0056 | -1.92 [-3.21, -0.63], p = 0.00349 |

Confidence slope: active 12.62, observed 10.89 (SE 0.87). Working correlation 0.012. n = 4860 trials, 20 subjects.

### Interpretation of the second pass
- **Calibration:** the `behavior` model is well calibrated in new participants (slope ≈ 0.99, intercept ≈ 0.005). The `task` model's slope of about 0.76–0.82 shows its weak predictions are slightly too spread.
- **Personalization:** learning a person's own mapping gives only a small extra gain (MAE about 0.010 lower). Most of what behavior explains about confidence generalizes across participants.
- **Metacognition:** confidence discriminates accuracy less steeply for observed than for committed decisions. The negative `confidence:observed` interaction excludes 0, and it holds without sub-20. This matches the direction reported by Pereira et al. (2020), and it is consistent with the type-2 AUROC result above. Caveats:
  1. GEE slopes are population-averaged; the paper used a mixed model with random slopes, so the magnitudes are not comparable.
  2. A pooled slope mixes within-person sensitivity with between-person differences in confidence bias.
  3. Observed trials replay the active stimuli and outcomes, which makes this a paired design, not independent samples.
  4. This analysis was specified after the earlier results were seen. It is confirmatory in direction only, not a preregistered test.

## Reproduce
```bash
ldm ds002158 fetch-metadata --dest data/ds002158
ldm ds002158 trials --bids-root data/ds002158 --out data/ds002158_trials.csv
ldm confidence-baseline --config configs/ds002158_confidence_baseline.yaml
ldm confidence-baseline --config configs/ds002158_confidence_baseline_excl_sub-20.yaml
```
Outputs are written to `reports/ds002158_confidence_baseline*.json` (git-ignored, regenerated deterministically with seed 0).

## Not yet done (remaining LDM-002 scope)
- A mixed-effects (random-slope) logistic model for a closer match to the paper, e.g. via an optional `bambi`/`pymer4` extra. Also a within-subject-centered confidence variant.
- Ordinal/beta-regression alternatives for the bounded confidence target.
- A computational confidence model (evidence accumulation with post-decision evidence), with a parameter-recovery test on synthetic data before fitting real data. **This is the next ticket.**
