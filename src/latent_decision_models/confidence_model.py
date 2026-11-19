"""Second-order signal-detection model of confidence for committed vs observed decisions.

The model is static (one sample per stream; no response-time dynamics). Per trial, with
stimulus side ``s`` in {-1, +1} and normalized evidence ``e`` (dots difference divided by
the subject's median dots difference; the staircase targets constant accuracy, so raw dots
are not comparable across subjects), evidence strength is ``mu = k * e``.

Generative process
------------------
* Confidence sample: ``x_c = s * mu + sigma_c * z`` with ``z ~ N(0, 1)``.
* **Active** (committed) trials: a decision sample
  ``x_a = s * mu + rho * z + sqrt(1 - rho**2) * eps`` with ``corr(x_a, x_c) = rho``, and the
  choice is ``a = sign(x_a)``.
* **Observed** trials: the choice ``a`` is a replay of the participant's own earlier
  decision on the same stimulus. It is therefore conditionally independent of the current
  ``x_c`` given ``s``. The model conditions on ``a`` and does not score it.

Readout
-------
The observer reports ``P(choice correct | x_c, a)`` under a *believed* correlation
``rho_hat`` between the choice and its own confidence sample:

    L(s') = N(x_c; s' mu, sigma_c^2) * Phi(a (s' mu + rho_hat (x_c - s' mu) / sigma_c)
                                            / sqrt(1 - rho_hat^2))
    conf  = L(a) / (L(a) + L(-a))

Rating: ``r* = expit(gain * logit(conf) + bias) + N(0, sigma_r^2)``, censored to [0, 1]
(ratings of exactly 0 or 1 are treated as censored).

Variants (how observed trials are read out)
-------------------------------------------
* ``agency_blind``: ``rho_hat_obs = rho``. The observer treats observed choices as its own.
* ``normative``: ``rho_hat_obs = 0``. Ideal observer of a replayed (independent) choice.
* ``free_rho``: ``rho_hat_obs`` is a free parameter.
* ``obs_noise``: normative, plus a separate confidence-noise ``sigma_c_obs`` for observed
  trials.

This is our own model in the tradition of second-order / post-decisional signal detection
accounts of confidence (e.g. Fleming & Daw 2017, Psychol Rev 124:91). It is not the model
fitted in Pereira et al. (2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, logsumexp
from scipy.stats import norm

RHO_MAX = 0.98
BASE_PARAMS = ("k", "sigma_c", "rho", "bias", "gain", "sigma_r")
MODELS: dict[str, tuple[str, ...]] = {
    "agency_blind": BASE_PARAMS,
    "normative": BASE_PARAMS,
    "free_rho": (*BASE_PARAMS, "rho_obs"),
    "obs_noise": (*BASE_PARAMS, "sigma_c_obs"),
}
_LOG_PARAMS = {"k", "sigma_c", "gain", "sigma_r", "sigma_c_obs"}
_RHO_PARAMS = {"rho", "rho_obs"}
DEFAULT_START = {"k": 0.6, "sigma_c": 1.0, "rho": 0.5, "bias": 0.0, "gain": 1.0,
                 "sigma_r": 0.15, "rho_obs": 0.0, "sigma_c_obs": 1.0}
FIXED_DEFAULTS = {"bias": 0.0, "gain": 1.0}  # used when a model does not free them
CENSOR_EPS = 1e-9


# ---------------------------------------------------------------- parameters
def to_theta(params: dict, model: str) -> np.ndarray:
    """Natural-scale parameters to the unconstrained optimization vector."""
    out = []
    for name in MODELS[model]:
        v = params[name]
        if name in _LOG_PARAMS:
            out.append(np.log(v))
        elif name in _RHO_PARAMS:
            out.append(np.arctanh(np.clip(v / RHO_MAX, -0.999999, 0.999999)))
        else:
            out.append(v)
    return np.asarray(out, dtype=float)


def from_theta(theta: np.ndarray, model: str) -> dict:
    """Unconstrained vector to natural-scale parameters, including derived readout rho."""
    p = {}
    for name, v in zip(MODELS[model], theta, strict=True):
        if name in _LOG_PARAMS:
            p[name] = float(np.exp(v))
        elif name in _RHO_PARAMS:
            p[name] = float(RHO_MAX * np.tanh(v))
        else:
            p[name] = float(v)
    for name, v in FIXED_DEFAULTS.items():
        p.setdefault(name, v)
    p.setdefault("sigma_c_obs", p["sigma_c"])
    if model.startswith("agency_blind"):
        p["rho_obs"] = p["rho"]
    elif "rho_obs" not in p:
        p["rho_obs"] = 0.0
    return p


# ---------------------------------------------------------------- data
@dataclass(frozen=True)
class ModelData:
    """Arrays for one set of trials (any number of subjects)."""

    s: np.ndarray        # stimulus side, +1 right / -1 left
    a: np.ndarray        # choice side, +1 right / -1 left
    e: np.ndarray        # normalized evidence (>= 0)
    r: np.ndarray        # rating in [0, 1]
    observed: np.ndarray  # bool

    def __len__(self) -> int:
        return len(self.s)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> ModelData:
        return cls(s=df["s"].to_numpy(float), a=df["a"].to_numpy(float),
                   e=df["e"].to_numpy(float), r=df["r"].to_numpy(float),
                   observed=df["observed"].to_numpy(bool))


def prepare_model_trials(trials: pd.DataFrame, exclude_subjects=()) -> pd.DataFrame:
    """Valid ds002158 trials to model columns ``s, a, e, r, observed`` (+ ids, accuracy).

    ``e`` is dots difference / the subject's median dots difference over valid trials (a
    stimulus-only normalization; no outcome information is used).
    """
    v = trials[trials["valid_trial"].astype(bool)].copy()
    v = v[~v["subject"].isin(exclude_subjects)]
    v = v.dropna(subset=["confidence_raw", "stim_side", "choice_side"])
    side = {"right": 1.0, "left": -1.0}
    v["s"] = v["stim_side"].map(side)
    v["a"] = v["choice_side"].map(side)
    if v[["s", "a"]].isna().any().any():
        raise ValueError("unexpected stim_side/choice_side codes")
    if not ((v["s"] == v["a"]).astype(int) == v["accuracy"].astype(int)).all():
        raise ValueError("choice_side/stim_side inconsistent with accuracy")
    med = v.groupby("subject")["evidence_dots_difference"].transform("median")
    v["e"] = v["evidence_dots_difference"] / med
    v["r"] = v["confidence_raw"].astype(float)
    v["observed"] = v["condition"].eq("observed")
    return v[["subject", "run", "condition", "observed", "s", "a", "e", "r",
              "accuracy"]].reset_index(drop=True)


# ---------------------------------------------------------------- likelihood
def _grid(n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(-7.0, 7.0, n_grid)
    w = norm.pdf(z)
    return z, np.log(w / w.sum())


def _logit_conf(x, a, mu, sigma, rho_hat):
    """logit P(s = a | x_c, a) under the readout model. Shapes broadcast."""
    c = np.sqrt(1.0 - rho_hat ** 2)
    la = -((x - a * mu) ** 2) / (2 * sigma ** 2) + log_ndtr(
        (mu + a * rho_hat * (x - a * mu) / sigma) / c)
    lb = -((x + a * mu) ** 2) / (2 * sigma ** 2) + log_ndtr(
        (-mu + a * rho_hat * (x + a * mu) / sigma) / c)
    return la - lb


def _rating_mean(logit_conf, p):
    return expit(p["gain"] * logit_conf + p["bias"])


def _tobit_logpdf(r, m, sigma_r):
    lo = r <= CENSOR_EPS
    hi = r >= 1 - CENSOR_EPS
    out = norm.logpdf(r, m, sigma_r)
    out = np.where(lo, log_ndtr((0.0 - m) / sigma_r), out)
    return np.where(hi, log_ndtr((m - 1.0) / sigma_r), out)


def trial_loglik(params: dict, data: ModelData, n_grid: int = 121) -> np.ndarray:
    """Per-trial log-likelihood: ``log p(a, r | s)`` for active, ``log p(r | a, s)`` observed."""
    z, logw = _grid(n_grid)
    obs = data.observed
    mu = (params["k"] * data.e)[:, None]
    sig = np.where(obs, params["sigma_c_obs"], params["sigma_c"])[:, None]
    rho_hat = np.where(obs, params["rho_obs"], params["rho"])[:, None]
    s, a = data.s[:, None], data.a[:, None]
    x = s * mu + sig * z[None, :]
    lr = _tobit_logpdf(data.r[:, None], _rating_mean(_logit_conf(x, a, mu, sig, rho_hat),
                                                     params), params["sigma_r"])
    rho = params["rho"]
    lpa = log_ndtr(a * (s * mu + rho * z[None, :]) / np.sqrt(1 - rho ** 2))
    lpa = np.where(obs[:, None], 0.0, lpa)
    return logsumexp(lr + lpa + logw[None, :], axis=1)


def negloglik(theta, model, data, n_grid=121):
    ll = trial_loglik(from_theta(theta, model), data, n_grid).sum()
    return -ll if np.isfinite(ll) else 1e12


# ---------------------------------------------------------------- simulation
def simulate(design: pd.DataFrame, params: dict, rng: np.random.Generator,
             observed_choice: str = "independent") -> pd.DataFrame:
    """Simulate choices and ratings for a design with columns ``s, e, observed``.

    Observed choices are drawn as an independent replay (``"independent"``) or taken from
    the design's ``a`` column (``"given"``). Active choices are always simulated.
    """
    out = design.copy()
    n = len(out)
    obs = out["observed"].to_numpy(bool)
    s = out["s"].to_numpy(float)
    mu = params["k"] * out["e"].to_numpy(float)
    z = rng.standard_normal(n)
    rho = params["rho"]
    a_act = np.sign(s * mu + rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(n))
    if observed_choice == "given":
        a_obs = out["a"].to_numpy(float)
    else:
        a_obs = np.sign(s * mu + rng.standard_normal(n))
    a = np.where(obs, a_obs, a_act)
    a[a == 0] = 1.0
    sig = np.where(obs, params["sigma_c_obs"], params["sigma_c"])
    rho_hat = np.where(obs, params["rho_obs"], rho)
    x = s * mu + sig * z
    m = _rating_mean(_logit_conf(x, a, mu, sig, rho_hat), params)
    out["a"] = a
    out["accuracy"] = (a == s).astype(int)
    out["r"] = np.clip(m + params["sigma_r"] * rng.standard_normal(n), 0.0, 1.0)
    return out


def synthetic_design(n_trials: int, rng: np.random.Generator) -> pd.DataFrame:
    """Half active / half observed, random sides, evidence ~ lognormal(0, 0.25)."""
    obs = np.arange(n_trials) % 2 == 1
    return pd.DataFrame({"observed": obs, "condition": np.where(obs, "observed", "active"),
                         "s": rng.choice([-1.0, 1.0], n_trials),
                         "e": rng.lognormal(0.0, 0.25, n_trials)})


# ---------------------------------------------------------------- fitting
def fit(df: pd.DataFrame, model: str, n_starts: int = 3, seed: int = 0,
        n_grid: int = 121) -> dict:
    """Maximum-likelihood fit with L-BFGS-B from ``n_starts`` starting points."""
    data = ModelData.from_frame(df)
    rng = np.random.default_rng(seed)
    base = to_theta(DEFAULT_START, model)
    bounds = [(-6.0, 4.0)] * len(base)
    best = None
    for i in range(n_starts):
        x0 = base if i == 0 else base + rng.normal(0, 0.5, len(base))
        res = minimize(negloglik, x0, args=(model, data, n_grid), method="L-BFGS-B",
                       bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    k = len(base)
    n = len(data)
    return {"model": model, "params": from_theta(best.x, model), "theta": best.x.tolist(),
            "nll": float(best.fun), "n_params": k, "n_trials": n,
            "aic": float(2 * best.fun + 2 * k), "bic": float(2 * best.fun + k * np.log(n)),
            "converged": bool(best.success)}


def heldout_loglik(fitted: dict, df: pd.DataFrame, n_grid: int = 121) -> float:
    return float(trial_loglik(fitted["params"], ModelData.from_frame(df), n_grid).sum())


# ---------------------------------------------------------------- recovery
RECOVERY_RANGES = {"k": (0.3, 1.2), "sigma_c": (0.3, 1.5), "rho": (0.1, 0.9),
                   "bias": (-0.5, 0.5), "gain": (0.5, 1.5), "sigma_r": (0.08, 0.25),
                   "rho_obs": (0.0, 0.8), "sigma_c_obs": (0.3, 1.5)}


def sample_params(model: str, rng: np.random.Generator, ranges=None) -> dict:
    ranges = {**RECOVERY_RANGES, **(ranges or {})}
    p = {name: float(rng.uniform(*ranges[name])) for name in MODELS[model]}
    return from_theta(to_theta(p, model), model)


def _recovery_one(model, i, seed, n_trials, n_starts, ranges=None):
    rng = np.random.default_rng([seed, i])
    true = sample_params(model, rng, ranges)
    sim = simulate(synthetic_design(n_trials, rng), true, rng)
    est = fit(sim, model, n_starts=n_starts, seed=i)["params"]
    return [{"model": model, "sim": i, "param": name, "true": true[name],
             "fitted": est[name]} for name in MODELS[model]]


def parameter_recovery(model: str, n_sims: int, n_trials: int, seed: int = 0,
                       n_starts: int = 2, n_jobs: int = 1, ranges=None) -> pd.DataFrame:
    """Simulate from ``model`` with random parameters, refit it, return true vs fitted."""
    out = Parallel(n_jobs=n_jobs)(delayed(_recovery_one)(model, i, seed, n_trials, n_starts,
                                                         ranges)
                                  for i in range(n_sims))
    return pd.DataFrame([row for rows in out for row in rows])


def summarize_recovery(rec: pd.DataFrame) -> pd.DataFrame:
    def f(g):
        return pd.Series({"pearson_r": np.corrcoef(g["true"], g["fitted"])[0, 1],
                          "median_abs_error": float((g["fitted"] - g["true"]).abs().median()),
                          "median_bias": float((g["fitted"] - g["true"]).median())})
    return rec.groupby(["model", "param"]).apply(f, include_groups=False).reset_index()


def _model_recovery_one(gen, models, i, seed, n_trials, n_starts, ranges=None):
    rng = np.random.default_rng([seed, i, list(MODELS).index(gen)])
    sim = simulate(synthetic_design(n_trials, rng), sample_params(gen, rng, ranges), rng)
    return gen, {m: fit(sim, m, n_starts=n_starts, seed=i)["bic"] for m in models}


def model_recovery(models, n_sims: int, n_trials: int, seed: int = 0, n_starts: int = 2,
                   n_jobs: int = 1, ranges=None) -> pd.DataFrame:
    """Confusion counts: rows = generating model, columns = model with lowest BIC."""
    out = Parallel(n_jobs=n_jobs)(
        delayed(_model_recovery_one)(gen, models, i, seed, n_trials, n_starts, ranges)
        for gen in models for i in range(n_sims))
    counts = pd.DataFrame(0, index=list(models), columns=list(models))
    for gen, bics in out:
        counts.loc[gen, min(bics, key=bics.get)] += 1
    return counts


def _group_subject_bics(gen, models, g, j, seed, n_trials, n_starts, ranges=None):
    rng = np.random.default_rng([seed, g, j, list(MODELS).index(gen), 99])
    sim = simulate(synthetic_design(n_trials, rng), sample_params(gen, rng, ranges), rng)
    return gen, g, {m: fit(sim, m, n_starts=n_starts, seed=j)["bic"] for m in models}


def group_model_recovery(models, n_groups: int, n_subjects: int, n_trials: int,
                         seed: int = 0, n_starts: int = 2, n_jobs: int = 1,
                         ranges=None) -> dict:
    """Group-level model recovery: per synthetic group of ``n_subjects`` (each with its own
    random parameters), select the model with the lowest summed per-subject BIC.

    Returns the confusion counts and, per generating model, the summed-BIC differences.
    """
    out = Parallel(n_jobs=n_jobs)(
        delayed(_group_subject_bics)(gen, models, g, j, seed, n_trials, n_starts, ranges)
        for gen in models for g in range(n_groups) for j in range(n_subjects))
    tot: dict = {}
    for gen, g, bics in out:
        acc = tot.setdefault((gen, g), dict.fromkeys(models, 0.0))
        for m, b in bics.items():
            acc[m] += b
    counts = pd.DataFrame(0, index=list(models), columns=list(models))
    for (gen, _), bics in tot.items():
        counts.loc[gen, min(bics, key=bics.get)] += 1
    return {"n_groups": n_groups, "n_subjects": n_subjects, "n_trials": n_trials,
            "confusion": {g: counts.loc[g].to_dict() for g in counts.index}}


# ---------------------------------------------------------------- real-data comparison
def _subject_fits(subj, g, models, n_starts, seed, n_grid):
    fits, cv = [], []
    runs = sorted(g["run"].unique())
    for m in models:
        f = fit(g, m, n_starts=n_starts, seed=seed, n_grid=n_grid)
        fits.append({"subject": subj, **{k: v for k, v in f.items() if k != "theta"}})
        ll = 0.0
        for run in runs:
            tr, te = g[g["run"] != run], g[g["run"] == run]
            ll += heldout_loglik(fit(tr, m, n_starts=n_starts, seed=seed, n_grid=n_grid),
                                 te, n_grid)
        cv.append({"subject": subj, "model": m, "heldout_ll": ll, "n_trials": len(g),
                   "n_runs": len(runs)})
    return fits, cv


def compare_models(df: pd.DataFrame, models=tuple(MODELS), reference: str = "normative",
                   n_starts: int = 3, n_boot: int = 2000, seed: int = 0,
                   n_grid: int = 121, n_jobs: int = 1) -> dict:
    """Per-subject fits; within-subject leave-one-run-out held-out log-likelihood.

    Returns per-subject parameters and fit statistics, and group comparisons of held-out
    log-likelihood relative to ``reference`` (participant bootstrap CIs).
    """
    rng = np.random.default_rng(seed)
    out = Parallel(n_jobs=n_jobs)(delayed(_subject_fits)(subj, g, models, n_starts, seed, n_grid)
                                  for subj, g in df.groupby("subject"))
    fits = [f for fs, _ in out for f in fs]
    cv = [c for _, cs in out for c in cs]
    cv_df = pd.DataFrame(cv)
    wide = cv_df.pivot(index="subject", columns="model", values="heldout_ll")
    n_tr = cv_df.groupby("subject")["n_trials"].first()
    best = wide.idxmax(axis=1)
    summary = {}
    for m in models:
        d = (wide[m] - wide[reference]).to_numpy()
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].sum(axis=1)
        summary[m] = {
            "heldout_ll_total": float(wide[m].sum()),
            "heldout_ll_per_trial": float(wide[m].sum() / n_tr.sum()),
            "delta_vs_reference_total": float(d.sum()),
            "delta_vs_reference_ci95": [float(np.quantile(boot, 0.025)),
                                        float(np.quantile(boot, 0.975))],
            "n_subjects_better_than_reference": int((d > 0).sum()),
            "n_subjects_best": int((best == m).sum()),
            "bic_total": float(sum(f["bic"] for f in fits if f["model"] == m)),
        }
    return {"reference": reference, "n_subjects": int(wide.shape[0]),
            "n_trials": int(n_tr.sum()), "summary": summary,
            "per_subject_heldout": cv_df.to_dict(orient="records"), "fits": fits}


def posterior_predictive(df: pd.DataFrame, fits: list[dict], model: str, n_rep: int = 50,
                         seed: int = 0) -> pd.DataFrame:
    """Per subject × condition: observed vs simulated mean rating by accuracy and type-2 AUROC.

    Active choices are simulated; observed trials keep the actual replayed choice.
    """
    from sklearn.metrics import roc_auc_score

    def stats(g):
        y = g["accuracy"].to_numpy()
        auc = roc_auc_score(y, g["r"]) if 0 < y.sum() < len(y) else np.nan
        return {"mean_conf_correct": g.loc[y == 1, "r"].mean(),
                "mean_conf_error": g.loc[y == 0, "r"].mean(), "auroc2": auc}

    rng = np.random.default_rng(seed)
    params = {f["subject"]: f["params"] for f in fits if f["model"] == model}
    rows = []
    for subj, g in df.groupby("subject"):
        sims = [simulate(g, params[subj], rng, observed_choice="given") for _ in range(n_rep)]
        for cond, gc in g.groupby("condition"):
            emp = stats(gc)
            sim = pd.DataFrame([stats(s[s["condition"] == cond]) for s in sims]).mean()
            rows.append({"subject": subj, "condition": cond, "model": model,
                         **{f"{k}_data": v for k, v in emp.items()},
                         **{f"{k}_model": float(sim[k]) for k in emp}})
    return pd.DataFrame(rows)
