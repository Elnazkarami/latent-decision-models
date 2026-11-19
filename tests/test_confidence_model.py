"""Second-order SDT confidence model: analytic checks, normalization, known-answer recovery.

All data here are synthetic except ``test_prepare_on_real_fixture`` (CC0 fixture).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr

from latent_decision_models import confidence_model as cm
from latent_decision_models.datasets import ds002158 as d

FIX = Path(__file__).parent / "fixtures" / "ds002158"
TRUE = {"k": 0.7, "sigma_c": 0.6, "rho": 0.7, "bias": 0.2, "gain": 1.1, "sigma_r": 0.12}


def _params(model, **extra):
    p = {**TRUE, **extra}
    return cm.from_theta(cm.to_theta({**cm.DEFAULT_START, **p}, model), model)


@pytest.mark.parametrize("model", list(cm.MODELS))
def test_theta_round_trip(model):
    p = _params(model, rho_obs=0.3, sigma_c_obs=0.9)
    back = cm.from_theta(cm.to_theta(p, model), model)
    for name in cm.MODELS[model]:
        assert back[name] == pytest.approx(p[name], rel=1e-9)
    assert back["rho_obs"] == {"agency_blind": p["rho"], "normative": 0.0,
                               "free_rho": 0.3, "obs_noise": 0.0}[model]


def test_readout_closed_form_when_choice_independent():
    # rho_hat = 0: logit conf = 2 a mu x / sigma^2 + log Phi(mu) - log Phi(-mu)
    x = np.linspace(-3, 3, 13)
    mu, sig = 0.8, 0.7
    for a in (-1.0, 1.0):
        got = cm._logit_conf(x, a, mu, sig, 0.0)
        want = 2 * a * mu * x / sig ** 2 + log_ndtr(mu) - log_ndtr(-mu)
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_readout_monotone_and_symmetric():
    x = np.linspace(-3, 3, 61)
    for rho_hat in (0.0, 0.5, 0.95):
        right = cm._logit_conf(x, 1.0, 0.6, 0.8, rho_hat)
        left = cm._logit_conf(-x, -1.0, 0.6, 0.8, rho_hat)
        assert np.all(np.diff(right) > 0)
        np.testing.assert_allclose(right, left, atol=1e-10)


@pytest.mark.parametrize("model", list(cm.MODELS))
def test_likelihood_normalizes(model):
    """Active: sum_a ∫ p(a, r | s) dr = 1 and ∫ p(a, r|s) dr = Phi(a s mu). Observed: ∫ = 1."""
    p = _params(model, rho_obs=0.4, sigma_c_obs=1.1)
    rg = np.linspace(0, 1, 4001)[1:-1]
    dr = rg[1] - rg[0]

    def total(a, observed, e=1.2, s=1.0):
        n = len(rg)
        dat = cm.ModelData(s=np.full(n, s), a=np.full(n, a), e=np.full(n, e), r=rg,
                           observed=np.full(n, observed))
        cens = cm.ModelData(s=np.full(2, s), a=np.full(2, a), e=np.full(2, e),
                            r=np.array([0.0, 1.0]), observed=np.full(2, observed))
        return (np.exp(cm.trial_loglik(p, dat, 241)).sum() * dr
                + np.exp(cm.trial_loglik(p, cens, 241)).sum())

    mu = p["k"] * 1.2
    for a in (-1.0, 1.0):
        assert total(a, True) == pytest.approx(1.0, abs=2e-3)
        assert total(a, False) == pytest.approx(np.exp(log_ndtr(a * mu)), abs=2e-3)


def test_simulated_accuracy_matches_model():
    rng = np.random.default_rng(0)
    design = pd.DataFrame({"observed": np.zeros(20000, bool), "condition": "active",
                           "s": rng.choice([-1.0, 1.0], 20000), "e": np.ones(20000)})
    sim = cm.simulate(design, _params("normative"), rng)
    assert sim["accuracy"].mean() == pytest.approx(np.exp(log_ndtr(TRUE["k"])), abs=0.01)
    assert sim["r"].between(0, 1).all()


def test_known_answer_parameter_recovery():
    rng = np.random.default_rng(3)
    true = _params("normative")
    sim = cm.simulate(cm.synthetic_design(3000, rng), true, rng)
    est = cm.fit(sim, "normative", n_starts=1)["params"]
    assert est["k"] == pytest.approx(true["k"], rel=0.15)
    assert est["sigma_c"] == pytest.approx(true["sigma_c"], rel=0.2)
    assert est["rho"] == pytest.approx(true["rho"], abs=0.2)
    assert est["sigma_r"] == pytest.approx(true["sigma_r"], rel=0.15)


def test_obs_noise_detected_when_present():
    rng = np.random.default_rng(5)
    sim = cm.simulate(cm.synthetic_design(2000, rng),
                      _params("obs_noise", sigma_c_obs=2.0), rng)
    bic = {m: cm.fit(sim, m, n_starts=1)["bic"] for m in ("normative", "obs_noise")}
    assert bic["obs_noise"] < bic["normative"] - 10


def test_compare_models_structure():
    rng = np.random.default_rng(1)
    frames = []
    for subj in ("sub-a", "sub-b", "sub-c"):
        sim = cm.simulate(cm.synthetic_design(120, rng), _params("normative"), rng)
        sim["subject"] = subj
        sim["run"] = np.repeat(["run-001", "run-002"], 60)
        frames.append(sim)
    res = cm.compare_models(pd.concat(frames), ("normative", "agency_blind"),
                            n_starts=1, n_boot=200)
    assert res["n_subjects"] == 3 and res["n_trials"] == 360
    s = res["summary"]
    assert s["normative"]["delta_vs_reference_total"] == 0.0
    assert s["normative"]["n_subjects_best"] + s["agency_blind"]["n_subjects_best"] == 3
    assert len(res["fits"]) == 6


def test_prepare_on_real_fixture():
    trials, _ = d.load_trials(FIX)
    df = cm.prepare_model_trials(trials)
    valid = trials[trials["valid_trial"]]
    assert len(df) == len(valid)
    assert df.groupby("subject")["e"].median().eq(1.0).all()
    assert ((df["s"] == df["a"]).astype(int) == df["accuracy"]).all()
    assert set(df["observed"]) == {True, False}


def test_prepare_rejects_inconsistent_accuracy():
    trials, _ = d.load_trials(FIX)
    bad = trials.copy()
    i = bad.index[bad["valid_trial"]][0]
    bad.loc[i, "accuracy"] = 1 - bad.loc[i, "accuracy"]
    with pytest.raises(ValueError, match="inconsistent"):
        cm.prepare_model_trials(bad)


def test_group_model_recovery_counts():
    res = cm.group_model_recovery(["agency_blind", "normative"], n_groups=2, n_subjects=2,
                                  n_trials=60, n_starts=1)
    for gen in ("agency_blind", "normative"):
        assert sum(res["confusion"][gen].values()) == 2


def test_unfreed_rating_parameters_take_fixed_defaults():
    cm.MODELS["_test_fixed"] = ("k", "sigma_c", "rho", "sigma_r")
    try:
        p = cm.from_theta(np.zeros(4), "_test_fixed")
        assert p["gain"] == 1.0 and p["bias"] == 0.0 and p["rho_obs"] == 0.0
    finally:
        del cm.MODELS["_test_fixed"]
