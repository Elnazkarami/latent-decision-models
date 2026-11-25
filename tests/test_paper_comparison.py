"""Quantities comparable to Pereira et al. (2020), on synthetic trials with known answers."""

import numpy as np
import pandas as pd
import pytest

from latent_decision_models.datasets import ds002158_paper as pp


def _trials(seed=0, n_sub=12, n=120, steeper_active=True):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        for cond in ("active", "observed"):
            acc = rng.random(n) < 0.72
            sens = 3.0 if (cond == "active") == steeper_active else 1.0
            conf = np.clip(0.6 + 0.12 * sens * (acc - 0.5) + rng.normal(0, 0.15, n), 0, 1)
            rt = 0.4 - 0.05 * (conf - 0.6) * np.where(acc, 1, -1) + rng.normal(0, 0.03, n)
            rows.append(pd.DataFrame({
                "subject": f"sub-{s:02d}", "condition": cond, "accuracy": acc.astype(int),
                "confidence_raw": conf, "response_time_seconds": rt,
                "response_latency_seconds": rt, "evidence_dots_difference": 13.0,
                "valid_trial": True, "exclusion_reasons": ""}))
    return pd.concat(rows, ignore_index=True)


def test_descriptives_and_mean_confidence():
    t = _trials()
    t.loc[0, ["valid_trial", "exclusion_reasons"]] = [False, "no_first_order_response"]
    d = pp.descriptives(t)
    assert d["P1_n_subjects"] == 12
    assert d["P2_valid_trials_per_subject"]["active"]["mean"] == pytest.approx(120 - 1 / 12)
    assert 65 < d["P3_accuracy_percent"]["mean"] < 79
    assert d["P5_rt_ms_active_valid"]["mean"] == pytest.approx(400, abs=10)
    assert pp.mean_confidence_test(t)["df"] == 11


def test_metacognitive_contrasts_detect_the_planted_direction():
    t = _trials()
    sl = pp.logistic_slopes(t)
    assert sl["n_active_steeper"] >= 10 and sl["p"] < 0.01
    a2 = pp.type2_test(t)
    assert a2["active"]["mean"] > a2["observed"]["mean"] and a2["p"] < 0.01
    assert a2["V"] > 60   # sum of positive ranks (active - observed), n = 12
    flipped = pp.logistic_slopes(_trials(steeper_active=False))
    assert flipped["n_active_steeper"] <= 2


def test_rt_confidence_signs():
    r = pp.rt_confidence(_trials())
    assert r["active_correct"]["mean"] < 0 and r["active_error"]["mean"] > 0


def test_wilcoxon_v_is_the_sum_of_positive_ranks():
    res = pp._wilcoxon([1.0, -2.0, 3.0, 0.0])
    assert res["n"] == 3 and res["V"] == 4.0   # ranks 1, 2, 3; positives 1 + 3


def test_ds007474_reward_probability_and_stay_regression():
    from latent_decision_models.datasets import ds007474_paper as d7

    rng = np.random.default_rng(1)
    rows = []
    for s in range(15):
        n = 200
        p1 = rng.uniform(0.2, 0.8, n)
        p2 = rng.uniform(0.2, 0.8, n)
        choice = rng.integers(1, 3, n)
        rare = rng.random(n) < 0.3
        planet = np.where(rare, 3 - choice, choice)
        win = (rng.random(n) < 0.5).astype(int)
        # model-based stay: repeat after win+common or loss+rare
        for i in range(1, n):
            if rng.random() < 0.8:
                mb_stay = (win[i - 1] == 1) != rare[i - 1]
                choice[i] = choice[i - 1] if mb_stay else 3 - choice[i - 1]
        rows.append(pd.DataFrame({
            "subject": f"s{s}", "session": "functional", "run": "run-1",
            "trial_index": np.arange(n), "valid_trial": True, "choice": choice,
            "planet": planet, "rare_transition": rare.astype(int), "reward_binary": win,
            "reward_contingency": np.where(np.arange(n) < n // 2, "stimulus", "state"),
            "reward_probability_planet1": p1, "reward_probability_planet2": p2}))
    t = pd.concat(rows, ignore_index=True)
    r = d7.reward_probability(t)
    assert r["null_percent"] == pytest.approx(50, abs=2)
    st = d7.stay_regression(t)
    assert st["win:rare"]["coef"] < 0 and st["win:rare"]["p"] < 0.001
    c = d7.compare(t)
    assert c["D2_common_transition_fraction"] == pytest.approx(0.7, abs=0.03)
