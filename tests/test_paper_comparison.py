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
