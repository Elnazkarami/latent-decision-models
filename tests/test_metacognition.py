"""Known-answer tests for the GEE metacognition model (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from latent_decision_models.metacognition import metacognition_gee


def _sim(slope_active, slope_observed, n_sub=20, n_trials=200, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        for t in range(n_trials):
            cond = "active" if t % 2 == 0 else "observed"
            conf = rng.uniform(0, 1)
            b = slope_active if cond == "active" else slope_observed
            p = 1 / (1 + np.exp(-(b * (conf - 0.5))))
            rows.append({"subject": f"sub-{s:02d}", "condition": cond, "confidence_raw": conf,
                         "accuracy": int(rng.random() < p)})
    return pd.DataFrame(rows)


def test_recovers_planted_slopes_and_interaction():
    r = metacognition_gee(_sim(8.0, 4.0))
    assert r["confidence_slope_active"] == pytest.approx(8.0, abs=1.0)
    assert r["confidence_slope_observed"] == pytest.approx(4.0, abs=1.0)
    inter = r["slope_difference_observed_minus_active"]
    assert inter["ci95"][1] < 0 and inter["p_value"] < 0.001


def test_no_interaction_when_slopes_equal():
    r = metacognition_gee(_sim(6.0, 6.0))
    lo, hi = r["slope_difference_observed_minus_active"]["ci95"]
    assert lo < 0 < hi


def test_rejects_unknown_condition_and_missing_columns():
    df = _sim(6.0, 6.0, n_sub=4, n_trials=20)
    with pytest.raises(ValueError, match="Missing"):
        metacognition_gee(df.drop(columns="accuracy"))
    df.loc[0, "condition"] = "passive"
    with pytest.raises(ValueError, match="Conditions"):
        metacognition_gee(df)
