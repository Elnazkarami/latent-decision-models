"""Known-answer tests for subject-disjoint confidence baselines (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from latent_decision_models.confidence_baseline import prepare, run_confidence_baseline, type2_auroc


def _synthetic(signal: bool, n_sub=10, n_trials=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        for t in range(n_trials):
            acc = int(rng.random() < 0.7)
            conf = (0.3 + 0.5 * acc if signal else 0.6) + rng.normal(0, 0.1)
            rows.append({"subject": f"sub-{s:02d}", "condition": ["active", "observed"][t % 2],
                         "valid_trial": True, "confidence_raw": float(np.clip(conf, 0, 1)),
                         "accuracy": acc, "evidence_dots_difference": rng.integers(5, 25),
                         "response_latency_seconds": rng.uniform(0.25, 0.5)})
    return pd.DataFrame(rows)


def test_known_signal_is_recovered():
    r = run_confidence_baseline(_synthetic(signal=True), n_splits=5, n_boot=200)
    beh = r["models"]["behavior"]
    assert beh["mae_improvement_vs_null_ci95"][0] > 0.1
    assert r["type2_auroc_by_condition"]["active"]["mean"] > 0.9


def test_no_signal_gives_no_improvement():
    r = run_confidence_baseline(_synthetic(signal=False), n_splits=5, n_boot=200)
    assert abs(r["models"]["behavior"]["mae_improvement_vs_null"]) < 0.01
    assert 0.35 < r["type2_auroc_by_condition"]["active"]["mean"] < 0.65


def test_folds_are_subject_disjoint_and_cover_all():
    r = run_confidence_baseline(_synthetic(True), n_splits=5, n_boot=50)
    tested = [s for f in r["folds"] for s in f["test_subjects"]]
    assert sorted(tested) == sorted(set(tested)) and len(tested) == 10


def test_invalid_and_missing_confidence_excluded_not_imputed():
    df = _synthetic(True)
    df.loc[0, "valid_trial"] = False
    df.loc[1, "confidence_raw"] = np.nan
    assert len(prepare(df)) == len(df) - 2


def test_exclude_subjects_and_reproducibility():
    df = _synthetic(True)
    a = run_confidence_baseline(df, n_boot=100, exclude_subjects=("sub-00",))
    b = run_confidence_baseline(df, n_boot=100, exclude_subjects=("sub-00",))
    assert a == b and a["n_subjects"] == 9


def test_unknown_condition_rejected():
    df = _synthetic(True)
    df.loc[0, "condition"] = "passive"
    with pytest.raises(ValueError, match="Unexpected conditions"):
        prepare(df)


def test_held_out_targets_do_not_leak_into_their_predictions():
    """Altering only fold-k test subjects' targets must leave fold-k predictions unchanged."""
    df = _synthetic(True)
    r1 = run_confidence_baseline(df, n_splits=5, n_boot=10, return_predictions=True)
    held = df["subject"].isin(r1["folds"][0]["test_subjects"]).to_numpy()
    df2 = df.copy()
    df2.loc[held, "confidence_raw"] = 1 - df2.loc[held, "confidence_raw"]
    r2 = run_confidence_baseline(df2, n_splits=5, n_boot=10, return_predictions=True)
    p1, p2 = r1["oof_predictions"], r2["oof_predictions"]
    for col in ("pred_null_train_mean", "pred_task", "pred_behavior"):
        np.testing.assert_allclose(p1.loc[held, col], p2.loc[held, col])
        assert not np.allclose(p1.loc[~held, col], p2.loc[~held, col])


def test_type2_auroc_nan_without_errors():
    df = _synthetic(True)
    df["accuracy"] = 1
    assert type2_auroc(df)["auroc2"].isna().all()
