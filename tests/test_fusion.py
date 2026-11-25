"""Behavior + EEG fusion (LDM-005) on synthetic subjects."""

import numpy as np
import pandas as pd
import pytest
from test_eeg import _write_fake_subject

from latent_decision_models import fusion
from latent_decision_models.datasets import ds002158_eeg as e


def _subjects(rng, n_sub=8, n_runs=4, per=40, beh_w=1.0, eeg_w=1.0, k_e=30):
    """y = beh_w * b0 + eeg_w * latent + noise; E carries the latent in 3 features."""
    data = {}
    for i in range(n_sub):
        n = n_runs * per
        B = rng.standard_normal((n, 3))
        latent = rng.standard_normal(n)
        E = rng.standard_normal((n, k_e))
        E[:, :3] += latent[:, None]
        y = beh_w * B[:, 0] + eeg_w * latent + rng.standard_normal(n)
        runs = np.repeat([f"run-{r}" for r in range(n_runs)], per)
        y = y + 3.0 * (np.unique(runs, return_inverse=True)[1] - 1.5)  # run offsets
        data[f"sub-{i:02d}"] = (B, E, y, runs)
    return data


def test_stacking_gains_over_behavior_when_eeg_adds_signal():
    rng = np.random.default_rng(0)
    data = _subjects(rng)
    sc = fusion.per_subject_scores(fusion.loso_predictions(data))
    assert (sc["B+E"] - sc["B"]).mean() > 0.1
    assert (sc["E"] > 0.2).all()
    w = fusion.per_subject_scores(fusion.within_predictions(data))
    assert (w["B+E"] - w["B"]).mean() > 0.05


def test_stacking_does_not_gain_when_eeg_is_noise():
    rng = np.random.default_rng(1)
    data = _subjects(rng, eeg_w=0.0)
    sc = fusion.per_subject_scores(fusion.loso_predictions(data))
    assert abs((sc["B+E"] - sc["B"]).mean()) < 0.02
    assert abs(sc["E"].mean()) < 0.1   # zero in expectation (20 replicates: 0.018 +- 0.007)


def test_per_run_scores_ignore_run_offsets_and_small_subjects_are_skipped():
    pred = pd.DataFrame({"subject": "s", "run": np.repeat(["a", "b"], 5),
                         "y": np.r_[np.arange(5), np.arange(5) + 100.0],
                         "pred_B": np.r_[np.arange(5) + 50.0, np.arange(5)]})
    assert fusion.per_subject_scores(pred)["B"].item() == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    data = _subjects(rng, n_sub=2, n_runs=2)
    with pytest.raises(ValueError):
        pd.concat([fusion.within_predictions(data)])  # all subjects skipped (< 3 runs)
    with pytest.raises(ValueError, match="two training groups"):
        fusion.stacked_predict(*[np.zeros((4, 2))] * 2, np.arange(4.0), np.zeros(4),
                               np.zeros((1, 2)), np.zeros((1, 2)))


def test_mlp_fusion_learns_and_is_reproducible():
    pytest.importorskip("torch")
    from latent_decision_models.fusion_torch import fit_predict_mlp

    rng = np.random.default_rng(3)
    data = _subjects(rng, n_sub=6)
    test = "sub-00"
    tr = [s for s in data if s != test]
    args = (np.concatenate([data[s][0] for s in tr]), np.concatenate([data[s][1] for s in tr]),
            np.concatenate([data[s][2] - data[s][2].mean() for s in tr]),
            np.concatenate([[s] * len(data[s][2]) for s in tr]), data[test][0], data[test][1])
    cfg = {"hidden": 16, "eeg_units": 8, "dropout": 0.2, "lr": 0.01, "weight_decay": 0.01,
           "max_epochs": 200, "patience": 20, "n_val_subjects": 2, "n_seeds": 2}
    p1 = fit_predict_mlp(*args, cfg, seed=0)
    p2 = fit_predict_mlp(*args, cfg, seed=0)
    np.testing.assert_allclose(p1, p2)
    y = data[test][2] - data[test][2].mean()
    runs = data[test][3]
    r = np.mean([np.corrcoef(p1[runs == k], y[runs == k])[0, 1] for k in np.unique(runs)])
    assert r > 0.5


def test_run_fusion_end_to_end_on_fake_epochs(tmp_path):
    """Fake epochs: confidence is planted in post-decision EEG only, so B+E beats B."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    subs = [f"sub-{i:02d}" for i in range(1, 8)]
    for s in subs:
        _write_fake_subject(tmp_path, s, rng, effect=1.0)
    cfg = {"epochs_dir": str(tmp_path), "subjects": subs, "confirmatory_subjects": subs[1:],
           "min_runs": 3, "min_trials": 40, "conditions": ["active", "observed"],
           "window": "post_decision", "bin_seconds": 0.05,
           "behavior": ["accuracy", "evidence_dots_difference", "response_latency_seconds",
                        "accuracy_x_evidence", "accuracy_x_latency"],
           "mlp": {"hidden": 8, "eeg_units": 4, "dropout": 0.2, "lr": 0.01,
                   "weight_decay": 0.01, "max_epochs": 50, "patience": 10,
                   "n_val_subjects": 2, "n_seeds": 1},
           "n_boot": 500, "seed": 0, "n_jobs": 1}
    inputs = e.fusion_inputs(cfg)
    B, E, y, runs = inputs["active"]["sub-01"]
    assert B.shape == (60, 5) and len(y) == len(runs) == len(E) == 60
    np.testing.assert_allclose(B[:, 3], B[:, 0] * B[:, 1])   # accuracy x evidence
    res = e.run_fusion(cfg, inputs)
    t = res["tests"].set_index(["condition", "scheme", "contrast"])
    primary = t.loc[("active", "participant_disjoint", "B+E - B")]
    assert primary["role"] == "primary" and primary["n_subjects"] == 6
    assert primary["mean"] > 0.1 and primary["p_wilcoxon"] < 0.05
    assert t["p_holm_secondary"].notna().sum() == 5
    assert {"B", "E", "B+E", "MLP"} <= set(res["scores"].columns)
