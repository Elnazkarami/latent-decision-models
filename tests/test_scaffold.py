import pandas as pd
import pytest

from latent_decision_models.benchmark import benchmark
from latent_decision_models.schema import validate_trials
from latent_decision_models.synthetic import generate_trials


def test_synthetic_repeatable():
    pd.testing.assert_frame_equal(generate_trials(), generate_trials())


def test_synthetic_valid():
    result = validate_trials(generate_trials())
    assert result["valid"], result["errors"]


def test_duplicate_trial_fails():
    df = generate_trials()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert "Duplicate trial identity keys" in validate_trials(df)["errors"]


def test_invalid_reaction_time_fails():
    df = generate_trials()
    df.loc[0, "reaction_time_seconds"] = -1
    assert not validate_trials(df)["valid"]


def test_group_disjoint_and_reproducible():
    df = generate_trials()
    a = benchmark(df, ["evidence", "accuracy", "reaction_time_seconds"])
    b = benchmark(df, ["evidence", "accuracy", "reaction_time_seconds"])
    assert a == b
    for f in a["folds"]:
        assert not (set(f["train_subjects"]) & set(f["test_subjects"]))


def test_not_enough_groups():
    df = generate_trials(subjects=4)
    with pytest.raises(ValueError, match="Not enough"):
        benchmark(df, ["evidence"], n_splits=5)
