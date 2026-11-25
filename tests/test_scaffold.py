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


def test_json_differences_tolerance_and_structure():
    from latent_decision_models.provenance import json_differences

    exp = {"a": 1.0, "b": [1, 2, {"c": "x"}], "t": "2026", "n": float("nan")}
    assert json_differences(exp, {"a": 1.0 + 1e-9, "b": [1, 2, {"c": "x"}], "t": "2026",
                                  "n": float("nan")}) == []
    d = json_differences(exp, {"a": 1.1, "b": [1, 2], "t": "2027", "extra": 0,
                               "n": float("nan")}, ignore_keys=("t",))
    assert any(x.startswith("/a:") for x in d) and any("length 3" in x for x in d)
    assert any("/extra: not expected" in x for x in d)
    assert not any(x.startswith("/t") for x in d)
    assert json_differences(True, 1) != []   # booleans are not numbers
    grown = {"a": 1.0, "new": {"z": 1}}
    assert json_differences({"a": 1.0}, grown, allow_new_keys=True) == []
    assert json_differences({"a": 1.0}, {"new": 1}, allow_new_keys=True) == \
        ["/a: missing in actual"]


def test_cli_compare_reports_exit_code(tmp_path):
    import json
    import subprocess
    import sys

    (tmp_path / "e").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "e" / "r.json").write_text(json.dumps({"x": 1.0}))
    (tmp_path / "a" / "r.json").write_text(json.dumps({"x": 1.0000000001}))
    cmd = [sys.executable, "-m", "latent_decision_models.cli", "compare-reports",
           "--expected", str(tmp_path / "e"), "--actual", str(tmp_path / "a"),
           "--files", "r.json"]
    assert subprocess.run(cmd, check=False, capture_output=True).returncode == 0
    (tmp_path / "a" / "r.json").write_text(json.dumps({"x": 2.0}))
    assert subprocess.run(cmd, check=False, capture_output=True).returncode == 1


def test_cli_verify_files(tmp_path):
    import json
    import subprocess
    import sys

    from latent_decision_models.provenance import sha256_file

    f = tmp_path / "sub-01" / "x.eeg"
    f.parent.mkdir()
    f.write_bytes(b"abc")
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"files": [{"path": "sub-01/x.eeg", "bytes": 3,
                                          "sha256": sha256_file(f)}]}))
    cmd = [sys.executable, "-m", "latent_decision_models.cli", "verify-files",
           "--manifest", str(man), "--root", str(tmp_path)]
    assert subprocess.run(cmd, check=False, capture_output=True).returncode == 0
    f.write_bytes(b"abd")
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert out.returncode == 1 and "SHA-256 mismatch" in out.stdout
