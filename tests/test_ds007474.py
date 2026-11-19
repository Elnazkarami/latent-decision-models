"""ds007474 adapter and OSF helpers.

The OSF behavior files carry no license, so no real data are committed: these tests build
files with the verified structure (column header, 105 structural + 2 x 77 functional
trials, codes) but synthetic values. A real-data check runs only if the files are present
locally (``ldm ds007474 fetch-behavior --dest data/ds007474_osf``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from latent_decision_models.datasets import ds007474 as d
from latent_decision_models.io import osf
from latent_decision_models.schema import validate_trials

REAL = Path(__file__).parents[1] / "data" / "ds007474_osf"


def _synthetic_csv(order="s2f3", seed=0, n_missed=3):
    rng = np.random.default_rng(seed)
    if order == "s2f3":
        runs = [1] * 105 + [2] * 77 + [3] * 77
        sess = ["structural"] * 105 + ["functional"] * 154
    else:
        runs = [1] * 77 + [2] * 77 + [3] * 105
        sess = ["functional"] * 154 + ["structural"] * 105
    n = len(runs)
    resp = rng.choice([1.0, 2.0], n)
    state = rng.integers(1, 5, n).astype(float)
    win = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame({c: 0.0 for c in d.CSV_COLUMNS}, index=range(n))
    df["runID"] = runs
    df["sess_type"] = sess
    df["trialID"] = np.arange(1, n + 1)
    df["condContingent"] = rng.integers(1, 3, n)
    df["condState"] = rng.integers(1, 3, n)
    df["condReward"] = rng.integers(1, 3, n)
    df["doRareTrans"] = (rng.random(n) < 0.3).astype(int)
    df["Resp1"] = resp
    df["RT1"] = rng.uniform(0.2, 1.5, n)
    df["outcome2"] = state
    df["outcome1"] = np.ceil(state / 2).astype(int)
    df["outcomeBin"] = win
    df["outcomeMag"] = win * rng.integers(1, 100, n)
    missed = rng.choice(np.flatnonzero(np.array(sess) == "functional"), n_missed, replace=False)
    df.loc[missed, ["Resp1", "outcome2", "outcomeBin", "outcomeMag"]] = np.nan
    df.loc[missed, "outcome1"] = 1  # placeholder written by the authors' pipeline
    return df[list(d.CSV_COLUMNS)], missed


def _write(tmp_path, df, name="sub0002s2f3.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("order", ["s2f3", "s3f2"])
def test_parse_synthetic(tmp_path, order):
    df, missed = _synthetic_csv(order)
    trials, issues = d.parse_behavior_csv(_write(tmp_path, df, f"sub0002{order}.csv"),
                                          "MBMF000202")
    assert issues == []
    assert validate_trials(trials)["valid"]
    assert trials.groupby("run").size().to_dict() == {"run-1": 77, "run-2": 77,
                                                      "structural": 105}
    func = trials[trials["session_type"] == "functional"]
    assert func.groupby("run")["trial_index"].agg(["min", "max"]).values.tolist() == [[0, 76]] * 2
    gone = trials.loc[missed]
    assert not gone["valid_trial"].any()
    assert gone[["choice", "planet", "second_state", "response_time_seconds"]].isna().all().all()
    ok = trials["valid_trial"]
    assert (trials.loc[ok, "planet"] == np.ceil(trials.loc[ok, "second_state"] / 2)).all()
    assert set(trials["choice_side"].dropna()) <= {"left", "right"}


def test_rejects_wrong_header(tmp_path):
    df, _ = _synthetic_csv()
    with pytest.raises(d.Ds007474SchemaError, match="header"):
        d.parse_behavior_csv(_write(tmp_path, df.rename(columns={"Resp1": "resp1"})))


def test_rejects_session_order_contradicting_filename(tmp_path):
    df, _ = _synthetic_csv("s2f3")
    with pytest.raises(d.Ds007474SchemaError, match="contradicts"):
        d.parse_behavior_csv(_write(tmp_path, df, "sub0002s3f2.csv"))


def test_rejects_inconsistent_state_coding(tmp_path):
    df, missed = _synthetic_csv()
    i = next(i for i in range(len(df)) if i not in set(missed))
    df.loc[i, "outcome1"] = 3 - df.loc[i, "outcome1"]
    with pytest.raises(d.Ds007474SchemaError, match="ceil"):
        d.parse_behavior_csv(_write(tmp_path, df))
    _, issues = d.parse_behavior_csv(_write(tmp_path, df), strict=False)
    assert [x.code for x in issues] == ["state_coding"]


def _timing_table(trials, rng):
    func = trials[trials["session_type"] == "functional"].reset_index(drop=True)
    start = np.array([1000.0, 2000.0])
    stim = np.concatenate([np.cumsum(rng.uniform(8, 10, 77)) for _ in range(2)])
    run = np.repeat([1.0, 2.0], 77)
    t_stim = stim + start[(run - 1).astype(int)]
    rt = func["response_time_seconds"].fillna(2.002).to_numpy()
    table = pd.DataFrame({
        "runID": run, "resp1": func["choice"].to_numpy(), "RT1": rt,
        "condState": func["state_condition"].map({"high": 1, "low": 2}).to_numpy(),
        "doRareTrans": func["rare_transition"].to_numpy(),
        "tStim1On": t_stim, "tResp1": t_stim + rt, "tState2On": t_stim + 3,
        "tTerminalState2On": t_stim + 4, "tFBOn_Bin": t_stim + 5, "tFBOn_Mag": t_stim + 5})
    return table, start, stim


def test_timing_merge(tmp_path):
    df, _ = _synthetic_csv()
    trials, _ = d.parse_behavior_csv(_write(tmp_path, df), "MBMF000202")
    table, start, stim = _timing_table(trials, np.random.default_rng(1))
    merged = d.merge_timing(trials, d._timing_frame(table, start))
    func = merged[merged["session_type"] == "functional"]
    np.testing.assert_allclose(func["onset_seconds"].to_numpy(), stim)
    assert (func["onset_reference"] == "first_scanner_pulse_of_run").all()
    assert merged.loc[merged["session_type"] == "structural", "onset_seconds"].isna().all()
    assert validate_trials(merged)["valid"] and len(merged) == 259


def test_timing_merge_detects_disagreement(tmp_path):
    df, _ = _synthetic_csv()
    trials, _ = d.parse_behavior_csv(_write(tmp_path, df), "MBMF000202")
    table, start, _ = _timing_table(trials, np.random.default_rng(2))
    i = int(np.flatnonzero(table["resp1"].notna())[0])
    table.loc[i, "resp1"] = 3 - table.loc[i, "resp1"]
    with pytest.raises(d.Ds007474SchemaError, match="choice"):
        d.merge_timing(trials, d._timing_frame(table, start))


def test_timing_requires_rt_consistency():
    table = pd.DataFrame({"runID": [1.0], "resp1": [1.0], "RT1": [0.5], "condState": [1],
                          "doRareTrans": [0], "tStim1On": [10.0], "tResp1": [10.9],
                          "tState2On": [12.0], "tTerminalState2On": [13.0],
                          "tFBOn_Bin": [14.0], "tFBOn_Mag": [14.0]})
    with pytest.raises(d.Ds007474SchemaError, match="RT1"):
        d._timing_frame(table, np.array([0.0]))


def test_subject_map(tmp_path):
    mat = tmp_path / d.MAT_DIR
    mat.mkdir()
    (mat / "subID_0002_sesID_2_MBMF_fMRI.mat").write_bytes(b"")
    (mat / "subID_0048_sesID_1_MBMF_fMRI.mat").write_bytes(b"")
    assert d.subject_map(tmp_path) == {"0002": "MBMF000202", "0048": "MBMF004801"}


def test_osf_selection_guard_and_verify():
    files = [{"path": "/a/x.csv", "size": 3, "sha256": None},
             {"path": "/a/y.nii.gz", "size": 3, "sha256": None}]
    with pytest.raises(PermissionError, match="allow-list"):
        osf.check_selection(files, (".csv",))
    with pytest.raises(PermissionError, match="cap"):
        osf.check_selection(files[:1], (".csv",), max_total_bytes=2)
    import hashlib
    good = {"path": "/a/x.csv", "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert osf.verify(b"abc", good)["sha256_ok"]
    assert not osf.verify(b"abd", good)["sha256_ok"]
    url = osf._with_view_only("https://api.osf.io/v2/nodes/x/files/?page=2", "tok")
    assert "page=2" in url and "view_only=tok" in url


@pytest.mark.skipif(not (REAL / d.CSV_DIR).exists(), reason="OSF behavior not downloaded")
def test_real_data_if_present():
    trials, issues = d.load_trials(REAL)
    assert issues == [] and trials["subject"].nunique() == 189 and len(trials) == 189 * 259
    assert validate_trials(trials)["valid"]
