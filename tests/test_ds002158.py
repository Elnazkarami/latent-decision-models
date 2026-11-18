"""Tests for the ds002158 adapter, BrainVision marker reader and metadata fetch guard.

Real-data assertions use the CC0 fixture in tests/fixtures/ds002158 (see PROVENANCE.md).
Edge cases are synthetic mutations of that fixture.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from latent_decision_models.datasets import ds002158 as d
from latent_decision_models.io import openneuro
from latent_decision_models.io.brainvision import read_vhdr_sampling_rate, read_vmrk
from latent_decision_models.provenance import git_blob_sha1
from latent_decision_models.schema import validate_trials

FIX = Path(__file__).parent / "fixtures" / "ds002158"
EVENTS = FIX / "sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_events.tsv"
VHDR = FIX / "sub-02/ses-001/eeg/sub-02_ses-001_task-main_run-001_eeg.vhdr"
VMRK = FIX / "sub-02/ses-001/eeg/sub-02_ses-001_task-main_run-001_eeg.vmrk"
# git blob ids in the OpenNeuroDatasets/ds002158 tree at tag 1.0.2 (audit 2026-11-18)
SNAPSHOT_BLOB_SHA = {
    EVENTS: "450ba0052208951c0f0755a21ec8fffbcbc8ad01",
    FIX / "task-main_events.json": "0b9230cf364641d275ccc8fc2c93b155eca2a205",
}


def _bids_copy(tmp_path, name=None):
    rel = EVENTS.relative_to(FIX)
    dst = tmp_path / "bids" / (rel if name is None else rel.parent / name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(EVENTS, dst)
    return dst


def _rewrite(path, fn):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = fn(df)
    (df if out is None else out).to_csv(path, sep="\t", index=False)


# ---------------------------------------------------------------- real fixture
@pytest.mark.parametrize("path", list(SNAPSHOT_BLOB_SHA))
def test_fixture_bytes_match_snapshot(path):
    assert git_blob_sha1(path.read_bytes()) == SNAPSHOT_BLOB_SHA[path]


def test_fixture_parses_with_verified_contract():
    trials, issues = d.parse_events(EVENTS)
    assert issues == []
    assert len(trials) == 48
    assert trials["condition"].value_counts().to_dict() == {"active": 24, "observed": 24}
    assert validate_trials(trials)["valid"]
    assert trials["run"].unique().tolist() == ["run-001"]
    assert trials["onset_seconds"].is_monotonic_increasing


def test_observed_trials_have_no_participant_rt():
    trials, _ = d.parse_events(EVENTS)
    obs = trials[trials["condition"] == "observed"]
    act = trials[trials["condition"] == "active"]
    assert obs["response_time_seconds"].isna().all()
    assert obs["response_latency_seconds"].notna().all()  # replayed virtual-hand latency
    assert act["response_time_seconds"].notna().all()


def test_missing_confidence_sentinel_becomes_nan_and_invalid():
    trials, _ = d.parse_events(EVENTS)
    row = trials.loc[33]  # fixture row with confidence == -1 and response_time2 == -0.02
    assert row["confidence_missing"] and np.isnan(row["confidence_raw"])
    assert not row["valid_trial"]
    assert "no_confidence_response" in row["exclusion_reasons"]
    assert trials["confidence_raw"].dropna().between(0, 1).all()


def test_choice_side_inference():
    trials, _ = d.parse_events(EVENTS)
    cor = trials["accuracy"] == 1
    assert (trials.loc[cor, "choice_side"] == trials.loc[cor, "stim_side"]).all()
    assert (trials.loc[~cor, "choice_side"] != trials.loc[~cor, "stim_side"]).all()


def test_author_exclusions_have_reasons():
    trials, _ = d.parse_events(EVENTS)
    assert (trials["author_excluded"] == (trials["exclusion_reasons"] != "")).all()
    assert (trials.loc[trials["valid_trial"], "response_latency_seconds"].between(0.2, 0.5)).all()


def test_vmrk_reader():
    sf = read_vhdr_sampling_rate(VHDR)
    assert sf == 5000.0
    mk = read_vmrk(VMRK, sf)
    assert (mk["description"] == "R128").sum() == 375
    assert mk["sample"].is_monotonic_increasing
    assert np.isclose(mk["time_seconds"].iloc[1], mk["sample"].iloc[1] / sf)


def test_alignment_verified_on_fixture():
    trials, _ = d.parse_events(EVENTS)
    table, summary = d.align_eeg_markers(trials, read_vmrk(VMRK, 5000.0))
    assert summary["status"] == "verified"
    assert summary["n_timing_matched"] == 48
    assert summary["max_abs_onset_residual_ms"] <= 0.5
    assert summary["n_response_side_mismatch"] == 0
    # EEG response latency should agree with events response_time within ~1 refresh
    lat = trials["response_latency_seconds"].to_numpy()
    eeg = table["eeg_response_latency_seconds"].astype(float).to_numpy()
    assert np.nanmax(np.abs(lat - eeg)) < 0.05


def test_alignment_does_not_use_row_order():
    trials, _ = d.parse_events(EVENTS)
    shuffled = trials.sample(frac=1, random_state=0)
    _, summary = d.align_eeg_markers(shuffled, read_vmrk(VMRK, 5000.0))
    assert summary["status"] == "verified"


def test_alignment_fails_for_wrong_run_timing():
    trials, _ = d.parse_events(EVENTS)
    shifted = trials.assign(onset_seconds=trials["onset_seconds"] + 0.75)
    _, summary = d.align_eeg_markers(shifted, read_vmrk(VMRK, 5000.0))
    assert summary["status"] == "failed"


def test_alignment_flags_label_disagreement():
    trials, _ = d.parse_events(EVENTS)
    swapped = trials.copy()
    swapped.loc[0, "stim_side"] = "left" if swapped.loc[0, "stim_side"] == "right" else "right"
    _, summary = d.align_eeg_markers(swapped, read_vmrk(VMRK, 5000.0))
    assert summary["status"] == "timing_only"
    assert summary["n_stimulus_side_mismatch"] == 1


# ---------------------------------------------------------------- strict validation
def test_duplicate_rows_rejected(tmp_path):
    p = _bids_copy(tmp_path)
    _rewrite(p, lambda df: df.loc.__setitem__((1, "onset"), df.loc[0, "onset"]))
    with pytest.raises(d.Ds002158SchemaError, match="onset_order"):
        d.parse_events(p)


def test_duplicate_trial_ids_across_files_rejected(tmp_path, monkeypatch):
    p = _bids_copy(tmp_path)
    monkeypatch.setattr(d, "discover_events", lambda root: [p, p])
    with pytest.raises(d.Ds002158SchemaError, match="Duplicate trial identity"):
        d.load_trials(tmp_path / "bids")
    _, issues = d.load_trials(tmp_path / "bids", strict=False)
    assert any(i.code == "duplicate_trial_id" for i in issues)


def test_mismatched_run_directory_rejected(tmp_path):
    dst = tmp_path / "sub-04/ses-001/func" / EVENTS.name  # sub-02 file under sub-04/
    dst.parent.mkdir(parents=True)
    shutil.copy(EVENTS, dst)
    with pytest.raises(d.Ds002158SchemaError, match="subject mismatch"):
        d.parse_events(dst)


def test_unknown_trial_type_rejected(tmp_path):
    p = _bids_copy(tmp_path)
    _rewrite(p, lambda df: df.loc.__setitem__((0, "trial_type"), "active"))
    with pytest.raises(d.Ds002158SchemaError, match="unknown_code"):
        d.parse_events(p)


def test_confidence_out_of_range_rejected(tmp_path):
    p = _bids_copy(tmp_path)
    _rewrite(p, lambda df: df.loc.__setitem__((2, "confidence"), "1.5"))
    with pytest.raises(d.Ds002158SchemaError, match="confidence_range"):
        d.parse_events(p)


def test_unexpected_columns_rejected(tmp_path):
    p = _bids_copy(tmp_path)
    _rewrite(p, lambda df: df.rename(columns={"confidence": "conf"}))
    with pytest.raises(d.Ds002158SchemaError, match="Columns"):
        d.parse_events(p)


def test_non_strict_returns_issues_instead_of_raising(tmp_path):
    p = _bids_copy(tmp_path)
    _rewrite(p, lambda df: df.loc.__setitem__((0, "stim_side"), "middle"))
    trials, issues = d.parse_events(p, strict=False)
    assert trials.empty
    assert any(i.code == "unknown_code" and i.row == 0 for i in issues)


def test_truncated_run_and_block_deviation_are_warnings(tmp_path):
    p = _bids_copy(tmp_path)
    _rewrite(p, lambda df: df.drop(index=range(40, 48)))
    trials, issues = d.parse_events(p)
    assert len(trials) == 40
    assert {i.code for i in issues} == {"short_run"}
    p2 = _bids_copy(tmp_path, name="sub-02_ses-001_task-main_run-002_events.tsv")
    _rewrite(p2, lambda df: df.loc.__setitem__((2, "trial_type"), "obs"))  # row 2: valid "act"
    _, issues2 = d.parse_events(p2)
    assert [i.code for i in issues2] == ["block_structure"]


def test_unflagged_rule_violation_warns(tmp_path):
    p = _bids_copy(tmp_path)
    _rewrite(p, lambda df: df.loc.__setitem__((2, "response_time"), "0.9"))  # row 2 is "act"
    _, issues = d.parse_events(p)
    assert any(i.code == "unflagged_exclusion" and i.row == 2 for i in issues)


def test_discover_skips_rejected(tmp_path):
    _bids_copy(tmp_path)
    rej = tmp_path / "bids/sub-02/ses-001/func/rejected"
    rej.mkdir()
    shutil.copy(EVENTS, rej / EVENTS.name.replace("run-001", "run-009"))
    assert len(d.discover_events(tmp_path / "bids")) == 1


# ---------------------------------------------------------------- fetch guard / provenance
def test_metadata_guard_refuses_signal_files():
    files = [{"filename": "sub-02/ses-001/eeg/x_eeg.eeg", "size": 10, "urls": ["u"]},
             {"filename": "sub-02/ses-001/func/x_bold.nii.gz", "size": 10, "urls": ["u"]}]
    with pytest.raises(PermissionError):
        openneuro.select_files(files, ["*"])
    with pytest.raises(PermissionError):
        openneuro.fetch_files(files[:1], "/nonexistent")


def test_metadata_guard_size_cap(tmp_path):
    files = [{"filename": "big_events.tsv", "size": 10_000_000, "urls": ["u"]}]
    with pytest.raises(PermissionError, match="cap"):
        openneuro.fetch_files(files, tmp_path, max_bytes=1000)


def test_git_blob_sha1_known_value():
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
