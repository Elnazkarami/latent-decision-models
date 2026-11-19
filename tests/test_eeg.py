"""EEG cleaning, decoding and ds002158 EEG helpers: known-answer tests on synthetic signals.

No EEG recordings are used; the only real file is the existing 48-trial marker fixture.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from latent_decision_models import eeg_clean, eeg_decoding
from latent_decision_models.datasets import ds002158 as d
from latent_decision_models.datasets import ds002158_eeg as e
from latent_decision_models.io import openneuro
from latent_decision_models.io.brainvision import read_vmrk

FIX = Path(__file__).parent / "fixtures" / "ds002158"
EVENTS = FIX / "sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_events.tsv"
VMRK = FIX / "sub-02/ses-001/eeg/sub-02_ses-001_task-main_run-001_eeg.vmrk"


# ------------------------------------------------------------------ artifact removal
def test_gradient_template_removes_locked_artifact_keeps_unlocked_signal():
    rng = np.random.default_rng(0)
    tr, n_vol, fs = 200, 60, 1000.0
    t = np.arange(tr * n_vol) / fs
    artifact = np.tile(500 * rng.standard_normal(tr), n_vol)  # identical every volume
    brain = 10 * np.sin(2 * np.pi * 3.3 * t)  # not locked to the volume period
    x = np.vstack([artifact + brain, 2 * artifact + brain])
    out = eeg_clean.gradient_correct(x, np.arange(n_vol) * tr, tr, n_template=21)
    # The artifact is gone; the unlocked signal survives (attenuated ~1/21 by the template).
    assert np.abs(out - brain).max() < 0.2 * np.abs(brain).max()
    assert np.corrcoef(out[0], brain)[0, 1] > 0.99


def test_gradient_correct_refuses_unsynchronised_triggers():
    x = np.zeros((1, 1000))
    with pytest.raises(ValueError, match="synchronised"):
        eeg_clean.gradient_correct(x, np.array([0, 100, 210]), 100)


def test_isolated_late_trigger_is_snapped_to_the_grid():
    # Artifact on the grid, one marker logged 2 samples late (as in sub-04 run-004).
    rng = np.random.default_rng(4)
    tr, n_vol = 200, 40
    x = np.tile(300 * rng.standard_normal(tr), n_vol)[None]
    markers = np.arange(n_vol) * tr
    markers[20] += 2
    grid, n_off = eeg_clean.volume_grid(markers, tr)
    assert n_off == 1 and (grid == np.arange(n_vol) * tr).all()
    assert np.abs(eeg_clean.gradient_correct(x, markers, tr)).max() < 1e-3


def test_template_subtract_overlapping_events_corrected_once():
    # Two events 30 samples apart with 50-sample windows overlap; each sample must be
    # corrected exactly once (by the nearest event), so a constant artifact is removed
    # without being subtracted twice.
    x = np.ones((1, 400))
    out, ok = eeg_clean.template_subtract(x, np.array([100, 130, 300]), 20, 30, n_template=3)
    assert ok.all()
    inside = np.r_[80:160, 280:330]
    assert np.allclose(out[0, inside], 0.0)  # not -1: no sample corrected twice
    assert np.allclose(np.delete(out[0], inside), 1.0)  # outside windows: untouched


def test_template_subtract_skips_events_past_the_edges():
    x = np.ones((1, 100))
    out, ok = eeg_clean.template_subtract(x, np.array([5, 50, 95]), 10, 10)
    assert ok.tolist() == [False, True, False]
    assert np.allclose(out[0, 40:60], 0.0) and np.allclose(out[0, :5], 1.0)


def test_causal_filter_never_responds_before_an_impulse():
    x = np.zeros((1, 2000))
    x[0, 1000] = 1.0
    for y in (eeg_clean.causal_bandpass(x, 250.0, 1.0, 10.0),
              eeg_clean.causal_decimate(np.repeat(x, 20, axis=1), 5000.0, 20)[0]):
        k = np.flatnonzero(np.abs(y[0]) > 0)
        assert k.min() >= 1000  # nothing before the impulse
    # A zero-phase filter would leak backwards; this is what the causal choice avoids.
    from scipy.signal import butter, sosfiltfilt
    zp = sosfiltfilt(butter(4, [1, 10], "bandpass", fs=250.0, output="sos"), x[0])
    assert np.abs(zp[:1000]).max() > 1e-3


def test_causal_decimate_rate_and_shape():
    y, fs = eeg_clean.causal_decimate(np.zeros((3, 5000)), 5000.0, 20)
    assert fs == 250.0 and y.shape == (3, 250)


def test_bcg_correct_removes_heartbeat_locked_waveform():
    rng = np.random.default_rng(1)
    fs = 250.0
    peaks = np.cumsum(rng.integers(220, 260, size=120)) + 100
    n = peaks[-1] + 400
    wave = np.hanning(200) * 40
    x = np.zeros((2, n))
    for p in peaks:
        x[:, p - 50: p + 150] += wave
    noise = rng.standard_normal((2, n))
    out, ok = eeg_clean.bcg_correct(x + noise, peaks, fs, median_rr_s=0.96)
    before = eeg_clean.causal_bandpass(x + noise, fs, 1, 10)
    after = eeg_clean.causal_bandpass(out, fs, 1, 10)
    assert ok.sum() >= 115
    assert after[:, 500:-500].std() < 0.3 * before[:, 500:-500].std()


# ------------------------------------------------------------------ decoding
def _synthetic_trials(n_runs=6, per_run=40, effect=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_runs * per_run
    y = rng.integers(0, 2, n).astype(float)
    X = rng.standard_normal((n, 30))
    X[:, :3] += effect * (y[:, None] - 0.5) * 2
    return X, y, np.repeat(np.arange(n_runs), per_run)


def test_decode_finds_planted_effect_and_not_noise():
    X, y, g = _synthetic_trials(effect=0.8)
    hit = eeg_decoding.decode(X, y, g, "classification", n_perm=30, seed=0)
    assert hit["score"] > 0.8 and hit["p_value"] <= 1 / 31 + 1e-9
    X0, y0, g0 = _synthetic_trials(effect=0.0, seed=3)
    miss = eeg_decoding.decode(X0, y0, g0, "classification", n_perm=30, seed=0)
    assert 0.35 < miss["score"] < 0.65 and miss["p_value"] > 0.05


def test_decode_residualised_target_removes_covariate_explained_part():
    # y is fully explained by a covariate that the EEG also carries: residualised
    # decoding must lose the effect, raw decoding must find it.
    rng = np.random.default_rng(2)
    n = 240
    cov = rng.standard_normal(n)
    X = rng.standard_normal((n, 20))
    X[:, 0] += 2 * cov
    y = cov + 0.1 * rng.standard_normal(n)
    g = np.repeat(np.arange(6), 40)
    raw = eeg_decoding.decode(X, y, g, "regression", n_perm=0)
    res = eeg_decoding.decode(X, y, g, "regression", covariates=cov[:, None], n_perm=0)
    assert raw["score"] > 0.6 and abs(res["score"]) < 0.25


def test_permutation_respects_groups():
    rng = np.random.default_rng(0)
    y = np.arange(12.0)
    g = np.repeat([0, 1, 2], 4)
    yp = eeg_decoding._permute_within(y, g, rng)
    for k in range(3):
        assert sorted(yp[g == k]) == sorted(y[g == k])


def test_transfer_and_holm():
    X, y, _ = _synthetic_trials(effect=0.8)
    r = eeg_decoding.transfer(X[:120], y[:120], X[120:], y[120:], "classification",
                              n_perm=50)
    assert r["score"] > 0.8 and r["p_value"] < 0.05
    adj = eeg_decoding.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_window_features_bins():
    times = np.arange(-0.2, 1.2, 1 / 250)
    ep = np.zeros((2, 3, len(times)))
    ep[:, :, (times >= 0.05) & (times < 0.1)] = 1.0
    f = eeg_decoding.window_features(ep, times, 0.0, 0.2, 0.05)
    assert f.shape == (2, 12)
    assert np.allclose(f[0, 3:6], 1.0) and np.allclose(f[0, :3], 0.0)


# ------------------------------------------------------------------ ds002158 EEG helpers
def test_pair_runs_matches_by_timing_not_filename():
    trials, _ = d.parse_events(EVENTS)
    markers = read_vmrk(VMRK, 5000.0)
    other = trials.assign(run="run-002", onset_seconds=trials["onset_seconds"] + 0.75)
    both = pd.concat([trials, other])
    # The EEG file is *named* run-002 but its markers belong to events run-001.
    pairs = e.pair_runs({"run-002": markers}, both)
    row = pairs.iloc[0]
    assert row["events_run"] == "run-001" and not row["same_name"]
    assert row["status"] == "verified"


def test_pair_runs_unmatched():
    trials, _ = d.parse_events(EVENTS)
    shifted = trials.assign(onset_seconds=trials["onset_seconds"] + 0.75)
    assert e.pair_runs({"run-001": read_vmrk(VMRK, 5000.0)}, shifted).iloc[0]["status"] == \
        "unmatched"


def test_decoding_trials_pre_decision_requires_late_enough_response():
    t = pd.DataFrame({"valid_trial": [True, True, True, False],
                      "eeg_rejected": [False, False, True, False],
                      "condition": ["active"] * 4, "eeg_has_response": [True] * 4,
                      "eeg_response_latency_seconds": [0.35, 0.19, 0.4, 0.4]})
    assert e.decoding_trials(t, "active", "pre_decision").tolist() == [True, False, False,
                                                                       False]
    assert e.decoding_trials(t, "active", "post_decision").tolist() == [True, True, False,
                                                                        False]


def test_reject_epochs_robust_rule():
    ptp = np.r_[np.random.default_rng(0).uniform(100, 140, 50), 400.0, 900.0, np.nan]
    t = e.reject_epochs(pd.DataFrame({"eeg_epoch_ptp_uv": ptp}))
    assert t["eeg_rejected"].tolist() == [False] * 50 + [True, True, True]


def test_windows_are_ordered_and_post_decision_precedes_rating_scale():
    assert e.WINDOWS["pre_decision"][1:] == (0.0, 0.2)
    assert e.WINDOWS["response_locked"][0] == "response"
    assert e.WINDOWS["post_decision"][2] <= 1.0  # scale appears ~1.5 s after response


# ------------------------------------------------------------------ IO
def test_signal_fetch_refuses_over_budget_before_any_network_call(tmp_path):
    files = [{"filename": "sub-02/ses-001/eeg/x.eeg", "size": 3_000_000_000,
              "urls": ["http://invalid.invalid/x"]},
             {"filename": "sub-02/ses-001/eeg/y.eeg", "size": 2_000_000_000,
              "urls": ["http://invalid.invalid/y"]}]
    with pytest.raises(PermissionError, match="approved budget"):
        openneuro.fetch_signal_files(files, tmp_path, approved_bytes=4_200_000_000)
    assert not any(tmp_path.iterdir())


def test_metadata_fetch_still_refuses_signal_files():
    with pytest.raises(PermissionError):
        openneuro.select_files([{"filename": "sub-02/ses-001/eeg/x.eeg", "size": 1}],
                               ["*.eeg"])


def test_read_raw_tolerates_stale_datafile_name(tmp_path):
    pytest.importorskip("mne")
    from latent_decision_models.io.brainvision import read_raw

    n_ch, n = 2, 100
    data = (np.arange(n_ch * n, dtype=np.int16) % 50).reshape(n, n_ch)  # multiplexed
    stem = tmp_path / "sub-01_task-x_eeg"
    stem.with_suffix(".eeg").write_bytes(data.tobytes())
    stem.with_suffix(".vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\n"
        "DataFile=old_name.eeg\n[Marker Infos]\nMk1=New Segment,,1,1,0\n"
        "Mk2=Stimulus,S 11,10,1,0\n")
    stem.with_suffix(".vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\n"
        "Codepage=UTF-8\nDataFile=old_name.eeg\nMarkerFile=old_name.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\n"
        "SamplingInterval=200\n[Binary Infos]\nBinaryFormat=INT_16\n[Channel Infos]\n"
        "Ch1=Cz,,0.5,µV\nCh2=ECG,,0.5,µV\n")
    raw = read_raw(stem.with_suffix(".vhdr"))
    assert raw.info["sfreq"] == 5000.0 and raw.ch_names == ["Cz", "ECG"]
    assert np.allclose(raw.get_data()[0] * 1e6, data[:, 0] * 0.5)
    assert (tmp_path / "old_name.eeg").exists() is False  # nothing written next to data
