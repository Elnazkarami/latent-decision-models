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
        eeg_clean.gradient_correct(x, np.array([0, 100, 211]), 100)


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


def test_grid_phase_ignores_a_late_first_trigger():
    # First marker logged 1 sample late (as in sub-07 run-002); a 6-sample late marker
    # later on (as in sub-15) is within tolerance.
    tr, n_vol = 200, 40
    markers = np.arange(n_vol) * tr
    markers[0] += 1
    markers[30] += 6
    grid, n_off = eeg_clean.volume_grid(markers, tr)
    assert n_off == 2 and (grid == np.arange(n_vol) * tr).all()


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


# ------------------------------------------------------------------ run drift, group level
def test_per_group_score_is_robust_to_run_drift():
    # The target drifts between runs and the features carry nothing: pooled
    # leave-one-run-out scoring is biased below chance, per-run scoring is not.
    rng = np.random.default_rng(5)
    n_runs, per = 6, 40
    g = np.repeat(np.arange(n_runs), per)
    run_means = np.linspace(-1, 1, n_runs)
    pooled, per_run = [], []
    for rep in range(8):
        y = run_means[g] + rng.standard_normal(len(g))
        X = rng.standard_normal((len(g), 10))
        pooled.append(eeg_decoding.decode(X, y, g, "regression", n_perm=0)["score"])
        per_run.append(eeg_decoding.decode(X, y, g, "regression", n_perm=0,
                                           score_mode="per_group")["score"])
    assert np.mean(pooled) < -0.15
    assert abs(np.mean(per_run)) < 0.05


def test_group_test_known_answers():
    hit = eeg_decoding.group_test(np.full(12, 0.6) + np.linspace(-0.02, 0.02, 12), 0.5)
    assert hit["n_above_chance"] == 12 and hit["p_wilcoxon"] < 0.001
    assert hit["ci95"][0] > 0.5
    miss = eeg_decoding.group_test(np.linspace(-0.1, 0.1, 11), 0.0)
    assert miss["p_wilcoxon"] > 0.3
    assert eeg_decoding.group_test([0.1, 0.2], 0.0)["p_wilcoxon"] is None


def _write_fake_subject(root, subject, rng, effect):
    sf, ch = 250, ["Cz", "Pz", "ECG", "Oz"]
    st = np.arange(*[round(v * sf) for v in e.STIM_EPOCH]) / sf
    rt = np.arange(*[round(v * sf) for v in e.RESP_EPOCH]) / sf
    n = 4 * 30
    conf = rng.random(n)
    side = rng.integers(0, 2, n)
    stim = rng.standard_normal((n, len(ch), len(st))).astype(np.float32)
    resp = rng.standard_normal((n, len(ch), len(rt))).astype(np.float32)
    resp[:, 0, (rt >= 0.1) & (rt < 0.6)] += effect * (conf[:, None] - 0.5) * 4
    resp[:, 1, (rt >= -0.2) & (rt < 0.1)] += 2.0 * (side[:, None] - 0.5)
    t = pd.DataFrame({
        "subject": subject, "run": np.repeat([f"run-00{i}" for i in range(1, 5)], 30),
        "condition": np.tile(["active", "observed"], n // 2), "valid_trial": True,
        "eeg_rejected": False, "eeg_has_response": True,
        "eeg_response_latency_seconds": 0.35, "response_latency_seconds": 0.35,
        "choice_side": np.where(side == 1, "right", "left"),
        "accuracy": rng.integers(0, 2, n), "evidence_dots_difference": rng.integers(1, 20, n),
        "confidence_raw": conf, "eeg_bad_channels": np.nan})
    t.to_csv(root / f"{subject}_trials.csv", index=False)
    np.savez_compressed(root / f"{subject}_epochs.npz", stim=stim, resp=resp, stim_times=st,
                        resp_times=rt, ch_names=np.array(ch))


def test_run_decoding_group_loso_recovers_planted_effects(tmp_path):
    rng = np.random.default_rng(0)
    subs = [f"sub-{i:02d}" for i in range(1, 8)]
    for s in subs:
        _write_fake_subject(tmp_path, s, rng, effect=1.0)
    cfg = {"epochs_dir": str(tmp_path), "subjects": subs, "confirmatory_subjects": subs[1:],
           "min_runs": 3, "min_trials": 40, "conditions": ["active"],
           "windows": ["response_locked", "post_decision"], "score_mode": "per_group",
           "n_perm": 0, "n_perm_transfer": 0, "loso": True, "n_jobs": 1,
           "targets": {"choice_side": {"task": "classification"},
                       "confidence": {"task": "regression", "primary": True}}}
    res = e.run_decoding(cfg)
    assert "ECG" in res["excluded_channels"]
    g = res["group"].set_index(["analysis", "target", "window"])
    assert g.loc[("within_subject", "choice_side", "response_locked"), "p_wilcoxon"] < 0.05
    assert g.loc[("within_subject", "confidence", "post_decision"), "p_holm_primary"] < 0.05
    assert g.loc[("leave_one_subject_out", "confidence", "post_decision"), "mean"] > 0.3
    assert g.loc[("within_subject", "confidence", "response_locked"), "p_wilcoxon"] > 0.05
    assert (g["n_subjects"] == 6).all()  # sub-01 is not confirmatory


def test_signal_fetch_refuses_to_fill_the_disk(tmp_path):
    import shutil

    from latent_decision_models.io import openneuro

    free = shutil.disk_usage(tmp_path).free
    files = [{"filename": "sub-01/x_eeg.eeg", "size": 10, "urls": ["https://invalid.example/x"]}]
    with pytest.raises(OSError, match="reserve"):
        openneuro.fetch_signal_files(files, tmp_path, approved_bytes=100,
                                     min_free_bytes=free + 1)


def test_obs_removes_artifact_with_varying_amplitude_and_keeps_unlocked_signal():
    # A BCG-like artifact whose amplitude drifts beat to beat: OBS (mean + PCs) removes it
    # better than a moving-average template; broadband non-locked activity survives.
    # (A single pure sinusoid would not: at random phases it lies in the span of two
    # PCs and is fitted away, a known OBS caveat.)
    rng = np.random.default_rng(5)
    fs, n_beats, period = 250.0, 300, 200
    shape = np.hanning(120) * 40
    n = n_beats * period + 200
    peaks = 100 + period * np.arange(n_beats) + rng.integers(-20, 21, n_beats)
    gain = 1 + 0.5 * np.sin(np.arange(n_beats) / 7)
    art = np.zeros(n)
    for p, g in zip(peaks, gain, strict=True):
        art[p - 40:p + 80] += g * shape
    brain = eeg_clean.causal_bandpass(5 * rng.standard_normal((1, n)), fs, 1.0, 30.0)[0]
    x = (art + brain)[None]
    out, ok = eeg_clean.obs_subtract(x, peaks, 60, 140, n_pc=3)
    aas, _ = eeg_clean.template_subtract(x, peaks, 60, 140, n_template=21)
    inner = slice(2000, n - 2000)
    err_obs = np.sqrt(np.mean((out[0, inner] - brain[inner]) ** 2))
    err_aas = np.sqrt(np.mean((aas[0, inner] - brain[inner]) ** 2))
    assert ok.all()
    assert err_obs < 0.5 * err_aas
    r_obs = np.corrcoef(out[0, inner], brain[inner])[0, 1]
    r_aas = np.corrcoef(aas[0, inner], brain[inner])[0, 1]
    assert r_obs > 0.9 and r_obs > r_aas


def test_run_decoding_custom_windows_time_course(tmp_path):
    # Confidence is planted at +0.1..0.6 s after the response: a sliding time course finds
    # it there (Holm across the windows) and not before the response.
    rng = np.random.default_rng(1)
    subs = [f"sub-{i:02d}" for i in range(1, 9)]  # 6 would floor Holm p at 0.0625
    for s in subs:
        _write_fake_subject(tmp_path, s, rng, effect=1.0)
    custom = {"pre": ["response", -0.4, -0.3], "early": ["response", 0.0, 0.1],
              "mid": ["response", 0.2, 0.3], "late": ["response", 0.4, 0.5]}
    cfg = {"epochs_dir": str(tmp_path), "subjects": subs, "confirmatory_subjects": subs,
           "conditions": ["active"], "custom_windows": custom, "windows": list(custom),
           "score_mode": "per_group", "n_perm": 0, "n_perm_transfer": 0, "loso": False,
           "n_jobs": 1, "holm_by": ["analysis", "target", "condition"],
           "targets": {"confidence": {"task": "regression"}}}
    res = e.run_decoding(cfg)
    assert res["windows"]["mid"] == ("response", 0.2, 0.3)
    g = res["group"].set_index("window")
    assert g.loc["mid", "p_holm_by"] < 0.05 and g.loc["late", "p_holm_by"] < 0.05
    assert g.loc["pre", "p_wilcoxon"] > 0.05
    assert (g["p_holm_by"] >= g["p_wilcoxon"]).all()


def test_cardiac_trial_features_known_answers():
    sf = 100.0
    peaks = np.array([0, 100, 200, 300, 400, 500, 600, 700])  # 1.0 s beats
    peaks[5:] += 20  # IBI 4 -> 1.2 s (plausible), later beats shifted
    stim = np.array([150.0, 150.0, np.nan])
    resp = np.array([330.0, 330.0, 330.0])
    f = eeg_clean.cardiac_trial_features(peaks, stim, resp, sf)
    assert f["hb_ibi_base_s"][0] == pytest.approx(1.0)       # 0-100, before the stim IBI
    assert f["hb_ibi_resp0_s"][0] == pytest.approx(1.0)      # 300-400 contains 330
    assert f["hb_ibi_resp1_s"][0] == pytest.approx(1.2)      # 400-520
    assert f["hb_ibi_resp2_s"][0] == pytest.approx(1.0)      # 520-620
    assert f["hb_post_response_change_s"][0] == pytest.approx(0.1)
    assert f["hb_prev_rpeak_s"][0] == pytest.approx(0.3)
    assert f["hb_next_rpeak_s"][0] == pytest.approx(0.7)
    assert np.isnan(f["hb_ibi_base_s"][2]) and np.isnan(f["hb_post_response_change_s"][2])
    # A missed beat (2 s interval) makes every feature that uses it NaN.
    gap = np.array([0, 100, 200, 300, 500, 600, 700, 800])
    g = eeg_clean.cardiac_trial_features(gap, np.array([150.0]), np.array([330.0]), sf)
    assert np.isnan(g["hb_ibi_resp0_s"][0]) and np.isnan(g["hb_next_rpeak_s"][0])
    assert g["hb_ibi_base_s"][0] == pytest.approx(1.0)


def test_ica_removes_heartbeat_component_and_keeps_other_sources():
    pytest.importorskip("mne")
    rng = np.random.default_rng(3)
    fs, n, n_ch = 250.0, 40_000, 8
    peaks = 100 + np.cumsum(rng.integers(190, 211, 200))
    peaks = peaks[peaks < n - 300]
    art = np.zeros(n)
    for p in peaks:
        art[p - 30:p + 70] += np.hanning(100) * 30
    sources = eeg_clean.causal_bandpass(rng.laplace(size=(n_ch - 1, n)) * 5, fs, 1.0, 30.0)
    mixing = rng.standard_normal((n_ch, n_ch - 1))
    brain = mixing @ sources
    x = brain + rng.standard_normal(n_ch)[:, None] * art[None]
    out, info = eeg_clean.ica_remove_cardiac(x, peaks, fs, 0.8, n_components=n_ch - 1)
    inner = slice(2000, n - 2000)
    err_before = np.sqrt(np.mean((x - brain)[:, inner] ** 2))
    err_after = np.sqrt(np.mean((out - brain)[:, inner] ** 2))
    assert 1 <= info["n_excluded"] <= 2
    assert err_after < 0.25 * err_before
    assert np.corrcoef(out[:, inner].ravel(), brain[:, inner].ravel())[0, 1] > 0.95


def test_run_decoding_column_target_drops_missing_rows(tmp_path):
    rng = np.random.default_rng(2)
    subs = [f"sub-{i:02d}" for i in range(1, 7)]
    for s in subs:
        _write_fake_subject(tmp_path, s, rng, effect=1.0)
        t = pd.read_csv(tmp_path / f"{s}_trials.csv")
        t["hb_next_rpeak_s"] = np.where(np.arange(len(t)) % 3 == 0, np.nan, rng.random(len(t)))
        t.to_csv(tmp_path / f"{s}_trials.csv", index=False)
    cfg = {"epochs_dir": str(tmp_path), "subjects": subs, "conditions": ["active"],
           "windows": ["post_decision"], "score_mode": "per_group", "n_perm": 0,
           "loso": False, "pairwise_transfer": False, "n_jobs": 1,
           "targets": {"rpeak": {"task": "regression", "column": "hb_next_rpeak_s"},
                       "confidence": {"task": "regression"},
                       "confidence_same_trials": {"task": "regression", "target": "confidence",
                                                  "require_columns": ["hb_next_rpeak_s"]},
                       "confidence_beyond_rpeak": {"task": "regression", "target": "confidence",
                                                   "covariates": ["hb_next_rpeak_s"]}}}
    w = e.run_decoding(cfg)["within_subject"].set_index(["subject", "target"])
    n = w["n_trials"].xs("sub-01")
    assert n["confidence"] == 60
    assert n["rpeak"] == n["confidence_same_trials"] == n["confidence_beyond_rpeak"] == 40  # 20 of 60 NaN
    assert w["score"].xs("confidence_beyond_rpeak", level="target").mean() > 0.3


def test_cardiac_behavior_finds_planted_association_and_ignores_run_drift(tmp_path):
    rng = np.random.default_rng(4)
    subs = [f"sub-{i:02d}" for i in range(1, 11)]
    for s in subs:
        n = 160
        run = np.repeat([f"run-00{i}" for i in range(1, 5)], n // 4)
        drift = np.repeat(np.arange(4), n // 4) * 0.5  # shared run drift: not an effect
        conf = rng.random(n) + drift
        cond = np.tile(["active", "observed"], n // 2)
        change = rng.standard_normal(n) * 0.02 + drift * 0.05
        change += np.where(cond == "active", 0.03 * (conf - drift - 0.5), 0.0)
        pd.DataFrame({"run": run, "condition": cond, "valid_trial": True,
                      "eeg_has_response": True, "confidence_raw": conf,
                      "accuracy": rng.integers(0, 2, n),
                      "hb_post_response_change_s": change}).to_csv(
            tmp_path / f"{s}_trials.csv", index=False)
    res = e.cardiac_behavior(tmp_path, subs, measures=("hb_post_response_change_s",))
    g = res["group"].set_index(["condition", "behavior"])
    assert g.loc[("active", "confidence"), "p_holm"] < 0.01
    assert g.loc[("active", "confidence"), "mean"] > 0.2
    assert g.loc[("observed", "confidence"), "p_wilcoxon"] > 0.05  # drift alone: no effect
    assert abs(g.loc[("observed", "confidence"), "mean"]) < 0.1


def test_group_test_two_sided():
    neg = [-0.2, -0.1, -0.3, -0.15, -0.25, -0.05]
    assert eeg_decoding.group_test(neg, 0.0)["p_wilcoxon"] > 0.9
    assert eeg_decoding.group_test(neg, 0.0, alternative="two-sided")["p_wilcoxon"] < 0.05


def test_cli_several_bcg_methods_need_a_method_placeholder(tmp_path, monkeypatch):
    import sys

    from latent_decision_models import cli

    pd.DataFrame({"subject": []}).to_csv(tmp_path / "t.csv", index=False)
    monkeypatch.setattr(sys, "argv", [
        "ldm", "ds002158", "eeg-preprocess", "--bids-root", str(tmp_path), "--trials",
        str(tmp_path / "t.csv"), "--subjects", "sub-01", "--bcg-method", "aas", "obs",
        "--out-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="method"):
        cli.main()


def test_planted_signal_retention_known_answers():
    rng = np.random.default_rng(6)
    sf, n = 250.0, 30_000
    x = rng.standard_normal((4, n)).astype(np.float32)
    w = e.planted_waveforms(sf)["slow"]
    onsets = np.arange(3000, n - 1000, 700)
    pattern = np.array([1.0, -1.0, 0.5, -0.5])
    amps = rng.standard_normal(len(onsets))
    keep = eeg_clean.planted_signal_retention(x, lambda y: y, onsets, w, pattern, amps, sf)
    assert keep["gain"] == pytest.approx(1.0, abs=1e-4)
    assert keep["trial_r"] == pytest.approx(1.0, abs=1e-4)
    drop = eeg_clean.planted_signal_retention(x, lambda y: y * 0, onsets, w, pattern, amps, sf)
    assert drop["gain"] == pytest.approx(0.0, abs=1e-4)
    half = eeg_clean.planted_signal_retention(x, lambda y: 0.5 * y, onsets, w, pattern, amps, sf)
    assert half["gain"] == pytest.approx(0.5, abs=1e-4)
    assert half["trial_slope"] == pytest.approx(0.5, abs=1e-4)
    # Template subtraction locked to the planted events removes their mean, not their
    # trial-to-trial variation.
    aas = eeg_clean.planted_signal_retention(
        x, lambda y: eeg_clean.template_subtract(y, onsets, 0, len(w), 21)[0],
        onsets, w, pattern, amps + 3.0, sf)
    assert aas["trial_r"] > 0.95 and aas["gain"] < 0.5


def test_planted_waveforms_sit_in_the_post_decision_window():
    sf = 250.0
    for w in e.planted_waveforms(sf).values():
        t = np.arange(len(w)) / sf
        assert np.all(w[(t < 0.1) | (t >= 0.6)] == 0) and np.abs(w).max() > 0.9


def test_cross_condition_decoding_and_contrast(tmp_path):
    # Confidence is planted in both conditions in half of the subjects ("shared") and only
    # in active trials in the other half: cross-condition decoding generalises only where
    # the signal is shared, and the paired contrast sees active > observed.
    rng = np.random.default_rng(7)
    subs = [f"sub-{i:02d}" for i in range(1, 9)]
    for s in subs:
        _write_fake_subject(tmp_path, s, rng, effect=1.0)
    only_active = tmp_path / "only_active"
    only_active.mkdir()
    for s in subs:
        t = pd.read_csv(tmp_path / f"{s}_trials.csv")
        t.to_csv(only_active / f"{s}_trials.csv", index=False)
        z = dict(np.load(tmp_path / f"{s}_epochs.npz"))
        obs = (t["condition"] == "observed").to_numpy()
        z["resp"][obs, 0] = rng.standard_normal(z["resp"][obs, 0].shape)
        np.savez_compressed(only_active / f"{s}_epochs.npz", **z)
    cfg = {"subjects": subs, "confirmatory_subjects": subs, "min_runs": 3, "min_trials": 40,
           "conditions": ["active", "observed"], "windows": ["post_decision"],
           "score_mode": "per_group", "n_perm": 0, "loso": False, "pairwise_transfer": False,
           "n_jobs": 1, "cross_condition": [["active", "observed"], ["observed", "active"]],
           "condition_contrasts": [["active", "observed"]],
           "targets": {"confidence": {"task": "regression", "primary": True}}}
    shared = e.run_decoding({**cfg, "epochs_dir": str(tmp_path)})
    g = shared["group"].set_index(["analysis", "condition"])
    assert g.loc[("within_subject_cross_condition", "active->observed"), "p_wilcoxon"] < 0.01
    assert g.loc[("within_subject_cross_condition", "observed->active"), "p_wilcoxon"] < 0.01
    assert shared["cross_condition"]["n_trials"].min() == 60
    assert shared["cross_subject"].empty  # pairwise transfer skipped
    specific = e.run_decoding({**cfg, "epochs_dir": str(only_active)})
    g = specific["group"].set_index(["analysis", "condition"])
    assert g.loc[("within_subject", "active"), "p_wilcoxon"] < 0.01
    assert g.loc[("within_subject_cross_condition", "active->observed"), "p_wilcoxon"] > 0.05
    assert g.loc[("within_subject_contrast", "active-observed"), "mean"] > 0.3
    assert g.loc[("within_subject_contrast", "active-observed"), "p_wilcoxon"] < 0.05


def test_decode_reports_no_p_value_when_the_score_is_undefined():
    # Every held-out run contains one class only: no AUC, so no p-value (it used to be
    # 1/(1 + n_perm), i.e. "significant").
    X = np.random.default_rng(0).standard_normal((20, 3))
    y = np.array([0] * 10 + [1] * 10)
    r = eeg_decoding.decode(X, y, np.array([0] * 10 + [1] * 10), "classification", n_perm=20)
    assert np.isnan(r["score"]) and r["p_value"] is None


def test_permutation_p_counts_failed_permutations_conservatively():
    assert eeg_decoding.permutation_p(0.5, [0.1, 0.2, np.nan]) == pytest.approx(2 / 4)
    assert eeg_decoding.permutation_p(float("nan"), [0.1]) is None


def test_holm_keeps_nan_and_excludes_it_from_the_family():
    adj = eeg_decoding.holm([0.01, np.nan, 0.04])
    assert np.isnan(adj[1])
    assert adj[0] == pytest.approx(0.02) and adj[2] == pytest.approx(0.04)


def test_cross_decode_skips_a_run_without_training_trials_even_with_covariates():
    rng = np.random.default_rng(1)
    Xa, ya = rng.standard_normal((30, 4)), rng.standard_normal(30)
    Xb, yb = rng.standard_normal((40, 4)), rng.standard_normal(40)
    ga = np.zeros(30, int)                  # condition A recorded in run 0 only
    gb = np.repeat([0, 1], 20)
    r = eeg_decoding.cross_decode(Xa, ya, ga, Xb, yb, gb, "regression",
                                  rng.standard_normal((30, 1)), rng.standard_normal((40, 1)))
    assert r["n_groups"] == 1 and r["n_trials"] == 20  # run 0 skipped, run 1 scored
