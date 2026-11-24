"""LSS single-trial estimates, ROI averaging and the ds002158 fMRI pilot (synthetic data)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from latent_decision_models import fmri_lss
from latent_decision_models.datasets import ds002158_fmri as f

TR = 1.28


def _schedule(n_trials=24, gap=9.0, start=6.0):
    onsets = start + gap * np.arange(n_trials)
    conditions = np.where((np.arange(n_trials) // 6) % 2 == 0, "active", "observed")
    n_vol = int((onsets[-1] + 25) / TR)
    return onsets, conditions, n_vol


def _signal(onsets, amplitudes, n_vol):
    """Same HRF model as the estimator: sum of amplitude-weighted trial regressors."""
    from latent_decision_models.fmri_design import _design

    names = [f"t{i:03d}" for i in range(len(onsets))]
    X = _design(np.arange(n_vol) * TR, pd.DataFrame({"onset": onsets, "duration": 0.0,
                                                     "trial_type": names}), "glover", 1 / 128)
    return X[names].to_numpy() @ amplitudes


def test_lss_is_exact_for_equal_amplitudes_with_nuisance_in_the_design():
    pytest.importorskip("nilearn")
    rng = np.random.default_rng(0)
    onsets, cond, n_vol = _schedule()
    motion = rng.normal(size=(n_vol, 6))
    y = 100 + _signal(onsets, np.full(len(onsets), 0.7), n_vol) + motion @ rng.normal(size=6)
    W, cn = fmri_lss.lss_weights(onsets, cond, n_vol, TR, confounds=motion)
    assert W.shape == (len(onsets), n_vol) and np.isfinite(cn).all()
    assert cn.max() < 100  # trials 9 s apart: well conditioned once columns are scaled
    np.testing.assert_allclose(W @ y, 0.7, atol=1e-8)
    # Linearity: weights applied to two columns give two sets of betas.
    B = W @ np.column_stack([y, 2 * y])
    np.testing.assert_allclose(B[:, 1], 2 * B[:, 0])


def test_lss_tracks_varying_single_trial_amplitudes():
    """LSS models the other trials of a condition with one shared amplitude, so varying
    amplitudes are recovered closely but not exactly (Mumford et al. 2012)."""
    pytest.importorskip("nilearn")
    rng = np.random.default_rng(0)
    onsets, cond, n_vol = _schedule()
    amp = rng.normal(1.0, 0.5, len(onsets))
    W, _ = fmri_lss.lss_weights(onsets, cond, n_vol, TR)
    assert np.corrcoef(W @ _signal(onsets, amp, n_vol), amp)[0, 1] > 0.85


def test_lss_rejects_bad_inputs():
    pytest.importorskip("nilearn")
    onsets, cond, n_vol = _schedule(6)
    with pytest.raises(ValueError, match="finite"):
        fmri_lss.lss_weights(np.r_[onsets[:-1], np.nan], cond, n_vol, TR)
    with pytest.raises(ValueError, match="confounds"):
        fmri_lss.lss_weights(onsets, cond, n_vol, TR, confounds=np.zeros((n_vol - 1, 6)))


def test_roi_timeseries_skips_unusable_voxels():
    rng = np.random.default_rng(1)
    data = rng.normal(10, 1, size=(4, 3, 2, 20)).astype(np.float32)
    labels = np.zeros((4, 3, 2), int)
    labels[:2] = 1
    labels[2:] = 2
    labels[3, 2, 1] = 3          # a label whose only voxel is constant
    data[3, 2, 1] = 5.0
    data[0, 0, 0, 4] = np.nan    # one non-finite voxel in label 1
    Y, n = fmri_lss.roi_timeseries(data, labels, [1, 2, 3])
    assert n == {1: 11, 2: 11, 3: 0}
    m1 = (labels == 1)
    m1[0, 0, 0] = False
    np.testing.assert_allclose(Y[:, 0], data[m1].mean(axis=0), rtol=1e-5)
    assert np.isnan(Y[:, 2]).all()
    with pytest.raises(ValueError, match="grid"):
        fmri_lss.roi_timeseries(data, labels[:3], [1])


def test_percent_signal_and_tsnr():
    Y = np.array([[99.0, 1.0], [101.0, 3.0]])
    np.testing.assert_allclose(fmri_lss.percent_signal(Y), [[-1, -50], [1, 50]])
    np.testing.assert_allclose(fmri_lss.tsnr(Y), [100.0, 2.0])


def _pilot_run(tmp_path, subject="sub-02", run="run-001", seed=0, effects=(1.0, 0.6)):
    """A tiny derivative run: region 1 'visual' responds to every trial, region 2
    'motor' to active trials only. Written in the dataset's folder layout."""
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("nilearn")
    rng = np.random.default_rng(seed)
    onsets, cond, n_vol = _schedule()
    vis = _signal(onsets, np.full(len(onsets), effects[0]), n_vol)
    mot = _signal(onsets, np.where(cond == "active", effects[1], 0.0), n_vol)
    shape = (4, 4, 2)
    labels = np.zeros(shape, int)
    labels[:2] = 1
    labels[2:] = 2
    # nilearn stick regressors peak near 0.009, so the noise must be small for a
    # 1%-per-unit response to be recoverable.
    data = 1000 * (1 + rng.normal(0, 1e-6, shape + (n_vol,)))
    data[labels == 1] += 10 * vis
    data[labels == 2] += 10 * mot
    out = f.preproc_path(tmp_path, subject, run)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), out)
    np.savetxt(out.with_name(f"{subject}_ses-001_task-main_{run}_headmove.txt"),
               rng.normal(0, 1e-3, (n_vol, 6)))
    trials = pd.DataFrame({"subject": subject, "run": run, "trial_index": np.arange(len(onsets)),
                           "condition": cond, "onset_seconds": onsets, "valid_trial": True})
    names = {1: "Left Visual", 2: "Left Motor"}
    return trials, (labels, names)


def test_run_lss_end_to_end_on_a_synthetic_derivative(tmp_path):
    trials, atlas = _pilot_run(tmp_path)
    betas, qc = f.run_lss(tmp_path, "sub-02", "run-001", trials, atlas=atlas)
    assert list(betas.columns[-2:]) == ["roi_Left Visual", "roi_Left Motor"]
    # Percent signal: 10 units on a mean of ~1000 -> ~1% per unit regressor amplitude.
    np.testing.assert_allclose(betas["roi_Left Visual"], 1.0, atol=0.15)
    active = betas["condition"] == "active"
    assert betas.loc[active, "roi_Left Motor"].mean() > 0.4
    assert abs(betas.loc[~active, "roi_Left Motor"].mean()) < 0.1
    assert qc["n_trials"] == 24 and qc["n_rois_empty"] == 0
    assert qc["lss_condition_number_max"] > 1


def test_run_lss_rejects_motion_volume_mismatch(tmp_path):
    trials, atlas = _pilot_run(tmp_path)
    p = f.preproc_path(tmp_path, "sub-02", "run-001").with_name(
        "sub-02_ses-001_task-main_run-001_headmove.txt")
    np.savetxt(p, np.loadtxt(p)[:-1])
    with pytest.raises(ValueError, match="motion rows"):
        f.run_lss(tmp_path, "sub-02", "run-001", trials, atlas=atlas)


def _names_with_sets():
    names, k = {}, 1
    for s in ("occipital", "sensorimotor"):
        for region in f.ROI_SETS[s]:
            for side in ("Left", "Right"):
                names[k] = f"{side} {region}"
                k += 1
    names[k] = "Left Frontal Pole"
    return names


def test_positive_controls_detect_planted_effects_and_ignore_invalid_trials():
    rng = np.random.default_rng(2)
    names = _names_with_sets()
    occ = f.roi_set_columns(names, "occipital")
    sm = f.roi_set_columns(names, "sensorimotor")
    rows = []
    for run in range(6):
        for i in range(48):
            rows.append({"subject": "sub-02", "run": f"run-00{run + 1}", "trial_index": i,
                         "condition": "active" if (i // 12) % 2 == 0 else "observed"})
    t = pd.DataFrame(rows)
    t["valid_trial"] = t["trial_index"] != 0
    b = t[["subject", "run", "trial_index", "condition"]].copy()
    for c in [f"roi_{n}" for n in names.values()]:
        b[c] = rng.normal(0, 0.3, len(b))
    b[occ] += 0.5
    b.loc[b["condition"] == "active", sm] += 0.5
    b.loc[t["trial_index"] == 0, occ] = -1e6   # invalid trials must not count
    res = f.positive_controls(b, t, names)["sub-02"]
    assert res["visual"]["n_runs"] == 6 and res["visual"]["p_one_sided"] < 0.001
    assert res["movement"]["p_one_sided"] < 0.001 and 0.3 < res["movement"]["mean"] < 0.7
    b[occ] -= 1.0                                 # now visual betas are negative
    b.loc[t["trial_index"] == 0, occ] = -1e6
    assert f.positive_controls(b, t, names)["sub-02"]["visual"]["p_one_sided"] > 0.9


def test_roi_set_columns_requires_every_region():
    names = _names_with_sets()
    assert len(f.roi_set_columns(names, "sensorimotor")) == 4
    names = {k: v for k, v in names.items() if v != "Right Postcentral Gyrus"}
    with pytest.raises(ValueError, match="sensorimotor"):
        f.roi_set_columns(names, "sensorimotor")


def test_decode_pilot_finds_a_planted_confidence_signal():
    rng = np.random.default_rng(3)
    n_run, n = 4, 40
    t = pd.DataFrame({"subject": "sub-02", "run": np.repeat([f"run-00{r}" for r in range(1, 5)], n),
                      "trial_index": np.tile(np.arange(n), n_run), "condition": "active",
                      "valid_trial": True})
    t["confidence_raw"] = rng.uniform(0, 1, len(t))
    t["accuracy"] = rng.integers(0, 2, len(t))
    t["response_latency_seconds"] = rng.uniform(0.3, 0.5, len(t))
    t["evidence_dots_difference"] = rng.normal(size=len(t))
    b = t[["subject", "run", "trial_index", "condition"]].copy()
    X = rng.normal(size=(len(t), 5))
    X[:, 0] += 2 * t["confidence_raw"]
    for j in range(5):
        b[f"roi_r{j}"] = X[:, j]
    cfg = {"conditions": ["active"], "n_perm": 50, "seed": 0,
           "targets": {"confidence": {"task": "regression"}}}
    res = f.decode_pilot(b, t, cfg)
    assert len(res) == 1 and res.loc[0, "n_groups"] == 4
    assert res.loc[0, "score"] > 0.3 and res.loc[0, "p_value"] < 0.05


def test_harvard_oxford_merge_keeps_cortex_and_selected_subcortex(monkeypatch):
    """No download: nilearn's fetcher is replaced by two tiny atlases."""
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("nilearn")
    from types import SimpleNamespace

    from nilearn import datasets

    cort = np.zeros((4, 4, 4), np.int16)
    cort[0] = 1
    cort[1, :2] = 2
    sub = np.zeros((4, 4, 4), np.int16)
    sub[1] = 2          # 'Left Cerebral Cortex': must not be used
    sub[2] = 4          # 'Left Thalamus'
    sub[3, 0] = 1       # 'Left Cerebral White Matter': dropped
    sub[0, 0, 0] = 4    # overlaps cortex: cortex wins

    def fake(name, symmetric_split=False):
        if name.startswith("cort"):
            return SimpleNamespace(maps=nib.Nifti1Image(cort, np.eye(4)),
                                   labels=["Background", "Left Frontal Pole",
                                           "Right Frontal Pole"])
        return SimpleNamespace(maps=nib.Nifti1Image(sub, np.eye(4)),
                               labels=["Background", "Left Cerebral White Matter",
                                       "Left Cerebral Cortex", "Left Lateral Ventricle",
                                       "Left Thalamus"])

    monkeypatch.setattr(datasets, "fetch_atlas_harvard_oxford", fake)
    labels, names = f.harvard_oxford_atlas(np.eye(4), (4, 4, 4, 10))
    assert names == {1: "Left Frontal Pole", 2: "Right Frontal Pole", 3: "Left Thalamus"}
    assert (labels[0] == 1).all() and (labels[1, :2] == 2).all()
    assert (labels[1, 2:] == 0).all()            # subcortical 'cortex' label not used
    assert (labels[2] == 3).all() and (labels[3] == 0).all()


def test_cli_fmri_lss_and_pilot_end_to_end(tmp_path, monkeypatch, capsys):
    """Three synthetic runs: visual ROIs respond to all trials, sensorimotor ROIs to
    active trials; both controls pass and decoding runs. The atlas is replaced (no
    download) by one region per voxel row."""
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("nilearn")
    import json

    import yaml

    from latent_decision_models import cli

    names = _names_with_sets()
    occ_ids = [k for k, v in names.items() if any(r in v for r in f.ROI_SETS["occipital"])]
    sm_ids = [k for k, v in names.items() if any(r in v for r in f.ROI_SETS["sensorimotor"])]
    labels = np.repeat(np.array(list(names))[:, None, None], 2, axis=1)  # (R, 2, 1)
    rng = np.random.default_rng(4)
    onsets, cond, n_vol = _schedule()
    vis = _signal(onsets, np.ones(len(onsets)), n_vol)
    mot = _signal(onsets, (cond == "active").astype(float), n_vol)
    rows = []
    for r in (1, 2, 3):
        run = f"run-00{r}"
        data = 1000 * (1 + rng.normal(0, 1e-6, labels.shape + (n_vol,)))
        data[np.isin(labels, occ_ids)] += 10 * vis
        data[np.isin(labels, sm_ids)] += 10 * mot
        p = f.preproc_path(tmp_path / "bids", "sub-02", run)
        p.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), p)
        np.savetxt(p.with_name(f"sub-02_ses-001_task-main_{run}_headmove.txt"),
                   rng.normal(0, 1e-3, (n_vol, 6)))
        rows.append(pd.DataFrame({
            "subject": "sub-02", "run": run, "trial_index": np.arange(len(onsets)),
            "condition": cond, "onset_seconds": onsets, "valid_trial": True,
            "confidence_raw": rng.uniform(size=len(onsets)),
            "accuracy": rng.integers(0, 2, len(onsets)),
            "response_latency_seconds": rng.uniform(0.3, 0.5, len(onsets)),
            "evidence_dots_difference": rng.normal(size=len(onsets))}))
    pd.concat(rows).to_csv(tmp_path / "trials.csv", index=False)
    cfg = yaml.safe_load(Path("configs/ds002158_fmri_pilot.yaml").read_text())
    cfg.update(input_trials=str(tmp_path / "trials.csv"), subjects=["sub-02"])
    cfg["decoding"]["n_perm"] = 5
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(f, "harvard_oxford_atlas", lambda affine, shape: (labels, names))

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["ldm", "ds002158", *argv])
        cli.main()

    run("fmri-lss", "--config", str(tmp_path / "cfg.yaml"), "--bids-root",
        str(tmp_path / "bids"), "--out-dir", str(tmp_path / "out"),
        "--report", str(tmp_path / "lss.json"))
    qc = json.loads((tmp_path / "lss.json").read_text())
    assert qc["n_runs"] == 3
    b = pd.read_csv(tmp_path / "out" / "sub-02_lss_roi_betas.csv")
    assert len(b) == 3 * len(onsets)
    run("fmri-pilot", "--config", str(tmp_path / "cfg.yaml"), "--betas-dir",
        str(tmp_path / "out"), "--out", str(tmp_path / "pilot.json"))
    res = json.loads((tmp_path / "pilot.json").read_text())
    assert res["controls_passed"] is True
    assert res["positive_controls"]["sub-02"]["movement"]["mean"] > 0.5
    assert len(res["decoding"]) == 4  # 2 conditions x 2 targets


def test_fetch_fmri_selects_derivative_bold_and_motion_only(monkeypatch, capsys):
    import json

    from latent_decision_models import cli
    from latent_decision_models.io import openneuro

    d = f.DERIV + "sub-20/ses-001/func/"
    files = [{"filename": n, "size": 10, "urls": ["https://example.invalid/x"]} for n in (
        d + "sub-20_ses-001_task-main_run-001_preproc.nii",
        d + "sub-20_ses-001_task-main_run-001_headmove.txt",
        d + "sub-20_ses-001_task-main_run-001_events.tsv",
        d + "reject/sub-20_ses-001_task-main_run-007_preproc.nii",
        "sub-20/ses-001/func/sub-20_ses-001_task-main_run-001_bold.nii.gz",
        f.DERIV + "sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_preproc.nii")]
    monkeypatch.setattr(openneuro, "snapshot_files", lambda *a, **k: files)
    monkeypatch.setattr("sys.argv", ["ldm", "ds002158", "fetch-fmri", "--subjects", "sub-20",
                                     "--dest", "unused", "--approved-gb", "1", "--dry-run"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"subjects": ["sub-20"], "n_files": 2,
                                                   "total_bytes": 20}
