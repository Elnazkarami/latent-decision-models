"""ds002158 fMRI metadata audit and design diagnostics (no network, no images)."""

import gzip

import numpy as np
import pandas as pd
import pytest

from latent_decision_models.datasets import ds002158_fmri as f
from latent_decision_models.io import openneuro


def _entry(name, size=1):
    return {"filename": name, "size": size, "urls": ["https://example.invalid/x"]}


def test_inventory_pairs_raw_and_derivative():
    d = f.DERIV
    files = [
        _entry("sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_bold.nii.gz"),
        _entry("sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_events.tsv"),
        _entry(d + "sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_preproc.nii"),
        _entry(d + "sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_headmove.txt"),
        _entry(d + "sub-19/ses-001/func/sub-19_ses-001_task-main_run-006_preproc.nii"),
        _entry(d + "sub-20/ses-001/func/reject/sub-20_ses-001_task-main_run-007_preproc.nii"),
        _entry("sub-20/ses-001/func/sub-20_ses-001_task-main_run-007_bold.nii.gz"),
        _entry("sub-02/ses-001/anat/sub-02_ses-001_T1w.nii.gz"),
    ]
    inv = f.inventory(files).set_index(["subject", "run"])
    assert len(inv) == 3
    assert inv.loc[("sub-02", "run-001"), "derivative_headmove"].endswith("headmove.txt")
    assert pd.isna(inv.loc[("sub-19", "run-006"), "raw_bold"])
    assert inv.loc[("sub-20", "run-007"), "derivative_folder"] == "reject"


def test_motion_and_framewise_displacement(tmp_path):
    p = np.zeros((5, 6))
    p[2, 0] = 1.0                    # 1 mm jump in x at volume 2
    p[4, 3] = 0.01                   # 0.01 rad rotation -> 0.5 mm at 50 mm radius
    path = tmp_path / "rp.txt"
    np.savetxt(path, p)
    x = f.read_motion(path)
    np.testing.assert_allclose(f.framewise_displacement(x), [0, 0, 1, 1, 0.5])
    s = f.motion_summary(x, fd_threshold=0.75)
    assert s["n_volumes"] == 5 and s["fraction_fd_above_threshold"] == pytest.approx(0.4)
    np.savetxt(path, np.ones((3, 6)))  # rotations of 1 rad are not plausible
    with pytest.raises(ValueError, match="radians"):
        f.read_motion(path)


def test_fetch_range_is_capped():
    with pytest.raises(PermissionError, match="cap"):
        openneuro.fetch_range(_entry("x.nii"), openneuro.MAX_RANGE_BYTES + 1)


def _header_bytes(gz=False):
    nib = pytest.importorskip("nibabel")
    img = nib.Nifti1Image(np.zeros((4, 5, 6, 7), np.float32), np.diag([1.5, 1.5, 1.5, 1]))
    img.header.set_xyzt_units("mm", "sec")
    img.header["descrip"] = b"Smoothed (5x5x5) Dartel normed"
    img.header.set_zooms((1.5, 1.5, 1.5, 0.0))
    img.header.set_sform(img.affine, code=4)
    raw = img.header.binaryblock + b"\x00" * 4 + np.asarray(img.dataobj).tobytes()
    return gzip.compress(raw) if gz else raw[:1024]


@pytest.mark.parametrize("gz", [False, True])
def test_parse_header_from_prefix(gz):
    from latent_decision_models.io import nifti

    h = nifti.parse_header(_header_bytes(gz))
    assert h["shape"] == [4, 5, 6, 7] and h["n_volumes"] == 7
    assert h["voxel_size_mm"] == [1.5, 1.5, 1.5] and h["repetition_time"] == 0.0
    assert h["sform_code"] == "mni_152" and h["descrip"].startswith("Smoothed (5x5x5)")
    with pytest.raises(ValueError):
        nifti.parse_header(_header_bytes()[:100])


def test_design_diagnostics_known_answers():
    pytest.importorskip("nilearn")
    from latent_decision_models import fmri_design as fd

    onsets = np.arange(10.0, 400.0, 30.0)  # widely spaced trials
    cond = np.where(np.arange(len(onsets)) % 2, "a", "b")
    r = fd.run_diagnostics(onsets, cond, 340, 1.28, response_offsets=np.full(len(onsets), 0.4),
                           post_offsets=np.full(len(onsets), np.nan))
    assert r["lsa_vif_max"] < 1.2 and r["lss_vif_max"] < 1.2
    assert r["adjacent_trial_abs_r_max"] < 0.1
    assert r["r_stimulus_vs_response"] > 0.95 and r["vif_stimulus"] > 10
    assert r["vif_stimulus_with_time_derivative"] > r["vif_stimulus"]
    # jittered post-decision events stay estimable
    rng = np.random.default_rng(0)
    r2 = fd.run_diagnostics(onsets, cond, 340, 1.28,
                            post_offsets=rng.uniform(4, 10, len(onsets)))
    assert r2["vif_post"] < 3
