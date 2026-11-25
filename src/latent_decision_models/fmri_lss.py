"""Single-trial fMRI estimates (LSS) and ROI time series. Dataset-agnostic.

* :func:`lss_weights` builds, for every trial, the least-squares-separate design of
  Mumford et al. (2012): the target trial, the other trials of each condition, optional
  confounds, cosine drift and an intercept. It returns the row of the pseudo-inverse that
  gives the target beta, so all single-trial betas of a run are ``W @ Y`` for any data
  ``Y`` (time x signals). Because this is linear, the beta of an ROI's mean time series
  equals the ROI mean of its voxel betas.
* :func:`roi_timeseries` averages voxels per atlas label, skipping voxels that are not
  finite or are constant in the run (e.g. outside the normalised field of view).

Design matrices come from nilearn (optional ``[fmri]`` extra), as in :mod:`fmri_design`.
Onsets are in seconds from the first acquired volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fmri_design import _design


def lss_weights(onsets, conditions, n_volumes: int, tr: float, confounds=None,
                hrf_model: str = "glover", high_pass: float = 1 / 128):
    """Rows of the LSS pseudo-inverses, one per trial: ``betas = W @ Y``.

    Returns ``(W, condition_numbers)``; ``W`` has shape (n_trials, n_volumes) and the
    condition numbers are of each design with unit-norm columns.
    ``confounds`` (n_volumes x k) are demeaned and added to every design.
    """
    onsets = np.asarray(onsets, float)
    conditions = np.asarray(conditions).astype(str)
    if not np.isfinite(onsets).all():
        raise ValueError("all trial onsets must be finite")
    frame_times = np.arange(n_volumes) * tr
    conf = None
    if confounds is not None:
        conf = np.asarray(confounds, float)
        if conf.shape[0] != n_volumes:
            raise ValueError(f"confounds have {conf.shape[0]} rows, expected {n_volumes}")
        conf = conf - conf.mean(axis=0)
    n = len(onsets)
    W = np.empty((n, n_volumes))
    cond_numbers = np.empty(n)
    for i in range(n):
        labels = np.where(np.arange(n) == i, "target", "other_" + conditions)
        X = _design(frame_times, pd.DataFrame({"onset": onsets, "duration": 0.0,
                                               "trial_type": labels}), hrf_model, high_pass)
        A = X.to_numpy(float)
        if conf is not None:
            A = np.column_stack([A, conf])
        # Collinearity, not column scale (stick regressors peak near 0.01, rotations are
        # in radians): condition number of the design with unit-norm columns.
        cond_numbers[i] = np.linalg.cond(A / np.linalg.norm(A, axis=0))
        W[i] = np.linalg.pinv(A)[X.columns.get_loc("target")]
    return W, cond_numbers


def roi_timeseries(data: np.ndarray, labels: np.ndarray, label_ids) -> tuple[np.ndarray, dict]:
    """Mean time series per label.

    ``data``: (x, y, z, t) array; ``labels``: (x, y, z) integer array on the same grid.
    Returns ``(Y, n_voxels)`` with ``Y`` of shape (t, len(label_ids)); a label with no
    usable voxel gives a NaN column.
    """
    if data.shape[:3] != labels.shape:
        raise ValueError(f"data grid {data.shape[:3]} != label grid {labels.shape}")
    label_ids = list(label_ids)
    flat_labels = labels.reshape(-1, order="F")
    in_atlas = np.isin(flat_labels, label_ids)
    # Fortran-ordered reshape is a view for NIfTI arrays (x varies fastest).
    vox = data.reshape(-1, data.shape[3], order="F")[in_atlas]  # input dtype (memory)
    lab = flat_labels[in_atlas]
    usable = np.isfinite(vox).all(axis=1) & (vox.std(axis=1, dtype=np.float64) > 0)
    Y = np.full((data.shape[3], len(label_ids)), np.nan)
    n_voxels = {}
    for j, k in enumerate(label_ids):
        m = (lab == k) & usable
        n_voxels[k] = int(m.sum())
        if m.any():
            Y[:, j] = vox[m].mean(axis=0, dtype=np.float64)
    return Y, n_voxels


def percent_signal(Y: np.ndarray) -> np.ndarray:
    """Scale each column to percent of its mean (so betas are in % signal change)."""
    mean = np.nanmean(Y, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (Y / mean - 1.0)


def tsnr(Y: np.ndarray) -> np.ndarray:
    """Temporal SNR per column: mean / standard deviation over time (no detrending)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nanmean(Y, axis=0) / np.nanstd(Y, axis=0)


def onset_shift_scan(y, onsets, n_volumes: int, tr: float, confounds=None,
                     shifts=None, hrf_model: str = "glover",
                     high_pass: float = 1 / 128) -> pd.DataFrame:
    """Fit one all-trials regressor with onsets shifted by each value in ``shifts`` (s).

    A timing check: the shift with the best positive fit estimates how much later (> 0)
    or earlier the response is than the canonical HRF at the recorded onsets predicts.
    With near-periodic trials the profile repeats every trial period, so only shifts
    well inside +/- half a period are interpretable. Returns one row per shift with the
    beta and the R^2 gained over the nuisance-only model (drift, intercept, confounds).
    """
    shifts = np.arange(-6.0, 6.01, 0.5) if shifts is None else shifts
    from nilearn.glm.first_level import make_first_level_design_matrix

    y = np.asarray(y, float)
    onsets = np.asarray(onsets, float)
    frame_times = np.arange(n_volumes) * tr
    N = make_first_level_design_matrix(frame_times, None, drift_model="cosine",
                                       high_pass=high_pass).to_numpy()
    if confounds is not None:
        c = np.asarray(confounds, float)
        N = np.column_stack([N, c - c.mean(axis=0)])
    res0 = y - N @ np.linalg.lstsq(N, y, rcond=None)[0]
    rows = []
    for s in shifts:
        X = _design(frame_times, pd.DataFrame({"onset": onsets + s, "duration": 0.0,
                                               "trial_type": "trial"}), hrf_model, high_pass)
        A = np.column_stack([X["trial"].to_numpy(), N])
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        rows.append({"shift_s": float(s), "beta": float(b[0]),
                     "r2_gain": float(1 - (y - A @ b).var() / res0.var())})
    return pd.DataFrame(rows)
