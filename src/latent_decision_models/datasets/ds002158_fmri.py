"""ds002158 fMRI: raw/derivative inventory, header audit and motion summary (no image data).

Status: experimental. Facts verified in ``docs/ds002158-fmri-audit.md``:

* ``derivatives/fmriprep`` is **not** fMRIPrep output. Its NIfTI ``descrip`` field reads
  ``"Smoothed (5x5x5) Dartel normed"``: SPM DARTEL normalisation to MNI-152 (qform/sform
  code 4), 1.5 mm isotropic, 121 x 145 x 121 grid, 5 mm FWHM smoothing.
* The derivative headers store TR = 0; the TR (1.28 s) comes from the raw data.
* Motion is only available as ``*_headmove.txt`` (realignment parameters).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TR_SECONDS = 1.28  # raw bold.json and raw NIfTI headers
DERIV = "derivatives/fmriprep/"
_RUN = re.compile(r"sub-(?P<sub>\d+)/ses-001/func/(?:(?P<folder>reject|rejected)/)?"
                  r"sub-\d+_ses-001_task-main_run-(?P<run>\d+)_(?P<kind>bold\.nii\.gz|preproc\.nii|"
                  r"headmove\.txt|events\.tsv)$")


def inventory(files: list[dict]) -> pd.DataFrame:
    """One row per (subject, run) with which raw and derivative files exist."""
    rows: dict[tuple, dict] = {}
    for f in files:
        name = f["filename"]
        source = "derivative" if name.startswith(DERIV) else "raw"
        m = _RUN.search(name)
        if not m or (source == "raw" and name.startswith("derivatives/")):
            continue
        key = (f"sub-{m['sub']}", f"run-{m['run']}")
        row = rows.setdefault(key, {"subject": key[0], "run": key[1]})
        kind = m["kind"].split(".")[0]
        row[f"{source}_{kind}"] = name
        row[f"{source}_{kind}_bytes"] = f["size"]
        if m["folder"] and kind == "preproc":
            row["derivative_folder"] = m["folder"]
    out = pd.DataFrame(rows.values()).sort_values(["subject", "run"]).reset_index(drop=True)
    for c in ("raw_bold", "raw_events", "derivative_preproc", "derivative_headmove",
              "derivative_events", "derivative_folder"):
        if c not in out:
            out[c] = None
    return out


def header_audit(inv: pd.DataFrame, files: list[dict], fetch, parse) -> pd.DataFrame:
    """Parse the header of every raw ``bold.nii.gz`` and derivative ``preproc.nii``.

    ``fetch(entry, n_bytes) -> bytes`` and ``parse(bytes) -> dict`` are injected
    (normally :func:`io.openneuro.fetch_range` and :func:`io.nifti.parse_header`).
    """
    by_name = {f["filename"]: f for f in files}
    rows = []
    for _, r in inv.iterrows():
        for col, n_bytes in (("raw_bold", 16_384), ("derivative_preproc", 4_096)):
            name = r[col]
            if not isinstance(name, str):
                continue
            h = parse(fetch(by_name[name], n_bytes))
            rows.append({"subject": r["subject"], "run": r["run"], "file": col,
                         **{k: h[k] for k in ("shape", "n_volumes", "voxel_size_mm",
                                              "repetition_time", "datatype", "qform_code",
                                              "sform_code", "descrip", "affine")}})
    return pd.DataFrame(rows)


def read_motion(path: str | Path) -> np.ndarray:
    """Realignment parameters: 6 columns (3 translations in mm, 3 rotations in rad)."""
    x = np.loadtxt(path)
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(f"{path}: expected 6 columns, got shape {x.shape}")
    if np.abs(x[:, 3:]).max() > 0.5:  # > 28 degrees: not radians
        raise ValueError(f"{path}: rotation columns do not look like radians")
    return x


def framewise_displacement(params: np.ndarray, radius_mm: float = 50.0) -> np.ndarray:
    """Power et al. (2012) FD: sum of absolute backward differences, rotations as arc length."""
    d = np.abs(np.diff(params, axis=0))
    d[:, 3:] *= radius_mm
    return np.concatenate([[0.0], d.sum(axis=1)])


def motion_summary(params: np.ndarray, fd_threshold: float = 0.5) -> dict:
    fd = framewise_displacement(params)
    return {"n_volumes": len(params),
            "max_abs_translation_mm": float(np.abs(params[:, :3]).max()),
            "max_abs_rotation_deg": float(np.degrees(np.abs(params[:, 3:]).max())),
            "mean_fd_mm": float(fd.mean()), "max_fd_mm": float(fd.max()),
            "fraction_fd_above_threshold": float((fd > fd_threshold).mean())}
