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


# --------------------------------------------------------------------------------------
# Trial-level pilot (LDM-004): LSS betas on atlas ROIs from the published derivative.
# --------------------------------------------------------------------------------------

SUBCORTICAL = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
               "Accumbens")
ROI_SETS = {
    "occipital": ("Occipital Pole", "Lateral Occipital Cortex, superior division",
                  "Lateral Occipital Cortex, inferior division", "Intracalcarine Cortex",
                  "Supracalcarine Cortex", "Lingual Gyrus", "Occipital Fusiform Gyrus",
                  "Cuneal Cortex"),
    "sensorimotor": ("Precentral Gyrus", "Postcentral Gyrus"),
}


def harvard_oxford_atlas(target_affine, target_shape):
    """Harvard-Oxford cortical (thr25, 2 mm, left/right) + subcortical grey matter.

    Resampled (nearest neighbour) to the given grid. Returns ``(labels, names)`` with
    ``names[k]`` the region of integer label ``k`` (``k >= 1``). nilearn downloads the
    atlas (a few MB, FSL) on first use.
    """
    from nilearn import datasets, image

    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm",
                                               symmetric_split=True)
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")

    def grid(img):
        return np.asarray(image.resample_img(
            img, target_affine=target_affine, target_shape=tuple(target_shape[:3]),
            interpolation="nearest", force_resample=True, copy_header=True).dataobj,
            dtype=int)

    c, s = grid(cort.maps), grid(sub.maps)
    names = {k: str(n) for k, n in enumerate(cort.labels) if k > 0}
    labels = c.copy()
    offset = max(names) + 1
    keep = [(k, str(n)) for k, n in enumerate(sub.labels)
            if any(str(n).endswith(r) for r in SUBCORTICAL) or str(n) == "Brain-Stem"]
    for j, (k, n) in enumerate(keep):
        labels[(c == 0) & (s == k)] = offset + j
        names[offset + j] = n
    present = set(np.unique(labels)) - {0}
    return labels, {k: v for k, v in names.items() if k in present}


def roi_set_columns(names, roi_set: str) -> list[str]:
    """Column names (``roi_<region>``) of a predefined ROI set, both hemispheres."""
    wanted = ROI_SETS[roi_set]
    cols = [f"roi_{n}" for n in names.values()
            if any(n in (f"Left {w}", f"Right {w}") for w in wanted)]
    if len(cols) != 2 * len(wanted):
        raise ValueError(f"ROI set {roi_set}: found {len(cols)} of {2 * len(wanted)} regions")
    return cols


def preproc_path(bids_root, subject: str, run: str) -> Path:
    return Path(bids_root, DERIV, subject, "ses-001", "func",
                f"{subject}_ses-001_task-main_{run}_preproc.nii")


def run_lss(bids_root, subject: str, run: str, trials: pd.DataFrame, atlas=None,
            tr: float = TR_SECONDS, hrf_model: str = "glover", high_pass: float = 1 / 128):
    """Single-trial ROI betas (% signal change) and QC for one run.

    ``trials``: this run's rows of the trial table (all trials enter the design).
    ``atlas``: ``(labels, names)`` on the derivative grid; built if None.
    Returns ``(betas, qc)``; ``betas`` has the trial identifiers plus ``roi_*`` columns.
    """
    import nibabel as nib

    from ..fmri_lss import lss_weights, percent_signal, roi_timeseries, tsnr

    img = nib.load(preproc_path(bids_root, subject, run))
    if atlas is None:
        atlas = harvard_oxford_atlas(img.affine, img.shape)
    labels, names = atlas
    data = np.asarray(img.dataobj, dtype=np.float32)
    n_vol = data.shape[3]
    motion_file = preproc_path(bids_root, subject, run).with_name(
        f"{subject}_ses-001_task-main_{run}_headmove.txt")
    motion = read_motion(motion_file)
    if len(motion) != n_vol:
        raise ValueError(f"{subject} {run}: {len(motion)} motion rows != {n_vol} volumes")
    Y, n_vox = roi_timeseries(data, labels, list(names))
    del data
    g = trials.sort_values("onset_seconds")
    W, cond = lss_weights(g["onset_seconds"], g["condition"], n_vol, tr, confounds=motion,
                          hrf_model=hrf_model, high_pass=high_pass)
    B = W @ np.nan_to_num(percent_signal(Y))
    B[:, np.isnan(Y).all(axis=0)] = np.nan
    ids = g[["subject", "run", "trial_index", "condition"]].reset_index(drop=True)
    betas = pd.concat([ids, pd.DataFrame(B, columns=[f"roi_{n}" for n in names.values()])],
                      axis=1)
    t = tsnr(Y)
    qc = {"subject": subject, "run": run, "n_volumes": int(n_vol), "n_trials": len(g),
          "last_onset_to_scan_end_s": float(n_vol * tr - g["onset_seconds"].max()),
          "lss_condition_number_max": float(cond.max()),
          "roi_tsnr_median": float(np.nanmedian(t)), "roi_tsnr_min": float(np.nanmin(t)),
          "n_rois": len(names), "n_rois_empty": int(sum(v == 0 for v in n_vox.values())),
          "voxels_per_roi_median": float(np.median(list(n_vox.values()))),
          **motion_summary(motion)}
    return betas, qc


def _one_sided_t(x) -> dict:
    from scipy.stats import ttest_1samp

    x = np.asarray(x, float)
    res = ttest_1samp(x, 0.0, alternative="greater")
    return {"per_run": [round(float(v), 4) for v in x], "mean": float(x.mean()),
            "t": float(res.statistic), "p_one_sided": float(res.pvalue), "n_runs": len(x)}


def positive_controls(betas: pd.DataFrame, trials: pd.DataFrame, names) -> dict:
    """The two pre-specified pipeline checks of configs/ds002158_fmri_pilot.yaml (per subject).

    visual: mean beta of valid trials in occipital ROIs > 0 (t over run means).
    movement: active - observed mean beta in pre/postcentral ROIs > 0 (t over runs).
    """
    keys = ["subject", "run", "trial_index"]
    d = betas.merge(trials[keys + ["valid_trial"]], on=keys, how="left", validate="1:1")
    d = d[d["valid_trial"].astype(bool)]
    occ = d[roi_set_columns(names, "occipital")].mean(axis=1)
    sm = d[roi_set_columns(names, "sensorimotor")].mean(axis=1)
    out = {}
    for subject, g in d.groupby("subject"):
        vis = occ[g.index].groupby(g["run"]).mean()
        s = sm[g.index]
        diff = (s[g["condition"] == "active"].groupby(g["run"]).mean()
                - s[g["condition"] == "observed"].groupby(g["run"]).mean()).dropna()
        out[subject] = {"visual": _one_sided_t(vis), "movement": _one_sided_t(diff)}
    return out


def decode_pilot(betas: pd.DataFrame, trials: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Exploratory within-subject, leave-one-run-out decoding from ROI betas."""
    from ..eeg_decoding import decode

    keys = ["subject", "run", "trial_index"]
    cols = ["valid_trial", "confidence_raw", "accuracy", "response_latency_seconds",
            "evidence_dots_difference"]
    d = betas.merge(trials[keys + cols], on=keys, how="left", validate="1:1")
    d = d[d["valid_trial"].astype(bool)]
    roi_cols = [c for c in d.columns if c.startswith("roi_") and d[c].notna().all()]
    rows = []
    for (subject, condition), g in d.groupby(["subject", "condition"]):
        if condition not in cfg["conditions"]:
            continue
        for name, spec in cfg["targets"].items():
            target = spec.get("target", name)
            target = "confidence_raw" if target == "confidence" else target
            covs = spec.get("covariates")
            need = [target] + (covs or [])
            h = g.dropna(subset=need)
            res = decode(h[roi_cols].to_numpy(), h[target].to_numpy(), h["run"].to_numpy(),
                         spec["task"], covariates=h[covs].to_numpy() if covs else None,
                         n_perm=cfg.get("n_perm", 1000), seed=cfg.get("seed", 0),
                         score_mode="per_group")
            rows.append({"subject": subject, "condition": condition, "target": name, **res})
    return pd.DataFrame(rows)
