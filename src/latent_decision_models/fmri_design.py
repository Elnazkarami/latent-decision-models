"""Design diagnostics for trial-level fMRI, computed from event timing alone (no images).

Answers, per run, before any BOLD data are touched:

* how well single-trial responses can be estimated: variance inflation factor (VIF) of
  each trial regressor under LSA (one regressor per trial) and LSS (the trial vs. all
  other trials of each condition; Mumford et al. 2012), and the correlation of
  neighbouring trial regressors;
* whether phases within a trial can be separated: correlation of HRF-convolved
  regressors locked to stimulus onset (pre-decision), the response (button press or
  virtual hand) and an optional post-decision event.

Uses nilearn's design-matrix builder (optional ``[fmri]`` extra). Onsets must be in
seconds relative to the first acquired volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _design(frame_times, events, hrf_model, high_pass):
    import warnings

    try:
        from nilearn.glm.first_level import make_first_level_design_matrix
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("fmri_design needs nilearn: pip install "
                          "'latent-decision-models[fmri]'") from e
    with warnings.catch_warnings():  # zero-duration (stick) events are intended
        warnings.filterwarnings("ignore", message=".*null duration.*")
        return make_first_level_design_matrix(frame_times, events, hrf_model=hrf_model,
                                              drift_model="cosine", high_pass=high_pass)


def vif(X: pd.DataFrame, columns) -> pd.Series:
    """VIF of ``columns`` against all other columns of ``X`` (``X`` includes the constant)."""
    out = {}
    A = X.to_numpy(float)
    for c in columns:
        j = X.columns.get_loc(c)
        y, others = A[:, j], np.delete(A, j, axis=1)
        resid = y - others @ np.linalg.lstsq(others, y, rcond=None)[0]
        r2 = 1 - resid.var() / y.var()
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out)


def run_diagnostics(onsets, conditions, n_volumes: int, tr: float, response_offsets=None,
                    post_offsets=None, hrf_model: str = "glover",
                    high_pass: float = 1 / 128) -> dict:
    """Diagnostics for one run.

    ``onsets``: trial onsets (s); ``conditions``: label per trial; ``response_offsets`` and
    ``post_offsets``: optional delays (s) from onset to the response and to a
    post-decision event.
    """
    onsets = np.asarray(onsets, float)
    conditions = np.asarray(conditions)
    frame_times = np.arange(n_volumes) * tr
    n = len(onsets)
    names = [f"trial_{i:03d}" for i in range(n)]

    lsa = _design(frame_times, pd.DataFrame({"onset": onsets, "duration": 0.0,
                                             "trial_type": names}), hrf_model, high_pass)
    lsa_vif = vif(lsa, names)
    corr = lsa[names].corr().to_numpy()
    adjacent = np.abs(np.diag(corr, 1)) if n > 1 else np.array([])

    lss_vif = []
    for i in range(n):
        labels = np.where(np.arange(n) == i, "target", "other_" + conditions.astype(str))
        X = _design(frame_times, pd.DataFrame({"onset": onsets, "duration": 0.0,
                                               "trial_type": labels}), hrf_model, high_pass)
        lss_vif.append(float(vif(X, ["target"]).iloc[0]))

    out = {"n_trials": n, "n_volumes": n_volumes,
           "min_onset_gap_s": float(np.diff(np.sort(onsets)).min()) if n > 1 else None,
           "lsa_vif_median": float(np.median(lsa_vif)), "lsa_vif_max": float(lsa_vif.max()),
           "lss_vif_median": float(np.median(lss_vif)), "lss_vif_max": float(np.max(lss_vif)),
           "adjacent_trial_abs_r_median": float(np.median(adjacent)) if n > 1 else None,
           "adjacent_trial_abs_r_max": float(adjacent.max()) if n > 1 else None}

    phases = {"stimulus": onsets}
    if response_offsets is not None:
        phases["response"] = onsets + np.asarray(response_offsets, float)
    if post_offsets is not None:
        post = onsets + np.asarray(post_offsets, float)
        if np.isfinite(post).any():
            phases["post"] = post[np.isfinite(post)]
    if len(phases) > 1:
        ev = pd.concat([pd.DataFrame({"onset": t[np.isfinite(t)], "duration": 0.0,
                                      "trial_type": k}) for k, t in phases.items()])
        X = _design(frame_times, ev, hrf_model, high_pass)
        for k in phases:
            if k != "stimulus":
                out[f"r_stimulus_vs_{k}"] = float(np.corrcoef(X["stimulus"], X[k])[0, 1])
        out.update({f"vif_{k}": float(v) for k, v in vif(X, list(phases)).items()})
        # Same phases with an HRF time derivative per phase (as in the original GLM):
        # separation that rests only on the canonical HRF's exact latency disappears.
        Xd = _design(frame_times, ev, hrf_model + " + derivative", high_pass)
        out.update({f"vif_{k}_with_time_derivative": float(v)
                    for k, v in vif(Xd, list(phases)).items()})
    return out
