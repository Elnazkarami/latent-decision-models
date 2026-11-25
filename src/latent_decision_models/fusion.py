"""Behavior + neural feature fusion with participant-disjoint and within-subject evaluation.

Dataset-agnostic. Each unit of data is one subject: behavior features ``B`` (trials x
k_b), neural features ``E`` (trials x k_e), target ``y`` and a run label per trial.

Models (every scaler and hyper-parameter fitted on training data only):

* ``B``: ridge on behavior;
* ``E``: ridge on neural features;
* ``B+E``: stacking. A ridge model on ``E`` is fitted inside the training data; its
  out-of-fold predictions there (inner folds = the training groups) and its prediction
  for the test data enter a ridge model together with ``B``. One penalty therefore never
  has to serve a handful of behavior features and hundreds of neural features at once;
* ``MLP``: optional small PyTorch network (:mod:`fusion_torch`), participant-disjoint only.

Scores: Spearman r within each held-out run, averaged over runs (``per_group``), so
differences in run or subject means do not count as prediction. Consistently, training
targets are centred within each training run (:func:`center_by`): otherwise run offsets,
which no feature can predict, leak into the stacked model through biased out-of-fold
predictions and add noise (found on synthetic data before any real-data fit).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 5, 13)
LINEAR_MODELS = ("B", "E", "B+E")


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def center_by(y, labels) -> np.ndarray:
    """``y`` minus its mean within each label (e.g. subject + run)."""
    y = np.asarray(y, float)
    return y - pd.Series(y).groupby(np.asarray(labels)).transform("mean").to_numpy()


def stacked_predict(B_tr, E_tr, y_tr, groups_tr, B_te, E_te) -> np.ndarray:
    """B+E stacking; ``groups_tr`` defines the inner folds for the E predictions."""
    if len(np.unique(groups_tr)) < 2:
        raise ValueError("stacking needs at least two training groups")
    e_oof = np.full(len(y_tr), np.nan)
    for tr, te in LeaveOneGroupOut().split(E_tr, y_tr, groups_tr):
        e_oof[te] = ridge().fit(E_tr[tr], y_tr[tr]).predict(E_tr[te])
    if np.isnan(e_oof).any():
        raise ValueError("stacking needs at least two training groups")
    e_te = ridge().fit(E_tr, y_tr).predict(E_te)
    top = ridge().fit(np.column_stack([B_tr, e_oof]), y_tr)
    return top.predict(np.column_stack([B_te, e_te]))


def fit_predict(model: str, B_tr, E_tr, y_tr, groups_tr, B_te, E_te) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if model == "B":
            return ridge().fit(B_tr, y_tr).predict(B_te)
        if model == "E":
            return ridge().fit(E_tr, y_tr).predict(E_te)
        if model == "B+E":
            return stacked_predict(B_tr, E_tr, y_tr, groups_tr, B_te, E_te)
    raise ValueError(f"unknown model {model!r}")


def _frame(subject, runs, y):
    return pd.DataFrame({"subject": subject, "run": runs, "y": y})


def loso_predictions(data: dict, models=LINEAR_MODELS, mlp=None, seed: int = 0,
                     n_jobs: int = 1) -> pd.DataFrame:
    """Leave-one-subject-out predictions for every subject in ``data``.

    ``data``: {subject: (B, E, y, runs)}. Inner folds for stacking are the training
    subjects. ``mlp``: config for :func:`fusion_torch.fit_predict_mlp` (adds ``MLP``).
    Returns one row per test trial with a ``pred_<model>`` column per model.
    """
    from joblib import Parallel, delayed

    subjects = list(data)

    def one(s):
        tr = [o for o in subjects if o != s]
        B_tr = np.concatenate([data[o][0] for o in tr])
        E_tr = np.concatenate([data[o][1] for o in tr])
        y_tr = np.concatenate([center_by(data[o][2], data[o][3]) for o in tr])
        g_tr = np.concatenate([[o] * len(data[o][2]) for o in tr])
        B_te, E_te, y_te, runs = data[s]
        out = _frame(s, runs, y_te)
        for m in models:
            out[f"pred_{m}"] = fit_predict(m, B_tr, E_tr, y_tr, g_tr, B_te, E_te)
        if mlp is not None:
            from .fusion_torch import fit_predict_mlp

            out["pred_MLP"] = fit_predict_mlp(B_tr, E_tr, y_tr, g_tr, B_te, E_te, mlp,
                                              seed=seed)
        return out

    return pd.concat(Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in subjects),
                     ignore_index=True)


def within_predictions(data: dict, models=LINEAR_MODELS, n_jobs: int = 1) -> pd.DataFrame:
    """Leave-one-run-out predictions within each subject (inner folds: training runs).

    Subjects with fewer than 3 runs are skipped (stacking needs 2 training runs).
    """
    from joblib import Parallel, delayed

    def one(s):
        B, E, y, runs = data[s]
        if len(np.unique(runs)) < 3:
            return None
        out = _frame(s, runs, y)
        for m in models:
            pred = np.full(len(y), np.nan)
            for tr, te in LeaveOneGroupOut().split(B, y, runs):
                pred[te] = fit_predict(m, B[tr], E[tr], center_by(y[tr], runs[tr]),
                                       runs[tr], B[te], E[te])
            out[f"pred_{m}"] = pred
        return out

    parts = Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in data)
    return pd.concat([p for p in parts if p is not None], ignore_index=True)


def per_subject_scores(pred: pd.DataFrame) -> pd.DataFrame:
    """Per subject and model: Spearman r within each run, averaged over runs."""
    models = [c[5:] for c in pred.columns if c.startswith("pred_")]
    rows = []
    for s, g in pred.groupby("subject", sort=False):
        row = {"subject": s, "n_trials": len(g), "n_runs": g["run"].nunique()}
        for m in models:
            per_run = [spearmanr(h["y"], h[f"pred_{m}"]).statistic
                       for _, h in g.groupby("run")
                       if h["y"].nunique() > 1 and h[f"pred_{m}"].nunique() > 1]
            per_run = [r for r in per_run if np.isfinite(r)]
            row[m] = float(np.mean(per_run)) if per_run else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
