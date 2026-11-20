"""Classical trial-level EEG decoding with run- or subject-held-out evaluation.

* Features: mean amplitude per channel in consecutive bins (default 50 ms) inside an
  analysis window (:func:`window_features`).
* Models: standardisation + L2 logistic regression (classification, AUC) or ridge
  regression (continuous target, Spearman r). Regularisation is chosen by inner
  cross-validation **inside each training fold**; the scaler is also fitted there.
* Optional ``covariates``: the target is residualised on them with a linear model fitted
  on the training fold only, and the test target is residualised with those coefficients.
  This asks whether EEG explains the target *beyond* the covariates (e.g. confidence
  beyond accuracy, response time and evidence).
* Scoring (``score_mode``): ``"pooled"`` scores all held-out predictions together;
  ``"per_group"`` scores each held-out group (run) separately and averages. Pooled scores
  are biased when the target drifts between runs (a training fold's mean sits on the
  wrong side of the held-out run's mean, pushing the pooled score below chance);
  per-group scores are invariant to such offsets and centred on chance under the null.
* Inference: labels permuted *within* each group (run), repeating the full pipeline.
  p = (1 + #{perm >= observed}) / (1 + n_perm). Across subjects: :func:`group_test`.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def window_features(epochs: np.ndarray, times: np.ndarray, tmin: float, tmax: float,
                    bin_s: float = 0.05) -> np.ndarray:
    """Mean amplitude in ``bin_s`` bins over ``[tmin, tmax)``: shape (n, ch * n_bins)."""
    edges = np.arange(tmin, tmax - 1e-9, bin_s)
    feats = []
    for lo in edges:
        m = (times >= lo - 1e-9) & (times < min(lo + bin_s, tmax) - 1e-9)
        if not m.any():
            raise ValueError(f"no samples in bin [{lo}, {lo + bin_s})")
        feats.append(epochs[:, :, m].mean(axis=2))
    return np.concatenate(feats, axis=1)


def _model(task: str):
    if task == "classification":
        est = LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, scoring="roc_auc",
                                   max_iter=2000)
    elif task == "regression":
        est = RidgeCV(alphas=np.logspace(-1, 5, 13))
    else:
        raise ValueError(task)
    return make_pipeline(StandardScaler(), est)


def _score(task, y, pred):
    if task == "classification":
        return float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    return float(spearmanr(y, pred).statistic)


def _combined_score(task, target, pred, groups, score_mode):
    ok = ~np.isnan(pred)
    if score_mode == "pooled":
        return _score(task, target[ok], pred[ok])
    if score_mode == "per_group":
        per = [_score(task, target[ok & (groups == g)], pred[ok & (groups == g)])
               for g in np.unique(groups[ok])]
        per = [v for v in per if np.isfinite(v)]
        return float(np.mean(per)) if per else float("nan")
    raise ValueError(f"unknown score_mode {score_mode!r}")


def _residualise(y_tr, y_te, c_tr, c_te):
    lr = LinearRegression().fit(c_tr, y_tr)
    return y_tr - lr.predict(c_tr), y_te - lr.predict(c_te)


def _cv_predict(X, y, groups, task, covariates):
    pred = np.full(len(y), np.nan)
    target = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        y_tr, y_te = y[tr], y[te]
        if covariates is not None:
            y_tr, y_te = _residualise(y_tr, y_te, covariates[tr], covariates[te])
        if task == "classification" and len(np.unique(y_tr)) < 2:
            continue
        m = _model(task).fit(X[tr], y_tr)
        pred[te] = (m.predict_proba(X[te])[:, 1] if task == "classification"
                    else m.predict(X[te]))
        target[te] = y_te
    return pred, target


def _permute_within(y, groups, rng):
    yp = y.copy()
    for g in np.unique(groups):
        m = groups == g
        yp[m] = rng.permutation(y[m])
    return yp


def decode(X, y, groups, task: str, covariates=None, n_perm: int = 200,
           seed: int = 0, score_mode: str = "pooled") -> dict:
    """Leave-one-group-out decoding with a within-group permutation test."""
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    cov = None if covariates is None else np.asarray(covariates, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)  # sklearn API transition notices
        pred, target = _cv_predict(X, y, groups, task, cov)
        ok = ~np.isnan(pred)
        obs = _combined_score(task, target, pred, groups, score_mode)
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_perm):
            yp = _permute_within(y, groups, rng)
            pp, tp = _cv_predict(X, yp, groups, task, cov)
            null.append(_combined_score(task, tp, pp, groups, score_mode))
    null = np.asarray(null)
    return {"score": obs, "metric": "auc" if task == "classification" else "spearman_r",
            "score_mode": score_mode,
            "chance": 0.5 if task == "classification" else 0.0,
            "n_trials": int(ok.sum()), "n_groups": len(np.unique(groups)),
            "n_features": int(X.shape[1]), "n_perm": n_perm,
            "null_mean": float(np.nanmean(null)) if n_perm else None,
            "null_95": float(np.nanpercentile(null, 95)) if n_perm else None,
            "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm)) if n_perm else None}


def transfer(X_train, y_train, X_test, y_test, task: str, cov_train=None, cov_test=None,
             n_perm: int = 1000, seed: int = 0, test_groups=None,
             score_mode: str = "pooled") -> dict:
    """Train on one subject, test on another (subject-held-out).

    The model is fitted once; the p-value permutes the *test* labels, i.e. it asks
    whether the fixed model's predictions track the new subject's labels.
    """
    Xtr, Xte = np.asarray(X_train, float), np.asarray(X_test, float)
    ytr, yte = np.asarray(y_train, float), np.asarray(y_test, float)
    if cov_train is not None:
        ytr, yte = _residualise(ytr, yte, np.asarray(cov_train, float),
                                np.asarray(cov_test, float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)  # sklearn API transition notices
        m = _model(task).fit(Xtr, ytr)
    pred = m.predict_proba(Xte)[:, 1] if task == "classification" else m.predict(Xte)
    g = np.zeros(len(yte)) if test_groups is None else np.asarray(test_groups)
    obs = _combined_score(task, yte, pred, g, score_mode)
    rng = np.random.default_rng(seed)
    null = np.array([_combined_score(task, _permute_within(yte, g, rng), pred, g, score_mode)
                     for _ in range(n_perm)])
    return {"score": obs, "score_mode": score_mode, "n_train": len(ytr), "n_test": len(yte),
            "null_95": float(np.nanpercentile(null, 95)) if n_perm else None,
            "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm)) if n_perm else None}


def group_test(scores, chance: float, n_boot: int = 10_000, seed: int = 0,
               alternative: str = "greater") -> dict:
    """Across-subject test of per-subject scores against chance.

    Wilcoxon signed-rank (one-sided, scores > chance, by default; ``alternative`` as in
    SciPy), mean with a subject-bootstrap 95% CI, and the number of subjects above chance.
    """
    from scipy.stats import wilcoxon

    x = np.asarray(scores, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {"n_subjects": len(x), "p_wilcoxon": None}
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x))).mean(axis=1)
    return {"n_subjects": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_above_chance": int((x > chance).sum()),
            "p_wilcoxon": float(wilcoxon(x - chance, alternative=alternative).pvalue)}


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
