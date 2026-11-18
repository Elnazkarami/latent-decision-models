"""Behavioral confidence baselines with subject-disjoint evaluation (LDM-002).

Target: ``confidence_raw`` treated as continuous on [0, 1]. Justification: ds002158 uses a
visual-analogue scale with an automatically moving cursor (0 = sure error, 1 = sure
correct); 2 decimals are stored. This choice is dataset-specific and documented in the audit.

All estimands are *post-decision*: features such as objective accuracy and the
response latency are known only after the first-order decision. These models describe
how well behavior explains confidence in **new participants**; they are not prospective
predictions and say nothing about neural signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "confidence_raw"

# Nested feature sets; each adds information to the previous one.
FEATURE_SETS: dict[str, list[str]] = {
    "null_train_mean": [],
    "task": ["evidence_dots_difference", "is_observed"],
    "behavior": [
        "evidence_dots_difference",
        "is_observed",
        "accuracy",
        "response_latency_seconds",
        "accuracy_x_observed",
        "latency_x_observed",
    ],
}


def prepare(trials: pd.DataFrame, exclude_subjects: tuple[str, ...] = ()) -> pd.DataFrame:
    """Keep valid trials with confidence; add design features. Never imputes the target."""
    need = {"subject", "condition", "valid_trial", TARGET, "accuracy",
            "evidence_dots_difference", "response_latency_seconds"}
    missing = need - set(trials.columns)
    if missing:
        raise ValueError(f"Trial table lacks columns: {sorted(missing)}")
    work = trials[trials["valid_trial"].astype(bool) & trials[TARGET].notna()].copy()
    work = work[~work["subject"].isin(exclude_subjects)]
    bad_cond = set(work["condition"]) - {"active", "observed"}
    if bad_cond:
        raise ValueError(f"Unexpected conditions {bad_cond}")
    if work["response_latency_seconds"].isna().any():
        raise ValueError("Valid trials must have a response latency")
    work["is_observed"] = (work["condition"] == "observed").astype(float)
    work["accuracy_x_observed"] = work["accuracy"] * work["is_observed"]
    work["latency_x_observed"] = work["response_latency_seconds"] * work["is_observed"]
    return work.reset_index(drop=True)


def _model(features: list[str], alpha: float):
    if not features:
        return DummyRegressor(strategy="mean")
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def type2_auroc(df: pd.DataFrame) -> pd.DataFrame:
    """Per subject × condition AUROC of confidence for discriminating correct vs error.

    Descriptive metacognitive sensitivity (0.5 = none). NaN when a cell lacks errors
    or corrects.
    """
    rows = []
    for (subj, cond), g in df.groupby(["subject", "condition"]):
        y = g["accuracy"].to_numpy()
        auc = roc_auc_score(y, g[TARGET]) if 0 < y.sum() < len(y) else np.nan
        rows.append({"subject": subj, "condition": cond, "n": len(g),
                     "n_errors": int((y == 0).sum()), "auroc2": auc})
    return pd.DataFrame(rows)


def _boot_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def calibration(y: np.ndarray, pred: np.ndarray) -> dict:
    """Calibration of continuous predictions: OLS ``y = intercept + slope * pred``.

    Perfect calibration is slope 1 and intercept 0. A slope below 1 means overconfident
    (too-spread) predictions; above 1 means predictions shrunk toward the mean.
    """
    pred = np.asarray(pred, float)
    if np.ptp(pred) == 0:
        return {"slope": None, "intercept": None}
    slope, intercept = np.polyfit(pred, np.asarray(y, float), 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def run_within_subject(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    alpha: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Leave-one-run-out *within each subject* (a separate estimand from subject-held-out).

    Each subject gets their own models, trained only on that subject's other runs, so this
    measures how well behavior explains confidence once a person's idiosyncratic scale use
    has been learned. ``df`` must come from :func:`prepare` and include ``run``.
    """
    feature_sets = feature_sets or FEATURE_SETS
    if "run" not in df.columns:
        raise ValueError("run-held-out evaluation needs a 'run' column")
    names = list(feature_sets)
    rows, skipped = [], []
    for subj, g in df.groupby("subject"):
        runs = g["run"].to_numpy()
        if len(np.unique(runs)) < 2:
            skipped.append(subj)
            continue
        y = g[TARGET].to_numpy(float)
        pred = {n: np.full(len(g), np.nan) for n in names}
        for tr, te in LeaveOneGroupOut().split(g, y, runs):
            if set(runs[tr]) & set(runs[te]):
                raise RuntimeError("Run leakage within subject")
            for n, feats in feature_sets.items():
                x = g[feats].to_numpy(float) if feats else np.zeros((len(g), 1))
                pred[n][te] = _model(feats, alpha).fit(x[tr], y[tr]).predict(x[te])
        rows.append({"subject": subj, "n_runs": len(np.unique(runs)),
                     **{n: float(np.mean(np.abs(y - pred[n]))) for n in names}})
    out = {"validation": "LeaveOneGroupOut over runs within each subject; per-subject models",
           "n_subjects": len(rows), "skipped_subjects_single_run": skipped, "models": {},
           "per_subject_mae": []}
    if not rows:
        out["note"] = "not estimable: no subject has more than one run"
        return out
    per = pd.DataFrame(rows).set_index("subject")
    rng = np.random.default_rng(seed)
    null = names[0]
    for n in names:
        e = {"mae_mean_of_subjects": float(per[n].mean()),
             "mae_mean_of_subjects_ci95": _boot_ci(per[n].to_numpy(), n_boot, rng)}
        if n != null:
            diff = (per[null] - per[n]).to_numpy()
            e.update(mae_improvement_vs_null=float(diff.mean()),
                     mae_improvement_vs_null_ci95=_boot_ci(diff, n_boot, rng),
                     n_subjects_improved=int((diff > 0).sum()))
        out["models"][n] = e
    out["per_subject_mae"] = per.reset_index().to_dict(orient="records")
    return out


def run_confidence_baseline(
    trials: pd.DataFrame,
    n_splits: int = 5,
    alpha: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
    exclude_subjects: tuple[str, ...] = (),
    feature_sets: dict[str, list[str]] | None = None,
    return_predictions: bool = False,
    within_subject: bool = True,
) -> dict:
    """Subject-disjoint cross-validated comparison of nested confidence models.

    Ridge ``alpha`` is fixed a priori (no tuning), so no nested CV is needed. Scaling is
    fit inside each training fold. Uncertainty: participant-level bootstrap of per-subject
    out-of-fold MAE (and of the improvement over the null model).
    """
    feature_sets = feature_sets or FEATURE_SETS
    df = prepare(trials, exclude_subjects)
    subjects = df["subject"].to_numpy()
    if len(np.unique(subjects)) < n_splits:
        raise ValueError("Not enough distinct participants for requested folds")
    y = df[TARGET].to_numpy(float)
    oof = {name: np.full(len(df), np.nan) for name in feature_sets}
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(df, y, subjects), 1):
        tr_s, te_s = set(subjects[tr]), set(subjects[te])
        if tr_s & te_s:
            raise RuntimeError("Subject leakage between folds")
        rec = {"fold": k, "n_train": len(tr), "n_test": len(te),
               "test_subjects": sorted(te_s)}
        for name, feats in feature_sets.items():
            x = df[feats].to_numpy(float) if feats else np.zeros((len(df), 1))
            m = _model(feats, alpha).fit(x[tr], y[tr])
            oof[name][te] = m.predict(x[te])
            rec[f"{name}_mae"] = float(mean_absolute_error(y[te], oof[name][te]))
        folds.append(rec)

    rng = np.random.default_rng(seed)
    null = next(iter(feature_sets))
    per_subject = pd.DataFrame({"subject": subjects, "condition": df["condition"], "y": y,
                                **{n: np.abs(y - p) for n, p in oof.items()}})
    subj_mae = per_subject.groupby("subject")[list(feature_sets)].mean()
    models = {}
    for name in feature_sets:
        rho = spearmanr(y, oof[name]).statistic if name != null else np.nan
        entry = {
            "features": feature_sets[name],
            "mae_trials": float(np.mean(np.abs(y - oof[name]))),
            "mae_mean_of_subjects": float(subj_mae[name].mean()),
            "mae_mean_of_subjects_ci95": _boot_ci(subj_mae[name].to_numpy(), n_boot, rng),
            "spearman_rho_pooled": None if np.isnan(rho) else float(rho),
            "mae_by_condition": {c: float(g[name].mean())
                                 for c, g in per_subject.groupby("condition")},
        }
        if name != null:
            entry["calibration_pooled_oof"] = calibration(y, oof[name])
            diff = (subj_mae[null] - subj_mae[name]).to_numpy()
            entry["mae_improvement_vs_null"] = float(diff.mean())
            entry["mae_improvement_vs_null_ci95"] = _boot_ci(diff, n_boot, rng)
            entry["n_subjects_improved"] = int((diff > 0).sum())
        models[name] = entry

    full = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    beh = feature_sets.get("behavior")
    coefs = None
    if beh:
        full.fit(df[beh].to_numpy(float), y)
        coefs = dict(zip(beh, map(float, full.named_steps["ridge"].coef_)))

    auc = type2_auroc(df)
    auc_summary = {c: {"mean": float(g["auroc2"].mean()), "sd": float(g["auroc2"].std()),
                       "n_subjects": int(g["auroc2"].notna().sum())}
                   for c, g in auc.groupby("condition")}
    paired = auc.pivot(index="subject", columns="condition", values="auroc2").dropna()
    auc_diff = None
    if {"active", "observed"} <= set(paired.columns) and len(paired) > 1:
        dvals = (paired["active"] - paired["observed"]).to_numpy()
        auc_diff = {"mean_active_minus_observed": float(dvals.mean()),
                    "ci95": _boot_ci(dvals, n_boot, rng), "n_subjects": len(dvals),
                    "n_active_higher": int((dvals > 0).sum())}

    result = {
        "target": TARGET,
        "target_treatment": "continuous [0,1]",
        "estimand": "post-decision confidence in held-out participants from behavior",
        "validation": f"GroupKFold(n_splits={n_splits}) by subject; scaler fit per train fold",
        "ridge_alpha_fixed": alpha,
        "n_trials": len(df),
        "n_subjects": int(df["subject"].nunique()),
        "excluded_subjects": list(exclude_subjects),
        "trials_by_condition": df["condition"].value_counts().to_dict(),
        "models": models,
        "behavior_standardized_coefficients_full_data_descriptive": coefs,
        "type2_auroc_by_condition": auc_summary,
        "type2_auroc_active_minus_observed": auc_diff,
        "folds": folds,
        "bootstrap": {"unit": "subject", "n_boot": n_boot, "seed": seed},
    }
    if within_subject and "run" in df.columns:
        ws = run_within_subject(df, feature_sets, alpha, n_boot, seed)
        between = subj_mae.loc[[r["subject"] for r in ws["per_subject_mae"]]]
        for n in feature_sets if ws["per_subject_mae"] else []:
            if n == null:
                continue
            gain = (between[n].to_numpy()
                    - np.array([r[n] for r in ws["per_subject_mae"]]))
            ws["models"][n]["personalization_gain_vs_subject_held_out"] = {
                "mean_mae_reduction": float(gain.mean()),
                "ci95": _boot_ci(gain, n_boot, rng),
                "n_subjects_better_within": int((gain > 0).sum())}
        result["within_subject_run_held_out"] = ws
    if return_predictions:
        result["oof_predictions"] = df[["subject", "condition", TARGET]].assign(
            **{f"pred_{n}": p for n, p in oof.items()})
    return result
