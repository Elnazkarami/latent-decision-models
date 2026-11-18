"""Group-level metacognitive sensitivity: does confidence track accuracy, by condition?

Mirrors the logic of Pereira et al.'s primary analysis (logistic regression of first-order
accuracy on confidence; the confidence slope indexes metacognitive sensitivity). Their
paper used a mixed-effects logistic model with a random intercept and a random confidence
slope per participant. Here a **GEE** (population-averaged logistic model with an
exchangeable within-subject correlation and subject-clustered robust standard errors) is
used instead. GEE estimates *population-averaged* slopes, which are generally smaller in
magnitude than the *subject-specific* slopes of a mixed model, so coefficient values are
not directly comparable to the paper; the direction and significance of the
condition interaction are the comparable quantities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def metacognition_gee(df: pd.DataFrame, reference: str = "active") -> dict:
    """Fit ``accuracy ~ confidence * observed`` by GEE clustered on subject.

    ``df`` needs ``subject``, ``condition`` (active/observed), ``accuracy`` (0/1) and
    ``confidence_raw`` in [0, 1]; use only valid trials. Confidence is entered on its
    native 0–1 scale, so the slope is the log-odds change from "sure error" to "sure correct".
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    need = {"subject", "condition", "accuracy", "confidence_raw"}
    if need - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(need - set(df.columns))}")
    d = df.dropna(subset=["confidence_raw", "accuracy"]).copy()
    if set(d["condition"]) - {"active", "observed"}:
        raise ValueError("Conditions must be 'active'/'observed'")
    d["observed"] = (d["condition"] != reference).astype(float)
    d["confidence"] = d["confidence_raw"].astype(float)
    d = d.sort_values(["subject"]).reset_index(drop=True)
    model = smf.gee("accuracy ~ confidence * observed", groups="subject", data=d,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable())
    fit = model.fit()
    ci = fit.conf_int()
    coefs = {name: {"estimate": float(fit.params[name]), "robust_se": float(fit.bse[name]),
                    "ci95": [float(ci.loc[name, 0]), float(ci.loc[name, 1])],
                    "p_value": float(fit.pvalues[name])}
             for name in fit.params.index}
    slope_obs = fit.params["confidence"] + fit.params["confidence:observed"]
    cov = fit.cov_params()
    se_obs = float(np.sqrt(cov.loc["confidence", "confidence"]
                           + cov.loc["confidence:observed", "confidence:observed"]
                           + 2 * cov.loc["confidence", "confidence:observed"]))
    return {
        "model": "GEE logistic: accuracy ~ confidence * observed; groups=subject; "
                 "exchangeable working correlation; robust (sandwich) SE",
        "deviation_from_paper": "paper used a mixed-effects logistic model (random intercept + "
                                "random confidence slope); GEE slopes are population-averaged",
        "n_trials": len(d),
        "n_subjects": int(d["subject"].nunique()),
        "coefficients": coefs,
        "confidence_slope_active": float(fit.params["confidence"]),
        "confidence_slope_observed": float(slope_obs),
        "confidence_slope_observed_se": se_obs,
        "slope_difference_observed_minus_active": coefs["confidence:observed"],
        "working_correlation": float(fit.cov_struct.dep_params),
        "converged": bool(getattr(fit, "converged", True)),
    }
