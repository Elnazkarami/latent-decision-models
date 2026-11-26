"""Quantities comparable to Pereira et al. (2020) PNAS 117:8382 (behavior, main text).

Each function returns our estimate in the paper's terms; ``docs/VALIDATION.md`` lists
the paper's values and the agreement rules (fixed before these were computed). Wilcoxon
``V`` is the sum of positive ranks (as reported by R, which the paper uses).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression

from ..confidence_baseline import type2_auroc

#: Pereira et al. (2020) main-text values as (value, reported ± interval); see
#: docs/VALIDATION.md (P2-P5, P8). The paper does not state whether ± is an SD or an SEM.
PAPER_VALUES = {
    "valid trials, own": (119, 5), "valid trials, observed": (118, 5),
    "accuracy, %": (71.2, 1.0), "dot difference": (13.1, 1.7), "RT, ms": (385, 8),
    "type-2 AUROC, own": (0.92, 0.02), "type-2 AUROC, observed": (0.90, 0.03),
}


def _wilcoxon(x, alternative: str = "two-sided") -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    ranks = rankdata(np.abs(x))
    return {"n": len(x), "V": float(ranks[x > 0].sum()),
            "p": float(wilcoxon(x, alternative=alternative).pvalue)}


def _mean_sd(values) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n_subjects": len(v)}


def descriptives(trials: pd.DataFrame) -> dict:
    """P1-P5: subjects, valid trials per condition, accuracy, evidence and RT."""
    valid = trials[trials["valid_trial"].astype(bool)]
    responded = trials[(trials["condition"] == "active")
                       & ~trials["exclusion_reasons"].fillna("").str.contains(
                           "no_first_order_response")]
    active_valid = valid[valid["condition"] == "active"]
    return {
        "P1_n_subjects": int(trials["subject"].nunique()),
        "P2_valid_trials_per_subject": {
            c: _mean_sd(g.groupby("subject").size()) for c, g in valid.groupby("condition")},
        "P3_accuracy_percent": _mean_sd(100 * responded.groupby("subject")["accuracy"].mean()),
        "P4_dot_difference": _mean_sd(trials.groupby("subject")["evidence_dots_difference"]
                                      .mean()),
        "P5_rt_ms_active_valid": _mean_sd(1000 * active_valid.groupby("subject")
                                          ["response_time_seconds"].mean()),
    }


def mean_confidence_test(trials: pd.DataFrame) -> dict:
    """P6: paired t-test of per-subject mean confidence, active vs observed (valid trials)."""
    v = trials[trials["valid_trial"].astype(bool) & trials["confidence_raw"].notna()]
    m = v.groupby(["subject", "condition"])["confidence_raw"].mean().unstack()
    res = ttest_rel(m["active"], m["observed"])
    return {"active": _mean_sd(m["active"]), "observed": _mean_sd(m["observed"]),
            "t": float(res.statistic), "df": len(m) - 1, "p": float(res.pvalue)}


def logistic_slopes(trials: pd.DataFrame, C: float = 100.0) -> dict:
    """P7: per-subject slope of accuracy on confidence, per condition; active - observed.

    Mild L2 (``C``) keeps slopes finite when confidence separates correct from error.
    """
    v = trials[trials["valid_trial"].astype(bool) & trials["confidence_raw"].notna()]
    rows = []
    for (s, c), g in v.groupby(["subject", "condition"]):
        y = g["accuracy"].to_numpy(int)
        if 0 < y.sum() < len(y):
            m = LogisticRegression(C=C).fit(g[["confidence_raw"]].to_numpy(), y)
            rows.append({"subject": s, "condition": c, "slope": float(m.coef_[0, 0])})
    sl = pd.DataFrame(rows).pivot(index="subject", columns="condition", values="slope")
    diff = (sl["active"] - sl["observed"]).dropna()
    return {"median_slope": sl.median().to_dict(), "active_minus_observed": _mean_sd(diff),
            "n_active_steeper": int((diff > 0).sum()), **_wilcoxon(diff)}


def type2_test(trials: pd.DataFrame) -> dict:
    """P8: per-subject type-2 AUROC, active vs observed, paired Wilcoxon."""
    v = trials[trials["valid_trial"].astype(bool) & trials["confidence_raw"].notna()]
    a = type2_auroc(v).pivot(index="subject", columns="condition", values="auroc2")
    diff = (a["active"] - a["observed"]).dropna()
    return {"active": _mean_sd(a["active"].dropna()),
            "observed": _mean_sd(a["observed"].dropna()), **_wilcoxon(diff)}


def rt_confidence(trials: pd.DataFrame, min_trials: int = 5) -> dict:
    """P9: per-subject Spearman(RT, confidence): active correct, active error, observed.

    Observed trials use the replayed latency (the RT the participant saw), as the paper
    analysed observation RTs.
    """
    v = trials[trials["valid_trial"].astype(bool) & trials["confidence_raw"].notna()]
    cells = {"active_correct": (v["condition"] == "active") & (v["accuracy"] == 1),
             "active_error": (v["condition"] == "active") & (v["accuracy"] == 0),
             "observed_all": v["condition"] == "observed"}
    out = {}
    for name, mask in cells.items():
        rt = "response_time_seconds" if name.startswith("active") else \
            "response_latency_seconds"
        r = [spearmanr(g[rt], g["confidence_raw"]).statistic
             for _, g in v[mask].groupby("subject") if len(g) >= min_trials]
        r = [x for x in r if np.isfinite(x)]
        out[name] = {**_mean_sd(r), **_wilcoxon(r)}
    return out


def compare(trials: pd.DataFrame) -> dict:
    return {"descriptives": descriptives(trials), "P6_mean_confidence": mean_confidence_test(trials),
            "P7_logistic_slopes": logistic_slopes(trials), "P8_type2_auroc": type2_test(trials),
            "P9_rt_confidence": rt_confidence(trials)}
