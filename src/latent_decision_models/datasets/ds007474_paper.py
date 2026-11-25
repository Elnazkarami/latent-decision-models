"""Quantities comparable to the ds007474 study's preprint (Ding, Cockburn et al.,
bioRxiv 10.1101/2025.09.17.676398; published in Cell Reports 45(6), 2026).

``docs/VALIDATION.md`` lists the preprint's values and the agreement rules (fixed before
these were computed). Valid trials of every subject in the trial table are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def _prob(t: pd.DataFrame, which: pd.Series) -> np.ndarray:
    return np.where(which == 1, t["reward_probability_planet1"],
                    t["reward_probability_planet2"])


def reward_probability(trials: pd.DataFrame) -> dict:
    """D3: mean reward probability obtained vs the mean over the two options (paired t).

    ``planet``: probability of the planet reached (the pre-specified reading);
    ``contingent``: of the option the reward depends on (the chosen ship under stimulus
    contingency, the planet reached under state contingency; post hoc, it matches the
    preprint).
    """
    v = trials[trials["valid_trial"].astype(bool)].copy()
    v["planet_p"] = _prob(v, v["planet"])
    v["contingent_p"] = np.where(v["reward_contingency"] == "stimulus",
                                 _prob(v, v["choice"]), v["planet_p"])
    v["null_p"] = (v["reward_probability_planet1"] + v["reward_probability_planet2"]) / 2
    m = v.groupby("subject")[["planet_p", "contingent_p", "null_p"]].mean()
    out = {"null_percent": float(100 * m["null_p"].mean()), "n_subjects": len(m)}
    for k in ("planet_p", "contingent_p"):
        r = ttest_rel(m[k], m["null_p"])
        out[k.replace("_p", "")] = {"percent": float(100 * m[k].mean()),
                                    "t": float(r.statistic), "p": float(r.pvalue)}
    return out


def stay_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Consecutive valid trial pairs within a run: stay, previous win, previous rare."""
    t = trials.sort_values(["subject", "session", "run", "trial_index"])
    parts = []
    for _, g in t.groupby(["subject", "session", "run"], sort=False):
        g = g.reset_index(drop=True)
        prev = g.shift(1)
        ok = g["valid_trial"].astype(bool) & prev["valid_trial"].eq(True)
        parts.append(pd.DataFrame({"subject": g["subject"],
                                   "stay": (g["choice"] == prev["choice"]).astype(int),
                                   "prev_win": prev["reward_binary"],
                                   "prev_rare": prev["rare_transition"]})[ok])
    return pd.concat(parts, ignore_index=True).dropna()


def stay_regression(trials: pd.DataFrame) -> dict:
    """D4: logistic GEE of stay on previous win x previous rare (+-0.5 coding), clustered
    on subject. Coefficients are also given for +-1 coding (the scale of most papers)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    st = stay_table(trials)
    st["win"] = st["prev_win"] - 0.5
    st["rare"] = st["prev_rare"] - 0.5
    fit = smf.gee("stay ~ win * rare", groups="subject", data=st,
                  family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    scale = {"win": 0.5, "rare": 0.5, "win:rare": 0.25}
    return {"n_pairs": len(st), "n_subjects": int(st["subject"].nunique()),
            **{k: {"coef": float(fit.params[k]), "coef_pm1_coding": float(fit.params[k]
                                                                         * s),
                   "p": float(fit.pvalues[k])} for k, s in scale.items()}}


def compare(trials: pd.DataFrame) -> dict:
    v = trials[trials["valid_trial"].astype(bool)]
    return {"D1_subjects_with_data": int(trials["subject"].nunique()),
            "D2_common_transition_fraction": float(1 - v["rare_transition"].mean()),
            "D3_reward_probability": reward_probability(trials),
            "D4_stay_regression": stay_regression(trials)}
