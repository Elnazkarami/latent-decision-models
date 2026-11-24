"""Model-free / model-based / hybrid reinforcement learning for the ds007474 two-step task.

Task (verified in ``docs/ds007474-audit.md`` and ``docs/ds007474-two-step-model.md``):
on each trial the participant picks one of two ships. The left ship commonly (p = 0.7)
reaches planet 1, the right ship planet 2; there is no second-stage choice. Under
*state* contingency the win depends on the planet reached (pads on a planet share
P(win)); under *stimulus* contingency it depends on the ship chosen, wherever it lands.

Agents (values reset at the start of every run, as in the authors' model):

* model-free (MF): ``Q_mf(a) += alpha * (r - Q_mf(a))`` for the chosen ship;
* model-based (MB): ``V(planet) += alpha * (r - V(planet))`` for the planet reached and
  ``Q_mb(a) = 0.7 V(common planet of a) + 0.3 V(other planet)`` with the instructed
  transition structure;
* choice: ``P(a) = softmax(beta * (w Q_mb + (1 - w) Q_mf) + stick * [a = previous choice])``.

``mf`` fixes ``w = 0``, ``mb`` fixes ``w = 1``, ``hybrid`` fits ``w`` and
``hybrid_contingency`` fits separate weights for stimulus- and state-contingent trials.

Extensions (motivated by the posterior-predictive misfit of ``hybrid``):

* ``hybrid_2lr``: separate learning rates ``alpha_mf`` and ``alpha_mb``;
* ``hybrid_wsls``: adds one-trial win-stay/lose-shift terms to the choice rule, as in the
  authors' model. ``wsls_mf`` favours repeating after a win and switching after a loss.
  ``wsls_mb`` favours the ship that commonly reaches the planet just rewarded: repeat
  after a win via a common transition or a loss via a rare one, otherwise switch;
* ``hybrid_2lr_wsls``: both.
* ``hybrid_wsls_arbitration``: ``hybrid_wsls`` whose MB weight varies trial by trial with the
  conditions just experienced (the authors' ``isPost_*`` terms):
  ``w_t = expit(logit(w) + arb_reliable * x_reliable + arb_high * x_high)``, where
  ``x_reliable`` is +0.5 if the previous trial had reliable pads (P = 0.9, low state
  uncertainty) and -0.5 if not (P = 0.5), and ``x_high`` is +0.5 / -0.5 for a high / low
  reward-magnitude block on the previous trial (0 on a run's first trial).
* ``hybrid_wsls_magmf`` / ``hybrid_wsls_magmf_arbitration``: as ``hybrid_wsls`` /
  ``hybrid_wsls_arbitration``, but model-free values learn from the points shown
  (magnitude / 100) and model-based values from win / no win, as in the authors' model
  (``magMF_binMB``). This is independent of the ``reward`` setting.
No-response trials neither update values nor enter the likelihood; the stickiness term
refers to the last choice actually made. Trials with RT below ``min_rt`` update values
(their outcome was shown) but are not scored.

Fits are MAP with an independent N(0, ``prior_sd``^2) prior on the unconstrained
parameters (the authors used N(0, 6.25) with a Laplace approximation). Model comparison
uses held-out log-likelihood, never in-sample fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logit

P_COMMON = 0.7
MODELS: dict[str, tuple[str, ...]] = {
    "mf": ("alpha", "beta", "stick"),
    "mb": ("alpha", "beta", "stick"),
    "hybrid": ("alpha", "beta", "w", "stick"),
    "hybrid_contingency": ("alpha", "beta", "w_stimulus", "w_state", "stick"),
    "hybrid_2lr": ("alpha_mf", "alpha_mb", "beta", "w", "stick"),
    "hybrid_wsls": ("alpha", "beta", "w", "stick", "wsls_mf", "wsls_mb"),
    "hybrid_2lr_wsls": ("alpha_mf", "alpha_mb", "beta", "w", "stick", "wsls_mf", "wsls_mb"),
    "hybrid_wsls_arbitration": ("alpha", "beta", "w", "stick", "wsls_mf", "wsls_mb",
                                "arb_reliable", "arb_high"),
    "hybrid_wsls_magmf": ("alpha", "beta", "w", "stick", "wsls_mf", "wsls_mb"),
    "hybrid_wsls_magmf_arbitration": ("alpha", "beta", "w", "stick", "wsls_mf", "wsls_mb",
                                      "arb_reliable", "arb_high"),
}
_UNIT = {"alpha", "alpha_mf", "alpha_mb", "w", "w_stimulus", "w_state"}
_POSITIVE = {"beta"}
RECOVERY_RANGES = {"alpha": (0.1, 0.9), "alpha_mf": (0.1, 0.9), "alpha_mb": (0.1, 0.9),
                   "beta": (1.0, 10.0), "w": (0.0, 1.0), "w_stimulus": (0.0, 1.0),
                   "w_state": (0.0, 1.0), "stick": (-0.5, 1.5), "wsls_mf": (-1.0, 1.0),
                   "wsls_mb": (-1.0, 1.0), "arb_reliable": (-1.0, 1.0),
                   "arb_high": (-1.0, 1.0)}
HIGH_MAGNITUDE = 0.25  # magnitude scale: high blocks U(0.30, 1.00), low blocks U(0.10, 0.19)


# --------------------------------------------------------------------------- parameters
def to_theta(params: dict, model: str) -> np.ndarray:
    out = []
    for k in MODELS[model]:
        v = params[k]
        out.append(logit(np.clip(v, 1e-6, 1 - 1e-6)) if k in _UNIT
                   else np.log(v) if k in _POSITIVE else v)
    return np.array(out, float)


def from_theta(theta: np.ndarray, model: str) -> dict:
    p = {}
    for k, t in zip(MODELS[model], theta, strict=True):
        p[k] = float(expit(t) if k in _UNIT else np.exp(t) if k in _POSITIVE else t)
    return p


def _weights(params: dict, model: str, contingency: np.ndarray, x_reliable=None,
             x_high=None) -> np.ndarray:
    """Per-sequence MB weight for the current trial (contingency: 0 stimulus, 1 state)."""
    if model == "mf":
        return np.zeros(contingency.shape)
    if model == "mb":
        return np.ones(contingency.shape)
    if "arb_reliable" in params:
        z = (logit(np.clip(params["w"], 1e-6, 1 - 1e-6))
             + params["arb_reliable"] * x_reliable + params["arb_high"] * x_high)
        return expit(z)
    if "w" in params:
        return np.full(contingency.shape, params["w"])
    return np.where(contingency == 1, params["w_state"], params["w_stimulus"])


# --------------------------------------------------------------------------- data
@dataclass
class TwoStepData:
    """Runs as rows (values reset per run), trials as columns; padded with ``choice = -1``.

    ``choice``: 0 left / 1 right / -1 none; ``planet``: 0/1/-1; ``reward``: learning signal;
    ``win``: binary outcome shown; ``rare``: rare first-stage transition; ``contingency``:
    0 stimulus / 1 state; ``scored``: enters the likelihood; ``x_reliable`` / ``x_high``:
    the previous trial's pad reliability and magnitude block, +-0.5 (see
    :func:`lagged_conditions`).
    """

    choice: np.ndarray
    planet: np.ndarray
    reward: np.ndarray
    win: np.ndarray
    rare: np.ndarray
    contingency: np.ndarray
    scored: np.ndarray
    subject: np.ndarray
    run: np.ndarray
    x_reliable: np.ndarray
    x_high: np.ndarray
    points: np.ndarray  # points shown / 100 (model-free signal of the magmf models)

    def subset(self, rows) -> TwoStepData:
        rows = np.asarray(rows)
        return TwoStepData(**{k: getattr(self, k)[rows] for k in self.__dataclass_fields__})

    @property
    def n_scored(self) -> int:
        return int(self.scored.sum())


@dataclass
class Schedule:
    """Everything needed to replay any agent on a task sequence (runs x trials)."""

    pad_if: np.ndarray  # (n, T, 2) landing pad 0-3 if the left/right ship were chosen
    draws: np.ndarray  # (n, T, 4) binary reward draw of each pad
    contingency: np.ndarray  # (n, T) 0 stimulus / 1 state
    magnitude: np.ndarray  # (n, T) magnitude scale in [0, 1]
    present: np.ndarray  # (n, T) trial exists
    responds: np.ndarray  # (n, T) a response is made (replays real no-response trials)
    subject: np.ndarray
    run: np.ndarray
    reliability: np.ndarray  # (n, T) P(common pad | planet): 0.5 or 0.9

    def subset(self, rows) -> Schedule:
        rows = np.asarray(rows)
        return Schedule(**{k: getattr(self, k)[rows] for k in self.__dataclass_fields__})


def lagged_conditions(reliability: np.ndarray, magnitude: np.ndarray,
                      present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Previous trial's conditions, coded +-0.5 (0 on each run's first trial).

    ``x_reliable``: +0.5 after a reliable-pad trial (P = 0.9), -0.5 after P = 0.5.
    ``x_high``: +0.5 after a high-magnitude trial (scale >= 0.25), -0.5 after a low one.
    """
    rel = np.where(present, np.where(reliability > 0.7, 0.5, -0.5), 0.0)
    high = np.where(present, np.where(magnitude >= HIGH_MAGNITUDE, 0.5, -0.5), 0.0)
    x_rel, x_high = np.zeros(rel.shape), np.zeros(high.shape)
    x_rel[:, 1:], x_high[:, 1:] = rel[:, :-1], high[:, :-1]
    return x_rel, x_high


def _runs(trials: pd.DataFrame, runs) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    t = trials[trials["run"].isin(runs)].sort_values(["subject", "run", "trial_index"])
    return list(t.groupby(["subject", "run"], sort=True))


def _pad(blocks: list[np.ndarray], fill, dtype) -> np.ndarray:
    T = max(len(b) for b in blocks)
    out = np.full((len(blocks), T) + blocks[0].shape[1:], fill, dtype)
    for i, b in enumerate(blocks):
        out[i, :len(b)] = b
    return out


def prepare(trials: pd.DataFrame, reward: str = "binary", min_rt: float | None = 0.15,
            runs=("run-1", "run-2")) -> TwoStepData:
    """Build model input from the ``ldm ds007474 trials`` table.

    ``reward``: ``"binary"`` (win/no win) or ``"points"`` (points shown / 100).
    """
    if reward not in ("binary", "points"):
        raise ValueError("reward must be 'binary' or 'points'")
    groups = _runs(trials, runs)
    if not groups:
        raise ValueError(f"no trials in runs {runs}")
    cols: dict[str, list] = {k: [] for k in ("choice", "planet", "reward", "win", "rare",
                                             "contingency", "scored", "rel", "mag",
                                             "points")}
    for _, g in groups:
        ok = g["choice"].notna().to_numpy()
        cols["choice"].append(np.where(ok, g["choice"].fillna(1).to_numpy() - 1, -1))
        cols["planet"].append(np.where(ok, g["planet"].fillna(1).to_numpy() - 1, -1))
        r = (g["reward_binary"] if reward == "binary" else g["reward_magnitude"] / 100)
        cols["reward"].append(r.fillna(0).to_numpy(float))
        cols["win"].append(g["reward_binary"].fillna(0).to_numpy(float))
        cols["points"].append((g["reward_magnitude"] / 100).fillna(0).to_numpy(float))
        cols["rare"].append(g["rare_transition"].to_numpy(int))
        cols["contingency"].append((g["reward_contingency"] == "state").to_numpy(int))
        fast = (g["response_time_seconds"] < min_rt).to_numpy() if min_rt else 0
        cols["scored"].append(ok & ~fast)
        cols["rel"].append(g["pad_transition_probability"].to_numpy(float)
                           if "pad_transition_probability" in g else np.full(len(g), 0.9))
        cols["mag"].append(g["magnitude_scale"].to_numpy(float)
                           if "magnitude_scale" in g else np.ones(len(g)))
    fills = {"choice": (-1, int), "planet": (-1, int), "reward": (0.0, float),
             "win": (0.0, float), "rare": (0, int), "contingency": (0, int),
             "scored": (False, bool), "rel": (0.9, float), "mag": (1.0, float),
             "points": (0.0, float)}
    arrays = {k: _pad(v, *fills[k]) for k, v in cols.items()}
    present = _pad([np.ones(len(v), bool) for v in cols["rel"]], False, bool)
    x_rel, x_high = lagged_conditions(arrays.pop("rel"), arrays.pop("mag"), present)
    return TwoStepData(**arrays, subject=np.array([k[0] for k, _ in groups]),
                       run=np.array([k[1] for k, _ in groups]), x_reliable=x_rel,
                       x_high=x_high)


def schedule_from_trials(trials: pd.DataFrame, runs=("run-1", "run-2")) -> Schedule:
    groups = _runs(trials, runs)
    pad_if, draws, cont, mag, resp, rel = [], [], [], [], [], []
    for _, g in groups:
        rel.append(g["pad_transition_probability"].to_numpy(float))
        pad_if.append(g[["pad_if_left", "pad_if_right"]].to_numpy(int) - 1)
        draws.append(g[[f"reward_draw_pad{k}" for k in range(1, 5)]].to_numpy(float))
        cont.append((g["reward_contingency"] == "state").to_numpy(int))
        mag.append(g["magnitude_scale"].to_numpy(float))
        resp.append(g["choice"].notna().to_numpy())
    present = _pad([np.ones(len(c), bool) for c in cont], False, bool)
    return Schedule(pad_if=_pad(pad_if, 0, int), draws=_pad(draws, 0.0, float),
                    contingency=_pad(cont, 0, int), magnitude=_pad(mag, 0.0, float),
                    present=present, responds=_pad(resp, False, bool) & present,
                    subject=np.array([k[0] for k, _ in groups]),
                    run=np.array([k[1] for k, _ in groups]), reliability=_pad(rel, 0.9, float))


def synthetic_schedule(n_runs: int = 2, n_trials: int = 77, seed: int = 0,
                       subject: str = "synthetic") -> Schedule:
    """A schedule with the verified *structure* of the ds007474 task (not its exact generator).

    One contingency switch per session at the midpoint; 70/30 ship -> planet transitions;
    26-trial blocks of pad reliability 0.5 / 0.9; reward-probability reversals every
    ~20 trials on a rare transition, each starting at 0/1 and settling at 0.3/0.5;
    magnitude scale in 26-trial blocks of U(0.3, 1) / U(0.1, 0.19). For tests and
    recovery studies only; recovery on real schedules uses :func:`schedule_from_trials`.
    """
    rng = np.random.default_rng(seed)
    T = n_runs * n_trials
    switch = T // 2 + rng.integers(-1, 2)
    first = rng.integers(0, 2)
    cont = np.where(np.arange(T) < switch, first, 1 - first)
    rare = rng.random(T) < 1 - P_COMMON
    block = (np.arange(T) // 26) % 2
    reliable = np.where(block == 0, 0.5, 0.9)
    common_pad = np.zeros((T, 2), int)
    for b in range(T // 26 + 1):
        common_pad[b * 26:(b + 1) * 26] = [rng.integers(0, 2), 2 + rng.integers(0, 2)]
    p = np.zeros((T, 2))
    anchor = rng.permutation([1.0, 0.0])
    since = 0
    for t in range(T):
        if since > 20 and rare[t]:
            anchor, since = anchor[::-1].copy(), 0
        settle = np.where(anchor == 1.0, 0.3, 0.5)
        p[t] = settle + (anchor - settle) * np.exp(-(since / 6.0) ** 3)
        since += 1
    draws = (rng.random((T, 4)) < np.repeat(p, 2, axis=1)).astype(float)
    pad_if = np.zeros((T, 2), int)
    for a in (0, 1):
        planet = np.where(rare, 1 - a, a)
        common = common_pad[np.arange(T), planet]
        other = np.where(planet == 0, 1 - common, 5 - common)
        pad_if[:, a] = np.where(rng.random(T) < reliable, common, other)
    mag_block = ((np.arange(T) + 13) // 26) % 2
    mag = np.where(mag_block == 0, rng.uniform(0.3, 1.0, T), rng.uniform(0.10, 0.19, T))
    shape = (n_runs, n_trials)
    return Schedule(pad_if=pad_if.reshape(shape + (2,)), draws=draws.reshape(shape + (4,)),
                    contingency=cont.reshape(shape), magnitude=mag.reshape(shape),
                    present=np.ones(shape, bool), responds=np.ones(shape, bool),
                    subject=np.array([subject] * n_runs),
                    run=np.array([f"run-{i + 1}" for i in range(n_runs)]),
                    reliability=reliable.reshape(shape))


def outcome(schedule: Schedule, t: int, choice: np.ndarray) -> tuple[np.ndarray, ...]:
    """Pad, planet and win for ``choice`` (0/1 per run) at trial ``t``: the task's rule."""
    idx = np.arange(len(choice))
    pad = schedule.pad_if[idx, t, choice]
    win_state = schedule.draws[idx, t, pad]
    win_stim = schedule.draws[idx, t, 2 * choice]  # pad 1 for left, pad 3 for right
    win = np.where(schedule.contingency[:, t] == 1, win_state, win_stim)
    return pad, pad // 2, win


# --------------------------------------------------------------------------- core
class _Agent:
    """Values and one-trial memory of ``n`` independent runs of one agent."""

    def __init__(self, params: dict, model: str, n: int):
        self.p, self.model = params, model
        self.magmf = "magmf" in model
        self.a_mf = params.get("alpha_mf", params.get("alpha"))
        self.a_mb = params.get("alpha_mb", params.get("alpha"))
        self.qmf, self.v = np.zeros((n, 2)), np.zeros((n, 2))
        self.prev = np.full(n, -1)  # last choice made
        self.prev_win = np.zeros(n)
        self.prev_rare = np.zeros(n, bool)

    def logits(self, contingency: np.ndarray, x_reliable=None, x_high=None) -> np.ndarray:
        """log P(right) - log P(left) for the current trial."""
        p, v = self.p, self.v
        qmb = P_COMMON * v + (1 - P_COMMON) * v[:, ::-1]
        w = _weights(p, self.model, contingency, x_reliable, x_high)[:, None]
        q = w * qmb + (1 - w) * self.qmf
        repeat = (self.prev == 1).astype(float) - (self.prev == 0).astype(float)
        dv = p["beta"] * (q[:, 1] - q[:, 0]) + p["stick"] * repeat
        if "wsls_mf" in p:
            win = np.where(self.prev_win > 0, 1.0, -1.0)
            mb_repeat = np.where((self.prev_win > 0) != self.prev_rare, 1.0, -1.0)
            dv = dv + repeat * (p["wsls_mf"] * win + p["wsls_mb"] * mb_repeat)
        return dv

    def update(self, choice, planet, reward, win, valid, points=None) -> None:
        idx = np.flatnonzero(valid)
        a, s, r = choice[idx], planet[idx], reward[idx]
        r_mf, r_mb = (points[idx], win[idx]) if self.magmf else (r, r)
        self.qmf[idx, a] += self.a_mf * (r_mf - self.qmf[idx, a])
        self.v[idx, s] += self.a_mb * (r_mb - self.v[idx, s])
        self.prev = np.where(valid, choice, self.prev)
        self.prev_win = np.where(valid, win, self.prev_win)
        self.prev_rare = np.where(valid, planet != choice, self.prev_rare)


def sequence_loglik(params: dict, model: str, data: TwoStepData) -> np.ndarray:
    """Log-likelihood of each run's scored choices."""
    n, T = data.choice.shape
    agent = _Agent(params, model, n)
    ll = np.zeros(n)
    for t in range(T):
        c = data.choice[:, t]
        dv = agent.logits(data.contingency[:, t], data.x_reliable[:, t], data.x_high[:, t])
        lp = np.where(c == 1, log_expit(dv), log_expit(-dv))
        ll += np.where(data.scored[:, t], lp, 0.0)
        agent.update(c, data.planet[:, t], data.reward[:, t], data.win[:, t], c >= 0,
                     data.points[:, t])
    return ll


def loglik(params: dict, model: str, data: TwoStepData) -> float:
    return float(sequence_loglik(params, model, data).sum())


def simulate(schedule: Schedule, model: str, params: dict, seed: int = 0,
             reward: str = "binary") -> TwoStepData:
    """Run an agent on a schedule (no-response trials of a real schedule are kept missing)."""
    rng = np.random.default_rng(seed)
    n, T = schedule.present.shape
    agent = _Agent(params, model, n)
    out = {k: np.zeros((n, T), int) for k in ("choice", "planet", "rare")}
    out.update({k: np.zeros((n, T)) for k in ("reward", "win", "points")})
    x_rel, x_high = lagged_conditions(schedule.reliability, schedule.magnitude,
                                      schedule.present)
    for t in range(T):
        dv = agent.logits(schedule.contingency[:, t], x_rel[:, t], x_high[:, t])
        c = (rng.random(n) < expit(dv)).astype(int)
        _, planet, win = outcome(schedule, t, c)
        pts = np.round(win * schedule.magnitude[:, t] * 100) / 100
        r = win if reward == "binary" else pts
        valid = schedule.responds[:, t]
        agent.update(c, planet, r, win, valid, pts)
        out["points"][:, t] = np.where(valid, pts, 0.0)
        out["choice"][:, t] = np.where(valid, c, -1)
        out["planet"][:, t] = np.where(valid, planet, -1)
        out["reward"][:, t] = np.where(valid, r, 0.0)
        out["win"][:, t] = np.where(valid, win, 0.0)
        out["rare"][:, t] = (planet != c).astype(int)
    return TwoStepData(**out, contingency=schedule.contingency.copy(),
                       scored=schedule.responds.copy(), subject=schedule.subject.copy(),
                       run=schedule.run.copy(), x_reliable=x_rel, x_high=x_high)


# --------------------------------------------------------------------------- fitting
def fit(data: TwoStepData, model: str, n_starts: int = 4, seed: int = 0,
        prior_sd=2.5, prior_mean=None) -> dict:
    """MAP fit (one parameter set for all runs in ``data``).

    The prior is independent Gaussian on the unconstrained parameters: mean ``prior_mean``
    (default 0) and SD ``prior_sd`` (scalar or one per parameter; falsy = no prior).
    """
    rng = np.random.default_rng(seed)
    k = len(MODELS[model])
    mu = np.zeros(k) if prior_mean is None else np.asarray(prior_mean, float)
    sd = None if prior_sd is None or np.all(np.asarray(prior_sd) == 0) else \
        np.broadcast_to(np.asarray(prior_sd, float), (k,))

    def objective(theta):
        val = -loglik(from_theta(theta, model), model, data)
        return val + 0.5 * float(np.sum(((theta - mu) / sd) ** 2)) if sd is not None else val

    best = None
    for i in range(n_starts):
        start = mu.copy() if i == 0 else mu + rng.normal(0, 1.0, k)
        res = minimize(objective, start, method="L-BFGS-B", bounds=[(-8, 8)] * k)
        if best is None or res.fun < best.fun:
            best = res
    params = from_theta(best.x, model)
    return {"model": model, "params": params, "loglik": loglik(params, model, data),
            "n_trials": data.n_scored, "n_params": k, "converged": bool(best.success),
            "theta": best.x}


# --------------------------------------------------------------------------- empirical Bayes
def _neg_log_posterior(theta, model, data, mu, sd) -> float:
    return (-loglik(from_theta(theta, model), model, data)
            + 0.5 * float(np.sum(((theta - mu) / sd) ** 2)))


def _hessian(f, x, h: float = 1e-3) -> np.ndarray:
    """Central finite-difference Hessian of a scalar function."""
    k = len(x)
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h ** 2
        for j in range(i):
            ej = np.zeros(k)
            ej[j] = h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                                 + f(x - ei - ej)) / (4 * h ** 2)
    return H


def _eb_subject(data, model, mu, sd, n_starts, seed):
    f = fit(data, model, n_starts=n_starts, seed=seed, prior_sd=sd, prior_mean=mu)
    theta = f["theta"]
    H = _hessian(lambda t: _neg_log_posterior(t, model, data, mu, sd), theta)
    try:
        var = np.diag(np.linalg.inv(H))
    except np.linalg.LinAlgError:
        var = np.full(len(theta), np.nan)
    # Fall back to the prior variance where the curvature is not positive (flat posterior).
    var = np.where(np.isfinite(var) & (var > 0), np.minimum(var, sd ** 2), sd ** 2)
    return theta, var


def fit_empirical_bayes(data: TwoStepData, model: str, n_iter: int = 30, tol: float = 1e-2,
                        n_starts: int = 2, seed: int = 0, init_sd: float = 2.5,
                        min_sd: float = 0.05, n_jobs: int = 1) -> dict:
    """Hierarchical fit by empirical Bayes (EM with a Laplace approximation; Huys et al. 2011).

    The group prior is independent Gaussian on the unconstrained parameters, N(mu, sd^2),
    starting from N(0, ``init_sd``^2) (the fixed prior of :func:`fit`). Each iteration
    fits every subject's MAP under the current prior, takes the posterior variance from
    the inverse Hessian, and sets ``mu = mean(theta)``, ``sd^2 = mean(theta^2 + var) -
    mu^2`` (floored at ``min_sd``). Stops when mu and sd change by less than ``tol``.
    Returns the group prior, per-subject parameters under the final prior, and the trace.
    """
    from joblib import Parallel, delayed

    subs = _subject_rows(data)
    k = len(MODELS[model])
    mu, sd = np.zeros(k), np.full(k, float(init_sd))
    trace, converged = [], False
    for it in range(n_iter):
        out = Parallel(n_jobs=n_jobs)(
            delayed(_eb_subject)(data.subset(r), model, mu, sd, n_starts, seed)
            for r in subs.values())
        thetas = np.array([o[0] for o in out])
        var = np.array([o[1] for o in out])
        new_mu = thetas.mean(axis=0)
        new_sd = np.sqrt(np.maximum((thetas ** 2 + var).mean(axis=0) - new_mu ** 2,
                                    min_sd ** 2))
        change = float(max(np.abs(new_mu - mu).max(), np.abs(new_sd - sd).max()))
        mu, sd = new_mu, new_sd
        trace.append({"iteration": it, "max_change": change})
        if change < tol:
            converged = True
            break
    final = Parallel(n_jobs=n_jobs)(
        delayed(fit)(data.subset(r), model, n_starts, seed, sd, mu) for r in subs.values())
    fits = pd.DataFrame([{"subject": s, "model": model, **f["params"]}
                         for s, f in zip(subs, final, strict=True)])
    return {"model": model, "prior_mean": mu.tolist(), "prior_sd": sd.tolist(),
            "parameters": list(MODELS[model]), "n_iter": len(trace),
            "converged": converged, "trace": trace, "fits": fits}


def heldout_runs_eb(data: TwoStepData, model: str, **eb_kw) -> pd.DataFrame:
    """Leave-one-run-out with an empirical-Bayes prior estimated on the training runs only.

    For each run label (e.g. run-1), the group prior and every subject's parameters are
    fitted on all subjects' *other* runs and score that subject's held-out run, so the
    held-out data never inform the prior.
    """
    rows = []
    for held in dict.fromkeys(data.run):
        train = data.subset(np.flatnonzero(data.run != held))
        test = data.subset(np.flatnonzero(data.run == held))
        eb = fit_empirical_bayes(train, model, **eb_kw)
        for _, f in eb["fits"].iterrows():
            m = test.subject == f["subject"]
            if not m.any():
                continue
            params = {k: float(f[k]) for k in MODELS[model]}
            sub = test.subset(np.flatnonzero(m))
            rows.append({"subject": f["subject"], "model": model, "heldout_run": held,
                         "heldout_loglik": loglik(params, model, sub),
                         "heldout_trials": sub.n_scored})
    return pd.DataFrame(rows)


def heldout_runs(data: TwoStepData, model: str, **fit_kw) -> dict:
    """Leave-one-run-out: fit on the other runs of the same subject, score the held-out run."""
    total, n = 0.0, 0
    for i in range(len(data.run)):
        train = data.subset([j for j in range(len(data.run)) if j != i])
        test = data.subset([i])
        params = fit(train, model, **fit_kw)["params"]
        total += loglik(params, model, test)
        n += test.n_scored
    return {"heldout_loglik": total, "heldout_trials": n}


def _subject_rows(data: TwoStepData) -> dict[str, np.ndarray]:
    return {s: np.flatnonzero(data.subject == s) for s in dict.fromkeys(data.subject)}


def _fit_subject(data: TwoStepData, models, n_starts, seed, prior_sd) -> list[dict]:
    rows = []
    for m in models:
        full = fit(data, m, n_starts=n_starts, seed=seed, prior_sd=prior_sd)
        ho = heldout_runs(data, m, n_starts=n_starts, seed=seed, prior_sd=prior_sd)
        rows.append({"subject": data.subject[0], "model": m, **full["params"],
                     "loglik": full["loglik"], "n_trials": full["n_trials"], **ho})
    return rows


def fit_subjects(data: TwoStepData, models, n_starts: int = 4, seed: int = 0,
                 prior_sd: float = 2.5, n_jobs: int = 1) -> pd.DataFrame:
    """Per-subject fits plus leave-one-run-out held-out log-likelihood, for every model."""
    from joblib import Parallel, delayed

    subs = _subject_rows(data)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_subject)(data.subset(r), models, n_starts, seed, prior_sd)
        for r in subs.values())
    return pd.DataFrame([row for rows in out for row in rows])


def heldout_subjects(data: TwoStepData, model: str, n_folds: int = 5, seed: int = 0,
                     n_starts: int = 4, prior_sd: float = 2.5) -> pd.DataFrame:
    """Subject-level k-fold: one pooled parameter set fitted on training subjects is
    scored on the held-out subjects (no subject is in both)."""
    subs = list(_subject_rows(data))
    order = np.random.default_rng(seed).permutation(len(subs))
    folds = np.array_split(np.array(subs)[order], n_folds)
    rows = []
    for k, test_subs in enumerate(folds):
        test = np.isin(data.subject, test_subs)
        params = fit(data.subset(np.flatnonzero(~test)), model, n_starts=n_starts,
                     seed=seed, prior_sd=prior_sd)["params"]
        held = data.subset(np.flatnonzero(test))
        ll = sequence_loglik(params, model, held)
        for s in test_subs:
            m = held.subject == s
            rows.append({"subject": s, "model": model, "fold": k,
                         "heldout_loglik": float(ll[m].sum()),
                         "heldout_trials": int(held.scored[m].sum()), **params})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summaries
def stay_table(data: TwoStepData) -> pd.DataFrame:
    """P(repeat previous choice) by previous win x previous transition, per contingency.

    Only consecutive responded trials within a run; the contingency is the previous
    trial's (the one whose outcome is being reacted to).
    """
    c, prev = data.choice[:, 1:], data.choice[:, :-1]
    ok = (c >= 0) & (prev >= 0)
    df = pd.DataFrame({
        "subject": np.repeat(data.subject, c.shape[1])[ok.ravel()],
        "contingency": np.where(data.contingency[:, :-1] == 1, "state", "stimulus")[ok],
        "prev_win": data.win[:, :-1][ok].astype(int),
        "prev_rare": data.rare[:, :-1][ok].astype(int),
        "stay": (c == prev)[ok].astype(int)})
    return df


def stay_summary(data: TwoStepData) -> pd.DataFrame:
    """Group mean of per-subject stay probabilities and the reward x transition indices."""
    per = (stay_table(data).groupby(["subject", "contingency", "prev_win", "prev_rare"])
           ["stay"].mean().unstack(["prev_win", "prev_rare"]))
    out = pd.DataFrame({
        "rewarded_common": per[(1, 0)], "rewarded_rare": per[(1, 1)],
        "unrewarded_common": per[(0, 0)], "unrewarded_rare": per[(0, 1)]})
    out["reward_effect"] = (out.rewarded_common + out.rewarded_rare
                            - out.unrewarded_common - out.unrewarded_rare) / 2
    out["interaction"] = (out.rewarded_common - out.rewarded_rare
                          - out.unrewarded_common + out.unrewarded_rare) / 2
    return out.groupby("contingency").mean()


def sample_params(model: str, rng: np.random.Generator, ranges: dict | None = None) -> dict:
    r = {**RECOVERY_RANGES, **(ranges or {})}
    return {k: float(rng.uniform(*r[k])) for k in MODELS[model]}


def _recover_one(model, i, seed, schedules, n_starts, prior_sd, ranges, reward):
    rng = np.random.default_rng([seed, i])
    sched = schedules[i % len(schedules)]
    true = sample_params(model, rng, ranges)
    data = simulate(sched, model, true, seed=int(rng.integers(2**31)), reward=reward)
    est = fit(data, model, n_starts=n_starts, seed=seed, prior_sd=prior_sd)["params"]
    return {"model": model, "sim": i, **{f"true_{k}": v for k, v in true.items()},
            **{f"est_{k}": v for k, v in est.items()}}


def parameter_recovery(model: str, schedules: list[Schedule], n_sims: int = 50,
                       seed: int = 0, n_starts: int = 4, prior_sd: float = 2.5,
                       ranges: dict | None = None, reward: str = "binary",
                       n_jobs: int = 1) -> pd.DataFrame:
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_recover_one)(model, i, seed, schedules, n_starts, prior_sd, ranges, reward)
        for i in range(n_sims))
    return pd.DataFrame(rows)


def summarize_recovery(df: pd.DataFrame, model: str) -> dict:
    out = {}
    for k in MODELS[model]:
        t, e = df[f"true_{k}"], df[f"est_{k}"]
        out[k] = {"pearson_r": float(np.corrcoef(t, e)[0, 1]),
                  "median_abs_error": float((t - e).abs().median())}
    return out


def _model_recovery_one(gen, i, seed, schedules, models, n_starts, prior_sd, ranges, reward):
    rng = np.random.default_rng([seed, 1000 + i, len(gen)])
    sched = schedules[i % len(schedules)]
    data = simulate(sched, gen, sample_params(gen, rng, ranges),
                    seed=int(rng.integers(2**31)), reward=reward)
    ho = {m: heldout_runs(data, m, n_starts=n_starts, seed=seed, prior_sd=prior_sd)
          ["heldout_loglik"] for m in models}
    return {"generating": gen, "sim": i, "selected": max(ho, key=ho.get),
            **{f"heldout_{m}": v for m, v in ho.items()}}


def model_recovery(models, schedules: list[Schedule], n_sims: int = 30, seed: int = 0,
                   n_starts: int = 3, prior_sd: float = 2.5, ranges: dict | None = None,
                   reward: str = "binary", n_jobs: int = 1) -> pd.DataFrame:
    """Simulate from each model; select by leave-one-run-out held-out log-likelihood."""
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_model_recovery_one)(g, i, seed, schedules, models, n_starts, prior_sd,
                                     ranges, reward)
        for g in models for i in range(n_sims))
    return pd.DataFrame(rows, columns=["generating", "sim", "selected"]) if not rows \
        else pd.DataFrame(rows)


def group_model_recovery(conf: pd.DataFrame, models, group_size: int = 10,
                         seed: int = 0) -> pd.DataFrame:
    """Group-level selection from :func:`model_recovery` output: simulated subjects of each
    generating model are split into *disjoint* groups of ``group_size``; each group picks
    the model with the highest summed held-out log-likelihood."""
    rng = np.random.default_rng(seed)
    rows = []
    for gen, g in conf.groupby("generating"):
        order = rng.permutation(len(g))
        for k in range(len(g) // group_size):
            part = g.iloc[order[k * group_size:(k + 1) * group_size]]
            totals = {m: part[f"heldout_{m}"].sum() for m in models}
            rows.append({"generating": gen, "group": k, "selected": max(totals, key=totals.get)})
    return pd.DataFrame(rows, columns=["generating", "group", "selected"])


# --------------------------------------------------------------------------- real-data run
def missed_fraction(trials: pd.DataFrame, runs=("run-1", "run-2")) -> pd.Series:
    t = trials[trials["run"].isin(runs)]
    return 1 - t.groupby("subject")["valid_trial"].mean()


def posterior_predictive(data: TwoStepData, schedule: Schedule, fits: pd.DataFrame,
                         model: str, n_rep: int = 20, seed: int = 0,
                         reward: str = "binary") -> pd.DataFrame:
    """Stay-probability summary of each subject's fitted agent replayed on that subject's
    real schedule (``n_rep`` replays), next to the observed summary."""
    sims = []
    for i, (s, rows) in enumerate(_subject_rows(schedule).items()):
        p = fits[(fits["subject"] == s) & (fits["model"] == model)]
        params = {k: float(p.iloc[0][k]) for k in MODELS[model]}
        sim = simulate(schedule.subset(np.tile(rows, n_rep)), model, params,
                       seed=seed + i, reward=reward)
        sim.subject = np.array([f"{s}|{r}" for r in range(n_rep) for _ in rows])
        sims.append(sim)
    sim_all = TwoStepData(**{k: np.concatenate([getattr(x, k) for x in sims])
                             for k in TwoStepData.__dataclass_fields__})
    out = pd.concat({"observed": stay_summary(data), model: stay_summary(sim_all)},
                    names=["source"])
    return out.reset_index()


def contingency_contrast(fits: pd.DataFrame, trials: pd.DataFrame, n_boot: int = 2000,
                         seed: int = 0, first_run: str = "run-1", runs=None) -> dict:
    """Order-balanced estimate of ``w_state - w_stimulus`` (``hybrid_contingency`` fits).

    The contingency switches once, at the run boundary, so contingency is confounded with
    time (run 1 vs run 2). The starting contingency is counterbalanced across subjects:
    with ``d`` the per-subject difference, stimulus-first subjects give
    ``d = contingency + time`` and state-first subjects ``d = contingency - time``. The
    contingency effect is the mean of the two group means; the time (late minus early)
    effect is half their difference. CIs resample subjects within order groups.
    Subjects who never experienced both contingencies in the analysed runs (``runs``;
    default: the functional runs, or ``first_run`` alone) are dropped: one of their
    weights is set by the prior only.
    """
    from scipy.stats import wilcoxon

    f = fits[fits["model"] == "hybrid_contingency"].set_index("subject")
    d = f["w_state"] - f["w_stimulus"]
    first = (trials[(trials["run"] == first_run) & (trials["trial_index"] == 0)]
             .set_index("subject")["reward_contingency"].reindex(d.index))
    runs = runs or (("run-1", "run-2") if first_run == "run-1" else (first_run,))
    t = trials[trials["run"].isin(runs)]
    both = t.groupby("subject")["reward_contingency"].nunique().reindex(d.index) == 2
    keep = first.notna() & d.notna() & both
    dropped = sorted(d.index[~keep])
    d, first = d[keep], first[keep]
    groups = {g: d[first == g].to_numpy() for g in ("stimulus", "state")}
    if min(len(x) for x in groups.values()) < 2:
        return {"excluded_single_contingency": dropped, "insufficient_subjects": True,
                "n_by_first_contingency": {g: len(x) for g, x in groups.items()}}
    rng = np.random.default_rng(seed)

    def effects(gs):
        a, b = gs["stimulus"].mean(), gs["state"].mean()
        return (a + b) / 2, (a - b) / 2

    boot = np.array([effects({g: x[rng.integers(0, len(x), len(x))] for g, x in groups.items()})
                     for _ in range(n_boot)])
    cont, time = effects(groups)
    return {
        "excluded_single_contingency": dropped,
        "by_first_contingency": {
            g: {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                "wilcoxon_p": float(wilcoxon(x).pvalue)} for g, x in groups.items()},
        "contingency_effect_w_state_minus_w_stimulus": float(cont),
        "contingency_effect_ci95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
        "time_effect_late_minus_early": float(time),
        "time_effect_ci95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
    }


def _contrast_one(sched, i, effect, seed, n_starts, prior_sd, ranges, reward):
    rng = np.random.default_rng([seed, 5000 + i])
    p = sample_params("hybrid_contingency", rng, ranges)
    lo, hi = (ranges or {}).get("w_state", RECOVERY_RANGES["w_state"])
    p["w_stimulus"] = float(rng.uniform(lo, hi - max(effect, 0)))
    p["w_state"] = p["w_stimulus"] + effect
    data = simulate(sched, "hybrid_contingency", p, seed=int(rng.integers(2**31)),
                    reward=reward)
    est = fit(data, "hybrid_contingency", n_starts=n_starts, seed=seed, prior_sd=prior_sd)
    label = {0: "stimulus", 1: "state"}
    rows = [{"subject": f"sim{i}", "run": sched.run[j], "trial_index": 0 if k == 0 else -1,
             "reward_contingency": label[int(c)]}
            for j in range(len(sched.run))
            for k, c in enumerate(dict.fromkeys(sched.contingency[j][sched.present[j]]))]
    return ({"subject": f"sim{i}", "model": "hybrid_contingency", **est["params"]}, rows)


def contingency_effect_recovery(schedules: list[Schedule], effect: float, seed: int = 0,
                                n_starts: int = 3, prior_sd: float = 2.5,
                                ranges: dict | None = None, reward: str = "binary",
                                n_boot: int = 2000, n_jobs: int = 1) -> dict:
    """Known-answer check of :func:`contingency_contrast`: one simulated subject per
    schedule with ``w_state - w_stimulus = effect``, refitted and contrasted."""
    from joblib import Parallel, delayed

    out = Parallel(n_jobs=n_jobs)(
        delayed(_contrast_one)(s, i, effect, seed, n_starts, prior_sd, ranges, reward)
        for i, s in enumerate(schedules))
    fits = pd.DataFrame([a for a, _ in out])
    trials = pd.DataFrame([r for _, rows in out for r in rows])
    runs = tuple(schedules[0].run)
    return {"true_effect": effect, "n_subjects": len(schedules),
            **contingency_contrast(fits, trials, n_boot, seed, first_run=runs[0], runs=runs)}


ARBITRATION_TERMS = ("arb_reliable", "arb_high")


def arbitration_summary(fits: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                        model: str = "hybrid_wsls_arbitration") -> dict | None:
    """Group summary of per-subject arbitration weights (logit units of the MB weight).

    For each term: mean with a subject-bootstrap 95% CI, median, number of subjects > 0,
    two-sided Wilcoxon signed-rank test against 0, and Holm over the terms. MAP estimates
    are shrunk towards 0 by the prior, so the means are conservative.
    """
    from scipy.stats import wilcoxon

    f = fits[fits["model"] == model]
    if f.empty:
        return None
    rng = np.random.default_rng(seed)
    out, pvals = {}, []
    for k in ARBITRATION_TERMS:
        x = f[k].to_numpy(float)
        boot = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
        p = float(wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else float("nan")
        pvals.append(p)
        out[k] = {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                  "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                  "n_positive": int((x > 0).sum()), "wilcoxon_p": p}
    order = np.argsort(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[ARBITRATION_TERMS[i]]["wilcoxon_p_holm"] = running
    return out


def _arbitration_one(sched, i, effects, seed, n_starts, prior_sd, ranges, reward,
                     model="hybrid_wsls_arbitration"):
    rng = np.random.default_rng([seed, 7000 + i])
    p = {**sample_params(model, rng, ranges), **effects}
    data = simulate(sched, model, p, seed=int(rng.integers(2**31)), reward=reward)
    est = fit(data, model, n_starts=n_starts, seed=seed, prior_sd=prior_sd)
    return {"subject": f"sim{i}", "model": model, **est["params"]}


def arbitration_effect_recovery(schedules: list[Schedule], effects: dict, seed: int = 0,
                                n_starts: int = 3, prior_sd: float = 2.5,
                                ranges: dict | None = None, reward: str = "binary",
                                n_boot: int = 2000, n_jobs: int = 1,
                                model: str = "hybrid_wsls_arbitration") -> dict:
    """Known-answer check of :func:`arbitration_summary`: one simulated subject per schedule,
    every subject with the planted arbitration weights ``effects`` (other parameters drawn
    from ``ranges``), refitted and summarised."""
    from joblib import Parallel, delayed

    fits = pd.DataFrame(Parallel(n_jobs=n_jobs)(
        delayed(_arbitration_one)(s, i, effects, seed, n_starts, prior_sd, ranges, reward,
                                  model)
        for i, s in enumerate(schedules)))
    return {"model": model, "true_effects": effects, "n_subjects": len(schedules),
            **arbitration_summary(fits, n_boot, seed, model)}


def _arbitration_null_rep(fits, schedule, rep, seed, n_starts, prior_sd, reward,
                          generator="hybrid_wsls", fitted="hybrid_wsls_arbitration"):
    rows = []
    for i, (s, r) in enumerate(_subject_rows(schedule).items()):
        p = fits[(fits["subject"] == s) & (fits["model"] == generator)]
        params = {k: float(p.iloc[0][k]) for k in MODELS[generator]}
        data = simulate(schedule.subset(r), generator, params,
                        seed=int(np.random.default_rng([seed, rep, i]).integers(2**31)),
                        reward=reward)
        est = fit(data, fitted, n_starts=n_starts, seed=seed, prior_sd=prior_sd)["params"]
        rows.append([est[k] for k in ARBITRATION_TERMS])
    return np.mean(rows, axis=0)


def arbitration_null(fits: pd.DataFrame, schedule: Schedule, n_reps: int = 200,
                     seed: int = 0, n_starts: int = 3, prior_sd: float = 2.5,
                     reward: str = "binary", n_jobs: int = 1, generator: str = "hybrid_wsls",
                     fitted: str = "hybrid_wsls_arbitration") -> np.ndarray:
    """Parametric-bootstrap null for the group-mean arbitration weights.

    Each replicate simulates every subject from their own fitted ``generator`` (a model
    without arbitration) on their own schedule, refits ``fitted`` and records the group
    means. Returns ``(n_reps, 2)`` (columns: :data:`ARBITRATION_TERMS`). This
    calibrates the group test for any bias of the estimator on the real schedules.
    """
    from joblib import Parallel, delayed

    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_arbitration_null_rep)(fits, schedule, k, seed, n_starts, prior_sd, reward,
                                       generator, fitted)
        for k in range(n_reps)))


def calibrated_arbitration_test(summary: dict, null: np.ndarray) -> dict:
    """Two-sided test of each observed group mean against the bootstrap null: empirical
    p = (1 + #|null - null mean| >= |obs - null mean|) / (1 + n), Holm over the terms."""
    out, pvals = {}, []
    for j, k in enumerate(ARBITRATION_TERMS):
        x = null[:, j]
        obs = summary[k]["mean"]
        dev = np.abs(x - x.mean())
        p = float((1 + np.sum(dev >= abs(obs - x.mean()))) / (1 + len(x)))
        pvals.append(p)
        out[k] = {"observed_mean": obs, "null_mean": float(x.mean()), "null_sd": float(x.std()),
                  "null_95": np.percentile(x, [2.5, 97.5]).tolist(),
                  "bias_corrected_mean": float(obs - x.mean()), "p_calibrated": p}
    order = np.argsort(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[ARBITRATION_TERMS[i]]["p_calibrated_holm"] = running
    return {"n_reps": len(null), **out}


def _delta_summary(wide: pd.DataFrame, n: pd.Series, models, reference, n_boot, rng) -> dict:
    out = {}
    for m in models:
        d = (wide[m] - wide[reference]).to_numpy()
        boot = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(n_boot)]
        out[m] = {
            "heldout_loglik_per_trial": float(wide[m].sum() / n.sum()),
            "delta_vs_reference": float(d.sum()),
            "delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "subjects_better_than_reference": int((d > 0).sum()),
            "subjects_best": int((wide.idxmax(axis=1) == m).sum()),
        }
    return out


def session_heldout(fits: pd.DataFrame, data: TwoStepData, models) -> pd.DataFrame:
    """Score another session's trials with each subject's parameters from ``fits``."""
    rows = []
    for s, r in _subject_rows(data).items():
        sub = data.subset(r)
        for m in models:
            p = fits[(fits["subject"] == s) & (fits["model"] == m)]
            if p.empty:
                continue
            params = {k: float(p.iloc[0][k]) for k in MODELS[m]}
            rows.append({"subject": s, "model": m, "heldout_loglik": loglik(params, m, sub),
                         "heldout_trials": sub.n_scored})
    return pd.DataFrame(rows)


def _fit_contingency_only(data, n_starts, seed, prior_sd, model="hybrid_contingency"):
    f = fit(data, model, n_starts=n_starts, seed=seed, prior_sd=prior_sd)
    return {"subject": data.subject[0], "model": model, **f["params"]}


def _calibrated(fits, sched, models, n_reps, seed, n_starts, prior_sd, reward, n_jobs,
                n_boot, generator="hybrid_wsls", fitted="hybrid_wsls_arbitration"):
    if not n_reps or not {generator, fitted} <= set(models):
        return None
    null = arbitration_null(fits, sched, n_reps, seed, n_starts, prior_sd, reward, n_jobs,
                            generator, fitted)
    return {"generator": generator, "fitted": fitted,
            **calibrated_arbitration_test(arbitration_summary(fits, n_boot, seed, fitted), null)}


def compare(trials: pd.DataFrame, models=tuple(MODELS), reference: str = "hybrid",
            reward: str = "binary", min_rt: float | None = 0.15,
            max_missed_fraction: float | None = 0.10, exclude_subjects=(),
            n_starts: int = 4, prior_sd: float = 2.5, n_folds: int = 5, n_boot: int = 2000,
            n_rep: int = 20, seed: int = 0, n_jobs: int = 1,
            replication_run: str | None = None, arbitration_null_reps: int = 0,
            arbitration_tests=()) -> dict:
    """Fit every model per subject; compare by held-out log-likelihood (runs and subjects).

    With ``replication_run`` (e.g. ``"structural"``), the parameters fitted on the
    functional runs also score that separate session, and the contingency contrast is
    repeated on ``hybrid_contingency`` fits to that session alone.

    ``arbitration_tests``: list of {generator, fitted, reps}: calibrated tests of the
    ``fitted`` model's group arbitration weights against a parametric-bootstrap null
    generated from each subject's fitted ``generator`` (:func:`arbitration_null`).
    ``arbitration_null_reps`` is the shorthand for generator ``hybrid_wsls``, fitted
    ``hybrid_wsls_arbitration``.
    """
    miss = missed_fraction(trials)
    excluded = sorted(set(exclude_subjects)
                      | (set(miss[miss > max_missed_fraction].index)
                         if max_missed_fraction is not None else set()))
    keep = trials[~trials["subject"].isin(excluded)]
    data = prepare(keep, reward=reward, min_rt=min_rt)
    sched = schedule_from_trials(keep)
    fits = fit_subjects(data, models, n_starts=n_starts, seed=seed, prior_sd=prior_sd,
                        n_jobs=n_jobs)

    rng = np.random.default_rng(seed)
    wide = fits.pivot(index="subject", columns="model", values="heldout_loglik")
    n = fits.groupby("subject")["heldout_trials"].first()
    summary = _delta_summary(wide, n, models, reference, n_boot, rng)

    from joblib import Parallel, delayed

    replication = None
    if replication_run:
        rep_data = prepare(keep, reward=reward, min_rt=min_rt, runs=(replication_run,))
        ho = session_heldout(fits, rep_data, models)
        rw = ho.pivot(index="subject", columns="model", values="heldout_loglik")
        rn = ho.groupby("subject")["heldout_trials"].first()
        rep_fits = pd.DataFrame(Parallel(n_jobs=n_jobs)(
            delayed(_fit_contingency_only)(rep_data.subset(r), n_starts, seed, prior_sd)
            for r in _subject_rows(rep_data).values()))
        rep_arbitration = {}
        for am in [m for m in models if m.endswith("_arbitration")]:
            arb_fits = pd.DataFrame(Parallel(n_jobs=n_jobs)(
                delayed(_fit_contingency_only)(rep_data.subset(r), n_starts, seed, prior_sd, am)
                for r in _subject_rows(rep_data).values()))
            rep_arbitration[am] = arbitration_summary(arb_fits, n_boot, seed, am)
        rep_arbitration = (rep_arbitration.get("hybrid_wsls_arbitration") if
                           list(rep_arbitration) == ["hybrid_wsls_arbitration"]
                           else rep_arbitration or None)
        replication = {
            "arbitration": rep_arbitration,
            "run": replication_run, "n_subjects": len(rw), "n_scored_trials": rep_data.n_scored,
            "session_heldout": _delta_summary(rw, rn, models, reference, n_boot, rng),
            "contingency_contrast": contingency_contrast(rep_fits, keep, n_boot, seed,
                                                         first_run=replication_run),
            "stay_probabilities_observed": stay_summary(rep_data).reset_index()
            .to_dict(orient="records"),
        }
    pooled = Parallel(n_jobs=min(n_jobs, len(models)))(
        delayed(heldout_subjects)(data, m, n_folds, seed, n_starts, prior_sd) for m in models)
    pooled = pd.concat(pooled)
    pooled_summary = {m: float(g["heldout_loglik"].sum() / g["heldout_trials"].sum())
                      for m, g in pooled.groupby("model")}
    ppc = pd.concat([posterior_predictive(data, sched, fits, m, n_rep, seed, reward)
                     for m in models])
    ppc = ppc.drop_duplicates(subset=["source", "contingency"])
    params = fits.drop(columns=["subject"]).groupby("model")
    contrast = (contingency_contrast(fits, keep, n_boot, seed)
                if "hybrid_contingency" in models else None)
    return {
        "n_subjects": len(wide), "excluded_subjects": excluded,
        "missed_fraction_threshold": max_missed_fraction, "n_scored_trials": data.n_scored,
        "chance_loglik_per_trial": float(np.log(0.5)),
        "run_heldout": summary,
        "subject_heldout_pooled_loglik_per_trial": pooled_summary,
        "parameter_medians": {m: g[list(MODELS[m])].median().to_dict() for m, g in params},
        "stay_probabilities": ppc.to_dict(orient="records"),
        "contingency_contrast": contrast,
        "arbitration": {am: arbitration_summary(fits, n_boot, seed, am)
                        for am in models if am.endswith("_arbitration")} or None,
        "arbitration_calibrated": _calibrated(fits, sched, models, arbitration_null_reps, seed,
                                              n_starts, prior_sd, reward, n_jobs, n_boot),
        "arbitration_tests": [_calibrated(fits, sched, models, t["reps"], seed, n_starts,
                                          prior_sd, reward, n_jobs, n_boot, t["generator"],
                                          t["fitted"]) for t in arbitration_tests],
        "replication": replication,
        "fits": fits,
    }


# --------------------------------------------------------------------------- EB studies
def _simulate_population(model, schedules, seed, ranges, reward):
    parts, true = [], []
    for i, sched in enumerate(schedules):
        rng = np.random.default_rng([seed, 9000 + i])
        p = sample_params(model, rng, ranges)
        sim = simulate(sched, model, p, seed=int(rng.integers(2**31)), reward=reward)
        sim.subject = np.array([f"sim{i}"] * len(sim.subject))
        parts.append(sim)
        true.append({"subject": f"sim{i}", **p})
    data = TwoStepData(**{k: np.concatenate([getattr(x, k) for x in parts])
                          for k in TwoStepData.__dataclass_fields__})
    return data, pd.DataFrame(true)


def empirical_bayes_recovery(model: str, schedules: list[Schedule], seed: int = 0,
                             ranges: dict | None = None, reward: str = "binary",
                             n_starts: int = 2, prior_sd: float = 2.5, n_jobs: int = 1,
                             **eb_kw) -> dict:
    """Parameter recovery of fixed-prior MAP vs empirical Bayes on one simulated population
    (one subject per schedule, parameters drawn from ``ranges``)."""
    data, true = _simulate_population(model, schedules, seed, ranges, reward)
    fixed = fit_subjects(data, [model], n_starts=n_starts, seed=seed, prior_sd=prior_sd,
                         n_jobs=n_jobs).set_index("subject")
    eb = fit_empirical_bayes(data, model, n_starts=n_starts, seed=seed, n_jobs=n_jobs,
                             **eb_kw)
    est = eb["fits"].set_index("subject")
    t = true.set_index("subject")
    out = {}
    for k in MODELS[model]:
        out[k] = {"r_fixed": float(np.corrcoef(t[k], fixed.loc[t.index, k])[0, 1]),
                  "r_eb": float(np.corrcoef(t[k], est.loc[t.index, k])[0, 1]),
                  "mae_fixed": float((t[k] - fixed.loc[t.index, k]).abs().median()),
                  "mae_eb": float((t[k] - est.loc[t.index, k]).abs().median())}
    return {"model": model, "n_subjects": len(t), "eb_converged": eb["converged"],
            "eb_n_iter": eb["n_iter"], "prior_mean": eb["prior_mean"],
            "prior_sd": eb["prior_sd"], "parameters": out}


def compare_empirical_bayes(trials: pd.DataFrame, models, min_rt: float | None = 0.15,
                            max_missed_fraction: float | None = 0.10, n_starts: int = 2,
                            prior_sd: float = 2.5, seed: int = 0, n_boot: int = 2000,
                            n_jobs: int = 1, replication_run: str | None = "structural",
                            **eb_kw) -> dict:
    """Fixed-prior MAP vs empirical Bayes on real data, per model.

    Held-out runs: both methods leave one run out (EB estimates its prior on the
    training runs only, :func:`heldout_runs_eb`). Other session: parameters fitted on
    the functional runs (fixed prior, or EB with the prior from all functional data)
    score ``replication_run``. Deltas are EB minus fixed, summed over subjects, with a
    subject-bootstrap CI.
    """
    miss = missed_fraction(trials)
    excluded = sorted(miss[miss > max_missed_fraction].index) if max_missed_fraction else []
    keep = trials[~trials["subject"].isin(excluded)]
    data = prepare(keep, min_rt=min_rt)
    rep_data = prepare(keep, min_rt=min_rt, runs=(replication_run,)) if replication_run \
        else None
    rng = np.random.default_rng(seed)

    def delta(a, b):
        d = (a - b).to_numpy()
        boot = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(n_boot)]
        return {"delta_eb_minus_fixed": float(d.sum()),
                "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                "subjects_better_with_eb": int((d > 0).sum())}

    out = {"n_subjects": int(len(set(data.subject))), "excluded_subjects": excluded,
           "models": {}}
    for m in models:
        fixed = fit_subjects(data, [m], n_starts=n_starts, seed=seed, prior_sd=prior_sd,
                             n_jobs=n_jobs)
        fx_runs = fixed.set_index("subject")["heldout_loglik"]
        eb_runs = (heldout_runs_eb(data, m, n_starts=n_starts, seed=seed, n_jobs=n_jobs,
                                   **eb_kw).groupby("subject")["heldout_loglik"].sum())
        n_runs = int(fixed["heldout_trials"].sum())
        eb = fit_empirical_bayes(data, m, n_starts=n_starts, seed=seed, n_jobs=n_jobs, **eb_kw)
        res = {"prior_mean": dict(zip(MODELS[m], eb["prior_mean"], strict=True)),
               "prior_sd": dict(zip(MODELS[m], eb["prior_sd"], strict=True)),
               "eb_converged": eb["converged"], "eb_n_iter": eb["n_iter"],
               "medians_fixed": fixed[list(MODELS[m])].median().to_dict(),
               "medians_eb": eb["fits"][list(MODELS[m])].median().to_dict(),
               "heldout_runs": {"per_trial_fixed": float(fx_runs.sum() / n_runs),
                                "per_trial_eb": float(eb_runs.sum() / n_runs),
                                **delta(eb_runs.reindex(fx_runs.index), fx_runs)}}
        if rep_data is not None:
            fx_s = session_heldout(fixed, rep_data, [m]).set_index("subject")["heldout_loglik"]
            eb_s = session_heldout(eb["fits"], rep_data, [m]).set_index("subject")
            n_s = int(eb_s["heldout_trials"].sum())
            eb_s = eb_s["heldout_loglik"]
            res["other_session"] = {"per_trial_fixed": float(fx_s.sum() / n_s),
                                    "per_trial_eb": float(eb_s.sum() / n_s),
                                    **delta(eb_s.reindex(fx_s.index), fx_s)}
        if m.endswith("_arbitration"):
            res["arbitration_eb_group_mean"] = {k: res["prior_mean"][k]
                                                for k in ARBITRATION_TERMS}
        out["models"][m] = res
    return out
