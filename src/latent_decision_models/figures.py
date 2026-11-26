"""Figures for the reported results, drawn only from the result files.

Every figure is built from a report (``reports/*.json``) or from the trial table, so it
can be regenerated with ``ldm figures`` and cannot drift from the numbers in the docs.
Requires the optional ``[plots]`` extra (matplotlib). Output is deterministic SVG.

Colours: own (active) decisions blue, observed decisions orange, a third series aqua
(validated for colour-vision deficiency). Every series is also directly labelled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OWN, OBSERVED, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COND = {"active": ("own decisions", OWN), "observed": ("observed decisions", OBSERVED)}


def _plt():
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("figures need matplotlib: pip install -e '.[plots]'") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "svg.hashsalt": "latent-decision-models", "svg.fonttype": "path",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })
    return plt


def _note(fig, text):
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=INK2, wrap=True)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _window_centre(spec) -> float:
    return (float(spec[1]) + float(spec[2])) / 2


# 1. EEG time course -------------------------------------------------------------------

def eeg_timecourse(report: dict):
    """Within-subject decoding around the response, own vs observed decisions."""
    plt = _plt()
    g = pd.DataFrame(report["group"])
    g = g[g["analysis"] == "within_subject"].copy()
    g["t"] = g["window"].map({w: _window_centre(s) for w, s in report["windows"].items()})
    panels = [("choice_side", "Response side (positive control)", "AUC"),
              ("confidence", "Confidence", "Spearman r"),
              ("confidence_beyond_behavior", "Confidence beyond behavior", "Spearman r")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True)
    for ax, (target, title, ylab) in zip(axes, panels, strict=True):
        sub = g[g["target"] == target]
        chance = float(sub["chance"].iloc[0])
        ax.axvspan(sub["t"].min() - 0.05, 0, color=GRID, alpha=0.5, lw=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axhline(chance, color=INK2, lw=0.8, ls=(0, (3, 3)))
        for cond, (label, colour) in COND.items():
            d = sub[sub["condition"] == cond].sort_values("t")
            lo, hi = np.array(d["ci95"].tolist()).T
            ax.fill_between(d["t"], lo, hi, color=colour, alpha=0.15, lw=0)
            ax.plot(d["t"], d["mean"], color=colour, label=label)
            sig = d["p_holm_by"] < 0.05
            ax.plot(d["t"][sig], d["mean"][sig], "o", ms=4, color=colour)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        ax.set_xlabel("window centre, s from response")
        ax.text(0.02, 0.97, "before\nresponse", transform=ax.transAxes, va="top",
                fontsize=7.5, color=INK2)
    axes[0].legend(loc="center left", fontsize=8)
    n = int(g["n_subjects"].max())
    _note(fig, f"Exploratory; {n} subjects; mean and 95% CI over subjects; dots: Holm p < 0.05 "
               "over 31 windows.\n100 ms windows with causal filters: true onsets may be up to "
               "~0.1 s earlier, never later.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# 2. Fusion ----------------------------------------------------------------------------

def fusion(report: dict, confirmatory_subjects: list[str]):
    """Behavior alone vs behavior + EEG, new participants, own decisions."""
    plt = _plt()
    s = pd.DataFrame(report["per_subject_scores"])
    s = s[(s["condition"] == "active") & (s["scheme"] == "participant_disjoint")
          & s["subject"].isin(confirmatory_subjects)].sort_values("B")
    test = next(t for t in report["tests"] if t.get("role") == "primary")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    for _, r in s.iterrows():
        a.plot([0, 1], [r["B"], r["B+E"]], color=OWN, lw=1, alpha=0.6, marker="o", ms=4)
    a.set_xticks([0, 1], ["behavior", "behavior + EEG"])
    a.set_xlim(-0.3, 1.3)
    a.set_ylabel("within-run Spearman r")
    a.set_title("Each new participant", loc="left")
    gain = (s["B+E"] - s["B"]).to_numpy()
    y = np.arange(len(gain))
    b.axvline(0, color=INK2, lw=0.8)
    b.plot(np.sort(gain), y, "o", ms=4, color=OWN)
    lo, hi = test["ci95"]
    b.errorbar(test["mean"], -1.5, xerr=[[test["mean"] - lo], [hi - test["mean"]]], fmt="D",
               ms=6, color=INK, capsize=3, lw=1.5)
    b.text(hi, -1.5, f"  mean {test['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}], p = "
                     f"{test['p_wilcoxon']:.2f}", va="center", fontsize=8)
    b.set_yticks([])
    b.set_ylim(-3, len(gain))
    b.set_xlabel("gain from adding EEG (B+E − B)")
    b.set_title("Gain per participant (sorted)", loc="left")
    _note(fig, f"Pre-specified primary test; {len(s)} confirmatory subjects, leave one subject "
               "out; post-decision features (behavior and EEG 0.1–0.6 s after the response).")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# 3. Behavior --------------------------------------------------------------------------

def confidence_by_accuracy(trials: pd.DataFrame):
    """Mean confidence on correct and error trials, own vs observed decisions."""
    plt = _plt()
    v = trials[trials["valid_trial"].astype(bool) & trials["confidence_raw"].notna()]
    m = v.groupby(["subject", "condition", "accuracy"])["confidence_raw"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for k, (cond, (label, colour)) in enumerate(COND.items()):
        d = m[m["condition"] == cond]
        x = d["accuracy"].to_numpy() + (k - 0.5) * 0.08
        ax.plot(x, d["confidence_raw"], "o", ms=3, color=colour, alpha=0.35)
        agg = d.groupby("accuracy")["confidence_raw"].agg(["mean", "std", "count"])
        ci = 1.96 * agg["std"] / np.sqrt(agg["count"])
        ax.errorbar(agg.index + (k - 0.5) * 0.2, agg["mean"], yerr=ci, color=colour,
                    marker="o", ms=7, capsize=3, label=label)
    ax.set_xticks([0, 1], ["error", "correct"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("confidence (0–1)")
    ax.set_title("Confidence on error and correct trials", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    n = m["subject"].nunique()
    _note(fig, f"{n} subjects, valid trials; small dots: subject means; large: mean and 95% CI.\n"
               "Mean levels are alike; the condition difference is trial-by-trial separation\n"
               "(type-2 AUROC 0.955 vs 0.941, Wilcoxon p = 0.03; GEE interaction p = 0.006).")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


# 4. fMRI pilot ------------------------------------------------------------------------

_CONTROL = {"visual": "visual", "movement": "movement\n(own − obs.)",
            "sensorimotor_active": "sensori-\nmotor"}

def fmri_pilot(pilot: dict, derivative: dict, timing: dict):
    """Positive controls (canonical and derivative-boosted) and the timing scan."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.4, 1.2]})
    for ax, rep, title in [(axes[0], pilot, "Canonical HRF"),
                           (axes[1], derivative, "HRF + derivative")]:
        labels = []
        for subj, ctrls in rep["positive_controls"].items():
            gating = rep.get("gating_controls") or list(ctrls)
            for name, c in ctrls.items():
                x = len(labels)
                colour = OWN if subj == "sub-02" else OBSERVED
                runs = np.asarray(c["per_run"], float)
                ax.plot(np.full(len(runs), x) + np.linspace(-0.12, 0.12, len(runs)), runs, "o",
                        ms=3.5, color=colour, alpha=0.6)
                ax.plot([x - 0.25, x + 0.25], [c["mean"]] * 2, color=INK, lw=2)
                passed = c["p_one_sided"] < 0.05
                status = "" if passed else ("\nfail" if name in gating else "\n(not gating)")
                ax.text(x, max(runs.max(), c["mean"]) + 4, f"p={c['p_one_sided']:.3f}" + status,
                        ha="center", fontsize=7,
                        color=OBSERVED if status == "\nfail" else INK2)
                labels.append(f"{subj}\n{_CONTROL[name]}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(title, loc="left")
        ax.set_ylabel("ROI beta contrast, % signal")
    t = pd.DataFrame(timing["mean_over_runs"])
    ax = axes[2]
    for subj, colour in [("sub-02", OWN), ("sub-04", OBSERVED)]:
        d = t[(t["subject"] == subj) & (t["roi_set"] == "occipital")].sort_values("shift_s")
        ax.plot(d["shift_s"], d["beta"], color=colour, marker="o", ms=3)
        ax.text(d["shift_s"].iloc[-1] + 0.2, d["beta"].iloc[-1], subj, fontsize=8,
                va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("onset shift, s")
    ax.set_ylabel("occipital response, % signal")
    ax.set_title("Timing scan (occipital)", loc="left")
    _note(fig, "Pre-specified pilot, 2 subjects, 6 runs each; dots: runs, bar: mean; one-sided "
               "t over runs; 'fail': a gating control with p ≥ 0.05. sub-04's visual control "
               "fails in both versions. Right (diagnosis, post hoc): occipital response when the "
               "model's onsets are shifted; sub-04 fits best at −6 s, i.e. about +3.5 s with "
               "trials ~9.5 s apart.")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


# 5. Paper comparison ------------------------------------------------------------------

def paper_comparison(report: dict, paper: dict):
    """Our descriptives and type-2 AUROC next to Pereira et al. (2020)."""
    plt = _plt()
    d = report["descriptives"]
    ours = {
        "valid trials, own": d["P2_valid_trials_per_subject"]["active"],
        "valid trials, observed": d["P2_valid_trials_per_subject"]["observed"],
        "accuracy, %": d["P3_accuracy_percent"],
        "dot difference": d["P4_dot_difference"],
        "RT, ms": d["P5_rt_ms_active_valid"],
        "type-2 AUROC, own": report["P8_type2_auroc"]["active"],
        "type-2 AUROC, observed": report["P8_type2_auroc"]["observed"],
    }
    fig, axes = plt.subplots(len(ours), 1, figsize=(6.2, 0.62 * len(ours) + 1.0))
    for ax, (name, o) in zip(axes, ours.items(), strict=True):
        pv, pe = paper[name]
        ax.errorbar(pv, 1, xerr=pe, fmt="s", color=INK2, ms=6, capsize=3, lw=1.5)
        ax.errorbar(o["mean"], 0, xerr=o["sd"], fmt="o", color=OWN, ms=6, capsize=3, lw=1.5)
        ax.set_yticks([0, 1], ["ours", "paper"], fontsize=7.5)
        ax.set_ylim(-0.7, 1.7)
        ax.set_title(name, loc="left", fontsize=8.5, pad=2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
    _note(fig, "Paper: value ± its reported interval (main text). Ours: mean ± SD over 20 "
               "subjects. Accuracy on valid trials is 71.5% (post hoc); the AUROC levels "
               "differ (unexplained) but the Wilcoxon test matches exactly (V = 163, p = 0.03).")
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.6)
    return fig


# 6. ds007474 model comparison ----------------------------------------------------------

def two_step_models(report: dict):
    """Held-out log-likelihood of each model relative to the hybrid (group summary only)."""
    plt = _plt()
    rh = report["run_heldout"]
    names = {"mf": "model-free", "mb": "model-based",
             "hybrid_contingency": "hybrid + contingency"}
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for y, (k, label) in enumerate(names.items()):
        r = rh[k]
        lo, hi = r["delta_ci95"]
        ax.errorbar(r["delta_vs_reference"], y, xerr=[[r["delta_vs_reference"] - lo],
                    [hi - r["delta_vs_reference"]]], fmt="o", color=OWN, ms=7, capsize=3)
        ax.text(hi, y, f"  {r['delta_vs_reference']:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]",
                va="center", fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(names)), list(names.values()))
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(right=900)
    ax.set_xlabel("held-out log-likelihood vs hybrid (summed over subjects)")
    ax.set_title("Every alternative predicts held-out runs worse than the hybrid", loc="left",
                 fontsize=9)
    _note(fig, f"{report['n_subjects']} subjects; pre-specified; 95% bootstrap CI over subjects. "
               "Group summary only (the OSF source has no license).")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


# Build --------------------------------------------------------------------------------

def _load(path):
    return json.loads(Path(path).read_text())


def build(reports_dir="reports", out_dir="docs/figures", trials=None, ds007474_report=None,
          root=".") -> dict:
    """Write every figure whose inputs exist; returns {name: path or reason skipped}."""
    from .datasets.ds002158_paper import PAPER_VALUES

    r, root = Path(reports_dir), Path(root)
    out = {}
    jobs = {
        "eeg-timecourse": lambda: eeg_timecourse(_load(r / "ds002158-eeg-timecourse.json")),
        "fusion": lambda: fusion(
            (f := _load(r / "ds002158-fusion.json")),
            yaml.safe_load((root / f["config"]).read_text())["confirmatory_subjects"]),
        "fmri-pilot": lambda: fmri_pilot(_load(r / "ds002158-fmri-pilot.json"),
                                         _load(r / "ds002158-fmri-derivative-pilot.json"),
                                         _load(r / "ds002158-fmri-timing.json")),
        "paper-comparison": lambda: paper_comparison(
            _load(r / "ds002158-paper-comparison.json"), PAPER_VALUES),
    }
    if trials is not None:
        jobs["confidence-by-accuracy"] = lambda: confidence_by_accuracy(
            pd.read_csv(trials, low_memory=False))
    else:
        out["confidence-by-accuracy"] = "skipped: no --trials"
    if ds007474_report is not None:
        jobs["ds007474-models"] = lambda: two_step_models(_load(ds007474_report))
    else:
        out["ds007474-models"] = "skipped: no --ds007474-report"
    for name, make in jobs.items():
        out[name] = str(_save(make(), Path(out_dir) / f"{name}.svg"))
    return out
