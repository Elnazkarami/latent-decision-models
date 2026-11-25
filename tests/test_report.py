"""docs/REPORT.md quotes the committed ds002158 reports correctly (no drift).

ds007474 numbers are not checked here: its real-data outputs are not stored in the
repository (the OSF source declares no license).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "docs" / "REPORT.md").read_text().replace("\u2212", "-")  # typographic minus


def _group(name):
    return pd.DataFrame(json.loads((ROOT / "reports" / name).read_text())["group"])


def _cell(g, **kw):
    m = pd.Series(True, index=g.index)
    for k, v in kw.items():
        m &= g[k] == v
    assert m.sum() == 1, kw
    return g[m].iloc[0]


def _in_report(text):
    assert text in REPORT, f"REPORT.md does not contain {text!r}"


def test_confirmatory_eeg_numbers():
    g = _group("ds002158-eeg-decoding-confirmatory.json")
    w = _cell(g, analysis="within_subject", target="confidence", condition="active",
              window="post_decision")
    lo = _cell(g, analysis="leave_one_subject_out", target="confidence", condition="active",
               window="post_decision")
    _in_report(f"Within subject r = {w['mean']:.2f} [{w['ci95'][0]:.2f}, {w['ci95'][1]:.2f}], "
               f"Holm p = {w['p_holm_primary']:.4f} ({w['n_above_chance']}/17)")
    _in_report(f"left-out subjects r = {lo['mean']:.2f} [{lo['ci95'][0]:.2f}, "
               f"{lo['ci95'][1]:.2f}], Holm p = {lo['p_holm_primary']:.3f}")
    obs = _cell(g, analysis="within_subject", target="confidence", condition="observed",
                window="post_decision")
    _in_report(f"observed r = {obs['mean']:.2f}, Holm p = {obs['p_holm_primary']:.2f}")
    beyond = _cell(g, analysis="within_subject", target="confidence_beyond_behavior",
                   condition="active", window="post_decision")
    _in_report(f"within r = {beyond['mean']:.2f} (p = {beyond['p_wilcoxon']:.3f}, uncorrected)")
    side = [_cell(g, analysis="within_subject", target="choice_side", condition="active",
                  window=wnd)["mean"] for wnd in ("response_locked", "post_decision")]
    _in_report(f"AUC {side[0]:.2f} at the response and {side[1]:.2f} after it")


def test_cardiac_obs1_and_own_vs_observed_numbers():
    c = _cell(_group("ds002158-eeg-decoding-cardiac-control-aas.json"),
              analysis="within_subject", target="confidence_beyond_cardiac", condition="active")
    _in_report(f"within r = {c['mean']:.3f}, Holm p = {c['p_holm_primary']:.3f}")
    o = _cell(_group("ds002158-eeg-decoding-confirmatory-obs1.json"),
              analysis="leave_one_subject_out", target="confidence", condition="active",
              window="post_decision")
    _in_report(f"OBS with 1 component: r = {o['mean']:.2f}, Holm p = {o['p_holm_primary']:.3f}")
    g = _group("ds002158-eeg-own-vs-observed-aas.json")
    a2o = _cell(g, target="confidence", condition="active->observed", window="post_decision")
    _in_report(f"Own → observed r = {a2o['mean']:.2f}")


def test_fusion_numbers():
    t = pd.DataFrame(json.loads((ROOT / "reports" / "ds002158-fusion.json").read_text())["tests"])
    p = _cell(t, role="primary")
    _in_report(f"gain from EEG +{p['mean']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}], "
               f"p = {p['p_wilcoxon']:.2f}")
    e = _cell(t, contrast="E vs 0")
    _in_report(f"EEG alone r = {e['mean']:.2f}, Holm p = {e['p_holm_secondary']:.3f}")
    m = _cell(t, contrast="MLP - B+E")
    _in_report(f"{m['mean']:.2f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}], "
               f"Holm p = {m['p_holm_secondary']:.2f}")


def test_fmri_pilot_numbers():
    d = json.loads((ROOT / "reports" / "ds002158-fmri-derivative-pilot.json").read_text())
    v = d["positive_controls"]["sub-04"]["visual"]["p_one_sided"]
    assert d["controls_passed"] is False
    _in_report(f"sub-04 visual still fails (p = {v:.2f})")
