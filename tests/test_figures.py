"""Figures rebuild from the committed reports (needs the optional [plots] extra)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from latent_decision_models import figures as F
from latent_decision_models.datasets import ds002158 as d

ROOT = Path(__file__).parents[1]
FIX = Path(__file__).parent / "fixtures" / "ds002158"


def test_build_from_committed_reports(tmp_path):
    out = F.build(ROOT / "reports", tmp_path, root=ROOT)
    made = {k for k, v in out.items() if not v.startswith("skipped")}
    assert made == {"eeg-timecourse", "fusion", "fmri-pilot", "paper-comparison"}
    for name in made:
        svg = (tmp_path / f"{name}.svg").read_text()
        assert svg.startswith("<?xml") and "<svg" in svg
        assert "/Users/" not in svg and "<dc:date>" not in svg


def test_svg_is_deterministic(tmp_path):
    rep = json.loads((ROOT / "reports" / "ds002158-paper-comparison.json").read_text())
    from latent_decision_models.datasets.ds002158_paper import PAPER_VALUES

    a = F._save(F.paper_comparison(rep, PAPER_VALUES), tmp_path / "a.svg").read_bytes()
    b = F._save(F.paper_comparison(rep, PAPER_VALUES), tmp_path / "b.svg").read_bytes()
    assert a == b


def _locked_matplotlib() -> str | None:
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        if line.lower().startswith("matplotlib=="):
            return line.split("==")[1].strip()
    return None


def test_committed_figures_are_current(tmp_path):
    """docs/figures must equal a fresh build from the committed reports.

    Byte equality only holds for the matplotlib version the figures were made with.
    """
    import matplotlib

    if matplotlib.__version__ != _locked_matplotlib():
        pytest.skip(f"matplotlib {matplotlib.__version__} differs from the lock file")
    F.build(ROOT / "reports", tmp_path, root=ROOT)
    for svg in ["eeg-timecourse", "fusion", "fmri-pilot", "paper-comparison"]:
        assert (tmp_path / f"{svg}.svg").read_bytes() == \
            (ROOT / "docs" / "figures" / f"{svg}.svg").read_bytes(), svg


def test_behavior_figure_from_fixture(tmp_path):
    trials, _ = d.load_trials(FIX)
    fig = F.confidence_by_accuracy(trials)
    assert F._save(fig, tmp_path / "c.svg").stat().st_size > 1000


def test_two_step_figure_uses_group_summaries_only(tmp_path):
    rh = {k: {"delta_vs_reference": v, "delta_ci95": [v - 100, v + 100]}
          for k, v in {"mf": -1000.0, "mb": -900.0, "hybrid_contingency": -1500.0}.items()}
    fig = F.two_step_models({"run_heldout": rh, "n_subjects": 10})
    svg = F._save(fig, tmp_path / "m.svg").read_text()
    assert "<svg" in svg
