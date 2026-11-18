"""CLI smoke tests (no network)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

FIX = Path(__file__).parent / "fixtures" / "ds002158"


def _ldm(*args):
    out = subprocess.run([sys.executable, "-m", "latent_decision_models.cli", *args],
                         check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def test_ds002158_trials_and_align(tmp_path):
    root = tmp_path / "bids"
    shutil.copytree(FIX, root)
    res = _ldm("ds002158", "trials", "--bids-root", str(root),
               "--out", str(tmp_path / "t.csv"), "--issues", str(tmp_path / "i.csv"))
    assert res["n_trials"] == 48 and res["schema"]["valid"] and res["issues"] == {}
    ev = root / "sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_events.tsv"
    mk = root / "sub-02/ses-001/eeg/sub-02_ses-001_task-main_run-001_eeg.vmrk"
    assert _ldm("ds002158", "align-eeg", "--events", str(ev), "--vmrk", str(mk))["status"] == \
        "verified"


def test_confidence_baseline_cli_on_synthetic(tmp_path):
    from latent_decision_models.synthetic import generate_trials

    df = generate_trials(subjects=8, trials_per_subject=30)
    df = df.rename(columns={"evidence": "evidence_dots_difference"})
    df["response_latency_seconds"] = df["reaction_time_seconds"].fillna(0.4)
    csv = tmp_path / "syn.csv"
    df.to_csv(csv, index=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"input": str(csv), "output": str(tmp_path / "r.json"),
                                   "n_splits": 4, "n_boot": 100, "data_kind": "synthetic"}))
    res = _ldm("confidence-baseline", "--config", str(cfg))
    assert res["data_kind"] == "synthetic" and res["n_subjects"] == 8
    assert (tmp_path / "r.json").exists()
