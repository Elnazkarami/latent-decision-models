"""Command-line interface (``ldm``). No command reads EEG/fMRI signal data."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .benchmark import benchmark
from .schema import validate_trials
from .synthetic import generate_trials


def _write_json(content, out=None):
    text = json.dumps(content, indent=2, default=str)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


def _ds002158_fetch(args):
    from .datasets import ds002158 as d
    from .io import openneuro

    files = openneuro.snapshot_files(d.DATASET_ID, args.tag)
    patterns = ["README", "CHANGES", "dataset_description.json", "participants.*",
                "task-main_events.json", "task-main_bold.json", ".bidsignore",
                "sub-*/ses-*/func/*_task-main_run-*_events.tsv", "sub-*/ses-*/sub-*_scans.tsv"]
    sel = [f for f in openneuro.select_files(files, patterns) if "/rejected/" not in f["filename"]]
    manifest = openneuro.fetch_files(sel, args.dest)
    out = {"dataset_id": d.DATASET_ID, "snapshot": args.tag, "n_files": len(manifest),
           "total_bytes": sum(m["bytes"] for m in manifest), "files": manifest}
    Path(args.dest, "fetch_manifest.json").write_text(json.dumps(out, indent=1) + "\n")
    _write_json({k: v for k, v in out.items() if k != "files"})


def _ds002158_trials(args):
    from .datasets import ds002158 as d

    trials, issues = d.load_trials(args.bids_root, strict=not args.lenient)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out, index=False)
    iss = d.issues_frame(issues)
    if args.issues:
        iss.to_csv(args.issues, index=False)
    _write_json({
        "path": str(out), "n_trials": len(trials), "n_subjects": trials["subject"].nunique(),
        "n_runs": int(trials.groupby(["subject", "run"]).ngroups),
        "valid_trials_by_condition": trials[trials["valid_trial"]]["condition"]
        .value_counts().to_dict(),
        "issues": {f"{lv}:{code}": int(n) for (lv, code), n
                   in iss.groupby(["level", "code"]).size().items()},
        "schema": validate_trials(trials),
    })


def _ds002158_align(args):
    from .datasets import ds002158 as d
    from .io.brainvision import read_vhdr_sampling_rate, read_vmrk

    trials, _ = d.parse_events(args.events)
    vhdr = args.vhdr or str(Path(args.vmrk).with_suffix(".vhdr"))
    markers = read_vmrk(args.vmrk, read_vhdr_sampling_rate(vhdr))
    table, summary = d.align_eeg_markers(trials, markers)
    if args.out:
        table.to_csv(args.out, index=False)
    _write_json({"events": args.events, "vmrk": args.vmrk, **summary})


def _confidence_baseline(args):
    from .confidence_baseline import prepare, run_confidence_baseline
    from .metacognition import metacognition_gee

    cfg = yaml.safe_load(Path(args.config).read_text())
    trials = pd.read_csv(cfg["input"])
    result = run_confidence_baseline(
        trials, n_splits=cfg.get("n_splits", 5), alpha=cfg.get("ridge_alpha", 1.0),
        n_boot=cfg.get("n_boot", 2000), seed=cfg.get("seed", 0),
        exclude_subjects=tuple(cfg.get("exclude_subjects", []) or []))
    excl = tuple(cfg.get("exclude_subjects", []) or [])
    result["metacognition_gee"] = metacognition_gee(prepare(trials, excl))
    result = {"config": cfg, "data_kind": cfg.get("data_kind", "unspecified"), **result}
    _write_json(result, cfg.get("output"))


def main():
    p = argparse.ArgumentParser(prog="ldm", description="Decision modeling research tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("simulate", help="Generate explicitly synthetic behavioral trials")
    a.add_argument("--out", required=True)
    a.add_argument("--subjects", type=int, default=12)
    a.add_argument("--trials-per-subject", type=int, default=35)
    a.add_argument("--seed", type=int, default=23)
    b = sub.add_parser("validate", help="Validate a trial CSV")
    b.add_argument("--input", required=True)
    c = sub.add_parser("benchmark", help="Group-wise confidence benchmark (synthetic)")
    c.add_argument("--config", required=True)
    cb = sub.add_parser("confidence-baseline",
                        help="Subject-disjoint behavioral confidence baselines (config)")
    cb.add_argument("--config", required=True)

    ds = sub.add_parser("ds002158", help="OpenNeuro ds002158 adapter (experimental)")
    dsub = ds.add_subparsers(dest="ds_cmd", required=True)
    f = dsub.add_parser("fetch-metadata",
                        help="Download sidecars + events.tsv only (~0.3 MB; no signal data)")
    f.add_argument("--dest", required=True)
    f.add_argument("--tag", default="1.0.2")
    t = dsub.add_parser("trials", help="Parse events into a validated trial table")
    t.add_argument("--bids-root", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--issues")
    t.add_argument("--lenient", action="store_true", help="Report errors instead of raising")
    al = dsub.add_parser("align-eeg", help="Match one run's events to its BrainVision markers")
    al.add_argument("--events", required=True)
    al.add_argument("--vmrk", required=True)
    al.add_argument("--vhdr")
    al.add_argument("--out")

    args = p.parse_args()
    if args.cmd == "simulate":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = generate_trials(args.subjects, args.trials_per_subject, args.seed)
        df.to_csv(out, index=False)
        _write_json({"status": "generated", "kind": "synthetic", "path": str(out),
                     "n_trials": len(df)})
    elif args.cmd == "validate":
        report = validate_trials(pd.read_csv(args.input))
        _write_json(report)
        if not report["valid"]:
            raise SystemExit(1)
    elif args.cmd == "benchmark":
        cfg = yaml.safe_load(Path(args.config).read_text())
        result = benchmark(pd.read_csv(cfg["input"]), cfg["features"], cfg.get("n_splits", 4))
        _write_json(result, cfg.get("output"))
    elif args.cmd == "confidence-baseline":
        _confidence_baseline(args)
    elif args.cmd == "ds002158":
        {"fetch-metadata": _ds002158_fetch, "trials": _ds002158_trials,
         "align-eeg": _ds002158_align}[args.ds_cmd](args)


if __name__ == "__main__":
    main()
