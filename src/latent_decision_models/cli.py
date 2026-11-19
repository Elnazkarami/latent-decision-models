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


def _ds007474_fetch(args):
    from .datasets import ds007474 as d
    from .io import osf

    meta = osf.project_metadata(d.OSF_NODE, d.OSF_VIEW_ONLY)
    files = osf.list_files(d.OSF_NODE, d.OSF_VIEW_ONLY)
    manifest = osf.fetch_files(files, args.dest, d.OSF_SUFFIXES, d.OSF_VIEW_ONLY)
    out = {"source": f"osf.io/{d.OSF_NODE}", "project": meta, "n_files": len(manifest),
           "total_bytes": sum(m["bytes"] for m in manifest), "files": manifest}
    Path(args.dest, "fetch_manifest.json").write_text(json.dumps(out, indent=1) + "\n")
    _write_json({k: v for k, v in out.items() if k != "files"})


def _ds007474_trials(args):
    from .datasets import ds007474 as d

    trials, issues = d.load_trials(args.osf_root, timing=args.timing, strict=not args.lenient)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out, index=False)
    iss = d.issues_frame(issues)
    if args.issues:
        iss.to_csv(args.issues, index=False)
    _write_json({
        "path": str(out), "n_trials": len(trials), "n_subjects": trials["subject"].nunique(),
        "trials_by_session": trials["session_type"].value_counts().to_dict(),
        "valid_fraction": float(trials["valid_trial"].mean()), "timing": args.timing,
        "issues": {f"{lv}:{code}": int(n) for (lv, code), n
                   in iss.groupby(["level", "code"]).size().items()},
        "schema": validate_trials(trials),
    })


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


def _confidence_model_recovery(args):
    from . import confidence_model as cm

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg.get("models", list(cm.MODELS))
    n_jobs = cfg.get("n_jobs", 1)
    ranges = {k: tuple(v) for k, v in (cfg.get("parameter_ranges") or {}).items()}
    rec = pd.concat([cm.parameter_recovery(m, cfg["n_sims"], cfg["n_trials"], cfg.get("seed", 0),
                                           cfg.get("n_starts", 2), n_jobs, ranges)
                     for m in models])
    conf = cm.model_recovery(models, cfg["n_model_sims"], cfg["n_trials"], cfg.get("seed", 0),
                             cfg.get("n_starts", 2), n_jobs, ranges)
    group = None
    if cfg.get("group_recovery"):
        gr = cfg["group_recovery"]
        group = cm.group_model_recovery(gr.get("models", models), gr["n_groups"],
                                        gr["n_subjects"], cfg["n_trials"], cfg.get("seed", 0),
                                        cfg.get("n_starts", 2), n_jobs, ranges)
    if cfg.get("recovery_table"):
        Path(cfg["recovery_table"]).parent.mkdir(parents=True, exist_ok=True)
        rec.to_csv(cfg["recovery_table"], index=False)
    _write_json({"config": cfg, "data_kind": "synthetic_parameter_and_model_recovery",
                 "parameter_ranges": {**cm.RECOVERY_RANGES, **ranges},
                 "parameter_recovery": cm.summarize_recovery(rec).to_dict(orient="records"),
                 "model_recovery_bic_confusion": {g: conf.loc[g].to_dict() for g in conf.index},
                 "group_model_recovery": group},
                cfg.get("output"))


def _confidence_model_compare(args):
    from . import confidence_model as cm

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = cm.prepare_model_trials(pd.read_csv(cfg["input"]),
                                 tuple(cfg.get("exclude_subjects", []) or []))
    models = cfg.get("models", list(cm.MODELS))
    res = cm.compare_models(df, models, cfg.get("reference", "normative"),
                            cfg.get("n_starts", 3), cfg.get("n_boot", 2000), cfg.get("seed", 0),
                            n_jobs=cfg.get("n_jobs", 1))
    ppc = pd.concat([cm.posterior_predictive(df, res["fits"], m, cfg.get("n_rep", 50),
                                             cfg.get("seed", 0)) for m in models])
    num = [c for c in ppc.columns if c.endswith(("_data", "_model"))]
    res["posterior_predictive_group_mean"] = (
        ppc.groupby(["model", "condition"])[num].mean().reset_index().to_dict(orient="records"))
    params = pd.DataFrame([{"subject": f["subject"], "model": f["model"], **f["params"]}
                           for f in res["fits"]])
    res["parameter_medians"] = (params.drop(columns="subject").groupby("model").median()
                                .to_dict(orient="index"))
    if cfg.get("ppc_table"):
        ppc.to_csv(cfg["ppc_table"], index=False)
    _write_json({"config": cfg, "data_kind": cfg.get("data_kind", "unspecified"), **res},
                cfg.get("output"))


def _two_step_schedules(cfg):
    from . import two_step_model as ts

    if cfg.get("schedules", "synthetic") == "synthetic":
        return [ts.synthetic_schedule(seed=i) for i in range(cfg.get("n_schedules", 20))], \
            "synthetic_schedule"
    from .datasets import ds007474 as d

    trials, _ = d.load_trials(cfg["osf_root"])
    sched = ts.schedule_from_trials(trials, runs=tuple(cfg.get("runs", ("run-1", "run-2"))))
    rows = ts._subject_rows(sched)
    return [sched.subset(r) for r in rows.values()], "real_ds007474_schedules"


def _two_step_recovery(args):
    from . import two_step_model as ts

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg.get("models", list(ts.MODELS))
    schedules, kind = _two_step_schedules(cfg)
    kw = {"seed": cfg.get("seed", 0), "n_starts": cfg.get("n_starts", 3),
          "prior_sd": cfg.get("prior_sd", 2.5), "reward": cfg.get("reward", "binary"),
          "ranges": {k: tuple(v) for k, v in (cfg.get("parameter_ranges") or {}).items()},
          "n_jobs": cfg.get("n_jobs", 1)}
    rec = {m: ts.parameter_recovery(m, schedules, cfg["n_sims"], **kw) for m in models}
    conf = ts.model_recovery(models, schedules, cfg["n_model_sims"], **kw)
    group = ts.group_model_recovery(conf, models, cfg.get("group_size", 10), cfg.get("seed", 0))
    confusion = (pd.crosstab(conf["generating"], conf["selected"]).to_dict(orient="index")
                 if len(conf) else None)
    group_confusion = (pd.crosstab(group["generating"], group["selected"])
                       .to_dict(orient="index") if len(group) else None)
    contrast = [ts.contingency_effect_recovery(schedules, e, **kw)
                for e in cfg.get("contingency_effects", [])]
    if cfg.get("recovery_table"):
        Path(cfg["recovery_table"]).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rec.values()).to_csv(cfg["recovery_table"], index=False)
    _write_json({"config": cfg, "data_kind": f"simulated_agents_on_{kind}",
                 "parameter_ranges": {**ts.RECOVERY_RANGES, **kw["ranges"]},
                 "parameter_recovery": {m: ts.summarize_recovery(r, m) for m, r in rec.items()},
                 "model_recovery_confusion": confusion,
                 "group_model_recovery_confusion": group_confusion,
                 "contingency_effect_recovery": contrast},
                cfg.get("output"))


def _two_step_compare(args):
    from . import two_step_model as ts

    cfg = yaml.safe_load(Path(args.config).read_text())
    res = ts.compare(pd.read_csv(cfg["input"]), models=cfg.get("models", list(ts.MODELS)),
                     reference=cfg.get("reference", "hybrid"), reward=cfg.get("reward", "binary"),
                     min_rt=cfg.get("min_rt", 0.15),
                     max_missed_fraction=cfg.get("max_missed_fraction", 0.10),
                     exclude_subjects=tuple(cfg.get("exclude_subjects", []) or []),
                     n_starts=cfg.get("n_starts", 4), prior_sd=cfg.get("prior_sd", 2.5),
                     n_folds=cfg.get("n_folds", 5), n_boot=cfg.get("n_boot", 2000),
                     n_rep=cfg.get("n_rep", 20), seed=cfg.get("seed", 0),
                     n_jobs=cfg.get("n_jobs", 1),
                     replication_run=cfg.get("replication_run"))
    fits = res.pop("fits")
    if cfg.get("fits_table"):
        fits.to_csv(cfg["fits_table"], index=False)
    _write_json({"config": cfg, "data_kind": cfg.get("data_kind", "unspecified"), **res},
                cfg.get("output"))


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

    cm = sub.add_parser("confidence-model",
                        help="Second-order SDT confidence model: recovery and comparison")
    cmsub = cm.add_subparsers(dest="cm_cmd", required=True)
    cmsub.add_parser("recovery", help="Synthetic parameter and model recovery").add_argument(
        "--config", required=True)
    cmsub.add_parser("compare", help="Fit models per subject; run-held-out comparison"
                     ).add_argument("--config", required=True)

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

    ts = sub.add_parser("two-step-model",
                        help="MF / MB / hybrid RL for the ds007474 two-step task")
    tssub = ts.add_subparsers(dest="ts_cmd", required=True)
    tssub.add_parser("recovery", help="Parameter and model recovery on task schedules"
                     ).add_argument("--config", required=True)
    tssub.add_parser("compare", help="Per-subject fits; held-out run and subject comparison"
                     ).add_argument("--config", required=True)
    d7 = sub.add_parser("ds007474", help="OpenNeuro ds007474 two-step task (experimental)")
    d7sub = d7.add_subparsers(dest="d7_cmd", required=True)
    f7 = d7sub.add_parser("fetch-behavior",
                          help="Download OSF behavior CSV + timing .mat (~13 MB; no MRI)")
    f7.add_argument("--dest", required=True)
    t7 = d7sub.add_parser("trials", help="Parse OSF behavior into a validated trial table")
    t7.add_argument("--osf-root", required=True)
    t7.add_argument("--out", required=True)
    t7.add_argument("--timing", action="store_true",
                    help="Merge event timing from .mat files (needs the [ds007474] extra)")
    t7.add_argument("--issues")
    t7.add_argument("--lenient", action="store_true", help="Report errors instead of raising")

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
    elif args.cmd == "confidence-model":
        {"recovery": _confidence_model_recovery,
         "compare": _confidence_model_compare}[args.cm_cmd](args)
    elif args.cmd == "two-step-model":
        {"recovery": _two_step_recovery, "compare": _two_step_compare}[args.ts_cmd](args)
    elif args.cmd == "ds007474":
        {"fetch-behavior": _ds007474_fetch, "trials": _ds007474_trials}[args.d7_cmd](args)
    elif args.cmd == "ds002158":
        {"fetch-metadata": _ds002158_fetch, "trials": _ds002158_trials,
         "align-eeg": _ds002158_align}[args.ds_cmd](args)


if __name__ == "__main__":
    main()
