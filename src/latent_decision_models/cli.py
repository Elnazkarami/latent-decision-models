"""Command-line interface (``ldm``). No command reads EEG/fMRI signal data."""

import argparse
import json
import re
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


def _fetch_signal(args, pattern: str, kind: str):
    """Download signal files of named subjects within an approved budget (EEG or fMRI)."""
    from .datasets import ds002158 as d
    from .io import openneuro

    files = openneuro.snapshot_files(d.DATASET_ID, args.tag)
    subjects = set(args.subjects)
    sel = [f for f in files if re.fullmatch(pattern, f["filename"])
           and re.search(r"sub-\d+", f["filename"]).group(0) in subjects]
    found = {re.search(r"sub-\d+", f["filename"]).group(0) for f in sel}
    if found != subjects:
        raise SystemExit(f"no {kind} files for {sorted(subjects - found)}")
    total = sum(f["size"] for f in sel)
    if args.dry_run:
        _write_json({"subjects": sorted(subjects), "n_files": len(sel), "total_bytes": total})
        return
    manifest = openneuro.fetch_signal_files(sel, args.dest, int(args.approved_gb * 1e9),
                                             min_free_bytes=int(args.min_free_gb * 1e9))
    # One manifest per destination and kind, accumulated over calls (one entry per file).
    mpath = Path(args.dest, f"{kind}_fetch_manifest.json")
    prev = json.loads(mpath.read_text())["files"] if mpath.exists() else []
    files_by_path = {m["path"]: m for m in prev} | {m["path"]: m for m in manifest}
    all_files = sorted(files_by_path.values(), key=lambda m: m["path"])
    out = {"dataset_id": d.DATASET_ID, "snapshot": args.tag,
           "subjects": sorted({re.search(r"sub-\d+", m["path"]).group(0) for m in all_files}),
           "n_files": len(all_files), "total_bytes": sum(m["bytes"] for m in all_files),
           "files": all_files}
    mpath.write_text(json.dumps(out, indent=1) + "\n")
    _write_json({"fetched_subjects": sorted(subjects), "n_files": len(manifest),
                 "bytes": total, "manifest_total_bytes": out["total_bytes"]})


def _ds002158_fetch_eeg(args):
    """Raw BrainVision EEG for named subjects: signal data, needs an approved budget."""
    _fetch_signal(args, r"(sub-\d+)/ses-001/eeg/\1_ses-001_task-main_run-\d+_eeg\."
                        r"(eeg|vhdr|vmrk)", "eeg")


def _ds002158_fetch_fmri(args):
    """Published derivative BOLD (preproc.nii) + realignment parameters for named subjects.

    Runs in the derivative's reject/rejected folders are never selected.
    """
    _fetch_signal(args, r"derivatives/fmriprep/(sub-\d+)/ses-001/func/\1_ses-001_task-main_"
                        r"run-\d+_(preproc\.nii|headmove\.txt)", "fmri")


def _ds002158_eeg_preprocess(args):
    from .datasets import ds002158_eeg as e

    methods = list(dict.fromkeys(args.bcg_method))
    if len(methods) > 1 and ("{method}" not in args.out_dir
                             or (args.report and "{method}" not in args.report)):
        raise SystemExit("with several --bcg-method values, --out-dir and --report must "
                         "contain {method}")
    trials = pd.read_csv(args.trials)
    reports = {m: {"band_hz": [args.l_freq, args.h_freq], "bcg_method": m,
                   **({"obs_n_pc": args.obs_n_pc} if m == "obs" else {}), "subjects": []}
               for m in methods}
    for s in args.subjects:
        res = e.preprocess_subject_methods(
            args.bids_root, s, trials, {m: args.out_dir.format(method=m) for m in methods},
            l_freq=args.l_freq, h_freq=args.h_freq, obs_n_pc=args.obs_n_pc)
        for m in methods:
            reports[m]["subjects"].append(res[m])
    if args.report:
        for m, report in reports.items():
            path = Path(args.report.format(method=m))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1, default=_json_default) + "\n")
    _write_json({m: [{k: r[k] for k in ("subject", "n_eeg_runs", "n_runs_used", "n_epochs",
                                        "n_rejected")} for r in report["subjects"]]
                 for m, report in reports.items()})


def _ds002158_eeg_bcg_retention(args):
    from .datasets import ds002158_eeg as e

    rows = []
    for s in args.subjects:
        rows += e.bcg_retention_subject(args.bids_root, s, args.epochs_dir,
                                        amplitude_uv=args.amplitude_uv, seed=args.seed)
    per_run = pd.DataFrame(rows)
    per_subject = (per_run.groupby(["subject", "waveform", "method"], sort=False)
                   [["gain", "trial_r"]].mean().reset_index())
    summary = (per_subject.groupby(["waveform", "method"], sort=False)
               .agg(n_subjects=("subject", "nunique"), gain_median=("gain", "median"),
                    gain_min=("gain", "min"), gain_max=("gain", "max"),
                    trial_r_median=("trial_r", "median"), trial_r_min=("trial_r", "min"))
               .reset_index())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"amplitude_uv": args.amplitude_uv, "seed": args.seed,
             "summary": _records(summary), "per_subject": _records(per_subject),
             "per_run": _records(per_run)}, indent=1, default=_json_default) + "\n")
    print(summary.round(3).to_string(index=False))


def _ds002158_eeg_cardiac(args):
    from .datasets import ds002158_eeg as e

    cfg = yaml.safe_load(Path(args.config).read_text())
    res = e.cardiac_behavior(args.epochs_dir or cfg["epochs_dir"], cfg["subjects"],
                             cfg.get("confirmatory_subjects"),
                             min_trials=cfg.get("min_trials", 40),
                             min_runs=cfg.get("min_runs", 3), seed=cfg.get("seed", 0))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"config": args.config, "per_subject": _records(res["per_subject"]),
             "group": _records(res["group"])}, indent=1, default=_json_default) + "\n")
    cols = ["condition", "behavior", "measure", "n_subjects", "mean", "ci95", "p_wilcoxon",
            "p_holm"]
    print(res["group"][cols].to_string(index=False))


def _ds002158_fusion(args):
    """LDM-005: behavior + post-decision EEG fusion for confidence (pre-specified plan)."""
    from .datasets import ds002158_eeg as e

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs_dir:
        cfg["epochs_dir"] = args.epochs_dir
    res = e.run_fusion(cfg)
    if args.predictions:
        Path(args.predictions).mkdir(parents=True, exist_ok=True)
        for (c, scheme), p in res["predictions"].items():
            p.to_csv(Path(args.predictions, f"{c}_{scheme}.csv"), index=False)
    tests = res["tests"]
    cols = ["role", "condition", "scheme", "contrast", "n_subjects", "mean", "ci95",
            "n_above_chance", "p_wilcoxon", "p_holm_secondary"]
    print(tests[cols].to_string(index=False))
    out = {"config": args.config, "data_kind": "real_eeg_behavior_openneuro_ds002158_v1.0.2",
           "timing": "post-decision features (behavior and 0.1-0.6 s after the response); "
                     "not prospective",
           "tests": _records(tests), "per_subject_scores": _records(res["scores"])}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, default=_json_default) + "\n")


def _ds002158_compare_paper(args):
    """Quantities comparable to Pereira et al. (2020), main-text behavior results."""
    from .datasets import ds002158_paper as pp

    res = pp.compare(pd.read_csv(args.trials))
    _write_json({"data_kind": "real_behavioral_openneuro_ds002158_v1.0.2",
                 "paper": "Pereira et al. (2020) PNAS 117:8382, main text",
                 "comparison_rules": "docs/VALIDATION.md", **res}, args.out)


def _ds002158_eeg_decode(args):
    from .datasets import ds002158_eeg as e

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs_dir:
        cfg["epochs_dir"] = args.epochs_dir
    res = e.run_decoding(cfg)
    out = {"config": args.config, "channels_used": res["channels_used"],
           "excluded_channels": res["excluded_channels"], "windows": res["windows"],
           "within_subject": _records(res["within_subject"]),
           "cross_subject": _records(res["cross_subject"]),
           "cross_condition": _records(res["cross_condition"]),
           "score_mode": res["score_mode"], "group": _records(res["group"])}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, default=_json_default) + "\n")
    g = res["group"]
    if not g.empty:
        cols = ["analysis", "target", "condition", "window", "n_subjects", "mean", "ci95",
                "n_above_chance", "p_wilcoxon", "p_holm_primary"]
        print(g[cols].to_string(index=False))
    else:
        w = res["within_subject"]
        cols = ["subject", "target", "condition", "window", "score", "null_95", "p_value",
                "p_holm_primary", "n_trials"]
        print(w[cols].round(3).to_string(index=False))


def _records(df):
    """DataFrame rows as dicts with NaN -> None (valid JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _json_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


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


def _ds002158_fmri_audit(args):
    from .datasets import ds002158 as d
    from .datasets import ds002158_fmri as f
    from .io import nifti, openneuro

    files = openneuro.snapshot_files(d.DATASET_ID, args.tag)
    inv = f.inventory(files)
    motion = openneuro.select_files(files, [f.DERIV + "sub-*/ses-001/func/*_headmove.txt"])
    manifest = openneuro.fetch_files(motion, args.dest)
    rows = []
    for m in manifest:
        sub, run = re.search(r"(sub-\d+)_ses-001_task-main_(run-\d+)", m["path"]).groups()
        rows.append({"subject": sub, "run": run,
                     **f.motion_summary(f.read_motion(Path(args.dest, m["path"])))})
    mot = pd.DataFrame(rows)
    headers = f.header_audit(inv, files, openneuro.fetch_range, nifti.parse_header)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = inv.merge(mot, on=["subject", "run"], how="left")
    hdr_wide = headers.pivot_table(index=["subject", "run"], columns="file",
                                   values="n_volumes").reset_index()
    table = table.merge(hdr_wide.rename(columns={"raw_bold": "raw_volumes",
                                                 "derivative_preproc": "derivative_volumes"}),
                        on=["subject", "run"], how="left")
    table.to_csv(out.with_suffix(".csv"), index=False)
    headers.to_csv(out.with_name(out.stem + "_headers.csv"), index=False)
    deriv = headers[headers["file"] == "derivative_preproc"]
    raw = headers[headers["file"] == "raw_bold"]

    def distinct(df, col):
        return pd.Series([str(v) for v in df[col]]).value_counts().to_dict()

    _write_json({
        "dataset_id": d.DATASET_ID, "snapshot": args.tag, "data_kind": "metadata_and_headers",
        "n_runs": len(inv), "n_raw_bold": int(inv["raw_bold"].notna().sum()),
        "n_derivative_preproc": int(inv["derivative_preproc"].notna().sum()),
        "derivatives_in_reject_folders": inv.loc[inv["derivative_folder"].notna(),
                                                 ["subject", "run", "derivative_folder"]]
        .to_dict(orient="records"),
        "raw_without_derivative": inv.loc[inv["raw_bold"].notna()
                                          & inv["derivative_preproc"].isna(),
                                          ["subject", "run"]].to_dict(orient="records"),
        "derivative_without_raw": inv.loc[inv["raw_bold"].isna()
                                          & inv["derivative_preproc"].notna(),
                                          ["subject", "run"]].to_dict(orient="records"),
        "volume_mismatch": table.loc[table["raw_volumes"].notna()
                                     & table["derivative_volumes"].notna()
                                     & (table["raw_volumes"] != table["derivative_volumes"]),
                                     ["subject", "run"]].to_dict(orient="records"),
        "motion_rows_mismatch": table.loc[table["n_volumes"].notna()
                                          & (table["n_volumes"] != table["derivative_volumes"]),
                                          ["subject", "run"]].to_dict(orient="records"),
        "raw_headers": {c: distinct(raw, c) for c in ("shape", "voxel_size_mm",
                                                      "repetition_time", "sform_code")},
        "derivative_headers": {c: distinct(deriv, c) for c in (
            "shape", "voxel_size_mm", "repetition_time", "datatype", "sform_code", "descrip",
            "affine")},
        "motion": {"fd_threshold_mm": 0.5,
                   "mean_fd_mm_quantiles": mot["mean_fd_mm"].quantile([0, .5, .9, 1]).round(3)
                   .tolist(),
                   "max_abs_translation_mm_max": float(mot["max_abs_translation_mm"].max()),
                   "max_abs_rotation_deg_max": float(mot["max_abs_rotation_deg"].max()),
                   "runs_over_20pct_high_fd": mot.loc[mot["fraction_fd_above_threshold"] > 0.2,
                                                      ["subject", "run"]]
                   .to_dict(orient="records")},
        "tables": [str(out.with_suffix(".csv")), str(out.with_name(out.stem + "_headers.csv"))],
    }, args.out)


def _ds002158_fmri_lss(args):
    """Single-trial ROI amplitudes (LSS) and per-run QC from the published derivative.

    With ``--roi-ts`` (or ``roi_ts_dir`` in the config) the ROI time series saved by an
    earlier run are reused and no image is read (LSS is linear in the data).
    """
    import numpy as np

    from .datasets import ds002158_fmri as f

    cfg = yaml.safe_load(Path(args.config).read_text())
    bids_root = args.bids_root or cfg["bids_root"]
    out_dir = Path(args.out_dir or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_ts = args.roi_ts or cfg.get("roi_ts_dir")
    trials = pd.read_csv(cfg["input_trials"])
    subjects = args.subjects or cfg["subjects"]
    kw = {"tr": cfg["tr"], "hrf_model": cfg["hrf_model"], "high_pass": cfg["high_pass_hz"],
          "amplitude": cfg.get("amplitude", "canonical")}
    qc_rows = []
    for subject in subjects:
        t = trials[trials["subject"] == subject]
        parts, names = [], None
        if roi_ts:
            files = sorted(Path(roi_ts).glob(f"{subject}_run-*_roi_ts.npz"))
            if not files:
                raise SystemExit(f"{subject}: no ROI time series in {roi_ts}")
            for path in files:
                run = path.name.split("_")[1]
                z = np.load(path)
                names = dict(enumerate((str(n) for n in z["names"]), start=1))
                b, qc = f.lss_from_timeseries(z["Y"], names, t[t["run"] == run],
                                              f._motion(bids_root, subject, run), **kw)
                parts.append(b)
                qc_rows.append({"subject": subject, "run": run, **qc})
        else:
            import nibabel as nib

            runs = sorted(r for r in t["run"].unique()
                          if f.preproc_path(bids_root, subject, r).exists())
            if not runs:
                raise SystemExit(f"{subject}: no derivative images under {bids_root}")
            ref = nib.load(f.preproc_path(bids_root, subject, runs[0]))
            atlas = f.harvard_oxford_atlas(ref.affine, ref.shape)
            names = atlas[1]
            for run in runs:
                img = nib.load(f.preproc_path(bids_root, subject, run))
                if not np.allclose(img.affine, ref.affine) or img.shape[:3] != ref.shape[:3]:
                    raise SystemExit(f"{subject} {run}: grid differs from {runs[0]}")
                b, qc, Y = f.run_lss(bids_root, subject, run, t[t["run"] == run],
                                     atlas=atlas, **kw)
                (out_dir / "roi_ts").mkdir(exist_ok=True)
                np.savez(out_dir / "roi_ts" / f"{subject}_{run}_roi_ts.npz", Y=Y,
                         names=np.array(list(names.values())))
                parts.append(b)
                qc_rows.append(qc)
        for qc in qc_rows[-len(parts):]:
            print(f"{subject} {qc['run']}: {qc['n_trials']} trials, ROI tSNR median "
                  f"{qc['roi_tsnr_median']:.0f}", flush=True)
        pd.concat(parts).to_csv(out_dir / f"{subject}_lss_roi_betas.csv", index=False)
        pd.Series(names).to_csv(out_dir / "atlas_labels.csv", header=["region"])
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out_dir / "lss_qc.csv", index=False)
    _write_json({"config": args.config, "subjects": subjects, "n_runs": len(qc),
                 "source": "saved ROI time series" if roi_ts else "derivative images",
                 "amplitude": kw["amplitude"], "hrf_model": kw["hrf_model"],
                 "beta_units": "percent of the ROI's run-mean signal per unit nilearn "
                               "regressor (stick at stimulus onset)",
                 "qc": _records(qc)}, args.report)


def _ds002158_fmri_timing(args):
    """Onset-shift scan of occipital / sensorimotor ROI means (timing diagnostic)."""
    from .datasets import ds002158_fmri as f

    cfg = yaml.safe_load(Path(args.config).read_text())
    ts_dir = Path(args.betas_dir or cfg["out_dir"]) / "roi_ts"
    trials = pd.read_csv(cfg["input_trials"])
    scan = f.timing_scan(ts_dir, args.bids_root or cfg["bids_root"], trials, cfg["subjects"],
                         tr=cfg["tr"])
    mean = (scan.groupby(["subject", "roi_set", "shift_s"])[["beta", "r2_gain"]].mean()
            .reset_index())
    best = (mean[mean["beta"] > 0].sort_values("r2_gain").groupby(["subject", "roi_set"])
            .tail(1).sort_values(["subject", "roi_set"]))
    print(best.round(3).to_string(index=False))
    _write_json({"config": args.config, "data_kind": "real_fmri_openneuro_ds002158_v1.0.2_"
                 "derivative", "best_positive_shift": _records(best),
                 "mean_over_runs": _records(mean),
                 "per_run": _records(scan)}, args.out)


def _ds002158_fmri_pilot(args):
    """Pre-specified positive controls, then (only if all pass) exploratory decoding."""
    from .datasets import ds002158_fmri as f

    cfg = yaml.safe_load(Path(args.config).read_text())
    betas_dir = Path(args.betas_dir or cfg["out_dir"])
    trials = pd.read_csv(cfg["input_trials"])
    names = pd.read_csv(betas_dir / "atlas_labels.csv", index_col=0)["region"].to_dict()
    betas = pd.concat([pd.read_csv(betas_dir / f"{s}_lss_roi_betas.csv")
                       for s in cfg["subjects"]])
    controls = f.positive_controls(betas, trials, names)
    passed = f.controls_passed(controls, cfg["controls"])
    out = {"config": args.config, "data_kind": "real_fmri_openneuro_ds002158_v1.0.2_derivative",
           "feature_window": "whole trial (stimulus, response and start of the confidence "
                             "report are not separable; docs/ds002158-fmri-audit.md)",
           "amplitude": cfg.get("amplitude", "canonical"),
           "gating_controls": [k for k, v in cfg["controls"].items()
                               if (v or {}).get("gate", True)],
           "positive_controls": controls, "controls_passed": passed}
    if passed:
        dec = f.decode_pilot(betas, trials, cfg["decoding"])
        out["decoding"] = _records(dec)
        print(dec[["subject", "condition", "target", "score", "null_95", "p_value",
                   "n_trials", "n_features"]].round(3).to_string(index=False))
    else:
        out["decoding"] = "not run: a positive control failed (plan, decision 2)"
    print(json.dumps(controls, indent=1))
    _write_json(out, args.out)


def _ds002158_fmri_design(args):
    from joblib import Parallel, delayed

    from .datasets import ds002158_fmri as f
    from .fmri_design import run_diagnostics

    trials = pd.read_csv(args.trials)
    vols = pd.read_csv(args.audit_table)
    vols["n_volumes"] = vols["derivative_volumes"].fillna(vols["raw_volumes"])
    vols = vols.set_index(["subject", "run"])["n_volumes"]
    runs = [(k, g.sort_values("onset_seconds")) for k, g in trials.groupby(["subject", "run"])
            if k in vols.index and pd.notna(vols[k])]

    def one(key, g):
        res = run_diagnostics(
            g["onset_seconds"], g["condition"], int(vols[key]), f.TR_SECONDS,
            response_offsets=g["response_latency_seconds"].fillna(0.0),
            post_offsets=g["response_latency_seconds"] + g["confidence_report_time_seconds"])
        return {"subject": key[0], "run": key[1], **res}

    table = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(delayed(one)(k, g) for k, g in runs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    num = table.drop(columns=["subject", "run"])
    _write_json({
        "dataset_id": "ds002158", "data_kind": "design_diagnostics_from_event_timing",
        "tr_seconds": f.TR_SECONDS, "hrf_model": "glover", "high_pass_hz": 1 / 128,
        "n_runs": len(table),
        "post_event_definition": "onset + response latency + response_time2 (assumes the "
                                 "rating time is measured from the response; not verified)",
        "quantiles_across_runs": num.quantile([0.0, 0.5, 0.95, 1.0]).round(4)
        .to_dict(orient="index"),
        "table": str(out.with_suffix(".csv")),
    }, args.out)


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
    arbitration = [ts.arbitration_effect_recovery(
        schedules, e, **kw, model=cfg.get("arbitration_model", "hybrid_wsls_arbitration"))
        for e in cfg.get("arbitration_effects", [])]
    if cfg.get("recovery_table"):
        Path(cfg["recovery_table"]).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rec.values()).to_csv(cfg["recovery_table"], index=False)
    _write_json({"config": cfg, "data_kind": f"simulated_agents_on_{kind}",
                 "parameter_ranges": {**ts.RECOVERY_RANGES, **kw["ranges"]},
                 "parameter_recovery": {m: ts.summarize_recovery(r, m) for m, r in rec.items()},
                 "model_recovery_confusion": confusion,
                 "group_model_recovery_confusion": group_confusion,
                 "contingency_effect_recovery": contrast,
                 "arbitration_effect_recovery": arbitration},
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
                     replication_run=cfg.get("replication_run"),
                     arbitration_null_reps=cfg.get("arbitration_null_reps", 0),
                     arbitration_tests=cfg.get("arbitration_tests", []) or [])
    fits = res.pop("fits")
    if cfg.get("fits_table"):
        fits.to_csv(cfg["fits_table"], index=False)
    _write_json({"config": cfg, "data_kind": cfg.get("data_kind", "unspecified"), **res},
                cfg.get("output"))


def _two_step_eb(args):
    from . import two_step_model as ts

    cfg = yaml.safe_load(Path(args.config).read_text())
    eb_kw = {"n_iter": cfg.get("eb_n_iter", 30), "tol": cfg.get("eb_tol", 0.01)}
    common = {"seed": cfg.get("seed", 0), "n_starts": cfg.get("n_starts", 2),
              "prior_sd": cfg.get("prior_sd", 2.5), "n_jobs": cfg.get("n_jobs", 1)}
    out = {"config": cfg}
    if cfg.get("recovery_models"):
        schedules, kind = _two_step_schedules(cfg)
        ranges = {k: tuple(v) for k, v in (cfg.get("parameter_ranges") or {}).items()}
        out["recovery"] = {"data_kind": f"simulated_agents_on_{kind}", "results": [
            ts.empirical_bayes_recovery(m, schedules, ranges=ranges,
                                        reward=cfg.get("reward", "binary"), **common, **eb_kw)
            for m in cfg["recovery_models"]]}
    if cfg.get("models"):
        out["real_data"] = {"data_kind": cfg.get("data_kind", "unspecified"),
                            **ts.compare_empirical_bayes(
                                pd.read_csv(cfg["input"]), cfg["models"],
                                reward=cfg.get("reward", "binary"),
                                min_rt=cfg.get("min_rt", 0.15),
                                max_missed_fraction=cfg.get("max_missed_fraction", 0.10),
                                n_boot=cfg.get("n_boot", 2000),
                                replication_run=cfg.get("replication_run", "structural"),
                                **common, **eb_kw)}
    _write_json(out, cfg.get("output"))


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
    fe = dsub.add_parser("fetch-eeg", help="Download raw BrainVision EEG for named subjects "
                         "(signal data: needs an approved size budget)")
    fe.add_argument("--subjects", nargs="+", required=True)
    fe.add_argument("--dest", required=True)
    fe.add_argument("--tag", default="1.0.2")
    fe.add_argument("--approved-gb", type=float, required=True,
                    help="approved download budget in GB; the request fails if it is exceeded")
    fe.add_argument("--min-free-gb", type=float, default=20.0,
                    help="refuse to download if less than this would stay free on the "
                         "destination disk (default 20)")
    fe.add_argument("--dry-run", action="store_true", help="only report what would be fetched")
    ff = dsub.add_parser("fetch-fmri", help="Download the published derivative BOLD "
                         "(preproc.nii, ~3.2 GB/run) and realignment parameters for named "
                         "subjects (signal data: needs an approved size budget)")
    ff.add_argument("--subjects", nargs="+", required=True)
    ff.add_argument("--dest", required=True)
    ff.add_argument("--tag", default="1.0.2")
    ff.add_argument("--approved-gb", type=float, required=True,
                    help="approved download budget in GB; the request fails if it is exceeded")
    ff.add_argument("--min-free-gb", type=float, default=20.0,
                    help="refuse to download if less than this would stay free on the "
                         "destination disk (default 20)")
    ff.add_argument("--dry-run", action="store_true", help="only report what would be fetched")
    cp = dsub.add_parser("compare-paper", help="Behavioral quantities comparable to the "
                         "original publication (Pereira et al. 2020); see docs/VALIDATION.md")
    cp.add_argument("--trials", required=True)
    cp.add_argument("--out", help="JSON report path (default: print)")
    fu = dsub.add_parser("fusion", help="Behavior + post-decision EEG fusion for "
                         "confidence: participant-disjoint and within-subject ablation "
                         "(configs/ds002158_fusion.yaml)")
    fu.add_argument("--config", required=True)
    fu.add_argument("--epochs-dir")
    fu.add_argument("--out", help="JSON report path")
    fu.add_argument("--predictions", help="directory for per-trial prediction tables")
    ep = dsub.add_parser("eeg-preprocess", help="Clean raw in-scanner EEG (gradient + BCG "
                         "subtraction, causal filter) and cut trial epochs")
    ep.add_argument("--bids-root", required=True)
    ep.add_argument("--trials", required=True, help="trial table from `ldm ds002158 trials`")
    ep.add_argument("--subjects", nargs="+", required=True)
    ep.add_argument("--bcg-method", nargs="+", choices=["aas", "obs", "ica", "aas_ica"],
                    default=["aas"],
                    help="heartbeat artifact removal: moving-average template (aas, default), "
                         "optimal basis set (obs), heartbeat-locked ICA components (ica), or "
                         "aas followed by ica (aas_ica). Several values share one gradient "
                         "pass; --out-dir and --report then need a {method} placeholder")
    ep.add_argument("--obs-n-pc", type=int, default=4,
                    help="principal components per heartbeat for obs (default 4, Niazy 2005)")
    ep.add_argument("--out-dir", required=True)
    ep.add_argument("--report", help="JSON QC report")
    ep.add_argument("--l-freq", type=float, default=1.0)
    ep.add_argument("--h-freq", type=float, default=10.0)
    ed = dsub.add_parser("eeg-decode", help="Within- and cross-subject decoding from "
                         "preprocessed epochs (config-driven)")
    ed.add_argument("--config", required=True)
    ed.add_argument("--epochs-dir", help="override the config's epochs_dir (e.g. an external drive)")
    ed.add_argument("--out")
    eb = dsub.add_parser("eeg-bcg-retention", help="Plant a response-locked signal in real "
                         "gradient-corrected EEG and measure how much each heartbeat-artifact "
                         "method keeps")
    eb.add_argument("--bids-root", required=True)
    eb.add_argument("--epochs-dir", required=True, help="preprocessed trial tables (response "
                    "times and the runs used)")
    eb.add_argument("--subjects", nargs="+", required=True)
    eb.add_argument("--amplitude-uv", type=float, default=2.0)
    eb.add_argument("--seed", type=int, default=0)
    eb.add_argument("--out")
    ec = dsub.add_parser("eeg-cardiac", help="Within-subject association of per-trial "
                         "heart period and R-peak timing with confidence and accuracy")
    ec.add_argument("--config", required=True, help="decoding config (subjects, "
                    "confirmatory_subjects, min_trials, min_runs)")
    ec.add_argument("--epochs-dir", help="override the config's epochs_dir")
    ec.add_argument("--out")
    t = dsub.add_parser("trials", help="Parse events into a validated trial table")
    t.add_argument("--bids-root", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--issues")
    t.add_argument("--lenient", action="store_true", help="Report errors instead of raising")
    fa = dsub.add_parser("fmri-audit", help="fMRI raw/derivative inventory, NIfTI headers "
                         "(range requests, no image data) and realignment-parameter summary")
    fa.add_argument("--dest", default="data/ds002158")
    fa.add_argument("--tag", default="1.0.2")
    fa.add_argument("--out", default="reports/ds002158-fmri-audit.json")
    fd = dsub.add_parser("fmri-design", help="Single-trial and phase separability of the "
                         "fMRI design, from event timing only")
    fd.add_argument("--trials", default="data/ds002158_trials.csv")
    fd.add_argument("--audit-table", default="reports/ds002158-fmri-audit.csv")
    fd.add_argument("--out", default="reports/ds002158-fmri-design.json")
    fd.add_argument("--n-jobs", type=int, default=8)
    fl = dsub.add_parser("fmri-lss", help="Single-trial (LSS) ROI betas and QC from the "
                         "published derivative images (see configs/ds002158_fmri_pilot.yaml)")
    fl.add_argument("--config", required=True)
    fl.add_argument("--bids-root", help="dataset root holding derivatives/fmriprep")
    fl.add_argument("--out-dir")
    fl.add_argument("--subjects", nargs="+")
    fl.add_argument("--roi-ts", help="reuse ROI time series saved by an earlier fmri-lss "
                    "run (no image is read)")
    fl.add_argument("--report", help="JSON report path (default: print)")
    ft = dsub.add_parser("fmri-timing", help="Onset-shift scan of occipital and "
                         "sensorimotor ROI means: is the response earlier/later than the "
                         "canonical HRF predicts?")
    ft.add_argument("--config", required=True)
    ft.add_argument("--betas-dir", help="fmri-lss output directory (holds roi_ts/)")
    ft.add_argument("--bids-root")
    ft.add_argument("--out", help="JSON report path (default: print)")
    fp = dsub.add_parser("fmri-pilot", help="Positive controls and exploratory decoding on "
                         "LSS ROI betas")
    fp.add_argument("--config", required=True)
    fp.add_argument("--betas-dir")
    fp.add_argument("--out", help="JSON report path (default: print)")
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
    tssub.add_parser("eb", help="Empirical-Bayes (hierarchical) fitting vs fixed-prior MAP: "
                     "recovery and real-data held-out comparison").add_argument(
        "--config", required=True)
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
        {"recovery": _two_step_recovery, "compare": _two_step_compare,
         "eb": _two_step_eb}[args.ts_cmd](args)
    elif args.cmd == "ds007474":
        {"fetch-behavior": _ds007474_fetch, "trials": _ds007474_trials}[args.d7_cmd](args)
    elif args.cmd == "ds002158":
        {"fetch-metadata": _ds002158_fetch, "trials": _ds002158_trials,
         "align-eeg": _ds002158_align, "fetch-eeg": _ds002158_fetch_eeg,
         "fetch-fmri": _ds002158_fetch_fmri, "fusion": _ds002158_fusion,
         "compare-paper": _ds002158_compare_paper,
         "eeg-preprocess": _ds002158_eeg_preprocess, "eeg-decode": _ds002158_eeg_decode,
         "eeg-cardiac": _ds002158_eeg_cardiac, "eeg-bcg-retention": _ds002158_eeg_bcg_retention, "fmri-audit": _ds002158_fmri_audit,
         "fmri-design": _ds002158_fmri_design,
         "fmri-lss": _ds002158_fmri_lss, "fmri-pilot": _ds002158_fmri_pilot,
         "fmri-timing": _ds002158_fmri_timing}[args.ds_cmd](args)


if __name__ == "__main__":
    main()
