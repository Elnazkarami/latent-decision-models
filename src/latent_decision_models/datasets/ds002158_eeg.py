"""ds002158 raw EEG: run pairing, in-scanner artifact removal and trial epochs.

Status: experimental (LDM-003). Verified facts (``docs/ds002158-eeg.md``):

* Raw BrainVision, 5000 Hz, 64 channels (Ch32 = ECG, stored as type EEG), INT_16.
  Headers name the original lab files (``DataFile=s4_run1_...``); see
  :func:`io.brainvision.read_raw`.
* ``R128`` volume triggers are 6400 samples (1.28 s) apart: EEG and scanner clocks are
  synchronised, so gradient templates need no sub-sample realignment. One trigger in
  sub-04 run-004 is logged 2 samples late; the artifact stays on the grid, which is used.
* The raw files still contain MR gradient and BCG artifacts (the authors removed them
  only in ``derivatives/eegprep``).

Pipeline per run (all filters causal, see :mod:`eeg_clean`):

1. pair the EEG run with an events run by marker *timing* (never by filename);
2. crop to the scan; gradient AAS (21-volume window) at 5000 Hz;
3. causal anti-alias low-pass and decimation to 250 Hz;
4. R-peak detection on ECG (fallback TP10); BCG AAS (21-beat window);
5. causal band-pass (default 1-10 Hz, the authors' band); flag bad channels
   (robust z of log SD > 3.5); common average of the good EEG channels;
6. epochs locked to the stimulus marker and to the response marker.

Nothing here uses trial labels, so it can run before any train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import eeg_clean
from ..io.brainvision import read_raw, read_vmrk
from . import ds002158 as d

SFREQ_RAW = 5000.0
TR_SAMPLES = 6400  # 1.28 s at 5000 Hz; checked per run by eeg_clean.gradient_correct
DECIMATE = 20  # -> 250 Hz
ECG_CHANNEL = "ECG"
FALLBACK_HEART_CHANNEL = "TP10"  # the authors' choice for heartbeat detection

# Analysis windows (s). Pre-decision ends before the earliest kept response; response-
# locked spans the press; post-decision ends before the confidence scale (~ +1.5 s,
# marker S171/S43, unverified) so it never contains the rating itself.
WINDOWS = {
    "pre_decision": ("stimulus", 0.0, 0.2),
    "response_locked": ("response", -0.2, 0.1),
    "post_decision": ("response", 0.1, 0.6),
}
STIM_EPOCH = (-0.2, 1.2)
RESP_EPOCH = (-0.6, 1.0)
# Label-free epoch rejection on good channels, per subject: log peak-to-peak more than
# REJECT_ROBUST_Z robust SDs above the subject median, or above an absolute ceiling.
# (A fixed 150 uV threshold, chosen before seeing the data, rejected 31% of sub-04's
# epochs because in-scanner epochs have a median peak-to-peak of ~130 uV; it was replaced
# before any decoding was run.)
REJECT_ROBUST_Z = 3.5
REJECT_CEILING_UV = 500.0


@dataclass
class CleanRun:
    data: np.ndarray  # (n_channels, n_samples) uV, 250 Hz, scan-cropped, CAR
    sfreq: float
    ch_names: list[str]
    first_sample_raw: int  # raw (5000 Hz) sample of data[:, 0]
    bad_channels: list[str]
    qc: dict = field(default_factory=dict)
    heartbeat_peaks: np.ndarray | None = None  # R peaks, samples of ``data`` (250 Hz)

    def to_clean_sample(self, raw_sample) -> np.ndarray:
        return np.round((np.asarray(raw_sample, float) - self.first_sample_raw)
                        / (SFREQ_RAW / self.sfreq)).astype(int)


def pair_runs(eeg_markers: dict[str, pd.DataFrame], trials: pd.DataFrame) -> pd.DataFrame:
    """Try every (EEG run, events run) pair of one subject; keep full timing matches.

    Returns one row per EEG run with the matched events run (or None) and the alignment
    summary. A pairing is accepted only if *every* trial matched in time and it is the
    only events run that does.
    """
    rows = []
    for eeg_run, markers in sorted(eeg_markers.items()):
        hits = []
        for ev_run, tr in trials.groupby("run"):
            _, s = d.align_eeg_markers(tr, markers)
            if s["n_timing_matched"] == s["n_trials"]:
                hits.append((ev_run, s))
        if len(hits) == 1:
            ev_run, s = hits[0]
            rows.append({"eeg_run": eeg_run, "events_run": ev_run,
                         "same_name": eeg_run == ev_run, **s})
        else:
            rows.append({"eeg_run": eeg_run, "events_run": None, "same_name": False,
                         "status": "ambiguous" if hits else "unmatched",
                         "n_candidate_runs": len(hits)})
    return pd.DataFrame(rows)


def _heartbeat_rms(x: np.ndarray, peaks: np.ndarray, sfreq: float) -> float:
    """RMS over channels and 0-0.6 s of the heartbeat-locked average (1-10 Hz, causal)."""
    y = eeg_clean.causal_bandpass(x, sfreq, 1.0, 10.0)
    n = int(0.6 * sfreq)
    p = peaks[(peaks >= 0) & (peaks + n <= y.shape[1])]
    if len(p) == 0:
        return float("nan")
    avg = y[:, p[:, None] + np.arange(n)].mean(axis=1)
    return float(np.sqrt(np.mean(avg**2)))


BCG_METHODS = ("aas", "obs", "ica", "aas_ica")


def clean_run(vhdr: str | Path, l_freq: float = 1.0, h_freq: float = 10.0,
              n_template: int = 21, bcg_method: str = "aas", obs_n_pc: int = 4) -> CleanRun:
    """Steps 2-5 of the module pipeline for one run (see module docstring).

    ``bcg_method``: ``"aas"`` (21-beat moving-average template, the default), ``"obs"``
    (optimal basis set, mean + ``obs_n_pc`` principal components; Niazy et al. 2005),
    ``"ica"`` (heartbeat-locked independent components removed) or ``"aas_ica"`` (AAS,
    then ICA on what AAS left behind).
    """
    return clean_run_methods(vhdr, [bcg_method], l_freq, h_freq, n_template,
                             obs_n_pc)[bcg_method]


def clean_run_methods(vhdr: str | Path, bcg_methods, l_freq: float = 1.0,
                      h_freq: float = 10.0, n_template: int = 21,
                      obs_n_pc: int = 4) -> dict[str, CleanRun]:
    """:func:`clean_run` for several BCG methods, sharing the gradient-correction stage."""
    bad_m = [m for m in bcg_methods if m not in BCG_METHODS]
    if bad_m:
        raise ValueError(f"unknown bcg_method {bad_m}; choose from {BCG_METHODS}")
    x, sfreq, names, s0, qc, hb = _gradient_stage(Path(vhdr), n_template)
    eeg_idx = [i for i, c in enumerate(names) if c != ECG_CHANNEL]
    return {m: _bcg_and_filter(x.copy(), sfreq, names, eeg_idx, s0, dict(qc), hb, m,
                               l_freq, h_freq, n_template, obs_n_pc)
            for m in bcg_methods}


def _gradient_stage(vhdr: Path, n_template: int):
    raw = read_raw(vhdr)
    if raw.info["sfreq"] != SFREQ_RAW:
        raise ValueError(f"{vhdr.name}: expected {SFREQ_RAW} Hz, got {raw.info['sfreq']}")
    names = list(raw.ch_names)
    markers = read_vmrk(vhdr.with_suffix(".vmrk"), SFREQ_RAW)
    vol = markers.loc[markers["description"] == d.VOLUME_MARKER, "sample"].to_numpy()
    # Crop on the volume grid, not on the first logged trigger (which may be logged late).
    grid, n_snapped = eeg_clean.volume_grid(vol, TR_SAMPLES)
    # Keep complete volumes only: the EEG recording may stop before the scan ends
    # (sub-02 run-004 ends 0.8 s into its last volume, after the last trial).
    complete = (grid >= 0) & (grid + TR_SAMPLES <= raw.n_times)
    n_incomplete = int((~complete).sum())
    vol, grid = vol[complete], grid[complete]
    s0 = int(grid[0])
    n_vol = len(vol)
    x = (raw.get_data(start=s0, stop=s0 + n_vol * TR_SAMPLES) * 1e6).astype(np.float32)
    del raw
    qc = {"n_volumes": n_vol, "n_volumes_incomplete_dropped": n_incomplete,
          "first_volume_raw_sample": s0,
          "n_volume_triggers_snapped_to_grid": n_snapped,
          "power_before_gradient_correction": eeg_clean.band_power(
              x[:, : 60 * int(SFREQ_RAW)], SFREQ_RAW, (5, 10, 12.5, 25))}
    x = eeg_clean.gradient_correct(x, vol - s0, TR_SAMPLES, n_template)
    qc["power_after_gradient_correction"] = eeg_clean.band_power(
        x[:, : 60 * int(SFREQ_RAW)], SFREQ_RAW, (5, 10, 12.5, 25))
    x, sfreq = eeg_clean.causal_decimate(x, SFREQ_RAW, DECIMATE)

    hb = eeg_clean.detect_heartbeats(x[names.index(ECG_CHANNEL)], sfreq)
    hb["channel"] = ECG_CHANNEL
    if not (40 <= hb["heart_rate_bpm"] <= 120) or hb["fraction_rr_implausible"] > 0.15:
        alt = eeg_clean.detect_heartbeats(x[names.index(FALLBACK_HEART_CHANNEL)], sfreq)
        if alt["fraction_rr_implausible"] < hb["fraction_rr_implausible"]:
            hb = {**alt, "channel": FALLBACK_HEART_CHANNEL}
    return x, sfreq, names, s0, qc, hb


def _bcg_and_filter(x, sfreq, names, eeg_idx, s0, qc, hb, bcg_method, l_freq, h_freq,
                    n_template, obs_n_pc) -> CleanRun:
    qc["bcg_evoked_rms_uv_before"] = _heartbeat_rms(x[eeg_idx], hb["peaks"], sfreq)
    var_before = float(x[eeg_idx].var(axis=1).mean())
    corrected = np.ones(len(hb["peaks"]), bool)
    if bcg_method in ("aas", "aas_ica"):
        x[eeg_idx], corrected = eeg_clean.bcg_correct(x[eeg_idx], hb["peaks"], sfreq,
                                                      hb["median_rr_s"], n_template)
    elif bcg_method == "obs":
        x[eeg_idx], corrected = eeg_clean.bcg_correct_obs(x[eeg_idx], hb["peaks"], sfreq,
                                                          hb["median_rr_s"], obs_n_pc)
    if bcg_method in ("ica", "aas_ica"):
        x[eeg_idx], qc["ica"] = eeg_clean.ica_remove_cardiac(x[eeg_idx], hb["peaks"], sfreq,
                                                            hb["median_rr_s"])
    qc["bcg_method"] = bcg_method
    qc["bcg_variance_removed_fraction"] = 1 - float(x[eeg_idx].var(axis=1).mean()) / var_before
    qc["heartbeat"] = {k: v for k, v in hb.items() if k != "peaks"}
    qc["heartbeat"]["n_corrected"] = int(corrected.sum())
    qc["bcg_evoked_rms_uv_after"] = _heartbeat_rms(x[eeg_idx], hb["peaks"], sfreq)

    x = eeg_clean.causal_bandpass(x, sfreq, l_freq, h_freq)
    # Bad channels: robust z of log SD over EEG channels (skip the first 10 s of filter
    # settling).
    settle = int(10 * sfreq)
    sd = np.log(x[eeg_idx, settle:].std(axis=1))
    mad = np.median(np.abs(sd - np.median(sd))) * 1.4826
    z = (sd - np.median(sd)) / (mad if mad > 0 else 1.0)
    bad = [names[eeg_idx[i]] for i in np.flatnonzero(np.abs(z) > 3.5)]
    good = [i for i in eeg_idx if names[i] not in bad]
    x[eeg_idx] -= x[good].mean(axis=0, keepdims=True)
    qc.update(bad_channels=bad, band=[l_freq, h_freq], sfreq=sfreq,
              filter="causal Butterworth (one-pass)",
              median_channel_sd_uv=float(np.median(x[good, settle:].std(axis=1))))
    return CleanRun(x, sfreq, names, s0, bad, qc, heartbeat_peaks=np.asarray(hb["peaks"]))


def epoch(run: CleanRun, raw_samples, tmin: float, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Epochs ``(n, n_channels, n_times)`` around raw-sample events; NaN where out of range."""
    centre = run.to_clean_sample(raw_samples)
    rel = np.arange(round(tmin * run.sfreq), round(tmax * run.sfreq))
    out = np.full((len(centre), run.data.shape[0], len(rel)), np.nan, np.float32)
    for i, c in enumerate(centre):
        idx = c + rel
        if idx[0] >= 0 and idx[-1] < run.data.shape[1]:
            out[i] = run.data[:, idx]
    return out, rel / run.sfreq


def run_epochs(run: CleanRun, trials: pd.DataFrame, markers: pd.DataFrame) -> dict:
    """Align one run's trials to its markers and cut stimulus- and response-locked epochs."""
    table, summary = d.align_eeg_markers(trials, markers)
    t = trials.reset_index(drop=True).merge(table, on="trial_index", how="left")
    stim_raw = t["eeg_stimulus_sample"].astype(float).to_numpy()
    lat = t["eeg_response_latency_seconds"].astype(float).to_numpy()
    resp_raw = stim_raw + lat * SFREQ_RAW
    stim, st_times = epoch(run, np.nan_to_num(stim_raw, nan=-1e9), *STIM_EPOCH)
    resp, rs_times = epoch(run, np.nan_to_num(resp_raw, nan=-1e9), *RESP_EPOCH)
    resp[np.isnan(resp_raw)] = np.nan
    good = [i for i, c in enumerate(run.ch_names) if c != ECG_CHANNEL
            and c not in run.bad_channels]
    with np.errstate(invalid="ignore"):
        ptp = np.nanmax(np.ptp(stim[:, good], axis=2), axis=1)
    t["eeg_epoch_ptp_uv"] = ptp  # rejection is decided per subject: reject_epochs()
    t["eeg_has_response"] = ~np.isnan(resp_raw)
    if run.heartbeat_peaks is not None:
        f = SFREQ_RAW / run.sfreq
        hb = eeg_clean.cardiac_trial_features(
            run.heartbeat_peaks, (stim_raw - run.first_sample_raw) / f,
            (resp_raw - run.first_sample_raw) / f, run.sfreq)
        for k, v in hb.items():
            t[k] = v
    return {"trials": t, "stim": stim, "stim_times": st_times, "resp": resp,
            "resp_times": rs_times, "alignment": summary}


_EEG_RUN = r"(sub-\d+)_ses-001_task-main_(run-\d+)_eeg\.vhdr$"


def reject_epochs(t: pd.DataFrame, robust_z: float = REJECT_ROBUST_Z,
                  ceiling_uv: float = REJECT_CEILING_UV) -> pd.DataFrame:
    """Flag epochs by a robust outlier rule on log peak-to-peak (labels are never used)."""
    ptp = t["eeg_epoch_ptp_uv"].to_numpy(float)
    lp = np.log(ptp)
    med = np.nanmedian(lp)
    mad = 1.4826 * np.nanmedian(np.abs(lp - med))
    z = (lp - med) / mad if mad > 0 else np.zeros_like(lp)
    bad = np.isnan(ptp) | (z > robust_z) | (ptp > ceiling_uv)
    return t.assign(eeg_epoch_ptp_robust_z=z, eeg_rejected=bad)


def preprocess_subject(bids_root: str | Path, subject: str, trials: pd.DataFrame,
                       out_dir: str | Path, bcg_method: str = "aas", **clean_kwargs) -> dict:
    """Pair, clean and epoch every EEG run of one subject; write epochs and QC.

    Writes ``<out_dir>/<subject>_epochs.npz`` (stimulus- and response-locked epochs),
    ``<subject>_trials.csv`` (one row per epoch, with alignment, rejection and per-trial
    cardiac columns) and returns a QC dict.
    """
    return preprocess_subject_methods(bids_root, subject, trials, {bcg_method: out_dir},
                                      **clean_kwargs)[bcg_method]


def preprocess_subject_methods(bids_root: str | Path, subject: str, trials: pd.DataFrame,
                               out_dirs: dict, **clean_kwargs) -> dict:
    """:func:`preprocess_subject` for several BCG methods ({method: out_dir}) in one pass.

    The gradient correction and heartbeat detection run once per EEG run and are shared.
    Returns {method: QC dict}.
    """
    import re

    from ..io.brainvision import read_vhdr_sampling_rate

    eeg_dir = Path(bids_root, subject, "ses-001", "eeg")
    # Skip macOS "._*" AppleDouble files (written on exFAT/FAT external drives).
    vhdrs = {re.search(_EEG_RUN, p.name)[2]: p for p in sorted(eeg_dir.glob("*_eeg.vhdr"))
             if not p.name.startswith(".")}
    if not vhdrs:
        raise FileNotFoundError(f"no EEG headers in {eeg_dir}")
    sub_trials = trials[trials["subject"] == subject]
    markers = {r: read_vmrk(p.with_suffix(".vmrk"), read_vhdr_sampling_rate(p))
               for r, p in vhdrs.items()}
    pairs = pair_runs(markers, sub_trials)
    methods = list(out_dirs)
    acc = {m: {"parts": [], "stim": [], "resp": [], "runs": []} for m in methods}
    for row in pairs.itertuples(index=False):
        rq = {"eeg_run": row.eeg_run, "events_run": row.events_run, "status": row.status}
        if row.events_run is None or row.status != "verified":
            for m in methods:
                acc[m]["runs"].append({**rq, "used": False})
            continue
        cleaned = clean_run_methods(vhdrs[row.eeg_run], methods, **clean_kwargs)
        for m, cr in cleaned.items():
            ep = run_epochs(cr, sub_trials[sub_trials["run"] == row.events_run],
                            markers[row.eeg_run])
            t = ep["trials"].assign(eeg_run=row.eeg_run,
                                    eeg_bad_channels=";".join(cr.bad_channels))
            acc[m]["parts"].append(t)
            acc[m]["stim"].append(ep["stim"])
            acc[m]["resp"].append(ep["resp"])
            acc[m]["runs"].append({**rq, "used": True, **cr.qc, "n_trials": len(t)})
            acc[m]["times"] = (ep["stim_times"], ep["resp_times"])
        del cleaned
    if not acc[methods[0]]["parts"]:
        raise ValueError(f"{subject}: no verified EEG run")
    ch_names = np.array(ep_channel_names(vhdrs))
    result = {}
    for m, a in acc.items():
        out = Path(out_dirs[m])
        out.mkdir(parents=True, exist_ok=True)
        table = reject_epochs(pd.concat(a["parts"], ignore_index=True))
        for rq in a["runs"]:
            if rq.get("used"):
                sel = table["eeg_run"] == rq["eeg_run"]
                rq.update(n_rejected=int(table.loc[sel, "eeg_rejected"].sum()),
                          n_valid_kept=int((table.loc[sel, "valid_trial"]
                                            & ~table.loc[sel, "eeg_rejected"]).sum()))
        table.to_csv(out / f"{subject}_trials.csv", index=False)
        np.savez_compressed(out / f"{subject}_epochs.npz", stim=np.concatenate(a["stim"]),
                            resp=np.concatenate(a["resp"]), stim_times=a["times"][0],
                            resp_times=a["times"][1], ch_names=ch_names)
        result[m] = {"subject": subject, "bcg_method": m, "n_eeg_runs": len(vhdrs),
                     "n_runs_used": int(sum(r["used"] for r in a["runs"])),
                     "n_epochs": len(table), "n_rejected": int(table["eeg_rejected"].sum()),
                     "runs": a["runs"], "pairing": pairs.to_dict(orient="records")}
    return result


def ep_channel_names(vhdrs: dict) -> list[str]:
    """Channel names from the first header (identical across runs; checked by MNE read)."""
    from ..io.brainvision import read_raw

    return list(read_raw(next(iter(vhdrs.values())), preload=False).ch_names)


def load_epochs(epochs_dir: str | Path, subject: str) -> tuple[pd.DataFrame, dict]:
    z = np.load(Path(epochs_dir, f"{subject}_epochs.npz"))
    t = pd.read_csv(Path(epochs_dir, f"{subject}_trials.csv"))
    return t, {k: z[k] for k in z.files}


def window_data(ep: dict, window: str,
                windows: dict | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    lock, tmin, tmax = (windows or WINDOWS)[window]
    return ((ep["stim"], ep["stim_times"]) if lock == "stimulus"
            else (ep["resp"], ep["resp_times"])) + (tmin, tmax)


def decoding_trials(t: pd.DataFrame, condition: str, window: str,
                    windows: dict | None = None) -> np.ndarray:
    """Rows used for one analysis: valid, not rejected, the condition, a response marker,
    and (pre-decision) a response at least as late as the window end."""
    m = (t["valid_trial"].astype(bool) & ~t["eeg_rejected"].astype(bool)
         & (t["condition"] == condition) & t["eeg_has_response"].astype(bool))
    lock, _, tmax = (windows or WINDOWS)[window]
    if lock == "stimulus":
        m &= t["eeg_response_latency_seconds"] >= tmax
    return m.to_numpy()


TARGET_COLUMNS = {"choice_side": "choice_side", "accuracy": "accuracy",
                  "confidence": "confidence_raw"}


def _target(t: pd.DataFrame, spec: dict) -> np.ndarray:
    name = spec.get("target", spec["_name"])
    if spec.get("column"):  # any trial-table column, e.g. a cardiac timing variable
        return t[spec["column"]].to_numpy(float)
    if name == "choice_side":
        return (t[TARGET_COLUMNS[name]] == "right").to_numpy(float)
    return t[TARGET_COLUMNS[name]].to_numpy(float)


def _analysis_inputs(t, ep, chans, condition, window, spec, bin_s, windows=None):
    from ..eeg_decoding import window_features

    rows = decoding_trials(t, condition, window, windows)
    # Trials missing the target column, a covariate or a required column are dropped
    # (e.g. cardiac features are NaN where a heartbeat interval was implausible).
    need = (list(spec.get("covariates", [])) + list(spec.get("require_columns", []))
            + ([spec["column"]] if spec.get("column") else []))
    if need:
        rows = rows & t[need].notna().all(axis=1).to_numpy()
    data, times, tmin, tmax = window_data(ep, window, windows)
    X = window_features(data[rows][:, chans], times, tmin, tmax, bin_s)
    tt = t[rows]
    y = _target(tt, spec)
    cov = (tt[spec["covariates"]].to_numpy(float) if spec.get("covariates") else None)
    return X, y, tt["run"].to_numpy(), cov


def run_decoding(cfg: dict) -> dict:
    """Within-subject (leave-one-run-out) and cross-subject decoding per the config.

    Config keys (defaults reproduce the two-subject analysis): ``score_mode`` ("pooled" or
    "per_group"), ``n_perm``; ``loso`` (train on all other subjects, test on each one) or,
    without it, pairwise transfer between subjects; ``confirmatory_subjects`` with
    ``min_runs`` / ``min_trials`` for across-subject tests (:func:`eeg_decoding.group_test`),
    Holm-corrected over the targets marked ``primary``.

    Target spec keys: ``task``, ``target`` (a name in ``TARGET_COLUMNS``) or ``column``
    (any trial-table column), ``covariates``, ``require_columns`` (drop trials where these
    are missing, e.g. to compare on the same trials as a covariate analysis), ``primary``.

    ``exclude_channels`` drops further channels for every subject; ``channels`` uses
    exactly the listed channels (e.g. ``[ECG]`` as a cardiac control). ``custom_windows`` ({name: [lock, tmin, tmax]}, lock "stimulus" or "response") adds
    analysis windows, e.g. a sliding time course. ``holm_by`` (list of columns, e.g.
    ``[analysis, target, condition]``) adds ``p_holm_by``: group-test p-values Holm-corrected
    within each such family (e.g. across the windows of one time course).
    """
    from joblib import Parallel, delayed

    from ..eeg_decoding import decode, group_test, holm, transfer

    loaded = {s: load_epochs(cfg["epochs_dir"], s) for s in cfg["subjects"]}
    names = [str(c) for c in next(iter(loaded.values()))[1]["ch_names"]]
    bad = set()
    for t, ep in loaded.values():
        if [str(c) for c in ep["ch_names"]] != names:
            raise ValueError("channel lists differ between subjects")
        for b in t["eeg_bad_channels"].dropna():
            bad |= set(filter(None, str(b).split(";")))
    bad |= set(cfg.get("exclude_channels", []))  # e.g. to match another channel set
    if cfg.get("channels"):  # explicit channel list, e.g. [ECG] for a cardiac control
        chans = [names.index(c) for c in cfg["channels"]]
    else:
        chans = [i for i, c in enumerate(names) if c != ECG_CHANNEL and c not in bad]
    bin_s = cfg.get("bin_seconds", 0.05)
    windows = {**WINDOWS, **{k: tuple(v) for k, v in cfg.get("custom_windows", {}).items()}}
    for w in cfg["windows"]:
        if windows[w][0] not in ("stimulus", "response"):
            raise ValueError(f"window {w}: lock must be 'stimulus' or 'response'")
    mode = cfg.get("score_mode", "pooled")
    seed = cfg.get("seed", 0)
    specs = {k: {**v, "_name": k} for k, v in cfg["targets"].items()}
    cells = [(k, c, w) for k in specs for c in cfg["conditions"] for w in cfg["windows"]]
    inputs = {(s, *cell): _analysis_inputs(*loaded[s], chans, cell[1], cell[2],
                                           specs[cell[0]], bin_s, windows)
              for s in cfg["subjects"] for cell in cells}
    n_jobs = cfg.get("n_jobs", 1)

    def one(s, k, c, w):
        X, y, g, cov = inputs[(s, k, c, w)]
        r = decode(X, y, g, specs[k]["task"], cov, cfg.get("n_perm", 200), seed, mode)
        return {"subject": s, "target": k, "condition": c, "window": w, **r}

    within = pd.DataFrame(Parallel(n_jobs=n_jobs)(
        delayed(one)(s, *cell) for s in cfg["subjects"] for cell in cells))
    within["primary"] = within["target"].map(lambda k: bool(specs[k].get("primary")))
    within["p_holm_primary"] = np.nan
    for s in cfg["subjects"]:
        m = (within["subject"] == s) & within["primary"] & within["p_value"].notna()
        if m.any():
            within.loc[m, "p_holm_primary"] = holm(within.loc[m, "p_value"])

    def loso_one(s, k, c, w):
        tr = [inputs[(o, k, c, w)] for o in cfg["subjects"] if o != s]
        Xtr = np.concatenate([a[0] for a in tr])
        ytr = np.concatenate([a[1] for a in tr])
        ctr = None if tr[0][3] is None else np.concatenate([a[3] for a in tr])
        Xte, yte, gte, cte = inputs[(s, k, c, w)]
        r = transfer(Xtr, ytr, Xte, yte, specs[k]["task"], ctr, cte,
                     cfg.get("n_perm_transfer", 0), seed, gte, mode)
        return {"train_subject": "all_others", "test_subject": s, "target": k,
                "condition": c, "window": w, **r}

    if cfg.get("loso"):
        cross = pd.DataFrame(Parallel(n_jobs=n_jobs)(
            delayed(loso_one)(s, *cell) for s in cfg["subjects"] for cell in cells))
    elif len(cfg["subjects"]) >= 2:
        rows = []
        for a in cfg["subjects"]:
            for b in cfg["subjects"]:
                if a == b:
                    continue
                for k, c, w in cells:
                    Xa, ya, _, ca = inputs[(a, k, c, w)]
                    Xb, yb, gb, cb = inputs[(b, k, c, w)]
                    # Test-run groups only matter for per-run scoring; pooled scoring
                    # keeps the original whole-subject permutation (reproducibility).
                    r = transfer(Xa, ya, Xb, yb, specs[k]["task"], ca, cb,
                                 cfg.get("n_perm_transfer", 1000), seed,
                                 gb if mode == "per_group" else None, mode)
                    rows.append({"train_subject": a, "test_subject": b, "target": k,
                                 "condition": c, "window": w, **r})
        cross = pd.DataFrame(rows).sort_values(["target", "condition", "window",
                                                "train_subject"])
    else:
        cross = pd.DataFrame()

    group = []
    conf = cfg.get("confirmatory_subjects")
    if conf:
        n_trials = within.set_index(["subject", "target", "condition", "window"])["n_trials"]
        n_groups = within.set_index(["subject", "target", "condition", "window"])["n_groups"]
        # Pairwise transfer (loso off) has several rows per test subject, so only
        # leave-one-subject-out results enter subject-level group tests.
        analyses = [("within_subject", within, "subject")]
        if cfg.get("loso"):
            analyses.append(("leave_one_subject_out", cross, "test_subject"))
        for kind, table, subj_col in analyses:
            if table.empty:
                continue
            for k, c, w in cells:
                rows = table[(table["target"] == k) & (table["condition"] == c)
                             & (table["window"] == w) & table[subj_col].isin(conf)]
                elig = [r for r in rows.itertuples()
                        if n_trials[(getattr(r, subj_col), k, c, w)] >= cfg.get("min_trials", 40)
                        and n_groups[(getattr(r, subj_col), k, c, w)] >= cfg.get("min_runs", 3)]
                chance = 0.5 if specs[k]["task"] == "classification" else 0.0
                g = group_test([r.score for r in elig], chance, seed=seed)
                group.append({"analysis": kind, "target": k, "condition": c, "window": w,
                              "primary": bool(specs[k].get("primary")), "chance": chance, **g})
        group = pd.DataFrame(group)
        group["p_holm_primary"] = np.nan
        for kind in group["analysis"].unique():
            m = (group["analysis"] == kind) & group["primary"] & group["p_wilcoxon"].notna()
            if m.any():
                group.loc[m, "p_holm_primary"] = holm(group.loc[m, "p_wilcoxon"])
        if cfg.get("holm_by"):
            group["p_holm_by"] = np.nan
            tested = group[group["p_wilcoxon"].notna()]
            for idx in tested.groupby(cfg["holm_by"]).groups.values():
                group.loc[idx, "p_holm_by"] = holm(group.loc[idx, "p_wilcoxon"])
    return {"channels_used": [names[i] for i in chans],
            "excluded_channels": sorted(set(names) - {names[i] for i in chans}),
            "windows": {w: windows[w] for w in cfg["windows"]},
            "score_mode": mode, "within_subject": within, "cross_subject": cross,
            "group": pd.DataFrame(group)}


CARDIAC_MEASURES = ("hb_post_response_change_s", "hb_ibi_resp1_s", "hb_prev_rpeak_s",
                    "hb_next_rpeak_s")


def cardiac_behavior(epochs_dir: str | Path, subjects, confirmatory_subjects=None,
                     conditions=("active", "observed"), behaviors=("confidence", "accuracy"),
                     measures=CARDIAC_MEASURES, min_trials: int = 40, min_runs: int = 3,
                     seed: int = 0) -> dict:
    """Within-subject association of per-trial cardiac measures with behaviour.

    Per subject and condition: valid trials with a response marker and a finite measure
    (EEG epoch rejection is not applied; the heartbeat does not depend on it). Run means
    are removed from both variables (confidence drifts across runs), then Spearman's r.
    Across the confirmatory subjects with >= ``min_trials`` trials in >= ``min_runs``
    runs: two-sided Wilcoxon of r against 0, with a bootstrap CI, Holm-corrected over
    all cells. Returns {"per_subject": DataFrame, "group": DataFrame}.
    """
    from scipy.stats import spearmanr

    from ..eeg_decoding import group_test, holm

    cols = {"confidence": "confidence_raw", "accuracy": "accuracy"}
    rows = []
    for s in subjects:
        t = pd.read_csv(Path(epochs_dir, f"{s}_trials.csv"))
        base = t["valid_trial"].astype(bool) & t["eeg_has_response"].astype(bool)
        for c in conditions:
            for b in behaviors:
                for m in measures:
                    tt = t[base & (t["condition"] == c)
                           & t[m].notna() & t[cols[b]].notna()]
                    y = tt[m] - tt.groupby("run")[m].transform("mean")
                    x = tt[cols[b]] - tt.groupby("run")[cols[b]].transform("mean")
                    ok = len(tt) >= 3 and x.std() > 0 and y.std() > 0
                    rows.append({"subject": s, "condition": c, "behavior": b, "measure": m,
                                 "n_trials": len(tt), "n_runs": tt["run"].nunique(),
                                 "r": float(spearmanr(x, y)[0]) if ok else np.nan})
    per = pd.DataFrame(rows)
    conf = set(confirmatory_subjects or subjects)
    elig = per[per["subject"].isin(conf) & (per["n_trials"] >= min_trials)
               & (per["n_runs"] >= min_runs)]
    group = []
    for (c, b, m), g in elig.groupby(["condition", "behavior", "measure"], sort=False):
        group.append({"condition": c, "behavior": b, "measure": m,
                      **group_test(g["r"], 0.0, seed=seed, alternative="two-sided")})
    group = pd.DataFrame(group)
    if not group.empty:
        group["p_holm"] = np.nan
        tested = group["p_wilcoxon"].notna()
        group.loc[tested, "p_holm"] = holm(group.loc[tested, "p_wilcoxon"])
    return {"per_subject": per, "group": group}
