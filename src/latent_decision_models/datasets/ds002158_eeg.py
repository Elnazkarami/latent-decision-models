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


def clean_run(vhdr: str | Path, l_freq: float = 1.0, h_freq: float = 10.0,
              n_template: int = 21) -> CleanRun:
    """Steps 2-5 of the module pipeline for one run (see module docstring)."""
    vhdr = Path(vhdr)
    raw = read_raw(vhdr)
    if raw.info["sfreq"] != SFREQ_RAW:
        raise ValueError(f"{vhdr.name}: expected {SFREQ_RAW} Hz, got {raw.info['sfreq']}")
    names = list(raw.ch_names)
    markers = read_vmrk(vhdr.with_suffix(".vmrk"), SFREQ_RAW)
    vol = markers.loc[markers["description"] == d.VOLUME_MARKER, "sample"].to_numpy()
    s0 = int(vol[0])
    # Keep complete volumes only: the EEG recording may stop before the scan ends
    # (sub-02 run-004 ends 0.8 s into its last volume, after the last trial).
    complete = vol + TR_SAMPLES <= raw.n_times
    n_incomplete = int((~complete).sum())
    vol = vol[complete]
    n_vol = len(vol)
    x = (raw.get_data(start=s0, stop=s0 + n_vol * TR_SAMPLES) * 1e6).astype(np.float32)
    del raw
    qc = {"n_volumes": n_vol, "n_volumes_incomplete_dropped": n_incomplete,
          "first_volume_raw_sample": s0,
          "n_volume_triggers_snapped_to_grid": eeg_clean.volume_grid(vol, TR_SAMPLES)[1],
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
    eeg_idx = [i for i, c in enumerate(names) if c != ECG_CHANNEL]
    qc["bcg_evoked_rms_uv_before"] = _heartbeat_rms(x[eeg_idx], hb["peaks"], sfreq)
    x[eeg_idx], corrected = eeg_clean.bcg_correct(x[eeg_idx], hb["peaks"], sfreq,
                                                  hb["median_rr_s"], n_template)
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
    return CleanRun(x, sfreq, names, s0, bad, qc)


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
                       out_dir: str | Path, **clean_kwargs) -> dict:
    """Pair, clean and epoch every EEG run of one subject; write epochs and QC.

    Writes ``<out_dir>/<subject>_epochs.npz`` (stimulus- and response-locked epochs),
    ``<subject>_trials.csv`` (one row per epoch, with alignment and rejection columns)
    and returns a QC dict.
    """
    import re

    from ..io.brainvision import read_vhdr_sampling_rate

    eeg_dir = Path(bids_root, subject, "ses-001", "eeg")
    vhdrs = {re.search(_EEG_RUN, p.name)[2]: p for p in sorted(eeg_dir.glob("*_eeg.vhdr"))}
    if not vhdrs:
        raise FileNotFoundError(f"no EEG headers in {eeg_dir}")
    sub_trials = trials[trials["subject"] == subject]
    markers = {r: read_vmrk(p.with_suffix(".vmrk"), read_vhdr_sampling_rate(p))
               for r, p in vhdrs.items()}
    pairs = pair_runs(markers, sub_trials)
    parts, stim, resp, runs_qc = [], [], [], []
    for row in pairs.itertuples(index=False):
        rq = {"eeg_run": row.eeg_run, "events_run": row.events_run, "status": row.status}
        if row.events_run is None or row.status != "verified":
            rq["used"] = False
            runs_qc.append(rq)
            continue
        cr = clean_run(vhdrs[row.eeg_run], **clean_kwargs)
        ep = run_epochs(cr, sub_trials[sub_trials["run"] == row.events_run],
                        markers[row.eeg_run])
        t = ep["trials"].assign(eeg_run=row.eeg_run,
                                eeg_bad_channels=";".join(cr.bad_channels))
        parts.append(t)
        stim.append(ep["stim"])
        resp.append(ep["resp"])
        rq.update(used=True, **cr.qc, n_trials=len(t))
        runs_qc.append(rq)
        del cr
    if not parts:
        raise ValueError(f"{subject}: no verified EEG run")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = reject_epochs(pd.concat(parts, ignore_index=True))
    for rq in runs_qc:
        if rq.get("used"):
            m = table["eeg_run"] == rq["eeg_run"]
            rq.update(n_rejected=int(table.loc[m, "eeg_rejected"].sum()),
                      n_valid_kept=int((table.loc[m, "valid_trial"]
                                        & ~table.loc[m, "eeg_rejected"]).sum()))
    table.to_csv(out / f"{subject}_trials.csv", index=False)
    np.savez_compressed(out / f"{subject}_epochs.npz", stim=np.concatenate(stim),
                        resp=np.concatenate(resp), stim_times=ep["stim_times"],
                        resp_times=ep["resp_times"], ch_names=np.array(ep_channel_names(vhdrs)))
    return {"subject": subject, "n_eeg_runs": len(vhdrs),
            "n_runs_used": int(sum(r["used"] for r in runs_qc)),
            "n_epochs": len(table), "n_rejected": int(table["eeg_rejected"].sum()),
            "runs": runs_qc, "pairing": pairs.to_dict(orient="records")}


def ep_channel_names(vhdrs: dict) -> list[str]:
    """Channel names from the first header (identical across runs; checked by MNE read)."""
    from ..io.brainvision import read_raw

    return list(read_raw(next(iter(vhdrs.values())), preload=False).ch_names)


def load_epochs(epochs_dir: str | Path, subject: str) -> tuple[pd.DataFrame, dict]:
    z = np.load(Path(epochs_dir, f"{subject}_epochs.npz"))
    t = pd.read_csv(Path(epochs_dir, f"{subject}_trials.csv"))
    return t, {k: z[k] for k in z.files}


def window_data(ep: dict, window: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    lock, tmin, tmax = WINDOWS[window]
    return ((ep["stim"], ep["stim_times"]) if lock == "stimulus"
            else (ep["resp"], ep["resp_times"])) + (tmin, tmax)


def decoding_trials(t: pd.DataFrame, condition: str, window: str) -> np.ndarray:
    """Rows used for one analysis: valid, not rejected, the condition, a response marker,
    and (pre-decision) a response at least as late as the window end."""
    m = (t["valid_trial"].astype(bool) & ~t["eeg_rejected"].astype(bool)
         & (t["condition"] == condition) & t["eeg_has_response"].astype(bool))
    lock, _, tmax = WINDOWS[window]
    if lock == "stimulus":
        m &= t["eeg_response_latency_seconds"] >= tmax
    return m.to_numpy()


TARGET_COLUMNS = {"choice_side": "choice_side", "accuracy": "accuracy",
                  "confidence": "confidence_raw"}


def _target(t: pd.DataFrame, name: str) -> np.ndarray:
    col = TARGET_COLUMNS[name]
    if name == "choice_side":
        return (t[col] == "right").to_numpy(float)
    return t[col].to_numpy(float)


def _analysis_inputs(t, ep, chans, condition, window, spec, bin_s):
    from ..eeg_decoding import window_features

    rows = decoding_trials(t, condition, window)
    data, times, tmin, tmax = window_data(ep, window)
    X = window_features(data[rows][:, chans], times, tmin, tmax, bin_s)
    tt = t[rows]
    y = _target(tt, spec.get("target", spec["_name"]))
    cov = (tt[spec["covariates"]].to_numpy(float) if spec.get("covariates") else None)
    return X, y, tt["run"].to_numpy(), cov


def run_decoding(cfg: dict) -> dict:
    """Within-subject (leave-one-run-out) and cross-subject decoding per the config."""
    from joblib import Parallel, delayed

    from ..eeg_decoding import decode, holm, transfer

    loaded = {s: load_epochs(cfg["epochs_dir"], s) for s in cfg["subjects"]}
    names = [str(c) for c in next(iter(loaded.values()))[1]["ch_names"]]
    bad = set()
    for t, ep in loaded.values():
        if [str(c) for c in ep["ch_names"]] != names:
            raise ValueError("channel lists differ between subjects")
        for b in t["eeg_bad_channels"].dropna():
            bad |= set(filter(None, str(b).split(";")))
    chans = [i for i, c in enumerate(names) if c != ECG_CHANNEL and c not in bad]
    bin_s = cfg.get("bin_seconds", 0.05)
    specs = {k: {**v, "_name": k} for k, v in cfg["targets"].items()}
    jobs = [(s, k, c, w) for s in cfg["subjects"] for k in specs
            for c in cfg["conditions"] for w in cfg["windows"]]

    def one(s, k, c, w):
        t, ep = loaded[s]
        X, y, g, cov = _analysis_inputs(t, ep, chans, c, w, specs[k], bin_s)
        r = decode(X, y, g, specs[k]["task"], cov, cfg.get("n_perm", 200),
                   cfg.get("seed", 0))
        return {"subject": s, "target": k, "condition": c, "window": w, **r}

    within = Parallel(n_jobs=cfg.get("n_jobs", 1))(delayed(one)(*j) for j in jobs)
    within = pd.DataFrame(within)
    within["primary"] = within["target"].map(lambda k: bool(specs[k].get("primary")))
    within["p_holm_primary"] = np.nan
    for s in cfg["subjects"]:
        m = (within["subject"] == s) & within["primary"]
        within.loc[m, "p_holm_primary"] = holm(within.loc[m, "p_value"])

    cross = []
    subs = cfg["subjects"]
    if len(subs) >= 2:
        for a in subs:
            for b in subs:
                if a == b:
                    continue
                for k, c, w in {(j[1], j[2], j[3]) for j in jobs}:
                    Xa, ya, _, ca = _analysis_inputs(*loaded[a], chans, c, w, specs[k], bin_s)
                    Xb, yb, _, cb = _analysis_inputs(*loaded[b], chans, c, w, specs[k], bin_s)
                    r = transfer(Xa, ya, Xb, yb, specs[k]["task"], ca, cb,
                                 cfg.get("n_perm_transfer", 1000), cfg.get("seed", 0))
                    cross.append({"train_subject": a, "test_subject": b, "target": k,
                                  "condition": c, "window": w, **r})
    cross = pd.DataFrame(cross).sort_values(["target", "condition", "window",
                                             "train_subject"]) if cross else pd.DataFrame()
    return {"channels_used": [names[i] for i in chans], "excluded_channels":
            sorted(bad | {ECG_CHANNEL}), "windows": WINDOWS, "within_subject": within,
            "cross_subject": cross}
