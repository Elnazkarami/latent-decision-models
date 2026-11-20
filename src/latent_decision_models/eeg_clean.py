"""Artifact removal for EEG recorded inside an MRI scanner, plus causal filtering.

Generic building blocks (NumPy / SciPy; MNE only for heartbeat detection):

* :func:`template_subtract` - sliding-window average artifact subtraction (AAS; Allen et
  al. 1998, 2000). Used for MR gradient artifacts (one template per volume, locked to
  the volume trigger) and for ballistocardiogram (BCG) artifacts (locked to heartbeats).
* :func:`obs_subtract` - optimal basis set (OBS; Niazy et al. 2005): each heartbeat
  segment is fitted with the mean segment plus the first principal components of all
  segments, and the fit is subtracted. Captures beat-to-beat shape changes that a moving
  average misses.
* :func:`causal_bandpass` / :func:`causal_decimate` - one-pass (causal) IIR filtering.
  A zero-phase (two-pass) filter spreads later activity backwards in time, which would
  let post-response signal leak into a "pre-decision" window. With a causal filter,
  every filtered sample depends only on the present and the past (at the cost of a
  frequency-dependent delay, which only makes window labels conservative).
* :func:`detect_heartbeats` - R-peak detection with MNE, plus plausibility metrics.

All functions take arrays shaped ``(n_channels, n_samples)`` in microvolts.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, welch


def template_subtract(x: np.ndarray, onsets: np.ndarray, before: int, after: int,
                      n_template: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a sliding-window average template from each event-locked segment.

    Segment ``i`` covers samples ``[onsets[i] - before, onsets[i] + after)``. Its template
    is the mean of the ``n_template`` segments centred on it (truncated at the edges).
    If segments overlap (e.g. a short heartbeat interval), each sample is corrected once,
    by the segment whose event is nearest. Segments that run past the data are skipped.

    Returns the corrected copy and a boolean mask of the events that were corrected.
    """
    if n_template < 1:
        raise ValueError("n_template must be >= 1")
    x = np.asarray(x)
    n_samp = x.shape[1]
    onsets = np.asarray(onsets, int)
    ok = (onsets - before >= 0) & (onsets + after <= n_samp)
    ev = onsets[ok]
    out = x.astype(np.float32, copy=True)
    if len(ev) == 0:
        return out, ok
    length = before + after
    idx = ev[:, None] - before + np.arange(length)[None, :]
    seg = x[:, idx].astype(np.float32)  # (ch, n_ev, length)
    cs = np.cumsum(np.concatenate([np.zeros_like(seg[:, :1]), seg], axis=1), axis=1,
                   dtype=np.float64)
    half = n_template // 2
    k = np.arange(len(ev))
    lo = np.clip(k - half, 0, None)
    hi = np.clip(k - half + n_template, None, len(ev))
    tmpl = ((cs[:, hi] - cs[:, lo]) / (hi - lo)[None, :, None]).astype(np.float32)
    # Ownership: each sample is corrected by the nearest event's template.
    owner = np.full(n_samp, -1)
    dist = np.full(n_samp, np.inf)
    for i, row in enumerate(idx):
        d = np.abs(row - ev[i])
        better = d < dist[row]
        owner[row[better]] = i
        dist[row[better]] = d[better]
    for i, row in enumerate(idx):
        mine = owner[row] == i
        out[:, row[mine]] -= tmpl[:, i, mine]
    return out, ok


def volume_grid(volume_onsets: np.ndarray, tr_samples: int,
                max_jitter: int = 10) -> tuple[np.ndarray, int]:
    """Regular volume grid ``phase + k * tr_samples`` and the number of triggers off it.

    With EEG and scanner clocks synchronised, the gradient artifact sits on this grid;
    triggers logged a few samples late must not move their template. In ds002158 isolated
    triggers are logged 1-6 samples late and the next one is back on the grid; for
    deviations of 2-6 samples the artifact was checked directly and sits on the grid
    (best-fitting shift 0; see docs/ds002158-eeg.md). The grid phase is the most common
    trigger phase, so one late *first* trigger does not shift the whole grid.
    Deviations above ``max_jitter`` samples mean the clocks are not synchronised.
    """
    on = np.asarray(volume_onsets, int)
    k = np.arange(len(on))
    phases, counts = np.unique(on - tr_samples * k, return_counts=True)
    grid = phases[np.argmax(counts)] + tr_samples * k
    dev = on - grid
    if np.abs(dev).max(initial=0) > max_jitter:
        raise ValueError(f"volume triggers deviate up to {np.abs(dev).max()} samples from a "
                         f"{tr_samples}-sample grid; clocks not synchronised")
    return grid, int(np.count_nonzero(dev))


def gradient_correct(x: np.ndarray, volume_onsets: np.ndarray, tr_samples: int,
                     n_template: int = 21, max_jitter: int = 10) -> np.ndarray:
    """MR gradient AAS: one template per volume, on the regular volume grid.

    Requires synchronised clocks (see :func:`volume_grid`); otherwise templates would be
    misaligned and the residual large.
    """
    grid, _ = volume_grid(volume_onsets, tr_samples, max_jitter)
    corrected, ok = template_subtract(x, grid, 0, tr_samples, n_template)
    if not ok.all():
        raise ValueError(f"{(~ok).sum()} volumes extend past the recording")
    return corrected


def causal_bandpass(x: np.ndarray, sfreq: float, l_freq: float | None,
                    h_freq: float | None, order: int = 4) -> np.ndarray:
    """One-pass Butterworth band/high/low-pass (causal)."""
    if l_freq and h_freq:
        sos = butter(order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
    elif l_freq:
        sos = butter(order, l_freq, btype="highpass", fs=sfreq, output="sos")
    elif h_freq:
        sos = butter(order, h_freq, btype="lowpass", fs=sfreq, output="sos")
    else:
        return np.asarray(x, np.float32).copy()
    return sosfilt(sos, x, axis=-1).astype(np.float32)


def causal_decimate(x: np.ndarray, sfreq: float, factor: int,
                    order: int = 8) -> tuple[np.ndarray, float]:
    """Causal anti-alias low-pass at 0.8 x the new Nyquist, then keep every ``factor``-th sample."""
    new = sfreq / factor
    y = causal_bandpass(x, sfreq, None, 0.8 * new / 2, order=order)
    return y[:, ::factor], new


def detect_heartbeats(x_ch: np.ndarray, sfreq: float) -> dict:
    """R peaks on one channel (ECG, or a mastoid electrode) via MNE.

    Returns sample indices and plausibility metrics: heart rate, and the fraction of
    beat-to-beat intervals outside 0.4-1.5 s (missed or spurious detections).
    """
    import mne

    info = mne.create_info(["ecg"], sfreq, ["ecg"])
    raw = mne.io.RawArray(np.asarray(x_ch, float)[None] * 1e-6, info, verbose="ERROR")
    events, _, hr = mne.preprocessing.find_ecg_events(raw, ch_name="ecg", verbose="ERROR")
    peaks = np.asarray(events[:, 0] - raw.first_samp, int)
    rr = np.diff(peaks) / sfreq
    return {"peaks": peaks, "n_beats": len(peaks), "heart_rate_bpm": float(hr),
            "median_rr_s": float(np.median(rr)) if len(rr) else float("nan"),
            "fraction_rr_implausible": float(np.mean((rr < 0.4) | (rr > 1.5)))
            if len(rr) else 1.0}


def bcg_correct(x: np.ndarray, peaks: np.ndarray, sfreq: float, median_rr_s: float,
                n_template: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """BCG AAS: template from 0.3 x RR before to 0.7 x RR after each R peak (21 beats)."""
    before = round(0.3 * median_rr_s * sfreq)
    after = round(0.7 * median_rr_s * sfreq)
    return template_subtract(x, peaks, before, after, n_template)


def _segment_owner(idx, ev, n_samp):
    owner = np.full(n_samp, -1)
    dist = np.full(n_samp, np.inf)
    for i, row in enumerate(idx):
        d = np.abs(row - ev[i])
        better = d < dist[row]
        owner[row[better]] = i
        dist[row[better]] = d[better]
    return owner


def obs_subtract(x: np.ndarray, onsets: np.ndarray, before: int, after: int,
                 n_pc: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Optimal-basis-set artifact removal (Niazy et al. 2005), per channel.

    Segments ``[onset - before, onset + after)`` are stacked; the basis is the mean segment
    plus the first ``n_pc`` principal components of the mean-removed segments. Each
    segment's least-squares fit on that basis is subtracted. Overlapping samples are
    corrected once, by the nearest event (as in :func:`template_subtract`).
    """
    x = np.asarray(x)
    n_samp = x.shape[1]
    onsets = np.asarray(onsets, int)
    ok = (onsets - before >= 0) & (onsets + after <= n_samp)
    ev = onsets[ok]
    out = x.astype(np.float32, copy=True)
    if len(ev) <= n_pc + 1:
        return out, ok & False
    idx = ev[:, None] - before + np.arange(before + after)[None, :]
    owner = _segment_owner(idx, ev, n_samp)
    mine = owner[idx] == np.arange(len(ev))[:, None]  # (n_ev, length)
    for ch in range(x.shape[0]):
        seg = x[ch, idx].astype(np.float64)  # (n_ev, length)
        mean = seg.mean(axis=0)
        _, _, vt = np.linalg.svd(seg - mean, full_matrices=False)
        basis = np.vstack([mean, vt[:n_pc]])  # (n_pc + 1, length)
        coef, *_ = np.linalg.lstsq(basis.T, seg.T, rcond=None)
        fit = (basis.T @ coef).T
        out[ch, idx[mine]] -= fit[mine].astype(np.float32)
    return out, ok


def bcg_correct_obs(x: np.ndarray, peaks: np.ndarray, sfreq: float, median_rr_s: float,
                    n_pc: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """BCG removal by OBS, same window as :func:`bcg_correct`."""
    before = round(0.3 * median_rr_s * sfreq)
    after = round(0.7 * median_rr_s * sfreq)
    return obs_subtract(x, peaks, before, after, n_pc)


def band_power(x: np.ndarray, sfreq: float, freqs) -> dict:
    """Median-over-channels Welch power (uV^2/Hz) at the given frequencies (QC)."""
    f, p = welch(x, sfreq, nperseg=int(min(x.shape[1], 4 * sfreq)))
    return {float(hz): float(np.median(p[:, np.argmin(np.abs(f - hz))])) for hz in freqs}
