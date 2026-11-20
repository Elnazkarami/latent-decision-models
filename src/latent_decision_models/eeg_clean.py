"""Artifact removal for EEG recorded inside an MRI scanner, plus causal filtering.

Generic building blocks (NumPy / SciPy; MNE only for heartbeat detection):

* :func:`template_subtract` - sliding-window average artifact subtraction (AAS; Allen et
  al. 1998, 2000). Used for MR gradient artifacts (one template per volume, locked to
  the volume trigger) and for ballistocardiogram (BCG) artifacts (locked to heartbeats).
* :func:`obs_subtract` - optimal basis set (OBS; Niazy et al. 2005): each heartbeat
  segment is fitted with the mean segment plus the first principal components of all
  segments, and the fit is subtracted. Captures beat-to-beat shape changes that a moving
  average misses.
* :func:`ica_remove_cardiac` - independent component analysis (MNE): components whose
  phase locks to the heartbeat (cross-trial phase statistics, Dammers et al. 2008) are
  removed. Can run alone or on the output of AAS, to remove what AAS left behind.
* :func:`cardiac_trial_features` - per-trial heart periods and R-peak timing around the
  response, for checking whether cardiac timing could confound a neural effect.
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


def ica_remove_cardiac(x: np.ndarray, peaks: np.ndarray, sfreq: float, median_rr_s: float,
                       n_components: int = 30, threshold: float = 0.3,
                       seed: int = 0) -> tuple[np.ndarray, dict]:
    """Remove heartbeat-locked independent components (MNE ICA, FastICA).

    ICA is fitted on a 1-40 Hz (causal) copy, every second sample; components are scored
    by cross-trial phase statistics over heartbeat epochs (-0.3 x RR to +0.7 x RR around
    each R peak, the BCG window) with :meth:`mne.preprocessing.ICA.find_bads_ecg`
    (``method="ctps"``; MNE's default threshold 0.3). The excluded components are removed from the unfiltered
    data; all other variance is kept. Uses no trial labels.
    """
    import mne

    x = np.asarray(x, np.float32)
    n_ch, n_samp = x.shape
    info = mne.create_info([f"e{i}" for i in range(n_ch)] + ["ecg"], sfreq,
                           ["eeg"] * n_ch + ["ecg"])
    # The ecg channel only satisfies MNE's channel check; components are scored on the
    # heartbeat epochs built from ``peaks``, and ICA uses the EEG channels only.
    fit = np.vstack([causal_bandpass(x, sfreq, 1.0, 40.0), np.zeros((1, n_samp))])
    raw_fit = mne.io.RawArray(fit.astype(float) * 1e-6, info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=min(n_components, n_ch - 1), method="fastica",
                                random_state=seed, max_iter="auto", verbose="ERROR")
    ica.fit(raw_fit, picks="eeg", decim=2, verbose="ERROR")  # fit on every 2nd sample
    before = round(0.3 * median_rr_s * sfreq)
    after = round(0.7 * median_rr_s * sfreq)
    p = np.asarray(peaks, int)
    p = p[(p - before >= 0) & (p + after < n_samp)]
    events = np.column_stack([p, np.zeros_like(p), np.ones_like(p)])
    ep = mne.Epochs(raw_fit, events, tmin=-before / sfreq, tmax=(after - 1) / sfreq,
                    baseline=None, preload=True, verbose="ERROR")
    bads, scores = ica.find_bads_ecg(ep, ch_name="ecg", method="ctps", threshold=threshold,
                                     verbose="ERROR")
    raw = mne.io.RawArray(np.vstack([x, np.zeros((1, n_samp))]).astype(float) * 1e-6, info,
                          verbose="ERROR")
    ica.apply(raw, exclude=list(bads), verbose="ERROR")
    out = (raw.get_data(picks="eeg") * 1e6).astype(np.float32)
    return out, {"n_components": int(ica.n_components_), "excluded": [int(b) for b in bads],
                 "n_excluded": len(bads),
                 "ctps_scores": [round(float(v), 3) for v in np.asarray(scores)]}


def cardiac_trial_features(peaks: np.ndarray, stim: np.ndarray, resp: np.ndarray,
                           sfreq: float, tolerance: float = 0.3) -> dict:
    """Per-trial heart periods and R-peak timing around stimulus and response.

    ``peaks``, ``stim`` and ``resp`` are sample indices at ``sfreq`` (``stim``/``resp`` may be
    NaN). Inter-beat interval (IBI) ``k`` is ``peaks[k]`` to ``peaks[k + 1]``. An IBI is used
    only if it lies within ``tolerance`` (30%) of the median IBI; otherwise it is treated
    as a missed or spurious detection and the feature is NaN. Returned arrays (seconds):

    * ``hb_ibi_base_s`` - the IBI before the one containing the stimulus onset;
    * ``hb_ibi_resp0_s`` / ``hb_ibi_resp1_s`` / ``hb_ibi_resp2_s`` - the IBI containing the
      response and the next two;
    * ``hb_post_response_change_s`` - mean(resp1, resp2) - base (cardiac slowing > 0);
    * ``hb_prev_rpeak_s`` / ``hb_next_rpeak_s`` - time from the last R peak to the response
      and from the response to the next R peak (where heartbeat artifacts fall in
      response-locked windows); NaN unless the containing IBI is plausible.
    """
    pk = np.sort(np.asarray(peaks, int))
    ibi = np.diff(pk) / sfreq
    med = np.median(ibi) if len(ibi) else np.nan
    ok = np.abs(ibi - med) <= tolerance * med

    def containing(t):
        t = np.asarray(t, float)
        i = np.full(len(t), -1)
        f = np.isfinite(t)
        i[f] = np.searchsorted(pk, t[f], side="right") - 1
        i[~f] = -10**6
        return i

    def ibi_at(i):
        out = np.full(len(i), np.nan)
        v = (i >= 0) & (i < len(ibi))
        out[v] = np.where(ok[i[v]], ibi[i[v]], np.nan)
        return out

    i_s, i_r = containing(stim), containing(resp)
    base = ibi_at(i_s - 1)
    r0, r1, r2 = ibi_at(i_r), ibi_at(i_r + 1), ibi_at(i_r + 2)
    resp = np.asarray(resp, float)
    prev = np.full(len(resp), np.nan)
    nxt = np.full(len(resp), np.nan)
    v = np.isfinite(r0)
    prev[v] = (resp[v] - pk[i_r[v]]) / sfreq
    nxt[v] = (pk[i_r[v] + 1] - resp[v]) / sfreq
    return {"hb_ibi_base_s": base, "hb_ibi_resp0_s": r0, "hb_ibi_resp1_s": r1,
            "hb_ibi_resp2_s": r2, "hb_post_response_change_s": (r1 + r2) / 2 - base,
            "hb_prev_rpeak_s": prev, "hb_next_rpeak_s": nxt}


def band_power(x: np.ndarray, sfreq: float, freqs) -> dict:
    """Median-over-channels Welch power (uV^2/Hz) at the given frequencies (QC)."""
    f, p = welch(x, sfreq, nperseg=int(min(x.shape[1], 4 * sfreq)))
    return {float(hz): float(np.median(p[:, np.argmin(np.abs(f - hz))])) for hz in freqs}


def planted_signal_retention(x: np.ndarray, clean, onsets: np.ndarray, waveform: np.ndarray,
                             pattern: np.ndarray, amplitudes: np.ndarray, sfreq: float,
                             band: tuple = (1.0, 10.0), settle_s: float = 10.0) -> dict:
    """How much of a planted event-locked signal survives an artifact-removal step.

    ``s = sum_i amplitudes[i] * pattern ⊗ waveform`` starting at ``onsets[i]`` is added to
    real data ``x``; ``clean`` (a function of the data array) is applied to ``x`` and to
    ``x + s``, and the difference ``d = clean(x + s) - clean(x)`` is the part of ``s`` that
    survives (exact for linear methods such as template subtraction; for data-dependent
    methods such as OBS it also includes the method's reaction to ``s``). Both ``d`` and
    ``s`` are causally band-passed to ``band``. Returns:

    * ``gain`` - regression of ``d`` on ``s`` (1 = fully kept, 0 = removed);
    * ``trial_r`` / ``trial_slope`` - single-trial amplitudes recovered from ``d`` (projection
      on the band-passed unit response) vs the planted ``amplitudes``: what a decoder of
      trial-to-trial variation could still use.
    """
    x = np.asarray(x, np.float32)
    n_samp = x.shape[1]
    onsets = np.asarray(onsets, int)
    L = len(waveform)
    keep = (onsets >= int(settle_s * sfreq)) & (onsets + 4 * L < n_samp)
    onsets, amplitudes = onsets[keep], np.asarray(amplitudes, float)[keep]
    trace = np.zeros(n_samp, np.float32)
    for o, a in zip(onsets, amplitudes, strict=True):
        trace[o:o + L] += a * waveform
    s = np.asarray(pattern, np.float32)[:, None] * trace[None]
    d = clean(x + s) - clean(x)
    sb = causal_bandpass(s, sfreq, *band)
    db = causal_bandpass(d, sfreq, *band)
    t0 = int(settle_s * sfreq)
    gain = float(np.sum(db[:, t0:] * sb[:, t0:]) / np.sum(sb[:, t0:] ** 2))
    # Band-passed response to one unit event (the filter is linear and time-invariant).
    unit = np.zeros(4 * L, np.float32)
    unit[:L] = waveform
    k = np.asarray(pattern, float)[:, None] * causal_bandpass(unit[None], sfreq, *band)[0][None]
    est = np.array([np.sum(db[:, o:o + 4 * L] * k) for o in onsets]) / np.sum(k ** 2)
    r = float(np.corrcoef(est, amplitudes)[0, 1]) if len(onsets) > 2 else float("nan")
    slope = float(np.polyfit(amplitudes, est, 1)[0]) if len(onsets) > 2 else float("nan")
    return {"gain": gain, "trial_r": r, "trial_slope": slope, "n_events": len(onsets)}
