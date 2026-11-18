"""Task adapter for OpenNeuro ds002158 (Pereira et al., 2020, PNAS) — status: experimental.

Every mapping below was verified against snapshot 1.0.2 on 2026-11-18; see
``docs/ds002158-audit.md`` for the evidence. Facts are labelled:

* VERIFIED-SIDECAR   – stated in ``task-main_events.json``
* VERIFIED-DATA      – checked exhaustively over all 118 BIDS-valid events files
* VERIFIED-MARKERS   – checked against BrainVision ``.vmrk`` files (sample, see audit)
* PAPER              – stated in the bioRxiv v1 preprint Methods (10.1101/496877)
* INFERRED           – derived deterministically from verified fields

Nothing here reads EEG/fMRI signals.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "ds002158"
SNAPSHOT = "1.0.2"
TASK = "main"

# VERIFIED-DATA: identical header in all 118 events files (column order included).
EXPECTED_COLUMNS = (
    "onset",
    "duration",
    "trial_type",
    "response_time",
    "dots_difference",
    "stim_side",
    "choice_accuracy",
    "confidence",
    "response_time2",
)
# VERIFIED-DATA: only these four codes occur. "-bad" = excluded by the authors.
TRIAL_TYPES = {
    "act": ("active", False),
    "act-bad": ("active", True),
    "obs": ("observed", False),
    "obs-bad": ("observed", True),
}
STIM_SIDES = ("left", "right")  # VERIFIED-SIDECAR: side with more dots
ACCURACY_CODES = {"cor": 1, "err": 0}  # VERIFIED-SIDECAR
CONFIDENCE_MISSING = -1.0  # VERIFIED-DATA: 45 rows, all flagged "-bad"
# VERIFIED-DATA: response_time is on a 20 ms grid; -0.02 (34 rows) and 0.0 (1 row)
# occur only in "-bad" rows and are treated as "no first-order response".
RT_NO_RESPONSE_MAX = 0.0
# PAPER: "Trials with RT smaller than 200 ms or higher than 500 ms ... were excluded".
RT_MIN_S, RT_MAX_S = 0.2, 0.5
# PAPER + VERIFIED-DATA: 4 blocks of 12 trials per run, active block first, alternating.
BLOCK_SIZE = 12
TRIALS_PER_RUN = 48
BLOCK_ORDER = ("active", "observed", "active", "observed")

_FNAME = re.compile(
    r"^(?P<subject>sub-[A-Za-z0-9]+)_(?P<session>ses-[A-Za-z0-9]+)_task-(?P<task>[A-Za-z0-9]+)"
    r"_run-(?P<run>[0-9]+)_events\.tsv$"
)


class Ds002158SchemaError(ValueError):
    """Raised in strict mode when an events file violates the verified contract."""


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    source_file: str
    row: int | None = None


def _entities(path: Path) -> dict:
    m = _FNAME.match(path.name)
    if not m:
        raise Ds002158SchemaError(f"Not a BIDS task events filename: {path.name}")
    ent = m.groupdict()
    # Cross-check against parent directories when present (sub-XX/ses-YYY/func/...).
    parts = path.parts
    for key in ("subject", "session"):
        prefix = "sub-" if key == "subject" else "ses-"
        dirs = [p for p in parts[:-1] if p.startswith(prefix)]
        if dirs and dirs[-1] != ent[key]:
            raise Ds002158SchemaError(
                f"{key} mismatch: directory {dirs[-1]!r} vs filename {ent[key]!r} in {path}"
            )
    return ent


def _exclusion_reasons(rt: float, conf_missing: bool, author_flag: bool) -> list[str]:
    reasons = []
    if np.isnan(rt) or rt <= RT_NO_RESPONSE_MAX:
        reasons.append("no_first_order_response")
    elif rt < RT_MIN_S:
        reasons.append("rt_below_200ms")
    elif rt > RT_MAX_S:
        reasons.append("rt_above_500ms")
    if conf_missing:
        reasons.append("no_confidence_response")
    if author_flag and not reasons:
        # e.g. button press during observation (PAPER) — not recorded in events.tsv.
        reasons.append("author_flag_reason_not_in_events")
    return reasons


def parse_events(path: str | Path, strict: bool = True) -> tuple[pd.DataFrame, list[Issue]]:
    """Parse one ds002158 ``task-main`` events file into canonical trial rows.

    No row is dropped. Author-excluded trials keep ``valid_trial=False`` and an
    ``exclusion_reasons`` string. In strict mode any error-level issue raises
    :class:`Ds002158SchemaError`; otherwise issues are returned.

    Output columns (beyond the generic schema):

    * ``condition`` – ``active`` (own/committed) or ``observed``.
    * ``response_latency_seconds`` – stimulus→button press (active) or stimulus→virtual-hand
      onset (observed; a *replay* of the participant's own earlier active trial, PAPER).
    * ``response_time_seconds`` – participant RT; **active only** (NaN for observed).
    * ``choice_side`` – INFERRED: ``stim_side`` if correct else the opposite side. For observed
      trials it is the side the virtual hand chose, not a participant choice.
    * ``confidence_raw`` – continuous 0 (sure error) … 1 (sure correct); NaN where missing.
    * ``confidence_report_time_seconds`` – ``response_time2``; the sidecar warns it has
      "little utility" because the scale cursor moved automatically.
    """
    path = Path(path)
    src = str(path)
    issues: list[Issue] = []

    def err(code, msg, row=None):
        issues.append(Issue("error", code, msg, src, row))

    def warn(code, msg, row=None):
        issues.append(Issue("warning", code, msg, src, row))

    ent = _entities(path)
    if ent["task"] != TASK:
        err("wrong_task", f"Expected task-{TASK}, got task-{ent['task']}")

    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if tuple(raw.columns) != EXPECTED_COLUMNS:
        err("columns", f"Columns {list(raw.columns)} != verified {list(EXPECTED_COLUMNS)}")
        if strict:
            raise Ds002158SchemaError(issues[-1].message)
        return pd.DataFrame(), issues
    if raw.empty:
        err("empty", "Events file has no rows")

    numeric = {}
    for col in ("onset", "duration", "response_time", "dots_difference", "confidence",
                "response_time2"):
        vals = pd.to_numeric(raw[col].replace({"n/a": np.nan}), errors="coerce")
        bad = vals.isna() & (raw[col] != "n/a")
        for i in np.flatnonzero(bad.to_numpy()):
            err("non_numeric", f"{col}={raw[col].iloc[i]!r} is not numeric", int(i))
        numeric[col] = vals.astype(float)

    for col, allowed in (
        ("trial_type", set(TRIAL_TYPES)),
        ("stim_side", set(STIM_SIDES)),
        ("choice_accuracy", set(ACCURACY_CODES)),
    ):
        for i in np.flatnonzero(~raw[col].isin(allowed).to_numpy()):
            err("unknown_code", f"{col}={raw[col].iloc[i]!r} not in {sorted(allowed)}", int(i))

    onset = numeric["onset"]
    if (onset < 0).any():
        err("negative_onset", "Negative onset values")
    if not onset.dropna().is_monotonic_increasing or onset.duplicated().any():
        err("onset_order", "Onsets are not strictly increasing (possible duplicate rows)")
    if (numeric["duration"].fillna(0) != 0).any():
        warn("nonzero_duration", "duration != 0; verified files use 0 (stick events)")

    conf = numeric["confidence"]
    out_of_range = conf.notna() & (conf != CONFIDENCE_MISSING) & ((conf < 0) | (conf > 1))
    for i in np.flatnonzero(out_of_range.to_numpy()):
        err("confidence_range", f"confidence={conf.iloc[i]} outside [0,1] and not -1", int(i))

    if len(raw) > TRIALS_PER_RUN:
        err("too_many_trials", f"{len(raw)} rows > {TRIALS_PER_RUN} trials per run")
    elif len(raw) < TRIALS_PER_RUN:
        warn("short_run", f"{len(raw)} rows < {TRIALS_PER_RUN} (truncated run?)")

    if strict and any(i.level == "error" for i in issues):
        raise Ds002158SchemaError("; ".join(f"[{i.code}] {i.message}" for i in issues
                                            if i.level == "error"))
    if any(i.level == "error" for i in issues):
        return pd.DataFrame(), issues

    cond = raw["trial_type"].map(lambda t: TRIAL_TYPES[t][0])
    author_flag = raw["trial_type"].map(lambda t: TRIAL_TYPES[t][1]).astype(bool)
    trial_index = np.arange(len(raw))
    block = trial_index // BLOCK_SIZE
    expected = [BLOCK_ORDER[b % len(BLOCK_ORDER)] for b in block]
    for i in np.flatnonzero(cond.to_numpy() != np.asarray(expected)):
        warn("block_structure", f"condition {cond.iloc[i]} where design expects {expected[i]}",
             int(i))

    latency = numeric["response_time"].where(numeric["response_time"] > RT_NO_RESPONSE_MAX)
    conf_missing = (conf == CONFIDENCE_MISSING) | conf.isna()
    accuracy = raw["choice_accuracy"].map(ACCURACY_CODES)
    other = {"left": "right", "right": "left"}
    choice = np.where(accuracy == 1, raw["stim_side"], raw["stim_side"].map(other))
    reasons = [
        _exclusion_reasons(float(numeric["response_time"].iloc[i]), bool(conf_missing.iloc[i]),
                           bool(author_flag.iloc[i]))
        for i in range(len(raw))
    ]
    for i, r in enumerate(reasons):
        if r and not author_flag.iloc[i]:
            warn("unflagged_exclusion", f"meets exclusion rule {r} but not flagged -bad", i)

    rt2 = numeric["response_time2"]
    trials = pd.DataFrame(
        {
            "dataset_id": DATASET_ID,
            "subject": ent["subject"],
            "session": ent["session"],
            "task": ent["task"],
            "run": f"run-{ent['run']}",
            "trial_index": trial_index,
            "block_index": block,
            "condition": cond.to_numpy(),
            "onset_seconds": onset.to_numpy(),
            "onset_reference": "first_fmri_volume_trigger",
            "evidence_dots_difference": numeric["dots_difference"].to_numpy(),
            "stim_side": raw["stim_side"].to_numpy(),
            "accuracy": accuracy.to_numpy(),
            "choice_side": choice,
            "response_latency_seconds": latency.to_numpy(),
            "response_time_seconds": latency.where(cond == "active").to_numpy(),
            "confidence_raw": conf.mask(conf_missing).to_numpy(),
            "confidence_missing": conf_missing.to_numpy(),
            "confidence_report_time_seconds": rt2.where(rt2 > 0).to_numpy(),
            "author_excluded": author_flag.to_numpy(),
            "exclusion_reasons": [";".join(r) for r in reasons],
            "valid_trial": (~author_flag & ~conf_missing).to_numpy(),
            "source_file": src,
        }
    )
    return trials, issues


def discover_events(bids_root: str | Path) -> list[Path]:
    """List BIDS-valid ``task-main`` events files under ``bids_root``.

    Files inside ``rejected/`` directories are excluded, mirroring the dataset's
    ``.bidsignore``. Derivative copies are ignored (verified byte-identical to raw).
    """
    root = Path(bids_root)
    files = sorted(root.glob(f"sub-*/ses-*/func/sub-*_task-{TASK}_run-*_events.tsv"))
    return [f for f in files if "rejected" not in f.parts]


def load_trials(bids_root: str | Path, strict: bool = True) -> tuple[pd.DataFrame, list[Issue]]:
    """Parse every BIDS-valid events file and check trial identity uniqueness."""
    files = discover_events(bids_root)
    if not files:
        raise FileNotFoundError(f"No task-{TASK} events files under {bids_root}")
    frames, issues = [], []
    for f in files:
        t, iss = parse_events(f, strict=strict)
        frames.append(t)
        issues.extend(iss)
    trials = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    key = ["dataset_id", "subject", "session", "task", "run", "trial_index"]
    if trials.duplicated(subset=key).any():
        msg = "Duplicate trial identity keys across files"
        if strict:
            raise Ds002158SchemaError(msg)
        issues.append(Issue("error", "duplicate_trial_id", msg, str(bids_root)))
    return trials, issues


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([asdict(i) for i in issues],
                        columns=["level", "code", "message", "source_file", "row"])


# --------------------------------------------------------------------------------------
# EEG marker alignment (VERIFIED-MARKERS; see audit for which runs were checked)
# --------------------------------------------------------------------------------------
VOLUME_MARKER = "R128"  # one per fMRI volume; count equals BOLD volumes in checked runs
STIMULUS_MARKERS = {"left": "S11", "right": "S12"}  # by stim_side, both conditions
RESPONSE_MARKERS = {
    ("active", "left"): "S7",  # participant button press, chosen side
    ("active", "right"): "S8",
    ("observed", "left"): "S17",  # virtual-hand onset, chosen side
    ("observed", "right"): "S18",
}


def align_eeg_markers(
    trials: pd.DataFrame,
    markers: pd.DataFrame,
    tolerance_seconds: float = 0.002,
    response_window_seconds: float = 1.2,
) -> tuple[pd.DataFrame, dict]:
    """Match one run's trials to EEG markers using *timing*, never row order.

    ``events.onset`` is defined relative to the first ``R128`` volume trigger. For each
    trial, the stimulus marker (``S11`` or ``S12``) nearest to ``first_R128 + onset`` must
    lie within ``tolerance_seconds`` (timing match). Labels are then checked separately:
    the marker's side vs ``stim_side``, and the first condition-specific response marker
    within ``response_window_seconds`` vs the inferred ``choice_side``.

    Summary ``status``:
      * ``verified``    – every trial timing-matched and no label disagreements
      * ``timing_only`` – every trial timing-matched but some labels disagree
      * ``partial``     – some trials timing-matched
      * ``failed``      – none matched
    """
    if trials["run"].nunique() != 1 or trials["subject"].nunique() != 1:
        raise ValueError("align_eeg_markers expects trials from exactly one subject/run")
    vol = markers[markers["description"] == VOLUME_MARKER]
    if vol.empty:
        raise ValueError(f"No {VOLUME_MARKER} volume markers found")
    t0 = float(vol["time_seconds"].iloc[0])
    rel = markers.assign(rel=markers["time_seconds"] - t0)
    side_of_stim = {code: side for side, code in STIMULUS_MARKERS.items()}
    stim = rel[rel["description"].isin(list(side_of_stim))]
    rows = []
    used: set[int] = set()
    for tr in trials.itertuples(index=False):
        rec = {"trial_index": tr.trial_index, "stimulus_marker": None,
               "eeg_stimulus_sample": None, "onset_residual_ms": None,
               "stimulus_side_matches": None, "response_marker": None,
               "eeg_response_latency_seconds": None, "response_side_matches_choice": None,
               "timing_matched": False}
        cand = stim[~stim.index.isin(used)]
        if not cand.empty:
            j = (cand["rel"] - tr.onset_seconds).abs().idxmin()
            resid = float(cand.at[j, "rel"] - tr.onset_seconds)
            if abs(resid) <= tolerance_seconds:
                used.add(j)
                code = cand.at[j, "description"]
                rec.update(stimulus_marker=code, eeg_stimulus_sample=int(cand.at[j, "sample"]),
                           onset_residual_ms=round(resid * 1000, 3), timing_matched=True,
                           stimulus_side_matches=bool(side_of_stim[code] == tr.stim_side))
                codes = {RESPONSE_MARKERS[(tr.condition, s)]: s for s in STIM_SIDES}
                t_stim = cand.at[j, "rel"]
                after = rel[(rel["description"].isin(list(codes))) & (rel["rel"] > t_stim)
                            & (rel["rel"] <= t_stim + response_window_seconds)]
                if not after.empty:
                    first = after.iloc[0]
                    rec.update(response_marker=first["description"],
                               eeg_response_latency_seconds=round(float(first["rel"] - t_stim), 4),
                               response_side_matches_choice=bool(
                                   codes[first["description"]] == tr.choice_side))
        rows.append(rec)
    table = pd.DataFrame(rows)
    n = len(table)
    n_matched = int(table["timing_matched"].sum())
    stim_mis = int((table["stimulus_side_matches"] == False).sum())
    resp_mis = int((table["response_side_matches_choice"] == False).sum())
    if n_matched == 0:
        status = "failed"
    elif n_matched < n:
        status = "partial"
    elif stim_mis or resp_mis:
        status = "timing_only"
    else:
        status = "verified"
    summary = {
        "subject": trials["subject"].iloc[0],
        "run": trials["run"].iloc[0],
        "n_trials": n,
        "n_timing_matched": n_matched,
        "n_volume_markers": len(vol),
        "max_abs_onset_residual_ms": (float(table["onset_residual_ms"].abs().max())
                                      if n_matched else None),
        "n_stimulus_side_mismatch": stim_mis,
        "n_response_marker_found": int(table["response_marker"].notna().sum()),
        "n_response_side_mismatch": resp_mis,
        "status": status,
    }
    return table, summary
