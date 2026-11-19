"""OpenNeuro ds007474 (two-step task fMRI): behavior and event timing from the linked OSF project.

Status: experimental. Semantics below were verified on all 189 subjects (see
``docs/ds007474-audit.md``). Tags: VERIFIED-DATA (checked on every file), VERIFIED-CODE
(authors' task/analysis code), INFERRED (consistent with data, not stated by the authors).

Key facts:
* The BIDS ``*_events.tsv`` files in snapshot 1.0.5 are header-only. Behavior and timing
  exist only on OSF (node ``ctfzd``, view-only link published in the dataset README).
* Each behavior CSV holds 259 trials: 105 from a session in the structural scanner (no
  fMRI) and 154 from two 77-trial functional runs.
* Event timing (``.mat``) is stored as MATLAB ``table`` objects; reading it needs the
  optional ``mat-io`` package (``pip install 'latent-decision-models[ds007474]'``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "ds007474"
SNAPSHOT = "1.0.5"
TASK = "MBMF"
OSF_NODE = "ctfzd"
OSF_VIEW_ONLY = "022584fa16944755982f0d06a713835c"  # published in the dataset README
OSF_SUFFIXES = (".csv", ".mat")
CSV_DIR = "Control_structural_functional_behavior_csv"
MAT_DIR = "Control_functional_behavior_event_timing_mat"

TR_SECONDS = 1.12  # VERIFIED-DATA: all 378 bold.json and NIfTI headers
EXPECTED_VOLUMES = 650  # VERIFIED-DATA: 377/378 runs (sub-MBMF025002 run-1 has 381)
TRIALS_PER_RUN = 77
STRUCTURAL_TRIALS = 105
FUNCTIONAL_RUNS = ("run-1", "run-2")  # INFERRED: OSF functional run k = BIDS run-k

# VERIFIED-CODE task constants (task_MBMF_fMRI struct); semantics of high/low not interpreted.
CONTINGENCY = {1: "stimulus", 2: "state"}  # CONT_STIM = 1, CONT_STATE = 2
STATE_CODES = {1: "high", 2: "low"}  # STATE_HIGH = 1, STATE_LOW = 2
REWARD_CODES = {1: "high", 2: "low"}  # REWARD_HIGH = 1, REWARD_LOW = 2
CHOICE_SIDES = {1: "left", 2: "right"}  # VERIFIED-CODE: authors use isLeft = Resp1 == 1
P_RARE_TRANSITION = 0.3  # VERIFIED-CODE: pTransRare; observed rare rate 0.28

# From the dataset README (VERIFIED-DATA for the reward condition).
STATE_CONTINGENT_ONLY = ("MBMF021501", "MBMF021601", "MBMF021701", "MBMF021801",
                         "MBMF021901")
NO_RUN1_FIELDMAP = ("MBMF025002",)
TRUNCATED_BOLD = {("MBMF025002", "run-1"): 381}  # 30 trials fall after the scan ended

CSV_COLUMNS = (
    "runID", "sess_type", "trialID", "condContingent", "condState", "condReward",
    "isPost_ContState", "isPost_StateLow", "isPost_RewHigh", "doRareTrans", "outcomeBin",
    "outcomeMag", "rewMagnitude", "rewPWin_1", "rewPWin_2", "rewPWin_3", "rewPWin_4",
    "rewBinary_1", "rewBinary_2", "rewBinary_3", "rewBinary_4", "outcome2", "outcome1",
    "state2_1", "state2_2", "Resp1", "RT1", "isStay", "isPrevRare", "isPrevWin",
    "isPrev2Win", "isPrev3Win", "PrevRT1", "PrevRewMag",
)
_CSV_NAME = re.compile(r"sub(?P<id>\d{4})s(?P<s>\d)f(?P<f>\d)\.csv$")
_MAT_NAME = re.compile(r"subID_(?P<id>\d{4})_sesID_(?P<ses>\d)_MBMF_fMRI\.mat$")


class Ds007474SchemaError(ValueError):
    """A behavior file does not match the verified structure."""


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    source_file: str


def subject_map(osf_root: str | Path) -> dict[str, str]:
    """4-digit OSF participant ID -> BIDS label (``MBMFxxxx0k``), from timing file names.

    ``k`` is the task order within the scan session (README); the timing file's ``sesID``
    equals it for all 189 subjects (VERIFIED-DATA).
    """
    out = {}
    for p in Path(osf_root, MAT_DIR).glob("*.mat"):
        m = _MAT_NAME.search(p.name)
        if m:
            out[m["id"]] = f"MBMF{m['id']}0{m['ses']}"
    return out


def parse_behavior_csv(path: str | Path, subject: str | None = None,
                       strict: bool = True) -> tuple[pd.DataFrame, list[Issue]]:
    """Parse one OSF behavior CSV into the canonical trial table.

    ``subject`` is the BIDS label (e.g. ``MBMF000202``); if omitted, ``MBMFxxxx`` is used.
    Structural-session trials get ``run = "structural"`` and no onset. On no-response
    trials the CSV's ``outcome1`` is a placeholder (always 1), so the reached planet and
    second-stage state are set to missing.
    """
    path = Path(path)
    issues: list[Issue] = []

    def problem(level, code, msg):
        issues.append(Issue(level, code, msg, str(path)))

    name = _CSV_NAME.search(path.name)
    if not name:
        raise Ds007474SchemaError(f"unexpected file name: {path.name}")
    raw = pd.read_csv(path)
    if tuple(raw.columns) != CSV_COLUMNS:
        raise Ds007474SchemaError(f"{path.name}: columns differ from the verified header")

    if len(raw) != STRUCTURAL_TRIALS + 2 * TRIALS_PER_RUN:
        problem("error", "row_count", f"{len(raw)} rows (expected 259)")
    sess = raw["sess_type"]
    if set(sess) - {"structural", "functional"}:
        problem("error", "unknown_session", f"sess_type values {sorted(set(sess))}")
    order = raw.groupby("runID", sort=True)["sess_type"].agg(lambda s: set(s))
    if any(len(v) != 1 for v in order):
        problem("error", "mixed_run", "a runID mixes structural and functional trials")
    run_types = [next(iter(v)) for v in order]
    expected = {("2", "3"): ["structural", "functional", "functional"],
                ("3", "2"): ["functional", "functional", "structural"]}.get((name["s"], name["f"]))
    if expected is not None and run_types != expected:
        problem("error", "session_order", f"run order {run_types} contradicts {path.name}")

    func_runs = [r for r, t in zip(order.index, run_types, strict=True) if t == "functional"]
    run_label = {r: FUNCTIONAL_RUNS[i] for i, r in enumerate(func_runs[:2])}
    for r in func_runs:
        if (raw["runID"] == r).sum() != TRIALS_PER_RUN:
            problem("error", "run_length", f"functional runID {r} has "
                    f"{(raw['runID'] == r).sum()} trials (expected {TRIALS_PER_RUN})")

    responded = raw["Resp1"].notna()
    bad_codes = ~raw.loc[responded, "Resp1"].isin(CHOICE_SIDES)
    if bad_codes.any():
        problem("error", "choice_code", f"Resp1 codes {sorted(raw.loc[responded, 'Resp1'][bad_codes].unique())}")
    if not raw["condContingent"].isin(CONTINGENCY).all():
        problem("error", "contingency_code", "condContingent outside {1, 2}")
    if not raw["doRareTrans"].isin([0, 1]).all():
        problem("error", "transition_code", "doRareTrans outside {0, 1}")
    planet_ok = (np.ceil(raw.loc[responded, "outcome2"] / 2) == raw.loc[responded, "outcome1"])
    if not planet_ok.all():
        problem("error", "state_coding", "outcome1 != ceil(outcome2 / 2) on responded trials")
    if raw.loc[~responded, ["outcome2", "outcomeBin"]].notna().any().any():
        problem("warning", "outcome_without_response", "outcome recorded on no-response trial")

    subj = subject or f"MBMF{name['id']}"
    func = sess.eq("functional")
    trials = pd.DataFrame({
        "dataset_id": DATASET_ID,
        "subject": subj,
        "session": "ses-01",
        "task": TASK,
        "run": [run_label.get(r, "structural") if f else "structural"
                for r, f in zip(raw["runID"], func, strict=True)],
        "osf_run_id": raw["runID"].astype(int),
        "session_type": sess,
        "osf_trial_id": raw["trialID"].astype(int),
        "condition": raw["condContingent"].map(CONTINGENCY) + "_contingent",
        "reward_contingency": raw["condContingent"].map(CONTINGENCY),
        "state_condition": raw["condState"].map(STATE_CODES),
        "reward_condition": raw["condReward"].map(REWARD_CODES),
        "choice": raw["Resp1"].where(responded),
        "choice_side": raw["Resp1"].map(CHOICE_SIDES),
        "response_time_seconds": raw["RT1"].where(responded),
        "rare_transition": raw["doRareTrans"].astype(int),
        "planet": raw["outcome1"].where(responded),
        "second_state": raw["outcome2"].where(responded),
        "reward_binary": raw["outcomeBin"].where(responded),
        "reward_magnitude": raw["outcomeMag"].where(responded),
        "onset_seconds": np.nan,
        "onset_reference": None,
        "valid_trial": responded,
        "source_file": str(path),
    })
    trials.insert(5, "trial_index", trials.groupby("run").cumcount())
    if strict and any(i.level == "error" for i in issues):
        raise Ds007474SchemaError("; ".join(i.message for i in issues if i.level == "error"))
    return trials, issues


def _timing_frame(table: pd.DataFrame, run_start: np.ndarray) -> pd.DataFrame:
    """Event times relative to each run's first scanner pulse (``tRunStart``)."""
    runs = sorted(table["runID"].unique())
    if len(runs) != len(run_start):
        raise Ds007474SchemaError(f"{len(runs)} runs but {len(run_start)} run starts")
    t0 = table["runID"].map(dict(zip(runs, run_start, strict=True))).to_numpy(float)
    cols = {"onset_seconds": "tStim1On", "response_onset_seconds": "tResp1",
            "second_state_onset_seconds": "tState2On",
            "terminal_state_onset_seconds": "tTerminalState2On",
            "feedback_binary_onset_seconds": "tFBOn_Bin",
            "feedback_magnitude_onset_seconds": "tFBOn_Mag"}
    out = pd.DataFrame({k: table[v].to_numpy(float) - t0 for k, v in cols.items()})
    out.insert(0, "run", table["runID"].map(dict(zip(runs, FUNCTIONAL_RUNS, strict=False))))
    out.insert(1, "trial_index", out.groupby("run").cumcount())
    for c in ("resp1", "RT1", "condState", "doRareTrans"):
        out[f"_check_{c}"] = table[c].to_numpy(float)
    rt_err = np.nanmax(np.abs((table["tResp1"] - table["tStim1On"]) - table["RT1"]))
    if rt_err > 1e-6:
        raise Ds007474SchemaError(f"tResp1 - tStim1On differs from RT1 by {rt_err:.3g} s")
    return out


def read_event_timing(path: str | Path) -> pd.DataFrame:
    """Read one OSF timing ``.mat`` (needs the optional ``mat-io`` package)."""
    try:
        import matio
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("Reading ds007474 timing files needs `pip install mat-io` "
                          "(extra: latent-decision-models[ds007474])") from e
    m = matio.load_from_mat(str(path))
    task = m["task_MBMF_fMRI"]
    table = task["trials"].ravel()[0]  # the top-level `trialEvents` is an empty placeholder
    run_start = np.asarray(task["tRunStart"].ravel()[0], float).ravel()
    return _timing_frame(table, run_start)


def merge_timing(trials: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    """Attach timing to functional trials after checking choices, RTs and design agree."""
    func = trials[trials["session_type"] == "functional"]
    merged = func.merge(timing, on=["run", "trial_index"], how="left", validate="one_to_one",
                        suffixes=("_drop", ""))
    if merged["onset_seconds"].isna().any():
        raise Ds007474SchemaError("functional trial without timing")
    checks = {"choice": "_check_resp1", "response_time_seconds": "_check_RT1",
              "rare_transition": "_check_doRareTrans"}
    responded = merged["choice"].notna().to_numpy()
    for a, b in checks.items():
        # On no-response trials the timing file stores the 2 s timeout as RT1.
        keep = responded if a == "response_time_seconds" else np.ones(len(merged), bool)
        x, y = merged[a].to_numpy(float)[keep], merged[b].to_numpy(float)[keep]
        if not np.allclose(x, y, equal_nan=True, atol=1e-6):
            raise Ds007474SchemaError(f"timing file disagrees with behavior CSV on {a}")
    merged["onset_reference"] = "first_scanner_pulse_of_run"
    merged = merged.drop(columns=[c for c in merged if c.endswith("_drop") or c.startswith("_check_")])
    rest = trials[trials["session_type"] != "functional"]
    return pd.concat([merged, rest], ignore_index=True)[list(trials.columns) + [
        c for c in merged.columns if c not in trials.columns]]


def load_trials(osf_root: str | Path, timing: bool = False,
                strict: bool = True) -> tuple[pd.DataFrame, list[Issue]]:
    """Parse every behavior CSV under ``osf_root`` (optionally merging event timing)."""
    osf_root = Path(osf_root)
    ids = subject_map(osf_root)
    frames, issues = [], []
    for p in sorted(Path(osf_root, CSV_DIR).glob("*.csv")):
        pid = _CSV_NAME.search(p.name)["id"]
        t, iss = parse_behavior_csv(p, ids.get(pid), strict=strict)
        if timing:
            mats = list(Path(osf_root, MAT_DIR).glob(f"subID_{pid}_sesID_*_MBMF_fMRI.mat"))
            if len(mats) != 1:
                raise Ds007474SchemaError(f"expected one timing file for {pid}, got {len(mats)}")
            t = merge_timing(t, read_event_timing(mats[0]))
        frames.append(t)
        issues += iss
    trials = pd.concat(frames, ignore_index=True)
    key = ["subject", "run", "trial_index"]
    if trials.duplicated(key).any():
        raise Ds007474SchemaError("duplicate trial identities across files")
    return trials, issues


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([asdict(i) for i in issues],
                        columns=["level", "code", "message", "source_file"])
