"""Minimal synthetic/general trial-table checks; no dataset-specific event mapping."""

import pandas as pd

TRIAL_KEY = ["dataset_id", "subject", "session", "task", "run", "trial_index"]
REQUIRED = TRIAL_KEY + ["condition", "onset_seconds", "source_file", "valid_trial"]


def validate_trials(df: pd.DataFrame) -> dict:
    errors = []
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        errors.append(f"Missing required columns: {missing}")
    else:
        if df.duplicated(subset=TRIAL_KEY).any():
            errors.append("Duplicate trial identity keys")
        if df[TRIAL_KEY].isna().any().any():
            errors.append("Null trial identity fields")
        if (pd.to_numeric(df["onset_seconds"], errors="coerce") < 0).any():
            errors.append("Negative onset_seconds")
    if "reaction_time_seconds" in df.columns:
        rt = pd.to_numeric(df["reaction_time_seconds"], errors="coerce")
        if (rt.dropna() <= 0).any():
            errors.append("Reaction time must be positive when present")
    if "confidence_raw" in df.columns:
        conf = pd.to_numeric(df["confidence_raw"], errors="coerce")
        if conf.notna().sum() == 0:
            errors.append("No nonmissing confidence_raw observations")
        # Do not enforce a confidence scale without dataset-specific metadata.
    return {"valid": not errors, "errors": errors, "n_trials": len(df), "n_subjects": int(df["subject"].nunique()) if "subject" in df else None}
