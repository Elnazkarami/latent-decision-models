"""Deterministic synthetic data for testing only; not empirical neuroscience evidence."""

import numpy as np
import pandas as pd


def generate_trials(subjects: int = 12, trials_per_subject: int = 35, seed: int = 23):
    if subjects < 4 or trials_per_subject < 2:
        raise ValueError("Require at least 4 subjects and 2 trials per subject")
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(subjects):
        subj = f"sub-{s+1:02d}"
        offset = float(rng.normal(0, .08))
        for t in range(trials_per_subject):
            evidence = float(rng.uniform(.05, 1.0))
            accuracy = int(rng.random() < (0.52 + .40 * evidence))
            rt = float(np.clip(rng.normal(1.3 - .5 * evidence, .16), .2, 3.0))
            conf = float(np.clip(.22 + .48 * evidence + .17 * accuracy - .08 * rt + offset + rng.normal(0,.10),0,1))
            rows.append({"dataset_id":"synthetic", "subject":subj, "session":"ses-01", "task":"confidence", "run":"run-01", "trial_index":t, "condition":"active" if t % 2 == 0 else "observed", "onset_seconds":t * 3.5, "evidence":evidence, "accuracy":accuracy, "reaction_time_seconds":rt if t % 2 == 0 else np.nan, "confidence_raw":conf, "source_file":"synthetic_generator", "valid_trial":True})
    return pd.DataFrame.from_records(rows)
