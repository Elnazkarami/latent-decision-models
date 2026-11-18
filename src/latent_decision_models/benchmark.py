"""Subject-disjoint confidence prediction baseline for synthetic pipeline checks."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import validate_trials


def benchmark(df: pd.DataFrame, features: list[str], n_splits: int = 4) -> dict:
    check = validate_trials(df)
    if not check["valid"]:
        raise ValueError("Invalid trial table: " + "; ".join(check["errors"]))
    required = ["subject", "confidence_raw", *features]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
    work = df.dropna(subset=["confidence_raw"]).copy()
    if work["subject"].nunique() < n_splits:
        raise ValueError("Not enough distinct participants for requested folds")
    x = work[features]
    y = work["confidence_raw"].astype(float)
    groups = work["subject"]
    prep = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for fold, (tr, te) in enumerate(splitter.split(x, y, groups), start=1):
        train_groups, test_groups = set(groups.iloc[tr]), set(groups.iloc[te])
        if train_groups & test_groups:
            raise RuntimeError("Subject leakage between folds")
        dummy = DummyRegressor(strategy="mean").fit(x.iloc[tr], y.iloc[tr])
        model = Pipeline([("preprocess", prep), ("ridge", Ridge(alpha=1.0))])
        model.fit(x.iloc[tr], y.iloc[tr])
        folds.append({"fold": fold, "n_train": len(tr), "n_test": len(te), "train_subjects": sorted(train_groups), "test_subjects": sorted(test_groups), "dummy_mae": float(mean_absolute_error(y.iloc[te], dummy.predict(x.iloc[te]))), "ridge_mae": float(mean_absolute_error(y.iloc[te], model.predict(x.iloc[te])))})
    return {"data_kind": "synthetic_or_user_supplied_unverified", "target": "confidence_raw", "validation": "group_k_fold_subject", "folds": folds, "mean_dummy_mae": float(np.mean([f["dummy_mae"] for f in folds])), "mean_ridge_mae": float(np.mean([f["ridge_mae"] for f in folds]))}
