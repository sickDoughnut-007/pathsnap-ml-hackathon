"""Loading the shared processed tables. Every model should use these functions.

Example:
    from pathsnap.dataset import load_candidates, load_points, FEATURES, model_rows
    train = model_rows(load_candidates("train"))
    test = model_rows(load_candidates("student_test"))
    X, y, w, fold = train[FEATURES], train["label"], train["trip_weight"], train["fold"]
"""
from pathlib import Path

import pandas as pd

from .features import FEATURES
from .io import REPO_ROOT

PROCESSED = REPO_ROOT / "data" / "processed"


def load_candidates(partition):
    if partition == "train":
        files = sorted(PROCESSED.glob("train_candidates_fold*.parquet"))
        if not files:
            raise FileNotFoundError("Run scripts/02_build_features.py first (no train_candidates_fold*.parquet).")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(PROCESSED / f"{partition}_candidates.parquet")


def load_points(partition):
    return pd.read_parquet(PROCESSED / f"{partition}_points.parquet")


def model_rows(cands):
    """Rows used for training / scoring: only observations with a reliable reference label."""
    return cands[cands["label_ok"] == 1].reset_index(drop=True)


__all__ = ["FEATURES", "load_candidates", "load_points", "model_rows", "PROCESSED"]
