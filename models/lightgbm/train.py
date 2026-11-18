"""PathSnap LightGBM candidate-ranking model."""

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from pathsnap.dataset import (
    FEATURES,
    load_candidates,
    load_points,
    model_rows,
)
from pathsnap.evaluate import (
    observation_metrics,
    top1_predictions,
    route_continuity,
)
from pathsnap.io import load_config, load_trip


SEED = 42

PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 127,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def make_model():
    return lgb.LGBMClassifier(**PARAMS)


def fit(model, X, y, weights):
    return model.fit(X, y, sample_weight=weights)


def main():
    HERE.mkdir(parents=True, exist_ok=True)

    print("Loading shared processed dataset...", flush=True)

    cfg = load_config()

    train_candidates = load_candidates("train")
    test_candidates = load_candidates("student_test")

    tr = model_rows(train_candidates)
    te = model_rows(test_candidates)

    tr_pts = load_points("train")
    te_pts = load_points("student_test")

    # Validate the exact shared feature set.
    missing = sorted(set(FEATURES) - set(tr.columns))
    if missing:
        raise ValueError(f"Missing shared features: {missing}")

    required = {"label", "fold", "trip_id", "point_idx", "trip_weight"}
    missing = sorted(required - set(tr.columns))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    if not set(tr["label"].unique()).issubset({0, 1}):
        raise ValueError("Training labels must be binary (0/1).")

    folds = sorted(int(f) for f in tr["fold"].unique())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected five folds 0-4, got {folds}")

    if tr.empty or te.empty:
        raise ValueError("Training or student-test candidate data is empty.")

    print(f"Training candidate rows: {len(tr):,}")
    print(f"Training trips: {tr['trip_id'].nunique()}")
    print(f"Training features: {len(FEATURES)}")
    print(f"Positive candidate rate: {tr['label'].mean():.4f}")
    print(f"Student-test candidate rows: {len(te):,}")
    print("\nStarting 5-fold trip-grouped cross-validation...\n")

    cv_rows = []

    for fold in folds:
        train_fold = tr[tr["fold"] != fold]
        valid_fold = tr[tr["fold"] == fold]

        model = make_model()

        fit(
            model,
            train_fold[FEATURES],
            train_fold["label"].astype(int),
            train_fold["trip_weight"].to_numpy(),
        )

        scores = model.predict_proba(valid_fold[FEATURES])[:, 1]

        metrics = observation_metrics(
            valid_fold,
            scores,
            tr_pts[tr_pts["fold"] == fold],
        )

        metrics["fold"] = fold
        cv_rows.append(metrics)

        print(
            f"Fold {fold}: "
            f"top1={metrics['top1_accuracy']:.4f} | "
            f"macro_f1={metrics['f1_macro']:.4f} | "
            f"logloss={metrics['log_loss']:.4f}",
            flush=True,
        )

        del model, scores

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(HERE / "cv_folds.csv", index=False)

    print("\nTraining final model on all training folds...", flush=True)

    model = make_model()

    start = time.perf_counter()

    fit(
        model,
        tr[FEATURES],
        tr["label"].astype(int),
        tr["trip_weight"].to_numpy(),
    )

    train_seconds = time.perf_counter() - start

    print("Evaluating on student-test...", flush=True)

    start = time.perf_counter()
    test_scores = model.predict_proba(te[FEATURES])[:, 1]

    inference_ms_per_1000 = (
        (time.perf_counter() - start) / len(te) * 1_000_000
    )

    test_metrics = observation_metrics(
        te,
        test_scores,
        te_pts,
    )

    # Evaluation-only route continuity.
    # The reference route is not used as a model feature.
    print("Calculating route continuity...", flush=True)

    arcs = {
        trip_id: load_trip(cfg, "student_test", trip_id).arcs
        for trip_id in te["trip_id"].unique()
    }

    best = top1_predictions(te, test_scores)

    test_metrics["route_continuity_1hop"], _ = route_continuity(
        best, arcs, 1
    )

    test_metrics["route_continuity_2hop"], _ = route_continuity(
        best, arcs, 2
    )

    output = {
        "model": "lightgbm",
        "library": "LightGBM",
        "library_version": lgb.__version__,
        "parameters": PARAMS,
        "features": list(FEATURES),
        "seed": SEED,
        "cv_top1_mean": float(cv["top1_accuracy"].mean()),
        "cv_top1_std": float(cv["top1_accuracy"].std()),
        "cv_f1_macro_mean": float(cv["f1_macro"].mean()),
        "cv_f1_macro_std": float(cv["f1_macro"].std()),
        "cv_log_loss_mean": float(cv["log_loss"].mean()),
        "train_seconds": train_seconds,
        "inference_ms_per_1000_rows": inference_ms_per_1000,
        "student_test": test_metrics,
    }

    # Save final model and evaluation results.
    joblib.dump(model, HERE / "model.joblib", compress=3)

    with open(HERE / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=float)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "importance_split": model.booster_.feature_importance(
            importance_type="split"
        ),
    }).sort_values("importance_gain", ascending=False)

    importance.to_csv(HERE / "feature_importance.csv", index=False)

    print("\n========== LIGHTGBM RESULTS ==========")
    print(f"CV top-1 mean:    {output['cv_top1_mean']:.4f}")
    print(f"CV top-1 std:     {output['cv_top1_std']:.4f}")
    print(f"CV macro F1:      {output['cv_f1_macro_mean']:.4f}")
    print(f"CV log loss:      {output['cv_log_loss_mean']:.4f}")
    print(f"Training time:    {train_seconds:.2f} seconds")
    print(f"Inference ms/1k:  {inference_ms_per_1000:.4f}")

    print("\n========== STUDENT TEST ==========")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("\n========== TOP 15 FEATURES ==========")
    print(importance.head(15).to_string(index=False))

    print(f"\nAll artifacts saved in: {HERE}", flush=True)


if __name__ == "__main__":
    main()
