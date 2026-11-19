"""Efficient Bayesian optimisation for PathSnap LightGBM."""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "tuning"

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
MAX_TRIALS = 8


def make_params(trial):
    max_depth = trial.suggest_int("max_depth", 3, 8)

    return {
        "objective": "binary",
        "n_estimators": trial.suggest_int(
            "n_estimators", 200, 500, step=50
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.12, log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 4, min(32, 2 ** max_depth)
        ),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 40, 160
        ),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-3, 5.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10.0, log=True
        ),
        "max_bin": trial.suggest_categorical(
            "max_bin", [63, 127]
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain", 0.0, 0.3
        ),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 0.8, 2.0
        ),
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def fit_model(params, X, y, weights):
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, sample_weight=weights)
    return model


def save_trials(study, trial):
    """Persist trial results immediately after each completed/pruned trial."""
    study.trials_dataframe().to_csv(
        OUT / "optuna_trials.csv", index=False
    )

    completed = sum(
        t.state == TrialState.COMPLETE for t in study.trials
    )
    pruned = sum(
        t.state == TrialState.PRUNED for t in study.trials
    )

    print(
        f"Saved progress: {completed} completed, {pruned} pruned trials.",
        flush=True,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Loading shared processed dataset...", flush=True)

    cfg = load_config()

    tr = model_rows(load_candidates("train"))
    te = model_rows(load_candidates("student_test"))

    tr_pts = load_points("train")
    te_pts = load_points("student_test")

    missing = sorted(set(FEATURES) - set(tr.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")

    folds = sorted(int(f) for f in tr["fold"].unique())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected folds 0-4, got {folds}")

    if tr.empty or te.empty:
        raise ValueError("Training or student-test data is empty.")

    print(f"Training rows: {len(tr):,}")
    print(f"Training trips: {tr['trip_id'].nunique()}")
    print(f"Features: {len(FEATURES)}")
    print(f"Maximum trials: {MAX_TRIALS}")
    print("CV folds: 5")
    print("Objective: mean observation-level Top-1 accuracy")
    print("\nStarting optimisation...\n", flush=True)

    def objective(trial):
        params = make_params(trial)

        fold_metrics = []
        fold_scores = []

        for fold in folds:
            train_fold = tr[tr["fold"] != fold]
            valid_fold = tr[tr["fold"] == fold]

            model = fit_model(
                params,
                train_fold[FEATURES],
                train_fold["label"].astype(int),
                train_fold["trip_weight"].to_numpy(),
            )

            scores = model.predict_proba(
                valid_fold[FEATURES]
            )[:, 1]

            metrics = observation_metrics(
                valid_fold,
                scores,
                tr_pts[tr_pts["fold"] == fold],
            )

            fold_metrics.append({
                "fold": fold,
                **metrics,
            })

            fold_scores.append(metrics["top1_accuracy"])

            del model, scores, train_fold, valid_fold
            gc.collect()

            current_mean = float(np.mean(fold_scores))
            trial.report(current_mean, step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_top1 = float(np.mean(fold_scores))

        trial.set_user_attr("fold_metrics", fold_metrics)
        trial.set_user_attr(
            "mean_macro_f1",
            float(np.mean([m["f1_macro"] for m in fold_metrics])),
        )
        trial.set_user_attr(
            "mean_log_loss",
            float(np.mean([m["log_loss"] for m in fold_metrics])),
        )

        print(
            f"Trial {trial.number + 1}: "
            f"Top-1={mean_top1:.4f} | "
            f"Macro F1={trial.user_attrs['mean_macro_f1']:.4f} | "
            f"Log loss={trial.user_attrs['mean_log_loss']:.4f}",
            flush=True,
        )

        return mean_top1

    db_path = (OUT / "optuna_study.db").resolve().as_posix()
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="pathsnap_lightgbm_fast_tuning",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=1,
        ),
    )

    existing = len(study.trials)
    remaining = max(0, MAX_TRIALS - existing)

    print(
        f"Existing trials: {existing} | Remaining: {remaining}",
        flush=True,
    )

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            callbacks=[save_trials],
            show_progress_bar=False,
        )

    completed = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
    ]

    if not completed:
        raise RuntimeError(
            "No completed trials. Check the logs before retrying."
        )

    best_trial = study.best_trial
    best_params = {
        "objective": "binary",
        **best_trial.params,
        "subsample_freq": 1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }

    print("\n========== BEST TRIAL ==========")
    print(f"Trial: {best_trial.number + 1}")
    print(f"CV Top-1: {best_trial.value:.4f}")
    print(f"Parameters: {best_params}", flush=True)

    with open(OUT / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    cv = pd.DataFrame(best_trial.user_attrs["fold_metrics"])
    cv.to_csv(OUT / "cv_folds.csv", index=False)

    print("\nTraining final tuned model...", flush=True)

    start = time.perf_counter()

    final_model = fit_model(
        best_params,
        tr[FEATURES],
        tr["label"].astype(int),
        tr["trip_weight"].to_numpy(),
    )

    train_seconds = time.perf_counter() - start

    print("Evaluating student-test once...", flush=True)

    start = time.perf_counter()
    test_scores = final_model.predict_proba(te[FEATURES])[:, 1]

    inference_ms_per_1000 = (
        (time.perf_counter() - start) / len(te) * 1_000_000
    )

    test_metrics = observation_metrics(
        te,
        test_scores,
        te_pts,
    )

    arcs = {
        tid: load_trip(cfg, "student_test", tid).arcs
        for tid in te["trip_id"].unique()
    }

    predictions = top1_predictions(te, test_scores)

    test_metrics["route_continuity_1hop"], _ = route_continuity(
        predictions, arcs, 1
    )
    test_metrics["route_continuity_2hop"], _ = route_continuity(
        predictions, arcs, 2
    )

    cv_top1 = float(cv["top1_accuracy"].mean())

    output = {
        "model": "lightgbm_bayesian_tuned",
        "library": "LightGBM",
        "library_version": lgb.__version__,
        "optuna_version": optuna.__version__,
        "optimization": {
            "method": "Optuna TPE Bayesian optimization",
            "max_trials": MAX_TRIALS,
            "total_trials": len(study.trials),
            "completed_trials": len(completed),
            "pruned_trials": sum(
                t.state == TrialState.PRUNED for t in study.trials
            ),
            "objective": "Mean 5-fold trip-grouped observation-level Top-1",
            "best_trial": best_trial.number + 1,
            "best_trial_top1": float(best_trial.value),
        },
        "parameters": best_params,
        "features": list(FEATURES),
        "seed": SEED,
        "cv_top1_mean": cv_top1,
        "cv_top1_std": float(cv["top1_accuracy"].std()),
        "cv_f1_macro_mean": float(cv["f1_macro"].mean()),
        "cv_log_loss_mean": float(cv["log_loss"].mean()),
        "train_seconds": train_seconds,
        "inference_ms_per_1000_rows": inference_ms_per_1000,
        "student_test": test_metrics,
    }

    joblib.dump(final_model, OUT / "model.joblib", compress=3)

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=float)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_gain": final_model.booster_.feature_importance(
            importance_type="gain"
        ),
        "importance_split": final_model.booster_.feature_importance(
            importance_type="split"
        ),
    }).sort_values("importance_gain", ascending=False)

    importance.to_csv(OUT / "feature_importance.csv", index=False)

    baseline_path = HERE / "metrics.json"
    comparison = {}

    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        comparison = {
            "baseline_cv_top1": baseline["cv_top1_mean"],
            "tuned_cv_top1": cv_top1,
            "cv_change": cv_top1 - baseline["cv_top1_mean"],
            "baseline_student_top1": baseline["student_test"]["top1_accuracy"],
            "tuned_student_top1": test_metrics["top1_accuracy"],
            "student_change": (
                test_metrics["top1_accuracy"]
                - baseline["student_test"]["top1_accuracy"]
            ),
            "baseline_student_macro_f1": baseline["student_test"]["f1_macro"],
            "tuned_student_macro_f1": test_metrics["f1_macro"],
        }

    with open(OUT / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n========== TUNED RESULTS ==========")
    print(f"CV Top-1 mean: {cv_top1:.4f}")
    print(f"CV Top-1 std: {output['cv_top1_std']:.4f}")
    print(f"CV Macro F1: {output['cv_f1_macro_mean']:.4f}")
    print(f"CV Log loss: {output['cv_log_loss_mean']:.4f}")
    print(f"Training time: {train_seconds:.2f} seconds")
    print(f"Inference ms/1k: {inference_ms_per_1000:.4f}")

    print("\n========== STUDENT TEST ==========")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("\n========== BASELINE COMPARISON ==========")
    print(json.dumps(comparison, indent=2))

    print(f"\nArtifacts saved in: {OUT}", flush=True)


if __name__ == "__main__":
    main()
