import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathsnap.io import load_config, load_trip
from pathsnap.dataset import load_candidates, load_points, model_rows
from pathsnap.features import FEATURES
from models.hmm_viterbi.decoder import (
    decode_trip,
    classifier_predictions,
)

RESULTS = ROOT / "models" / "hmm_viterbi" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "models" / "lightgbm" / "tuning" / "model.joblib"


def metric_summary(points, predictions, arcs):
    labelled = points[points["label_ok"] == 1].copy()
    labelled["prediction"] = labelled["point_idx"].map(predictions)

    valid = labelled["prediction"].notna()
    labelled_valid = labelled[valid]

    predicted = labelled["prediction"].fillna(-1).astype(int)
    correct = predicted == labelled["ref_arc"].astype(int)
    top1 = float(correct.mean()) if len(correct) else 0.0
    candidate_miss = float(
        (labelled["ref_in_cands"] == 0).mean()
    ) if "ref_in_cands" in labelled else None

    transition_count = 0
    one_hop_count = 0
    two_hop_count = 0

    for _, group in points.groupby("segment", sort=False):
        group = group.sort_values("point_idx")
        rows = list(group.itertuples(index=False))

        for prev, curr in zip(rows[:-1], rows[1:]):
            if int(curr.point_idx) != int(prev.point_idx) + 1:
                continue

            a = predictions.get(int(prev.point_idx))
            b = predictions.get(int(curr.point_idx))
            if a is None or b is None:
                continue

            transition_count += 1

            if a == b or arcs[a, 1] == arcs[b, 0]:
                one_hop_count += 1
                two_hop_count += 1
                continue

            middle = np.flatnonzero(arcs[:, 0] == arcs[a, 1])
            reaches = np.any(arcs[middle, 1] == arcs[b, 0]) if len(middle) else False
            if reaches:
                two_hop_count += 1

    return {
        "top1_accuracy": top1,
        "labelled_observations": int(len(labelled)),
        "predicted_labelled_observations": int(len(labelled_valid)),
        "correct_labelled_observations": int(correct.sum()),
        "candidate_miss_rate": candidate_miss,
        "route_continuity_1hop": (
            one_hop_count / transition_count if transition_count else 0.0
        ),
        "route_continuity_2hop": (
            two_hop_count / transition_count if transition_count else 0.0
        ),
        "evaluated_transitions": int(transition_count),
    }


def score_partition(points, candidates, probabilities, params, cfg, partition):
    all_predictions = {}
    all_classifier = {}
    arcs_by_trip = {}

    for trip_id, trip_points in points.groupby("trip_id", sort=False):
        trip_candidates = candidates[candidates["trip_id"] == trip_id].copy()
        trip_probs = probabilities[trip_candidates.index.to_numpy()]

        trip = load_trip(cfg, partition, trip_id)
        arcs = trip.arcs
        arcs_by_trip[trip_id] = arcs

        classifier = classifier_predictions(trip_candidates, trip_probs)
        decoded = decode_trip(
            trip_points,
            trip_candidates,
            arcs,
            trip_probs,
            params
        )

        all_classifier[trip_id] = classifier
        all_predictions[trip_id] = decoded

    classifier_metrics = {}
    hmm_metrics = {}

    for trip_id, trip_points in points.groupby("trip_id", sort=False):
        arcs = arcs_by_trip[trip_id]
        classifier_metrics[trip_id] = metric_summary(
            trip_points, all_classifier[trip_id], arcs
        )
        hmm_metrics[trip_id] = metric_summary(
            trip_points, all_predictions[trip_id], arcs
        )

    def aggregate(per_trip):
        result = {}
        total_labelled = sum(
            v["labelled_observations"] for v in per_trip.values()
        )
        total_correct = sum(
            v["correct_labelled_observations"] for v in per_trip.values()
        )
        result["top1_accuracy"] = (
            total_correct / total_labelled if total_labelled else 0.0
        )
        for key in ["route_continuity_1hop", "route_continuity_2hop"]:
            values = [v[key] for v in per_trip.values()]
            result[key] = float(np.mean(values)) if values else 0.0
        result["trips"] = len(per_trip)
        result["labelled_observations"] = int(sum(
            v["labelled_observations"] for v in per_trip.values()
        ))
        result["evaluated_transitions"] = int(sum(
            v["evaluated_transitions"] for v in per_trip.values()
        ))
        return result

    prediction_rows = []
    for trip_id, trip_points in points.groupby("trip_id", sort=False):
        for row in trip_points.itertuples(index=False):
            idx = int(row.point_idx)
            prediction_rows.append({
                "trip_id": trip_id,
                "point_idx": idx,
                "segment": int(row.segment),
                "ref_arc": int(row.ref_arc) if row.label_ok == 1 else None,
                "label_ok": int(row.label_ok),
                "classifier_pred_arc": all_classifier[trip_id].get(idx),
                "hmm_pred_arc": all_predictions[trip_id].get(idx),
            })

    return (
        aggregate(classifier_metrics),
        aggregate(hmm_metrics),
        pd.DataFrame(prediction_rows)
    )


def main():
    start = time.time()
    cfg = load_config()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing tuned LightGBM model: {MODEL_PATH}"
        )

    print("Loading tuned LightGBM model...")
    saved_model = joblib.load(MODEL_PATH)
    model_params = saved_model.get_params()

    print("Loading training data...")
    train_candidates = load_candidates("train")
    train_points = load_points("train")
    train_labelled = model_rows(train_candidates)

    print(f"Training candidate rows: {len(train_candidates):,}")
    print(f"Training labelled rows: {len(train_labelled):,}")

    folds = sorted(train_candidates["fold"].dropna().unique())
    if len(folds) < 2:
        raise RuntimeError("Expected at least two training CV folds.")

    oof_probabilities = np.full(len(train_candidates), np.nan, dtype=float)

    print("Generating out-of-fold LightGBM predictions...")
    for fold in folds:
        fold_start = time.time()

        train_mask = (
            (train_labelled["fold"] != fold)
            & train_labelled["label_ok"].eq(1)
        )
        validation_mask = train_candidates["fold"] == fold

        fit_rows = train_labelled.loc[train_mask]
        validation_rows = train_candidates.loc[validation_mask]

        fold_model = LGBMClassifier(**model_params)
        fold_model.fit(
            fit_rows[FEATURES],
            fit_rows["label"].astype(int)
        )

        positions = np.flatnonzero(validation_mask.to_numpy())
        oof_probabilities[positions] = fold_model.predict_proba(
            validation_rows[FEATURES]
        )[:, 1]

        print(
            f"  Fold {fold}: {len(fit_rows):,} fit rows, "
            f"{len(validation_rows):,} scored rows "
            f"({time.time() - fold_start:.1f}s)"
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Some training rows did not receive OOF predictions.")

    np.save(RESULTS / "train_oof_probabilities.npy", oof_probabilities)

    # Tune decoder parameters only on out-of-fold training predictions.
    parameter_grid = [
        {"weight": 0.5, "sigma_m": 30.0, "disconnected_penalty": 6.0,
         "backward_penalty": 3.0, "backward_tolerance_m": 5.0, "error_cap": 8.0},
        {"weight": 1.0, "sigma_m": 30.0, "disconnected_penalty": 6.0,
         "backward_penalty": 3.0, "backward_tolerance_m": 5.0, "error_cap": 8.0},
        {"weight": 1.0, "sigma_m": 60.0, "disconnected_penalty": 8.0,
         "backward_penalty": 5.0, "backward_tolerance_m": 5.0, "error_cap": 8.0},
        {"weight": 2.0, "sigma_m": 60.0, "disconnected_penalty": 8.0,
         "backward_penalty": 5.0, "backward_tolerance_m": 5.0, "error_cap": 8.0},
    ]

    print("Tuning Viterbi parameters on training OOF predictions...")
    tuning_results = []

    for i, params in enumerate(parameter_grid, start=1):
        tune_start = time.time()
        _, hmm_metrics, _ = score_partition(
            train_points,
            train_candidates,
            oof_probabilities,
            params,
            cfg,
            "train"
        )

        result = {
            "trial": i,
            "params": params,
            "train_oof_top1": hmm_metrics["top1_accuracy"],
            "train_oof_continuity_1hop": hmm_metrics["route_continuity_1hop"],
            "train_oof_continuity_2hop": hmm_metrics["route_continuity_2hop"],
            "seconds": time.time() - tune_start,
        }
        tuning_results.append(result)

        print(
            f"  Trial {i}/{len(parameter_grid)}: "
            f"Top-1={result['train_oof_top1']:.4%}, "
            f"1-hop={result['train_oof_continuity_1hop']:.4%}, "
            f"time={result['seconds']:.1f}s"
        )

    best = max(tuning_results, key=lambda x: x["train_oof_top1"])
    best_params = best["params"]

    with open(RESULTS / "tuning_results.json", "w") as f:
        json.dump(tuning_results, f, indent=2)

    with open(RESULTS / "best_decoder_config.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nSelected decoder parameters:")
    print(json.dumps(best_params, indent=2))

    # Final LightGBM model is used for student-test inference only.
    print("\nScoring student-test candidates...")
    student_candidates = load_candidates("student_test")
    student_points = load_points("student_test")

    student_probabilities = saved_model.predict_proba(
        student_candidates[FEATURES]
    )[:, 1]

    classifier_metrics, hmm_metrics, predictions = score_partition(
        student_points,
        student_candidates,
        student_probabilities,
        best_params,
        cfg,
        "student_test"
    )

    predictions.to_csv(RESULTS / "student_test_predictions.csv", index=False)

    comparison = {
        "classifier_only": classifier_metrics,
        "classifier_plus_hmm_viterbi": hmm_metrics,
        "selected_decoder_params": best_params,
        "training_oof_best_trial": best,
        "runtime_seconds": time.time() - start,
    }

    with open(RESULTS / "student_test_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n========== STUDENT TEST COMPARISON ==========")
    print(json.dumps(comparison, indent=2))
    print(f"\nResults saved to: {RESULTS}")
    print(f"Total runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
