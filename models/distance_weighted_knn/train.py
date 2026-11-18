"""Train and evaluate KNN using the shared PathSnap candidate and point tables."""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.dataset import FEATURES, load_candidates, load_points, model_rows
from pathsnap.evaluate import group_cv_folds, observation_metrics, route_continuity, top1_predictions
from pathsnap.io import load_config, load_trip
from pathsnap.knn import TripWeightedKNN


def fit(rows):
    model = make_pipeline(StandardScaler(), PCA(n_components=8, svd_solver="covariance_eigh"),
                          TripWeightedKNN(n_neighbors=25, workers=4))
    start = perf_counter()
    model.fit(rows[FEATURES], rows.label, tripweightedknn__sample_weight=rows.trip_weight)
    return model, perf_counter() - start


def evaluate(model, rows, points):
    X = rows[FEATURES]
    start = perf_counter()
    scores = model.predict_proba(X)[:, 1]
    seconds = perf_counter() - start
    metrics = observation_metrics(rows, scores, points)
    metrics["inference_ms_per_1000_rows"] = seconds * 1e6 / len(rows)
    metrics["pca_variance_fraction"] = float(model[1].explained_variance_ratio_.sum())
    return metrics, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Raw trip directory for route-continuity evaluation")
    args = parser.parse_args()
    train = model_rows(load_candidates("train"))
    points = load_points("train")
    folds = []
    for f, (fit_idx, valid_idx) in enumerate(group_cv_folds(train)):
        model, seconds = fit(train.iloc[fit_idx])
        val = train.iloc[valid_idx]
        assert set(val.fold) == {f}
        metrics, _ = evaluate(model, val, points[points.fold == f])
        folds.append(dict(fold=f, fit_seconds=seconds, **metrics))
        print(f"Fold {f}: top-1 {metrics['top1_accuracy']:.4f}", flush=True)
        del model, val
    cv = pd.DataFrame(folds)
    cv.to_csv(HERE / "cv_folds.csv", index=False)
    model, seconds = fit(train)
    n_train = len(train)
    del train
    test = model_rows(load_candidates("student_test"))
    points = load_points("student_test")
    metrics, scores = evaluate(model, test, points)
    best = top1_predictions(test, scores)
    cfg = load_config()
    arcs = {t: load_trip(cfg, "student_test", t, args.data_root, with_route=False).arcs
            for t in points.trip_id.unique()}
    for hops in (1, 2):
        metrics[f"route_continuity_{hops}hop"] = route_continuity(best, arcs, hops)[0]
    trip_rows = []
    for tid, group in points.groupby("trip_id"):
        labelled = group[group.label_ok == 1]
        correct = int(best.loc[best.trip_id == tid, "label"].sum())
        trip_rows.append(dict(trip_id=tid, labelled_points=len(labelled), correct=correct,
                              top1_accuracy=correct / len(labelled)))
    pd.DataFrame(trip_rows).to_csv(HERE / "student_trips.csv", index=False)
    result = dict(model="distance_weighted_knn", train_rows=n_train, features=FEATURES,
                  config=str(model), train_seconds=seconds, fold_seed=cfg["cv"]["seed"],
                  cv_top1_mean=cv.top1_accuracy.mean(), cv_top1_std=cv.top1_accuracy.std(),
                  cv_f1_macro_mean=cv.f1_macro.mean(), cv_f1_macro_std=cv.f1_macro.std(),
                  cv_log_loss_mean=cv.log_loss.mean(), student_test=metrics,
                  python=platform.python_version(), packages={p: version(p) for p in
                  ("scikit-learn", "scipy", "numpy", "pandas", "pyarrow", "joblib")})
    joblib.dump(model, HERE / "model.joblib", compress=3)
    restored = joblib.load(HERE / "model.joblib")
    np.testing.assert_allclose(restored.predict_proba(test[FEATURES].iloc[:256])[:, 1], scores[:256])
    (HERE / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
