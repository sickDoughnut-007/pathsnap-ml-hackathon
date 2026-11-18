"""TEMPLATE: copy to models/<your_model>/train.py and replace make_model().

Guarantees every model is trained and scored identically:
  - same features (pathsnap.dataset.FEATURES), same labelled rows, same trip-level CV folds
  - same metrics (pathsnap.evaluate), same student-testing partition
Writes models/<name>/metrics.json, cv_folds.csv and model.joblib.

Run from repo root:  python models/<your_model>/train.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pathsnap.dataset import FEATURES, load_candidates, load_points, model_rows  # noqa: E402
from pathsnap.evaluate import observation_metrics, top1_predictions, route_continuity  # noqa: E402
from pathsnap.io import load_config, load_trip  # noqa: E402

NAME = HERE.name
SEED = 42


def make_model():
    # Replace with your model. Scale features inside a Pipeline for LR / KNN / SVM / MLP.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))


def fit(model, X, y, w):
    # trip_weight makes every trip count equally; drop it if your model can't take weights.
    try:
        step = model.steps[-1][0] if hasattr(model, "steps") else None
        kw = {f"{step}__sample_weight": w} if step else {"sample_weight": w}
        return model.fit(X, y, **kw)
    except TypeError:
        return model.fit(X, y)


def main():
    cfg = load_config()
    tr, te = model_rows(load_candidates("train")), model_rows(load_candidates("student_test"))
    tr_pts, te_pts = load_points("train"), load_points("student_test")

    cv_rows = []
    for f in sorted(tr["fold"].unique()):
        a, v = tr[tr["fold"] != f], tr[tr["fold"] == f]
        m = fit(make_model(), a[FEATURES], a["label"], a["trip_weight"].values)
        s = m.predict_proba(v[FEATURES])[:, 1]
        r = observation_metrics(v, s, tr_pts[tr_pts["fold"] == f])
        r["fold"] = int(f)
        cv_rows.append(r)
        print(f"fold {f}: top1={r['top1_accuracy']:.4f} f1_macro={r['f1_macro']:.4f} logloss={r['log_loss']:.4f}")
    cv = pd.DataFrame(cv_rows)

    t0 = time.perf_counter()
    model = fit(make_model(), tr[FEATURES], tr["label"], tr["trip_weight"].values)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = model.predict_proba(te[FEATURES])[:, 1]
    infer_ms_per_1k = (time.perf_counter() - t0) / len(te) * 1e6
    test = observation_metrics(te, s, te_pts)
    arcs = {t: load_trip(cfg, "student_test", t).arcs for t in te["trip_id"].unique()}
    best = top1_predictions(te, s)
    test["route_continuity_1hop"], _ = route_continuity(best, arcs, 1)
    test["route_continuity_2hop"], _ = route_continuity(best, arcs, 2)

    out = {
        "model": NAME, "config": str(model), "seed": SEED,
        "cv_top1_mean": cv["top1_accuracy"].mean(), "cv_top1_std": cv["top1_accuracy"].std(),
        "cv_f1_macro_mean": cv["f1_macro"].mean(), "cv_f1_macro_std": cv["f1_macro"].std(),
        "cv_log_loss_mean": cv["log_loss"].mean(),
        "train_seconds": train_s, "inference_ms_per_1000_rows": infer_ms_per_1k,
        "student_test": test,
    }
    cv.to_csv(HERE / "cv_folds.csv", index=False)
    (HERE / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    joblib.dump(model, HERE / "model.joblib", compress=3)
    print(json.dumps({k: v for k, v in out.items() if k != "student_test"}, indent=1, default=float))
    print("student_test top1", test["top1_accuracy"], "f1_macro", test["f1_macro"], "log_loss", test["log_loss"])


if __name__ == "__main__":
    main()
