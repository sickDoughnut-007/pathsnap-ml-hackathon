"""Logistic Regression baseline for PathSnap candidate scoring.

This is a learned version of the classic Newson-Krumm Gaussian emission model.
Logistic regression is a linear, probabilistic classifier that is well-calibrated
by construction — its predicted P(label=1) can be used directly as HMM emission
probabilities without post-hoc calibration.

Key design choices:
  - StandardScaler: LR needs feature scaling since features have different units
    (metres, seconds, degrees, counts).
  - Interaction features: a linear model cannot learn interactions on its own, so
    we engineer the ones that matter most:
      * heading × speed indicators: GPS heading is unreliable at low speed (jitter),
        so heading_diff matters less when speed is low or straightness is poor.
      * dist × context: distance combined with contextual distance signals.
      * rank × n_cand: being rank 0 out of 2 candidates is different from rank 0 out of 16.
  - No class_weight="balanced": that would distort probabilities and break the HMM.
  - trip_weight as sample_weight: so large trips don't dominate training.

Run from repo root:
    python models/logistic_regression/train.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pathsnap.dataset import FEATURES, load_candidates, load_points, model_rows  # noqa: E402
from pathsnap.evaluate import observation_metrics, top1_predictions, route_continuity  # noqa: E402
from pathsnap.io import load_config, load_trip  # noqa: E402

NAME = HERE.name
SEED = 42
# Keep the search small enough for a practical training run while covering
# under-, well-, and over-regularised logistic models.
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


# ---------------------------------------------------------------------------
# Interaction feature engineering
# ---------------------------------------------------------------------------
class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Add hand-crafted interaction features that a linear model cannot learn.

    Motivation (from EDA / domain knowledge):
      1. GPS heading is meaningless at low speed (vehicle stopped, jitter).
         The raw heading_diff feature misleads the model for ~15% of observations.
         Interactions heading_diff × speed and heading_diff × straightness let the
         model down-weight heading when it is unreliable.
      2. dist × ctx_dist_mean: if both the observation and its neighbours are close
         to a candidate, that candidate is more likely correct.
      3. rank × n_cand: being the nearest candidate (rank 0) out of 2 is a stronger
         signal than rank 0 out of 16.
      4. cos_heading × speed_window: cosine similarity is more meaningful at speed.
      5. dist_rel × link_prev_nearest: relative distance combined with topological
         link to the previous observation's nearest arc.
    """

    # Map feature name -> column index (computed at fit time)
    _idx = None

    # Define the interactions as (feat_a, feat_b) pairs
    INTERACTIONS = [
        # Heading reliability at low speed / jitter
        ("heading_diff", "speed"),
        ("heading_diff", "straightness"),
        ("heading_diff_undirected", "speed"),
        ("cos_heading", "speed"),
        ("cos_heading", "speed_window"),
        ("cos_heading", "straightness"),
        # Distance × context
        ("dist", "ctx_dist_mean"),
        ("dist_rel", "ctx_dist_mean"),
        ("dist_rel", "link_prev_nearest"),
        ("dist_rel", "link_next_nearest"),
        # Rank interactions
        ("rank", "n_cand"),
        # Context continuity interactions
        ("ctx_cand_prev", "ctx_cand_next"),
        ("link_prev_nearest", "link_next_nearest"),
        # Along-arc motion at speed
        ("progress_prev", "speed"),
        ("progress_next", "speed"),
    ]

    def fit(self, X, y=None):
        # Build a name -> column index lookup
        self._idx = {name: i for i, name in enumerate(FEATURES)}
        # Validate that all interaction features exist
        for a, b in self.INTERACTIONS:
            if a not in self._idx:
                raise ValueError(f"Feature '{a}' not found in FEATURES list")
            if b not in self._idx:
                raise ValueError(f"Feature '{b}' not found in FEATURES list")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        extras = []
        for a, b in self.INTERACTIONS:
            ia, ib = self._idx[a], self._idx[b]
            extras.append((X[:, ia] * X[:, ib]).reshape(-1, 1))
        return np.hstack([X] + extras)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = FEATURES
        base = list(input_features)
        for a, b in self.INTERACTIONS:
            base.append(f"{a}×{b}")
        return base


def make_model(C=1.0):
    """Build the logistic regression pipeline.

    Pipeline:
      1. InteractionFeatures: add domain-driven interactions
      2. StandardScaler: zero-mean unit-variance (required for L2 regularisation)
      3. LogisticRegression: L2-regularised, lbfgs solver

    The lbfgs solver is efficient for the ~50 features (37 base + 15 interactions)
    and handles L2 well. max_iter=2000 gives plenty of room for convergence.
    """
    return make_pipeline(
        InteractionFeatures(),
        StandardScaler(),
        LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=3000,
            random_state=SEED,
            # NO class_weight="balanced" — would distort emission probabilities
        ),
    )


def fit(model, X, y, w):
    """Fit the pipeline, passing trip_weight as sample_weight to the LR step."""
    try:
        step = model.steps[-1][0] if hasattr(model, "steps") else None
        kw = {f"{step}__sample_weight": w} if step else {"sample_weight": w}
        return model.fit(X, y, **kw)
    except TypeError:
        return model.fit(X, y)


def tune_c(tr, tr_pts):
    """Select C using only the fixed trip-level training folds.

    The score balances the three objectives used by this project: candidate
    selection (top-1), classification quality (macro F1), and calibrated
    probabilities (log loss). All candidates are evaluated on identical folds.
    """
    rows = []
    for C in C_GRID:
        fold_rows = []
        for f in sorted(tr["fold"].unique()):
            a, v = tr[tr["fold"] != f], tr[tr["fold"] == f]
            model = fit(make_model(C), a[FEATURES], a["label"], a["trip_weight"].values)
            scores = model.predict_proba(v[FEATURES])[:, 1]
            metrics = observation_metrics(v, scores, tr_pts[tr_pts["fold"] == f])
            fold_rows.append(metrics)
        fold_metrics = pd.DataFrame(fold_rows)
        row = {
            "C": C,
            "top1_accuracy": fold_metrics["top1_accuracy"].mean(),
            "f1_macro": fold_metrics["f1_macro"].mean(),
            "log_loss": fold_metrics["log_loss"].mean(),
        }
        row["selection_score"] = row["top1_accuracy"] + row["f1_macro"] - row["log_loss"]
        rows.append(row)
        print(f"  C={C:g}: top1={row['top1_accuracy']:.4f}"
              f"  f1_macro={row['f1_macro']:.4f}  log_loss={row['log_loss']:.4f}"
              f"  score={row['selection_score']:.4f}")

    results = pd.DataFrame(rows)
    best = results.sort_values(
        ["selection_score", "log_loss", "f1_macro", "top1_accuracy"],
        ascending=[False, True, False, False],
    ).iloc[0]
    return float(best["C"]), results


def main():
    cfg = load_config()
    print(f"Loading data...")
    tr = model_rows(load_candidates("train"))
    te = model_rows(load_candidates("student_test"))
    tr_pts = load_points("train")
    te_pts = load_points("student_test")
    print(f"  Train: {len(tr):,} candidate rows, {tr['trip_id'].nunique()} trips")
    print(f"  Test:  {len(te):,} candidate rows, {te['trip_id'].nunique()} trips")
    print(f"  Features: {len(FEATURES)} base + {len(InteractionFeatures.INTERACTIONS)} interactions"
          f" = {len(FEATURES) + len(InteractionFeatures.INTERACTIONS)} total")
    print()

    # -----------------------------------------------------------------------
    # Tune regularization on the fixed trip-level folds.
    # -----------------------------------------------------------------------
    print("Tuning Logistic Regression regularization...")
    best_c, tuning = tune_c(tr, tr_pts)
    print(f"  Selected C={best_c:g}")
    print()

    # -----------------------------------------------------------------------
    # 5-fold cross-validation with the selected configuration
    # -----------------------------------------------------------------------
    cv_rows = []
    for f in sorted(tr["fold"].unique()):
        a, v = tr[tr["fold"] != f], tr[tr["fold"] == f]
        m = fit(make_model(best_c), a[FEATURES], a["label"], a["trip_weight"].values)
        s = m.predict_proba(v[FEATURES])[:, 1]
        r = observation_metrics(v, s, tr_pts[tr_pts["fold"] == f])
        r["fold"] = int(f)
        cv_rows.append(r)
        print(f"  fold {f}: top1={r['top1_accuracy']:.4f}  f1_macro={r['f1_macro']:.4f}"
              f"  log_loss={r['log_loss']:.4f}  recall+={r['recall_pos']:.4f}")
    cv = pd.DataFrame(cv_rows)
    print()
    print(f"  CV mean  top1={cv['top1_accuracy'].mean():.4f} ± {cv['top1_accuracy'].std():.4f}"
          f"   f1_macro={cv['f1_macro'].mean():.4f}   log_loss={cv['log_loss'].mean():.4f}")
    print()

    # -----------------------------------------------------------------------
    # Refit on all 80 training trips
    # -----------------------------------------------------------------------
    print("Refitting on all training data...")
    t0 = time.perf_counter()
    model = fit(make_model(best_c), tr[FEATURES], tr["label"], tr["trip_weight"].values)
    train_s = time.perf_counter() - t0
    print(f"  Training time: {train_s:.2f}s")

    # -----------------------------------------------------------------------
    # Score the 10 student-testing trips
    # -----------------------------------------------------------------------
    print("Scoring student_test...")
    t0 = time.perf_counter()
    s = model.predict_proba(te[FEATURES])[:, 1]
    infer_ms_per_1k = (time.perf_counter() - t0) / len(te) * 1e6
    test = observation_metrics(te, s, te_pts)
    best = top1_predictions(te, s)
    try:
        arcs = {t: load_trip(cfg, "student_test", t).arcs for t in te["trip_id"].unique()}
        test["route_continuity_1hop"], _ = route_continuity(best, arcs, 1)
        test["route_continuity_2hop"], _ = route_continuity(best, arcs, 2)
        print(f"  route_continuity 1-hop={test['route_continuity_1hop']:.4f}"
              f"  2-hop={test['route_continuity_2hop']:.4f}")
    except FileNotFoundError:
        print("  (route continuity skipped — raw data not found; set PATHSNAP_DATA_ROOT)")
    print(f"  student_test top1={test['top1_accuracy']:.4f}  f1_macro={test['f1_macro']:.4f}"
          f"  log_loss={test['log_loss']:.4f}")

    # -----------------------------------------------------------------------
    # Print learned coefficients (for interpretability / report)
    # -----------------------------------------------------------------------
    lr = model.named_steps["logisticregression"]
    interaction_feat = InteractionFeatures()
    interaction_feat.fit(np.zeros((1, len(FEATURES))))
    all_names = interaction_feat.get_feature_names_out()
    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    print(f"\n  Intercept: {intercept:.4f}")
    print(f"\n  Top 15 coefficients (by |weight|):")
    order = np.argsort(np.abs(coefs))[::-1]
    for i, idx in enumerate(order[:15]):
        print(f"    {i+1:2d}. {all_names[idx]:30s}  {coefs[idx]:+.4f}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    out = {
        "model": NAME,
        "config": str(model),
        "seed": SEED,
        "selected_C": best_c,
        "tuning": tuning.to_dict(orient="records"),
        "n_base_features": len(FEATURES),
        "n_interaction_features": len(InteractionFeatures.INTERACTIONS),
        "n_total_features": len(FEATURES) + len(InteractionFeatures.INTERACTIONS),
        "cv_top1_mean": cv["top1_accuracy"].mean(),
        "cv_top1_std": cv["top1_accuracy"].std(),
        "cv_f1_macro_mean": cv["f1_macro"].mean(),
        "cv_f1_macro_std": cv["f1_macro"].std(),
        "cv_log_loss_mean": cv["log_loss"].mean(),
        "train_seconds": train_s,
        "inference_ms_per_1000_rows": infer_ms_per_1k,
        "student_test": test,
        "intercept": float(intercept),
        "top_coefficients": {all_names[idx]: float(coefs[idx]) for idx in order[:20]},
    }
    cv.to_csv(HERE / "cv_folds.csv", index=False)
    (HERE / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    joblib.dump(model, HERE / "model.joblib", compress=3)
    print(f"\nSaved to {HERE}:")
    print(f"  metrics.json, cv_folds.csv, model.joblib")
    print(json.dumps({k: v for k, v in out.items() if k != "student_test"}, indent=1, default=float))


if __name__ == "__main__":
    main()
