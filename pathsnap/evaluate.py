"""Shared evaluation. Every model must be scored with these functions so results are comparable.

Definitions (state these in the report):
  Observation-level Top-1 accuracy: for each labelled GPS observation (label_ok == 1), take the
      candidate with the highest score; correct if it is the reference arc. Observations whose
      reference arc is missing from the candidate set count as WRONG (denominator = all labelled
      observations), so Top-1 <= 1 - candidate miss rate.
  Candidate miss rate: fraction of labelled observations whose reference arc is not a candidate.
  Accuracy / precision / recall / macro F1 / log loss: on candidate rows (binary label), at a
      0.5 threshold unless given. Macro F1 averages the F1 of class 0 and class 1.
  Route continuity: over consecutive observations in the same GPS segment whose predicted arc
      changes, the fraction where the new arc is reachable from the old one within `max_hops`
      network arcs (1 = directly connected head-to-tail). Same arc -> continuous.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def top1_predictions(cands, scores):
    """Return one row per observation with its highest-scoring candidate."""
    df = cands[["trip_id", "point_idx", "arc_id", "label", "segment"]].copy()
    df["score"] = np.asarray(scores)
    idx = df.groupby(["trip_id", "point_idx"], sort=False)["score"].idxmax()
    return df.loc[idx].rename(columns={"arc_id": "pred_arc"}).reset_index(drop=True)


def observation_metrics(cands, scores, points, threshold=0.5):
    """cands: model rows (label_ok == 1); scores: P(label=1) per row; points: *_points table."""
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    y = cands["label"].values
    pred = (scores >= threshold).astype(int)
    lab_pts = points[points["label_ok"] == 1]
    n_lab = len(lab_pts)
    best = top1_predictions(cands, scores)
    return {
        "top1_accuracy": float(best["label"].sum() / n_lab),
        "top1_given_ref_in_candidates": float(best["label"].sum() / max(1, lab_pts["ref_in_cands"].sum())),
        "candidate_miss_rate": float(1 - lab_pts["ref_in_cands"].mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "precision_pos": float(precision_score(y, pred, zero_division=0)),
        "recall_pos": float(recall_score(y, pred, zero_division=0)),
        "f1_pos": float(f1_score(y, pred)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "n_observations": int(n_lab),
        "n_candidate_rows": int(len(cands)),
    }


def _reachable(arcs, max_hops):
    """Build a function reachable(a, b) on the arc table (arc ids)."""
    out = {}
    for i, (u, v) in enumerate(arcs):
        out.setdefault(int(u), []).append(i)

    def reach(a, b):
        if a == b:
            return True
        frontier = {a}
        for _ in range(max_hops):
            nxt = set()
            for x in frontier:
                for c in out.get(int(arcs[x, 1]), []):
                    if c == b:
                        return True
                    nxt.add(c)
            frontier = nxt
        return False

    return reach


def route_continuity(pred_by_point, arcs_by_trip, max_hops=1):
    """pred_by_point: DataFrame [trip_id, point_idx, segment, pred_arc] (any column name for the arc
    can be passed via rename). arcs_by_trip: {trip_id: arcs array}. Returns overall fraction."""
    ok_total, n_total = 0, 0
    per_trip = {}
    for tid, g in pred_by_point.sort_values(["trip_id", "point_idx"]).groupby("trip_id"):
        reach = _reachable(arcs_by_trip[tid], max_hops)
        a = g["pred_arc"].values
        s = g["segment"].values
        ch = np.flatnonzero((a[1:] != a[:-1]) & (s[1:] == s[:-1]))
        ok = sum(reach(int(a[i]), int(a[i + 1])) for i in ch)
        per_trip[tid] = ok / len(ch) if len(ch) else 1.0
        ok_total += ok
        n_total += len(ch)
    return (ok_total / n_total if n_total else 1.0), per_trip


def timed_predict(predict_fn, X):
    """Returns (scores, seconds_per_1000_rows)."""
    t0 = time.perf_counter()
    s = predict_fn(X)
    dt = time.perf_counter() - t0
    return s, dt / max(1, len(X)) * 1000.0


def group_cv_folds(df):
    """Yield (train_idx, val_idx) using the shared, precomputed trip-level folds."""
    for f in sorted(df["fold"].unique()):
        yield np.flatnonzero(df["fold"].values != f), np.flatnonzero(df["fold"].values == f)


def metrics_frame(rows):
    return pd.DataFrame(rows)
