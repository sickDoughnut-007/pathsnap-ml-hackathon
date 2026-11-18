"""Build the shared candidate-feature tables used by every model.

Outputs (data/processed/):
  train_candidates_fold{0..4}.parquet  one row per (GPS point, candidate arc), 80 training trips,
                                   split by CV fold (load with pathsnap.dataset.load_candidates)
  student_test_candidates.parquet  same for the 10 student-testing trips
  train_points.parquet / student_test_points.parquet   one row per GPS point (labels, quality)
  folds.csv                        trip_id -> CV fold (GroupKFold by trip; identical for everyone)
  feature_list.txt                 the exact model feature columns
results/preprocessing/trip_quality.csv   per-trip label/candidate quality

Usage (from repo root):
  python scripts/02_build_features.py [--data-root PATH]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.features import FEATURES, build_trip_tables  # noqa: E402
from pathsnap.io import REPO_ROOT, list_trips, load_config, load_trip  # noqa: E402
from pathsnap.preprocess import prepare_trip  # noqa: E402


def assign_folds(points_per_trip, n_folds, seed):
    """Seeded, size-balanced assignment of whole trips to folds."""
    rng = np.random.default_rng(seed)
    trips = list(points_per_trip.index)
    rng.shuffle(trips)
    trips.sort(key=lambda t: -points_per_trip[t])  # stable: shuffled order breaks ties
    load = np.zeros(n_folds)
    fold = {}
    for t in trips:
        f = int(np.argmin(load))
        fold[t] = f
        load[f] += points_per_trip[t]
    return pd.Series(fold, name="fold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    cfg = load_config()
    out = REPO_ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    qdir = REPO_ROOT / "results" / "preprocessing"
    qdir.mkdir(parents=True, exist_ok=True)

    quality = []
    tables = {}
    for part in ["train", "student_test"]:
        cands, points = [], []
        for tid in list_trips(cfg, part, args.data_root):
            t0 = time.time()
            trip = load_trip(cfg, part, tid, args.data_root)
            pt, order = prepare_trip(trip, cfg)
            df, pts = build_trip_tables(trip, pt, cfg, order)
            cands.append(df)
            points.append(pts)
            ok = pts["label_ok"] == 1
            quality.append({
                "partition": part, "trip_id": tid, "n_points": len(pts),
                "n_segments": int(pts["segment"].nunique()),
                "frac_label_ok": float(ok.mean()),
                "frac_no_candidates": float((pts["n_cand"] == 0).mean()),
                "candidate_miss_rate_labelled": float(1 - pts.loc[ok, "ref_in_cands"].mean()) if ok.any() else np.nan,
                "nearest_is_ref_labelled": float((pts.loc[ok, "nearest_arc"] == pts.loc[ok, "ref_arc"]).mean()) if ok.any() else np.nan,
                "ref_dist_median_m": float(pts.loc[ok, "ref_dist"].median()) if ok.any() else np.nan,
                "ref_dist_p95_m": float(pts.loc[ok, "ref_dist"].quantile(0.95)) if ok.any() else np.nan,
                "mean_candidates": float(pts["n_cand"].mean()),
                "n_candidate_rows": len(df), "seconds": round(time.time() - t0, 2),
            })
            print(f"{part:12s} {tid}  pts={len(pts):6d} rows={len(df):7d} "
                  f"label_ok={ok.mean():.3f} nearest=ref={quality[-1]['nearest_is_ref_labelled']:.3f} "
                  f"({quality[-1]['seconds']}s)", flush=True)
        tables[part] = (pd.concat(cands, ignore_index=True), pd.concat(points, ignore_index=True))

    # Folds on training trips, balanced by labelled points.
    tr_c, tr_p = tables["train"]
    lab = tr_p[tr_p["label_ok"] == 1].groupby("trip_id").size().reindex(tr_p["trip_id"].unique(), fill_value=0)
    folds = assign_folds(lab, cfg["cv"]["n_folds"], cfg["cv"]["seed"])
    folds.index.name = "trip_id"
    folds.reset_index().to_csv(out / "folds.csv", index=False)

    for part, (c, p) in tables.items():
        if part == "train":
            c["fold"] = c["trip_id"].map(folds).astype(np.int8)
            p["fold"] = p["trip_id"].map(folds).astype(np.int8)
        else:
            c["fold"] = np.int8(-1)
            p["fold"] = np.int8(-1)
        # Per-point weight so every trip contributes equally regardless of its length
        # (trips 00000054, 00000084, 00000021 have 15k-19k points).
        n_lab = p[p["label_ok"] == 1].groupby("trip_id").size()
        w = (n_lab.mean() / n_lab).rename("trip_weight")
        c["trip_weight"] = c["trip_id"].map(w).fillna(0).astype(np.float32)
        if part == "train":
            # One file per CV fold keeps every file well under GitHub's 100 MB limit.
            for f in sorted(c["fold"].unique()):
                c[c["fold"] == f].to_parquet(out / f"train_candidates_fold{f}.parquet",
                                             index=False, compression="zstd")
        else:
            c.to_parquet(out / f"{part}_candidates.parquet", index=False, compression="zstd")
        p.to_parquet(out / f"{part}_points.parquet", index=False, compression="zstd")

    (out / "feature_list.txt").write_text("\n".join(FEATURES) + "\n")
    pd.DataFrame(quality).to_csv(qdir / "trip_quality.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
