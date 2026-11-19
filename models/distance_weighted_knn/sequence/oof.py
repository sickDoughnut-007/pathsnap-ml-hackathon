"""Score every training candidate with a KNN fitted on the other four whole-trip folds."""
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train import ROOT, FEATURES, fit, load_candidates, model_rows

CACHE = ROOT / "outputs" / "knn_sequence"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = load_candidates("train")
    scores = np.empty(len(rows), np.float64)
    for fold in sorted(rows.fold.unique()):
        valid = rows.fold == fold
        model, _ = fit(model_rows(rows[~valid]))
        scores[valid] = model.predict_proba(rows.loc[valid, FEATURES])[:, 1]
        print(f"Scored held-out fold {fold}", flush=True)
    keyed = rows[["trip_id", "point_idx", "arc_id"]].copy()
    keyed["knn"] = scores
    keyed.to_parquet(CACHE / "oof.parquet", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
