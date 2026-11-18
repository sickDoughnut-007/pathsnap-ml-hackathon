"""Predict a raw trip using the saved KNN and selected sequence settings."""
import argparse
import json
import sys
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from pathsnap.features import FEATURES, build_trip_tables
from pathsnap.io import load_config, load_trip_dir
from pathsnap.knn_sequence import Settings, TripLattice
from pathsnap.preprocess import prepare_trip


def predict(trip_dir):
    trip = load_trip_dir(trip_dir, with_route=False)
    config = load_config()
    prepared, order = prepare_trip(trip, config)
    candidates, points = build_trip_tables(trip, prepared, config, order)
    points["pred_arc"] = -1
    if len(candidates):
        model = joblib.load(HERE.parent / "model.joblib")
        scores = model.predict_proba(candidates[FEATURES])[:, 1]
        params = Settings(**json.loads((HERE / "settings.json").read_text()))
        lattice = TripLattice(candidates, scores, trip)
        points.loc[lattice.points, "pred_arc"] = lattice.decode(params)
    return points[["point_idx", "orig_row", "t", "segment", "pred_arc"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        predict(args.trip_dir).to_csv(args.output_csv, index=False)
