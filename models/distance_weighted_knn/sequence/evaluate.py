"""Select a sequence decoder on out-of-fold training predictions, then evaluate once on student test."""
import argparse
import json
from dataclasses import asdict
from time import perf_counter

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from oof import CACHE, HERE
from pathsnap.dataset import FEATURES, load_candidates, load_points
from pathsnap.evaluate import route_continuity
from pathsnap.io import load_config, load_trip
from pathsnap.knn_sequence import Settings, TripLattice


# Each tuple controls distance cost, transition mismatch, U-turn penalty, KNN weight.
TRIALS = [Settings(*args) for args in [
    (6, 4, 3, 1), (3, 3, 3, 1), (2, 2, 4, 1), (1, 2, 6, 1),
    (2, 1, 8, .4), (2, 3, 8, .4), (1, 1, 8, .2), (4, 2, 8, .2),
    (1, 2, 12, .05), (6, 2, 4, 2), (4, 6, 6, 1),
    (6, 8, 8, 1, 2), (8, 12, 10, 1, 4), (10, 15, 10, 1, 6),
    (4, 6, 6, 1, 0, True), (6, 8, 8, 1, 2, True),
    (8, 12, 10, 1, 4, True), (10, 15, 10, 1, 6, True),
    (5, 12, 10, .3, 4, True), (10, 12, 10, .3, 6, True),
    (10, 8, 12, 2, 4, True),
    (6, 8, 8, 1, 2, True, True), (4, 6, 6, 1, 0, True, True),
    (4, 8, 10, 1, 1, True, True, 60), (6, 8, 10, .7, 1, True, True, 60),
    (6, 8, 10, 1.2, 2, True, True, 60), (4, 10, 16, .7, 2, True, True, 100),
    (8, 8, 12, 1, 1, True, True, 100), (3, 8, 12, .4, 1, True, True, 100),
    (3, 8, 12, .4, 1, True, True, 100, True),
    (4, 10, 16, .7, 2, True, True, 100, True),
    (4, 8, 20, .3, 1, True, True, 180, True),
    (4, 8, 20, .3, 1, True, True, 180),
    (3, 8, 30, .15, .6, True, True, 180, True),
    (4, 12, 25, .3, 1, True, True, 180, True),
    (3, 12, 35, .2, 1, True, True, 180),
    (2, 8, 50, .2, 1, True, True, 250, True),
    (4, 15, 35, .4, 1.5, True, True, 250, True),
]]


def compare(part, configs, root):
    c, points = load_candidates(part), load_points(part)
    if part == "train":
        keyed = pd.read_parquet(CACHE / "oof.parquet")
        keys = ["trip_id", "point_idx", "arc_id"]
        if not c[keys].equals(keyed[keys]):
            raise ValueError("Rebuild out-of-fold predictions for this dataset")
        c["knn"] = keyed.knn.to_numpy()
    else:
        model = joblib.load(HERE.parent / "model.joblib")
        c["knn"] = model.predict_proba(c[FEATURES])[:, 1]
    cfg = load_config()
    results, student = [], []
    for tid, rows in c.groupby("trip_id", sort=True):
        trip = load_trip(cfg, part, tid, root, with_route=False)
        obs = points[points.trip_id == tid].set_index("point_idx")
        nearest = rows.loc[rows.groupby("point_idx").knn.idxmax()].set_index("point_idx").arc_id
        usable = obs.label_ok.eq(1)
        ref = obs.ref_arc
        start = perf_counter()
        lattice = TripLattice(rows, rows.knn.to_numpy(), trip)
        for trial, params in enumerate(configs):
            pred = pd.Series(lattice.decode(params), index=lattice.points).reindex(obs.index)
            record = dict(trip_id=tid, fold=int(obs.fold.iloc[0]), trial=trial,
                          eligible=int(usable.sum()), correct=int((pred.eq(ref) & usable).sum()),
                          baseline_correct=int((nearest.reindex(obs.index).eq(ref) & usable).sum()))
            if part == "student_test":
                record["decoder_seconds"] = perf_counter() - start
                pred_rows = obs[usable & pred.notna()][["trip_id", "segment"]].copy()
                pred_rows["pred_arc"] = pred.loc[pred_rows.index].astype(int)
                student.append((pred_rows.reset_index(), trip.arcs))
            results.append(record)
        if (len(results) // len(configs)) % 10 == 0:
            print(f"{part}: {len(results) // len(configs)} trips", flush=True)
    return pd.DataFrame(results), student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "test"])
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    if args.stage == "select":
        rows, _ = compare("train", TRIALS, args.data_root)
        rows.to_csv(HERE / "validation_trips.csv", index=False)
        fold = rows.groupby(["trial", "fold"])[["correct", "eligible", "baseline_correct"]].sum()
        fold["top1"] = fold.correct / fold.eligible
        fold["knn_top1"] = fold.baseline_correct / fold.eligible
        fold.to_csv(HERE / "validation_folds.csv")
        scores = fold.groupby("trial").top1.mean()
        trials = pd.DataFrame([asdict(cfg) for cfg in TRIALS])
        trials["mean_fold_top1"] = scores
        trials.to_csv(HERE / "trials.csv", index_label="trial")
        best = int(scores.idxmax())
        (HERE / "settings.json").write_text(json.dumps(asdict(TRIALS[best]), indent=2) + "\n")
        print(trials.to_string(), flush=True)
    else:
        settings = Settings(**json.loads((HERE / "settings.json").read_text()))
        rows, student = compare("student_test", [settings], args.data_root)
        rows.drop(columns=["trial"]).to_csv(HERE / "student_trips.csv", index=False)
        pred = pd.concat([p for p, _ in student], ignore_index=True)
        arcs = {p.trip_id.iloc[0]: arcs for p, arcs in student}
        report = dict(correct=int(rows.correct.sum()), eligible=int(rows.eligible.sum()),
                      top1=rows.correct.sum()/rows.eligible.sum(),
                      knn_top1=rows.baseline_correct.sum()/rows.eligible.sum(),
                      decoder_seconds=rows.decoder_seconds.sum())
        for hop in (1, 2):
            report[f"continuity_{hop}hop"] = route_continuity(pred, arcs, hop)[0]
        (HERE / "student_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
