"""Compare multi-scale route guidance against the fixed first decoder."""
import argparse
import json
from dataclasses import replace
from time import perf_counter

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from oof import CACHE, HERE
from pathsnap.dataset import FEATURES, load_candidates, load_points
from pathsnap.evaluate import route_continuity
from pathsnap.io import load_config, load_trip
from pathsnap.knn_anchors import anchor_corridor
from pathsnap.knn_sequence import Settings, TripLattice

OUT = HERE / "iteration2"
BASE = Settings(**json.loads((HERE / "settings.json").read_text()))


def experiment(part, configs, root):
    candidates, points = load_candidates(part), load_points(part)
    if part == "train":
        cache = pd.read_parquet(CACHE / "oof.parquet")
        keys = ["trip_id", "point_idx", "arc_id"]
        if not candidates[keys].equals(cache[keys]):
            raise ValueError("Rebuild out-of-fold predictions for this dataset")
        candidates["knn"] = cache.knn.to_numpy()
    else:
        model = joblib.load(HERE.parent / "model.joblib")
        candidates["knn"] = model.predict_proba(candidates[FEATURES])[:, 1]
    results, decoded, networks = [], [], {}
    config = load_config()
    for trip_number, (tid, rows) in enumerate(candidates.groupby("trip_id"), start=1):
        obs = points[points.trip_id == tid].set_index("point_idx")
        ok = obs.label_ok.eq(1)
        trip = load_trip(config, part, tid, root, with_route=False)
        start = perf_counter()
        full = TripLattice(rows, rows.knn.to_numpy(), trip)
        build_s = perf_counter() - start
        guides = {}
        for trial, (spacing, weight, turn) in enumerate(configs):
            cfg = replace(BASE, turn_cost=turn)
            key = spacing, turn
            decode_start = perf_counter()
            if spacing and key not in guides:
                guides[key] = anchor_corridor(rows, rows.knn.to_numpy(), trip, full, cfg, spacing)
            bonus = guides[key] * weight if spacing else None
            path = pd.Series(full.decode(cfg, bonus), index=full.points).reindex(obs.index)
            correct = int((ok & path.eq(obs.ref_arc)).sum())
            results.append(dict(trip_id=tid, fold=int(obs.fold.iloc[0]), trial=trial, spacing=spacing,
                                weight=weight, turn_cost=turn, correct=correct, eligible=int(ok.sum()),
                                decoder_seconds=build_s + perf_counter() - decode_start))
            if part == "student_test":
                selected = obs.loc[ok & path.notna(), ["trip_id", "segment"]].copy()
                selected["pred_arc"] = path.loc[selected.index].astype(int)
                decoded.append(selected.reset_index())
                networks[tid] = trip.arcs
        if trip_number % 10 == 0:
            print(f"{part}: {trip_number} trips", flush=True)
    return pd.DataFrame(results), decoded, networks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["select", "test"])
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage == "select":
        configs = [(0, 0, 35)] + [(spacing, weight, turn) for spacing in (15, 35, 70)
                                 for weight in (1, 4, 12) for turn in (35, 90)]
        rows, _, _ = experiment("train", configs, args.data_root)
        keys = ["trial", "spacing", "weight", "turn_cost"]
        folds = rows.groupby(keys + ["fold"])[["correct", "eligible"]].sum().reset_index()
        folds["top1"] = folds.correct / folds.eligible
        folds.to_csv(OUT / "folds.csv", index=False)
        trials = folds.groupby(keys).top1.mean().reset_index()
        trials.to_csv(OUT / "trials.csv", index=False)
        best = trials.loc[trials.top1.idxmax()]
        params = {k: int(best[k]) for k in ("spacing", "weight", "turn_cost")}
        (OUT / "settings.json").write_text(json.dumps(params, indent=2) + "\n")
        print(trials.to_string(index=False))
    else:
        cfg = json.loads((OUT / "settings.json").read_text())
        selected = (cfg["spacing"], cfg["weight"], cfg["turn_cost"])
        rows, predictions, networks = experiment("student_test", [selected], args.data_root)
        rows.drop(columns=["trial"]).to_csv(OUT / "student_trips.csv", index=False)
        result = dict(correct=int(rows.correct.sum()), eligible=int(rows.eligible.sum()),
                      top1=rows.correct.sum() / rows.eligible.sum(),
                      decoder_seconds=rows.decoder_seconds.sum())
        for hops in (1, 2):
            result[f"continuity_{hops}hop"] = route_continuity(pd.concat(predictions), networks, hops)[0]
        (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
