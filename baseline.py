"""Run the first PathSnap comparison on CPU and save reproducible results."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from map_matching import FEATURES, MAX_CANDIDATES, MAX_LABEL_DISTANCE_M, RADIUS_M, prepare


def trip_ids(archive, split):
    return sorted(Path(name).stem for name in archive.namelist()
                  if name.startswith(f"Student folder/{split}/") and name.endswith(".track"))


def fit(trips, cache):
    rows, labels = [], []
    for trip in trips:
        with np.load(cache / f"{trip}.npz") as d:
            obs, reference = d["obs"], d["reference"]
            correct = d["arc"] == reference[obs]
            covered = np.bincount(obs[correct], minlength=len(reference)) > 0
            keep = (covered & (d["error"] <= MAX_LABEL_DISTANCE_M))[obs]
            rows.append(d["x"][keep])
            labels.append(correct[keep])
    x, y = np.concatenate(rows), np.concatenate(labels)
    del rows, labels
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=42))
    model.fit(x, y)
    if model[-1].n_iter_.max() >= model[-1].max_iter:
        raise RuntimeError("Logistic regression did not converge")
    return model


def predictions(obs, arc, scores, size):
    nearest = np.full(size, -1, np.int32)
    ranked = nearest.copy()
    if len(obs):
        starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
        nearest[obs[starts]] = arc[starts]
        # Preserve nearest-road order for exact score ties.
        order = np.lexsort((np.arange(len(obs)), -scores, obs))
        ranked[obs[starts]] = arc[order[starts]]
    return nearest, ranked


def evaluate(model, trips, cache, split, output):
    trip_rows = []
    for trip in trips:
        with np.load(cache / f"{trip}.npz") as d:
            obs, arc, reference = d["obs"], d["arc"], d["reference"]
            features = d["x"]
            scores = model.decision_function(features) if len(obs) else np.empty(0)
            nearest, ranked = predictions(obs, arc, scores, len(reference))
            close = d["error"] <= MAX_LABEL_DISTANCE_M
            positive = arc == reference[obs]
            predicted = scores >= 0
            covered = np.unique(obs[positive])
            # logaddexp computes binary log loss directly from the decision score.
            loss_sum = np.logaddexp(0, np.where(positive, -scores, scores)).sum()
            trip_rows.append(dict(
                split=split, trip=trip, observations=len(reference), candidates=len(arc),
                no_candidates=int((nearest == -1).sum()), reference_in_candidates=len(covered),
                nearest_correct=int((nearest == reference).sum()),
                logistic_correct=int((ranked == reference).sum()), within_30m=int(close.sum()),
                logistic_correct_within_30m=int(((ranked == reference) & close).sum()),
                tp=int((positive & predicted).sum()), fp=int((~positive & predicted).sum()),
                tn=int((~positive & ~predicted).sum()), fn=int((positive & ~predicted).sum()),
                candidate_log_loss_sum=float(loss_sum)))
            if trip == "00000001" and split == "validation":
                np.savez(output / "example_predictions.npz", nearest=nearest, logistic=ranked,
                         reference=reference, alignment_distance_m=d["error"])
    counts = {key: sum(row[key] for row in trip_rows)
              for key in trip_rows[0] if key not in ("trip", "split")}
    n = counts["observations"]
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    counts.update(trips=len(trips), nearest_top1=counts["nearest_correct"] / n,
                  logistic_top1=counts["logistic_correct"] / n,
                  candidate_miss_rate=1 - counts["reference_in_candidates"] / n,
                  mean_trip_logistic_top1=float(np.mean([r["logistic_correct"] / r["observations"]
                                                        for r in trip_rows])),
                  candidate_macro_f1=0.5 * (2*tp / max(2*tp+fp+fn, 1) + 2*tn / max(2*tn+fp+fn, 1)),
                  candidate_log_loss=counts["candidate_log_loss_sum"] / max(counts["candidates"], 1))
    return counts, trip_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with args.archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    result = {"archive_sha256": digest, "features": FEATURES, "seed": 42, "seconds": {},
              "candidate_radius_m": RADIUS_M, "max_candidates": MAX_CANDIDATES,
              "max_fit_label_distance_m": MAX_LABEL_DISTANCE_M}
    with (ZipFile(args.archive) as archive, tempfile.TemporaryDirectory(dir=args.output) as temp,
          threadpool_limits(limits=2)):
        cache = Path(temp)
        train, student = (trip_ids(archive, s) for s in ("train", "student_test"))
        if len(train) != 80 or len(student) != 10 or set(train) & set(student):
            raise ValueError("Expected 80 training and 10 distinct student-testing trips")
        example = "00000001"
        rng = np.random.default_rng(42)
        validation = sorted([example, *rng.choice([t for t in train if t != example], 15, replace=False)])
        fitting = sorted(set(train) - set(validation))
        result["trip_ids"] = dict(fitting=fitting, validation=validation, student_test=student)
        all_rows = []
        for split, trips in [("train", train), ("student_test", student)]:
            clock = perf_counter()
            for i, trip in enumerate(trips, 1):
                np.savez(cache / f"{trip}.npz", **prepare(archive, split, trip))
                if i % 10 == 0:
                    print(f"Prepared {split} {i}/{len(trips)}", flush=True)
            result["seconds"][f"prepare_{split}"] = perf_counter() - clock
            training = fitting if split == "train" else train
            testing = validation if split == "train" else student
            stage = "validation" if split == "train" else "student_test"
            clock = perf_counter()
            model = fit(training, cache)
            result["seconds"][f"fit_{len(training)}_trips"] = perf_counter() - clock
            clock = perf_counter()
            result[stage], rows = evaluate(model, testing, cache, stage, args.output)
            result["seconds"][f"evaluate_{stage}"] = perf_counter() - clock
            all_rows.extend(rows)
            print(stage, result[stage], flush=True)
        joblib.dump(dict(model=model, features=FEATURES), args.output / "logistic.joblib")
    with (args.output / "trip_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0])
        writer.writeheader()
        writer.writerows(all_rows)
    result["seconds"]["total"] = perf_counter() - start
    mem = psutil.Process().memory_info()
    if hasattr(mem, "peak_wset"):
        result["peak_process_working_set_mib"] = mem.peak_wset / 2**20
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
