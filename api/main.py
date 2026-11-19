import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap.io import load_config, load_trip_dir
from pathsnap.preprocess import prepare_trip
from pathsnap.features import FEATURES, build_trip_tables
from pathsnap.labels import align_to_route
from models.hmm_viterbi.decoder import (
    classifier_predictions,
    decode_trip,
)

MODEL_PATH = ROOT / "models" / "lightgbm" / "tuning" / "model.joblib"
HMM_CONFIG_PATH = ROOT / "models" / "hmm_viterbi" / "results" / "best_decoder_config.json"

EVALUATOR_ROOT = Path(os.environ.get(
    "PATHSNAP_EVALUATOR_ROOT",
    r"D:\Akshya\MLhackathon\evaluator_test\evaluator_test"
))

app = FastAPI(title="PathSnap Route Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config()
model = None
hmm_params = None


@app.on_event("startup")
def load_artifacts():
    global model, hmm_params

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model: {MODEL_PATH}")

    if not HMM_CONFIG_PATH.exists():
        raise RuntimeError(f"Missing HMM config: {HMM_CONFIG_PATH}")

    model = joblib.load(MODEL_PATH)

    import json
    with open(HMM_CONFIG_PATH, encoding="utf-8") as f:
        hmm_params = json.load(f)


def evaluator_trip_dir(trip_id):
    if not re.fullmatch(r"\d{8}", trip_id):
        raise HTTPException(status_code=400, detail="Trip ID must contain 8 digits.")

    trip_dir = EVALUATOR_ROOT / trip_id

    if not trip_dir.is_dir():
        raise HTTPException(status_code=404, detail="Evaluator trip not found.")

    required = [
        trip_dir / f"{trip_id}.track",
        trip_dir / f"{trip_id}.nodes",
        trip_dir / f"{trip_id}.arcs",
    ]

    if not all(p.is_file() for p in required):
        raise HTTPException(status_code=500, detail="Trip input files are incomplete.")

    return trip_dir


def make_paths(predictions, points, arcs, nodes_lonlat):
    paths = []
    current = []
    previous_idx = None
    previous_segment = None
    previous_arc = None

    for row in points.sort_values("point_idx").itertuples(index=False):
        idx = int(row.point_idx)
        segment = int(row.segment)
        arc_id = predictions.get(idx)

        if arc_id is None:
            if current:
                paths.append(current)
                current = []
            previous_idx = None
            previous_arc = None
            previous_segment = None
            continue

        start_node, end_node = arcs[int(arc_id)]
        start = [float(nodes_lonlat[start_node, 1]), float(nodes_lonlat[start_node, 0])]
        end = [float(nodes_lonlat[end_node, 1]), float(nodes_lonlat[end_node, 0])]

        continuous = (
            current
            and previous_idx is not None
            and idx == previous_idx + 1
            and segment == previous_segment
            and (
                int(arc_id) == previous_arc
                or int(arcs[previous_arc, 1]) == int(start_node)
            )
        )

        if not continuous:
            if current:
                paths.append(current)
            current = [start, end]
        elif current[-1] != end:
            current.append(end)

        previous_idx = idx
        previous_segment = segment
        previous_arc = int(arc_id)

    if current:
        paths.append(current)

    return paths


def calculate_server_metrics(points, candidates, predictions, arcs, pt, trip):
    if trip.route is None or len(trip.route) == 0:
        return {"evaluation_available": False}

    ref_arc, _, ref_dist = align_to_route(
        pt,
        trip.route,
        cfg["labels"]["alignment_cap_m"],
        cfg["labels"]["alignment_heading_weight_m"],
    )

    label_ok = ref_dist <= cfg["labels"]["max_label_dist_m"]
    labelled_indices = np.flatnonzero(label_ok)

    if len(labelled_indices):
        correct = sum(
            predictions.get(int(i), -1) == int(ref_arc[i])
            for i in labelled_indices
        )
        top1 = float(correct / len(labelled_indices))
    else:
        top1 = None

    candidate_sets = {
        int(point_idx): set(group["arc_id"].astype(int))
        for point_idx, group in candidates.groupby("point_idx")
    }

    candidate_misses = sum(
        int(ref_arc[i]) not in candidate_sets.get(int(i), set())
        for i in labelled_indices
    )

    transition_total = 0
    transition_connected = 0

    ordered = points.sort_values("point_idx")
    rows = list(ordered.itertuples(index=False))

    for prev, curr in zip(rows[:-1], rows[1:]):
        if (
            int(curr.point_idx) != int(prev.point_idx) + 1
            or int(curr.segment) != int(prev.segment)
        ):
            continue

        a = predictions.get(int(prev.point_idx))
        b = predictions.get(int(curr.point_idx))

        if a is None or b is None:
            continue

        transition_total += 1
        if a == b or arcs[a, 1] == arcs[b, 0]:
            transition_connected += 1

    return {
        "evaluation_available": True,
        "labelled_observations": int(len(labelled_indices)),
        "top1_accuracy": top1,
        "candidate_miss_rate": (
            float(candidate_misses / len(labelled_indices))
            if len(labelled_indices) else None
        ),
        "route_continuity_1hop": (
            float(transition_connected / transition_total)
            if transition_total else None
        ),
        "evaluated_transitions": int(transition_total),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "hmm_config_loaded": hmm_params is not None,
        "evaluator_data_available": EVALUATOR_ROOT.is_dir(),
    }


@app.get("/api/trips")
def list_evaluator_trips():
    if not EVALUATOR_ROOT.is_dir():
        raise HTTPException(status_code=500, detail="Evaluator dataset directory not found.")

    trips = sorted(
        p.name for p in EVALUATOR_ROOT.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{8}", p.name)
    )

    return {"count": len(trips), "trip_ids": trips}


@app.post("/api/predict/{trip_id}")
def predict_trip(trip_id: str):
    trip_dir = evaluator_trip_dir(trip_id)

    start_time = time.perf_counter()

    # Prediction input deliberately excludes the reference .route file.
    trip = load_trip_dir(trip_dir, with_route=False)
    pt, order = prepare_trip(trip, cfg)
    candidates, points = build_trip_tables(trip, pt, cfg, order=order)

    if candidates.empty:
        raise HTTPException(status_code=422, detail="No road candidates generated for this trip.")

    missing = sorted(set(FEATURES) - set(candidates.columns))
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing model features: {missing}")

    probabilities = model.predict_proba(candidates[FEATURES])[:, 1]

    classifier = classifier_predictions(candidates, probabilities)

    hmm = decode_trip(
        points,
        candidates,
        trip.arcs,
        probabilities,
        hmm_params,
    )

    classifier_paths = make_paths(classifier, points, trip.arcs, trip.nodes)
    hmm_paths = make_paths(hmm, points, trip.arcs, trip.nodes)

    gps_points = [
        {
            "point_idx": int(row.point_idx),
            "latitude": float(row.lat),
            "longitude": float(row.lon),
            "segment": int(row.segment),
        }
        for row in points.itertuples(index=False)
    ]

    prediction_points = []
    for row in points.itertuples(index=False):
        idx = int(row.point_idx)
        prediction_points.append({
            "point_idx": idx,
            "classifier_arc": classifier.get(idx),
            "hmm_arc": hmm.get(idx),
        })

    # Load reference labels only after inference, strictly for server-side scoring.
    evaluation_trip = load_trip_dir(trip_dir, with_route=True)
    metrics = calculate_server_metrics(
        points, candidates, hmm, trip.arcs, pt, evaluation_trip
    )

    return {
        "trip_id": trip_id,
        "observation_count": len(gps_points),
        "candidate_count": int(len(candidates)),
        "inference_seconds": float(time.perf_counter() - start_time),
        "gps_points": gps_points,
        "prediction_points": prediction_points,
        "classifier_paths": classifier_paths,
        "hmm_paths": hmm_paths,
        "metrics": metrics,
    }
