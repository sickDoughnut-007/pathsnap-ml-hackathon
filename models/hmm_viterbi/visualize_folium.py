import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import folium

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pathsnap.io import load_config, load_trip
from pathsnap.geo import LocalProjection


PREDICTIONS_FILE = (
    REPO_ROOT / "models" / "hmm_viterbi" / "results"
    / "student_test_predictions.csv"
)
OUTPUT_DIR = REPO_ROOT / "models" / "hmm_viterbi" / "results"

COLORS = {
    "reference": "#16a34a",
    "classifier": "#f59e0b",
    "hmm": "#dc2626",
    "gps": "#2563eb",
    "network": "#9ca3af",
}


def unique_arc_ids(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return list(dict.fromkeys(values.astype(int).tolist()))


def add_arc_layer(map_obj, nodes, arcs, arc_ids, name, color, weight):
    layer = folium.FeatureGroup(name=name, show=True)
    added = 0

    for arc_id in arc_ids:
        if arc_id < 0 or arc_id >= len(arcs):
            continue

        u, v = arcs[arc_id]
        if u < 0 or v < 0 or u >= len(nodes) or v >= len(nodes):
            continue

        start = [float(nodes[u, 1]), float(nodes[u, 0])]
        end = [float(nodes[v, 1]), float(nodes[v, 0])]

        folium.PolyLine(
            locations=[start, end],
            color=color,
            weight=weight,
            opacity=0.9,
            tooltip=f"{name} | Arc ID: {arc_id}",
        ).add_to(layer)

        added += 1

    layer.add_to(map_obj)
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-id", default="00000000")
    args = parser.parse_args()

    trip_id = str(args.trip_id).zfill(8)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions["trip_id"] = pd.to_numeric(
        predictions["trip_id"], errors="coerce"
    )

    trip_predictions = predictions[
        predictions["trip_id"] == int(trip_id)
    ].copy()

    if trip_predictions.empty:
        raise ValueError(f"No predictions found for trip {trip_id}")

    if trip_predictions["point_idx"].duplicated().any():
        raise ValueError(f"Duplicate point_idx values for trip {trip_id}")

    cfg = load_config()
    trip = load_trip(
        cfg,
        partition="student_test",
        trip_id=trip_id,
        with_route=False,
    )

    # Match the chronological ordering used by preprocessing.
    order = np.argsort(trip.track[:, 2], kind="stable")
    track = trip.track[order]

    lon = track[:, 0]
    lat = track[:, 1]
    timestamps = track[:, 2]

    if trip_predictions["point_idx"].max() >= len(track):
        raise ValueError(
            "Prediction point_idx exceeds the number of GPS observations."
        )

    proj = LocalProjection(
        lon.mean(),
        lat.mean(),
        cfg["projection"]["earth_radius_m"],
    )

    gps_x, gps_y = proj(lon, lat)
    node_x, node_y = proj(trip.nodes[:, 0], trip.nodes[:, 1])

    margin_m = 200.0
    xmin, xmax = gps_x.min() - margin_m, gps_x.max() + margin_m
    ymin, ymax = gps_y.min() - margin_m, gps_y.max() + margin_m

    u = trip.arcs[:, 0]
    v = trip.arcs[:, 1]

    ax, bx = node_x[u], node_x[v]
    ay, by = node_y[u], node_y[v]

    context_mask = (
        (np.maximum(ax, bx) >= xmin)
        & (np.minimum(ax, bx) <= xmax)
        & (np.maximum(ay, by) >= ymin)
        & (np.minimum(ay, by) <= ymax)
    )

    context_arc_ids = np.flatnonzero(context_mask).tolist()

    m = folium.Map(
        location=[float(lat.mean()), float(lon.mean())],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    add_arc_layer(
        m, trip.nodes, trip.arcs, context_arc_ids,
        "Nearby road network", COLORS["network"], 2
    )

    gps_layer = folium.FeatureGroup(
        name="Original GPS observations", show=True
    )

    for idx, (latitude, longitude, timestamp) in enumerate(
        zip(lat, lon, timestamps)
    ):
        folium.CircleMarker(
            location=[float(latitude), float(longitude)],
            radius=2.5,
            color=COLORS["gps"],
            fill=True,
            fill_color=COLORS["gps"],
            fill_opacity=0.85,
            weight=1,
            tooltip=f"GPS point: {idx} | Timestamp: {timestamp:.2f}",
        ).add_to(gps_layer)

    gps_layer.add_to(m)

    reference_rows = trip_predictions[
        trip_predictions["label_ok"] == 1
    ]

    reference_count = add_arc_layer(
        m, trip.nodes, trip.arcs,
        unique_arc_ids(reference_rows, "ref_arc"),
        "Supplied reference route", COLORS["reference"], 5
    )

    classifier_count = add_arc_layer(
        m, trip.nodes, trip.arcs,
        unique_arc_ids(trip_predictions, "classifier_pred_arc"),
        "LightGBM classifier predictions", COLORS["classifier"], 4
    )

    hmm_count = add_arc_layer(
        m, trip.nodes, trip.arcs,
        unique_arc_ids(trip_predictions, "hmm_pred_arc"),
        "LightGBM + HMM predictions", COLORS["hmm"], 6
    )

    folium.Marker(
        [float(lat[0]), float(lon[0])],
        tooltip="Trip start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        [float(lat[-1]), float(lon[-1])],
        tooltip="Trip end",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    m.fit_bounds([
        [float(lat.min()), float(lon.min())],
        [float(lat.max()), float(lon.max())],
    ])

    folium.LayerControl(collapsed=False).add_to(m)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"trip_{trip_id}_folium.html"
    m.save(str(output_file))

    print("\nFOLIUM MAP GENERATED")
    print("--------------------")
    print(f"Trip ID:                 {trip_id}")
    print(f"GPS observations:        {len(track)}")
    print(f"Nearby road arcs:        {len(context_arc_ids)}")
    print(f"Reference route arcs:    {reference_count}")
    print(f"Classifier route arcs:   {classifier_count}")
    print(f"HMM route arcs:          {hmm_count}")
    print(f"Output:                  {output_file.resolve()}")


if __name__ == "__main__":
    main()
