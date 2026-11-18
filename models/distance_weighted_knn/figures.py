"""Render the measured fold results and a student-test prediction map."""
import argparse
import json
import sys
from pathlib import Path

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pathsnap.dataset import FEATURES, load_points
from pathsnap.features import build_trip_tables
from pathsnap.io import load_config, load_trip
from pathsnap.preprocess import prepare_trip


def diagnostics():
    cv = pd.read_csv(HERE / "cv_folds.csv")
    test = json.loads((HERE / "metrics.json").read_text())["student_test"]
    fig, (ax, cm) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for key, label, color in [("top1_accuracy", "Observation top-1", "#156981"),
                              ("f1_macro", "Candidate macro F1", "#984723")]:
        ax.plot(cv.fold + 1, cv[key], "o-", label=label, color=color)
        ax.scatter(6, test[key], marker="D", color=color)
    ax.set(xticks=range(1, 7), xticklabels=["1", "2", "3", "4", "5", "Test"],
           ylabel="Score", xlabel="Validation fold or student test", title="Trip-level evaluation")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    matrix = np.array(test["confusion_matrix"])
    cm.imshow(matrix / matrix.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        cm.text(j, i, f"{count:,}", ha="center", va="center",
                color="white" if count / matrix[i].sum() > .6 else "black")
    cm.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Other", "Match"], yticklabels=["Other", "Match"],
           xlabel="Predicted candidate class", ylabel="Estimated reference class", title="Student-test candidate counts")
    fig.savefig(HERE / "evaluation.png", dpi=160)
    plt.close(fig)


def example_map(root):
    tid = "00000052"
    cfg = load_config()
    trip = load_trip(cfg, "student_test", tid, root, with_route=False)
    pt, order = prepare_trip(trip, cfg)
    candidates, _ = build_trip_tables(trip, pt, cfg, order)
    model = joblib.load(HERE / "model.joblib")
    candidates["score"] = model.predict_proba(candidates[FEATURES])[:, 1]
    best = candidates.loc[candidates.groupby("point_idx").score.idxmax()].set_index("point_idx")
    # Only the comparison overlay reads reference data, after scoring the raw trip.
    points = load_points("student_test").query("trip_id == @tid").set_index("point_idx")
    reference = load_trip(cfg, "student_test", tid, root).route
    m = folium.Map(location=[points.lat.mean(), points.lon.mean()], prefer_canvas=True)
    for title, ids, color, weight in [
        ("Candidate roads · grey", candidates.arc_id, "#7d898d", 2),
        ("Reference route · green", reference, "#27844f", 7),
        ("KNN predictions · purple", best.arc_id, "#6c40ae", 3),
    ]:
        layer = folium.FeatureGroup(name=title).add_to(m)
        lines = trip.nodes[trip.arcs[np.unique(ids)]][:, :, ::-1]
        folium.PolyLine(lines.tolist(), color=color, weight=weight, opacity=.85).add_to(layer)
    layer = folium.FeatureGroup(name="GPS points · blue (match), red (miss), grey (unscored)").add_to(m)
    for idx, row in points.iterrows():
        arc = int(best.at[idx, "arc_id"]) if idx in best.index else -1
        color = "#777" if not row.label_ok else ("#176487" if arc == row.ref_arc else "#c64127")
        folium.CircleMarker([row.lat, row.lon], radius=3, fill=True, color=color,
                            tooltip=f"Point {idx} · KNN arc {arc} · reference {int(row.ref_arc)}").add_to(layer)
    m.fit_bounds([[points.lat.min(), points.lon.min()], [points.lat.max(), points.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    html = m.get_root().render()
    (HERE / "trip_00000052.html").write_text(
        "\n".join(line.rstrip() for line in html.splitlines()).strip() + "\n", encoding="utf-8")
    correct = int((best.arc_id.reindex(points.index).eq(points.ref_arc) & points.label_ok.eq(1)).sum())
    saved = pd.read_csv(HERE / "student_trips.csv", dtype={"trip_id": str}).set_index("trip_id")
    assert correct == saved.at[tid, "correct"], "Raw trip and Parquet predictions disagree on correct count"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    args = parser.parse_args()
    diagnostics()
    example_map(args.data_root)
