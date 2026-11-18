"""Exploratory data analysis of the raw PathSnap dataset (train + student_test partitions).

Outputs (results/eda/):
  inventory.csv            per-trip file sizes, row counts, sampling, extent, network size
  missing_values.csv       NaN / invalid-value / ordering checks per file type, per partition
  network_summary.csv      connectivity: components, isolated nodes, duplicates, self-loops
  route_summary.csv        reference-route length, continuity, coverage of the GPS track
  plots/*.png              distributions and trajectory plots
  maps/*.html              interactive Folium maps (raw GPS, road network, reference route)

Usage (from repo root):  python scripts/01_eda.py [--data-root PATH]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.geo import LocalProjection, haversine  # noqa: E402
from pathsnap.io import REPO_ROOT, data_root, list_trips, load_config, load_trip  # noqa: E402
from pathsnap.network import build_network, connectivity_summary  # noqa: E402

OUT = REPO_ROOT / "results" / "eda"


def file_checks(trip):
    tr, nd, ar = trip.track, trip.nodes, trip.arcs
    dt = np.diff(tr[:, 2])
    return {
        "track_nan": int(np.isnan(tr).sum()),
        "track_invalid_coord": int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum()),
        "track_dup_timestamps": int((dt == 0).sum()),
        "track_non_monotonic": int((dt < 0).sum()),
        "track_dup_positions": int(((np.diff(tr[:, 0]) == 0) & (np.diff(tr[:, 1]) == 0)).sum()),
        "nodes_nan": int(np.isnan(nd).sum()),
        "nodes_invalid_coord": int(((np.abs(nd[:, 0]) > 180) | (np.abs(nd[:, 1]) > 90)).sum()),
        "arcs_bad_node_ref": int(((ar < 0) | (ar >= len(nd))).sum()),
        "route_bad_arc_ref": int(((trip.route < 0) | (trip.route >= len(ar))).sum()) if trip.route is not None else 0,
    }


def folium_map(trip, net, path, title):
    import folium
    tr = trip.track
    lon_c, lat_c = tr[:, 0].mean(), tr[:, 1].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=14, tiles="OpenStreetMap")
    # Road network: arcs inside the track bounding box (+ margin), drawn once per node pair.
    x0, x1 = tr[:, 0].min() - 0.003, tr[:, 0].max() + 0.003
    y0, y1 = tr[:, 1].min() - 0.003, tr[:, 1].max() + 0.003
    nd, ar = trip.nodes, trip.arcs
    a, b = nd[ar[:, 0]], nd[ar[:, 1]]
    inside = ((a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1))
    key = np.sort(ar, axis=1)
    _, uniq = np.unique(key, axis=0, return_index=True)
    sel = np.intersect1d(np.flatnonzero(inside), uniq)
    if len(sel) > 4000:
        sel = sel[np.random.default_rng(0).choice(len(sel), 4000, replace=False)]
    g_net = folium.FeatureGroup(name=f"Road network ({len(sel)} segments in view)")
    for i in sel:
        folium.PolyLine([[a[i, 1], a[i, 0]], [b[i, 1], b[i, 0]]], color="#9e9e9e", weight=1, opacity=0.6).add_to(g_net)
    g_net.add_to(m)
    if trip.route is not None:
        g_r = folium.FeatureGroup(name="Reference route (.route)")
        for arc in trip.route:
            u, v = ar[arc]
            folium.PolyLine([[nd[u, 1], nd[u, 0]], [nd[v, 1], nd[v, 0]]], color="#2e7d32", weight=5, opacity=0.8).add_to(g_r)
        g_r.add_to(m)
    g_gps = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(tr) // 2000)
    for lon, lat, t in tr[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(g_gps)
    g_gps.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    legend = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;'
              f'border:1px solid #999;font-size:13px"><b>{title}</b><br>'
              '<span style="color:#1565c0">&#9679;</span> Raw GPS<br>'
              '<span style="color:#2e7d32">&#9644;</span> Reference route<br>'
              '<span style="color:#9e9e9e">&#9644;</span> Road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--maps", nargs="*", default=["train/00000005", "train/00000054", "student_test/00000052"])
    args = ap.parse_args()
    cfg = load_config()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    root = data_root(cfg, args.data_root)

    inv, miss, netrows, routes = [], [], [], []
    all_dt = {"train": [], "student_test": []}
    trips_cache = {}
    for part in ["train", "student_test"]:
        for tid in list_trips(cfg, part, args.data_root):
            trip = load_trip(cfg, part, tid, args.data_root)
            d = root / cfg["data"][{"train": "train_dir", "student_test": "student_test_dir"}[part]] / tid
            tr = trip.track
            dt = np.diff(tr[:, 2])
            all_dt[part].append(dt)
            proj = LocalProjection(tr[:, 0].mean(), tr[:, 1].mean())
            net = build_network(trip.nodes, trip.arcs, proj)
            step_m = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
            inv.append({
                "partition": part, "trip_id": tid,
                "files_present": sum((d / f"{tid}.{e}").exists() for e in ["track", "nodes", "arcs", "route"]),
                "track_kb": round((d / f"{tid}.track").stat().st_size / 1024, 1),
                "nodes_kb": round((d / f"{tid}.nodes").stat().st_size / 1024, 1),
                "arcs_kb": round((d / f"{tid}.arcs").stat().st_size / 1024, 1),
                "n_gps": len(tr), "duration_s": float(tr[-1, 2] - tr[0, 2]),
                "track_length_km": round(step_m.sum() / 1000, 2),
                "frac_dt_1s": round(float((dt == 1).mean()), 4), "median_dt_s": float(np.median(dt)),
                "max_gap_s": float(dt.max()), "n_gaps_gt_30s": int((dt > cfg["preprocessing"]["gap_split_s"]).sum()),
                "max_step_m": round(float(step_m.max()), 1),
                "lon_center": round(tr[:, 0].mean(), 4), "lat_center": round(tr[:, 1].mean(), 4),
                "extent_ew_km": round(float(np.ptp(proj(tr[:, 0], tr[:, 1])[0]) / 1000), 2),
                "extent_ns_km": round(float(np.ptp(proj(tr[:, 0], tr[:, 1])[1]) / 1000), 2),
                "projection_max_scale_error": round(proj.max_scale_error(tr[:, 1]), 5),
                "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
                "route_len": len(trip.route) if trip.route is not None else 0,
            })
            miss.append({"partition": part, "trip_id": tid, **file_checks(trip)})
            netrows.append({"partition": part, "trip_id": tid, **connectivity_summary(net)})
            if trip.route is not None:
                r = trip.route
                conn = trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0]
                routes.append({
                    "partition": part, "trip_id": tid, "route_arcs": len(r),
                    "unique_arcs": len(np.unique(r)),
                    "route_length_km": round(float(net.length[r].sum() / 1000), 2),
                    "track_length_km": inv[-1]["track_length_km"],
                    "route_to_track_length_ratio": round(float(net.length[r].sum() / 1000 / max(inv[-1]["track_length_km"], 1e-9)), 3),
                    "consecutive_connected_frac": round(float(conn.mean()) if len(conn) else 1.0, 4),
                    "n_breaks": int((~conn).sum()),
                    "uses_both_directions_of_a_street": int(len(set(map(tuple, trip.arcs[r]))
                                                            & set(map(tuple, trip.arcs[r][:, ::-1]))) > 0),
                })
            if f"{part}/{tid}" in args.maps:
                trips_cache[f"{part}/{tid}"] = (trip, net)
            print(part, tid, "ok", flush=True)

    inv = pd.DataFrame(inv)
    miss = pd.DataFrame(miss)
    netdf = pd.DataFrame(netrows)
    rdf = pd.DataFrame(routes)
    inv.to_csv(OUT / "inventory.csv", index=False)
    netdf.to_csv(OUT / "network_summary.csv", index=False)
    rdf.to_csv(OUT / "route_summary.csv", index=False)
    msum = miss.drop(columns="trip_id").groupby("partition").sum()
    msum.T.to_csv(OUT / "missing_values.csv")
    miss.to_csv(OUT / "missing_values_per_trip.csv", index=False)

    # ---------------- plots ----------------
    def save(name):
        plt.tight_layout()
        plt.savefig(OUT / "plots" / name, dpi=130)
        plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for part, c in [("train", "#1565c0"), ("student_test", "#ef6c00")]:
        s = inv[inv.partition == part]
        ax[0].hist(s.n_gps, bins=30, alpha=0.7, label=part, color=c)
        ax[1].hist(np.log10(s.n_arcs), bins=30, alpha=0.7, label=part, color=c)
    ax[0].set(xlabel="GPS observations per trip", ylabel="trips", title="Trajectory size")
    ax[1].set(xlabel="log10(road segments per trip)", title="Road-network size")
    ax[0].legend()
    save("trip_sizes.png")

    dt_all = np.concatenate(all_dt["train"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnts = np.unique(np.clip(dt_all, 0, 11), return_counts=True)
    ax[0].bar([str(int(v)) if v < 11 else ">10" for v in vals], cnts / cnts.sum(), color="#1565c0")
    ax[0].set(xlabel="time step between consecutive GPS points (s)", ylabel="fraction of steps",
              title="Sampling interval (train)")
    ax[1].bar(range(len(inv)), np.sort(inv.max_gap_s.values)[::-1], color="#c62828")
    ax[1].set_yscale("log")
    ax[1].axhline(cfg["preprocessing"]["gap_split_s"], color="k", ls="--", label="gap split threshold")
    ax[1].set(xlabel="trips (sorted)", ylabel="largest time gap (s, log)", title="Largest GPS gap per trip")
    ax[1].legend()
    save("sampling_and_gaps.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for part, c in [("train", "#1565c0"), ("student_test", "#ef6c00")]:
        s = inv[inv.partition == part]
        ax.scatter(s.lon_center, s.lat_center, s=18, c=c, label=part)
    ax.set(xlabel="longitude", ylabel="latitude", title="Trip locations (each trip has its own local road network)",
           xlim=(-180, 180), ylim=(-60, 75))
    ax.grid(alpha=0.3)
    ax.legend()
    save("trip_locations.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(netdf.n_nodes, netdf.weak_components, s=14, c="#6a1b9a")
    ax.set(xscale="log", yscale="log", xlabel="nodes", ylabel="weakly connected components",
           title="Road-network connectivity")
    save("network_connectivity.png")

    for key, (trip, net) in trips_cache.items():
        tr = trip.track
        fig, ax = plt.subplots(figsize=(6, 6))
        proj = LocalProjection(tr[:, 0].mean(), tr[:, 1].mean())
        x, y = proj(tr[:, 0], tr[:, 1])
        nx_, ny_ = proj(trip.nodes[:, 0], trip.nodes[:, 1])
        m = 300
        box = (nx_ > x.min() - m) & (nx_ < x.max() + m) & (ny_ > y.min() - m) & (ny_ < y.max() + m)
        sel = np.flatnonzero(box[trip.arcs[:, 0]] & box[trip.arcs[:, 1]])
        for i in sel[:30000]:
            u, v = trip.arcs[i]
            ax.plot([nx_[u], nx_[v]], [ny_[u], ny_[v]], color="#bdbdbd", lw=0.5)
        if trip.route is not None:
            for i in trip.route:
                u, v = trip.arcs[i]
                ax.plot([nx_[u], nx_[v]], [ny_[u], ny_[v]], color="#2e7d32", lw=2.5)
        ax.scatter(x, y, s=2, c="#1565c0", zorder=3)
        ax.set_aspect("equal")
        ax.set(title=f"{key}: GPS (blue), route (green), network (grey)", xlabel="x (m)", ylabel="y (m)")
        save(f"trajectory_{key.replace('/', '_')}.png")
        folium_map(trip, net, OUT / "maps" / f"{key.replace('/', '_')}.html", key)

    print("EDA outputs written to", OUT)


if __name__ == "__main__":
    main()
