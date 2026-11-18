"""Feature construction for (GPS observation, candidate road segment) pairs.

All features are location-independent (metres, seconds, degrees, counts) because every
trip has its own road network and arc ids carry no meaning across trips.
Features never use the reference route; labels are added as a separate column.
"""
import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .geo import angle_diff, point_segment
from .labels import align_to_route

FEATURES = [
    # geometry of the candidate relative to the observation
    "dist", "dist_rel", "dist_ratio", "rank", "n_cand", "dist_min",
    "proj_t", "at_endpoint", "arc_len", "log_arc_len",
    # direction agreement
    "heading_ok", "heading_diff", "heading_diff_undirected", "cos_heading",
    # kinematics of the observation
    "speed", "turn", "dt_prev", "dt_next", "step_prev",
    # road topology
    "out_deg_to", "in_deg_from", "has_reverse",
    # temporal context: neighbouring observations (same segment only)
    "ctx_dist_prev1", "ctx_dist_next1", "ctx_dist_prev2", "ctx_dist_next2", "ctx_dist_mean",
    "ctx_cand_prev", "ctx_cand_next", "link_prev_nearest", "link_next_nearest",
    # along-arc motion: signed metres moved along the arc direction, projection overshoot
    "overshoot_m", "progress_prev", "progress_next", "progress_2s",
    # GPS jitter indicators (heading is meaningless while stopped)
    "straightness", "speed_window",
]


def _raw_t(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * vx + (py - ay) * vy) / L2
    return np.where(L2 > 0, t, 0.0)


def _neighbour(pt, offset):
    n = len(pt.x)
    j = np.arange(n) + offset
    valid = (j >= 0) & (j < n)
    jc = np.clip(j, 0, n - 1)
    valid &= pt.segment[jc] == pt.segment
    return jc, valid


def build_trip_tables(trip, pt, cfg, order=None):
    """Return (candidates_df, points_df) for one prepared trip."""
    net = pt.net
    p, a, d, t = generate_candidates(pt, cfg)
    n = len(pt.x)

    n_cand = np.bincount(p, minlength=n)
    dmin = np.full(n, np.nan)
    if len(p):
        first = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
        dmin[p[first]] = d[first]
        nearest = np.full(n, -1, dtype=np.int64)
        nearest[p[first]] = a[first]
        rank = np.arange(len(p)) - np.repeat(first, np.diff(np.r_[first, len(p)]))
    else:
        nearest = np.full(n, -1, dtype=np.int64)
        rank = np.array([], dtype=np.int64)

    df = pd.DataFrame({"point_idx": p, "arc_id": a})
    df["dist"] = d
    df["dist_min"] = dmin[p]
    df["dist_rel"] = d - dmin[p]
    df["dist_ratio"] = d / (dmin[p] + 1.0)
    df["rank"] = rank
    df["n_cand"] = n_cand[p]
    df["proj_t"] = t
    df["at_endpoint"] = ((t <= 1e-9) | (t >= 1 - 1e-9)).astype(np.int8)
    df["arc_len"] = net.length[a]
    df["log_arc_len"] = np.log1p(net.length[a])

    hok = pt.heading_ok[p]
    hd = angle_diff(pt.heading[p], net.bearing[a])
    df["heading_ok"] = hok.astype(np.int8)
    df["heading_diff"] = np.where(hok, hd, 90.0)
    df["heading_diff_undirected"] = np.where(hok, np.minimum(hd, 180.0 - hd), 45.0)
    df["cos_heading"] = np.where(hok, np.cos(np.radians(hd)), 0.0)

    df["speed"] = pt.speed[p]
    df["turn"] = pt.turn[p]
    df["dt_prev"] = pt.dt_prev[p]
    df["dt_next"] = pt.dt_next[p]
    df["step_prev"] = pt.step_prev[p]
    df["straightness"] = pt.straightness[p]
    df["speed_window"] = pt.speed_window[p]

    df["out_deg_to"] = net.out_deg[net.arcs[a, 1]]
    df["in_deg_from"] = net.in_deg[net.arcs[a, 0]]
    df["has_reverse"] = net.has_reverse[a].astype(np.int8)

    # Temporal context: distance of neighbouring observations to the same candidate arc.
    ctx = []
    for name, off in [("ctx_dist_prev1", -1), ("ctx_dist_next1", 1), ("ctx_dist_prev2", -2), ("ctx_dist_next2", 2)]:
        j, valid = _neighbour(pt, off)
        dd, _ = point_segment(pt.x[j[p]], pt.y[j[p]], net.ax[a], net.ay[a], net.bx[a], net.by[a])
        dd = np.where(valid[p], dd, d)
        df[name] = dd
        ctx.append(dd)
    df["ctx_dist_mean"] = (d + sum(ctx)) / 5.0

    key = p.astype(np.int64) * (len(net.arcs) + 1) + a
    keyset = np.sort(key)

    def is_cand(off):
        j, valid = _neighbour(pt, off)
        k = j[p].astype(np.int64) * (len(net.arcs) + 1) + a
        pos = np.searchsorted(keyset, k)
        hit = (pos < len(keyset)) & (keyset[np.minimum(pos, len(keyset) - 1)] == k)
        return (hit & valid[p]).astype(np.int8)

    df["ctx_cand_prev"] = is_cand(-1)
    df["ctx_cand_next"] = is_cand(1)

    # Topological link to the neighbour's nearest candidate (same arc, or arcs share a node).
    def link(off):
        j, valid = _neighbour(pt, off)
        nb = nearest[j[p]]
        ok = valid[p] & (nb >= 0)
        nbc = np.where(ok, nb, 0)
        u, v = net.arcs[a, 0], net.arcs[a, 1]
        nu, nv = net.arcs[nbc, 0], net.arcs[nbc, 1]
        same = nbc == a
        touch = (u == nu) | (u == nv) | (v == nu) | (v == nv)
        return (ok & (same | touch)).astype(np.int8)

    df["link_prev_nearest"] = link(-1)
    df["link_next_nearest"] = link(1)

    # Along-arc motion. Positive progress = moving in the arc's direction.
    L = net.length[a]
    t0 = _raw_t(pt.x[p], pt.y[p], net.ax[a], net.ay[a], net.bx[a], net.by[a])
    df["overshoot_m"] = (np.maximum(0.0, -t0) + np.maximum(0.0, t0 - 1.0)) * L
    prog = {}
    for name, off in [("progress_prev", -1), ("progress_next", 1), ("_m2", -2), ("_p2", 2)]:
        j, valid = _neighbour(pt, off)
        tj = _raw_t(pt.x[j[p]], pt.y[j[p]], net.ax[a], net.ay[a], net.bx[a], net.by[a])
        s = np.sign(off)
        prog[name] = np.where(valid[p], s * (tj - t0) * L, 0.0)
    df["progress_prev"] = prog["progress_prev"]
    df["progress_next"] = prog["progress_next"]
    df["progress_2s"] = prog["_m2"] + prog["_p2"]

    lon, lat = pt.proj.inverse(pt.x, pt.y)
    pts = pd.DataFrame({
        "point_idx": np.arange(n), "orig_row": order if order is not None else np.arange(n),
        "t": pt.t, "lon": lon, "lat": lat, "segment": pt.segment,
        "speed": pt.speed, "heading_ok": pt.heading_ok.astype(np.int8),
        "n_cand": n_cand, "dist_min": dmin, "nearest_arc": nearest,
    })

    if trip.route is not None:
        ref_arc, ref_pos, ref_dist = align_to_route(pt, trip.route, cfg["labels"]["alignment_cap_m"],
                                                   cfg["labels"]["alignment_heading_weight_m"])
        pts["ref_arc"] = ref_arc
        pts["ref_route_pos"] = ref_pos
        pts["ref_dist"] = ref_dist
        pts["label_ok"] = (ref_dist <= cfg["labels"]["max_label_dist_m"]).astype(np.int8)
        df["label"] = (a == ref_arc[p]).astype(np.int8)
        has_ref = np.zeros(n, dtype=np.int8)
        has_ref[p[df["label"].values == 1]] = 1
        pts["ref_in_cands"] = has_ref
        df["label_ok"] = pts["label_ok"].values[p]

    df.insert(0, "trip_id", trip.trip_id)
    pts.insert(0, "trip_id", trip.trip_id)
    df["segment"] = pt.segment[p]
    df["t"] = pt.t[p]
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df, pts
