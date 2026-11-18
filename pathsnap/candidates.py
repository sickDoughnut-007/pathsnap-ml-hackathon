"""Candidate road-segment generation with a KD-tree over densified arcs."""
import numpy as np
from scipy.spatial import cKDTree

from .geo import point_segment


def generate_candidates(pt, cfg):
    """Return arrays (point_idx, arc_id, dist, t) of candidates within radius, top-K per point.

    Only canonical arcs are considered (duplicated node pairs collapse to one arc id).
    """
    net = pt.net
    R = cfg["candidates"]["radius_m"]
    K = cfg["candidates"]["max_candidates"]
    step = cfg["candidates"]["index_step_m"]

    canon_ids = np.unique(net.canon)
    # Prefilter arcs to the track's bounding box (+ margin).
    m = R + step
    x0, x1, y0, y1 = pt.x.min() - m, pt.x.max() + m, pt.y.min() - m, pt.y.max() + m
    ax, ay, bx, by = net.ax[canon_ids], net.ay[canon_ids], net.bx[canon_ids], net.by[canon_ids]
    keep = ~((np.maximum(ax, bx) < x0) | (np.minimum(ax, bx) > x1) |
             (np.maximum(ay, by) < y0) | (np.minimum(ay, by) > y1))
    arc_ids = canon_ids[keep]
    if len(arc_ids) == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([]), np.array([])

    L = net.length[arc_ids]
    ns = np.maximum(2, np.ceil(L / step).astype(int) + 1)
    owner = np.repeat(np.arange(len(arc_ids)), ns)
    frac = np.concatenate([np.linspace(0, 1, k) for k in ns])
    sx = net.ax[arc_ids][owner] + frac * (net.bx[arc_ids] - net.ax[arc_ids])[owner]
    sy = net.ay[arc_ids][owner] + frac * (net.by[arc_ids] - net.ay[arc_ids])[owner]
    tree = cKDTree(np.column_stack([sx, sy]))

    hits = tree.query_ball_point(np.column_stack([pt.x, pt.y]), r=R + step / 2.0, return_sorted=False)
    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if lens.sum() == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([]), np.array([])
    p_idx = np.repeat(np.arange(len(hits)), lens)
    s_idx = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])
    a_local = owner[s_idx]
    pair = np.unique(p_idx * (len(arc_ids) + 1) + a_local)
    p_idx = pair // (len(arc_ids) + 1)
    a_id = arc_ids[pair % (len(arc_ids) + 1)]

    d, t = point_segment(pt.x[p_idx], pt.y[p_idx], net.ax[a_id], net.ay[a_id], net.bx[a_id], net.by[a_id])
    ok = d <= R
    p_idx, a_id, d, t = p_idx[ok], a_id[ok], d[ok], t[ok]

    # Top-K by distance per point (ties broken by arc id for determinism).
    order = np.lexsort((a_id, d, p_idx))
    p_idx, a_id, d, t = p_idx[order], a_id[order], d[order], t[order]
    first = np.r_[0, np.flatnonzero(np.diff(p_idx)) + 1]
    rank = np.arange(len(p_idx)) - np.repeat(first, np.diff(np.r_[first, len(p_idx)]))
    k = rank < K
    return p_idx[k], a_id[k], d[k], t[k]
