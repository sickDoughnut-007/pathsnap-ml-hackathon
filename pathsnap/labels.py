"""Per-observation reference labels derived from the ordered .route file.

The .route file lists the arcs the vehicle traversed, in order, but does not say which
GPS observation belongs to which arc. We assign each observation i a route position j(i)
by a monotone alignment: j(0) <= j(1) <= ... (the vehicle never goes back along the
route list), minimising the total perpendicular distance sum_i d(i, route[j(i)]).
This is solved exactly with dynamic programming in O(n * m).

The per-point cost is truncated at `cap_m` (robust alignment): observations that are far
from every route arc (e.g. a track that continues after the reference route ends, as in
trip 00000054) contribute a constant and cannot drag the alignment of on-route points.

Monotonicity resolves the ambiguities that pure nearest-arc labelling gets wrong:
two-way roads (u->v vs v->u), U-turns and routes that pass the same place twice.

Two further rules make the alignment direction-consistent (found during EDA: without
them ~20% of slow-speed labels pointed against the direction of travel, because a route
that uses a street in both directions gives equal geometric cost to both passes):
  * a direction term: where the GPS heading is reliable, an arc pointing against the
    heading costs up to `heading_weight_m` extra metres, (1 - cos(diff)) / 2 * weight;
  * ties are broken toward the EARLIEST route position (the vehicle has not yet
    progressed), never by jumping ahead.
"""
import numpy as np

from .geo import angle_diff, point_segment


def align_to_route(pt, route, cap_m=100.0, heading_weight_m=20.0):
    net = pt.net
    r = net.canon[np.asarray(route, dtype=np.int64)]
    n, m = len(pt.x), len(r)
    ax, ay, bx, by = net.ax[r], net.ay[r], net.bx[r], net.by[r]
    rb = net.bearing[r]

    ptr = np.empty((n, m), dtype=np.int32 if m > 32000 else np.int16)
    idx = np.arange(m)
    prev = None
    dists = np.empty((n, m), dtype=np.float32)
    for i in range(n):
        d, _ = point_segment(pt.x[i], pt.y[i], ax, ay, bx, by)
        dists[i] = d
        c = np.minimum(d, cap_m)
        if pt.heading_ok[i] and heading_weight_m > 0:
            c = c + heading_weight_m * (1.0 - np.cos(np.radians(angle_diff(pt.heading[i], rb)))) / 2.0
        if prev is None:
            cost = c.astype(np.float64)
            ptr[i] = idx
        else:
            acc = np.minimum.accumulate(prev)
            # earliest index attaining the running minimum: strict new-minimum positions
            rec = np.empty(m, dtype=bool)
            rec[0] = True
            rec[1:] = prev[1:] < acc[:-1]
            arg = np.maximum.accumulate(np.where(rec, idx, 0))
            cost = acc + c
            ptr[i] = arg
        prev = cost

    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = ptr[i, j[i]]
    ref_arc = r[j]
    ref_dist = dists[np.arange(n), j].astype(np.float64)
    return ref_arc, j, ref_dist
