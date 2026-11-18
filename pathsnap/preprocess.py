"""Track preprocessing: projection, gap segmentation, kinematics (speed, heading, turn)."""
from dataclasses import dataclass

import numpy as np

from .geo import LocalProjection, angle_diff, bearing_xy
from .network import build_network


@dataclass
class PreparedTrip:
    trip_id: str
    proj: LocalProjection
    t: np.ndarray            # seconds
    x: np.ndarray            # metres
    y: np.ndarray
    segment: np.ndarray      # int segment id (new segment after each gap > gap_split_s)
    dt_prev: np.ndarray      # seconds since previous point (0 for first)
    dt_next: np.ndarray
    step_prev: np.ndarray    # metres from previous point
    speed: np.ndarray        # m/s, central difference inside the segment
    heading: np.ndarray      # degrees; NaN where unreliable
    heading_ok: np.ndarray   # bool
    turn: np.ndarray         # |heading change| vs previous point, degrees (0 if unknown)
    net: object


def _within_segment_index(seg, offset):
    """Index i+offset clipped to the same segment as i."""
    n = len(seg)
    idx = np.arange(n)
    j = np.clip(idx + offset, 0, n - 1)
    # clip back toward i while crossing a segment boundary
    bad = seg[j] != seg
    while bad.any():
        j[bad] -= np.sign(offset)
        bad = seg[j] != seg
    return j


def prepare_trip(trip, cfg):
    tr = trip.track
    order = np.argsort(tr[:, 2], kind="stable")  # enforce chronological order
    tr = tr[order]
    lon, lat, t = tr[:, 0], tr[:, 1], tr[:, 2]
    proj = LocalProjection(lon.mean(), lat.mean(), cfg["projection"]["earth_radius_m"])
    x, y = proj(lon, lat)

    dt = np.diff(t)
    dt_prev = np.concatenate([[0.0], dt])
    dt_next = np.concatenate([dt, [0.0]])
    gap = cfg["preprocessing"]["gap_split_s"]
    segment = np.concatenate([[0], np.cumsum(dt > gap)])
    step_prev = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y))])
    step_prev[np.concatenate([[True], dt > gap])] = 0.0

    ip, inx = _within_segment_index(segment, -1), _within_segment_index(segment, 1)
    span_t = t[inx] - t[ip]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(span_t > 0, np.hypot(x[inx] - x[ip], y[inx] - y[ip]) / span_t, 0.0)

    w = cfg["preprocessing"]["heading_window"]
    a, b = _within_segment_index(segment, -w), _within_segment_index(segment, w)
    dx, dy = x[b] - x[a], y[b] - y[a]
    disp = np.hypot(dx, dy)
    heading_ok = disp >= cfg["preprocessing"]["min_heading_disp_m"]
    heading = np.where(heading_ok, bearing_xy(dx, dy), np.nan)
    prev_h = np.concatenate([[np.nan], heading[:-1]])
    same_seg = np.concatenate([[False], segment[1:] == segment[:-1]])
    turn = angle_diff(heading, prev_h)
    turn = np.where(same_seg & np.isfinite(turn), turn, 0.0)

    # Jitter indicators: when stopped, GPS noise fakes speed and random headings.
    # straightness = net displacement / path length over +-jw points (1 = straight drive, ~0 = jitter).
    jw = cfg["preprocessing"]["jitter_window"]
    ja, jb = _within_segment_index(segment, -jw), _within_segment_index(segment, jw)
    cum = np.cumsum(step_prev)
    path = cum[jb] - cum[ja]
    net_disp = np.hypot(x[jb] - x[ja], y[jb] - y[ja])
    with np.errstate(invalid="ignore", divide="ignore"):
        straightness = np.where(path > 0, net_disp / path, 0.0)
        span = t[jb] - t[ja]
        speed_window = np.where(span > 0, net_disp / span, 0.0)

    net = build_network(trip.nodes, trip.arcs, proj)
    pt = PreparedTrip(trip.trip_id, proj, t, x, y, segment, dt_prev, dt_next,
                      step_prev, speed, heading, heading_ok, turn, net)
    pt.straightness = straightness
    pt.speed_window = speed_window
    return pt, order
