"""Trip-local geometry, reference alignment and road candidate generation."""

from io import BytesIO

import numpy as np
import shapely


FEATURES = ("distance_m", "log_distance", "direction_cosine", "heading_available",
            "speed_mps", "log_segment_length", "distance_to_endpoint_m")
RADIUS_M = 60
MAX_CANDIDATES = 8
MAX_LABEL_DISTANCE_M = 30


def load_trip(archive, split, trip):
    base = f"Student folder/{split}/{trip}/{trip}."

    def read(extension, dtype, ndim):
        return np.loadtxt(BytesIO(archive.read(base + extension)), dtype=dtype, ndmin=ndim)

    nodes = read("nodes", np.float64, 2)
    arcs = read("arcs", np.int32, 2)
    track = read("track", np.float64, 2)
    route = read("route", np.int32, 1)
    if nodes.shape[1] != 2 or arcs.shape[1] != 2 or track.shape[1] != 3:
        raise ValueError(f"{trip}: unexpected file dimensions")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"{trip}: nonfinite coordinates or timestamps")
    if (np.diff(track[:, 2]) <= 0).any():
        raise ValueError(f"{trip}: timestamps must increase")
    if arcs.min() < 0 or arcs.max() >= len(nodes) or route.min() < 0 or route.max() >= len(arcs):
        raise ValueError(f"{trip}: invalid node or arc reference")
    return nodes, arcs, track, route


def project(nodes, track):
    # Local equirectangular coordinates centred on the trip, in metres.
    origin = track[:, :2].mean(axis=0)
    scale = np.deg2rad(1.0) * 6371008.8 * np.array([np.cos(np.deg2rad(origin[1])), 1])
    return (nodes - origin) * scale, (track[:, :2] - origin) * scale


def segment_geometry(points, starts, vectors, lengths_sq):
    offset = points - starts
    fraction = (offset * vectors).sum(axis=-1) / np.maximum(lengths_sq, 1e-12)
    fraction = fraction.clip(0, 1)
    residual = offset - fraction[..., None] * vectors
    return np.sqrt((residual * residual).sum(axis=-1)), fraction


def align_reference(points, segments, route):
    """Minimise total GPS-to-reference distance subject to nondecreasing route index."""
    reference = segments[route]
    starts = reference[:, 0]
    vectors = reference[:, 1] - starts
    lengths_sq = (vectors * vectors).sum(axis=1)
    positions = np.arange(len(route), dtype=np.int32)
    # Each row stores the best predecessor for every possible reference position.
    parents = np.empty((len(points), len(route)), dtype=np.int32)
    cost = np.zeros(len(route))
    for begin in range(0, len(points), 64):
        distances, _ = segment_geometry(points[begin:begin + 64, None], starts, vectors, lengths_sq)
        for offset, row in enumerate(distances):
            prefix = np.minimum.accumulate(cost)
            improved = np.r_[True, cost[1:] < prefix[:-1]]
            parents[begin + offset] = np.maximum.accumulate(np.where(improved, positions, 0))
            cost = prefix + row
    path = np.empty(len(points), dtype=np.int32)
    position = cost.argmin()
    for i in range(len(points) - 1, -1, -1):
        path[i] = position
        position = parents[i, position]
    distance, _ = segment_geometry(points, starts[path], vectors[path], lengths_sq[path])
    return route[path], distance


def motion(points, times):
    # Only past observations contribute. Drop unreliable direction at stops or after gaps.
    delta = np.diff(points, axis=0, prepend=points[:1])
    dt = np.diff(times, prepend=times[0])
    length = np.linalg.norm(delta, axis=1)
    recent = (dt > 0) & (dt <= 10)
    valid = recent & (length >= 3)
    unit = np.divide(delta, np.maximum(length[:, None], 1e-9)) * valid[:, None]
    speed = np.divide(length, dt, out=np.zeros_like(dt), where=recent)
    return unit, valid, speed


def candidates(points, times, segments):
    """Return sparse candidate rows ordered by observation, distance, then arc ID."""
    tree = shapely.STRtree(shapely.linestrings(segments))
    obs, arc = tree.query(shapely.points(points), predicate="dwithin", distance=RADIUS_M)
    vectors = segments[:, 1] - segments[:, 0]
    lengths_sq = (vectors * vectors).sum(axis=1)
    distance, fraction = segment_geometry(points[obs], segments[arc, 0], vectors[arc], lengths_sq[arc])
    # Opposite arcs can differ by floating-point roundoff at the same physical distance.
    order = np.lexsort((arc, distance.round(6), obs))
    obs = obs[order]
    starts = np.searchsorted(obs, obs)
    keep = np.arange(len(obs)) - starts < MAX_CANDIDATES
    order = order[keep]
    obs, arc = obs[keep], arc[order]
    distance, fraction = distance[order], fraction[order]
    heading, valid, speed = motion(points, times)
    length = np.sqrt(lengths_sq[arc])
    cosine = (heading[obs] * vectors[arc]).sum(axis=1) / np.maximum(length, 1e-9)
    endpoint = np.minimum(fraction, 1 - fraction) * length
    x = np.column_stack((distance, np.log1p(distance), cosine, valid[obs], speed[obs],
                         np.log1p(length), endpoint)).astype(np.float32)
    return obs.astype(np.int32), arc.astype(np.int32), x


def prepare(archive, split, trip):
    nodes, arcs, track, route = load_trip(archive, split, trip)
    projected, points = project(nodes, track)
    segments = projected[arcs]
    obs, arc, x = candidates(points, track[:, 2], segments)
    reference, error = align_reference(points, segments, route)
    return dict(obs=obs, arc=arc, x=x, reference=reference, error=error)
