"""Viterbi decoding of KNN candidate scores over a directed trip network."""
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from scipy.sparse import csr_matrix

from .geo import LocalProjection


@dataclass(frozen=True)
class Settings:
    gps_scale: float
    transition_scale: float
    turn_cost: float
    knn_strength: float
    switch_cost: float = 0
    realign: bool = False
    bridge: bool = False
    escape_cost: float = 20
    damp_jitter: bool = False


def _connecting_distances(pairs, nodes, arcs, track):
    """Shortest directed paths for requested node pairs within 500 m."""
    project = LocalProjection(*track[:, :2].mean(axis=0))
    x, y = project(nodes[:, 0], nodes[:, 1])
    # Duplicate directed arcs represent the same physical length.
    edges, original = np.unique(arcs, axis=0, return_index=True)
    u, v = edges.T
    length = np.hypot(x[u] - x[v], y[u] - y[v])
    graph = csr_matrix((length, (u, v)), shape=(len(nodes), len(nodes)))
    sources, destinations = np.divmod(pairs, len(nodes))
    result = np.full(len(pairs), np.inf, np.float32)
    bridges = {}
    splits = np.r_[0, np.flatnonzero(np.diff(sources)) + 1, len(pairs)]
    for lo, hi in zip(splits[:-1], splits[1:]):
        source = int(sources[lo])
        dist = {source: 0.0}
        parents = {}
        queue = [(0.0, source)]
        while queue:
            d, node = heappop(queue)
            if d > dist[node]:
                continue
            for j in range(graph.indptr[node], graph.indptr[node + 1]):
                nxt = int(graph.indices[j])
                new = d + graph.data[j]
                if new <= 500 and new < dist.get(nxt, np.inf):
                    dist[nxt] = new
                    parents[nxt] = (node, int(original[j]))
                    heappush(queue, (new, nxt))
        result[lo:hi] = [dist.get(n, np.inf) for n in destinations[lo:hi]]
        for key, dest in zip(pairs[lo:hi], destinations[lo:hi]):
            if dest in parents:
                path = []
                node = dest
                while node != source:
                    node, arc = parents[node]
                    path.append(arc)
                bridges[int(key)] = tuple(reversed(path))
    return result, bridges


class TripLattice:
    """Candidate matrices ordered by GPS time. Inputs contain no reference route."""

    def __init__(self, rows, scores, trip):
        rows = rows.assign(knn=scores).sort_values(["point_idx", "rank"])
        point, index = np.unique(rows.point_idx.to_numpy(), return_inverse=True)
        self.points = point
        if not len(point):
            return
        rank = rows["rank"].to_numpy(dtype=int)
        shape = (len(point), int(rank.max()) + 1)

        def matrix(values, default, dtype):
            out = np.full(shape, default, dtype=dtype)
            out[index, rank] = values
            return out

        self.arc = matrix(rows.arc_id, -1, np.int32)
        self.dist = matrix(rows.dist, np.inf, np.float32)
        self.log_knn = matrix(np.log(np.clip(rows.knn, 1e-8, 1)), 0, np.float32)
        length = matrix(rows.arc_len, 0, np.float32)
        along = length * matrix(rows.proj_t, 0, np.float32)
        first = rows.loc[rank == 0]
        self.motion_confidence = np.clip(first.speed_window.to_numpy(np.float32) / 5, .15, 1)
        self.restart = np.r_[True, (np.diff(point) != 1) | (np.diff(first.segment) != 0)]
        self.network_arcs = trip.arcs
        self.node_count = len(trip.nodes)
        self.bridges = {}
        if len(point) == 1:
            self.error = self.reverse = np.empty((0, shape[1], shape[1]), np.float32)
            return
        u, v = trip.arcs[np.maximum(self.arc, 0)].transpose(2, 0, 1)
        keys = v[:-1, :, None].astype(np.int64) * len(trip.nodes) + u[1:, None, :]
        unique = np.unique(keys)
        lookup, self.bridges = _connecting_distances(unique, trip.nodes, trip.arcs, trip.track)
        travel = length[:-1, :, None] - along[:-1, :, None] + lookup[np.searchsorted(unique, keys)] + along[1:, None, :]
        step = first.step_prev.to_numpy(np.float32)[1:, None, None]
        self.error = np.abs(travel - step)
        same = self.arc[:-1, :, None] == self.arc[1:, None, :]
        progress = along[1:, None, :] - along[:-1, :, None]
        self.error = np.where(same, np.abs(progress - step), self.error)
        self.reverse = ((u[:-1, :, None] == v[1:, None, :]) &
                        (v[:-1, :, None] == u[1:, None, :]) & ~same)

    def decode(self, cfg):
        if not len(self.points):
            return np.empty(0, np.int32)
        emission = cfg.knn_strength * self.log_knn - self.dist / cfg.gps_scale
        if cfg.damp_jitter:
            emission *= self.motion_confidence[:, None]
        # A finite escape cost lets the sequence recover at incomplete road networks.
        penalty = np.minimum(self.error / cfg.transition_scale + self.reverse * cfg.turn_cost, cfg.escape_cost)
        if cfg.switch_cost:
            penalty += cfg.switch_cost * (self.arc[:-1, :, None] != self.arc[1:, None, :])
        back = np.zeros(emission.shape, np.uint8)
        state = emission[0].copy()
        ends = {}
        cols = np.arange(emission.shape[1])
        for i in range(1, len(emission)):
            if self.restart[i]:
                ends[i - 1] = state.argmax()
                state = emission[i].copy()
                continue
            scores = state[:, None] - penalty[i - 1]
            back[i] = scores.argmax(axis=0)
            state = emission[i] + scores[back[i], cols]
            state -= state.max()
        ends[len(emission) - 1] = state.argmax()
        path = np.empty(len(emission), np.int32)
        for i in range(len(path) - 1, -1, -1):
            if i in ends:
                chosen = ends[i]
            path[i] = self.arc[i, chosen]
            chosen = back[i, chosen]
        if cfg.realign:
            for segment in np.split(np.arange(len(path)), np.flatnonzero(self.restart[1:]) + 1):
                ids = path[segment]
                starts = np.r_[True, ids[1:] != ids[:-1]]
                route = ids[starts]
                center = np.cumsum(starts) - 1
                if cfg.bridge:
                    expanded = [route[0]]
                    centers = [0]
                    for old, new in zip(route[:-1], route[1:]):
                        key = self.network_arcs[old, 1] * self.node_count + self.network_arcs[new, 0]
                        expanded.extend(self.bridges.get(key, ()))
                        centers.append(len(expanded))
                        expanded.append(new)
                    route = np.asarray(expanded)
                    center = np.asarray(centers)[center]
                path[segment] = _realign(route, center, self.arc[segment], self.dist[segment])
        return path


def _realign(route, center, candidates, distances):
    """Refine switch timing along the predicted route using only candidate geometry."""
    positions = np.clip(center[:, None] + np.arange(-3, 4), 0, len(route) - 1)
    options = route[positions]
    match = candidates[:, None, :] == options[:, :, None]
    cost = np.min(np.where(match, distances[:, None, :], np.inf), axis=2)
    state = cost[0]
    parent = np.zeros(cost.shape, dtype=np.uint8)
    for i in range(1, len(center)):
        # Stay at this route position or advance, preserving trip order.
        before = np.where(positions[i-1, :, None] <= positions[i, None, :], state[:, None], np.inf)
        parent[i] = before.argmin(axis=0)
        state = cost[i] + before.min(axis=0)
    chosen = state.argmin()
    output = np.empty(len(center), dtype=route.dtype)
    for i in range(len(output) - 1, -1, -1):
        output[i] = options[i, chosen]
        chosen = parent[i, chosen]
    return output
