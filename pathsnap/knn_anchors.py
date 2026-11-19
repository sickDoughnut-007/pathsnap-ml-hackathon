"""Build a coarse route from distance-spaced observations, then guide full-rate decoding."""
from dataclasses import replace

import numpy as np

from .geo import LocalProjection
from .knn_sequence import TripLattice


def anchor_corridor(rows, scores, trip, full, settings, spacing):
    """Mark candidate arcs on the local anchor route, including directed connectors."""
    if not len(full.points):
        return np.empty((0, 0), dtype=bool)
    corridor = np.zeros_like(full.arc, dtype=bool)
    track = trip.track[np.argsort(trip.track[:, 2], kind="stable")]
    project = LocalProjection(*track[:, :2].mean(axis=0))
    x, y = project(track[full.points, 0], track[full.points, 1])
    anchors, groups = [], []
    segments = np.split(np.arange(len(x)), np.flatnonzero(full.restart[1:]) + 1)
    for group, segment in enumerate(segments):
        last = segment[0]
        for i in segment:
            if i in (segment[0], segment[-1]) or np.hypot(x[i] - x[last], y[i] - y[last]) >= spacing:
                anchors.append(i)
                groups.append(group)
                last = i
    anchors = np.asarray(anchors)
    source_ids = full.points[anchors]
    coarse = rows.assign(knn=scores).loc[rows.point_idx.isin(source_ids)].copy()
    index = np.searchsorted(source_ids, coarse.point_idx)
    coarse["point_idx"] = index
    coarse["segment"] = np.asarray(groups)[index]
    coarse["step_prev"] = np.hypot(np.diff(x[anchors], prepend=x[anchors[0]]),
                                   np.diff(y[anchors], prepend=y[anchors[0]]))[index]
    lattice = TripLattice(coarse, coarse.knn.to_numpy(), trip)
    arcs = lattice.decode(replace(settings, realign=False))
    for j, start in enumerate(anchors):
        if j + 1 == len(anchors) or groups[j] != groups[j + 1]:
            corridor[start] |= full.arc[start] == arcs[j]
            continue
        stop = anchors[j + 1] + 1
        old, new = arcs[j:j + 2]
        choices = [old, new]
        if old != new:
            key = trip.arcs[old, 1] * len(trip.nodes) + trip.arcs[new, 0]
            choices.extend(lattice.bridges.get(key, ()))
        corridor[start:stop] |= np.isin(full.arc[start:stop], choices)
    return corridor
