"""Road-network preparation: projection, duplicate-arc canonicalisation, connectivity."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .geo import bearing_xy


@dataclass
class Network:
    node_xy: np.ndarray        # (N, 2) projected metres
    arcs: np.ndarray           # (A, 2) from, to (original arc ids = row index)
    canon: np.ndarray          # (A,) canonical arc id for each arc (duplicates of a node pair -> lowest id)
    ax: np.ndarray
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    length: np.ndarray         # (A,) metres
    bearing: np.ndarray        # (A,) degrees
    out_deg: np.ndarray        # (N,)
    in_deg: np.ndarray         # (N,)
    has_reverse: np.ndarray    # (A,) bool: arc v->u exists for u->v
    adj: csr_matrix            # node adjacency (directed)

    def successors(self, arc_id):
        """Arc ids that start where arc_id ends."""
        return self._out_arcs[self.arcs[arc_id, 1]]

    def build_out_arcs(self):
        order = np.argsort(self.arcs[:, 0], kind="stable")
        starts = np.searchsorted(self.arcs[order, 0], np.arange(len(self.node_xy) + 1))
        self._out_arcs = [order[starts[i]:starts[i + 1]] for i in range(len(self.node_xy))]
        return self


def build_network(nodes_lonlat, arcs, proj):
    x, y = proj(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
    node_xy = np.column_stack([x, y])
    u, v = arcs[:, 0], arcs[:, 1]
    N = len(node_xy)

    # Canonicalise duplicated node pairs (same u->v listed more than once).
    key = u.astype(np.int64) * (N + 1) + v
    _, first_idx, inv = np.unique(key, return_index=True, return_inverse=True)
    canon = first_idx[inv]

    ax, ay = node_xy[u, 0], node_xy[u, 1]
    bx, by = node_xy[v, 0], node_xy[v, 1]
    length = np.hypot(bx - ax, by - ay)
    bearing = bearing_xy(bx - ax, by - ay)
    out_deg = np.bincount(u, minlength=N)
    in_deg = np.bincount(v, minlength=N)
    rkey = v.astype(np.int64) * (N + 1) + u
    has_reverse = np.isin(rkey, key)
    adj = csr_matrix((np.ones(len(u)), (u, v)), shape=(N, N))
    return Network(node_xy, arcs, canon, ax, ay, bx, by, length, bearing,
                   out_deg, in_deg, has_reverse, adj).build_out_arcs()


def connectivity_summary(net):
    n_weak, lab_w = connected_components(net.adj, directed=True, connection="weak")
    n_strong, lab_s = connected_components(net.adj, directed=True, connection="strong")
    used = np.zeros(len(net.node_xy), bool)
    used[net.arcs.ravel()] = True
    sizes_w = np.bincount(lab_w)
    sizes_s = np.bincount(lab_s)
    return {
        "n_nodes": int(len(net.node_xy)),
        "n_arcs": int(len(net.arcs)),
        "isolated_nodes": int((~used).sum()),
        "self_loops": int((net.arcs[:, 0] == net.arcs[:, 1]).sum()),
        "duplicate_arcs": int((net.canon != np.arange(len(net.arcs))).sum()),
        "zero_length_arcs": int((net.length < 1e-6).sum()),
        "frac_two_way": float(net.has_reverse.mean()),
        "weak_components": int(n_weak),
        "largest_weak_frac": float(sizes_w.max() / len(net.node_xy)),
        "strong_components": int(n_strong),
        "largest_strong_frac": float(sizes_s.max() / len(net.node_xy)),
        "median_arc_len_m": float(np.median(net.length)),
        "total_len_km": float(net.length.sum() / 1000),
    }
