"""Reading raw PathSnap trip files and the shared config."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def data_root(cfg, override=None):
    root = override or os.environ.get("PATHSNAP_DATA_ROOT") or cfg["data"]["root"]
    root = Path(root)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    return root


def list_trips(cfg, partition, root=None):
    """partition: 'train' or 'student_test'. Returns sorted trip ids (8-digit strings)."""
    key = {"train": "train_dir", "student_test": "student_test_dir"}[partition]
    d = data_root(cfg, root) / cfg["data"][key]
    return sorted(p.name for p in d.iterdir() if p.is_dir())


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray          # (n, 3): lon, lat, t_seconds
    nodes: np.ndarray          # (N, 2): lon, lat ; row index = node_id
    arcs: np.ndarray           # (A, 2): from_node, to_node ; row index = arc_id
    route: np.ndarray | None   # (m,): ordered reference arc_ids, None if not supplied


def _load(path, dtype=float, ndmin=2):
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(cfg, partition, trip_id, root=None, with_route=True):
    """Load one trip directory. Set with_route=False in any prediction code path."""
    key = {"train": "train_dir", "student_test": "student_test_dir"}.get(partition)
    base = (data_root(cfg, root) / cfg["data"][key]) if key else Path(partition)
    d = base / trip_id
    track = _load(d / f"{trip_id}.track")
    nodes = _load(d / f"{trip_id}.nodes")
    arcs = _load(d / f"{trip_id}.arcs", dtype=np.int64)
    route = None
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        route = _load(rp, dtype=np.int64, ndmin=1)
    return Trip(trip_id, partition, track, nodes, arcs, route)


def load_trip_dir(trip_dir, with_route=False):
    """Load a trip from an arbitrary directory (e.g. evaluator trips at demo time)."""
    d = Path(trip_dir)
    tid = d.name
    track = _load(d / f"{tid}.track")
    nodes = _load(d / f"{tid}.nodes")
    arcs = _load(d / f"{tid}.arcs", dtype=np.int64)
    route = None
    if with_route and (d / f"{tid}.route").exists():
        route = _load(d / f"{tid}.route", dtype=np.int64, ndmin=1)
    return Trip(tid, "external", track, nodes, arcs, route)
