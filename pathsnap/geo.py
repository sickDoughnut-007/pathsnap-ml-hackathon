"""Geometry helpers. All distances in metres, bearings in degrees clockwise from north."""
import numpy as np

R_EARTH = 6371008.8


class LocalProjection:
    """Equirectangular tangent-plane projection centred at (lon0, lat0).

    Adequate for a single trip (tens of km): distances here are always local
    (GPS point to a nearby road), so the scale error over the trip extent is small.
    """

    def __init__(self, lon0, lat0, radius=R_EARTH):
        self.lon0, self.lat0, self.R = float(lon0), float(lat0), radius
        self.kx = np.radians(1.0) * radius * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * radius

    def __call__(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def inverse(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0

    def max_scale_error(self, lat):
        """Relative x-scale error at the most distant latitude of the trip."""
        lat = np.asarray(lat)
        c0 = np.cos(np.radians(self.lat0))
        return float(np.max(np.abs(np.cos(np.radians(lat)) / c0 - 1.0)))


def haversine(lon1, lat1, lon2, lat2, radius=R_EARTH):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_xy(dx, dy):
    """Bearing of vector (dx east, dy north), degrees in [0, 360)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (broadcasting). Returns (dist, t) with t in [0,1]."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * vx + (py - ay) * vy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t
