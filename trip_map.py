"""Build a self-contained Folium map using only the supplied road geometry."""

import argparse
import base64
import hashlib
import tarfile
from pathlib import Path
from zipfile import ZipFile

import folium
import numpy as np
import shapely
from branca.element import Element

from map_matching import load_trip, project


def embed_leaflet(map_view, bundle):
    expected = "84c65a256e50657896f54c33bd857b6849ebe94c817803be818bf32a3dde0b77"
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected:
        raise ValueError("Expected the Leaflet 1.9.4 npm release archive")
    map_view.default_js, map_view.default_css = [], []
    with tarfile.open(bundle) as package:
        def read(name):
            return package.extractfile("package/" + name).read()

        js = read("dist/leaflet.js").decode().split("//# sourceMappingURL=")[0]
        css = read("dist/leaflet.css").decode()
        for name in ("layers.png", "layers-2x.png", "marker-icon.png"):
            image = base64.b64encode(read("dist/images/" + name)).decode()
            css = css.replace("images/" + name, "data:image/png;base64," + image)
        license_text = read("LICENSE").decode()
    header = map_view.get_root().header
    header.add_child(Element(f"<!-- {license_text} -->\n<script>{js}</script><style>{css}</style>"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        nodes, arcs, track, route = load_trip(archive, "train", "00000001")
    with np.load(args.output / "example_predictions.npz") as saved:
        nearest, logistic = saved["nearest"], saved["logistic"]
    metric_nodes, gps = project(nodes, track)
    tree = shapely.STRtree(shapely.linestrings(metric_nodes[arcs]))
    corridor = shapely.linestrings(gps)
    road_ids = tree.query(corridor, predicate="dwithin", distance=180)
    latlon = nodes[:, ::-1]
    m = folium.Map(tiles=None, prefer_canvas=True, control_scale=True, max_zoom=20)
    embed_leaflet(m, args.output / "leaflet-1.9.4.tgz")

    def roads(title, ids, color, width, visible=True):
        group = folium.FeatureGroup(name=title, show=visible).add_to(m)
        valid = np.unique(ids[ids >= 0])
        if len(valid):
            edges = np.unique(np.sort(arcs[valid], axis=1), axis=0)
            folium.PolyLine(latlon[edges].round(6).tolist(), color=color, weight=width,
                            opacity=0.85, smooth_factor=0).add_to(group)

    roads("Road network · grey", road_ids, "#83949b", 1.5)
    roads("Reference route · green", route, "#138453", 8)
    roads("Nearest road · orange", nearest, "#d07d13", 5, False)
    roads("Logistic regression · purple", logistic, "#8243ad", 3)
    group = folium.FeatureGroup(name="GPS observations · blue").add_to(m)
    locations = track[:, [1, 0]]
    for point in locations:
        folium.CircleMarker(point.tolist(), radius=2, stroke=False, fill=True,
                            fill_color="#1256bc", fill_opacity=0.85).add_to(group)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([locations.min(axis=0).tolist(), locations.max(axis=0).tolist()])
    m.get_root().html.add_child(Element(
        '<div style="position:fixed;z-index:1000;bottom:22px;left:12px;background:white;'
        'padding:8px 12px;border-radius:4px;font:14px system-ui;color:#172d3a">'
        '<strong>PathSnap · Trip 00000001</strong><br>Held-out training trip · 1,315 GPS points</div>'))
    m.save(str(args.output / "trip_00000001.html"))


if __name__ == "__main__":
    main()
