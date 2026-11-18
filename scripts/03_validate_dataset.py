"""Sanity-check the shared dataset and record reference numbers every model should beat.

Not one of the five models: it scores two NON-learned rules with the shared evaluator,
and measures the route continuity of the reference labels themselves (the ceiling).

  rule_nearest            highest score = smallest perpendicular distance
  rule_dist_heading       score = -(dist + 0.2 * heading_diff)

Output: results/preprocessing/validation.json
Usage:  python scripts/03_validate_dataset.py [--data-root PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.dataset import FEATURES, load_candidates, load_points, model_rows  # noqa: E402
from pathsnap.evaluate import observation_metrics, route_continuity, top1_predictions  # noqa: E402
from pathsnap.io import REPO_ROOT, load_config, load_trip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    cfg = load_config()
    res = {}
    for part in ["train", "student_test"]:
        c = model_rows(load_candidates(part))
        p = load_points(part)
        assert not c[FEATURES].isna().any().any(), "NaN in features"
        arcs = {t: load_trip(cfg, part, t, args.data_root).arcs for t in c["trip_id"].unique()}
        out = {"rows": int(len(c)), "positive_rate": float(c["label"].mean()),
               "observations": int(len(p)), "label_ok_frac": float(p["label_ok"].mean())}
        for name, s in [("rule_nearest", -c["dist"].values),
                        ("rule_dist_heading", -(c["dist"].values + 0.2 * c["heading_diff"].values))]:
            # rules give scores, not probabilities: report ranking metrics only
            m = observation_metrics(c, 1 / (1 + np.exp(-s / 10)), p)
            best = top1_predictions(c, s)
            cont, _ = route_continuity(best, arcs, max_hops=1)
            cont2, _ = route_continuity(best, arcs, max_hops=2)
            out[name] = {"top1_accuracy": m["top1_accuracy"], "route_continuity_1hop": cont,
                         "route_continuity_2hop": cont2}
        lab = p[p["label_ok"] == 1].rename(columns={"ref_arc": "pred_arc"})
        c1, _ = route_continuity(lab, arcs, max_hops=1)
        c2, _ = route_continuity(lab, arcs, max_hops=2)
        out["reference_labels"] = {"route_continuity_1hop": c1, "route_continuity_2hop": c2,
                                   "candidate_miss_rate": float(1 - p.loc[p.label_ok == 1, "ref_in_cands"].mean())}
        res[part] = out
        print(part, json.dumps(out, indent=1))
    d = REPO_ROOT / "results" / "preprocessing"
    d.mkdir(parents=True, exist_ok=True)
    (d / "validation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
