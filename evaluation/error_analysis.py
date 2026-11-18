"""Out-of-fold error analysis of the untuned Random Forest (80 training trips, shared 5 folds).

Answers: what kind of mistakes does the point-wise classifier make, and could HMM decoding fix them?
Uses only the training trips (the student-test trips are not touched here).
Writes results/model_comparison/rf_error_analysis.json.

    python evaluation/error_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from pathsnap import modeling as M  # noqa: E402
from pathsnap.dataset import FEATURES  # noqa: E402

ROWS_PER_FOLD = 60_000  # same as models/random_forest
PARAMS = dict(n_estimators=100, max_depth=20, min_samples_leaf=5, max_features="sqrt",
              n_jobs=-1, random_state=M.SEED)

tr, _, tr_pts, _ = M.load_data()
oof = np.zeros(len(tr))
for f in sorted(tr["fold"].unique()):
    a = M.subsample_by_fold(tr[tr.fold != f], ROWS_PER_FOLD)
    vm = (tr["fold"] == f).values
    m = RandomForestClassifier(**PARAMS).fit(a[FEATURES], a["label"], sample_weight=a["trip_weight"].values)
    oof[vm] = m.predict_proba(tr.loc[vm, FEATURES])[:, 1]

tr["score"] = oof
tr["rank_pred"] = tr.groupby(["trip_id", "point_idx"])["score"].rank(ascending=False, method="first")
n_lab = int(tr_pts["label_ok"].sum())
pos = tr[tr["label"] == 1]
res = {f"oof_top{k}_accuracy": float((pos["rank_pred"] <= k).sum() / n_lab) for k in (1, 2, 3)}

best = tr[tr["rank_pred"] == 1].set_index(["trip_id", "point_idx"])
true = pos.set_index(["trip_id", "point_idx"])[["dist", "arc_len", "proj_t"]]
err = best[best["label"] == 0].join(true, rsuffix="_true", how="inner")
same_street = (np.abs(err["dist"] - err["dist_true"]) < 0.05) & (np.abs(err["arc_len"] - err["arc_len_true"]) < 0.05)
boundary = ~same_street & ((err["proj_t_true"] < 0.1) | (err["proj_t_true"] > 0.9)
                           | (err["proj_t"] < 0.1) | (err["proj_t"] > 0.9))
res.update({
    "n_labelled_observations": n_lab,
    "n_top1_errors_with_ref_in_candidates": int(len(err)),
    "error_share_opposite_direction_same_street": float(same_street.mean()),
    "error_share_adjacent_segment_at_boundary": float(boundary.mean()),
    "error_share_other": float((~same_street & ~boundary).mean()),
    "error_share_vehicle_slow_lt_2mps": float((err["speed_window"] < 2).mean()),
    "definitions": {
        "opposite_direction_same_street": "predicted and reference arcs have equal perpendicular distance and length (reverse arc of a two-way street)",
        "adjacent_segment_at_boundary": "otherwise, the observation projects within 10% of an end of either arc",
    },
})
out = ROOT / "results" / "model_comparison" / "rf_error_analysis.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
