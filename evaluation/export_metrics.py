"""Export the team-standard metric list for each trained model, read from models/<name>/metrics.json.

    python evaluation/export_metrics.py random_forest mlp

Writes results/model_comparison/metrics_summary.csv (one column per model, values in %
except log loss) and prints it. Every number comes from the saved metrics.json files.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
names = sys.argv[1:] or sorted(p.parent.name for p in (ROOT / "models").glob("*/metrics.json")
                               if not p.parent.name.startswith("_"))

ROWS = [  # (label, source, key, as_percent)
    ("CV Top-1 Accuracy", "cv", "cv_top1_mean", True),
    ("CV Macro F1", "cv", "cv_f1_macro_mean", True),
    ("Student-test Top-1 Accuracy", "st", "top1_accuracy", True),
    ("Student-test Macro F1", "st", "f1_macro", True),
    ("Student-test Log Loss", "st", "log_loss", False),
    ("Route Continuity - 1-hop", "st", "route_continuity_1hop", True),
    ("Route Continuity - 2-hop", "st", "route_continuity_2hop", True),
    ("Candidate Miss Rate", "st", "candidate_miss_rate", True),
    ("Candidate-row Accuracy", "st", "accuracy", True),
    ("Macro Precision", "st", "precision_macro", True),
    ("Macro Recall", "st", "recall_macro", True),
    ("Positive-class F1", "st", "f1_pos", True),
    ("Positive-class Precision", "st", "precision_pos", True),
    ("Positive-class Recall", "st", "recall_pos", True),
]

table = {}
for name in names:
    r = json.loads((ROOT / "models" / name / "metrics.json").read_text())
    col = {}
    for label, src, key, pct in ROWS:
        v = r[key] if src == "cv" else r["student_test"].get(key)
        col[label] = None if v is None else (f"{100 * v:.3f}%" if key == "candidate_miss_rate" else
                                             f"{100 * v:.2f}%" if pct else f"{v:.4f}")
    col["CV Top-1 std (over 5 folds)"] = f"{100 * r['cv_top1_std']:.2f}%"
    col["Train Top-1 (overfitting check)"] = f"{100 * r['cv_train_top1_mean']:.2f}%"
    col["Train time (s, final refit)"] = f"{r['train_seconds']:.1f}"
    col["Inference (ms / 1000 candidate rows)"] = f"{r['inference_ms_per_1000_rows']:.2f}"
    table[name] = col

df = pd.DataFrame(table)
df.index.name = "Metric"
out = ROOT / "results" / "model_comparison" / "metrics_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out)
print(df.to_string())
