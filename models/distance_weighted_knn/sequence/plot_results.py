"""Plot score changes by whole training fold and student-testing trip."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    trials = pd.read_csv(HERE / "trials.csv")
    best = trials.loc[trials.mean_fold_top1.idxmax(), "trial"]
    folds = pd.read_csv(HERE / "validation_folds.csv").query("trial == @best")
    trips = pd.read_csv(HERE / "student_trips.csv", dtype={"trip_id": str})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, frame, labels, title in [
        (axes[0], folds, folds.fold + 1, "Training folds used for selection"),
        (axes[1], trips, trips.trip_id.str[-2:], "Student-test trips"),
    ]:
        x = np.arange(len(frame))
        baseline = 100 * frame.baseline_correct / frame.eligible
        sequence = 100 * frame.correct / frame.eligible
        ax.vlines(x, baseline, sequence, color="#94a1a7")
        ax.scatter(x, baseline, color="#707780", label="Pointwise KNN", zorder=3)
        ax.scatter(x, sequence, color="#087b76", label="KNN + decoder", zorder=3)
        ax.axhline(95, ls="--", lw=1, color="#ab581e", label="95% target")
        ax.set(xticks=x, xticklabels=labels, ylim=(65, 101), title=title)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Correct directed arc (% of eligible GPS points)")
    axes[0].set_xlabel("Fold")
    axes[1].set_xlabel("Trip ID (last two digits)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.savefig(HERE / "accuracy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
