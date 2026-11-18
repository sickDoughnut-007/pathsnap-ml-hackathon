"""Generate visualizations from measured KNN and team model results."""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
KNN = HERE.parent
SEQUENCE = KNN / "sequence"
ITERATION2 = SEQUENCE / "iteration2"


def save(fig, name):
    fig.savefig(HERE / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def model_comparison():
    df = pd.read_csv(HERE / "model_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, mean_col, std_col, title, fmt in [
        (axes[0], "cv_top1_mean", "cv_top1_std", "Cross-validation Top-1", ".1%"),
        (axes[1], "cv_f1_macro_mean", "cv_f1_macro_std", "Cross-validation Macro F1", ".1%"),
        (axes[2], "student_top1", None, "Student-test Top-1", ".1%"),
    ]:
        y = np.arange(len(df))
        values = df[mean_col]
        err = df[std_col] if std_col else None
        ax.barh(y, values * 100, xerr=(err.values * 100 if err is not None else None),
                color=["#087b76" if m == "Distance-weighted KNN" else "#8da0b2" for m in df.model])
        ax.set_yticks(y, df.model)
        ax.set_xlim(
            82 if mean_col == "cv_top1_mean" or mean_col == "student_top1" else 91,
            98 if mean_col == "cv_top1_mean" or mean_col == "student_top1" else 94,
        )
        ax.set_xlabel("Score (%)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.25)
        for i, value in enumerate(values):
            ax.text(value * 100 + .05, i, f"{value:{fmt}}", va="center", fontsize=8)
    save(fig, "model_comparison_cv_student.png")


def macro_and_log_loss():
    df = pd.read_csv(HERE / "model_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for ax, cols, title in [
        (axes[0], ["cv_f1_macro_mean", "student_f1_macro"], "Macro F1"),
        (axes[1], ["cv_log_loss_mean", "student_log_loss"], "Log Loss"),
    ]:
        x = np.arange(len(df))
        for i, col in enumerate(cols):
            ax.plot(x, df[col], "o-", label=["Cross-validation", "Student test"][i],
                    color=["#087b76", "#984723"][i])
        ax.set_xticks(x, df.model, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    save(fig, "macro_f1_and_log_loss.png")


def confusion():
    metrics = json.loads((KNN / "metrics.json").read_text())
    matrix = np.array(metrics["student_test"]["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, f"{count:,}\n({normalized[i, j]:.1%})", ha="center", va="center",
                color="white" if normalized[i, j] > .6 else "black", fontsize=9)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Other", "Match"], yticklabels=["Other", "Match"],
           xlabel="Predicted candidate class", ylabel="Estimated reference class",
           title="Distance-weighted KNN student-test confusion")
    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    save(fig, "knn_confusion_matrix.png")


def decoder_accuracy():
    cv = pd.read_csv(KNN / "cv_folds.csv")
    folds = pd.read_csv(SEQUENCE / "validation_folds.csv")
    first_trial = folds.trial.nunique() and folds.trial.value_counts().index[0]
    best_trial = pd.read_csv(SEQUENCE / "trials.csv").mean_fold_top1.idxmax()
    folds = folds.query("trial == @best_trial")
    second = pd.read_csv(ITERATION2 / "folds.csv")
    second_mean = second.groupby("trial").top1.mean()
    best_second = second_mean.idxmax()
    second = second.query("trial == @best_second")
    first_trips = pd.read_csv(SEQUENCE / "student_trips.csv", dtype={"trip_id": str})
    second_trips = pd.read_csv(ITERATION2 / "student_trips.csv", dtype={"trip_id": str})
    trips = first_trips.merge(second_trips, on="trip_id", suffixes=("_first", "_second"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    x = np.arange(5)
    baseline = 100 * cv.top1_accuracy
    first = 100 * folds.top1
    second_values = 100 * second.groupby("fold").top1.mean()
    axes[0].plot(x, baseline, "o-", label="Pointwise KNN", color="#707780")
    axes[0].plot(x, first, "o-", label="First decoder", color="#087b76")
    axes[0].plot(x, second_values, "o-", label="Second decoder", color="#984723")
    axes[0].axhline(95, ls="--", lw=1, color="#ab581e", label="95% target")
    axes[0].set(xticks=x, xticklabels=[str(i + 1) for i in x],
                xlabel="Training fold", ylabel="Correct directed arc (%)",
                title="Cross-validation accuracy", ylim=(82, 100))
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8)

    x = np.arange(len(trips))
    first_trip = 100 * trips.correct_first / trips.eligible_first
    second_trip = 100 * trips.correct_second / trips.eligible_second
    axes[1].vlines(x, first_trip, second_trip, color="#94a1a7")
    axes[1].scatter(x, first_trip, color="#087b76", label="First decoder", zorder=3)
    axes[1].scatter(x, second_trip, color="#984723", label="Second decoder", zorder=3)
    axes[1].axhline(95, ls="--", lw=1, color="#ab581e", label="95% target")
    axes[1].set_xticks(x, trips.trip_id.str[-2:])
    axes[1].set(xlabel="Trip ID (last two digits)", ylabel="Correct directed arc (%)",
                title="Student-test trip accuracy", ylim=(88, 100))
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(fontsize=8)
    save(fig, "decoder_accuracy_comparison.png")


def tuning():
    first = pd.read_csv(SEQUENCE / "trials.csv")
    second = pd.read_csv(ITERATION2 / "trials.csv")
    first_best = first.groupby("trial").mean_fold_top1.mean().cummax()
    second_best = second.top1.cummax()
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    ax.plot(first.trial.drop_duplicates(), first.groupby("trial").mean_fold_top1.mean() * 100,
            alpha=.35, color="#087b76", label="First-decoder trials")
    ax.plot(second.trial, second.top1 * 100, alpha=.35, color="#984723",
            label="Second-decoder trials")
    ax.plot(first.trial.drop_duplicates(), first_best * 100, lw=2, color="#087b76",
            label="First-decoder running best")
    ax.plot(second.trial, second_best * 100, lw=2, color="#984723",
            label="Second-decoder running best")
    ax.set(xlabel="Trial", ylabel="Mean training-fold top-1 (%)",
           title="Route-decoder configuration search")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    save(fig, "decoder_trial_search.png")


def copy_map():
    source = KNN / "trip_00000052.html"
    if source.exists():
        shutil.copy2(source, HERE / "trip_00000052.html")


def main():
    model_comparison()
    macro_and_log_loss()
    confusion()
    decoder_accuracy()
    tuning()
    copy_map()


if __name__ == "__main__":
    main()
