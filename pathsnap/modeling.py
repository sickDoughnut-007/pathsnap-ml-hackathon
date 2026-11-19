"""Shared model-training helpers used by the model notebooks and the tuning notebook.

Everything here uses the shared dataset (pathsnap.dataset), the precomputed trip-level folds
and the shared metrics (pathsnap.evaluate), so results from different team members are
directly comparable. Output files follow the same schema as models/_template_train.py.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dataset import FEATURES, load_candidates, load_points, model_rows
from .evaluate import observation_metrics, route_continuity, top1_predictions
from .io import REPO_ROOT, load_config, load_trip

SEED = 42


# ----------------------------------------------------------------------------- data
def subsample_by_fold(df, rows_per_fold, seed=SEED):
    """Random row subsample inside each CV fold (keeps every trip and the fold structure)."""
    if rows_per_fold is None:
        return df.reset_index(drop=True)
    parts = []
    for f, g in df.groupby("fold", sort=True):
        parts.append(g.sample(n=min(len(g), rows_per_fold), random_state=seed + int(f)))
    return pd.concat(parts).sort_index().reset_index(drop=True)


def load_data():
    """Returns the full labelled training rows, student-test rows, and both points tables.
    Subsampling (for speed) is applied only to the rows a model is FITTED on, never to the
    rows it is evaluated on, so every model is validated on exactly the same observations."""
    tr = model_rows(load_candidates("train"))
    te = model_rows(load_candidates("student_test"))
    return tr, te, load_points("train"), load_points("student_test")


# ----------------------------------------------------------------------------- CV + test
def cross_validate(make_model, fit_fn, tr, tr_pts, rows_per_fold=None, predict_fn=None, verbose=True):
    """Trip-level CV on the shared folds. fit_fn(model, train_df) -> fitted model.
    The training part may be subsampled (rows_per_fold per training fold); the validation
    fold is always used in full. Returns one row of shared metrics per fold."""
    predict_fn = predict_fn or (lambda m, X: m.predict_proba(X)[:, 1])
    rows = []
    for f in sorted(tr["fold"].unique()):
        a_full, v = tr[tr["fold"] != f], tr[tr["fold"] == f]
        a = subsample_by_fold(a_full, rows_per_fold)
        t0 = time.perf_counter()
        m = fit_fn(make_model(), a)
        fit_s = time.perf_counter() - t0
        s = predict_fn(m, v[FEATURES])
        r = observation_metrics(v, s, tr_pts[tr_pts["fold"] == f])
        r.update(fold=int(f), fit_seconds=fit_s,
                 train_top1=_train_top1(m, a_full, predict_fn))
        rows.append(r)
        if verbose:
            print(f"fold {f}: val top1={r['top1_accuracy']:.4f}  train top1={r['train_top1']:.4f}  "
                  f"f1_macro={r['f1_macro']:.4f}  log_loss={r['log_loss']:.4f}  ({fit_s:.0f}s)")
    return pd.DataFrame(rows)


def _train_top1(model, a_full, predict_fn, n_trips=20):
    """Top-1 on ALL candidate rows of 20 random training trips (full rows, so every observation
    keeps its positive candidate). Compared with validation Top-1 = over/underfitting evidence."""
    trips = a_full["trip_id"].unique()
    rng = np.random.default_rng(SEED)
    sub = a_full[a_full["trip_id"].isin(rng.choice(trips, size=min(len(trips), n_trips), replace=False))]
    s = predict_fn(model, sub[FEATURES])
    best = top1_predictions(sub, s)
    return float(best["label"].mean())


def fit_and_test(make_model, fit_fn, tr, te, te_pts, rows_per_fold=None, predict_fn=None, data_root=None):
    """Refit on all 80 training trips (optionally subsampled), score the 10 student-testing trips in full."""
    predict_fn = predict_fn or (lambda m, X: m.predict_proba(X)[:, 1])
    t0 = time.perf_counter()
    model = fit_fn(make_model(), subsample_by_fold(tr, rows_per_fold))
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = predict_fn(model, te[FEATURES])
    infer_ms = (time.perf_counter() - t0) / len(te) * 1e6
    test = observation_metrics(te, scores, te_pts)
    test.update(continuity(te, scores, data_root))
    return model, scores, test, train_s, infer_ms


def continuity(cands, scores, data_root=None):
    """Route continuity of the classifier-only Top-1 route (needs the raw .arcs files)."""
    try:
        cfg = load_config()
        arcs = {t: load_trip(cfg, "student_test", t, data_root, with_route=False).arcs
                for t in cands["trip_id"].unique()}
        best = top1_predictions(cands, scores)
        c1, _ = route_continuity(best, arcs, 1)
        c2, _ = route_continuity(best, arcs, 2)
        return {"route_continuity_1hop": c1, "route_continuity_2hop": c2}
    except FileNotFoundError:
        print("Raw data not found: route continuity skipped (set PATHSNAP_DATA_ROOT).")
        return {"route_continuity_1hop": None, "route_continuity_2hop": None}


# ----------------------------------------------------------------------------- saving
def save_run(out_dir, name, family, config, cv, test, train_s, infer_ms, model=None,
             rows_per_fold=None, extra=None):
    """Write metrics.json (template-compatible), cv_folds.csv and model.joblib."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "model": name, "algorithm_family": family, "config": config, "seed": SEED,
        "train_rows_per_fold": rows_per_fold,
        "cv_top1_mean": float(cv["top1_accuracy"].mean()), "cv_top1_std": float(cv["top1_accuracy"].std()),
        "cv_f1_macro_mean": float(cv["f1_macro"].mean()), "cv_f1_macro_std": float(cv["f1_macro"].std()),
        "cv_log_loss_mean": float(cv["log_loss"].mean()),
        "cv_train_top1_mean": float(cv["train_top1"].mean()) if "train_top1" in cv else None,
        "train_seconds": float(train_s), "inference_ms_per_1000_rows": float(infer_ms),
        "student_test": test,
    }
    if extra:
        res.update(extra)
    cv.to_csv(out / "cv_folds.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=_json_default))
    if model is not None:
        path = out / "model.joblib"
        joblib.dump(model, path, compress=3)
        mb = path.stat().st_size / 1e6
        print(f"saved {path} ({mb:.1f} MB)" + ("  WARNING: >95 MB, too big for GitHub" if mb > 95 else ""))
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------- MLP with real loss curves
class EpochMLP:
    """StandardScaler + sklearn MLPClassifier trained epoch by epoch.

    Records the training and validation log loss after every epoch (real epoch-wise
    history for the report) and early-stops on validation loss. The validation set is a
    random 10% of the TRAINING trips (never the CV validation fold or student-test).
    Defined here, not in a notebook, so the saved model can be loaded by the API.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                 batch_size=512, max_epochs=60, patience=5, es_trip_frac=0.1, seed=SEED):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.es_trip_frac = es_trip_frac
        self.seed = seed

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in ["hidden_layer_sizes", "alpha", "learning_rate_init",
                                               "batch_size", "max_epochs", "patience", "es_trip_frac", "seed"]}

    def __repr__(self):
        return "EpochMLP(" + ", ".join(f"{k}={v}" for k, v in self.get_params().items()) + ")"

    def fit_df(self, df):
        import copy
        from sklearn.metrics import log_loss
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(self.seed)
        trips = df["trip_id"].unique()
        es_trips = rng.choice(trips, size=max(1, int(round(len(trips) * self.es_trip_frac))), replace=False)
        es_mask = df["trip_id"].isin(es_trips).values
        Xf, yf = df.loc[~es_mask, FEATURES].values, df.loc[~es_mask, "label"].values
        Xv, yv = df.loc[es_mask, FEATURES].values, df.loc[es_mask, "label"].values
        self.scaler_ = StandardScaler().fit(Xf)
        Xf, Xv = self.scaler_.transform(Xf), self.scaler_.transform(Xv)
        mon = rng.choice(len(Xf), size=min(len(Xf), 100_000), replace=False)  # train-loss monitor sample

        self.net_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  learning_rate_init=self.learning_rate_init, batch_size=self.batch_size,
                                  solver="adam", random_state=self.seed)
        hist, best, best_state, wait = [], np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(Xf))
            self.net_.partial_fit(Xf[order], yf[order], classes=[0, 1])
            tl = log_loss(yf[mon], self.net_.predict_proba(Xf[mon])[:, 1], labels=[0, 1])
            vl = log_loss(yv, self.net_.predict_proba(Xv)[:, 1], labels=[0, 1])
            hist.append({"epoch": epoch, "train_loss": tl, "val_loss": vl})
            if vl < best - 1e-5:
                best, best_state, wait = vl, copy.deepcopy(self.net_), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_ = best_state
        self.history_ = pd.DataFrame(hist)
        self.best_epoch_ = int(self.history_.loc[self.history_["val_loss"].idxmin(), "epoch"])
        return self

    def predict_proba(self, X):
        X = X.values if hasattr(X, "values") else X
        return self.net_.predict_proba(self.scaler_.transform(X))


def plot_loss_history(hist, title, path=None, best_epoch=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist["epoch"], hist["train_loss"], "o-", ms=3, label="training loss")
    ax.plot(hist["epoch"], hist["val_loss"], "o-", ms=3, label="validation loss (held-out trips)")
    if best_epoch:
        ax.axvline(best_epoch, color="k", ls="--", lw=1, label=f"best epoch ({best_epoch})")
    ax.set(xlabel="epoch", ylabel="log loss", title=title)
    ax.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.show()


# ----------------------------------------------------------------------------- plots
def plot_confusion(cm, title, path=None, ax=None):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4, 3.6))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title, fontsize=10)
    if own:
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=130)
        plt.show()


def plot_cv_folds(cv, title, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(cv))
    ax.bar(x - 0.2, cv["train_top1"], 0.4, label="train Top-1", color="#90caf9")
    ax.bar(x + 0.2, cv["top1_accuracy"], 0.4, label="validation Top-1", color="#1565c0")
    ax.set_xticks(x, [f"fold {f}" for f in cv["fold"]])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.show()
