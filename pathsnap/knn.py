"""Distance-weighted KNN for binary candidate scoring in a PCA feature space."""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class TripWeightedKNN(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=25, workers=4):
        self.n_neighbors = n_neighbors
        self.workers = workers

    def fit(self, X, y, sample_weight):
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Expected binary candidate labels 0 and 1")
        self.coordinates_ = np.ascontiguousarray(X, dtype=np.float32)
        self.labels_ = np.asarray(y, dtype=np.uint8)
        self.trip_weights_ = np.asarray(sample_weight, dtype=np.float32)
        self.tree_ = cKDTree(self.coordinates_, leafsize=32)
        return self

    def __getstate__(self):
        state = super().__getstate__()
        # Rebuild the search tree on load rather than also storing its float64 copy.
        state.pop("tree_", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if hasattr(self, "coordinates_"):
            self.tree_ = cKDTree(self.coordinates_, leafsize=32)

    def predict_proba(self, X):
        check_is_fitted(self, "tree_")
        p = np.empty(len(X))
        k = min(self.n_neighbors, len(self.coordinates_))
        for start in range(0, len(X), 8192):
            batch = np.asarray(X[start:start + 8192], dtype=np.float32)
            dist, idx = self.tree_.query(batch, k=list(range(1, k + 1)), workers=self.workers)
            exact = dist == 0
            inverse = np.divide(1, dist, out=np.zeros_like(dist), where=~exact)
            weights = np.where(exact.any(axis=1, keepdims=True), exact, inverse) * self.trip_weights_[idx]
            share = (weights * self.labels_[idx]).sum(axis=1) / weights.sum(axis=1)
            # Half a vote per class keeps pure neighbourhoods away from 0 and 1.
            p[start:start + len(batch)] = (k * share + 0.5) / (k + 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
