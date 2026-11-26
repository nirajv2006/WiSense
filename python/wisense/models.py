"""Candidate models. Random Forest is the main model; the rest are baselines
that show whether the extra complexity is actually earning its keep."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import FEATURE_NAMES


class VarianceThreshold(BaseEstimator, ClassifierMixin):
    """The method the ESP32 firmware uses today: two thresholds on the variance
    of the mean amplitude. Thresholds are learned from the training folds
    (grid search over quantiles, maximising macro-F1), so the comparison is fair."""

    def __init__(self, feature: str = "meanamp_var"):
        self.feature = feature

    def fit(self, X, y):
        self.classes_ = np.array(config.CLASSES)
        x = np.asarray(X)[:, FEATURE_NAMES.index(self.feature)]
        y = np.asarray(y)
        qs = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 49)))
        best = (-1, qs[0], qs[-1])
        for i, a in enumerate(qs):
            for b in qs[i:]:
                score = f1_score(y, self._apply(x, a, b), average="macro")
                if score > best[0]:
                    best = (score, a, b)
        _, self.t_low_, self.t_high_ = best
        return self

    def _apply(self, x, a, b):
        lo, mid, hi = config.CLASSES
        return np.where(x < a, lo, np.where(x < b, mid, hi))

    def predict(self, X):
        return self._apply(np.asarray(X)[:, FEATURE_NAMES.index(self.feature)], self.t_low_, self.t_high_)

    def predict_proba(self, X):
        pred = self.predict(X)
        return (pred[:, None] == self.classes_[None, :]).astype(float)


def make_models(seed: int = 0) -> dict:
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Variance threshold": VarianceThreshold(),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "k-NN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Decision tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                                class_weight="balanced", random_state=seed),
        "Random forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=seed),
    }


MAIN_MODEL = "Random forest"
