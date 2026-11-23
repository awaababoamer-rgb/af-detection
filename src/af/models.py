"""Baseline classifiers and evaluation helpers."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from af.config import CLASS_ORDER, RANDOM_STATE


def make_logistic_regression(C: float = 1.0) -> Pipeline:
    # Imputer inside the pipeline so its medians are learned from training
    # data only (fitting it on all data would leak test information).
    # add_indicator: "this feature was missing" is itself informative, e.g.
    # sample entropy is undefined mostly for short/noisy recordings.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),  # LR is sensitive to feature scale
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",  # upweight rare AF/Noisy classes
                    max_iter=5000,
                ),
            ),
        ]
    )


def make_random_forest(n_estimators: int = 300, min_samples_leaf: int = 2, max_features="sqrt") -> Pipeline:
    # Trees don't need scaling. "balanced_subsample" recomputes class weights
    # for each tree's bootstrap sample.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    """Per-class F1, macro F1 and confusion matrix (rows = true, cols = predicted)."""
    per_class = f1_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    return {
        "per_class_f1": dict(zip(CLASS_ORDER, np.round(per_class, 4).tolist())),
        "macro_f1": round(float(np.mean(per_class)), 4),
        # The official 2017 challenge score averaged F1 over N, A, O only (not Noisy).
        "challenge_f1_NAO": round(float(np.mean(per_class[:3])), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }
