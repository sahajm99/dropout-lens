"""Test-set metrics (DECISIONS S8): the five-metric suite, percentile bootstrap intervals,
quantile-binned calibration and the confusion matrix.

`y_true` always holds values from `classes`; probabilities are the positive-class column for a
binary target (`classes[1]` is the positive class, predicted at 0.5) or an n x k matrix in
`classes` order for a multiclass target (predicted at the argmax).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from pipeline.constants import SEED
from pipeline.stats import round6

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]
THRESHOLD = 0.5


def _codes(y, classes: list) -> np.ndarray:
    """Map class values onto their positions in `classes`; every value must be a member."""
    codes = np.asarray(pd.Categorical(list(y), categories=list(classes)).codes, dtype=int)
    if (codes < 0).any():
        raise ValueError(f"y_true holds values outside classes {list(classes)}")
    return codes


def predict_class(proba: np.ndarray) -> np.ndarray:
    """Class codes from probabilities: 0.5 on a positive-class column, argmax on a matrix."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return (proba >= THRESHOLD).astype(int)
    if proba.shape[1] == 2:
        return (proba[:, 1] >= THRESHOLD).astype(int)
    return proba.argmax(axis=1)


def _suite(y: np.ndarray, proba: np.ndarray, k: int) -> dict[str, float | None]:
    y_pred = predict_class(proba)
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
    }
    if len(np.unique(y)) < k:
        # A ranking metric needs every class present; a degenerate resample gets no value.
        out["roc_auc"] = None
        out["pr_auc"] = None
    elif proba.ndim == 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    else:
        labels = list(range(k))
        out["roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels))
        out["pr_auc"] = float(np.mean([average_precision_score(y == c, proba[:, c]) for c in labels]))
    return out


def metric_suite(y_true: np.ndarray, proba: np.ndarray, classes: list) -> dict[str, float | None]:
    """accuracy, balanced_accuracy, macro_f1, roc_auc and pr_auc on one set of predictions."""
    y = _codes(y_true, classes)
    proba = np.asarray(proba, dtype=float)
    if len(proba) != len(y):
        raise ValueError("y_true and proba differ in length")
    return _suite(y, proba, len(classes))


def bootstrap_ci(y_true, proba, classes: list, n: int = 1000, seed: int = SEED) -> dict[str, dict]:
    """{"value", "lo", "hi"} per metric: the full-sample value with a 2.5/97.5 percentile interval
    over `n` resamples of the rows, drawn with `np.random.default_rng(seed).integers`."""
    y = _codes(y_true, classes)
    proba = np.asarray(proba, dtype=float)
    k = len(classes)
    value = _suite(y, proba, k)
    draws = np.random.default_rng(seed).integers(0, len(y), size=(n, len(y)))
    samples: dict[str, list[float]] = {m: [] for m in METRICS}
    for rows in draws:
        s = _suite(y[rows], proba[rows], k)
        for m in METRICS:
            if s[m] is not None:
                samples[m].append(s[m])
    out: dict[str, dict] = {}
    for m in METRICS:
        if value[m] is None or not samples[m]:
            out[m] = {"value": value[m], "lo": None, "hi": None}
            continue
        lo, hi = np.percentile(samples[m], [2.5, 97.5])
        out[m] = {"value": value[m], "lo": float(lo), "hi": float(hi)}
    return out


def calibration(y_true, p, bins: int = 10) -> dict:
    """Quantile-binned reliability points and the Brier score for a positive-class probability.

    Every row lands in exactly one bin (edges are quantiles of `p`; a value on an interior edge goes
    up); empty bins, which tied probabilities can produce, are skipped, so the `n` sum to len(y).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and p must be non-empty and the same length")
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    which = np.searchsorted(edges[1:-1], p, side="right")
    points = []
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        points.append({
            "mean_predicted": round6(p[mask].mean()),
            "fraction_positive": round6(y[mask].mean()),
            "n": int(mask.sum()),
        })
    return {"points": points, "brier": round6(np.mean((p - y) ** 2))}


def confusion(y_true, y_pred, classes: list) -> list[list[int]]:
    """Rows are true classes, columns predicted classes, both in `classes` order."""
    labels = list(range(len(classes)))
    matrix = confusion_matrix(_codes(y_true, classes), _codes(y_pred, classes), labels=labels)
    return [[int(v) for v in row] for row in matrix]
