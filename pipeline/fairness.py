"""Fairness slice (DECISIONS S9): error rates of the best binary model per feature set, by group.

For each feature set the predictor is `bundle.best["binary"][feature_set]` (chosen by CV macro F1,
never by the test score). On the test rows, a student is predicted to drop out when the Dropout
probability is at least 0.5 (the one threshold in `pipeline.metrics`). Per level of each group:
the false negative rate is FN over the actual dropouts, the false positive rate FP over the
actual non-dropouts, each with a Wilson interval and suppressed (null, flagged) when its
denominator is under 30.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.constants import LEVELS, SUPPRESS_BELOW, VARIABLE_LABELS
from pipeline.features import FEATURE_SETS, TARGETS, y_codes
from pipeline.io import now_iso, write_json
from pipeline.metrics import THRESHOLD, predict_class
from pipeline.models import FitBundle, fit_all
from pipeline.stats import share_cell

FAIRNESS_GROUPS = ["gender", "scholarship", "age_band", "tuition"]


def _level_row(level: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Counts and error rates for one group level; `actual` and `predicted` are 0/1 (1 = Dropout)."""
    positive = actual == 1
    n = int(len(actual))
    n_positive = int(positive.sum())
    n_negative = n - n_positive
    false_negatives = int((positive & (predicted == 0)).sum())
    false_positives = int((~positive & (predicted == 1)).sum())
    fnr = share_cell(false_negatives, n_positive)
    fpr = share_cell(false_positives, n_negative)
    return {
        "level": level,
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "fnr": fnr,
        "fpr": fpr,
        "suppressed_fnr": fnr is None,
        "suppressed_fpr": fpr is None,
    }


def build_fairness(bundle: FitBundle) -> dict:
    test = bundle.test
    actual = y_codes(test, "binary")
    results = []
    for feature_set, fs in FEATURE_SETS.items():
        pred = bundle.best_predictor("binary", feature_set)
        if pred.classes != TARGETS["binary"]["classes"]:
            raise RuntimeError(f"{pred.model} classes are not {TARGETS['binary']['classes']}")
        predicted = predict_class(pred.predict_proba(test))
        groups = []
        for group in FAIRNESS_GROUPS:
            levels = []
            for level in LEVELS[group]:
                mask = (test[group] == level).to_numpy()
                levels.append(_level_row(level, actual[mask], predicted[mask]))
            groups.append({"id": group, "label": VARIABLE_LABELS[group], "levels": levels})
        results.append({
            "feature_set": feature_set,
            "model": pred.model,
            "leaks": bool(fs["leaks"]),
            "n_test": int(len(test)),
            "groups": groups,
        })
    return {
        "schema": "fairness.v1",
        "id": "fairness",
        "generated_at": now_iso(),
        "target": "binary",
        "threshold": THRESHOLD,
        "ci": {"method": "wilson", "level": 0.95},
        "suppress_below": SUPPRESS_BELOW,
        "results": results,
    }


def write_fairness(df: pd.DataFrame) -> Path:
    return write_json("fairness.json", build_fairness(fit_all(df)))
