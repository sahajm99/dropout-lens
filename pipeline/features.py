"""Feature sets, targets, the one train/test split and the logistic formula (DECISIONS S4 to S6, R3, R5).

Two feature sets are modelled side by side. "At enrolment" holds what the institution knew at
admission (plus the two administrative flags R2 keeps). "After first semester" adds the
first-semester unit counts and grade, which leak the outcome and are marked so everywhere.
The logistic models use only the categorical variables with pinned references (S6); the tree
models add the numeric columns and attendance (R3, R5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.constants import OUTCOMES, REFERENCE, SEED, VARIABLE_LABELS

# Logistic covariates at enrolment; attendance is excluded because it is exactly a function of course (R3).
CAT_ENROLMENT = [
    "age_band",
    "gender",
    "scholarship",
    "tuition",
    "debtor",
    "displaced",
    "international",
    "special_needs",
    "application_mode",
    "course",
    "marital",
    "previous_qualification",
]
NUM_ENROLMENT = ["age", "app_order", "unemployment", "inflation", "gdp"]
TREE_EXTRA_ENROLMENT = ["attendance"]
CAT_AFTER_S1 = CAT_ENROLMENT + ["approval_band_s1"]
NUM_AFTER_S1 = NUM_ENROLMENT + [
    "s1_credited",
    "s1_enrolled",
    "s1_evaluations",
    "s1_approved",
    "s1_grade",
    "s1_without_eval",
]

FEATURE_SETS: dict[str, dict] = {
    "enrolment": {
        "label": "At enrolment",
        "leaks": False,
        "cat": CAT_ENROLMENT,
        "num": NUM_ENROLMENT,
        "tree_cat": CAT_ENROLMENT + TREE_EXTRA_ENROLMENT,
    },
    "after_s1": {
        "label": "After first semester",
        "leaks": True,
        "cat": CAT_AFTER_S1,
        "num": NUM_AFTER_S1,
        "tree_cat": CAT_AFTER_S1 + TREE_EXTRA_ENROLMENT,
    },
}

# MNLogit base class is the lowest code, so Graduate is 0 (S6).
OUTCOME_CODE: dict[str, int] = {"Graduate": 0, "Dropout": 1, "Enrolled": 2}

# `column` is the label column on the frame; `formula_target` is the integer column the
# statsmodels formula regresses; `classes` is the order of every probability matrix.
TARGETS: dict[str, dict] = {
    "binary": {
        "label": "Dropout or not",
        "classes": ["Not dropout", "Dropout"],
        "positive": "Dropout",
        "column": "dropout",
        "formula_target": "dropout",
    },
    "multiclass": {
        "label": "Dropout, Enrolled or Graduate",
        "classes": list(OUTCOMES),
        "column": "outcome",
        "formula_target": "outcome_code",
    },
}

# Labels for the numeric tree-model features; the macro triple identifies the intake cohort (R5).
NUMERIC_LABELS: dict[str, str] = {
    "age": "Age at enrolment (years)",
    "app_order": "Application order",
    "unemployment": "Unemployment rate at enrolment (cohort marker)",
    "inflation": "Inflation rate at enrolment (cohort marker)",
    "gdp": "GDP at enrolment (cohort marker)",
    "s1_credited": "First-semester units credited",
    "s1_enrolled": "First-semester units enrolled",
    "s1_evaluations": "First-semester evaluations",
    "s1_approved": "First-semester units approved",
    "s1_grade": "First-semester grade",
    "s1_without_eval": "First-semester units without evaluation",
}


def variable_label(name: str) -> str:
    if name in VARIABLE_LABELS:
        return VARIABLE_LABELS[name]
    return NUMERIC_LABELS[name]


def split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Positional (train, test) index arrays, each sorted ascending: stratified 80/20 on the outcome (S4)."""
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=df["outcome"], random_state=SEED
    )
    return np.sort(train_idx), np.sort(test_idx)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """A fresh copy with a 0-based positional index and the integer `outcome_code` column."""
    out = frame.copy().reset_index(drop=True)
    out["outcome_code"] = out["outcome"].astype(str).map(OUTCOME_CODE).astype(int)
    return out


def formula(target: str, cats: list[str]) -> str:
    return f"{target} ~ " + " + ".join(f"C({c}, Treatment('{REFERENCE[c]}'))" for c in cats)


def y_codes(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Integer class codes in TARGETS[target]["classes"] order."""
    if target == "binary":
        return frame["dropout"].to_numpy(dtype=int)
    if list(frame["outcome"].cat.categories) != OUTCOMES:
        raise ValueError("outcome categories are not in OUTCOMES order")
    return frame["outcome"].cat.codes.to_numpy(dtype=int)


def y_labels(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Class labels (strings from TARGETS[target]["classes"]) for every row."""
    classes = np.asarray(TARGETS[target]["classes"], dtype=object)
    return classes[y_codes(frame, target)]
