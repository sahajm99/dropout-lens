"""Hypothesis tests writer (DECISIONS S2, S3).

Chi-square test of independence for every categorical predictor against the three-way
outcome, with Cramer's V (plain and Bergsma 2013 bias-corrected); one-way ANOVA and
Kruskal-Wallis for every numeric predictor, with eta squared and epsilon squared; Holm
adjustment across the whole family of tests. One output file, `hypothesis_tests.json`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from pipeline.constants import GROUPS, LEVELS, OUTCOMES
from pipeline.io import now_iso, write_json
from pipeline.stats import cramers_v, epsilon_squared, eta_squared, holm, mean_ci, round4, round6, sig3

ALPHA = 0.05

# Numeric predictors tested across the three outcomes. A semester grade recorded as 0 means
# no evaluated units, a missing mark rather than a mark, so those rows are excluded and
# counted (S3). Age is tested over every row.
NUMERICS: list[dict] = [
    {"id": "age", "label": "Age at enrolment", "column": "age", "post_enrolment": False, "exclude_zero": False},
    {"id": "grade_s1", "label": "First-semester grade", "column": "s1_grade", "post_enrolment": True, "exclude_zero": True},
    {"id": "grade_s2", "label": "Second-semester grade", "column": "s2_grade", "post_enrolment": True, "exclude_zero": True},
]


def categorical_groups() -> list[dict]:
    """The 14 GROUPS entries in contract order: pre-enrolment groups first, post-enrolment last."""
    return [g for g in GROUPS if not g["post_enrolment"]] + [g for g in GROUPS if g["post_enrolment"]]


def outcome_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Level x outcome counts with every level in LEVELS order, then all-zero levels dropped.

    `dropna=False` is how pandas crosstab spells observed=False (it forwards
    observed=dropna to pivot_table; there is no `observed` argument), so an
    unobserved level appears as a zero row and is then dropped, as the plan asks.
    """
    table = pd.crosstab(df[column], df["outcome"], dropna=False)
    table = table.reindex(index=LEVELS[column], columns=OUTCOMES, fill_value=0)
    return table.loc[table.sum(axis=1) > 0]


def _chi_square(df: pd.DataFrame, group: dict) -> dict:
    counts = outcome_table(df, group["column"]).to_numpy(dtype=float)
    statistic, p_raw, dof, expected = sps.chi2_contingency(counts, correction=False)
    v, v_corrected = cramers_v(counts)
    return {
        "id": group["id"],
        "label": group["label"],
        "post_enrolment": bool(group["post_enrolment"]),
        "test": "chi_square",
        "n": int(counts.sum()),
        "levels": int(counts.shape[0]),
        "df": int(dof),
        "statistic": round4(statistic),
        "p_raw": float(p_raw),  # unrounded until Holm has run over the family
        "p_holm": None,
        "cramers_v": round6(v),
        "cramers_v_corrected": round6(v_corrected),
        "min_expected": round4(expected.min()),
    }


def _summary(outcome: str, values: np.ndarray) -> dict:
    mean, lo, hi = mean_ci(values)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "outcome": outcome,
        "n": int(len(values)),
        "mean": round4(mean),
        "mean_lo": round4(lo),
        "mean_hi": round4(hi),
        "min": round4(values.min()),
        "q1": round4(q1),
        "median": round4(median),
        "q3": round4(q3),
        "max": round4(values.max()),
    }


def _numeric(df: pd.DataFrame, spec: dict) -> dict:
    column = spec["column"]
    sub = df[df[column] > 0] if spec["exclude_zero"] else df
    groups = [sub.loc[sub["outcome"] == outcome, column].to_numpy(dtype=float) for outcome in OUTCOMES]
    n = int(sum(len(g) for g in groups))
    k = len(groups)
    f_stat, f_p = sps.f_oneway(*groups)
    h_stat, h_p = sps.kruskal(*groups)
    return {
        "id": spec["id"],
        "label": spec["label"],
        "post_enrolment": bool(spec["post_enrolment"]),
        "n": n,
        "n_excluded": int(len(df) - len(sub)),
        "anova": {
            "statistic": round4(f_stat),
            "df_between": k - 1,
            "df_within": n - k,
            "p_raw": float(f_p),
            "p_holm": None,
            "eta_squared": round6(eta_squared(groups)),
        },
        "kruskal": {
            "statistic": round4(h_stat),
            "df": k - 1,
            "p_raw": float(h_p),
            "p_holm": None,
            "epsilon_squared": round6(epsilon_squared(float(h_stat), n)),
        },
        "by_outcome": [_summary(outcome, values) for outcome, values in zip(OUTCOMES, groups)],
    }


def build_hypothesis_tests(df: pd.DataFrame) -> dict:
    categorical = [_chi_square(df, group) for group in categorical_groups()]
    numeric = [_numeric(df, spec) for spec in NUMERICS]
    # The Holm family in contract order: every categorical test, then ANOVA and
    # Kruskal-Wallis for each numeric predictor in turn. Adjusted on the raw values,
    # rounded afterwards, so p_holm >= p_raw survives the rounding.
    family = categorical + [entry[test] for entry in numeric for test in ("anova", "kruskal")]
    adjusted = holm([slot["p_raw"] for slot in family])
    for slot, p_holm in zip(family, adjusted):
        slot["p_raw"] = sig3(slot["p_raw"])
        slot["p_holm"] = sig3(p_holm)
    return {
        "schema": "hypothesis_tests.v1",
        "id": "hypothesis_tests",
        "generated_at": now_iso(),
        "family_size": len(family),
        "adjustment": "holm",
        "alpha": ALPHA,
        "categorical": categorical,
        "numeric": numeric,
    }


def write_hypothesis_tests(df: pd.DataFrame) -> Path:
    return write_json("hypothesis_tests.json", build_hypothesis_tests(df))
