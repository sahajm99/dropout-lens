"""Browser estimator tables (DECISIONS S6, S11, W7, W8, R4).

`estimator.json` carries the enrolment binary Logit's coefficients and their covariance so the
browser can compute, for a chosen profile, the linear predictor, its delta-method standard error
and the transformed interval. `evaluate` is the Python twin of that browser function, and
`build_parity` writes the 100 seeded profiles the vitest parity test checks it against.

The coefficient table is the same fit (`bundle.binary_enrolment`, training rows only) that
supplies the odds ratios, rounded the same way, so the forest and the panel cannot disagree.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.constants import LEVELS, REFERENCE, SEED, VARIABLE_LABELS, Z95
from pipeline.features import CAT_ENROLMENT
from pipeline.io import now_iso, write_json
from pipeline.models import TERM_RE, FitBundle, fit_all
from pipeline.stats import round4, round6, round10

ESTIMATOR_MODEL = "binary_enrolment"
PARITY_ROWS = 100

# The eight inputs the panel exposes (W7) and the covariates held at their reference level (R2).
ESTIMATOR_INPUTS = [
    "age_band",
    "gender",
    "scholarship",
    "displaced",
    "application_mode",
    "course",
    "previous_qualification",
    "marital",
]
HELD_AT_REFERENCE = ["tuition", "debtor", "international", "special_needs"]

if set(ESTIMATOR_INPUTS) & set(HELD_AT_REFERENCE):
    raise ValueError("an estimator input cannot also be held at reference")
if set(ESTIMATOR_INPUTS) | set(HELD_AT_REFERENCE) != set(CAT_ENROLMENT):
    raise ValueError("estimator inputs plus held-at-reference covariates must be exactly CAT_ENROLMENT")


def sigmoid(x: float) -> float:
    """Logistic function in the overflow-safe form; equal to 1 / (1 + exp(-x))."""
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def build_estimator(bundle: FitBundle) -> dict:
    res = bundle.binary_enrolment
    names = list(res.params.index)
    if not names or names[0] != "Intercept":
        raise RuntimeError("the estimator model's first parameter is not the intercept")
    params = np.asarray(res.params, dtype=float)
    cov_df = res.cov_params()
    if list(cov_df.index) != names or list(cov_df.columns) != names:
        raise RuntimeError("cov_params() is not indexed in params order")
    cov = np.asarray(cov_df, dtype=float)
    if not (np.isfinite(params).all() and np.isfinite(cov).all()):
        raise RuntimeError("non-finite estimator coefficient or covariance entry")

    terms = ["Intercept"]
    term_index: dict[str, dict[str, int]] = {c: {} for c in CAT_ENROLMENT}
    coefficients: dict[str, dict[str, float]] = {c: {lvl: 0.0 for lvl in LEVELS[c]} for c in CAT_ENROLMENT}
    for i, name in enumerate(names[1:], start=1):
        m = TERM_RE.match(name)
        if m is None:
            raise RuntimeError(f"unparseable statsmodels term name: {name!r}")
        variable, reference, level = m.groups()
        if (
            variable not in term_index
            or level not in LEVELS[variable]
            or reference != REFERENCE[variable]
            or level == reference
        ):
            raise RuntimeError(f"unexpected term {name!r}")
        terms.append(f"{variable}={level}")
        term_index[variable][level] = i
        coefficients[variable][level] = round4(params[i])
    for c in CAT_ENROLMENT:
        expected = sorted(lvl for lvl in LEVELS[c] if lvl != REFERENCE[c])
        if sorted(term_index[c]) != expected:
            raise RuntimeError(f"{c}: fitted levels {sorted(term_index[c])} are not {expected}")

    return {
        "schema": "estimator.v1",
        "id": "estimator",
        "generated_at": now_iso(),
        "model": ESTIMATOR_MODEL,
        "fitted_on": "train",
        "z": Z95,
        "intercept": round4(params[0]),
        "base_rate": round6(bundle.train["dropout"].mean()),
        "n_obs": int(res.nobs),
        "inputs": [
            {"variable": v, "label": VARIABLE_LABELS[v], "levels": list(LEVELS[v]), "default": REFERENCE[v]}
            for v in ESTIMATOR_INPUTS
        ],
        "held_at_reference": [
            {"variable": v, "label": VARIABLE_LABELS[v], "level": REFERENCE[v]} for v in HELD_AT_REFERENCE
        ],
        "coefficients": coefficients,
        "terms": terms,
        "term_index": term_index,
        "cov": [[round10(v) for v in row] for row in cov],
    }


def evaluate(est: dict, chosen: dict[str, str]) -> tuple[float, float, float]:
    """(p, lo, hi) for one profile: the browser function's twin, on the published table.

    `chosen` must name every estimator input and nothing else; a level outside the input's
    levels raises. The held-at-reference covariates contribute nothing, as in the browser.
    """
    inputs = {i["variable"]: i for i in est["inputs"]}
    if set(chosen) != set(inputs):
        raise ValueError(f"chosen must name exactly {list(inputs)}, got {sorted(chosen)}")
    k = len(est["terms"])
    beta = np.zeros(k)
    beta[0] = float(est["intercept"])
    for variable, index in est["term_index"].items():
        for level, i in index.items():
            beta[i] = float(est["coefficients"][variable][level])
    x = np.zeros(k)
    x[0] = 1.0
    for variable, spec in inputs.items():
        level = chosen[variable]
        if level not in spec["levels"]:
            raise ValueError(f"{variable}: {level!r} is not one of {spec['levels']}")
        i = est["term_index"][variable].get(level)
        if i is not None:
            x[i] = 1.0
    cov = np.asarray(est["cov"], dtype=float)
    logit = float(x @ beta)
    variance = float(x @ cov @ x)
    if variance < 0:
        raise ValueError("negative variance from the published covariance")
    se = math.sqrt(variance)
    z = float(est["z"])
    return sigmoid(logit), sigmoid(logit - z * se), sigmoid(logit + z * se)


def build_parity(estimator: dict, seed: int = SEED, n: int = PARITY_ROWS) -> dict:
    """`n` seeded profiles (a uniform random level per input, inputs in order) with their (p, lo, hi)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        chosen = {
            spec["variable"]: spec["levels"][int(rng.integers(len(spec["levels"])))]
            for spec in estimator["inputs"]
        }
        p, lo, hi = evaluate(estimator, chosen)
        rows.append({"inputs": chosen, "p": round10(p), "lo": round10(lo), "hi": round10(hi)})
    return {
        "schema": "estimator_parity.v1",
        "id": "estimator_parity",
        "generated_at": now_iso(),
        "seed": seed,
        "rows": rows,
    }


def write_estimator(df: pd.DataFrame) -> list[Path]:
    estimator = build_estimator(fit_all(df))
    return [
        write_json("estimator.json", estimator),
        write_json("estimator_parity.json", build_parity(estimator)),
    ]
