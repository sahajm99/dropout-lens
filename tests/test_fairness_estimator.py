import numpy as np
import pandas as pd
import pytest

from pipeline.constants import LEVELS, REFERENCE
from pipeline.estimator import (
    ESTIMATOR_INPUTS,
    HELD_AT_REFERENCE,
    build_estimator,
    build_parity,
    evaluate,
    sigmoid,
)
from pipeline.fairness import FAIRNESS_GROUPS, build_fairness
from pipeline.features import CAT_ENROLMENT
from pipeline.models import fit_all


@pytest.fixture(scope="module")
def bundle(df):
    return fit_all(df)


@pytest.fixture(scope="module")
def est(bundle):
    return build_estimator(bundle)


def test_fairness_two_results_with_bracketed_rates(bundle):
    obj = build_fairness(bundle)
    assert obj["threshold"] == 0.5 and obj["suppress_below"] == 30
    assert [r["feature_set"] for r in obj["results"]] == ["enrolment", "after_s1"]
    for r in obj["results"]:
        assert r["model"] == bundle.best["binary"][r["feature_set"]] and r["n_test"] == 885
        assert [g["id"] for g in r["groups"]] == FAIRNESS_GROUPS == ["gender", "scholarship", "age_band", "tuition"]
        for g in r["groups"]:
            assert [lvl["level"] for lvl in g["levels"]] == LEVELS[g["id"]]
            assert sum(lvl["n"] for lvl in g["levels"]) == 885, g["id"]
            for lvl in g["levels"]:
                assert lvl["n_positive"] + lvl["n_negative"] == lvl["n"]
                for rate, denominator in (("fnr", lvl["n_positive"]), ("fpr", lvl["n_negative"])):
                    cell = lvl[rate]
                    if cell is None:
                        assert lvl[f"suppressed_{rate}"] and denominator < 30, (g["id"], lvl["level"], rate)
                    else:
                        assert not lvl[f"suppressed_{rate}"] and denominator >= 30, (g["id"], lvl["level"], rate)
                        assert 0 <= cell["lo"] <= cell["p"] <= cell["hi"] <= 1, (g["id"], lvl["level"], rate)


def test_estimator_terms_cov_index_and_reference_profile(est):
    assert est["terms"][0] == "Intercept" and est["n_obs"] == 3539
    k = len(est["terms"])
    cov = np.asarray(est["cov"], dtype=float)
    assert cov.shape == (k, k) and len(est["cov"]) == k
    assert np.abs(cov - cov.T).max() < 1e-9
    assert [i["variable"] for i in est["inputs"]] == ESTIMATOR_INPUTS
    assert [h["variable"] for h in est["held_at_reference"]] == HELD_AT_REFERENCE
    assert all(h["level"] == REFERENCE[h["variable"]] for h in est["held_at_reference"])
    for spec in est["inputs"]:
        variable = spec["variable"]
        assert spec["default"] == REFERENCE[variable] and spec["levels"] == LEVELS[variable]
        for level in spec["levels"]:
            if level == spec["default"]:
                assert level not in est["term_index"][variable]
                assert est["coefficients"][variable][level] == 0.0
            else:
                index = est["term_index"][variable][level]
                assert est["terms"][index] == f"{variable}={level}"
    defaults = {spec["variable"]: spec["default"] for spec in est["inputs"]}
    p, lo, hi = evaluate(est, defaults)
    assert abs(p - sigmoid(est["intercept"])) < 1e-9
    assert 0 < lo <= p <= hi < 1


def test_parity_rows_bracket_and_reproduce_the_fitted_model(est, bundle):
    parity = build_parity(est)
    rows = parity["rows"]
    assert len(rows) == 100 and parity["seed"] == 20260912
    for row in rows:
        assert list(row["inputs"]) == ESTIMATOR_INPUTS
        assert 0 <= row["lo"] <= row["p"] <= row["hi"] <= 1
    assert build_parity(est)["rows"] == rows
    # The published table must reproduce the statsmodels fit itself (coefficients are rounded to
    # 4 decimals, which moves a probability by at most a few 1e-5), so a misplaced term or a
    # permuted covariance entry cannot pass unnoticed.
    columns = {v: [row["inputs"][v] for row in rows] for v in ESTIMATOR_INPUTS}
    columns.update({v: [REFERENCE[v]] * len(rows) for v in HELD_AT_REFERENCE})
    frame = pd.DataFrame({v: pd.Categorical(columns[v], categories=LEVELS[v]) for v in CAT_ENROLMENT})
    linear = bundle.binary_enrolment.get_prediction(frame, which="linear")
    logit = np.asarray(linear.predicted, dtype=float)
    se = np.asarray(linear.se, dtype=float)
    z = est["z"]
    for key, model_values in (("p", logit), ("lo", logit - z * se), ("hi", logit + z * se)):
        ours = np.array([row[key] for row in rows])
        theirs = 1 / (1 + np.exp(-model_values))
        assert np.abs(ours - theirs).max() < 5e-4, key
