import json

import pytest

import pipeline.io as io
from pipeline.hypothesis import build_hypothesis_tests, write_hypothesis_tests

CATEGORICAL_IDS = [
    "gender", "scholarship", "tuition", "debtor", "age_band", "application_mode", "course",
    "attendance", "displaced", "marital", "previous_qualification", "international",
    "special_needs", "approval_band_s1",
]


@pytest.fixture(scope="module")
def ht(df):
    return build_hypothesis_tests(df)


def test_family_of_20_with_holm_never_below_raw(df, tmp_path, monkeypatch):
    monkeypatch.setattr(io, "OUT_DIR", tmp_path)
    d = json.loads(write_hypothesis_tests(df).read_text(encoding="utf-8"))
    assert d["family_size"] == 20 and d["adjustment"] == "holm" and d["alpha"] == 0.05
    assert [c["id"] for c in d["categorical"]] == CATEGORICAL_IDS
    assert [e["id"] for e in d["numeric"]] == ["age", "grade_s1", "grade_s2"]
    family = d["categorical"] + [e[t] for e in d["numeric"] for t in ("anova", "kruskal")]
    assert len(family) == 20
    for slot in family:
        assert 0.0 <= slot["p_raw"] <= slot["p_holm"] <= 1.0


def test_approval_band_s1_has_the_largest_corrected_v(ht):
    cats = ht["categorical"]
    top = max(cats, key=lambda c: c["cramers_v_corrected"])
    assert top["id"] == "approval_band_s1" and top["post_enrolment"] is True
    assert all(0.0 <= c["cramers_v_corrected"] <= 1.0 for c in cats)


def test_gender_has_two_levels_and_two_df(ht):
    gender = next(c for c in ht["categorical"] if c["id"] == "gender")
    assert gender["levels"] == 2 and gender["df"] == 2 and gender["n"] == 4424


def test_grade_s1_excludes_718_zero_grades(ht):
    grade_s1 = next(e for e in ht["numeric"] if e["id"] == "grade_s1")
    assert grade_s1["n_excluded"] == 718 and grade_s1["n"] == 4424 - 718
    assert sum(row["n"] for row in grade_s1["by_outcome"]) == grade_s1["n"]
    assert grade_s1["anova"]["df_within"] == grade_s1["n"] - 3
    age = next(e for e in ht["numeric"] if e["id"] == "age")
    assert age["n_excluded"] == 0 and age["n"] == 4424
