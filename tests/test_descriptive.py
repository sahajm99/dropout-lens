from pipeline.constants import LEVELS, OUTCOMES
from pipeline.descriptive import build_outcome_by_group, build_outcome_overall, largest_remainder


def test_tiles_are_largest_remainder_integers_summing_to_100(df):
    obj = build_outcome_overall(df)
    assert [r["outcome"] for r in obj["rows"]] == OUTCOMES
    assert [r["n"] for r in obj["rows"]] == [1421, 794, 2209]
    assert sum(obj["tiles"].values()) == 100
    assert obj["tiles"] == {"Dropout": 32, "Enrolled": 18, "Graduate": 50}
    # Ties go to the earlier outcome: three equal shares leave one unit, and it lands on Dropout.
    assert largest_remainder({"Dropout": 1, "Enrolled": 1, "Graduate": 1}) == {
        "Dropout": 34,
        "Enrolled": 33,
        "Graduate": 33,
    }


def test_course_group_suppresses_only_biofuel_and_every_group_sums_to_n(df):
    obj = build_outcome_by_group(df)
    assert [g["id"] for g in obj["groups"]] == [
        "gender", "scholarship", "tuition", "debtor", "age_band",
        "application_mode", "course", "attendance", "displaced", "approval_band_s1",
    ]
    course = next(g for g in obj["groups"] if g["id"] == "course")
    suppressed = [r for r in course["rows"] if r["suppressed"]]
    assert len(suppressed) == 1
    assert suppressed[0]["level"] == "Biofuel Production Technologies"
    assert suppressed[0]["n"] == 12 and suppressed[0]["shares"] is None
    for g in obj["groups"]:
        assert sum(r["n"] for r in g["rows"]) == 4424 == obj["n_total"], g["id"]
        assert [r["level"] for r in g["rows"]] == g["levels"] == LEVELS[g["id"]], g["id"]
        assert g["post_enrolment"] is (g["id"] == "approval_band_s1"), g["id"]


def test_shares_are_ordered_wilson_cells_summing_to_one(df):
    obj = build_outcome_by_group(df)
    checked = 0
    for g in obj["groups"]:
        for r in g["rows"]:
            if r["suppressed"]:
                assert r["shares"] is None and r["small_n"]
                continue
            assert sum(r["counts"].values()) == r["n"], (g["id"], r["level"])
            cells = [r["shares"][o] for o in OUTCOMES]
            for cell, outcome in zip(cells, OUTCOMES):
                assert 0 <= cell["lo"] <= cell["p"] <= cell["hi"] <= 1, (g["id"], r["level"], outcome)
                assert abs(cell["p"] - r["counts"][outcome] / r["n"]) < 1e-6
            # Each p is rounded to 6 decimals, so three of them can miss 1 by up to 1.5e-6
            # (the debtor row lands on exactly 0.999999); a wrong denominator would miss by far more.
            assert abs(sum(cell["p"] for cell in cells) - 1) < 2e-6, (g["id"], r["level"])
            checked += 1
    assert checked == 44  # 45 rows across the ten groups, one suppressed
