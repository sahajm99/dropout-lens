from pipeline.constants import CATEGORICALS, LEVELS


def test_shape_and_outcome_counts(df):
    assert len(df) == 4424
    assert df["outcome"].value_counts().to_dict() == {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794}


def test_evening_attendance_is_exactly_the_two_evening_courses(df):
    evening_courses = {"Social Service (evening)", "Management (evening)"}
    assert set(df.loc[df["attendance"] == "Evening", "course"].unique()) == evening_courses
    assert not df.loc[df["attendance"] == "Daytime", "course"].isin(evening_courses).any()


def test_over_23_route_and_categorical_integrity(df):
    assert (df.loc[df["application_mode"] == "Over 23 years old", "age"] >= 23).all()
    assert df["age_band"].cat.categories.tolist() == LEVELS["age_band"]
    for name in CATEGORICALS:
        assert df[name].isna().sum() == 0, name
        assert df[name].cat.categories.tolist() == LEVELS[name], name
        assert df[name].cat.ordered, name


def test_approval_band_s1_counts(df):
    counts = df["approval_band_s1"].value_counts().to_dict()
    assert counts == {
        "No units enrolled": 180,
        "0% approved": 538,
        "1 to 49%": 310,
        "50 to 99%": 1668,
        "100% approved": 1728,
    }


def test_course_order_is_by_count_descending(df):
    assert len(LEVELS["course"]) == 17
    assert LEVELS["course"][0] == "Nursing"
    assert df["course"].cat.categories.tolist() == LEVELS["course"]
    counts = df["course"].value_counts(sort=False).reindex(LEVELS["course"]).tolist()
    assert counts == sorted(counts, reverse=True)
