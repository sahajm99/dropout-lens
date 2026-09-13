"""Shared constants: paths, seed, outcome order, code tables, level orders, references, labels, groups.

Every list of levels lives here so that each JSON file, chart and model uses one order.
`LEVELS["course"]` is the one exception: it starts in code order and `pipeline.load`
replaces it in place, on first load, with the labels sorted by row count descending
(ties by code ascending), so every module that holds a reference sees the same order.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "dataset.csv"
OUT_DIR = ROOT / "site" / "public" / "data"

SEED = 20260912
OUTCOMES = ["Dropout", "Enrolled", "Graduate"]
YEAR_FROM = 2008
YEAR_TO = 2019
SUPPRESS_BELOW = 30
SMALL_N_BELOW = 100
Z95 = 1.959964

# Raw column name (the CSV header, read with encoding="utf-8-sig") -> short name.
RENAME: dict[str, str] = {
    "Marital status": "marital_code",
    "Application mode": "app_mode_code",
    "Application order": "app_order",
    "Course": "course_code",
    "Daytime/evening attendance": "attendance_code",
    "Previous qualification": "prev_qual_code",
    "Nacionality": "nationality_code",
    "Mother's qualification": "mother_qual",
    "Father's qualification": "father_qual",
    "Mother's occupation": "mother_occ",
    "Father's occupation": "father_occ",
    "Displaced": "displaced_code",
    "Educational special needs": "special_needs_code",
    "Debtor": "debtor_code",
    "Tuition fees up to date": "tuition_code",
    "Gender": "gender_code",
    "Scholarship holder": "scholarship_code",
    "Age at enrollment": "age",
    "International": "international_code",
    "Curricular units 1st sem (credited)": "s1_credited",
    "Curricular units 1st sem (enrolled)": "s1_enrolled",
    "Curricular units 1st sem (evaluations)": "s1_evaluations",
    "Curricular units 1st sem (approved)": "s1_approved",
    "Curricular units 1st sem (grade)": "s1_grade",
    "Curricular units 1st sem (without evaluations)": "s1_without_eval",
    "Curricular units 2nd sem (credited)": "s2_credited",
    "Curricular units 2nd sem (enrolled)": "s2_enrolled",
    "Curricular units 2nd sem (evaluations)": "s2_evaluations",
    "Curricular units 2nd sem (approved)": "s2_approved",
    "Curricular units 2nd sem (grade)": "s2_grade",
    "Curricular units 2nd sem (without evaluations)": "s2_without_eval",
    "Unemployment rate": "unemployment",
    "Inflation rate": "inflation",
    "GDP": "gdp",
    "Target": "outcome_raw",
}

# ---------------------------------------------------------------------------
# Code tables (Table 1 of Realinho et al. 2022; verified in docs/DATA_AUDIT.md section 4)
# ---------------------------------------------------------------------------

COURSE_LABEL: dict[int, str] = {
    1: "Biofuel Production Technologies",
    2: "Animation and Multimedia Design",
    3: "Social Service (evening)",
    4: "Agronomy",
    5: "Communication Design",
    6: "Veterinary Nursing",
    7: "Informatics Engineering",
    8: "Equinculture",
    9: "Management",
    10: "Social Service",
    11: "Tourism",
    12: "Nursing",
    13: "Oral Hygiene",
    14: "Advertising and Marketing Management",
    15: "Journalism and Communication",
    16: "Basic Education",
    17: "Management (evening)",
}


def _spread(groups: dict[str, tuple[int, ...]]) -> dict[int, str]:
    """Turn {label: codes} into {code: label}; the insertion order of `groups` is the level order."""
    return {code: label for label, codes in groups.items() for code in codes}


_APP_MODE_GROUPS: dict[str, tuple[int, ...]] = {
    "1st phase, general contingent": (1,),
    "2nd or 3rd phase, general contingent": (8, 9),
    "Over 23 years old": (12,),
    "Transfer or change of course or institution": (13, 14, 16, 18),
    "Prior higher or technical qualification": (4, 15, 17),
    "Special contingents and ordinances": (2, 3, 5, 6, 7, 10, 11),
}
_MARITAL_GROUPS: dict[str, tuple[int, ...]] = {
    "Single": (1,),
    "Married or de facto union": (2, 5),
    "Divorced, separated or widowed": (3, 4, 6),
}
_PREV_QUAL_GROUPS: dict[str, tuple[int, ...]] = {
    "Secondary": (1,),
    "Higher or post-secondary": (2, 3, 4, 5, 6, 14, 15, 16, 17),
    "Basic or incomplete secondary": (7, 8, 9, 10, 11, 12, 13),
}

APP_MODE_GROUP: dict[int, str] = _spread(_APP_MODE_GROUPS)
MARITAL_GROUP: dict[int, str] = _spread(_MARITAL_GROUPS)
PREV_QUAL_GROUP: dict[int, str] = _spread(_PREV_QUAL_GROUPS)

# Binary flags: code -> label, in level order (the reference level first).
BINARY_LABEL: dict[str, dict[int, str]] = {
    "gender": {0: "Female", 1: "Male"},
    "scholarship": {0: "No scholarship", 1: "Scholarship holder"},
    "tuition": {1: "Fees up to date", 0: "Fees not up to date"},
    "debtor": {0: "Not a debtor", 1: "Debtor"},
    "displaced": {0: "Not displaced", 1: "Displaced"},
    "international": {0: "Portuguese", 1: "International"},
    "special_needs": {0: "No special needs", 1: "Special needs"},
    "attendance": {1: "Daytime", 0: "Evening"},
}

# pd.cut edges (right-closed): (16,19] (19,22] (22,29] (29,39] (39,200].
AGE_BAND_EDGES = [16, 19, 22, 29, 39, 200]

# ---------------------------------------------------------------------------
# Level orders, references, group registry, labels
# ---------------------------------------------------------------------------

LEVELS: dict[str, list[str]] = {
    "outcome": list(OUTCOMES),
    **{name: list(table.values()) for name, table in BINARY_LABEL.items()},
    "age_band": ["17 to 19", "20 to 22", "23 to 29", "30 to 39", "40 and over"],
    "application_mode": list(_APP_MODE_GROUPS),
    # Placeholder in code order; replaced in place by pipeline.load on first load.
    "course": [COURSE_LABEL[code] for code in sorted(COURSE_LABEL)],
    "marital": list(_MARITAL_GROUPS),
    "previous_qualification": list(_PREV_QUAL_GROUPS),
    "approval_band_s1": ["No units enrolled", "0% approved", "1 to 49%", "50 to 99%", "100% approved"],
}

# The derived categorical columns that pipeline.load adds, in this order.
CATEGORICALS: list[str] = list(LEVELS)

REFERENCE: dict[str, str] = {
    "gender": "Female",
    "scholarship": "No scholarship",
    "tuition": "Fees up to date",
    "debtor": "Not a debtor",
    "displaced": "Not displaced",
    "international": "Portuguese",
    "special_needs": "No special needs",
    "attendance": "Daytime",
    "age_band": "17 to 19",
    "application_mode": "1st phase, general contingent",
    "course": "Nursing",
    "marital": "Single",
    "previous_qualification": "Secondary",
    "approval_band_s1": "100% approved",
}


def _group(id_: str, label: str, post_enrolment: bool = False) -> dict:
    return {"id": id_, "label": label, "column": id_, "post_enrolment": post_enrolment}


# The first ten are the outcome_by_group.json groups, in contract order; the last
# four are used by the hypothesis tests only (the descriptive writer skips them).
GROUPS: list[dict] = [
    _group("gender", "Gender"),
    _group("scholarship", "Scholarship"),
    _group("tuition", "Tuition fees"),
    _group("debtor", "Debtor status"),
    _group("age_band", "Age at enrolment"),
    _group("application_mode", "Application route"),
    _group("course", "Course"),
    _group("attendance", "Attendance"),
    _group("displaced", "Displaced"),
    _group("approval_band_s1", "First-semester approval rate", post_enrolment=True),
    _group("marital", "Marital status"),
    _group("previous_qualification", "Previous qualification"),
    _group("international", "Nationality"),
    _group("special_needs", "Educational special needs"),
]

VARIABLE_LABELS: dict[str, str] = {g["id"]: g["label"] for g in GROUPS}

# Human phrasing per (variable, level) for odds-ratio rows and the estimator.
LABELS: dict[str, dict[str, str]] = {
    "age_band": {band: f"Age {band}" for band in LEVELS["age_band"]},
    "gender": {"Female": "Female", "Male": "Male"},
    "scholarship": {"No scholarship": "No scholarship", "Scholarship holder": "Scholarship holder"},
    "tuition": {
        "Fees up to date": "Tuition fees up to date",
        "Fees not up to date": "Tuition fees not up to date",
    },
    "debtor": {"Not a debtor": "Not a debtor", "Debtor": "Debtor"},
    "displaced": {"Not displaced": "Not displaced", "Displaced": "Displaced"},
    "international": {"Portuguese": "Portuguese nationality", "International": "International student"},
    "special_needs": {
        "No special needs": "No educational special needs",
        "Special needs": "Educational special needs",
    },
    "attendance": {"Daytime": "Daytime attendance", "Evening": "Evening attendance"},
    "application_mode": {mode: f"Application route: {mode}" for mode in LEVELS["application_mode"]},
    "course": {course: f"Course: {course}" for course in COURSE_LABEL.values()},
    "marital": {status: status for status in LEVELS["marital"]},
    "previous_qualification": {q: f"Previous qualification: {q}" for q in LEVELS["previous_qualification"]},
    "approval_band_s1": {
        "No units enrolled": "First semester: no units enrolled",
        "0% approved": "First semester: 0% of units approved",
        "1 to 49%": "First semester: 1 to 49% of units approved",
        "50 to 99%": "First semester: 50 to 99% of units approved",
        "100% approved": "First semester: 100% of units approved",
    },
}

for _var, _ref in REFERENCE.items():
    if _ref not in LEVELS[_var]:
        raise ValueError(f"REFERENCE[{_var!r}] = {_ref!r} is not one of LEVELS[{_var!r}]")
for _var, _levels in LEVELS.items():
    if _var != "outcome" and set(LABELS[_var]) != set(_levels):
        raise ValueError(f"LABELS[{_var!r}] does not cover exactly LEVELS[{_var!r}]")
del _var, _ref, _levels
