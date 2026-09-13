"""Load the raw Dropout Lens CSV and derive labelled, ordered categoricals for analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.constants import (
    AGE_BAND_EDGES,
    APP_MODE_GROUP,
    BINARY_LABEL,
    CATEGORICALS,
    COURSE_LABEL,
    LEVELS,
    MARITAL_GROUP,
    OUTCOMES,
    PREV_QUAL_GROUP,
    RAW_PATH,
    RENAME,
)

# Derived categorical -> (short code column, code table).
_CODE_TABLES: dict[str, tuple[str, dict[int, str]]] = {
    **{name: (f"{name}_code", table) for name, table in BINARY_LABEL.items()},
    "application_mode": ("app_mode_code", APP_MODE_GROUP),
    "course": ("course_code", COURSE_LABEL),
    "marital": ("marital_code", MARITAL_GROUP),
    "previous_qualification": ("prev_qual_code", PREV_QUAL_GROUP),
}


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Read the CSV (UTF-8 with BOM) and rename its 35 columns to the short names in RENAME."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    expected, found = set(RENAME), set(raw.columns)
    if found != expected:
        raise ValueError(
            f"unexpected raw column set: missing {sorted(expected - found)}, extra {sorted(found - expected)}"
        )
    return raw.rename(columns=RENAME)


def _decode(codes: pd.Series, table: dict[int, str], name: str) -> pd.Series:
    unknown = sorted(set(codes.unique()) - set(table))
    if unknown:
        raise ValueError(f"{name}: codes outside the table: {unknown}")
    return codes.map(table)


def course_order(course_code: pd.Series) -> list[str]:
    """Course labels sorted by row count descending, ties broken by code ascending."""
    counts = course_code.value_counts()
    codes = sorted(COURSE_LABEL, key=lambda code: (-int(counts.get(code, 0)), code))
    return [COURSE_LABEL[code] for code in codes]


def approval_band_s1(enrolled: pd.Series, approved: pd.Series) -> pd.Series:
    """Band the first-semester approval rate (approved over enrolled); no enrolment is its own band."""
    e = enrolled.to_numpy(dtype=float)
    a = approved.to_numpy(dtype=float)
    if (e < 0).any() or (a < 0).any():
        raise ValueError("negative first-semester unit counts")
    rate = np.divide(a, e, out=np.full_like(e, np.nan), where=e > 0)
    bands = np.select(
        [e == 0, rate == 0, rate < 0.5, rate < 1, rate >= 1],
        LEVELS["approval_band_s1"],
        default="",
    )
    if (bands == "").any():
        raise ValueError("first-semester units that fit no approval band")
    return pd.Series(bands, index=enrolled.index)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ordered categoricals listed in CATEGORICALS plus the `dropout` flag."""
    df = df.copy()
    unknown = sorted(set(df["outcome_raw"].unique()) - set(OUTCOMES))
    if unknown:
        raise ValueError(f"outcome: values outside {OUTCOMES}: {unknown}")
    # Validate every code before touching module state.
    derived = {name: _decode(df[col], table, name) for name, (col, table) in _CODE_TABLES.items()}
    derived["outcome"] = df["outcome_raw"]
    derived["age_band"] = pd.cut(df["age"], bins=AGE_BAND_EDGES, labels=LEVELS["age_band"])
    derived["approval_band_s1"] = approval_band_s1(df["s1_enrolled"], df["s1_approved"])

    # In place, so every module holding a reference to LEVELS["course"] sees the order.
    LEVELS["course"][:] = course_order(df["course_code"])

    for name in CATEGORICALS:
        df[name] = pd.Categorical(derived[name], categories=LEVELS[name], ordered=True)
        missing = int(df[name].isna().sum())
        if missing:
            raise ValueError(f"{name}: {missing} rows outside the expected levels")
    df["dropout"] = (df["outcome"] == "Dropout").astype(int)
    return df


def load_data() -> pd.DataFrame:
    return add_derived(load_raw())
