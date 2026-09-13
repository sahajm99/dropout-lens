"""Descriptive writers: outcome shares overall and by group, with Wilson intervals.

Two pure builders (`build_outcome_overall`, `build_outcome_by_group`) return the
objects described by the `outcome_overall.v1` and `outcome_by_group.v1` contracts;
the `write_*` functions serialise them under `site/public/data/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.constants import GROUPS, LEVELS, OUTCOMES, SMALL_N_BELOW, SUPPRESS_BELOW
from pipeline.io import now_iso, write_json
from pipeline.stats import share_cell

CI = {"method": "wilson", "level": 0.95}

# The groups of outcome_by_group.json, in contract order. The remaining GROUPS
# entries (marital, previous_qualification, international, special_needs) are
# used by the hypothesis tests only.
BY_GROUP_IDS = [
    "gender",
    "scholarship",
    "tuition",
    "debtor",
    "age_band",
    "application_mode",
    "course",
    "attendance",
    "displaced",
    "approval_band_s1",
]

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def largest_remainder(counts: dict[str, int], total: int = 100) -> dict[str, int]:
    """Integers summing to `total`, in proportion to `counts` (Hamilton apportionment).

    Each key gets floor(total * share); the units left over go, one each, to the
    keys with the largest fractional parts, ties broken by key order. Integer
    arithmetic throughout, so equal fractional parts really are ties.
    """
    n = sum(int(v) for v in counts.values())
    if n <= 0:
        raise ValueError("counts must sum to a positive number")
    floors = {k: (total * int(v)) // n for k, v in counts.items()}
    remainders = {k: (total * int(v)) % n for k, v in counts.items()}
    leftover = total - sum(floors.values())
    # sorted() is stable, so keys with equal remainders keep their input order.
    for k in sorted(counts, key=lambda key: -remainders[key])[:leftover]:
        floors[k] += 1
    return floors


def _outcome_counts(outcome: pd.Series) -> dict[str, int]:
    counts = outcome.value_counts(sort=False).reindex(OUTCOMES, fill_value=0)
    return {o: int(counts[o]) for o in OUTCOMES}


def _crosstab(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Level x outcome counts with every level present (observed=False), in constants order.

    Both columns are ordered categoricals whose categories are the constants' level
    lists, so observed=False already yields every level in that order; a mismatch is
    an error, not something to paper over with a reindex.
    """
    table = df.groupby([column, "outcome"], observed=False).size().unstack("outcome", fill_value=0)
    if list(table.index) != LEVELS[column] or list(table.columns) != OUTCOMES:
        raise ValueError(f"{column}: crosstab levels differ from pipeline.constants")
    return table


def _group_rows(table: pd.DataFrame) -> list[dict]:
    rows = []
    for level, row in table.iterrows():
        counts = {o: int(row[o]) for o in OUTCOMES}
        n = sum(counts.values())
        suppressed = n < SUPPRESS_BELOW
        shares = None if suppressed else {o: share_cell(counts[o], n) for o in OUTCOMES}
        rows.append({
            "level": str(level),
            "n": n,
            "counts": counts,
            "shares": shares,
            "suppressed": suppressed,
            "small_n": n < SMALL_N_BELOW,
        })
    return rows


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_outcome_overall(df: pd.DataFrame) -> dict:
    n_total = int(len(df))
    counts = _outcome_counts(df["outcome"])
    rows = []
    for outcome in OUTCOMES:
        cell = share_cell(counts[outcome], n_total)
        if cell is None:
            raise ValueError(f"fewer than {SUPPRESS_BELOW} rows in total; nothing to describe")
        rows.append({"outcome": outcome, "n": counts[outcome], **cell})
    return {
        "schema": "outcome_overall.v1",
        "id": "outcome_overall",
        "generated_at": now_iso(),
        "n_total": n_total,
        "outcomes": list(OUTCOMES),
        "ci": dict(CI),
        "rows": rows,
        "tiles": largest_remainder(counts),
    }


def build_outcome_by_group(df: pd.DataFrame) -> dict:
    by_id = {g["id"]: g for g in GROUPS}
    groups = []
    for group_id in BY_GROUP_IDS:
        group = by_id[group_id]
        table = _crosstab(df, group["column"])
        groups.append({
            "id": group["id"],
            "label": group["label"],
            "post_enrolment": bool(group["post_enrolment"]),
            "levels": [str(level) for level in table.index],
            "rows": _group_rows(table),
        })
    return {
        "schema": "outcome_by_group.v1",
        "id": "outcome_by_group",
        "generated_at": now_iso(),
        "outcomes": list(OUTCOMES),
        "ci": dict(CI),
        "suppress_below": SUPPRESS_BELOW,
        "small_n_below": SMALL_N_BELOW,
        "n_total": int(len(df)),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_outcome_overall(df: pd.DataFrame) -> Path:
    return write_json("outcome_overall.json", build_outcome_overall(df))


def write_outcome_by_group(df: pd.DataFrame) -> Path:
    return write_json("outcome_by_group.json", build_outcome_by_group(df))
