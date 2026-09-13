"""Structural checks on the committed JSON under site/public/data.

These read the files as committed (CI regenerates them first and `scripts/check_stale.py`
proves they match), so they never fit a model. A wrong number in the prose would otherwise
go unnoticed: the site prints whatever claims.json says.
"""

import json
import pathlib

import pytest

from pipeline.constants import OUT_DIR

EXPECTED = sorted([
    "calibration.json",
    "claims.json",
    "estimator.json",
    "estimator_parity.json",
    "fairness.json",
    "hypothesis_tests.json",
    "meta.json",
    "model_metrics.json",
    "odds_ratios.json",
    "outcome_by_group.json",
    "outcome_overall.json",
    "permutation_importance.json",
])
GOLDEN = pathlib.Path(__file__).parent / "golden" / "outcome_overall.json"


def _read(name: str) -> tuple[str, dict]:
    text = (OUT_DIR / name).read_text(encoding="utf-8")
    return text, json.loads(text)


def _triples(obj):
    """Every {p|value, lo, hi} dict anywhere in `obj`, with its path for the assertion message."""
    stack = [("$", obj)]
    while stack:
        where, node = stack.pop()
        if isinstance(node, dict):
            if "lo" in node and "hi" in node and ("p" in node or "value" in node):
                yield where, node
            stack.extend((f"{where}.{k}", v) for k, v in node.items())
        elif isinstance(node, list):
            stack.extend((f"{where}[{i}]", v) for i, v in enumerate(node))


@pytest.mark.parametrize("name", EXPECTED)
def test_file_is_small_clean_and_bracketed(name):
    text, obj = _read(name)
    assert len(text.encode("utf-8")) < 200_000
    assert "NaN" not in text and "Infinity" not in text
    assert obj["id"] == name.removesuffix(".json")
    assert obj["schema"] == f"{obj['id']}.v1" and obj["generated_at"]
    for where, cell in _triples(obj):
        centre = cell["p"] if "p" in cell else cell["value"]
        values = (cell["lo"], centre, cell["hi"])
        if all(v is None for v in values):
            continue
        assert all(v is not None for v in values), where
        assert 0 <= cell["lo"] <= centre <= cell["hi"] <= 1, where


def test_meta_and_claims_anchors():
    _, meta = _read("meta.json")
    _, claims = _read("claims.json")
    assert meta["n_rows"] == 4424 and meta["n_features"] == 34 and claims["n_rows"] == 4424
    assert meta["n_columns"] == 35 and meta["n_duplicate_rows"] == 0
    assert meta["files"] == EXPECTED == sorted(p.name for p in OUT_DIR.glob("*.json"))


def test_best_model_accuracy_is_bracketed_and_the_leaking_set_is_not_worse():
    _, c = _read("claims.json")
    assert c["acc_best_enrolment_lo"] <= c["acc_best_enrolment"] <= c["acc_best_enrolment_hi"]
    assert c["acc_best_after_s1"] >= c["acc_best_enrolment"]


def test_golden_outcome_overall():
    _, actual = _read("outcome_overall.json")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual.pop("generated_at")
    golden.pop("generated_at")
    assert actual == golden
