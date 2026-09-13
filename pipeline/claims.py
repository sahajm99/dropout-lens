"""`claims.json` and `meta.json`: the site's prose numbers and the run provenance.

Every numeral the site prints in a sentence is read from `claims.json` rather than typed
into the HTML, so prose can never drift from the data. The claims are *derived from the
already-written files*, never recomputed here, so a claim and the chart beside it cannot
disagree. `build_claims` is pure: it takes the loaded frame and the parsed output objects.
`write_claims_and_meta` is the writer registered last in `pipeline.run`, because it reads
the files every other writer has just produced.

Decisions the contract leaves open, fixed here:
- `top_v`, `second_v`, `third_v` rank the corrected Cramer's V of the *pre-enrolment*
  categorical tests only; the page names the leaking approval band separately, with its
  own `v_approval_band_s1`.
- `fnr_gap_gender` is Female minus Male (the gender levels in order); the page prints it
  unsigned in percentage points.
- `top_importance_label_*` read the binary ("dropout or not") results, the target every
  other model claim on the page describes.
- A claimed cell that is suppressed raises instead of emitting null: the site's `Claims`
  type has no null, and a null would print as "0.0%".
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from pipeline.constants import OUT_DIR, OUTCOMES, RAW_PATH, RENAME, SEED, YEAR_FROM, YEAR_TO
from pipeline.io import now_iso, write_json
from pipeline.stats import round6

SOURCE_FILE = "data/raw/dataset.csv"
OUTCOME_COLUMN = "Target"
SHA_CHUNK_BYTES = 1024 * 1024
ALPHA = 0.05

# The contract's key list, verbatim; build_claims must produce exactly these.
CLAIM_KEYS = (
    "n_rows, n_features, n_columns, year_from, year_to, n_dropout, n_enrolled, n_graduate, "
    "p_dropout, p_dropout_lo, p_dropout_hi, p_enrolled, p_graduate, tiles_dropout, tiles_enrolled, "
    "tiles_graduate, p_dropout_female, p_dropout_male, p_dropout_scholarship, p_dropout_no_scholarship, "
    "p_dropout_fees_late, p_dropout_fees_ok, p_dropout_debtor, p_dropout_not_debtor, "
    "p_dropout_age_youngest, p_dropout_age_oldest, p_dropout_evening, p_dropout_daytime, "
    "p_dropout_approval_0, p_dropout_approval_100, course_max_label, p_dropout_course_max, "
    "course_min_label, p_dropout_course_min, n_tests, n_tests_holm_sig, top_v_label, top_v, "
    "second_v_label, second_v, third_v_label, third_v, v_gender, v_age_band, v_approval_band_s1, "
    "eta_age, eta_grade_s1, n_train, n_test, test_positive_share, acc_majority_binary, "
    "acc_logistic_enrolment, acc_best_enrolment, acc_best_enrolment_lo, acc_best_enrolment_hi, "
    "best_enrolment_label, f1_best_enrolment, auc_best_enrolment, auc_best_enrolment_lo, "
    "auc_best_enrolment_hi, acc_best_after_s1, acc_best_after_s1_lo, acc_best_after_s1_hi, "
    "best_after_s1_label, f1_best_after_s1, auc_best_after_s1, acc_majority_multiclass, "
    "acc_best_multiclass_enrolment, best_multiclass_enrolment_label, f1_best_multiclass_enrolment, "
    "acc_best_multiclass_after_s1, best_multiclass_after_s1_label, f1_best_multiclass_after_s1, "
    "or_fees_late, or_debtor, or_scholarship, or_male, or_age_40, or_over23, fnr_female, fnr_male, "
    "fnr_gap_gender, fpr_female, fpr_male, top_importance_label_enrolment, "
    "top_importance_label_after_s1, estimator_base_rate"
).split(", ")


# ---------------------------------------------------------------------------
# Lookups into the already-built outputs
# ---------------------------------------------------------------------------


def _outcome_row(outputs: dict, outcome: str) -> dict:
    for row in outputs["outcome_overall.json"]["rows"]:
        if row["outcome"] == outcome:
            return row
    raise KeyError(f"outcome_overall.json has no row for {outcome!r}")


def _group(outputs: dict, group_id: str) -> dict:
    for group in outputs["outcome_by_group.json"]["groups"]:
        if group["id"] == group_id:
            return group
    raise KeyError(f"outcome_by_group.json has no group {group_id!r}")


def _dropout_share(outputs: dict, group_id: str, level: str) -> float:
    for row in _group(outputs, group_id)["rows"]:
        if row["level"] == level:
            if row["suppressed"] or row["shares"] is None:
                raise ValueError(f"outcome_by_group.json {group_id}={level!r} is suppressed; no claim can quote it")
            return float(row["shares"]["Dropout"]["p"])
    raise KeyError(f"outcome_by_group.json group {group_id!r} has no level {level!r}")


def _extreme_course(outputs: dict, highest: bool) -> tuple[str, float]:
    """(label, dropout share) of the unsuppressed course with the highest or lowest share; ties by level order."""
    rows = [r for r in _group(outputs, "course")["rows"] if not r["suppressed"]]
    if not rows:
        raise ValueError("every course row is suppressed")
    row = (max if highest else min)(rows, key=lambda r: r["shares"]["Dropout"]["p"])
    return row["level"], float(row["shares"]["Dropout"]["p"])


def _categorical_test(outputs: dict, test_id: str) -> dict:
    for test in outputs["hypothesis_tests.json"]["categorical"]:
        if test["id"] == test_id:
            return test
    raise KeyError(f"hypothesis_tests.json has no categorical test {test_id!r}")


def _numeric_test(outputs: dict, test_id: str) -> dict:
    for test in outputs["hypothesis_tests.json"]["numeric"]:
        if test["id"] == test_id:
            return test
    raise KeyError(f"hypothesis_tests.json has no numeric test {test_id!r}")


def _family(ht: dict) -> list[dict]:
    """The Holm family in the writer's order: every categorical test, then ANOVA and Kruskal per numeric."""
    return list(ht["categorical"]) + [entry[test] for entry in ht["numeric"] for test in ("anova", "kruskal")]


def _top_pre_enrolment_v(outputs: dict, k: int = 3) -> list[dict]:
    """The `k` pre-enrolment categorical tests with the largest corrected V, ties by contract order."""
    tests = [t for t in outputs["hypothesis_tests.json"]["categorical"] if not t["post_enrolment"]]
    ranked = sorted(tests, key=lambda t: -t["cramers_v_corrected"])
    if len(ranked) < k:
        raise ValueError(f"fewer than {k} pre-enrolment categorical tests")
    return ranked[:k]


def _result(outputs: dict, target: str, feature_set: str, model: str) -> dict:
    for result in outputs["model_metrics.json"]["results"]:
        if (result["target"], result["feature_set"], result["model"]) == (target, feature_set, model):
            return result
    raise KeyError(f"model_metrics.json has no result for {(target, feature_set, model)}")


def _best(outputs: dict, target: str, feature_set: str) -> tuple[dict, str]:
    """The best model's result row and its label, for one (target, feature set)."""
    mm = outputs["model_metrics.json"]
    model = mm["best"][target][feature_set]
    return _result(outputs, target, feature_set, model), mm["model_labels"][model]


def _metric(result: dict, metric: str) -> dict:
    cell = result["test"][metric]
    if cell is None or any(cell[k] is None for k in ("value", "lo", "hi")):
        raise ValueError(f"model_metrics.json {result['target']}/{result['feature_set']}/{result['model']} has no {metric}")
    return cell


def _or(outputs: dict, model: str, variable: str, level: str) -> float:
    for term in outputs["odds_ratios.json"]["models"][model]["terms"]:
        if term["variable"] == variable and term["level"] == level:
            if term["or"] is None:
                raise ValueError(f"odds_ratios.json {model} term {variable}={level!r} has no estimate")
            return float(term["or"])
    raise KeyError(f"odds_ratios.json {model} model has no term {variable}={level!r}")


def _fairness_rate(outputs: dict, feature_set: str, group_id: str, level: str, rate: str) -> float:
    for result in outputs["fairness.json"]["results"]:
        if result["feature_set"] != feature_set:
            continue
        for group in result["groups"]:
            if group["id"] != group_id:
                continue
            for row in group["levels"]:
                if row["level"] == level:
                    if row[rate] is None:
                        raise ValueError(f"fairness.json {feature_set} {group_id}={level!r} {rate} is suppressed")
                    return float(row[rate]["p"])
            raise KeyError(f"fairness.json {feature_set} group {group_id!r} has no level {level!r}")
    raise KeyError(f"fairness.json has no result for {feature_set!r} with group {group_id!r}")


def _top_importance_label(outputs: dict, target: str, feature_set: str) -> str:
    for result in outputs["permutation_importance.json"]["results"]:
        if (result["target"], result["feature_set"]) == (target, feature_set):
            # Features are written sorted by mean importance descending.
            return str(result["features"][0]["label"])
    raise KeyError(f"permutation_importance.json has no result for {(target, feature_set)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_claims(df, outputs: dict[str, dict]) -> dict:
    """Assemble the flat claims object from the parsed output JSONs (and the raw header)."""
    overall = outputs["outcome_overall.json"]
    dropout, enrolled, graduate = (_outcome_row(outputs, o) for o in OUTCOMES)
    course_max_label, p_course_max = _extreme_course(outputs, highest=True)
    course_min_label, p_course_min = _extreme_course(outputs, highest=False)

    ht = outputs["hypothesis_tests.json"]
    top, second, third = _top_pre_enrolment_v(outputs)

    mm = outputs["model_metrics.json"]
    best_enrolment, best_enrolment_label = _best(outputs, "binary", "enrolment")
    best_after_s1, best_after_s1_label = _best(outputs, "binary", "after_s1")
    best_multi_enrolment, best_multi_enrolment_label = _best(outputs, "multiclass", "enrolment")
    best_multi_after_s1, best_multi_after_s1_label = _best(outputs, "multiclass", "after_s1")

    fnr_female = _fairness_rate(outputs, "enrolment", "gender", "Female", "fnr")
    fnr_male = _fairness_rate(outputs, "enrolment", "gender", "Male", "fnr")

    n_columns = len(read_raw_columns())

    claims = {
        "schema": "claims.v1",
        "id": "claims",
        "generated_at": now_iso(),
        "n_rows": int(overall["n_total"]),
        "n_features": n_columns - 1,
        "n_columns": n_columns,
        "year_from": YEAR_FROM,
        "year_to": YEAR_TO,
        "n_dropout": int(dropout["n"]),
        "n_enrolled": int(enrolled["n"]),
        "n_graduate": int(graduate["n"]),
        "p_dropout": dropout["p"],
        "p_dropout_lo": dropout["lo"],
        "p_dropout_hi": dropout["hi"],
        "p_enrolled": enrolled["p"],
        "p_graduate": graduate["p"],
        "tiles_dropout": int(overall["tiles"]["Dropout"]),
        "tiles_enrolled": int(overall["tiles"]["Enrolled"]),
        "tiles_graduate": int(overall["tiles"]["Graduate"]),
        "p_dropout_female": _dropout_share(outputs, "gender", "Female"),
        "p_dropout_male": _dropout_share(outputs, "gender", "Male"),
        "p_dropout_scholarship": _dropout_share(outputs, "scholarship", "Scholarship holder"),
        "p_dropout_no_scholarship": _dropout_share(outputs, "scholarship", "No scholarship"),
        "p_dropout_fees_late": _dropout_share(outputs, "tuition", "Fees not up to date"),
        "p_dropout_fees_ok": _dropout_share(outputs, "tuition", "Fees up to date"),
        "p_dropout_debtor": _dropout_share(outputs, "debtor", "Debtor"),
        "p_dropout_not_debtor": _dropout_share(outputs, "debtor", "Not a debtor"),
        "p_dropout_age_youngest": _dropout_share(outputs, "age_band", "17 to 19"),
        "p_dropout_age_oldest": _dropout_share(outputs, "age_band", "40 and over"),
        "p_dropout_evening": _dropout_share(outputs, "attendance", "Evening"),
        "p_dropout_daytime": _dropout_share(outputs, "attendance", "Daytime"),
        "p_dropout_approval_0": _dropout_share(outputs, "approval_band_s1", "0% approved"),
        "p_dropout_approval_100": _dropout_share(outputs, "approval_band_s1", "100% approved"),
        "course_max_label": course_max_label,
        "p_dropout_course_max": p_course_max,
        "course_min_label": course_min_label,
        "p_dropout_course_min": p_course_min,
        "n_tests": int(ht["family_size"]),
        "n_tests_holm_sig": sum(1 for slot in _family(ht) if slot["p_holm"] < ALPHA),
        "top_v_label": top["label"],
        "top_v": top["cramers_v_corrected"],
        "second_v_label": second["label"],
        "second_v": second["cramers_v_corrected"],
        "third_v_label": third["label"],
        "third_v": third["cramers_v_corrected"],
        "v_gender": _categorical_test(outputs, "gender")["cramers_v_corrected"],
        "v_age_band": _categorical_test(outputs, "age_band")["cramers_v_corrected"],
        "v_approval_band_s1": _categorical_test(outputs, "approval_band_s1")["cramers_v_corrected"],
        "eta_age": _numeric_test(outputs, "age")["anova"]["eta_squared"],
        "eta_grade_s1": _numeric_test(outputs, "grade_s1")["anova"]["eta_squared"],
        "n_train": int(mm["split"]["n_train"]),
        "n_test": int(mm["split"]["n_test"]),
        "test_positive_share": mm["targets"]["binary"]["test_positive_share"],
        "acc_majority_binary": _metric(_result(outputs, "binary", "enrolment", "majority"), "accuracy")["value"],
        "acc_logistic_enrolment": _metric(_result(outputs, "binary", "enrolment", "logistic"), "accuracy")["value"],
        "acc_best_enrolment": _metric(best_enrolment, "accuracy")["value"],
        "acc_best_enrolment_lo": _metric(best_enrolment, "accuracy")["lo"],
        "acc_best_enrolment_hi": _metric(best_enrolment, "accuracy")["hi"],
        "best_enrolment_label": best_enrolment_label,
        "f1_best_enrolment": _metric(best_enrolment, "macro_f1")["value"],
        "auc_best_enrolment": _metric(best_enrolment, "roc_auc")["value"],
        "auc_best_enrolment_lo": _metric(best_enrolment, "roc_auc")["lo"],
        "auc_best_enrolment_hi": _metric(best_enrolment, "roc_auc")["hi"],
        "acc_best_after_s1": _metric(best_after_s1, "accuracy")["value"],
        "acc_best_after_s1_lo": _metric(best_after_s1, "accuracy")["lo"],
        "acc_best_after_s1_hi": _metric(best_after_s1, "accuracy")["hi"],
        "best_after_s1_label": best_after_s1_label,
        "f1_best_after_s1": _metric(best_after_s1, "macro_f1")["value"],
        "auc_best_after_s1": _metric(best_after_s1, "roc_auc")["value"],
        "acc_majority_multiclass": _metric(_result(outputs, "multiclass", "enrolment", "majority"), "accuracy")["value"],
        "acc_best_multiclass_enrolment": _metric(best_multi_enrolment, "accuracy")["value"],
        "best_multiclass_enrolment_label": best_multi_enrolment_label,
        "f1_best_multiclass_enrolment": _metric(best_multi_enrolment, "macro_f1")["value"],
        "acc_best_multiclass_after_s1": _metric(best_multi_after_s1, "accuracy")["value"],
        "best_multiclass_after_s1_label": best_multi_after_s1_label,
        "f1_best_multiclass_after_s1": _metric(best_multi_after_s1, "macro_f1")["value"],
        "or_fees_late": _or(outputs, "binary_enrolment", "tuition", "Fees not up to date"),
        "or_debtor": _or(outputs, "binary_enrolment", "debtor", "Debtor"),
        "or_scholarship": _or(outputs, "binary_enrolment", "scholarship", "Scholarship holder"),
        "or_male": _or(outputs, "binary_enrolment", "gender", "Male"),
        "or_age_40": _or(outputs, "binary_enrolment", "age_band", "40 and over"),
        "or_over23": _or(outputs, "binary_enrolment", "application_mode", "Over 23 years old"),
        "fnr_female": fnr_female,
        "fnr_male": fnr_male,
        "fnr_gap_gender": round6(fnr_female - fnr_male),
        "fpr_female": _fairness_rate(outputs, "enrolment", "gender", "Female", "fpr"),
        "fpr_male": _fairness_rate(outputs, "enrolment", "gender", "Male", "fpr"),
        "top_importance_label_enrolment": _top_importance_label(outputs, "binary", "enrolment"),
        "top_importance_label_after_s1": _top_importance_label(outputs, "binary", "after_s1"),
        "estimator_base_rate": outputs["estimator.json"]["base_rate"],
    }
    _check_claims(claims)
    return claims


def _check_claims(claims: dict) -> None:
    """Exactly the contract's keys, every value a finite number or a non-empty string."""
    keys = [k for k in claims if k not in ("schema", "id", "generated_at")]
    if keys != CLAIM_KEYS:
        missing = sorted(set(CLAIM_KEYS) - set(keys))
        extra = sorted(set(keys) - set(CLAIM_KEYS))
        raise ValueError(f"claims keys differ from the contract: missing {missing}, extra {extra}")
    for key in CLAIM_KEYS:
        value = claims[key]
        if isinstance(value, bool) or not (
            (isinstance(value, (int, float)) and math.isfinite(value)) or (isinstance(value, str) and value)
        ):
            raise ValueError(f"claims[{key!r}] = {value!r} is not a finite number or a non-empty string")


def sha256_of(path) -> str:
    """Streaming sha256 so the raw CSV is never held in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(SHA_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_columns() -> list[str]:
    """Header of the raw CSV, so `n_columns` counts source columns, not derived ones."""
    columns = list(pd.read_csv(RAW_PATH, nrows=0, encoding="utf-8-sig").columns)
    if set(columns) != set(RENAME) or OUTCOME_COLUMN not in columns:
        raise ValueError("raw header differs from pipeline.constants.RENAME")
    return columns


def build_meta(df, files: list[str]) -> dict:
    raw_columns = [c for c in RENAME.values() if c in df.columns]
    counts = df["outcome"].value_counts()
    n_columns = len(read_raw_columns())
    return {
        "schema": "meta.v1",
        "id": "meta",
        "generated_at": now_iso(),
        "source_file": SOURCE_FILE,
        "sha256": sha256_of(RAW_PATH),
        "n_rows": int(len(df)),
        "n_columns": n_columns,
        "n_features": n_columns - 1,
        "n_duplicate_rows": int(df[raw_columns].duplicated().sum()),
        "outcome_counts": {o: int(counts.get(o, 0)) for o in OUTCOMES},
        "seed": SEED,
        "files": sorted(files),
    }


def read_outputs() -> dict[str, dict]:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in OUT_DIR.glob("*.json")}


def write_claims_and_meta(df) -> list[Path]:
    """Write `claims.json` then `meta.json`; must run after every other writer."""
    claims_path = write_json("claims.json", build_claims(df, read_outputs()))
    files = sorted({p.name for p in OUT_DIR.glob("*.json")} | {"meta.json"})
    meta_path = write_json("meta.json", build_meta(df, files))
    return [claims_path, meta_path]
