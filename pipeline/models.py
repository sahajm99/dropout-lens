"""Predictive models on the one split (DECISIONS S4 to S10, R3, R4) and their four JSON files.

Four models (majority baseline, statsmodels logistic, HistGradientBoosting, RandomForest) are fitted
for two targets (dropout or not; Dropout, Enrolled or Graduate) on two feature sets (at enrolment;
after first semester, which leaks). Model selection is 5-fold stratified CV on the training 80% by
macro F1; the test 20% is touched once per model. `fit_all` is memoised on the module so
`fairness.py` and `estimator.py` reuse one bundle inside a run.

The logistic fits are Newton first and BFGS on non-convergence (a Newton fit whose parameters
are not finite counts as non-converged: its step test passes vacuously on NaN). The multinomial
fits need BFGS: "Biofuel Production Technologies" has no Graduate in the training split (6 Dropout,
3 Enrolled), so its contrasts against the Graduate base are separated. A level whose rows all
share one outcome has no finite maximum-likelihood coefficient, so its `coef`, `or`, `lo`, `hi`,
`se` and `p_value` are emitted as null (never the optimiser's stopping point) and the row is
flagged `unstable`, as is every row with a 2x2 cell under 10.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

from pipeline.constants import LABELS, LEVELS, OUTCOMES, REFERENCE, SEED
from pipeline.features import (
    FEATURE_SETS,
    OUTCOME_CODE,
    TARGETS,
    formula,
    prepare,
    split,
    variable_label,
    y_codes,
    y_labels,
)
from pipeline.io import now_iso, write_json
from pipeline.metrics import METRICS, bootstrap_ci, calibration, confusion, predict_class
from pipeline.stats import round4, round6, sig3

MODELS = ["majority", "logistic", "hgb", "rf"]
MODEL_LABELS = {
    "majority": "Majority baseline",
    "logistic": "Logistic regression",
    "hgb": "Gradient boosting",
    "rf": "Random forest",
}

CV_FOLDS = 5
BOOTSTRAP_RESAMPLES = 1000
CALIBRATION_BINS = 10
PERMUTATION_REPEATS = 10
UNSTABLE_MIN_CELL = 10
TEST_SHARE = 0.2

# DECISIONS S7.
HGB_GRID = {"model__learning_rate": [0.05, 0.1], "model__max_iter": [100, 200]}
RF_GRID = {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 5]}

TERM_RE = re.compile(r"^C\((\w+), Treatment\('([^']*)'\)\)\[T\.(.+)\]$")

# The three logistic fits published as odds ratios; the multiclass after_s1 fit serves the metrics only.
ODDS_RATIO_LABELS = {
    "binary_enrolment": "Dropout or not, at enrolment",
    "binary_after_s1": "Dropout or not, after first semester",
    "multiclass_enrolment": "Dropout, Enrolled or Graduate, at enrolment",
}
MULTICLASS_BASE = "Graduate"


# ---------------------------------------------------------------------------
# Fitted objects
# ---------------------------------------------------------------------------


@dataclass
class Predictor:
    """One fitted model: probabilities in `classes` order for any frame with the raw columns."""

    target: str
    feature_set: str
    model: str
    classes: list[str]
    label: str
    chosen_params: dict | None
    cv_macro_f1_mean: float
    cv_macro_f1_sd: float
    variables: list[str]
    proba_fn: Callable[[pd.DataFrame], np.ndarray] = field(repr=False)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.asarray(self.proba_fn(frame), dtype=float)
        if out.shape != (len(frame), len(self.classes)):
            raise RuntimeError(f"{self.model} returned probabilities of shape {out.shape}")
        return out


@dataclass
class FitBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray
    predictors: dict[tuple[str, str, str], Predictor]
    best: dict[str, dict[str, str]]
    logits: dict[str, Any]
    fit_methods: dict[str, str]

    @property
    def binary_enrolment(self):
        return self.logits["binary_enrolment"]

    @property
    def binary_after_s1(self):
        return self.logits["binary_after_s1"]

    @property
    def multiclass_enrolment(self):
        return self.logits["multiclass_enrolment"]

    def predictor(self, target: str, feature_set: str, model: str) -> Predictor:
        return self.predictors[(target, feature_set, model)]

    def best_predictor(self, target: str, feature_set: str) -> Predictor:
        return self.predictors[(target, feature_set, self.best[target][feature_set])]


def term_name(variable: str, level: str) -> str:
    """The patsy term name of a non-reference level, as it appears in `res.params.index`."""
    return f"C({variable}, Treatment('{REFERENCE[variable]}'))[T.{level}]"


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _skf() -> StratifiedKFold:
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)


def _macro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    return float(f1_score(y, predict_class(proba), average="macro", zero_division=0))


def _cv_summary(scores: list[float]) -> tuple[float, float]:
    """Mean and population sd, the same convention as GridSearchCV's std_test_score."""
    return float(np.mean(scores)), float(np.std(scores))


def _converged(res) -> bool:
    """The optimiser's flag, and finite parameters and likelihood: Newton's step test passes on NaN."""
    params = np.asarray(res.params, dtype=float)
    return bool(res.mle_retvals["converged"]) and bool(np.isfinite(res.llf)) and bool(np.isfinite(params).all())


def _fit_formula(model_fn, f: str, data: pd.DataFrame):
    """Fit `f` on `data` by Newton (300 iterations), then BFGS (3000) if that did not converge."""
    model = model_fn(f, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            res = model.fit(method="newton", maxiter=300, disp=0)
            converged = _converged(res)
        except np.linalg.LinAlgError:
            converged = False
        if converged:
            return res, "newton"
        res = model.fit(method="bfgs", maxiter=3000, disp=0)
    if not _converged(res):
        raise RuntimeError(f"logistic fit did not converge under newton or bfgs: {f}")
    return res, "bfgs"


def _logit_proba(res, target: str) -> Callable[[pd.DataFrame], np.ndarray]:
    if target == "binary":

        def proba(frame):
            p = np.asarray(res.predict(frame), dtype=float)
            return np.column_stack([1 - p, p])

    else:
        order = [OUTCOME_CODE[o] for o in OUTCOMES]

        def proba(frame):
            # statsmodels columns are in code order (Graduate, Dropout, Enrolled).
            return np.asarray(res.predict(frame), dtype=float)[:, order]

    return proba


def _fit_majority(target: str, feature_set: str, train: pd.DataFrame) -> Predictor:
    spec = TARGETS[target]
    k = len(spec["classes"])
    y = y_codes(train, target)

    def one_hot(index: int, n: int) -> np.ndarray:
        out = np.zeros((n, k))
        out[:, index] = 1.0
        return out

    majority = int(np.bincount(y, minlength=k).argmax())
    scores = []
    for a, b in _skf().split(train, y):
        fold_majority = int(np.bincount(y[a], minlength=k).argmax())
        scores.append(_macro_f1(y[b], one_hot(fold_majority, len(b))))
    mean, sd = _cv_summary(scores)
    return Predictor(
        target, feature_set, "majority", list(spec["classes"]), MODEL_LABELS["majority"],
        None, mean, sd, [], lambda frame: one_hot(majority, len(frame)),
    )


def _fit_logistic(target: str, feature_set: str, train: pd.DataFrame) -> tuple[Predictor, Any, str]:
    spec = TARGETS[target]
    cats = list(FEATURE_SETS[feature_set]["cat"])
    f = formula(spec["formula_target"], cats)
    model_fn = smf.logit if target == "binary" else smf.mnlogit
    res, method = _fit_formula(model_fn, f, train)
    y = y_codes(train, target)
    scores = []
    for a, b in _skf().split(train, y):
        fold_res, _ = _fit_formula(model_fn, f, train.iloc[a])
        scores.append(_macro_f1(y[b], _logit_proba(fold_res, target)(train.iloc[b])))
    mean, sd = _cv_summary(scores)
    predictor = Predictor(
        target, feature_set, "logistic", list(spec["classes"]), MODEL_LABELS["logistic"],
        None, mean, sd, cats, _logit_proba(res, target),
    )
    return predictor, res, method


def _fit_tree(model: str, target: str, feature_set: str, train: pd.DataFrame) -> Predictor:
    spec = TARGETS[target]
    fs = FEATURE_SETS[feature_set]
    tree_cat, num = list(fs["tree_cat"]), list(fs["num"])
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(
            categories=[list(LEVELS[c]) for c in tree_cat], handle_unknown="ignore", sparse_output=False,
        ), tree_cat),
        ("num", "passthrough", num),
    ])
    if model == "hgb":
        estimator = HistGradientBoostingClassifier(max_depth=4, random_state=SEED)
        grid = HGB_GRID
    else:
        estimator = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)
        grid = RF_GRID
    pipe = Pipeline([("prep", prep), ("model", estimator)])
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=_skf(), n_jobs=1)
    y = y_codes(train, target)
    search.fit(train, y)
    fitted = search.best_estimator_
    if list(fitted.named_steps["model"].classes_) != list(range(len(spec["classes"]))):
        raise RuntimeError(f"{model} classes_ are not the expected codes")
    chosen = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    i = search.best_index_
    return Predictor(
        target, feature_set, model, list(spec["classes"]), MODEL_LABELS[model], chosen,
        float(search.cv_results_["mean_test_score"][i]), float(search.cv_results_["std_test_score"][i]),
        tree_cat + num, fitted.predict_proba,
    )


_BUNDLE: FitBundle | None = None
_BUNDLE_KEY: tuple | None = None


def fit_all(df: pd.DataFrame) -> FitBundle:
    """Split once, fit every (target, feature set, model) and pick the best by CV macro F1.

    Memoised on the module for the frame it was last given, so one run fits once.
    """
    global _BUNDLE, _BUNDLE_KEY
    key = (id(df), df.shape)
    if _BUNDLE is not None and _BUNDLE_KEY == key:
        return _BUNDLE
    train_idx, test_idx = split(df)
    train = prepare(df.iloc[train_idx])
    test = prepare(df.iloc[test_idx])
    predictors: dict[tuple[str, str, str], Predictor] = {}
    logits: dict[str, Any] = {}
    fit_methods: dict[str, str] = {}
    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            predictors[(target, feature_set, "majority")] = _fit_majority(target, feature_set, train)
            logistic, res, method = _fit_logistic(target, feature_set, train)
            predictors[(target, feature_set, "logistic")] = logistic
            logits[f"{target}_{feature_set}"] = res
            fit_methods[f"{target}_{feature_set}"] = method
            for model in ("hgb", "rf"):
                predictors[(target, feature_set, model)] = _fit_tree(model, target, feature_set, train)
    # max() keeps the first of equal scores, so ties go by MODELS order.
    best = {
        target: {
            feature_set: max(MODELS, key=lambda m: predictors[(target, feature_set, m)].cv_macro_f1_mean)
            for feature_set in FEATURE_SETS
        }
        for target in TARGETS
    }
    _BUNDLE = FitBundle(train, test, train_idx, test_idx, predictors, best, logits, fit_methods)
    _BUNDLE_KEY = key
    return _BUNDLE


# ---------------------------------------------------------------------------
# model_metrics.json and calibration.json
# ---------------------------------------------------------------------------


def _interval(cell: dict) -> dict:
    return {k: (None if cell[k] is None else round6(cell[k])) for k in ("value", "lo", "hi")}


def build_model_metrics(bundle: FitBundle) -> dict:
    test = bundle.test
    results = []
    for target, spec in TARGETS.items():
        classes = list(spec["classes"])
        y = y_labels(test, target)
        for feature_set, fs in FEATURE_SETS.items():
            for model in MODELS:
                pred = bundle.predictor(target, feature_set, model)
                proba = pred.predict_proba(test)
                scored = proba[:, 1] if target == "binary" else proba
                ci = bootstrap_ci(y, scored, classes, n=BOOTSTRAP_RESAMPLES, seed=SEED)
                test_block: dict[str, dict | None] = {m: _interval(ci[m]) for m in METRICS}
                if model == "majority":
                    # A constant prediction has no ranking to score.
                    test_block["roc_auc"] = None
                    test_block["pr_auc"] = None
                y_pred = np.asarray(classes, dtype=object)[predict_class(proba)]
                results.append({
                    "target": target,
                    "feature_set": feature_set,
                    "model": model,
                    "leaks": bool(fs["leaks"]),
                    "chosen_params": pred.chosen_params,
                    "cv": {"macro_f1_mean": round6(pred.cv_macro_f1_mean), "macro_f1_sd": round6(pred.cv_macro_f1_sd)},
                    "test": test_block,
                    "confusion": confusion(y, y_pred, classes),
                    "classes": classes,
                })
    return {
        "schema": "model_metrics.v1",
        "id": "model_metrics",
        "generated_at": now_iso(),
        "seed": SEED,
        "split": {
            "n_train": int(len(bundle.train)),
            "n_test": int(len(test)),
            "test_share": TEST_SHARE,
            "stratified_on": "outcome",
        },
        "cv": {"folds": CV_FOLDS, "selection_metric": "macro_f1"},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "level": 0.95},
        "feature_sets": {
            name: {
                "label": fs["label"],
                "leaks": bool(fs["leaks"]),
                "variables": [variable_label(v) for v in list(fs["tree_cat"]) + list(fs["num"])],
            }
            for name, fs in FEATURE_SETS.items()
        },
        "targets": {
            "binary": {
                "label": TARGETS["binary"]["label"],
                "classes": list(TARGETS["binary"]["classes"]),
                "positive": TARGETS["binary"]["positive"],
                "test_positive_share": round6(test["dropout"].mean()),
            },
            "multiclass": {
                "label": TARGETS["multiclass"]["label"],
                "classes": list(TARGETS["multiclass"]["classes"]),
            },
        },
        "models": list(MODELS),
        "model_labels": dict(MODEL_LABELS),
        "results": results,
        "best": {t: dict(v) for t, v in bundle.best.items()},
    }


def build_calibration(bundle: FitBundle) -> dict:
    test = bundle.test
    y = test["dropout"].to_numpy(dtype=int)
    curves = []
    for feature_set, fs in FEATURE_SETS.items():
        for model in ("logistic", "hgb", "rf"):
            p = bundle.predictor("binary", feature_set, model).predict_proba(test)[:, 1]
            cal = calibration(y, p, bins=CALIBRATION_BINS)
            curves.append({
                "feature_set": feature_set,
                "model": model,
                "leaks": bool(fs["leaks"]),
                "brier": cal["brier"],
                "points": cal["points"],
            })
    return {
        "schema": "calibration.v1",
        "id": "calibration",
        "generated_at": now_iso(),
        "bins": CALIBRATION_BINS,
        "strategy": "quantile",
        "target": "binary",
        "curves": curves,
    }


# ---------------------------------------------------------------------------
# odds_ratios.json
# ---------------------------------------------------------------------------


@dataclass
class _Coefs:
    """One equation's statistics, each a Series indexed by patsy term name."""

    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    lo: pd.Series
    hi: pd.Series


def _finite4(x: float) -> float | None:
    x = float(x)
    return round4(x) if np.isfinite(x) else None


def _finite_sig3(p: float) -> float | None:
    p = float(p)
    return sig3(p) if np.isfinite(p) else None


def _exp(x: float) -> float:
    with np.errstate(over="ignore"):
        return float(np.exp(x))


def _unstable(n_total: int, events_total: int, n_level: int, events_level: int) -> bool:
    """True when any cell of the level-vs-outcome 2x2 table is thinner than UNSTABLE_MIN_CELL."""
    cells = (
        events_level,
        n_level - events_level,
        events_total - events_level,
        (n_total - n_level) - (events_total - events_level),
    )
    return min(cells) < UNSTABLE_MIN_CELL


def _separated(n_level: int, events_level: int) -> bool:
    """True when every row at the level shares one outcome: the coefficient has no finite estimate."""
    return events_level == 0 or events_level == n_level


def _counts(frame: pd.DataFrame, variable: str, level: str, events: np.ndarray) -> tuple[int, int]:
    mask = (frame[variable] == level).to_numpy()
    return int(mask.sum()), int(events[mask].sum())


def _parse_terms(names, covs: list[str]) -> dict[str, list[tuple[str, str]]]:
    parsed: dict[str, list[tuple[str, str]]] = {c: [] for c in covs}
    for name in names:
        if name == "Intercept":
            continue
        m = TERM_RE.match(name)
        if m is None:
            raise RuntimeError(f"unparseable statsmodels term name: {name!r}")
        variable, _ref, level = m.groups()
        if variable not in parsed or level not in LEVELS[variable]:
            raise RuntimeError(f"unexpected term {name!r}")
        parsed[variable].append((level, name))
    return parsed


def _terms(coefs: _Coefs, frame: pd.DataFrame, events: np.ndarray, covs: list[str]) -> list[dict]:
    """Term rows: one synthesised reference row per covariate, then its fitted levels in design order.

    `frame` and `events` are the rows the equation compares (every training row for a binary
    model; the contrast class plus the base class for a multinomial contrast), so `n_level`,
    `events_level` and `unstable` describe the 2x2 table behind that odds ratio.
    """
    n_total = len(frame)
    events_total = int(events.sum())
    parsed = _parse_terms(coefs.params.index, covs)
    terms: list[dict] = []
    for cov in covs:
        reference = REFERENCE[cov]
        n_ref, ev_ref = _counts(frame, cov, reference, events)
        terms.append({
            "variable": cov,
            "level": reference,
            "reference": reference,
            "label": LABELS[cov][reference],
            "or": 1.0,
            "lo": None,
            "hi": None,
            "coef": 0.0,
            "se": None,
            "p_value": None,
            "n_level": n_ref,
            "events_level": ev_ref,
            "is_reference": True,
            "unstable": _unstable(n_total, events_total, n_ref, ev_ref),
        })
        for level, name in parsed[cov]:
            n_level, events_level = _counts(frame, cov, level, events)
            coef = float(coefs.params[name])
            if _separated(n_level, events_level):
                stats = {"or": None, "lo": None, "hi": None, "coef": None, "se": None, "p_value": None}
            else:
                stats = {
                    "or": _finite4(_exp(coef)),
                    "lo": _finite4(_exp(coefs.lo[name])),
                    "hi": _finite4(_exp(coefs.hi[name])),
                    "coef": _finite4(coef),
                    "se": _finite4(coefs.bse[name]),
                    "p_value": _finite_sig3(coefs.pvalues[name]),
                }
            terms.append({
                "variable": cov,
                "level": level,
                "reference": reference,
                "label": LABELS[cov][level],
                **stats,
                "n_level": n_level,
                "events_level": events_level,
                "is_reference": False,
                "unstable": _unstable(n_total, events_total, n_level, events_level),
            })
    return terms


def _binary_model_obj(bundle: FitBundle, key: str, feature_set: str) -> dict:
    res = bundle.logits[key]
    train = bundle.train
    covs = list(FEATURE_SETS[feature_set]["cat"])
    ci = res.conf_int()
    coefs = _Coefs(res.params, res.bse, res.pvalues, ci[0], ci[1])
    events = train["dropout"].to_numpy(dtype=int)
    return {
        "label": ODDS_RATIO_LABELS[key],
        "target": "binary",
        "feature_set": feature_set,
        "leaks": bool(FEATURE_SETS[feature_set]["leaks"]),
        "covariates": covs,
        "n_obs": int(res.nobs),
        "n_events": int(events.sum()),
        "converged": bool(res.mle_retvals["converged"]),
        "pseudo_r2_mcfadden": round4(res.prsquared),
        "intercept": round4(res.params["Intercept"]),
        "terms": _terms(coefs, train, events, covs),
    }


def _multiclass_model_obj(bundle: FitBundle, key: str, feature_set: str) -> dict:
    res = bundle.logits[key]
    train = bundle.train
    covs = list(FEATURE_SETS[feature_set]["cat"])
    ci = res.conf_int()
    contrasts = {}
    for cls in OUTCOMES:
        if cls == MULTICLASS_BASE:
            continue
        code = OUTCOME_CODE[cls]
        col = code - 1  # params columns are 0-based contrast indices over the non-base codes
        ci_eq = ci.xs(str(code), level=0)
        coefs = _Coefs(res.params[col], res.bse[col], res.pvalues[col], ci_eq["lower"], ci_eq["upper"])
        scope = train["outcome"].isin([cls, MULTICLASS_BASE]).to_numpy()
        frame = train[scope]
        events = (frame["outcome"] == cls).to_numpy().astype(int)
        contrasts[cls] = {
            "intercept": round4(res.params.loc["Intercept", col]),
            "terms": _terms(coefs, frame, events, covs),
        }
    return {
        "label": ODDS_RATIO_LABELS[key],
        "target": "multiclass",
        "feature_set": feature_set,
        "leaks": bool(FEATURE_SETS[feature_set]["leaks"]),
        "base": MULTICLASS_BASE,
        "covariates": covs,
        "n_obs": int(res.nobs),
        "converged": bool(res.mle_retvals["converged"]),
        "pseudo_r2_mcfadden": round4(res.prsquared),
        "contrasts": contrasts,
    }


def build_odds_ratios(bundle: FitBundle) -> dict:
    return {
        "schema": "odds_ratios.v1",
        "id": "odds_ratios",
        "generated_at": now_iso(),
        "fitted_on": "train",
        "models": {
            "binary_enrolment": _binary_model_obj(bundle, "binary_enrolment", "enrolment"),
            "binary_after_s1": _binary_model_obj(bundle, "binary_after_s1", "after_s1"),
            "multiclass_enrolment": _multiclass_model_obj(bundle, "multiclass_enrolment", "enrolment"),
        },
    }


# ---------------------------------------------------------------------------
# permutation_importance.json
# ---------------------------------------------------------------------------

PERMUTATION_SCORING = {"binary": "roc_auc", "multiclass": "macro_f1"}


def _score_fn(target: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if target == "binary":
        return lambda y, proba: float(roc_auc_score(y, proba[:, 1]))
    return lambda y, proba: float(f1_score(y, proba.argmax(axis=1), average="macro", zero_division=0))


def build_permutation_importance(bundle: FitBundle) -> dict:
    """Importance of each raw variable to the best model per (target, feature set), on the test rows.

    One generator serves the whole build; a variable is permuted as a whole column before encoding,
    so a categorical's dummies move together.
    """
    rng = np.random.default_rng(SEED)
    test = bundle.test
    results = []
    for target in TARGETS:
        y = y_codes(test, target)
        score = _score_fn(target)
        for feature_set, fs in FEATURE_SETS.items():
            pred = bundle.best_predictor(target, feature_set)
            baseline = score(y, pred.predict_proba(test))
            frame = test.copy()
            features = []
            for variable in pred.variables:
                original = frame[variable].copy()
                drops = []
                for _ in range(PERMUTATION_REPEATS):
                    frame[variable] = original.array[rng.permutation(len(frame))]
                    if frame[variable].dtype != original.dtype:
                        raise RuntimeError(f"permuting {variable} changed its dtype")
                    drops.append(baseline - score(y, pred.predict_proba(frame)))
                frame[variable] = original
                features.append({
                    "variable": variable,
                    "label": variable_label(variable),
                    "mean": round6(np.mean(drops)),
                    "sd": round6(np.std(drops)),
                })
            features.sort(key=lambda f: -f["mean"])
            results.append({
                "target": target,
                "feature_set": feature_set,
                "model": pred.model,
                "leaks": bool(fs["leaks"]),
                "baseline_score": round6(baseline),
                "features": features,
            })
    return {
        "schema": "permutation_importance.v1",
        "id": "permutation_importance",
        "generated_at": now_iso(),
        "repeats": PERMUTATION_REPEATS,
        "seed": SEED,
        "scoring": dict(PERMUTATION_SCORING),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_models(df: pd.DataFrame) -> list[Path]:
    bundle = fit_all(df)
    return [
        write_json("model_metrics.json", build_model_metrics(bundle)),
        write_json("calibration.json", build_calibration(bundle)),
        write_json("odds_ratios.json", build_odds_ratios(bundle)),
        write_json("permutation_importance.json", build_permutation_importance(bundle)),
    ]
