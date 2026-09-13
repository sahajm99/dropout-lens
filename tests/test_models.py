import pytest

from pipeline.constants import LEVELS, OUTCOMES, REFERENCE
from pipeline.features import FEATURE_SETS, TARGETS
from pipeline.metrics import METRICS
from pipeline.models import MODELS, build_model_metrics, build_odds_ratios, fit_all


@pytest.fixture(scope="module")
def bundle(df):
    return fit_all(df)


@pytest.fixture(scope="module")
def metrics_obj(bundle):
    return build_model_metrics(bundle)


def test_split_is_stratified_and_partitions_the_rows(bundle, df):
    assert len(bundle.test) == 885 and len(bundle.train) == 3539
    full = df["outcome"].value_counts(normalize=True)
    part = bundle.test["outcome"].value_counts(normalize=True)
    for outcome in OUTCOMES:
        assert abs(full[outcome] - part[outcome]) < 0.01, outcome
    train_set, test_set = set(bundle.train_idx.tolist()), set(bundle.test_idx.tolist())
    assert not (train_set & test_set) and len(train_set | test_set) == len(df)


def test_odds_ratio_terms_have_one_reference_and_every_level(bundle):
    obj = build_odds_ratios(bundle)
    assert list(obj["models"]) == ["binary_enrolment", "binary_after_s1", "multiclass_enrolment"]
    blocks = []
    for key, model in obj["models"].items():
        assert model["converged"] is True, key
        if "contrasts" in model:
            assert model["base"] == "Graduate" and list(model["contrasts"]) == ["Dropout", "Enrolled"]
            blocks += [(f"{key}/{c}", model["covariates"], v["terms"]) for c, v in model["contrasts"].items()]
        else:
            blocks.append((key, model["covariates"], model["terms"]))
    assert len(blocks) == 4
    for name, covs, terms in blocks:
        assert [t["variable"] for t in terms if t["is_reference"]] == covs, name
        for cov in covs:
            rows = [t for t in terms if t["variable"] == cov]
            refs = [t for t in rows if t["is_reference"]]
            assert len(refs) == 1, (name, cov)
            assert refs[0]["or"] == 1.0 and refs[0]["level"] == REFERENCE[cov] == refs[0]["reference"]
            assert len(rows) - 1 == len(LEVELS[cov]) - 1, (name, cov)
            assert [t["level"] for t in rows if not t["is_reference"]] == [
                lvl for lvl in LEVELS[cov] if lvl != REFERENCE[cov]
            ], (name, cov)
            for t in rows:
                if t["is_reference"]:
                    continue
                if t["or"] is None:
                    # No finite estimate exists for a separated level: every statistic is null.
                    assert t["unstable"], (name, cov, t["level"])
                    assert all(t[k] is None for k in ("lo", "hi", "coef", "se", "p_value")), (name, cov, t["level"])
                else:
                    assert t["lo"] <= t["or"] <= t["hi"], (name, cov, t["level"])


def test_model_metrics_has_sixteen_bracketed_results(metrics_obj, bundle):
    results = metrics_obj["results"]
    assert len(results) == 16
    assert [(r["target"], r["feature_set"], r["model"]) for r in results] == [
        (t, f, m) for t in TARGETS for f in FEATURE_SETS for m in MODELS
    ]
    for r in results:
        assert r["leaks"] is FEATURE_SETS[r["feature_set"]]["leaks"]
        assert r["classes"] == TARGETS[r["target"]]["classes"]
        assert sum(map(sum, r["confusion"])) == 885
        for m in METRICS:
            cell = r["test"][m]
            if cell is None:
                assert r["model"] == "majority" and m in ("roc_auc", "pr_auc"), (r["model"], m)
                continue
            assert 0 <= cell["lo"] <= cell["value"] <= cell["hi"] <= 1, (r["target"], r["feature_set"], r["model"], m)
        if r["model"] in ("majority", "logistic"):
            assert r["chosen_params"] is None
        else:
            assert set(r["chosen_params"]) == (
                {"learning_rate", "max_iter"} if r["model"] == "hgb" else {"max_depth", "min_samples_leaf"}
            )
    majority = {(r["target"], r["feature_set"]): r for r in results if r["model"] == "majority"}
    not_dropout_share = 1 - bundle.test["dropout"].mean()
    for feature_set in FEATURE_SETS:
        acc = majority[("binary", feature_set)]["test"]["accuracy"]["value"]
        assert abs(acc - not_dropout_share) < 1e-6
    for r in results:
        if r["model"] != "majority":
            base = majority[(r["target"], r["feature_set"])]["test"]["balanced_accuracy"]["value"]
            assert r["test"]["balanced_accuracy"]["value"] > base, (r["target"], r["feature_set"], r["model"])


def test_best_names_the_top_cv_macro_f1(metrics_obj):
    results = metrics_obj["results"]
    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            group = [r for r in results if r["target"] == target and r["feature_set"] == feature_set]
            top = max(r["cv"]["macro_f1_mean"] for r in group)
            chosen = next(r for r in group if r["model"] == metrics_obj["best"][target][feature_set])
            assert chosen["cv"]["macro_f1_mean"] == top
            assert chosen["model"] != "majority"
