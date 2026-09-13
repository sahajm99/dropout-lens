import numpy as np
import pytest

from pipeline.constants import SEED
from pipeline.metrics import METRICS, bootstrap_ci, calibration, metric_suite


def test_metric_suite_known_values():
    # Predictions at 0.5 are [0, 0, 0, 1] against [0, 0, 1, 1].
    m = metric_suite(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), [0, 1])
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert m["macro_f1"] == pytest.approx(0.733333, abs=1e-6)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["pr_auc"] == pytest.approx(0.833333, abs=1e-6)


def test_bootstrap_ci_is_deterministic_and_brackets_the_value():
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 30)
    p = np.clip(0.35 * y + 0.15 + 0.5 * rng.random(60), 0, 1)
    a = bootstrap_ci(y, p, [0, 1], n=300, seed=SEED)
    b = bootstrap_ci(y, p, [0, 1], n=300, seed=SEED)
    assert a == b
    assert list(a) == METRICS
    for m in METRICS:
        assert 0 <= a[m]["lo"] <= a[m]["value"] <= a[m]["hi"] <= 1, m


def test_calibration_points_cover_every_row():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 * y + 0.2 + 0.5 * rng.random(200), 0, 1)
    cal = calibration(y, p, bins=10)
    assert sum(pt["n"] for pt in cal["points"]) == len(y)
    assert 0 <= cal["brier"] <= 1
    for pt in cal["points"]:
        assert 0 <= pt["fraction_positive"] <= 1 and 0 <= pt["mean_predicted"] <= 1
    # Heavy ties still place every row in exactly one bin.
    tied = calibration(y, np.where(y == 1, 0.8, 0.2), bins=10)
    assert sum(pt["n"] for pt in tied["points"]) == len(y)
