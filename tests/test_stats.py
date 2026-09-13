import numpy as np
import pytest

from pipeline.stats import cramers_v, eta_squared, holm, share_cell, wilson


def test_wilson_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0
    assert abs(hi - 0.2775) < 1e-3
    lo, hi = wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and abs(lo - 0.7225) < 1e-3


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson(1, 0)
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_cramers_v_plain_and_bergsma_corrected():
    v, v_corr = cramers_v(np.array([[20, 10], [10, 20]]))
    assert abs(v - 1 / 3) < 1e-6
    assert abs(v_corr - 0.3095) < 1e-3


def test_eta_squared_three_groups():
    groups = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([3, 4, 5])]
    assert abs(eta_squared(groups) - 0.5) < 1e-9


def test_holm_step_down():
    adjusted = holm([0.01, 0.04, 0.03])
    assert len(adjusted) == 3
    assert all(abs(a - b) < 1e-9 for a, b in zip(adjusted, [0.03, 0.06, 0.06]))


def test_share_cell_suppresses_below_30():
    assert share_cell(3, 29) is None
    cell = share_cell(15, 30)
    assert cell["p"] == 0.5
    assert cell["lo"] <= cell["p"] <= cell["hi"]
