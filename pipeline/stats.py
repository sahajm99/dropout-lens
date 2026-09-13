"""Statistical helpers: rounding, Wilson interval, Cramer's V, eta and epsilon squared, Holm, mean CI."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from pipeline.constants import SUPPRESS_BELOW, Z95


def round6(x: float) -> float:
    return float(round(float(x), 6))


def round4(x: float) -> float:
    return float(round(float(x), 4))


def round10(x: float) -> float:
    return float(round(float(x), 10))


def sig3(p: float) -> float:
    """Round a p-value to 3 significant digits; underflow to exactly 0.0."""
    p = float(p)
    if not math.isfinite(p):
        raise ValueError(f"non-finite p-value: {p}")
    if p < 1e-300:
        return 0.0
    return float(f"{p:.3g}")


def wilson(events: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if events < 0 or events > n:
        raise ValueError("events must be within [0, n]")
    p = events / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # At the degenerate boundaries (p=0 or p=1) the Wilson interval touches 0/1
    # exactly in real arithmetic; pin it there rather than leave a float residue.
    if events == 0:
        lo = 0.0
    if events == n:
        hi = 1.0
    return (lo, hi)


def share_cell(events: int, n: int) -> dict | None:
    """{"p", "lo", "hi"} rounded to 6 decimals, or None when n is under SUPPRESS_BELOW."""
    if n < SUPPRESS_BELOW:
        return None
    lo, hi = wilson(events, n)
    return {"p": round6(events / n), "lo": round6(lo), "hi": round6(hi)}


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    """Cramer's V and the Bergsma (2013) bias-corrected V for an r x k contingency table."""
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or min(t.shape) < 2:
        raise ValueError("table must be at least 2 x 2")
    n = float(t.sum())
    if n <= 1:
        raise ValueError("table must hold more than one observation")
    chi2 = float(sps.chi2_contingency(t, correction=False)[0])
    r, k = t.shape
    phi2 = chi2 / n
    v = math.sqrt(phi2 / min(r - 1, k - 1))
    phi2_corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    v_corr = math.sqrt(phi2_corr / denom) if denom > 0 else 0.0
    return (float(v), float(v_corr))


def eta_squared(groups: list[np.ndarray]) -> float:
    """Between-group sum of squares over total sum of squares (one-way ANOVA effect size)."""
    arrays = [np.asarray(g, dtype=float) for g in groups]
    pooled = np.concatenate(arrays)
    grand = float(pooled.mean())
    ss_between = sum(len(a) * (float(a.mean()) - grand) ** 2 for a in arrays)
    ss_total = float(((pooled - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    return float(ss_between / ss_total)


def epsilon_squared(h: float, n: int) -> float:
    """Kruskal-Wallis effect size from the H statistic and total n."""
    return float(h * (n + 1) / (n * n - 1))


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    if len(p) == 0:
        return []
    adjusted = multipletests(list(p), alpha=0.05, method="holm")[1]
    return [float(x) for x in adjusted]


def mean_ci(x: pd.Series, z: float = Z95) -> tuple[float, float, float]:
    """Sample mean with a normal-approximation confidence interval."""
    x = pd.Series(x).astype(float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return (m, m - z * se, m + z * se)
