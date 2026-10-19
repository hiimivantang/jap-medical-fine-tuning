"""Bootstrap confidence intervals and paired significance tests (docs/INTERFACES.md, "Statistics").

A *data* object is a dict of equal-length per-item numpy arrays plus ``"ids"`` (list[str]); any other value is
a scalar carried along unchanged. A statistic ``fn(data, idx)`` returns the metric over the rows ``idx``
(``None`` = all rows), so the same function serves the point estimate and every bootstrap resample.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest

Data = dict[str, Any]
Statistic = Callable[[Data, np.ndarray | None], float]


def _resample_idx(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples, n) item indices drawn with replacement; identical for identical (n, seed)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def _percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Percentile interval of the non-NaN resampled values (NaN, NaN if there are none)."""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(data: Data, fn: Statistic, n_resamples: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap over items: (point estimate, lower, upper) of the 1 - alpha interval."""
    n = len(data["ids"])
    if n == 0:
        return math.nan, math.nan, math.nan
    idx = _resample_idx(n, n_resamples, seed)
    boot = np.fromiter((fn(data, row) for row in idx), dtype=float, count=n_resamples)
    lo, hi = _percentile_ci(boot, alpha)
    return float(fn(data, None)), lo, hi


def _reorder(data: Data, ids: list[str]) -> Data:
    pos = {item_id: k for k, item_id in enumerate(data["ids"])}
    order = np.array([pos[item_id] for item_id in ids], dtype=np.intp)
    out: Data = {}
    for key, val in data.items():
        if key == "ids":
            out[key] = list(ids)
        elif isinstance(val, np.ndarray) and val.ndim > 0:
            if len(val) != len(order):
                raise ValueError(f"array {key!r} has {len(val)} rows but there are {len(order)} ids")
            out[key] = val[order]
        else:
            out[key] = val
    return out


def align_by_id(a: Data, b: Data) -> tuple[Data, Data]:
    """Reorder both data objects by their sorted ids. The id sets must be identical and duplicate-free."""
    ids_a, ids_b = list(a["ids"]), list(b["ids"])
    for name, ids in (("a", ids_a), ("b", ids_b)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids in {name}")
    if set(ids_a) != set(ids_b):
        unshared = sorted(set(ids_a) ^ set(ids_b))
        raise ValueError(f"id sets differ: {len(unshared)} ids not shared, e.g. {unshared[:3]}")
    ids = sorted(ids_a)
    return _reorder(a, ids), _reorder(b, ids)


def paired_bootstrap_diff(a: Data, b: Data, fn: Statistic, n_resamples: int = 10000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Paired bootstrap of fn(a) - fn(b), resampling the same items for both conditions.

    Returns (diff, lower, upper, p_two_sided) with p = min(1, 2 * min(P(d <= 0), P(d >= 0))) over the
    resampled differences d (p = 1.0 when every d is exactly 0). Inputs must be aligned (see `align_by_id`).
    """
    if list(a["ids"]) != list(b["ids"]):
        raise ValueError("a and b must have identical ids in the same order; call align_by_id first")
    n = len(a["ids"])
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan
    idx = _resample_idx(n, n_resamples, seed)
    d = np.fromiter((fn(a, row) - fn(b, row) for row in idx), dtype=float, count=n_resamples)
    lo, hi = _percentile_ci(d, alpha)
    d = d[~np.isnan(d)]
    if d.size == 0:
        p = math.nan
    elif np.all(d == 0):
        p = 1.0
    else:
        p = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
    return float(fn(a, None) - fn(b, None)), lo, hi, p


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, Any]:
    """Exact McNemar test on paired per-item correctness.

    b = #(a right, b wrong), c = #(a wrong, b right); p = two-sided binomial test of min(b, c) out of b + c
    at 0.5 (1.0 when there are no discordant pairs).
    """
    ca, cb = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    if ca.shape != cb.shape:
        raise ValueError(f"correctness vectors differ in shape: {ca.shape} vs {cb.shape}")
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"b": b, "c": c, "p_value": float(p)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone in the sorted order, capped at 1).

    NaN p-values stay NaN and do not count towards the family size.
    """
    order = sorted((k for k, p in pvalues.items() if not math.isnan(p)), key=lambda k: pvalues[k])
    m = len(order)
    out = {k: math.nan for k in pvalues}
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[k]))
        out[k] = running
    return out
