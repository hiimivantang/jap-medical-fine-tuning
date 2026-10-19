"""Tests for jmedft.evaluation.stats. Every expected value is written out by hand."""

import math
import time

import numpy as np
import pytest

from jmedft.evaluation.stats import align_by_id, bootstrap_ci, holm, mcnemar_exact, paired_bootstrap_diff


def mean_x(data, idx):
    x = data["x"]
    return float(np.mean(x if idx is None else x[idx]))


def make(values, ids=None):
    values = np.asarray(values, dtype=float)
    return {"ids": ids if ids is not None else [f"item{i:04d}" for i in range(len(values))], "x": values}


# --- bootstrap_ci ------------------------------------------------------------------------------------------

def test_bootstrap_ci_deterministic_and_brackets_point():
    data = make([0.0, 1.0] * 100)
    first = bootstrap_ci(data, mean_x, n_resamples=2000, seed=0)
    assert bootstrap_ci(data, mean_x, n_resamples=2000, seed=0) == first
    point, lo, hi = first
    assert point == 0.5
    assert lo < point < hi


def test_bootstrap_ci_constant_collapses():
    assert bootstrap_ci(make([0.25] * 20), mean_x) == (0.25, 0.25, 0.25)


def test_bootstrap_ci_single_item():
    assert bootstrap_ci(make([3.0]), mean_x) == (3.0, 3.0, 3.0)


def test_bootstrap_ci_two_items():
    # resampled means are 0 (p=1/4), 0.5 (p=1/2) and 1 (p=1/4): the 2.5th/97.5th percentiles are 0 and 1
    assert bootstrap_ci(make([0.0, 1.0]), mean_x) == (0.5, 0.0, 1.0)


def test_bootstrap_ci_empty_is_nan():
    assert all(math.isnan(v) for v in bootstrap_ci(make([]), mean_x))


def test_bootstrap_ci_smaller_alpha_gives_wider_interval():
    data = make([0.0, 1.0] * 50)
    _, lo95, hi95 = bootstrap_ci(data, mean_x, n_resamples=2000)
    _, lo50, hi50 = bootstrap_ci(data, mean_x, n_resamples=2000, alpha=0.5)
    assert lo95 < lo50 <= 0.5 <= hi50 < hi95


def test_bootstrap_statistic_contract():
    seen = []

    def fn(data, idx):
        if idx is None:
            return 42.0
        seen.append(idx)
        return 0.0

    assert bootstrap_ci(make([1.0, 2.0, 3.0]), fn, n_resamples=50) == (42.0, 0.0, 0.0)
    assert len(seen) == 50
    assert all(i.shape == (3,) and i.min() >= 0 and i.max() <= 2 for i in seen)


def test_bootstrap_ci_ignores_undefined_resamples():
    # the statistic is undefined (NaN) on resamples without a positive item; the finite resampled values are
    # 0.5 (p=2/3) and 1 (p=1/3), so the percentile interval is [0.5, 1]
    def mean_if_any_positive(data, idx):
        x = data["x"] if idx is None else data["x"][idx]
        return float(np.mean(x)) if x.any() else math.nan

    assert bootstrap_ci(make([0.0, 1.0]), mean_if_any_positive) == (0.5, 0.5, 1.0)


def test_bootstrap_ci_fast_enough():
    data = make(np.random.default_rng(0).random(1000))
    t0 = time.perf_counter()
    bootstrap_ci(data, mean_x, n_resamples=10000, seed=0)
    assert time.perf_counter() - t0 < 5.0


# --- align_by_id -------------------------------------------------------------------------------------------

def test_align_by_id_reorders_arrays_and_keeps_scalars():
    a = {"ids": ["c", "a", "b"], "x": np.array([3.0, 1.0, 2.0]),
         "counts": np.array([[30, 3], [10, 1], [20, 2]]), "n_types": 2}
    b = {"ids": ["b", "c", "a"], "x": np.array([20.0, 30.0, 10.0])}
    a2, b2 = align_by_id(a, b)
    assert a2["ids"] == ["a", "b", "c"]
    assert b2["ids"] == ["a", "b", "c"]
    assert a2["x"].tolist() == [1.0, 2.0, 3.0]
    assert a2["counts"].tolist() == [[10, 1], [20, 2], [30, 3]]
    assert a2["n_types"] == 2
    assert b2["x"].tolist() == [10.0, 20.0, 30.0]
    assert a["ids"] == ["c", "a", "b"]
    assert a["x"].tolist() == [3.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("ids_a", "ids_b"),
    [(["a", "b"], ["a", "c"]), (["a", "b"], ["a", "b", "c"]), (["a", "a"], ["a", "a"])],
)
def test_align_by_id_rejects_mismatched_or_duplicate_ids(ids_a, ids_b):
    with pytest.raises(ValueError):
        align_by_id(make([0.0] * len(ids_a), ids_a), make([0.0] * len(ids_b), ids_b))


# --- paired_bootstrap_diff ---------------------------------------------------------------------------------

def test_paired_identical_data():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    assert paired_bootstrap_diff(make(values), make(values), mean_x) == (0.0, 0.0, 0.0, 1.0)


def test_paired_uses_same_resampled_items():
    # b = a + 1 on every item, so every paired resampled difference is exactly -1 (n = 8 keeps the means
    # exact in floating point); P(d >= 0) = 0 gives p = 0. Unpaired resampling would spread d out.
    x = np.arange(8, dtype=float)
    assert paired_bootstrap_diff(make(x), make(x + 1), mean_x) == (-1.0, -1.0, -1.0, 0.0)


def test_paired_two_items():
    # a - b is 1 on item x and 0 on item y: resampled differences are 0 (p=1/4), 0.5 (1/2) and 1 (1/4), so
    # the interval is [0, 1] and p = 2 * P(d <= 0) = 0.5 up to resampling noise
    diff, lo, hi, p = paired_bootstrap_diff(make([1.0, 0.0], ["x", "y"]), make([0.0, 0.0], ["x", "y"]), mean_x)
    assert (diff, lo, hi) == (0.5, 0.0, 1.0)
    assert p == pytest.approx(0.5, abs=0.03)


def test_paired_deterministic_and_brackets_diff():
    rng = np.random.default_rng(7)
    a, b = make(rng.random(100)), make(rng.random(100))
    first = paired_bootstrap_diff(a, b, mean_x, n_resamples=1000, seed=3)
    assert paired_bootstrap_diff(a, b, mean_x, n_resamples=1000, seed=3) == first
    diff, lo, hi, p = first
    assert lo <= diff <= hi
    assert 0.0 <= p <= 1.0


def test_paired_requires_aligned_ids():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(make([1.0, 0.0], ["x", "y"]), make([1.0, 0.0], ["y", "x"]), mean_x)


# --- mcnemar_exact -----------------------------------------------------------------------------------------

def test_mcnemar_counts_discordant_pairs():
    # 8 items only a gets right, 2 only b gets right, 5 both right, 3 both wrong.
    # Two-sided exact binomial: 2 * P(X <= 2 | n=10, p=0.5) = 2 * (1 + 10 + 45) / 1024 = 0.109375
    a = [True] * 8 + [False] * 2 + [True] * 5 + [False] * 3
    b = [False] * 8 + [True] * 2 + [True] * 5 + [False] * 3
    res = mcnemar_exact(a, b)
    assert (res["b"], res["c"]) == (8, 2)
    assert res["p_value"] == pytest.approx(0.109375)
    swapped = mcnemar_exact(b, a)
    assert (swapped["b"], swapped["c"]) == (2, 8)
    assert swapped["p_value"] == pytest.approx(0.109375)


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (0, 0, 1.0),  # no discordant pairs
        (5, 5, 1.0),  # observed count is the mode of Binomial(10, 0.5)
        (1, 0, 1.0),  # both outcomes of Binomial(1, 0.5) have probability 1/2
        (6, 0, 0.03125),  # 2 * (1/2)^6
        (0, 6, 0.03125),
    ],
)
def test_mcnemar_exact_p_values(b, c, p):
    correct_a = [True] * b + [False] * c + [True, False]
    correct_b = [False] * b + [True] * c + [True, False]
    res = mcnemar_exact(correct_a, correct_b)
    assert (res["b"], res["c"]) == (b, c)
    assert res["p_value"] == pytest.approx(p)


def test_mcnemar_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mcnemar_exact([True, False], [True])


# --- holm --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("pvalues", "expected"),
    [
        ({"x": 0.01, "y": 0.04, "z": 0.03}, {"x": 0.03, "y": 0.06, "z": 0.06}),
        # 0.02 * 2 = 0.04 is raised to 0.057 to keep the step-down sequence monotone
        ({"p1": 0.001, "p2": 0.02, "p3": 0.019, "p4": 0.5}, {"p1": 0.004, "p2": 0.057, "p3": 0.057, "p4": 0.5}),
        ({"a": 0.6, "b": 0.7}, {"a": 1.0, "b": 1.0}),
        ({"a": 0.02, "b": 0.02}, {"a": 0.04, "b": 0.04}),
        ({"only": 0.2}, {"only": 0.2}),
        ({}, {}),
    ],
)
def test_holm(pvalues, expected):
    assert holm(pvalues) == pytest.approx(expected)


def test_holm_nan_is_kept_and_excluded_from_family_size():
    out = holm({"a": math.nan, "b": 0.01, "c": 0.04})
    assert math.isnan(out["a"])
    assert (out["b"], out["c"]) == pytest.approx((0.02, 0.04))
