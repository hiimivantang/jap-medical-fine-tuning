"""Pre-registered decision rules must not pass vacuously (missing splits/tasks/predictions)."""

from jmedft.evaluation.run import hypotheses

K1 = "E3_qwen3_8b_lora__vs__E1_qwen3_8b"


def _c(diff, p_holm, lo=None, hi=None, incomplete=False):
    lo = diff - 0.01 if lo is None else lo
    hi = diff + 0.01 if hi is None else hi
    return {"diff": diff, "ci95": [lo, hi], "p_bootstrap": p_holm, "p_holm": p_holm, "incomplete": incomplete}


SIG = {t: _c(0.05, 0.001) for t in ("A_entities", "B_doctype", "C_structured")}


def test_h1_not_supported_when_test_xgen_missing():
    h = hypotheses({K1: {"test_synth": SIG}})
    assert h["H1"]["supported"] is False and h["H1"]["all_endpoints_present"] is False


def test_h1_supported_when_significant_and_same_direction():
    assert hypotheses({K1: {"test_synth": SIG, "test_xgen": SIG}})["H1"]["supported"] is True


def test_h1_uses_holm_adjusted_p():
    synth = {"A_entities": _c(0.05, 0.03), "B_doctype": _c(0.05, 0.08), "C_structured": _c(0.0, 0.5)}
    assert hypotheses({K1: {"test_synth": synth, "test_xgen": SIG}})["H1"]["supported"] is False


def test_h1_blocked_by_incomplete_predictions():
    xgen = dict(SIG, C_structured=_c(0.05, 0.001, incomplete=True))
    assert hypotheses({K1: {"test_synth": SIG, "test_xgen": xgen}})["H1"]["supported"] is False


def test_h3_requires_all_general_tasks():
    only_one = {"gen_jcqa": _c(0.0, 1.0, lo=-0.01, hi=0.01)}
    assert hypotheses({K1: {"test_general": only_one}})["H3"]["supported"] is False
    all_three = {t: _c(0.0, 1.0, lo=-0.01, hi=0.01) for t in ("gen_jcqa", "gen_jnli", "gen_jmmlu_med")}
    assert hypotheses({K1: {"test_general": all_three}})["H3"]["supported"] is True
    one_bad = dict(all_three, gen_jnli=_c(-0.02, 0.01, lo=-0.03, hi=-0.01))
    assert hypotheses({K1: {"test_general": one_bad}})["H3"]["supported"] is False


def test_subsampled_condition_pairs_on_shared_items_only():
    import numpy as np

    from jmedft.evaluation import run

    full = {"subsampled": False, "n_missing_predictions": 0,
            "items": [{"id": f"i{k}", "correct": k % 2 == 0, "strict_correct": k % 2 == 0} for k in range(10)]}
    sub = {"subsampled": True, "n_missing_predictions": 0,
           "items": [{"id": f"i{k}", "correct": True, "strict_correct": True} for k in (0, 1, 2, 3)]}
    row = run._paired("gen_jcqa", sub, full)
    assert row["n_paired"] == 4 and row["incomplete"] is False
    # Shared items i0..i3: subsample all correct (1.0); full run correct on i0, i2 (0.5) → diff +0.5.
    assert np.isclose(row["diff"], 0.5)
