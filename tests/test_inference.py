"""Decoding parity: every local no-think condition must send identical decoding parameters, and the
inference prompt must equal the LLaMA-Factory training prompt."""

from jmedft.inference.runner import _request_kwargs
from jmedft.utils import REPO_ROOT, load_yaml

CONDS = load_yaml(REPO_ROOT / "experiments/01_finetuning/configs/conditions.yaml")["conditions"]
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def _kw(name: str) -> dict:
    return _request_kwargs(dict(CONDS[name], name=name), MSGS)


def test_local_no_think_conditions_share_decoding():
    names = ["E1_qwen3_8b", "E2_swallow_8b", "E2m_medical_swallow_8b", "E3_qwen3_8b_lora"]
    kws = [_kw(n) for n in names]
    for kw in kws:
        kw.pop("model")
    assert all(kw == kws[0] for kw in kws)
    assert kws[0]["temperature"] == 0.0
    assert kws[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_e1_and_e3_differ_only_in_served_model():
    e1, e3 = _kw("E1_qwen3_8b"), _kw("E3_qwen3_8b_lora")
    assert e1.pop("model") == "qwen3-8b" and e3.pop("model") == "jmed-lora"
    assert e1 == e3


def test_thinking_condition_enables_reasoning():
    kw = _kw("E2t_swallow_8b_think")
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert kw["max_tokens"] >= 4096


def test_openai_conditions_do_not_send_vllm_only_params():
    for name in ("E4_gpt41", "E4b_gpt54mini"):
        kw = _kw(name)
        assert "extra_body" not in kw and "max_tokens" not in kw
        assert "max_completion_tokens" in kw
    assert "temperature" not in _kw("E4b_gpt54mini")  # reasoning models reject temperature


def test_inference_prompt_matches_training_prompt():
    from transformers import AutoTokenizer

    from jmedft.tasks import build_messages

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
    msgs = build_messages("A_entities", text="患者は昨日から頭痛と発熱を訴えている。")
    infer = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "{}"}], tokenize=False,
                                   enable_thinking=False)
    assert full.startswith(infer)
    assert infer.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
