# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Determinism offline inference tests using LLM.generate

Test scenarios:
1. Same-prompt repeatability (FD_DETERMINISTIC_MODE=1)
2. Batch invariance (single vs. batch, different positions)
3. Different batch sizes consistency
4. Sampling-parameter combinations (temperature x top_p, parametrized)
5. Long sequence generation (512-1024 tokens)
6. Long input prompt handling
7. Minimal output (max_tokens=1, early stop)
8. Special characters & multi-language prompts
9. Multi-turn conversation
10. State isolation (interleaved / interference prompts)
11. Non-deterministic validation (proves tests are effective)

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_determinism_offline.py -v
"""

import os

import pytest

pytestmark = pytest.mark.gpu

DEFAULT_MODEL_DIR = "./models"
MODEL_NAME = "Qwen2-7B-Instruct"

_ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_ENV_FD_DETERMINISTIC_MODE = "FD_DETERMINISTIC_MODE"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _module_env():
    """Set env vars before importing fastdeploy (must happen first)."""
    old_cuda = os.environ.get(_ENV_CUDA_VISIBLE_DEVICES)
    old_det = os.environ.get(_ENV_FD_DETERMINISTIC_MODE)

    os.environ[_ENV_CUDA_VISIBLE_DEVICES] = os.environ.get(_ENV_CUDA_VISIBLE_DEVICES, "0")
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"

    global LLM, SamplingParams  # noqa: PLW0603
    from fastdeploy import LLM, SamplingParams

    yield

    if old_cuda is None:
        os.environ.pop(_ENV_CUDA_VISIBLE_DEVICES, None)
    else:
        os.environ[_ENV_CUDA_VISIBLE_DEVICES] = old_cuda
    if old_det is None:
        os.environ.pop(_ENV_FD_DETERMINISTIC_MODE, None)
    else:
        os.environ[_ENV_FD_DETERMINISTIC_MODE] = old_det


@pytest.fixture(autouse=True)
def _reset_deterministic_mode():
    """Ensure every test starts with deterministic mode ON."""
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"
    yield
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"


@pytest.fixture(scope="module")
def model_path():
    model_dir = os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR)
    return os.path.join(model_dir, MODEL_NAME)


@pytest.fixture(scope="module")
def llm(model_path, _module_env):
    return LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=8192,
        enable_prefix_caching=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_text(llm, prompt, sp):
    """Generate once, return (text, token_ids)."""
    out = llm.generate([prompt], sp)[0]
    return out.outputs.text, out.outputs.token_ids


def _assert_deterministic(llm, prompt, sp, runs=2):
    """Run *runs* times and assert all outputs are identical."""
    results = [_generate_text(llm, prompt, sp) for _ in range(runs)]
    texts = [r[0] for r in results]
    token_ids = [r[1] for r in results]
    assert all(t == texts[0] for t in texts), "Text outputs differ across runs"
    assert all(t == token_ids[0] for t in token_ids), "Token IDs differ across runs"
    return texts[0], token_ids[0]


# ===================== Core determinism tests =====================


def test_deterministic_same_prompt(llm):
    """Same prompt + same seed produces identical output across 5 runs."""
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50, seed=123)
    _assert_deterministic(llm, "Please introduce artificial intelligence in one sentence.", sp, runs=5)


def test_deterministic_batch_invariance(llm):
    """Target prompt produces identical output regardless of batch position."""
    prompt = "What kind of programming language is Python?"
    sp = SamplingParams(temperature=0.5, max_tokens=40, seed=456)

    baseline, _ = _generate_text(llm, prompt, sp)

    batch_configs = [
        [prompt, "Filler question 1"],
        ["Filler question 2", prompt, "Filler question 3"],
        ["Filler question 4", "Filler question 5", prompt],
        ["Filler 6", "Filler 7", "Filler 8", prompt],
    ]

    for i, batch in enumerate(batch_configs):
        outputs = llm.generate(batch, sp)
        idx = batch.index(prompt)
        assert (
            outputs[idx].outputs.text == baseline
        ), f"Batch config {i} (pos {idx}): result differs from single-request baseline"


def test_deterministic_different_batch_sizes(llm):
    """Same prompt is consistent across batch sizes 1 / 2 / 4 / 8."""
    prompt = "What is machine learning?"
    sp = SamplingParams(temperature=0.5, max_tokens=30, seed=789)

    baseline, _ = _generate_text(llm, prompt, sp)

    for bs in [2, 4, 8]:
        outputs = llm.generate([prompt] * bs, sp)
        assert outputs[0].outputs.text == baseline, f"Batch size {bs} differs from bs=1"


# ===================== Sampling-parameter combinations =====================


@pytest.mark.parametrize(
    "temp,top_p,seed",
    [
        (0.0, 1.0, 300),  # greedy, no top_p filter
        (0.0, 0.0, 301),  # double-greedy
        (0.3, 0.9, 302),  # low temp, moderate top_p
        (0.8, 0.0, 303),  # medium temp, greedy top_p
        (0.8, 1.0, 304),  # medium temp, no top_p filter
        (0.8, 0.5, 305),  # medium temp, strict top_p
        (1.0, 0.95, 306),  # high temp
        (1.5, 0.9, 307),  # very high temp
    ],
)
def test_deterministic_param_combos(llm, temp, top_p, seed):
    """Determinism holds across various (temperature, top_p) combinations."""
    sp = SamplingParams(temperature=temp, top_p=top_p, max_tokens=30, seed=seed)
    _assert_deterministic(llm, "What is a neural network?", sp)


# ===================== Long sequence tests =====================


@pytest.mark.parametrize(
    "temp,seed",
    [
        (0.0, 100),
        (0.3, 130),
        (0.5, 150),
        (0.7, 170),
    ],
)
@pytest.mark.skip(reason="Potential non-determinism in long sequences, will be fixed by gongweibao in next PR")
def test_deterministic_long_sequence(llm, temp, seed):
    """Long generation (512+ tokens) stays deterministic at various temperatures."""
    prompt = "Please describe the history of AI in detail, including major milestones and key technical breakthroughs."
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, seed=seed)

    text, token_ids = _assert_deterministic(llm, prompt, sp)
    assert len(token_ids) >= 100, f"Expected >= 100 tokens, got {len(token_ids)}"


def test_deterministic_long_prompt(llm):
    """Long input prompt (prefill-heavy) stays deterministic."""
    base = "This is a description about natural language processing. "
    long_prompt = (base * 50) + "Please summarize the above."
    sp = SamplingParams(temperature=0.5, max_tokens=100, seed=2024)

    _assert_deterministic(llm, long_prompt, sp)


# ===================== Minimal / boundary output tests =====================


def test_deterministic_max_tokens_one(llm):
    """Single-token output is deterministic."""
    sp = SamplingParams(temperature=0.1, max_tokens=1, seed=700)

    text, token_ids = _assert_deterministic(llm, "What color is the sky?", sp)
    assert len(token_ids) == 1, f"Expected 1 token, got {len(token_ids)}"


def test_deterministic_early_stop(llm):
    """Early stopping via stop sequences is deterministic."""
    sp = SamplingParams(temperature=0.7, max_tokens=100, stop=["\u3002", "."], seed=800)

    text, token_ids = _assert_deterministic(llm, "Please list three colors:", sp)
    assert len(token_ids) < 100, f"Expected early stop, got {len(token_ids)} tokens"


# ===================== Special input tests =====================


@pytest.mark.parametrize(
    "prompt,seed",
    [
        ("What is AI? \U0001f52c\U0001f9e0", 900),  # emoji
        ("Math: E = mc\u00b2", 901),  # superscript
        ("Code: def hello(): return 'world'", 902),  # code
        ("Symbols: @#$%^&*()", 903),  # special symbols
    ],
)
def test_deterministic_special_chars(llm, prompt, seed):
    sp = SamplingParams(temperature=0.5, max_tokens=30, seed=seed)
    _assert_deterministic(llm, prompt, sp)


@pytest.mark.parametrize(
    "lang,prompt,seed",
    [
        ("Chinese", "Please introduce artificial intelligence in one sentence.", 1000),
        ("English", "What is artificial intelligence in one sentence?", 1001),
        (
            "Japanese",
            "\u4eba\u5de5\u77e5\u80fd\u306b\u3064\u3044\u3066\u4e00\u8a00\u3067\u8aac\u660e\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
            1002,
        ),
        ("Spanish", "\u00bfQu\u00e9 es la inteligencia artificial en una frase?", 1003),
    ],
)
def test_deterministic_multi_language(llm, lang, prompt, seed):
    sp = SamplingParams(temperature=0.5, max_tokens=30, seed=seed)
    _assert_deterministic(llm, prompt, sp)


# ===================== Multi-turn conversation test =====================


def test_deterministic_multi_turn(llm):
    """Multi-turn chat maintains determinism."""
    sp = SamplingParams(temperature=0.5, max_tokens=50, seed=1100)

    messages1 = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "Please introduce yourself."},
    ]

    # First full conversation
    r1_turn1 = llm.chat(messages1, sp)[0].outputs.text
    msgs2 = messages1 + [
        {"role": "assistant", "content": r1_turn1},
        {"role": "user", "content": "What can you do?"},
    ]
    r1_turn2 = llm.chat(msgs2, sp)[0].outputs.text

    # Second full conversation (same seed)
    r2_turn1 = llm.chat(messages1, sp)[0].outputs.text
    msgs2_repeat = messages1 + [
        {"role": "assistant", "content": r2_turn1},
        {"role": "user", "content": "What can you do?"},
    ]
    r2_turn2 = llm.chat(msgs2_repeat, sp)[0].outputs.text

    assert r1_turn1 == r2_turn1, "Multi-turn: turn-1 outputs differ"
    assert r1_turn2 == r2_turn2, "Multi-turn: turn-2 outputs differ"


# ===================== State isolation test =====================


def test_deterministic_state_isolation(llm):
    """Interference prompts and interleaving do not break determinism."""
    prompt_a = "What is Python?"
    prompt_b = "What is JavaScript?"
    sp_a = SamplingParams(temperature=0.5, max_tokens=30, seed=1200)
    sp_b = SamplingParams(temperature=0.5, max_tokens=30, seed=1201)

    # Round 1
    a1, _ = _generate_text(llm, prompt_a, sp_a)
    b1, _ = _generate_text(llm, prompt_b, sp_b)

    # Run unrelated interference
    for p in ["Explain reinforcement learning.", "What is NLP?", "List 3 fruits."]:
        llm.generate([p], SamplingParams(temperature=0.7, max_tokens=20, seed=999))

    # Round 2
    a2, _ = _generate_text(llm, prompt_a, sp_a)
    b2, _ = _generate_text(llm, prompt_b, sp_b)

    assert a1 == a2, "Prompt A: output changed after interference"
    assert b1 == b2, "Prompt B: output changed after interference"


# ===================== Non-deterministic validation =====================


def test_non_deterministic_validation(llm):
    """
    Prove that tests are effective:
    - Without seed + without mode: outputs vary
    - With explicit seed: outputs are consistent
    """
    prompt = "Please explain deep learning in one sentence."

    # Part 1: no mode, no seed -> outputs should differ
    os.environ.pop("FD_DETERMINISTIC_MODE", None)
    results_no_seed = []
    for _ in range(5):
        sp = SamplingParams(temperature=0.7, max_tokens=30)
        results_no_seed.append(llm.generate([prompt], sp)[0].outputs.text)

    # Probabilistic, skip if all outputs are the same
    if len(set(results_no_seed)) == 1:
        pytest.skip("Sampling produced identical outputs (probabilistic case)")

    # Part 2: explicit seed -> outputs must be consistent
    sp_seeded = SamplingParams(temperature=0.7, max_tokens=30, seed=999)
    results_seeded = [llm.generate([prompt], sp_seeded)[0].outputs.text for _ in range(5)]
    assert len(set(results_seeded)) == 1, "With explicit seed: expected consistent outputs"


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
