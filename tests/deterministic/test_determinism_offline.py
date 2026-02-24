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
1. Test determinism mode with same prompt (FD_DETERMINISTIC_MODE=1)
2. Test batch invariance (single request vs batch request)
3. Test different batch sizes consistency
4. Test non-deterministic mode variation (FD_DETERMINISTIC_MODE=0)
5. Test explicit seed determinism
6. Test long sequence generation (1024+ tokens) - validates accumulated errors
7. Test long sequence with different temperatures
8. Test long input prompt handling
9. Test extreme sampling parameters (boundary cases)
10. Test empty/minimal outputs
11. Test special characters and multi-language prompts
12. Test multi-turn conversations
13. Validate non-deterministic behavior - proves tests are effective

IMPORTANT: All tests use explicit seeds for better reproducibility.
In non-deterministic mode with explicit seed, results should still be consistent.
Without explicit seed, non-deterministic mode produces different results.

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_determinism_offline.py -v
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FD_DETERMINISTIC_MODE"] = "1"  # Must be set before importing fastdeploy to enable batch_invariant_ops

import pytest

pytestmark = pytest.mark.gpu

from fastdeploy import LLM, SamplingParams

# Small model path for fast testing
DEFAULT_MODEL_DIR = "./models"
MODEL_NAME = "Qwen2-7B-Instruct"


@pytest.fixture(scope="function", autouse=True)
def cleanup_env():
    """Clean up environment variables before and after each test"""
    yield
    # Clean up FD_DETERMINISTIC_MODE after each test
    os.environ.pop("FD_DETERMINISTIC_MODE", None)


@pytest.fixture(scope="module")
def model_path():
    """Get model path from MODEL_PATH env variable or use default ./models/"""
    model_dir = os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR)
    return os.path.join(model_dir, MODEL_NAME)


@pytest.fixture(scope="module")
def llm(model_path):
    """Initialize LLM model for offline inference"""
    return LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=8192,
        enable_prefix_caching=False,
    )


def test_deterministic_mode_same_prompt(llm):
    """
    Test: Deterministic mode produces consistent output for same prompt

    Sets FD_DETERMINISTIC_MODE=1 and verifies that running the same
    prompt multiple times produces identical results.

    Uses explicit seed for reproducibility.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "请用一句话介绍人工智能。"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50, seed=123)

    results = []
    for _ in range(5):
        outputs = llm.generate([prompt], sampling_params)
        results.append(outputs[0].outputs.text)

    assert all(r == results[0] for r in results), "Deterministic mode: same input should produce consistent output"

    print(f"[PASS] Deterministic mode test passed, output: {results[0][:50]}...")


def test_deterministic_mode_batch_invariance(llm):
    """
    Test: Batch invariance in deterministic mode

    Verifies that a single request produces the same result as when it
    is part of a batch with different requests at different positions.

    Uses explicit seed for reproducibility.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "Python 是一种什么编程语言？"
    sampling_params = SamplingParams(temperature=0.5, max_tokens=40, seed=456)

    # Single request
    output_single = llm.generate([prompt], sampling_params)[0].outputs.text

    # Batch requests with target prompt at different positions
    batch_configs = [
        [prompt, "干扰问题1"],
        ["干扰问题2", prompt, "干扰问题3"],
        ["干扰问题4", "干扰问题5", prompt],
        ["干扰问题6", "干扰问题7", "干扰问题8", prompt],
    ]

    for i, batch in enumerate(batch_configs):
        outputs = llm.generate(batch, sampling_params)
        target_idx = batch.index(prompt)
        output_batch = outputs[target_idx].outputs.text

        assert output_batch == output_single, f"Batch config {i}: batch request result differs from single request"

        print(f"[PASS] Batch config {i} (position {target_idx}) passed")


def test_deterministic_mode_different_batch_sizes(llm):
    """
    Test: Consistency across different batch sizes in deterministic mode

    Verifies that the same prompt produces consistent results
    regardless of batch size.

    Uses explicit seed for reproducibility.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "什么是机器学习？"
    sampling_params = SamplingParams(temperature=0.5, max_tokens=30, seed=789)

    # Batch size 1
    output_bs1 = llm.generate([prompt], sampling_params)[0].outputs.text

    # Different batch sizes with same prompt repeated
    for bs in [2, 4, 8]:
        batch = [prompt] * bs
        outputs = llm.generate(batch, sampling_params)
        output_bs = outputs[0].outputs.text

        assert output_bs == output_bs1, f"Batch size {bs} result differs from batch size 1"

        print(f"[PASS] Batch size {bs} test passed")


def test_deterministic_with_explicit_seed(llm):
    """
    Test: Explicit seed produces deterministic results

    Verifies that using the same explicit seed produces consistent
    results regardless of FD_DETERMINISTIC_MODE.

    This test should PASS with or without FD_DETERMINISTIC_MODE
    as long as the same explicit seed is used.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "请列举三种水果。"
    sampling_params = SamplingParams(temperature=0.7, seed=42, max_tokens=30)

    results = []
    for _ in range(3):
        outputs = llm.generate([prompt], sampling_params)
        results.append(outputs[0].outputs.text)

    assert all(r == results[0] for r in results), "Explicit seed should produce consistent results"

    # Different seed should likely produce different result
    sampling_params2 = SamplingParams(temperature=0.7, seed=123, max_tokens=30)
    output2 = llm.generate([prompt], sampling_params2)[0].outputs.text

    print(f"[PASS] Seed 42: {results[0][:30]}...")
    print(f"[PASS] Seed 123: {output2[:30]}...")


def test_non_deterministic_mode_variation_validation(llm):
    """
    Test: Validate non-deterministic behavior (mode validation)

    This test validates that:
    1. WITHOUT FD_DETERMINISTIC_MODE and WITHOUT seed: results are different
    2. WITH explicit seed: results are consistent (regardless of mode)

    This proves the determinism tests are effective.
    """
    prompt = "请用一句话解释什么是深度学习。"

    # Test 1: Non-deterministic mode without seed - should produce different results
    print("\n[TEST 1] Non-deterministic mode WITHOUT explicit seed")
    os.environ.pop("FD_DETERMINISTIC_MODE", None)

    results_no_seed = []
    for i in range(5):
        # Create new SamplingParams each time to get different random seeds
        sampling_params_no_seed = SamplingParams(temperature=0.7, max_tokens=30)
        print(f"  Run {i+1} seed: {sampling_params_no_seed.seed}")
        outputs = llm.generate([prompt], sampling_params_no_seed)
        results_no_seed.append(outputs[0].outputs.text)
        print(f"  Run {i+1}: {outputs[0].outputs.text[:30]}...")

    unique_no_seed = len(set(results_no_seed))
    assert (
        unique_no_seed > 1
    ), f"Non-deterministic without seed: expected different results, got {unique_no_seed} unique"

    print(f"  Result: {unique_no_seed} unique results (PASS)")

    # Test 2: With explicit seed - should produce consistent results
    print("\n[TEST 2] WITH explicit seed (mode independent)")
    sampling_params_with_seed = SamplingParams(temperature=0.7, max_tokens=30, seed=999)

    results_with_seed = []
    for i in range(5):
        outputs = llm.generate([prompt], sampling_params_with_seed)
        results_with_seed.append(outputs[0].outputs.text)
        print(f"  Run {i+1}: {outputs[0].outputs.text[:30]}...")

    unique_with_seed = len(set(results_with_seed))
    assert unique_with_seed == 1, f"With explicit seed: expected consistent results, got {unique_with_seed} unique"

    print(f"  Result: {unique_with_seed} unique result (PASS)")

    # Test 3: Deterministic mode with default seed - should produce consistent results
    print("\n[TEST 3] Deterministic mode with DEFAULT seed (42)")
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    sampling_params_det_default = SamplingParams(temperature=0.7, max_tokens=30)

    results_det_default = []
    for i in range(3):
        outputs = llm.generate([prompt], sampling_params_det_default)
        results_det_default.append(outputs[0].outputs.text)
        print(f"  Run {i+1}: {outputs[0].outputs.text[:30]}...")

    unique_det_default = len(set(results_det_default))
    assert (
        unique_det_default == 1
    ), f"Deterministic mode with default seed: expected consistent results, got {unique_det_default} unique"

    print(f"  Result: {unique_det_default} unique result (PASS)")

    print("\n[PASS] Non-deterministic behavior validation test passed")
    print("  Confirmed: No mode/seed → different results")
    print("  Confirmed: With explicit seed → consistent results")
    print("  Confirmed: Deterministic mode → consistent results")


def test_deterministic_long_sequence_generation(llm):
    """
    Test: Deterministic mode produces consistent output for long sequence generation

    This test validates determinism when generating long sequences (1024+ tokens),
    which is critical for detecting:
    - Accumulated errors over many decode steps
    - KV Cache state consistency during long generation
    - Randomness issues that may appear in long-running processes

    Uses explicit seed for reproducibility.

    Note: This test takes longer due to the long sequence generation.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    # Use a prompt that encourages longer output
    prompt = "请详细介绍一下人工智能的发展历史，包括主要里程碑和关键技术突破。"
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.95,
        max_tokens=1024,
        seed=42,
    )

    # Run the same prompt multiple times
    results = []
    token_ids_list = []

    for i in range(2):
        outputs = llm.generate([prompt], sampling_params)
        result_text = outputs[0].outputs.text
        result_tokens = outputs[0].outputs.token_ids

        results.append(result_text)
        token_ids_list.append(result_tokens)

        generated_len = len(result_tokens)
        print(f"[RUN {i+1}] Generated {generated_len} tokens, " f"text length: {len(result_text)} chars")

    # Verify all text outputs are identical
    assert all(
        r == results[0] for r in results
    ), "Long sequence: text outputs should be identical in deterministic mode"

    # Verify all token ID sequences are identical (stronger check)
    assert all(
        tokens == token_ids_list[0] for tokens in token_ids_list
    ), "Long sequence: token ID sequences should be identical in deterministic mode"

    # Verify we actually generated a substantial sequence
    generated_tokens = len(token_ids_list[0])
    assert generated_tokens >= 100, f"Expected at least 100 tokens, got {generated_tokens}"

    print("[PASS] Long sequence determinism test passed")
    print(f"  Generated {generated_tokens} tokens across 2 runs")
    print("  All outputs identical (text and token IDs)")


def test_deterministic_long_sequence_different_temperatures(llm):
    """
    Test: Long sequence determinism with different temperature values

    Validates that determinism holds for long sequences at different
    temperature settings (greedy, low, medium).

    Uses explicit seed for reproducibility.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "请解释机器学习的基本概念和应用场景."

    # Test different temperatures with long sequence generation
    temperatures = [0.0, 0.3, 0.7]

    for temp in temperatures:
        sampling_params = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, seed=100 + int(temp * 100))

        # Run twice
        outputs1 = llm.generate([prompt], sampling_params)[0]
        outputs2 = llm.generate([prompt], sampling_params)[0]

        text1 = outputs1.outputs.text
        text2 = outputs2.outputs.text

        assert text1 == text2, f"Long sequence at temp={temp}: outputs should be identical"

        tokens1 = outputs1.outputs.token_ids
        tokens2 = outputs2.outputs.token_ids

        assert tokens1 == tokens2, f"Long sequence at temp={temp}: token IDs should be identical"

        print(f"[PASS] Long sequence determinism at temperature={temp}, " f"generated {len(tokens1)} tokens")


def test_deterministic_long_prompt(llm):
    """
    Test: Deterministic mode with long input prompt

    Validates that long input prompts (pre-fill stage) are handled
    consistently in deterministic mode.

    Uses explicit seed for reproducibility.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    # Create a long prompt by repeating a pattern
    base_text = "这是一段关于自然语言处理的说明。"
    long_prompt = (base_text * 50) + "请总结以上内容。"

    sampling_params = SamplingParams(temperature=0.5, max_tokens=100, seed=2024)

    # Run twice
    outputs1 = llm.generate([long_prompt], sampling_params)[0]
    outputs2 = llm.generate([long_prompt], sampling_params)[0]

    assert (
        outputs1.outputs.text == outputs2.outputs.text
    ), "Long prompt: outputs should be identical in deterministic mode"

    print("[PASS] Long prompt determinism test passed")
    print(f"  Prompt length: {len(long_prompt)} chars")
    print(f"  Output length: {len(outputs1.outputs.text)} chars")


# ==================== Extreme Sampling Parameters Tests ====================


def test_deterministic_extreme_top_p(llm):
    """
    Test: Deterministic mode with extreme top_p values

    Tests boundary cases: top_p=0.0 (greedy) and top_p=1.0 (no filtering)
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "什么是神经网络？"

    for top_p in [0.0, 1.0]:
        sampling_params = SamplingParams(temperature=0.8, top_p=top_p, max_tokens=30, seed=300 + int(top_p * 100))

        outputs1 = llm.generate([prompt], sampling_params)[0]
        outputs2 = llm.generate([prompt], sampling_params)[0]

        assert outputs1.outputs.text == outputs2.outputs.text, f"Extreme top_p={top_p}: outputs should be identical"

        print(f"[PASS] Extreme top_p={top_p} test passed")


def test_deterministic_extreme_temperature(llm):
    """
    Test: Deterministic mode with extreme temperature values

    Tests boundary cases: temperature=0.0 (greedy) and high temperature
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "简述机器学习的三个主要分支。"

    for temp in [0.0, 1.5]:
        sampling_params = SamplingParams(temperature=temp, top_p=0.9, max_tokens=30, seed=400 + int(temp * 100))

        outputs1 = llm.generate([prompt], sampling_params)[0]
        outputs2 = llm.generate([prompt], sampling_params)[0]

        assert (
            outputs1.outputs.text == outputs2.outputs.text
        ), f"Extreme temperature={temp}: outputs should be identical"

        print(f"[PASS] Extreme temperature={temp} test passed")


def test_deterministic_parameter_combinations(llm):
    """
    Test: Deterministic mode with various parameter combinations

    Tests interactions between temperature and top_p at different combinations
    to ensure determinism holds across code paths.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "什么是强化学习？"

    # Test various parameter combinations
    combinations = [
        (0.0, 1.0),  # Greedy, no top_p filter
        (0.3, 0.9),  # Low temp, moderate top_p
        (0.8, 0.5),  # Medium temp, strict top_p
        (1.0, 0.95),  # High temp, permissive top_p
    ]

    for i, (temp, top_p) in enumerate(combinations):
        sampling_params = SamplingParams(temperature=temp, top_p=top_p, max_tokens=30, seed=500 + i)

        outputs1 = llm.generate([prompt], sampling_params)[0]
        outputs2 = llm.generate([prompt], sampling_params)[0]

        assert (
            outputs1.outputs.text == outputs2.outputs.text
        ), f"Parameter combination {i} (temp={temp}, top_p={top_p}): outputs should be identical"

        print(f"[PASS] Parameter combination test {i} passed (temp={temp}, top_p={top_p})")


# ==================== Minimal Output Tests ====================


def test_deterministic_max_tokens_one(llm):
    """
    Test: Deterministic mode with max_tokens=1 (single token output)

    Tests boundary case with minimal generation.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "天空是什么颜色的？"
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1, seed=700)

    outputs1 = llm.generate([prompt], sampling_params)[0]
    outputs2 = llm.generate([prompt], sampling_params)[0]

    assert outputs1.outputs.text == outputs2.outputs.text, "max_tokens=1: outputs should be identical"

    tokens1 = outputs1.outputs.token_ids
    tokens2 = outputs2.outputs.token_ids

    assert tokens1 == tokens2, "max_tokens=1: token IDs should be identical"
    assert len(tokens1) == 1, f"max_tokens=1: expected 1 token, got {len(tokens1)}"

    print("[PASS] max_tokens=1 test passed")


def test_deterministic_early_stopping_with_stop(llm):
    """
    Test: Deterministic mode with early stopping via stop sequences

    Tests that early stopping with stop sequences is deterministic.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt = "请列举三种颜色："
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
        stop=["。", "."],
        seed=800,
    )

    outputs1 = llm.generate([prompt], sampling_params)[0]
    outputs2 = llm.generate([prompt], sampling_params)[0]

    assert outputs1.outputs.text == outputs2.outputs.text, "Early stopping: outputs should be identical"

    # Verify output was truncated by stop sequence (not max_tokens)
    tokens = outputs1.outputs.token_ids
    assert len(tokens) < 100, f"Expected early stop before max_tokens, got {len(tokens)} tokens"

    print("[PASS] Early stopping with stop sequences test passed")


# ==================== Special Characters and Multi-Language Tests ====================


def test_deterministic_special_characters(llm):
    """
    Test: Deterministic mode with special characters in prompt

    Tests handling of unicode, emojis, and special symbols.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompts_with_special_chars = [
        "What is AI? 🔬🧠",  # Emoji
        "数学公式：E = mc²",  # Superscript
        "Code: def hello(): return 'world'",  # Code block
        "Symbols: @#$%^&*()",  # Special symbols
    ]

    for i, prompt in enumerate(prompts_with_special_chars):
        sampling_params = SamplingParams(temperature=0.5, max_tokens=30, seed=900 + i)

        outputs1 = llm.generate([prompt], sampling_params)[0]
        outputs2 = llm.generate([prompt], sampling_params)[0]

        assert outputs1.outputs.text == outputs2.outputs.text, f"Special chars test {i}: outputs should be identical"

        print(f"[PASS] Special characters test {i} passed")


def test_deterministic_multi_language(llm):
    """
    Test: Deterministic mode with multi-language prompts

    Tests that determinism holds across different languages.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    multi_lang_prompts = [
        ("中文", "请用一句话介绍人工智能。"),
        ("English", "What is artificial intelligence in one sentence?"),
        ("日本語", "人工知能について一言で説明してください。"),
        ("Español", "¿Qué es la inteligencia artificial en una frase?"),
    ]

    for i, (lang_name, prompt) in enumerate(multi_lang_prompts):
        sampling_params = SamplingParams(temperature=0.5, max_tokens=30, seed=1000 + i)

        outputs1 = llm.generate([prompt], sampling_params)[0]
        outputs2 = llm.generate([prompt], sampling_params)[0]

        assert (
            outputs1.outputs.text == outputs2.outputs.text
        ), f"Multi-language test {i} ({lang_name}): outputs should be identical"

        print(f"[PASS] Multi-language test {i} ({lang_name}) passed")


# ==================== Multi-Turn Conversation Tests ====================


def test_deterministic_multi_turn_conversation(llm):
    """
    Test: Deterministic mode with multi-turn conversations

    Validates that multi-turn conversations maintain determinism,
    which is critical for chat applications.
    Uses llm.chat() which supports message-list input with chat template.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    sampling_params = SamplingParams(temperature=0.5, max_tokens=50, seed=1100)

    # First turn
    prompt1 = "请介绍一下你自己。"
    messages1 = [
        {"role": "user", "content": "你好！"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
        {"role": "user", "content": prompt1},
    ]
    outputs1 = llm.chat(messages1, sampling_params)[0]
    response1 = outputs1.outputs.text

    # Second turn (append first turn result)
    prompt2 = "你能做什么？"
    messages2 = messages1 + [
        {"role": "assistant", "content": response1},
        {"role": "user", "content": prompt2},
    ]
    outputs2 = llm.chat(messages2, sampling_params)[0]
    response2 = outputs2.outputs.text

    # Repeat the same conversation with same seed
    outputs1_repeat = llm.chat(messages1, sampling_params)[0]
    response1_repeat = outputs1_repeat.outputs.text

    messages2_repeat = messages1 + [
        {"role": "assistant", "content": response1_repeat},
        {"role": "user", "content": prompt2},
    ]
    outputs2_repeat = llm.chat(messages2_repeat, sampling_params)[0]
    response2_repeat = outputs2_repeat.outputs.text

    assert response1 == response1_repeat, "Multi-turn turn 1: outputs should be identical"
    assert response2 == response2_repeat, "Multi-turn turn 2: outputs should be identical"

    print("[PASS] Multi-turn conversation determinism test passed")
    print(f"  Turn 1: {response1[:50]}...")
    print(f"  Turn 2: {response2[:50]}...")


# ==================== State Isolation Tests ====================


def test_deterministic_state_isolation_between_prompts(llm):
    """
    Test: Model state isolation between different prompts

    Validates that running different prompts doesn't affect the
    determinism of subsequent runs of the same prompt.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    target_prompt = "什么是深度学习？"
    sampling_params = SamplingParams(temperature=0.5, max_tokens=30, seed=1200)

    # Run target prompt first
    output_first = llm.generate([target_prompt], sampling_params)[0].outputs.text

    # Run interference prompts
    interference_prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What are the benefits of AI?",
    ]

    for prompt in interference_prompts:
        llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=20, seed=999))

    # Run target prompt again
    output_second = llm.generate([target_prompt], sampling_params)[0].outputs.text

    assert output_first == output_second, "State isolation: outputs should be identical after interference"

    print("[PASS] State isolation test passed")
    print(f"  Output: {output_first[:50]}...")


def test_deterministic_interleaved_prompts(llm):
    """
    Test: Determinism with interleaved different prompts

    Validates that interleaved runs of different prompts don't affect
    each other's determinism.
    """
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    prompt_a = "什么是Python？"
    prompt_b = "什么是JavaScript？"

    sampling_params_a = SamplingParams(temperature=0.5, max_tokens=30, seed=1300)
    sampling_params_b = SamplingParams(temperature=0.5, max_tokens=30, seed=1301)

    # Run A, then B, then A again
    output_a1 = llm.generate([prompt_a], sampling_params_a)[0].outputs.text
    output_b1 = llm.generate([prompt_b], sampling_params_b)[0].outputs.text
    output_a2 = llm.generate([prompt_a], sampling_params_a)[0].outputs.text
    output_b2 = llm.generate([prompt_b], sampling_params_b)[0].outputs.text

    assert output_a1 == output_a2, "Interleaved A: outputs should be identical"
    assert output_b1 == output_b2, "Interleaved B: outputs should be identical"

    print("[PASS] Interleaved prompts determinism test passed")


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
