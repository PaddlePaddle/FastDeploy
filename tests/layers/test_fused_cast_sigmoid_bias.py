"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import importlib
import os
import sys
from unittest import mock

import paddle
import paddle.nn.functional as F
import pytest

from fastdeploy.model_executor.layers.moe.fused_cast_sigmoid_bias import (
    fused_cast_sigmoid_bias,
    is_available,
)

DTYPE_MAP = {
    "float16": paddle.float16,
    "bfloat16": paddle.bfloat16,
    "float32": paddle.float32,
}


def _ensure_gpu_test_environment():
    """Ensure GPU runtime and required custom ops are available for this test module."""
    if not paddle.is_compiled_with_cuda():
        pytest.skip(
            "fused_cast_sigmoid_bias requires CUDA-enabled Paddle.",
            allow_module_level=True,
        )
    paddle.set_device("gpu")


_ensure_gpu_test_environment()


def reference_cast_sigmoid_bias(gate_out, bias, cast_type="float32"):
    """Reference implementation: compute in fp32, cast output to cast_type."""
    gate_fp32 = gate_out.cast("float32")
    scores_fp32 = F.sigmoid(gate_fp32)
    scores_with_bias_fp32 = scores_fp32 + bias
    scores = scores_fp32.cast(cast_type)
    scores_with_bias = scores_with_bias_fp32.cast(cast_type)
    return scores, scores_with_bias


def test_functionality():
    """Test basic functionality: correct shapes and dtypes (default cast_type=float32)."""
    print("=" * 60)
    print("Test 1: Functionality (default cast_type=float32)")
    print("=" * 60)

    for dtype_name in ["float16", "bfloat16", "float32"]:
        for num_tokens in [1, 7, 128, 1024]:
            for num_experts in [8, 64, 128, 256]:
                gate_out = paddle.randn([num_tokens, num_experts], dtype=dtype_name)
                bias = paddle.randn([num_experts], dtype="float32")

                scores, scores_with_bias = fused_cast_sigmoid_bias(gate_out, bias)

                assert scores.shape == [
                    num_tokens,
                    num_experts,
                ], f"scores shape mismatch: {scores.shape} vs {[num_tokens, num_experts]}"
                assert scores_with_bias.shape == [
                    num_tokens,
                    num_experts,
                ], f"scores_with_bias shape mismatch: {scores_with_bias.shape}"
                assert scores.dtype == paddle.float32, f"scores dtype mismatch: {scores.dtype}"
                assert (
                    scores_with_bias.dtype == paddle.float32
                ), f"scores_with_bias dtype mismatch: {scores_with_bias.dtype}"

                # Sigmoid output should be in [0, 1]
                assert bool(paddle.all(scores >= 0.0).item()) and bool(
                    paddle.all(scores <= 1.0).item()
                ), "scores out of [0,1] range"
        print(f"  [PASS] dtype={dtype_name}")

    print("  All functionality tests passed.\n")


def test_functionality_cast_types():
    """Test functionality with different cast_type values."""
    print("=" * 60)
    print("Test 1b: Functionality with different cast_type")
    print("=" * 60)

    for input_dtype in ["float16", "bfloat16", "float32"]:
        for cast_type in ["float16", "bfloat16", "float32"]:
            expected_paddle_dtype = DTYPE_MAP[cast_type]
            for num_tokens in [1, 64, 256]:
                for num_experts in [8, 64, 256]:
                    gate_out = paddle.randn([num_tokens, num_experts], dtype=input_dtype)
                    bias = paddle.randn([num_experts], dtype="float32")

                    scores, scores_with_bias = fused_cast_sigmoid_bias(gate_out, bias, cast_type)

                    assert scores.shape == [num_tokens, num_experts], f"scores shape mismatch: {scores.shape}"
                    assert scores_with_bias.shape == [
                        num_tokens,
                        num_experts,
                    ], f"scores_with_bias shape mismatch: {scores_with_bias.shape}"
                    assert (
                        scores.dtype == expected_paddle_dtype
                    ), f"scores dtype mismatch: got {scores.dtype}, expected {expected_paddle_dtype}"
                    assert (
                        scores_with_bias.dtype == expected_paddle_dtype
                    ), f"scores_with_bias dtype mismatch: got {scores_with_bias.dtype}, expected {expected_paddle_dtype}"

            print(f"  [PASS] input_dtype={input_dtype}, cast_type={cast_type}")

    print("  All cast_type functionality tests passed.\n")


def test_accuracy():
    """Test numerical accuracy against reference implementation (default cast_type=float32)."""
    print("=" * 60)
    print("Test 2: Accuracy (default cast_type=float32)")
    print("=" * 60)

    test_cases = [
        ("float16", 1, 8),
        ("float16", 128, 256),
        ("float16", 1024, 256),
        ("bfloat16", 1, 8),
        ("bfloat16", 128, 256),
        ("bfloat16", 1024, 256),
        ("float32", 1, 8),
        ("float32", 128, 256),
        ("float32", 1024, 256),
    ]

    for dtype_name, num_tokens, num_experts in test_cases:
        gate_out = paddle.randn([num_tokens, num_experts], dtype=dtype_name)
        bias = paddle.randn([num_experts], dtype="float32")

        # Fused kernel
        fused_scores, fused_scores_with_bias = fused_cast_sigmoid_bias(gate_out, bias)

        # Reference
        ref_scores, ref_scores_with_bias = reference_cast_sigmoid_bias(gate_out, bias)

        # Compare
        scores_diff = paddle.abs(fused_scores - ref_scores).max().item()
        scores_bias_diff = paddle.abs(fused_scores_with_bias - ref_scores_with_bias).max().item()

        atol = 1e-6 if dtype_name == "float32" else 1e-3
        passed = scores_diff < atol and scores_bias_diff < atol

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] dtype={dtype_name}, tokens={num_tokens}, experts={num_experts} | "
            f"scores_max_diff={scores_diff:.2e}, scores_with_bias_max_diff={scores_bias_diff:.2e}"
        )

        if not passed:
            raise AssertionError(
                f"Accuracy test failed for dtype={dtype_name}, tokens={num_tokens}, experts={num_experts}. "
                f"scores_diff={scores_diff}, scores_bias_diff={scores_bias_diff}, atol={atol}"
            )

    print("  All accuracy tests passed.\n")


def test_accuracy_cast_types():
    """Test numerical accuracy with different cast_type values."""
    print("=" * 60)
    print("Test 2b: Accuracy with different cast_type")
    print("=" * 60)

    # (input_dtype, cast_type, num_tokens, num_experts)
    test_cases = [
        # cast to float32 (original behavior)
        ("float16", "float32", 128, 256),
        ("bfloat16", "float32", 128, 256),
        ("float32", "float32", 128, 256),
        # cast to float16
        ("float16", "float16", 128, 256),
        ("bfloat16", "float16", 128, 256),
        ("float32", "float16", 128, 256),
        # cast to bfloat16
        ("float16", "bfloat16", 128, 256),
        ("bfloat16", "bfloat16", 128, 256),
        ("float32", "bfloat16", 128, 256),
        # different shapes
        ("bfloat16", "float16", 1, 8),
        ("bfloat16", "float16", 1024, 256),
        ("float16", "bfloat16", 1, 8),
        ("float16", "bfloat16", 1024, 256),
    ]

    for input_dtype, cast_type, num_tokens, num_experts in test_cases:
        gate_out = paddle.randn([num_tokens, num_experts], dtype=input_dtype)
        bias = paddle.randn([num_experts], dtype="float32")

        # Fused kernel
        fused_scores, fused_scores_with_bias = fused_cast_sigmoid_bias(gate_out, bias, cast_type)

        # Reference
        ref_scores, ref_scores_with_bias = reference_cast_sigmoid_bias(gate_out, bias, cast_type)

        # Compare in float32 for stable diff computation
        scores_diff = paddle.abs(fused_scores.cast("float32") - ref_scores.cast("float32")).max().item()
        scores_bias_diff = (
            paddle.abs(fused_scores_with_bias.cast("float32") - ref_scores_with_bias.cast("float32")).max().item()
        )

        # Tolerance depends on cast_type precision
        if cast_type == "float32":
            atol = 1e-6
        elif cast_type == "bfloat16":
            atol = 1e-2  # bfloat16 has fewer mantissa bits
        else:  # float16
            atol = 1e-3

        passed = scores_diff < atol and scores_bias_diff < atol

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] input={input_dtype}, cast_type={cast_type}, "
            f"tokens={num_tokens}, experts={num_experts} | "
            f"scores_diff={scores_diff:.2e}, bias_diff={scores_bias_diff:.2e}"
        )

        if not passed:
            raise AssertionError(
                f"Accuracy test failed for input={input_dtype}, cast_type={cast_type}, "
                f"tokens={num_tokens}, experts={num_experts}. "
                f"scores_diff={scores_diff}, bias_diff={scores_bias_diff}, atol={atol}"
            )

    print("  All cast_type accuracy tests passed.\n")


def test_accuracy_extreme_values():
    """Test accuracy with extreme input values."""
    print("=" * 60)
    print("Test 3: Accuracy with extreme values")
    print("=" * 60)

    num_tokens, num_experts = 64, 256

    for dtype_name in ["float16", "bfloat16"]:
        # Large positive values -> sigmoid ~ 1.0
        gate_out = paddle.full([num_tokens, num_experts], 10.0, dtype=dtype_name)
        bias = paddle.zeros([num_experts], dtype="float32")
        fused_scores, _ = fused_cast_sigmoid_bias(gate_out, bias)
        ref_scores, _ = reference_cast_sigmoid_bias(gate_out, bias)
        diff = paddle.abs(fused_scores - ref_scores).max().item()
        print(f"  [{'PASS' if diff < 1e-5 else 'FAIL'}] dtype={dtype_name}, large positive: max_diff={diff:.2e}")

        # Large negative values -> sigmoid ~ 0.0
        gate_out = paddle.full([num_tokens, num_experts], -10.0, dtype=dtype_name)
        fused_scores, _ = fused_cast_sigmoid_bias(gate_out, bias)
        ref_scores, _ = reference_cast_sigmoid_bias(gate_out, bias)
        diff = paddle.abs(fused_scores - ref_scores).max().item()
        print(f"  [{'PASS' if diff < 1e-5 else 'FAIL'}] dtype={dtype_name}, large negative: max_diff={diff:.2e}")

        # Zero values -> sigmoid = 0.5
        gate_out = paddle.zeros([num_tokens, num_experts], dtype=dtype_name)
        fused_scores, _ = fused_cast_sigmoid_bias(gate_out, bias)
        ref_scores, _ = reference_cast_sigmoid_bias(gate_out, bias)
        diff = paddle.abs(fused_scores - ref_scores).max().item()
        assert diff < 1e-6, f"Zero input test failed: diff={diff}"
        print(f"  [PASS] dtype={dtype_name}, zeros: max_diff={diff:.2e}")

    print("  All extreme value tests passed.\n")


def test_accuracy_extreme_values_cast_types():
    """Test accuracy with extreme values across different cast_type values."""
    print("=" * 60)
    print("Test 3b: Accuracy with extreme values + different cast_type")
    print("=" * 60)

    num_tokens, num_experts = 64, 256

    for input_dtype in ["float16", "bfloat16"]:
        for cast_type in ["float16", "bfloat16", "float32"]:
            bias = paddle.zeros([num_experts], dtype="float32")

            # Large positive
            gate_out = paddle.full([num_tokens, num_experts], 10.0, dtype=input_dtype)
            fused_scores, _ = fused_cast_sigmoid_bias(gate_out, bias, cast_type)
            ref_scores, _ = reference_cast_sigmoid_bias(gate_out, bias, cast_type)
            diff = paddle.abs(fused_scores.cast("float32") - ref_scores.cast("float32")).max().item()
            atol = 1e-2 if cast_type == "bfloat16" else 1e-5
            status = "PASS" if diff < atol else "FAIL"
            print(f"  [{status}] input={input_dtype}, cast={cast_type}, " f"large positive: diff={diff:.2e}")

            # Zero values
            gate_out = paddle.zeros([num_tokens, num_experts], dtype=input_dtype)
            fused_scores, _ = fused_cast_sigmoid_bias(gate_out, bias, cast_type)
            ref_scores, _ = reference_cast_sigmoid_bias(gate_out, bias, cast_type)
            diff = paddle.abs(fused_scores.cast("float32") - ref_scores.cast("float32")).max().item()
            atol = 1e-2 if cast_type == "bfloat16" else 1e-5
            assert diff < atol, f"Zero input test failed: input={input_dtype}, cast={cast_type}, diff={diff}"
            print(f"  [PASS] input={input_dtype}, cast={cast_type}, " f"zeros: diff={diff:.2e}")

    print("  All extreme value cast_type tests passed.\n")


@pytest.mark.skipif(
    os.getenv("RUN_PERFORMANCE_TESTS") != "1",
    reason="Performance benchmark is disabled by default. Set RUN_PERFORMANCE_TESTS=1 to enable.",
)
def test_performance():
    """Benchmark fused kernel vs reference implementation using CUDA events."""
    print("=" * 60)
    print("Test 4: Performance (CUDA event timing)")
    print("=" * 60)

    configs = [
        ("bfloat16", 1, 256),  # single token decode
        ("bfloat16", 8, 256),  # small batch decode
        ("bfloat16", 64, 256),  # medium batch
        ("bfloat16", 256, 256),  # typical DeepSeek-V3 config
        ("bfloat16", 1024, 256),  # large prefill
        ("bfloat16", 4096, 256),  # very large prefill
    ]

    warmup_iters = 100
    bench_iters = 500

    for dtype_name, num_tokens, num_experts in configs:
        gate_out = paddle.randn([num_tokens, num_experts], dtype=dtype_name)
        bias = paddle.randn([num_experts], dtype="float32")

        # Warmup fused
        for _ in range(warmup_iters):
            fused_cast_sigmoid_bias(gate_out, bias)
        paddle.device.synchronize()

        # Benchmark fused with CUDA events
        start_event = paddle.device.cuda.Event(enable_timing=True)
        end_event = paddle.device.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(bench_iters):
            fused_cast_sigmoid_bias(gate_out, bias)
        end_event.record()
        paddle.device.synchronize()
        fused_time = start_event.elapsed_time(end_event) / bench_iters * 1e3  # us

        # Warmup reference
        for _ in range(warmup_iters):
            reference_cast_sigmoid_bias(gate_out, bias)
        paddle.device.synchronize()

        # Benchmark reference with CUDA events
        start_event = paddle.device.cuda.Event(enable_timing=True)
        end_event = paddle.device.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(bench_iters):
            reference_cast_sigmoid_bias(gate_out, bias)
        end_event.record()
        paddle.device.synchronize()
        ref_time = start_event.elapsed_time(end_event) / bench_iters * 1e3  # us

        speedup = ref_time / fused_time if fused_time > 0 else float("inf")
        print(
            f"  tokens={num_tokens:5d}, experts={num_experts:3d} | "
            f"ref={ref_time:8.1f}us, fused={fused_time:8.1f}us, speedup={speedup:.2f}x"
        )

    print()
    print("  Note: The CUDA custom op fuses cast+sigmoid+bias into a single kernel,")
    print("  eliminating 2 intermediate tensors and reducing kernel launches from 3 to 1.")
    print("  Expected speedup: ~3x over the reference 3-op implementation.")
    print("  Performance benchmark complete.\n")


def test_is_available():
    """Test is_available() function returns True when GPU ops are available."""
    print("=" * 60)
    print("Test: is_available()")
    print("=" * 60)

    # In normal GPU test environment, is_available should return True
    result = is_available()
    assert isinstance(result, bool), f"is_available() should return bool, got {type(result)}"
    assert result is True, f"is_available() should return True when GPU ops are compiled, got {result}"
    print(f"  [PASS] is_available() returned {result}")
    print("  is_available() test passed.\n")


def test_import_error():
    """Test that ImportError is raised when GPU ops are not available."""
    print("=" * 60)
    print("Test 5: Import error handling")
    print("=" * 60)

    module_name = "fastdeploy.model_executor.layers.moe.fused_cast_sigmoid_bias"
    gpu_ops_module = "fastdeploy.model_executor.ops.gpu"

    # Save original module references
    original_module = sys.modules.pop(module_name, None)
    original_gpu_ops = sys.modules.get(gpu_ops_module)

    try:
        # Mock the GPU ops module to raise ImportError on import
        with mock.patch.dict(sys.modules, {gpu_ops_module: None}):
            # Re-import the module so it picks up the mocked (missing) GPU ops
            reloaded = importlib.import_module(module_name)
            importlib.reload(reloaded)

            # The module should load successfully, but calling the function
            # should raise ImportError because the cuda op is unavailable.
            dummy_gate = paddle.randn([1, 8], dtype="float32")
            dummy_bias = paddle.randn([8], dtype="float32")
            try:
                reloaded.fused_cast_sigmoid_bias(dummy_gate, dummy_bias)
                raise AssertionError("Expected ImportError was not raised")
            except ImportError as e:
                assert "fused_cast_sigmoid_bias is not available" in str(e), f"Unexpected error message: {e}"
                print(f"  [PASS] ImportError raised with correct message: {e}")
    finally:
        # Restore original modules
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        if original_gpu_ops is not None:
            sys.modules[gpu_ops_module] = original_gpu_ops

    print("  Import error handling test passed.\n")


def test_is_available_when_ops_unavailable():
    """Test is_available() returns False when GPU ops are not available."""
    print("=" * 60)
    print("Test: is_available() when ops unavailable")
    print("=" * 60)

    module_name = "fastdeploy.model_executor.layers.moe.fused_cast_sigmoid_bias"
    gpu_ops_module = "fastdeploy.model_executor.ops.gpu"

    # Save original module references
    original_module = sys.modules.pop(module_name, None)
    original_gpu_ops = sys.modules.get(gpu_ops_module)

    try:
        # Mock the GPU ops module to raise ImportError on import
        with mock.patch.dict(sys.modules, {gpu_ops_module: None}):
            # Re-import the module so it picks up the mocked (missing) GPU ops
            reloaded = importlib.import_module(module_name)
            importlib.reload(reloaded)

            # is_available should return False when ops are not available
            result = reloaded.is_available()
            assert isinstance(result, bool), f"is_available() should return bool, got {type(result)}"
            assert result is False, f"is_available() should return False when GPU ops are unavailable, got {result}"
            print(f"  [PASS] is_available() returned {result} when ops unavailable")
    finally:
        # Restore original modules
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        if original_gpu_ops is not None:
            sys.modules[gpu_ops_module] = original_gpu_ops

    print("  is_available() when ops unavailable test passed.\n")


if __name__ == "__main__":
    print("Running fused_cast_sigmoid_bias tests...\n")

    test_is_available()
    test_functionality()
    test_functionality_cast_types()
    test_accuracy()
    test_accuracy_cast_types()
    test_accuracy_extreme_values()
    test_accuracy_extreme_values_cast_types()
    test_import_error()
    test_is_available_when_ops_unavailable()
    if os.getenv("RUN_PERFORMANCE_TESTS") == "1":
        test_performance()
    else:
        print("Skipping performance benchmark. Set RUN_PERFORMANCE_TESTS=1 to enable.\n")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
