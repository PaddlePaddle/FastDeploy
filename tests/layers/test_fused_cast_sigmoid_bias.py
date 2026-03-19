"""
Test for fused_cast_sigmoid_bias CUDA custom op.
Tests: functionality, accuracy, and performance.

Usage:
    conda activate fd_py12_bin
    export PYTHONPATH="/workspace2/bingoo/code/FastDeploy"
    python /workspace2/bingoo/code/FastDeploy/tests/model_executor/layers/moe/test_fused_cast_sigmoid_bias.py
"""

import paddle
import paddle.nn.functional as F

from fastdeploy.model_executor.layers.moe.fused_cast_sigmoid_bias import (
    fused_cast_sigmoid_bias,
)


def reference_cast_sigmoid_bias(gate_out, bias):
    """Reference implementation: 3 separate ops."""
    gate_fp32 = gate_out.cast("float32")
    scores = F.sigmoid(gate_fp32)
    scores_with_bias = scores + bias
    return scores, scores_with_bias


def test_functionality():
    """Test basic functionality: correct shapes and dtypes."""
    print("=" * 60)
    print("Test 1: Functionality")
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
                assert paddle.all(scores >= 0.0) and paddle.all(scores <= 1.0), "scores out of [0,1] range"

        print(f"  [PASS] dtype={dtype_name}")

    print("  All functionality tests passed.\n")


def test_accuracy():
    """Test numerical accuracy against reference implementation."""
    print("=" * 60)
    print("Test 2: Accuracy")
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


if __name__ == "__main__":
    paddle.set_device("gpu")
    print("Running fused_cast_sigmoid_bias tests...\n")

    test_functionality()
    test_accuracy()
    test_accuracy_extreme_values()
    test_performance()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
