#!/usr/bin/env python3
"""
Test to compare optimized vs unoptimized w4afp8_gemm performance using pytest
This script compares the performance of the kernel with and without max_tokens_per_expert optimization
"""

import time

import numpy as np
import pytest

# Test parameters
BATCH_PER_EXPERT = 64
WARMUP = 10
ITERS = 20  # Reduced for faster testing
GROUP_SIZE = 128

# All supported test cases from the original benchmark
TEST_CASES = [
    [256, 256, 2, 0, 128],
    [512, 256, 2, 0, 128],
    [256, 5120, 128, 0, 128],
    [3072, 2560, 64, 0, 128],
    [2560, 1536, 64, 0, 128],
    [1536, 2560, 64, 0, 128],
    [2560, 768, 64, 0, 128],
    [768, 2048, 128, 0, 128],
    [2048, 384, 128, 0, 128],
    [3072, 1536, 128, 0, 128],
    [1536, 1536, 128, 0, 128],
    [1536, 768, 128, 0, 128],
    [768, 1536, 128, 0, 128],
]


def bench_fastdeploy_unoptimized(n, k, num_experts, group_size, batch_per_expert):
    """Benchmark without optimization (using max_tokens only)"""
    import paddle

    from fastdeploy.model_executor.ops.gpu import (
        w4afp8_gemm,
        w4afp8_gemm_scale_permute,
        w4afp8_gemm_weight_convert,
    )

    all_tokens = batch_per_expert * num_experts
    tokens = [batch_per_expert] * num_experts

    # Create input (fp8)
    input_fp8 = paddle.randn([all_tokens, k], dtype="bfloat16").astype(paddle.float8_e4m3fn)

    # Create weight
    weight_bf16 = paddle.randn([num_experts, n, k], dtype="bfloat16")
    weight_scale = 7 / weight_bf16.abs().max(axis=-1).reshape([num_experts, n, 1])
    weight_quant = (weight_bf16 * weight_scale).astype("int")
    weight_quant = paddle.clip(weight_quant, -7, 7).astype("bfloat16")
    weight_quant = paddle.where(weight_quant > 0, weight_quant, 8 - weight_quant)
    weight_dequant_scale = 1 / weight_scale.astype("float32")

    weight_int4 = w4afp8_gemm_weight_convert(weight_quant.astype("uint8").cpu()).cuda()

    # Process scale
    processed_scale = weight_dequant_scale * 512
    processed_scale = processed_scale.repeat_interleave(k // group_size, axis=-1)
    origin_shape = processed_scale.shape
    processed_scale = processed_scale.transpose([0, 2, 1])
    processed_scale = processed_scale.reshape([-1, processed_scale.shape[-1]])
    processed_scale = w4afp8_gemm_scale_permute(processed_scale)
    processed_scale = processed_scale.reshape([origin_shape[0], origin_shape[2], origin_shape[1] // 128, 128])
    processed_scale = processed_scale.transpose([0, 2, 1, 3]).astype("float32")

    tokens_prefix_sum = paddle.to_tensor(np.cumsum(tokens), dtype="int64")

    # Warmup - without optimization (pass None for max_tokens_per_expert)
    for _ in range(WARMUP):
        _ = w4afp8_gemm(
            input_fp8, weight_int4, tokens_prefix_sum, processed_scale, None, None, 0, all_tokens, True
        )  # No optimization
    paddle.device.synchronize()

    # Benchmark - without optimization
    times = []
    for _ in range(ITERS):
        paddle.device.synchronize()
        start = time.perf_counter()
        _ = w4afp8_gemm(
            input_fp8, weight_int4, tokens_prefix_sum, processed_scale, None, None, 0, all_tokens, True
        )  # No optimization
        paddle.device.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times) / len(times)
    flops = 2 * all_tokens * n * k
    tflops = flops / (avg_ms / 1000) / 1e12
    return avg_ms, tflops


def bench_fastdeploy_optimized(n, k, num_experts, group_size, batch_per_expert):
    """Benchmark with optimization (using max_tokens_per_expert)"""
    import paddle

    from fastdeploy.model_executor.ops.gpu import (
        w4afp8_gemm,
        w4afp8_gemm_scale_permute,
        w4afp8_gemm_weight_convert,
    )

    all_tokens = batch_per_expert * num_experts
    tokens = [batch_per_expert] * num_experts

    # Create input (fp8)
    input_fp8 = paddle.randn([all_tokens, k], dtype="bfloat16").astype(paddle.float8_e4m3fn)

    # Create weight
    weight_bf16 = paddle.randn([num_experts, n, k], dtype="bfloat16")
    weight_scale = 7 / weight_bf16.abs().max(axis=-1).reshape([num_experts, n, 1])
    weight_quant = (weight_bf16 * weight_scale).astype("int")
    weight_quant = paddle.clip(weight_quant, -7, 7).astype("bfloat16")
    weight_quant = paddle.where(weight_quant > 0, weight_quant, 8 - weight_quant)
    weight_dequant_scale = 1 / weight_scale.astype("float32")

    weight_int4 = w4afp8_gemm_weight_convert(weight_quant.astype("uint8").cpu()).cuda()

    # Process scale
    processed_scale = weight_dequant_scale * 512
    processed_scale = processed_scale.repeat_interleave(k // group_size, axis=-1)
    origin_shape = processed_scale.shape
    processed_scale = processed_scale.transpose([0, 2, 1])
    processed_scale = processed_scale.reshape([-1, processed_scale.shape[-1]])
    processed_scale = w4afp8_gemm_scale_permute(processed_scale)
    processed_scale = processed_scale.reshape([origin_shape[0], origin_shape[2], origin_shape[1] // 128, 128])
    processed_scale = processed_scale.transpose([0, 2, 1, 3]).astype("float32")

    tokens_prefix_sum = paddle.to_tensor(np.cumsum(tokens), dtype="int64")

    # Create max_tokens_per_expert tensor - this enables the optimization!
    max_tokens_per_expert = paddle.to_tensor([batch_per_expert] * num_experts, dtype="int64")

    # Warmup - with optimization
    for _ in range(WARMUP):
        _ = w4afp8_gemm(
            input_fp8,
            weight_int4,
            tokens_prefix_sum,
            processed_scale,
            None,
            max_tokens_per_expert,
            0,
            all_tokens,
            True,
        )  # With optimization
    paddle.device.synchronize()

    # Benchmark - with optimization
    times = []
    for _ in range(ITERS):
        paddle.device.synchronize()
        start = time.perf_counter()
        _ = w4afp8_gemm(
            input_fp8,
            weight_int4,
            tokens_prefix_sum,
            processed_scale,
            None,
            max_tokens_per_expert,
            0,
            all_tokens,
            True,
        )  # With optimization
        paddle.device.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times) / len(times)
    flops = 2 * all_tokens * n * k
    tflops = flops / (avg_ms / 1000) / 1e12
    return avg_ms, tflops


header_printed = False


@pytest.mark.parametrize("case", TEST_CASES)
def test_optimization_performance(case):
    """
    Test to compare optimized vs unoptimized performance for w4afp8_gemm
    Each test case runs both versions and verifies the optimized version is not slower
    """
    global header_printed
    n, k, num_experts, _, group_size = case

    try:
        # Print header only once
        if not header_printed:
            print("\nCase\tN\tK\tE\t优化前 ms\t现在 ms\t加速比")
            print("-" * 60)
            header_printed = True

        # Test unoptimized version
        unopt_ms, unopt_tf = bench_fastdeploy_unoptimized(n, k, num_experts, group_size, BATCH_PER_EXPERT)

        # Test optimized version
        opt_ms, opt_tf = bench_fastdeploy_optimized(n, k, num_experts, group_size, BATCH_PER_EXPERT)

        # Calculate improvement
        speedup = unopt_ms / opt_ms

        # Print individual test result in required format
        case_index = TEST_CASES.index([n, k, num_experts, 0, group_size]) + 1
        print(f"{case_index}\t{n}\t{k}\t{num_experts}\t{unopt_ms:.3f}\t{opt_ms:.3f}\t{speedup:.2f}x")

    except NotImplementedError:
        # Skip unsupported cases
        pytest.skip(f"Skipping unsupported case: N={n}, K={k}, Experts={num_experts}")
        return None


def test_all_cases_summary():
    """Summary test that runs all cases and checks if 50% of cases have speedup > 1"""
    print("\n" + "=" * 80)
    print("SUMMARY: Optimization Performance Comparison")
    print(f"Batch per expert: {BATCH_PER_EXPERT}, Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 80)
    print("Case\tN\tK\tE\t优化前 ms\t现在 ms\t加速比")
    print("-" * 80)

    successful_tests = []
    failed_tests = []

    for idx, case in enumerate(TEST_CASES):
        n, k, num_experts, _, group_size = case

        try:
            # Run individual test
            unopt_ms, unopt_tf = bench_fastdeploy_unoptimized(n, k, num_experts, group_size, BATCH_PER_EXPERT)
            opt_ms, opt_tf = bench_fastdeploy_optimized(n, k, num_experts, group_size, BATCH_PER_EXPERT)

            speedup = unopt_ms / opt_ms

            case_num = idx + 1
            print(f"{case_num}\t{n}\t{k}\t{num_experts}\t{unopt_ms:.3f}\t{opt_ms:.3f}\t{speedup:.2f}x")

            successful_tests.append(
                {
                    "case_num": case_num,
                    "n": n,
                    "k": k,
                    "experts": num_experts,
                    "unopt_ms": unopt_ms,
                    "opt_ms": opt_ms,
                    "speedup": speedup,
                }
            )

        except NotImplementedError as e:
            print(f"{idx+1}\t{n}\t{k}\t{num_experts}\tERROR\tERROR\tSKIPPED")
            failed_tests.append({"case_num": idx + 1, "error": str(e)})
        except Exception as e:
            print(f"{idx+1}\t{n}\t{k}\t{num_experts}\tERROR\tERROR\tERROR")
            failed_tests.append({"case_num": idx + 1, "error": str(e)})

    # Calculate statistics
    if successful_tests:
        total_tests = len(successful_tests) + len(failed_tests)
        fast_cases = sum(1 for test in successful_tests if test["speedup"] > 1.0)
        success_rate = len(successful_tests) / total_tests if total_tests > 0 else 0
        fast_rate = fast_cases / len(successful_tests) if successful_tests else 0

        print("-" * 80)
        print(f"Total cases: {total_tests}")
        print(f"Successful tests: {len(successful_tests)}")
        print(f"Failed/Skipped tests: {len(failed_tests)}")
        print(
            f"Cases with speedup > 1.0: {fast_cases}/{len(successful_tests)} ({fast_rate*100:.1f}% of successful tests)"
        )
        print(f"Overall success rate: {success_rate*100:.1f}%")

        # Check if at least 50% of successful cases have speedup > 1
        if fast_rate >= 0.5:
            print(f"✅ PASS: {fast_rate*100:.1f}% of successful cases show performance improvement (≥50% threshold)")
        else:
            print(
                f"❌ FAIL: Only {fast_rate*100:.1f}% of successful cases show performance improvement (<50% threshold)"
            )

        print("=" * 80)

        # Final assertion for pytest
        assert fast_rate >= 0.5, f"Only {fast_rate*100:.1f}% of cases had speedup > 1, below required 50% threshold"
    else:
        print("No successful tests completed")
        assert False, "No successful tests completed"


if __name__ == "__main__":
    # Run the parametrized tests
    pytest.main([__file__, "-v"])

    # Also run summary
    print("\nRunning summary test...")
    test_all_cases_summary()
