"""Benchmark: cuBLAS paddle.mm vs Triton matmul_persistent for MoE Gate."""

import time

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
    matmul_persistent,
)

WARMUP = 50
TIMED = 200
K = 7168  # hidden_size (DeepSeek V3)
N = 256  # n_routed_experts
M_LIST = [32, 64, 128, 256, 512, 1024, 2048]
DTYPES = [("fp32", paddle.float32), ("bf16", paddle.bfloat16)]


def bench(fn, a, b, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        fn(a, b)
    paddle.device.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        fn(a, b)
    paddle.device.cuda.synchronize()
    return (time.perf_counter() - t0) / timed * 1000  # ms


def main():
    paddle.set_device("gpu")
    rows = []
    for dtype_name, dtype in DTYPES:
        for M in M_LIST:
            a = paddle.randn([M, K], dtype=dtype)
            b = paddle.randn([K, N], dtype=dtype)

            cublas_ms = bench(paddle.mm, a, b)
            triton_ms = bench(matmul_persistent, a, b)
            slowdown = triton_ms / cublas_ms if cublas_ms > 0 else float("inf")
            rows.append((M, dtype_name, cublas_ms, triton_ms, slowdown))

    # Print table
    print(f"| {'M':>5} | {'dtype':>5} | {'cuBLAS (ms)':>12} | {'Triton persistent (ms)':>23} | {'Slowdown':>8} |")
    print(f"|{'-'*7}|{'-'*7}|{'-'*14}|{'-'*25}|{'-'*10}|")
    for M, dt, cb, tr, sd in rows:
        print(f"| {M:>5} | {dt:>5} | {cb:>12.4f} | {tr:>23.4f} | {sd:>7.2f}x |")


if __name__ == "__main__":
    main()
