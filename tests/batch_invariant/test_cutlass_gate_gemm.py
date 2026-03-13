"""Test & benchmark for CUTLASS batch-invariant GEMM kernel.

Three-phase validation:
  Phase 1 - Correctness: result matches cuBLAS (within tolerance)
  Phase 2 - Determinism: batch invariance (same token, different M → bitwise identical)
  Phase 3 - Performance: latency comparison vs cuBLAS, Triton persistent, Triton Split-K
"""

import sys
import time
from functools import partial

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
    matmul_cutlass_gate,
    matmul_persistent,
    matmul_splitk,
)

# ── Constants ──────────────────────────────────────────────────────────
K_DIM = 7168  # hidden_size (DeepSeek V3)
N_DIM = 256  # n_routed_experts
M_LIST = [32, 64, 128, 256, 512, 1024, 2048]
DTYPES = [("bf16", paddle.bfloat16), ("fp16", paddle.float16), ("fp32", paddle.float32)]
WARMUP = 50
TIMED = 200


# ── Phase 1: Correctness ──────────────────────────────────────────────
def check_correctness():
    """CUTLASS batch-invariant GEMM result vs cuBLAS (paddle.mm) reference."""
    print("\n[Phase 1: Correctness]")
    all_ok = True

    # Warmup: first CUTLASS call may trigger JIT compilation
    _a = paddle.randn([32, K_DIM], dtype=paddle.bfloat16)
    _b = paddle.randn([K_DIM, N_DIM], dtype=paddle.bfloat16)
    matmul_cutlass_gate(_a, _b)
    paddle.device.cuda.synchronize()

    for dt_name, dtype in DTYPES:
        for M in [1, 32, 128, 256, 512, 2048]:
            a = paddle.randn([M, K_DIM], dtype=dtype)
            b = paddle.randn([K_DIM, N_DIM], dtype=dtype)

            ref_cublas = paddle.mm(a, b)
            out_cutlass = matmul_cutlass_gate(a, b)
            ref_persistent = matmul_persistent(a, b)

            cutlass_err = (out_cutlass.cast("float32") - ref_cublas.cast("float32")).abs().max().item()
            persistent_err = (ref_persistent.cast("float32") - ref_cublas.cast("float32")).abs().max().item()

            # SplitK changes K-accumulation order vs single-pass, expect ~1-2 ULP diff.
            # Use dtype-aware tolerance: bf16 ULP ~0.5-2.0, fp16 ~0.25, fp32 ~1e-5.
            dtype_tol = {paddle.bfloat16: 4.0, paddle.float16: 0.5, paddle.float32: 1e-3}
            ok = cutlass_err <= max(persistent_err * 3, dtype_tol[dtype])
            status = "PASS" if ok else "FAIL"
            print(
                f"  M={M:>5} {dt_name}: {status} "
                f"(cutlass_vs_cublas={cutlass_err:.2e}, "
                f"persistent_vs_cublas={persistent_err:.2e})"
            )
            if not ok:
                all_ok = False
    return all_ok


def check_correctness_bias():
    """CUTLASS batch-invariant GEMM with bias vs cuBLAS + bias reference."""
    print("\n[Phase 1b: Correctness with bias]")
    all_ok = True
    # Fused epilogue does acc+bias in fp32 then truncates to dtype,
    # while manual add operates in dtype. Allow dtype-scale tolerance.
    dtype_atol = {"bf16": 4.0, "fp16": 0.5, "fp32": 1e-4}
    for dt_name, dtype in DTYPES:
        for M in [1, 128, 2048]:
            a = paddle.randn([M, K_DIM], dtype=dtype)
            b = paddle.randn([K_DIM, N_DIM], dtype=dtype)
            bias = paddle.randn([N_DIM], dtype=dtype)

            ref = paddle.mm(a, b) + bias.unsqueeze(0)
            out = matmul_cutlass_gate(a, b, bias=bias)

            err = (out.cast("float32") - ref.cast("float32")).abs().max().item()
            # Fair baseline: CUTLASS no-bias result + manual bias add
            out_nobias = matmul_cutlass_gate(a, b)
            ref_manual = out_nobias + bias.unsqueeze(0)
            baseline_err = (ref_manual.cast("float32") - ref.cast("float32")).abs().max().item()
            tol = max(baseline_err * 3, dtype_atol[dt_name])
            ok = err <= tol
            status = "PASS" if ok else "FAIL"
            print(
                f"  M={M:>5} {dt_name}: {status} "
                f"(fused_vs_cublas={err:.2e}, manual_vs_cublas={baseline_err:.2e}, tol={tol:.2e})"
            )
            if not ok:
                all_ok = False
    return all_ok


# ── Phase 2: Determinism (batch invariance) ────────────────────────────
def check_determinism(iters=10):
    """Verify same token produces bitwise identical output under different M.

    Core test: fix token data for rows 0..min_M-1, vary total M,
    and check that the output for those rows is bitwise identical.
    This is the definition of batch invariance.
    """
    print("\n[Phase 2: Batch Invariance]")
    all_ok = True
    M_sizes = [64, 256, 1024, 2048]
    min_M = min(M_sizes)

    for dt_name, dtype in DTYPES:
        # Fixed token data: the first min_M rows are always the same
        a_full = paddle.randn([max(M_sizes), K_DIM], dtype=dtype)
        b = paddle.randn([K_DIM, N_DIM], dtype=dtype)

        max_diff = 0.0
        for _ in range(iters):
            # Reference: compute with smallest M
            ref = matmul_cutlass_gate(a_full[:min_M], b)

            for M in M_sizes:
                # Compute with larger M, check first min_M rows match
                out = matmul_cutlass_gate(a_full[:M], b)[:min_M]
                diff = (out - ref).abs().max().item()
                max_diff = max(max_diff, diff)

        ok = max_diff == 0
        status = "PASS" if ok else f"FAIL (max_diff={max_diff})"
        print(f"  {dt_name} M={M_sizes}: {status}")
        if not ok:
            all_ok = False
    return all_ok


def check_determinism_across_runs(runs=100):
    """Same inputs produce identical outputs across multiple runs."""
    print("\n[Phase 2b: Run-to-run Determinism]")
    all_ok = True
    for dt_name, dtype in [("bf16", paddle.bfloat16)]:
        a = paddle.randn([256, K_DIM], dtype=dtype)
        b = paddle.randn([K_DIM, N_DIM], dtype=dtype)
        ref = matmul_cutlass_gate(a, b)
        max_diff = 0.0
        for _ in range(runs):
            out = matmul_cutlass_gate(a, b)
            diff = (out - ref).abs().max().item()
            max_diff = max(max_diff, diff)
        ok = max_diff == 0
        status = "PASS" if ok else f"FAIL (max_diff={max_diff})"
        print(f"  {dt_name} 256x{K_DIM}x{N_DIM} ({runs} runs): {status}")
        if not ok:
            all_ok = False
    return all_ok


# ── Phase 3: Performance ──────────────────────────────────────────────
def bench(fn, a, b, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        fn(a, b)
    paddle.device.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        fn(a, b)
    paddle.device.cuda.synchronize()
    return (time.perf_counter() - t0) / timed * 1000


def run_perf():
    print("\n[Phase 3: Performance]")
    splitk_fn = partial(matmul_splitk, split_k=8)

    header = (
        f"  | {'M':>5} | {'dtype':>5} | {'cuBLAS':>10} | {'persistent':>10} | "
        f"{'Split-K':>10} | {'CUTLASS':>10} | {'vs persist':>10} | {'vs cuBLAS':>10} |"
    )
    sep = f"  |{'-'*7}|{'-'*7}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|"
    print(header)
    print(sep)

    for dt_name, dtype in DTYPES:
        for M in M_LIST:
            a = paddle.randn([M, K_DIM], dtype=dtype)
            b = paddle.randn([K_DIM, N_DIM], dtype=dtype)

            ms_cublas = bench(paddle.mm, a, b)
            ms_persistent = bench(matmul_persistent, a, b)
            ms_splitk = bench(splitk_fn, a, b)
            ms_cutlass = bench(matmul_cutlass_gate, a, b)

            vs_persist = ms_persistent / ms_cutlass if ms_cutlass > 0 else float("inf")
            vs_cublas = ms_cutlass / ms_cublas if ms_cublas > 0 else float("inf")

            print(
                f"  | {M:>5} | {dt_name:>5} | {ms_cublas:>10.4f} | {ms_persistent:>10.4f} | "
                f"{ms_splitk:>10.4f} | {ms_cutlass:>10.4f} | {vs_persist:>9.2f}x | {vs_cublas:>9.2f}x |"
            )


def main():
    paddle.set_device("gpu")
    props = paddle.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, SM count: {props.multi_processor_count}")
    print(f"Matrix: [M, {K_DIM}] x [{K_DIM}, {N_DIM}]  (batch-invariant GEMM benchmark)")

    # Phase 1: Correctness
    c_ok = check_correctness()
    c_bias_ok = check_correctness_bias()
    if not (c_ok and c_bias_ok):
        print("\n>>> CORRECTNESS FAILED, skipping remaining phases.")
        sys.exit(1)

    # Phase 2: Determinism
    d_ok = check_determinism()
    d_run_ok = check_determinism_across_runs()
    if not (d_ok and d_run_ok):
        print("\n>>> DETERMINISM FAILED, skipping performance.")
        sys.exit(1)

    # Phase 3: Performance
    run_perf()

    print(f"\n{'='*80}")
    print("  All phases passed.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
