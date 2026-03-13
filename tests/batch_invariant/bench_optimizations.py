"""Test & benchmark optimizations for matmul_persistent kernel.

Three-phase validation per experiment:
  Phase 1 - Correctness: result matches baseline matmul_persistent (bit-exact)
  Phase 2 - Determinism: batch invariance (fn(a[:1],b) == fn(a,b)[:1], diff=0)
  Phase 3 - Performance: latency comparison vs baseline and cuBLAS

Does NOT modify existing code.
"""

import time

import paddle
import triton

from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
    matmul_kernel_persistent,
    matmul_persistent,
)

# ── Constants ──────────────────────────────────────────────────────────
K_DIM = 7168  # hidden_size (DeepSeek V3)
N_DIM = 256  # n_routed_experts
M_LIST = [32, 64, 128, 256, 512, 1024, 2048]
DTYPES = [("fp32", paddle.float32), ("bf16", paddle.bfloat16)]
WARMUP = 50
TIMED = 200

_NUM_SMS = None


def get_num_sms():
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = paddle.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SMS


# ── Generic kernel launcher ───────────────────────────────────────────
def launch_persistent(a, b, block_m, block_n, block_k, group_m=8, stages=3, warps=8):
    M, K_ = a.shape
    K_, N_ = b.shape
    c = paddle.empty((M, N_), dtype=a.dtype)
    NUM_SMS = get_num_sms()

    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_, META["BLOCK_SIZE_N"])),)

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        None,
        M,
        N_,
        K_,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=int(M * K_ > 2**31),
        B_LARGE=int(K_ * N_ > 2**31),
        C_LARGE=int(M * N_ > 2**31),
        HAS_BIAS=0,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        num_stages=stages,
        num_warps=warps,
    )
    return c


# ── Baseline config ───────────────────────────────────────────────────
BASELINE_CFG = {
    paddle.float32: {"block_m": 128, "block_n": 128, "block_k": 32, "stages": 3},
    paddle.bfloat16: {"block_m": 128, "block_n": 128, "block_k": 64, "stages": 3},
}


# ── Experiment factories ──────────────────────────────────────────────
def make_exp1_cached_sms():
    """Same config as baseline, but NUM_SMS cached."""

    def fn(a, b):
        c = BASELINE_CFG[a.dtype]
        return launch_persistent(a, b, c["block_m"], c["block_n"], c["block_k"], stages=c["stages"])

    return fn


def make_exp2_adaptive_block_m():
    """Adaptive BLOCK_SIZE_M based on M."""

    def fn(a, b):
        M = a.shape[0]
        bsm = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
        c = BASELINE_CFG[a.dtype]
        return launch_persistent(a, b, bsm, c["block_n"], c["block_k"], stages=c["stages"])

    return fn


def make_exp3_fp32_k64():
    """fp32 BLOCK_SIZE_K 32->64, but reduce BLOCK_SIZE_N to 64 and stages=2 to fit shared memory."""

    def fn(a, b):
        if a.dtype == paddle.float32:
            # shared mem = stages * (block_m*block_k + block_k*block_n) * 4 bytes
            # 2 * (128*64 + 64*64) * 4 = 2 * (8192+4096) * 4 = 98304 < 232448 OK
            return launch_persistent(a, b, 128, 64, 64, stages=2)
        c = BASELINE_CFG[a.dtype]
        return launch_persistent(a, b, c["block_m"], c["block_n"], c["block_k"], stages=c["stages"])

    return fn


def make_exp4_stages(stages):
    """Change num_stages."""

    def fn(a, b):
        c = BASELINE_CFG[a.dtype]
        return launch_persistent(a, b, c["block_m"], c["block_n"], c["block_k"], stages=stages)

    return fn


def make_exp5_combined():
    """Cached SMS + adaptive BLOCK_M + tuned stages."""

    def fn(a, b):
        M = a.shape[0]
        bsm = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
        c = BASELINE_CFG[a.dtype]
        return launch_persistent(a, b, bsm, c["block_n"], c["block_k"], stages=4)

    return fn


# ── Phase 1: Correctness (vs baseline matmul_persistent, bit-exact) ──
def check_correctness(fn, label):
    """Compare fn output vs matmul_persistent (baseline). Must be bit-exact for same config,
    or numerically close for different config (changed K tiling changes accumulation order)."""
    print(f"\n  [Correctness] {label}")
    all_ok = True
    for dt_name, dtype in DTYPES:
        for M in [32, 128, 512, 2048]:
            a = paddle.randn([M, K_DIM], dtype=dtype)
            b = paddle.randn([K_DIM, N_DIM], dtype=dtype)
            ref = matmul_persistent(a, b)
            out = fn(a, b)
            diff = (out - ref).abs().max().item()
            # For variants that don't change K tiling, expect bit-exact (diff=0)
            # For variants that change K tiling (exp3), expect small diff
            ok = True
            if diff == 0:
                status = "PASS (exact)"
            else:
                # Check relative error vs cuBLAS as sanity
                ref_cublas = paddle.mm(a, b)
                baseline_vs_cublas = (ref - ref_cublas).abs().max().item()
                opt_vs_cublas = (out - ref_cublas).abs().max().item()
                # Optimized variant should not be significantly worse than baseline vs cuBLAS
                if opt_vs_cublas <= baseline_vs_cublas * 2 + 1e-6:
                    status = f"PASS (diff={diff:.2e}, same order as baseline vs cuBLAS)"
                else:
                    status = f"FAIL (diff={diff:.2e}, baseline_vs_cublas={baseline_vs_cublas:.2e}, opt_vs_cublas={opt_vs_cublas:.2e})"
                    ok = False
            print(f"    M={M:>5} {dt_name}: {status}")
            if not ok:
                all_ok = False
    return all_ok


# ── Phase 2: Determinism (batch invariance) ───────────────────────────
def check_determinism(fn, label, iters=5):
    """Verify fn(a[:1],b) == fn(a,b)[:1]."""
    print(f"\n  [Determinism] {label}")
    all_ok = True
    for dt_name, dtype in DTYPES:
        for M in [64, 256, 1024, 2048]:
            D = 4096
            a = paddle.linspace(-100, 100, M * D, dtype=dtype).reshape([M, D])
            b = paddle.linspace(-100, 100, D * D, dtype=dtype).reshape([D, D])
            max_diff = 0.0
            for _ in range(iters):
                out_single = fn(a[:1], b)
                out_batch = fn(a, b)[:1]
                diff = (out_single - out_batch).abs().max().item()
                max_diff = max(max_diff, diff)
            ok = max_diff == 0
            status = "PASS" if ok else f"FAIL (max_diff={max_diff})"
            print(f"    M={M:>5} {dt_name}: {status}")
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


def run_perf(fn, label):
    print(f"\n  [Performance] {label}")
    rows = []
    for dt_name, dtype in DTYPES:
        for M in M_LIST:
            a = paddle.randn([M, K_DIM], dtype=dtype)
            b = paddle.randn([K_DIM, N_DIM], dtype=dtype)
            ms = bench(fn, a, b)
            rows.append((M, dt_name, ms))
            print(f"    M={M:>5} {dt_name}: {ms:.4f} ms")
    return rows


def print_perf_table(label, cublas_rows, baseline_rows, opt_rows):
    print(f"\n  {label}: Performance Table")
    hdr = f"  | {'M':>5} | {'dtype':>5} | {'cuBLAS':>10} | {'baseline':>10} | {'optimized':>10} | {'vs base':>8} | {'vs cuBLAS':>9} |"
    print(hdr)
    sep = f"  |{'-'*7}|{'-'*7}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*11}|"
    print(sep)
    for (M, dt, cb), (_, _, bl), (_, _, op) in zip(cublas_rows, baseline_rows, opt_rows):
        vs_base = bl / op if op > 0 else float("inf")
        vs_cublas = op / cb if cb > 0 else float("inf")
        print(
            f"  | {M:>5} | {dt:>5} | {cb:>10.4f} | {bl:>10.4f} | {op:>10.4f} | {vs_base:>7.2f}x | {vs_cublas:>8.2f}x |"
        )


# ── Run one experiment ────────────────────────────────────────────────
def run_experiment(name, fn, cublas_rows, baseline_rows):
    print(f"\n{'#'*80}")
    print(f"# {name}")
    print(f"{'#'*80}")

    c_ok = check_correctness(fn, name)
    if not c_ok:
        print(f"\n  >>> {name}: CORRECTNESS FAILED, skipping remaining phases.")
        return False, False, None

    d_ok = check_determinism(fn, name)
    if not d_ok:
        print(f"\n  >>> {name}: DETERMINISM FAILED, skipping performance.")
        return True, False, None

    perf = run_perf(fn, name)
    print_perf_table(name, cublas_rows, baseline_rows, perf)
    return True, True, perf


# ── Main ──────────────────────────────────────────────────────────────
def main():
    paddle.set_device("gpu")
    props = paddle.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, SM count: {get_num_sms()}")
    print(f"Matrix: [M, {K_DIM}] x [{K_DIM}, {N_DIM}]  (DeepSeek V3 Gate)")

    experiments = [
        ("Exp1: cached NUM_SMS", make_exp1_cached_sms()),
        ("Exp2: adaptive BLOCK_SIZE_M", make_exp2_adaptive_block_m()),
        ("Exp3: fp32 K=64 (N=64,s=2)", make_exp3_fp32_k64()),
        ("Exp4a: num_stages=4", make_exp4_stages(4)),
        ("Exp4b: num_stages=5", make_exp4_stages(5)),
        ("Exp5: combined", make_exp5_combined()),
    ]

    # Reference runs
    print("\n--- cuBLAS reference ---")
    cublas_rows = run_perf(paddle.mm, "cuBLAS")

    print("\n--- Baseline (matmul_persistent, no cache) ---")
    baseline_rows = run_perf(matmul_persistent, "baseline")

    # Experiments
    results = {}
    for name, fn in experiments:
        c_ok, d_ok, perf = run_experiment(name, fn, cublas_rows, baseline_rows)
        results[name] = {"correctness": c_ok, "determinism": d_ok, "perf": perf}

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"| {'Experiment':<30} | {'Correct':>8} | {'Determ.':>8} | {'Perf':>8} |")
    print(f"|{'-'*32}|{'-'*10}|{'-'*10}|{'-'*10}|")
    for name, r in results.items():
        c = "PASS" if r["correctness"] else "FAIL"
        d = "PASS" if r["determinism"] else "FAIL"
        p = "YES" if r["perf"] else "SKIP"
        print(f"| {name:<30} | {c:>8} | {d:>8} | {p:>8} |")


if __name__ == "__main__":
    main()
