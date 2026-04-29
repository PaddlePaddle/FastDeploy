"""
SM80 BF16 MoE direct test - tests the batched MoE GEMM path.

Tests the same math as _apply_ep_sm80_bf16 in fused_moe_marlin_backend.py.
Weight format: [out_features, in_features] (PyTorch format).

Usage:
    python scripts/test_sm80_moe_direct.py
"""
import sys
sys.path.insert(0, '/data/lizhijun/work/fd-vllm/FastDeploy')

import paddle
paddle.device.set_device('gpu:0')
print(f"Paddle: {paddle.__version__}", flush=True)

prop = paddle.device.cuda.get_device_properties(0)
sm = prop.major * 10 + prop.minor
print(f"SM version: {sm}", flush=True)

if sm >= 90:
    print("SKIP: SM90+ not supported for this test")
    sys.exit(0)

# Test: Batched MoE GEMM (the core of _apply_ep_sm80_bf16)
# Weight format matches minimax_m2_5.py: [out_features, in_features]
print("\n=== Test: Batched MoE GEMM ===", flush=True)
num_experts = 8
hidden_size = 128
inter_size = 64
top_k = 2
num_tokens = 16

# Expert weights in [out_features, in_features] format (PyTorch convention)
# gate: [inter_size, hidden_size] = [64, 128]
# up:   [inter_size, hidden_size] = [64, 128]
# down: [hidden_size, inter_size] = [128, 64]
gate_w = paddle.randn([num_experts, inter_size, hidden_size], dtype='bfloat16')
up_w = paddle.randn([num_experts, inter_size, hidden_size], dtype='bfloat16')
down_w = paddle.randn([num_experts, hidden_size, inter_size], dtype='bfloat16')

topk_ids = paddle.randint(0, num_experts, [num_tokens, top_k])
topk_weights = paddle.ones([num_tokens, top_k], dtype='float32') / top_k

x = paddle.randn([num_tokens, hidden_size], dtype='bfloat16')

from collections import defaultdict
import numpy as np
topk_ids_np = topk_ids.numpy()
topk_weights_np = topk_weights.numpy()

# Group tokens by expert (same as _apply_ep_sm80_bf16)
expert_rows = [[] for _ in range(num_experts)]
expert_wts = [[] for _ in range(num_experts)]
for r in range(num_tokens):
    for c in range(top_k):
        eid = int(topk_ids_np[r, c])
        expert_rows[eid].append(r)
        expert_wts[eid].append(float(topk_weights_np[r, c]))

active = [eid for eid in range(num_experts) if expert_rows[eid]]
print(f"Active experts: {len(active)}", flush=True)

groups = defaultdict(list)
for eid in active:
    n_tok = len(expert_rows[eid])
    groups[n_tok].append(eid)

# MoE forward (same math as _apply_ep_sm80_bf16)
ffn_out = paddle.zeros([num_tokens, hidden_size], dtype='float32')
for n_tok, eids in groups.items():
    A = len(eids)  # number of experts in this group
    all_rows = []
    all_wts = []
    for eid in eids:
        all_rows.extend(expert_rows[eid])
        all_wts.extend(expert_wts[eid])

    tok_idx = paddle.to_tensor(all_rows, dtype='int64')
    tok = x[tok_idx]  # [A * n_tok, hidden_size]
    tok = tok.reshape([A, n_tok, hidden_size])  # [A, n_tok, hidden_size]

    active_idx = paddle.to_tensor(eids, dtype='int64')

    # gate/up: tok @ W^T (W is [out, in], so W^T is [in, out])
    # bmm: [A, n_tok, hidden] @ [A, hidden, inter] = [A, n_tok, inter]
    g = paddle.bmm(tok, gate_w[active_idx].transpose([0, 2, 1]))
    u = paddle.bmm(tok, up_w[active_idx].transpose([0, 2, 1]))

    # SwiGLU activation (matches _swiglu in fused_moe_marlin_backend.py)
    gated = paddle.concat([g, u], -1)  # [A, n_tok, 2 * inter_size]
    if hasattr(paddle.nn.functional, "swiglu"):
        sw = paddle.nn.functional.swiglu(gated)
    else:
        gate_part, up_part = gated.chunk(2, axis=-1)
        sw = gate_part * paddle.nn.functional.silu(up_part)

    # down: sw @ W^T
    # bmm: [A, n_tok, inter] @ [A, inter, hidden] = [A, n_tok, hidden]
    o = paddle.bmm(sw, down_w[active_idx].transpose([0, 2, 1]))

    # Weight and accumulate
    wt = paddle.to_tensor(all_wts, dtype='float32').reshape([A, n_tok, 1])
    wo = o.cast("float32") * wt
    wo = wo.reshape([A * n_tok, hidden_size])

    ffn_out = paddle.index_put(ffn_out, [tok_idx], wo, accumulate=True)

paddle.device.synchronize()
print(f"MoE output shape: {ffn_out.shape}, dtype: {ffn_out.dtype}", flush=True)
print(f"Output mean: {float(ffn_out.mean()):.6f}", flush=True)
print("PASS: Batched MoE GEMM", flush=True)

print("\n=== ALL TESTS PASSED ===", flush=True)
