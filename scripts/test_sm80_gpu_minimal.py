"""
SM80 BF16 minimal test - single process, no engine.
Tests GPU matmul and MoE forward path without distributed setup.
"""
import os, sys
sys.path.insert(0, '/data/lizhijun/work/fd-vllm/FastDeploy')

import paddle
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"PaddlePaddle devices: {paddle.device.cuda.device_count()}")

# Test GPU matmul
paddle.device.set_device('gpu:0')
x = paddle.randn([6, 3072], dtype='bfloat16')
w = paddle.randn([3072, 1536], dtype='bfloat16')
import time
t0 = time.time()
for _ in range(100):
    y = paddle.matmul(x, w)
paddle.device.synchronize()
print(f"100 matmuls took {time.time()-t0:.3f}s")

# Test _apply_ep_sm80_bf16 batched bmm
gate_all = paddle.randn([64, 1536, 3072], dtype='bfloat16')
up_all = paddle.randn([64, 1536, 3072], dtype='bfloat16')
down_all = paddle.randn([64, 3072, 1536], dtype='bfloat16')

M = 6
top_k = 8
hidden_size = 3072
moe_inter = 1536
num_experts = 64

# Simulate routing
import numpy as np
topk_ids_np = np.array([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],
                         [0,8,16,24,32,40,48,56],[1,9,17,25,33,41,49,57],
                         [2,10,18,26,34,42,50,58],[3,11,19,27,35,43,51,59]])
topk_weights_np = np.ones((M, top_k), dtype=np.float32) / top_k

local_start = 0
num_local = 64
expert_rows = [[] for _ in range(num_local)]
expert_wts = [[] for _ in range(num_local)]
for r in range(M):
    for c in range(top_k):
        eid = int(topk_ids_np[r, c])
        if local_start <= eid < local_start + num_local:
            lid = eid - local_start
            expert_rows[lid].append(r)
            expert_wts[lid].append(float(topk_weights_np[r, c]))

active = [lid for lid in range(num_local) if expert_rows[lid]]
print(f"Active experts: {len(active)}")

x_bf16 = paddle.randn([M, hidden_size], dtype='bfloat16')
ffn_out = paddle.zeros([M, hidden_size], dtype='float32')

from collections import defaultdict
groups = defaultdict(list)
for lid in active:
    n_tok = len(expert_rows[lid])
    if n_tok > 0:
        groups[n_tok].append(lid)

t0 = time.time()
for n_tok, lids in groups.items():
    A = len(lids)
    all_rows = []
    all_wts = []
    for lid in lids:
        all_rows.extend(expert_rows[lid])
        all_wts.extend(expert_wts[lid])

    tok_idx = paddle.to_tensor(all_rows, dtype='int64')
    tok = x_bf16[tok_idx]
    tok = tok.reshape([A, n_tok, hidden_size])

    active_idx = paddle.to_tensor(lids, dtype='int64')
    gate_w = gate_all[active_idx]
    up_w = up_all[active_idx]

    g = paddle.bmm(tok, gate_w.transpose([0, 2, 1]))
    u = paddle.bmm(tok, up_w.transpose([0, 2, 1]))

    sw = paddle.nn.functional.swiglu(paddle.concat([g, u], -1))

    down_w = down_all[active_idx]
    o = paddle.bmm(sw, down_w.transpose([0, 2, 1]))

    wt = paddle.to_tensor(all_wts, dtype='float32').reshape([A, n_tok, 1])
    wo = o.cast("float32") * wt
    wo = wo.reshape([A * n_tok, hidden_size])

    ffn_out = paddle.index_put(ffn_out, [tok_idx], wo, accumulate=True)

paddle.device.synchronize()
print(f"MoE forward (M={M}, {len(active)} experts) took {time.time()-t0:.3f}s")
print(f"Output shape: {ffn_out.shape}, dtype: {ffn_out.dtype}")
print("SUCCESS")
