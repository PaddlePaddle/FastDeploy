import paddle
import numpy as np
from fastdeploy.model_executor.layers.attention.ops import flash_attn_v4

paddle.seed(0)
dtype = paddle.bfloat16

def attention_naive(q_input, k_input, v_input, cu_seq_q, cu_seq_k):
    bsz = cu_seq_q.shape[0] - 1  
    q_token_sum, num_head, head_dim = q_input.shape 
    k_token_sum, num_kv_head, _ = k_input.shape 
    gqa_group_size = num_head // num_kv_head 
    qk_scale = 1 / np.sqrt(head_dim)  
    out = paddle.zeros([num_head, q_token_sum, head_dim], q_input.dtype)  
    for bi in range(bsz):  
        q = q_input[cu_seq_q[bi]: cu_seq_q[bi + 1], :, :].transpose([1, 0, 2]).astype("float32").numpy()
        k = k_input[cu_seq_k[bi]: cu_seq_k[bi + 1], :, :].transpose([1, 2, 0]).astype("float32").numpy() 
        v = v_input[cu_seq_k[bi]: cu_seq_k[bi + 1], :, :].transpose([1, 0, 2]).astype("float32").numpy()  
        qk = np.matmul(q, np.repeat(k, gqa_group_size, 0)) 
        qk *= qk_scale  
        condition = np.tril(np.ones(qk.shape), q.shape[1] - k.shape[2])  
        mask = np.ones(condition.shape).astype("float32") * -1000000 
        qk = np.where(condition>0, qk, mask) 
        qk_max = qk.max(axis=-1, keepdims=True)  
        qk -= qk_max  
        qk = np.exp(qk)
        exp_sum = qk.sum(axis=-1, keepdims=True) 
        exp_sum_inv = 1.0 / exp_sum  
        temp_out = paddle.to_tensor(np.matmul(qk, np.repeat(v, gqa_group_size, 0))) 
        out[:, cu_seq_q[bi]: cu_seq_q[bi + 1], :] = temp_out * exp_sum_inv 
    return out.transpose([1, 0, 2])  


for hq, hk in [(56, 4)]:
    for seq_len_q in [1, 33, 888, 1024, 2048]:
        cu_seqlens_q = paddle.to_tensor([0, seq_len_q]).astype("int32")
        cu_seqlens_k = paddle.to_tensor([0, seq_len_q]).astype("int32")
        q_token_num = cu_seqlens_q[-1]
        k_token_num = cu_seqlens_k[-1]
        q = np.random.normal(0, 1, size=(q_token_num, hq, 128))
        k = np.random.normal(0, 1, size=(k_token_num, hk, 128))
        v = np.random.normal(0, 1, size=(k_token_num, hk, 128))
        q = paddle.to_tensor(q).astype("bfloat16")
        k = paddle.to_tensor(k).astype("bfloat16")
        v = paddle.to_tensor(v).astype("bfloat16")
        mask = paddle.arange(q_token_num).astype("int32") + 1
        out1 = paddle.empty((q_token_num, hq, 128), dtype=dtype, device="cuda")
        out2 = paddle.empty((q_token_num, hq, 128), dtype=dtype, device="cuda")
        flash_attn_v4(q, k, v, cu_seqlens_q, cu_seqlens_k, out1, mask)
        flash_attn_v4(q, k, v, cu_seqlens_q, cu_seqlens_k, out2, mask)
        naive_out = attention_naive(q, k, v, cu_seqlens_q, cu_seqlens_k)
        gap = abs(out1 - naive_out)
        diff = float(abs(out1 - out2).max())
        print(f"hq:{hq}  hk:{hk}  seq_len_q:{seq_len_q}  max_gap:{float(gap.max())}  mean_gap:{float(gap.mean())}  diff:{diff}")

