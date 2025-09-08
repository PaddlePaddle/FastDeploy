from fastdeploy.model_executor.ops.gpu import merge_prefill_decode_output, flash_attention_mask, split_qkv_and_rope, fused_block_mean_and_rope
import paddle 

seq_len = 1500
bsz = 4
num_heads = 20
kv_num_heads = 4
head_dim = 128
q = paddle.zeros([seq_len * bsz, num_heads * head_dim], dtype="float16")
k = paddle.zeros([seq_len * bsz, kv_num_heads * head_dim], dtype="float16")
v = paddle.zeros([seq_len * bsz, kv_num_heads * head_dim], dtype="float16")

qkv = paddle.rand([seq_len * bsz, (num_heads + 2 * kv_num_heads) * head_dim], dtype='bfloat16')

rotary_embs = paddle.ones([2, 8192, 64], dtype="float32")

seq_lens_encoder = paddle.ones([bsz], dtype="int32") * seq_len
seq_lens_decoder = paddle.zeros([bsz], dtype="int32")

cu_seqlens_q = paddle.arange(bsz + 1).astype("int32") * seq_len
cu_seqlens_k = paddle.arange(bsz + 1).astype("int32") * seq_len

split_qkv_and_rope(
    qkv,
    q,
    k,
    v,
    rotary_embs,
    seq_lens_encoder,
    seq_lens_decoder,
    cu_seqlens_q,
    cu_seqlens_k,
    None,
    num_heads,
    kv_num_heads,
    head_dim,
    int(seq_len),
    int(8192),
    "none"
) 