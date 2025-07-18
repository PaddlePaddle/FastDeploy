"""
# Copyright (c) 2025 MetaX-tech Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import paddle
import paddle.nn.functional as F
import math
from fastdeploy.model_executor.layers.attention import flash_attn_unpadded_func, flash_attn_kvcache_func
from fastdeploy.model_executor.forward_meta import ForwardMeta

def apply_rotary_embedding_neox(q, k):
    dim = q.shape[-1]
    half = dim // 2

    seq_len = q.shape[-2]
    freqs = paddle.arange(half, dtype=q.dtype)
    inv_freq = 1.0 / (10000 ** (freqs / half))
    t = paddle.arange(seq_len, dtype=q.dtype)
    freqs = paddle.einsum('i,j->ij', t, inv_freq)  # [seq_len, half]

    cos = paddle.cos(freqs)[None, :, None, :]  # [1, seq, 1, half]
    sin = paddle.sin(freqs)[None, :, None, :]  # [1, seq, 1, half]

    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return paddle.concat([-x2, x1], axis=-1)

    def apply(x):
        x1, x2 = x[..., :half], x[..., half:]
        x_out = paddle.concat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], axis=-1)
        return x_out

    return apply(q), apply(k)

def split_qkv(qkv, num_head_q, num_head_kv, dim):
    q = qkv[:, :num_head_q * dim].reshape([-1, num_head_q, dim])
    k = qkv[:, num_head_q * dim:num_head_q * dim + num_head_kv * dim].reshape([-1, num_head_kv, dim])
    v = qkv[:, num_head_q * dim + num_head_kv * dim:].reshape([-1, num_head_kv, dim])
    return q, k, v


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    # TODO to replace it with original flash_attn_varlen!
    total_q = q.shape[0]
    total_k = k.shape[0]
    num_head = q.shape[1]
    dim = q.shape[2]

    q_ = q.reshape([-1, num_head, dim])
    k_ = k.reshape([-1, num_head, dim])
    v_ = v.reshape([-1, num_head, dim])

    bsz = cu_seqlens_q.shape[0] - 1
    out = []
    for i in range(bsz):
        start_q, end_q = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        start_k, end_k = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        qi = q_[start_q:end_q]  # [seq_q, nh, dim]
        ki = k_[start_k:end_k]  # [seq_k, nh, dim]
        vi = v_[start_k:end_k]  # [seq_k, nh, dim]
        qi = qi.transpose([1, 0, 2])  # [nh, seq_q, dim]
        ki = ki.transpose([1, 2, 0])  # [nh, dim, seq_k]
        vi = vi.transpose([1, 0, 2])  # [nh, seq_k, dim]

        score = paddle.matmul(qi, ki) / math.sqrt(dim)  # [nh, seq_q, seq_k]
        prob = F.softmax(score, axis=-1)
        o = paddle.matmul(prob, vi)  # [nh, seq_q, dim]
        o = o.transpose([1, 0, 2])  # [seq_q, nh, dim]
        out.append(o)

    return paddle.concat(out, axis=0)  # [total_q, nh, dim]


def flash_attn_with_kvcache(q, cache_k, cache_v, cache_seqlens, block_tables=None):
    # TODO to implement it!
    bs, _, nh, dim = q.shape
    out = []
    for i in range(bs):
        q_i = q[i]  # [1, nh, dim]
        k_i = cache_k[i, :cache_seqlens[i, 0]]  # [seqlen, nh, dim]
        v_i = cache_v[i, :cache_seqlens[i, 0]]
        qi = q_i.transpose([1, 0, 2])  # [nh, 1, dim]
        ki = k_i.transpose([1, 2, 0])  # [nh, dim, seqlen]
        vi = v_i.transpose([1, 0, 2])  # [nh, seqlen, dim]
        score = paddle.matmul(qi, ki) / math.sqrt(dim)
        prob = F.softmax(score, axis=-1)
        o = paddle.matmul(prob, vi).transpose([1, 0, 2])  # [1, nh, dim]
        out.append(o)
    return paddle.concat(out, axis=0)  # [bs, nh, dim]

def block_cache_to_naive_cache(
    cache_k, cache_v, bsz, block_tables, cache_seq_len
):
    _, num_head, blocksize, dim_head = cache_k.shape
    out_cache_k = paddle.zeros(
        shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_k.dtype
    )
    out_cache_v = paddle.zeros(
        shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_v.dtype
    )
    for i in range(bsz):
        for j in range(cache_seq_len):
            out_cache_k[i, :, j, :] = cache_k[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
            out_cache_v[i, :, j, :] = cache_v[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
    return out_cache_k, out_cache_v

def block_cache_to_naive_cache__(
    cache_k, cache_v, bsz, block_tables, max_cache_seq_len
):
    _, num_head, blocksize, dim_head = cache_k.shape
    out_cache_k = paddle.zeros(
        shape=[bsz, max_cache_seq_len + 1, num_head, dim_head], dtype=cache_k.dtype
    )
    out_cache_v = paddle.zeros(
        shape=[bsz, max_cache_seq_len + 1, num_head, dim_head], dtype=cache_v.dtype
    )
    for i in range(bsz):
        for j in range(max_cache_seq_len):
            out_cache_k[i, j, :, :] = cache_k[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
            out_cache_v[i, j, :, :] = cache_v[
                block_tables[i, j // blocksize], :, j % blocksize, :
            ]
    return out_cache_k, out_cache_v

def update_encoder_kv_cache(k, v, seq_lens_encoder, cache_k, cache_v, block_tables):
    _, num_head, blocksize, dim_head = cache_k.shape
    offset = 0
    for batch_idx, seq_len in enumerate(seq_lens_encoder.numpy()):
        if seq_len == 0:
            continue
        for seq_idx in range(seq_len):
            block_id = block_tables[batch_idx, seq_idx // blocksize]
            assert block_id != -1
            index = offset + seq_idx
            cache_k[block_id, :, seq_idx % blocksize, :] = k[index, :, :]
            cache_v[block_id, :, seq_idx % blocksize, :] = v[index, :, :]

        offset += seq_len

def update_decoder_kv_cache(k, v, seq_lens_decoder, cache_k, cache_v, block_tables):
    _, num_head, blocksize, dim_head = cache_k.shape
    for batch_idx, seq_idx in enumerate(seq_lens_decoder.numpy()):
        if seq_idx == 0:
            continue
        block_id = block_tables[batch_idx, seq_idx // blocksize]
        assert block_id != -1
        cache_k[block_id, :, seq_idx % blocksize, :] = k[batch_idx, :, :]
        cache_v[block_id, :, seq_idx % blocksize, :] = v[batch_idx, :, :]

def apply_rope(qk, cos, sin):
    rotate_half = paddle.reshape(
        paddle.stack([-qk[..., 1::2], qk[..., 0::2]], axis=-1),
        paddle.shape(qk),
    )
    out = paddle.add(paddle.multiply(qk, cos),
                     paddle.multiply(rotate_half, sin))
    return paddle.cast(out, qk.dtype)

def forward_native_backend(qkv, layer, attn_backend, forward_meta: ForwardMeta):
    bsz = forward_meta.seq_lens_this_time.shape[0]
    num_head_q, num_head_kv, dim = layer.num_heads, layer.kv_num_heads, layer.head_dim

    # 1. 分离 encoder / decoder 的 mask
    seq_lens_encoder = forward_meta.seq_lens_encoder.squeeze(-1)
    seq_lens_decoder = forward_meta.seq_lens_decoder.squeeze(-1)
    seq_lens_this_time = forward_meta.seq_lens_this_time.squeeze(-1)
    encoder_indices = []
    decoder_indices = []

    offset = 0
    for i in range(bsz):
        length = seq_lens_this_time[i].item()
        if seq_lens_encoder[i] > 0:
            encoder_indices.extend(range(offset, offset + length))
        elif seq_lens_decoder[i] > 0:
            decoder_indices.extend(range(offset, offset + length))
        offset += length

    encoder_indices = paddle.to_tensor(encoder_indices, dtype='int32')
    decoder_indices = paddle.to_tensor(decoder_indices, dtype='int32')

    encoder_qkv = paddle.index_select(qkv, encoder_indices, axis=0)
    decoder_qkv = paddle.index_select(qkv, decoder_indices, axis=0)

    # 2. 分解 encoder 和 decoder 的 qkv
    encoder_q, encoder_k, encoder_v = split_qkv(encoder_qkv, num_head_q, num_head_kv, dim)
    decoder_q, decoder_k, decoder_v = split_qkv(decoder_qkv, num_head_q, num_head_kv, dim)
    cache_k = forward_meta.caches[2 * layer.layer_id]
    cache_v = forward_meta.caches[2 * layer.layer_id + 1]

    # 3. Rotary Embedding
    if paddle.device.is_compiled_with_custom_device('metax_gpu'):
        if decoder_q.numel() != 0 or encoder_q.numel() != 0:
            for batch_idx in range(forward_meta.seq_lens_this_time.shape[0]):
                seq_len_i = forward_meta.seq_lens_this_time[batch_idx]
                if seq_len_i == 0:
                    continue
                cached_kv_len = seq_lens_decoder[batch_idx]
                cu_seq_start_q = forward_meta.cu_seqlens_q[batch_idx]
                cu_seq_end_q = forward_meta.cu_seqlens_q[batch_idx + 1]
                if forward_meta.rotary_embs is not None and cu_seq_end_q > cu_seq_start_q:
                    cos = forward_meta.rotary_embs[0, 0,
                                                cached_kv_len:cached_kv_len +
                                                seq_len_i, :, :]
                    sin = forward_meta.rotary_embs[1, 0,
                                                cached_kv_len:cached_kv_len +
                                                seq_len_i, :, :]

                    def rope_func(qk):
                        qk[cu_seq_start_q:cu_seq_end_q] = apply_rope(
                        qk[cu_seq_start_q:cu_seq_end_q], cos, sin)
                    if encoder_q.numel() != 0:
                        rope_func(encoder_q)
                        rope_func(encoder_k)
                    if decoder_q.numel() != 0:
                        rope_func(decoder_q)
                        rope_func(decoder_k)
    else:
        encoder_q, encoder_k = apply_rotary_embedding_neox(encoder_q, encoder_k)
        decoder_q, decoder_k = apply_rotary_embedding_neox(decoder_q, decoder_k)

    # 4. Flash Attention for encoder
    encoder_v = encoder_v
    cu_seqlens_q = forward_meta.cu_seqlens_q
    cu_seqlens_k = forward_meta.cu_seqlens_k
    max_seqlen_q = paddle.max(seq_lens_this_time)
    max_seqlen_k = max_seqlen_q

    if paddle.device.is_compiled_with_custom_device('metax_gpu'):
        if encoder_q.numel() > 0:
            encoder_out = flash_attn_unpadded_func(
                encoder_q, encoder_k, encoder_v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                attn_mask = forward_meta.attn_mask,
                causal = attn_backend.causal
            )
            update_encoder_kv_cache(encoder_k, encoder_v, seq_lens_encoder, cache_k, cache_v, forward_meta.block_tables)
        else:
            encoder_out = None
    else:
        encoder_out = flash_attn_varlen(
            encoder_q, encoder_k, encoder_v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k
        )

    # 5. decoder attention with kv cache
    bs = decoder_q.shape[0]
    decoder_q = decoder_q.reshape([bs, 1, num_head_q, dim])
    decoder_k_ = decoder_k.reshape([bs, 1, num_head_kv, dim])
    decoder_v_ = decoder_v.reshape([bs, 1, num_head_kv, dim])
    cache_seqlens = paddle.index_select(forward_meta.seq_lens_decoder, decoder_indices, axis=0)

    # 5.1 convert paged kv cache to continuous cache
    if paddle.device.is_compiled_with_custom_device('metax_gpu'):
            if decoder_q.numel() > 0:
                max_cache_seq_len = paddle.max(cache_seqlens)
                c_cache_k, c_cache_v = block_cache_to_naive_cache__(cache_k, cache_v, bs, forward_meta.block_tables, max_cache_seq_len)
                decoder_out = flash_attn_kvcache_func(
                decoder_q, c_cache_k, c_cache_v,
                cache_seqlens.squeeze(-1),
                None,
                decoder_k_, decoder_v_,
                causal = attn_backend.causal
                )
                update_decoder_kv_cache(decoder_k, decoder_v, seq_lens_decoder, cache_k, cache_v, forward_meta.block_tables)
            else:
                decoder_out = None
    else:
        c_cache_k, c_cache_v = block_cache_to_naive_cache(cache_k, cache_v,  bsz, forward_meta.block_tables, cache_seqlens)
        decoder_out = flash_attn_with_kvcache(
            decoder_q, c_cache_k, c_cache_v,
            cache_seqlens,
            # forward_meta.block_tables
        )

    # 6. 拼接 encoder_out 和 decoder_out
    total_len = qkv.shape[0]
    out = paddle.zeros([total_len, num_head_q, dim])
    if paddle.device.is_compiled_with_custom_device('metax_gpu'):
        if encoder_out is not None:
            out = paddle.tensor.put_along_axis(out, encoder_indices.unsqueeze(-1).unsqueeze(-1), encoder_out[0], axis=0)
        if decoder_out is not None:
            new_decoder_out = decoder_out[0].squeeze(1)
            out = paddle.tensor.put_along_axis(out, decoder_indices.unsqueeze(-1).unsqueeze(-1), new_decoder_out, axis=0)

        out.reshape_([total_len, num_head_q * dim])
    else:
        out = paddle.tensor.put_along_axis(out, decoder_indices.unsqueeze(-1), encoder_out, axis=0)
        out = paddle.tensor.put_along_axis(out, decoder_indices.unsqueeze(-1), decoder_out, axis=0)

    return out