"""
Integration tests for the deterministic attention pipeline (ACTION-8).

Tests the full 3-step pipeline as a whole:
  Step 1: pre_cache_len_concat
  Step 2: gqa_rope_write_cache (RoPE + KV cache write)
  Step 3: _deterministic_build_triton_indices + extend_attention_fwd_unified

Scenarios:
  E1: Mixed batch end-to-end (prefill + decode through unified pipeline)
  E2: Decode via unified kernel correctness (extend_len=1)
  E3: RoPE position consistency (cache miss vs cache hit bit-identical)
  E4: All decode batch through unified path
  E5: Multi-sequence different prefix split-invariance (pipeline level)
  E6: Multi-round conversation determinism

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_integration_deterministic.py -v -s
"""

import numpy as np
import paddle
import pytest

# bfloat16 required by gqa_rope_write_cache CUDA kernel
COMPUTE_DTYPE = "bfloat16"
ATOL = 3e-2  # bfloat16 tolerance for full-pipeline comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rotary_embs(max_seq_len, head_dim, base=10000.0):
    """Build rotary_embs: [2, 1, max_seq_len, 1, head_dim]."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (paddle.arange(0, half_dim, dtype="float32") / half_dim))
    positions = paddle.arange(max_seq_len, dtype="float32")
    freqs = paddle.outer(positions, inv_freq)
    cos = paddle.cos(freqs)
    sin = paddle.sin(freqs)
    cos_full = paddle.concat([cos, cos], axis=-1)
    sin_full = paddle.concat([sin, sin], axis=-1)
    rotary_embs = paddle.stack([cos_full, sin_full], axis=0)
    rotary_embs = rotary_embs.unsqueeze(1).unsqueeze(3)
    return rotary_embs


def ref_neox_rope(x, cos, sin):
    """Reference neox-style RoPE."""
    D = x.shape[-1]
    x_left = x[..., : D // 2]
    x_right = x[..., D // 2 :]
    return paddle.concat(
        [
            x_left * cos[..., : D // 2] - x_right * sin[..., : D // 2],
            x_right * cos[..., D // 2 :] + x_left * sin[..., D // 2 :],
        ],
        axis=-1,
    )


def naive_attention_flat(q, k, v, qo_indptr, kv_indptr, prefix_lens, num_q_heads, num_kv_heads, head_dim):
    """
    Reference attention on flat token tensors with CSR indptrs.

    Args:
        q: [total_q_tokens, num_q_heads, head_dim] (float32)
        k, v: lists of [kv_len, num_kv_heads, head_dim] per sequence (float32)
        qo_indptr: [bs+1] CSR indptr for queries
        kv_indptr: not used here, k/v provided per-seq
        prefix_lens: [bs] int
    Returns:
        output: [total_q_tokens, num_q_heads, head_dim]
    """
    bs = len(k)
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / (head_dim**0.5)
    outputs = []

    for b in range(bs):
        q_start = int(qo_indptr[b])
        q_end = int(qo_indptr[b + 1])
        q_len = q_end - q_start
        if q_len == 0:
            continue

        q_b = q[q_start:q_end]  # [q_len, num_q_heads, head_dim]
        k_b = k[b]  # [kv_len, num_kv_heads, head_dim]
        v_b = v[b]
        kv_len = k_b.shape[0]
        plen = int(prefix_lens[b])

        # GQA expansion
        k_exp = k_b.unsqueeze(2).expand([-1, -1, group_size, -1]).reshape([kv_len, num_q_heads, head_dim])
        v_exp = v_b.unsqueeze(2).expand([-1, -1, group_size, -1]).reshape([kv_len, num_q_heads, head_dim])

        # [num_q_heads, q_len, kv_len]
        scores = paddle.einsum("qhd,khd->hqk", q_b, k_exp) * scale

        # Causal mask: prefix always visible, extend uses causal
        for qi in range(q_len):
            for ki in range(kv_len):
                if ki >= plen and qi < (ki - plen):
                    scores[:, qi, ki] = float("-inf")

        attn = paddle.nn.functional.softmax(scores, axis=-1)
        out_b = paddle.einsum("hqk,khd->qhd", attn, v_exp)
        outputs.append(out_b)

    return paddle.concat(outputs, axis=0)


def run_full_pipeline(
    qkv_list,
    cache_k,
    cache_v,
    block_tables,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    rotary_embs,
    num_heads,
    kv_num_heads,
    head_dim,
    block_size,
    max_seq_len,
):
    """
    Run the full 3-step deterministic pipeline:
      1. pre_cache_len_concat
      2. gqa_rope_write_cache
      3. build_triton_indices + extend_attention_fwd_unified

    Args:
        qkv_list: packed QKV tensor [token_nums, (num_heads+2*kv_num_heads)*head_dim]
        cache_k, cache_v: paged KV cache [num_blocks, kv_num_heads, block_size, head_dim]
        block_tables: [bs, max_blocks_per_seq]
        seq_lens_encoder: [bs] int32
        seq_lens_decoder: [bs] int32
        seq_lens_this_time: [bs] int32
        rotary_embs: [2, 1, max_seq_len, 1, head_dim]
        num_heads, kv_num_heads, head_dim, block_size, max_seq_len: config

    Returns:
        output: [token_nums, num_heads * head_dim]
        q_roped: [token_nums, num_heads, head_dim] (for RoPE verification)
    """
    from fastdeploy.model_executor.layers.attention.ops import (
        get_block_shape_and_split_kv_block,
    )
    from fastdeploy.model_executor.layers.attention.ops.gqa_rope_write_cache import (
        gqa_rope_write_cache,
    )
    from fastdeploy.model_executor.layers.attention.ops.pre_cache_len_concat import (
        pre_cache_len_concat,
    )
    from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
        build_kv_indices_from_block_tables,
        build_unified_kv_indices,
        extend_attention_fwd_unified,
    )

    bs = seq_lens_encoder.shape[0]

    # -- Compute cu_seqlens_q, batch_id_per_token --
    cu_list = [0]
    batch_ids = []
    running = 0
    for i in range(bs):
        stt = int(seq_lens_this_time[i].item())
        running += stt
        cu_list.append(running)
        batch_ids.extend([i] * stt)
    cu_seqlens_q = paddle.to_tensor(cu_list, dtype="int32")
    batch_id_per_token = paddle.to_tensor(batch_ids, dtype="int32")

    # -- get_block_shape_and_split_kv_block (needed for launch metadata) --
    max_blocks_total = bs * ((max_seq_len + block_size - 1) // block_size)
    decoder_batch_ids = paddle.full([max(max_blocks_total, 1)], 0, dtype="int32")
    decoder_tile_ids = paddle.full([max(max_blocks_total, 1)], 0, dtype="int32")
    decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
    decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
    decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
    max_len_tensor_cpu = paddle.full([9], 0, dtype="int32").cpu()
    encoder_batch_ids = paddle.full([max(bs, 1)], 0, dtype="int32")
    encoder_tile_ids = paddle.full([max(bs, 1)], 0, dtype="int32")
    encoder_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
    kv_batch_ids = paddle.full([max(max_blocks_total, 1)], 0, dtype="int32")
    kv_tile_ids = paddle.full([max(max_blocks_total, 1)], 0, dtype="int32")
    kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
    group_size = num_heads // kv_num_heads

    get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        decoder_batch_ids,
        decoder_tile_ids,
        decoder_num_blocks_cpu,
        decoder_num_blocks_device,
        decoder_chunk_size_device,
        max_len_tensor_cpu,
        encoder_batch_ids,
        encoder_tile_ids,
        encoder_num_blocks_x_cpu,
        kv_batch_ids,
        kv_tile_ids,
        kv_num_blocks_x_cpu,
        64,
        12,
        group_size,
        block_size,
    )

    max_dec_len = int(max_len_tensor_cpu[2].item())

    # -- Step 1: pre_cache_len_concat --
    (cu_seqlens_k, cache_batch_ids, cache_tile_ids, cache_num_blocks, kv_token_num_cpu) = pre_cache_len_concat(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        max_dec_len,
        block_size,
    )

    # -- Step 2: gqa_rope_write_cache --
    q_roped, _, _, _ = gqa_rope_write_cache(
        qkv_list,
        cache_k,
        cache_v,
        cu_seqlens_q,
        cu_seqlens_k,
        rotary_embs,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        batch_id_per_token,
        block_tables,
        kv_batch_ids,
        kv_tile_ids,
        kv_num_blocks_x_cpu,
        cache_batch_ids,
        cache_tile_ids,
        cache_num_blocks,
        None,
        None,  # q/k_norm_weight
        None,
        None,  # cache_k/v_quant_scales
        None,
        None,  # cache_k/v_dequant_scales
        None,
        None,  # cache_k/v_zp
        None,  # kv_signal_data
        kv_token_num_cpu[0].item(),
        max_seq_len,
        1e-6,
        True,
        "none",
        False,
    )

    # -- Step 3: build triton indices + unified attention --
    prefix_lens = seq_lens_decoder[:bs].clone().astype("int32")
    # For decode seqs (encoder=0), prefix_lens = seq_lens_decoder (total KV)
    # For prefill seqs (encoder>0), prefix_lens = seq_lens_decoder (cached prefix)
    # This matches gpu_model_runner behavior

    # Filter active sequences (seq_lens_this_time > 0)
    active_bs = int((seq_lens_this_time > 0).sum().item())
    prefix_lens_active = prefix_lens[:active_bs]
    extend_seq_lens = seq_lens_this_time[:active_bs]

    qo_indptr = paddle.concat(
        [
            paddle.zeros([1], dtype="int32"),
            paddle.cumsum(extend_seq_lens).astype("int32"),
        ]
    )

    prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
        block_tables,
        prefix_lens_active,
        block_size,
        active_bs,
    )
    total_seq_lens = prefix_lens_active + extend_seq_lens
    all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(
        block_tables,
        total_seq_lens,
        block_size,
        active_bs,
    )

    extend_start_loc = (
        paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(extend_seq_lens[:-1]).astype("int32"),
            ]
        )
        if active_bs > 1
        else paddle.zeros([1], dtype="int32")
    )

    total_extend_len = int(paddle.sum(extend_seq_lens).item())
    extend_kv_indices = paddle.empty([max(total_extend_len, 1)], dtype="int32")
    for s in range(active_bs):
        plen = int(prefix_lens_active[s].item())
        elen = int(extend_seq_lens[s].item())
        if elen == 0:
            continue
        src_start = int(all_kv_indptr[s].item()) + plen
        dst_start = int(extend_start_loc[s].item())
        extend_kv_indices[dst_start : dst_start + elen] = all_kv_indices[src_start : src_start + elen]

    unified_kv_indptr, unified_kv_indices, _ = build_unified_kv_indices(
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        extend_kv_indices,
        active_bs,
    )

    max_extend_len = int(paddle.max(extend_seq_lens).item())
    token_nums = q_roped.shape[0]
    o = paddle.zeros([token_nums, num_heads, head_dim], dtype=q_roped.dtype)
    res = extend_attention_fwd_unified(
        q_roped,
        o,
        cache_k,
        cache_v,
        qo_indptr,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens_active,
        num_heads,
        kv_num_heads,
        head_dim,
        max_extend_len,
        True,
    ).reshape([-1, num_heads * head_dim])

    return res, q_roped


def allocate_blocks(bs, max_seq_len, block_size, extra_factor=1):
    """Allocate sequential block_tables and empty KV cache."""
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks = bs * max_blocks_per_seq * max(extra_factor, 1)
    block_tables = paddle.zeros([bs, max_blocks_per_seq], dtype="int32")
    for b in range(bs):
        for j in range(max_blocks_per_seq):
            block_tables[b, j] = b * max_blocks_per_seq + j
    return block_tables, total_blocks, max_blocks_per_seq


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestIntegrationDeterministic:

    # ---- E1: Mixed batch end-to-end ----
    def test_e1_mixed_batch_end_to_end(self):
        """
        Mixed batch: 1 prefill seq (prefix=384, extend=128) + 3 decode seqs.
        All go through unified pipeline. Verify output shape and no NaN/Inf.
        """
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 1024
        bs = 4

        # seq0: prefill with prefix cache hit (prefix=384, extend=128)
        # seq1-3: decode (extend=1 each, with different KV lengths)
        extend_lens = [128, 1, 1, 1]
        prefix_lens = [384, 200, 300, 100]  # for decode, this is total KV len
        encoder_lens = [128, 0, 0, 0]  # >0 means prefill

        token_nums = sum(extend_lens)
        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        block_tables, total_blocks, _ = allocate_blocks(bs, max_seq_len, block_size)
        cache_k = paddle.randn([total_blocks, kv_num_heads, block_size, head_dim]).astype(COMPUTE_DTYPE)
        cache_v = paddle.randn([total_blocks, kv_num_heads, block_size, head_dim]).astype(COMPUTE_DTYPE)

        seq_lens_encoder = paddle.to_tensor(encoder_lens, dtype="int32")
        seq_lens_decoder = paddle.to_tensor(prefix_lens, dtype="int32")
        seq_lens_this_time = paddle.to_tensor(extend_lens, dtype="int32")

        res, q_roped = run_full_pipeline(
            qkv,
            cache_k,
            cache_v,
            block_tables,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # Verify output shape
        assert res.shape == [token_nums, num_heads * head_dim], f"Output shape: {res.shape}"
        # Verify no NaN/Inf
        assert not paddle.any(paddle.isnan(res)).item(), "Output contains NaN"
        assert not paddle.any(paddle.isinf(res)).item(), "Output contains Inf"
        # Verify q_roped shape
        assert q_roped.shape == [token_nums, num_heads, head_dim]
        print(
            f"\n[E1] Mixed batch passed: shape={res.shape}, "
            f"max_abs={float(res.astype('float32').abs().max().item()):.4f}"
        )

    # ---- E2: Decode via unified kernel correctness ----
    def test_e2_decode_single_token(self):
        """
        Single decode sequence (extend_len=1) through full pipeline.
        Compare against reference naive attention.
        """
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        bs = 1
        prefix_len = 128  # total cached KV
        extend_len = 1  # decode: 1 new token

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim

        # First, populate cache with prefix_len tokens via a prefill call
        qkv_prefill = paddle.randn([prefix_len, total_dim]).astype(COMPUTE_DTYPE)
        block_tables, total_blocks, _ = allocate_blocks(bs, max_seq_len, block_size)
        cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        # Prefill: write KV to cache via the full pipeline
        run_full_pipeline(
            qkv_prefill,
            cache_k,
            cache_v,
            block_tables,
            paddle.to_tensor([prefix_len], dtype="int32"),
            paddle.to_tensor([0], dtype="int32"),
            paddle.to_tensor([prefix_len], dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # Now decode: 1 new token, prefix_len cached
        qkv_decode = paddle.randn([extend_len, total_dim]).astype(COMPUTE_DTYPE)

        res, q_roped = run_full_pipeline(
            qkv_decode,
            cache_k,
            cache_v,
            block_tables,
            paddle.to_tensor([0], dtype="int32"),  # encoder=0 for decode
            paddle.to_tensor([prefix_len], dtype="int32"),
            paddle.to_tensor([extend_len], dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        assert res.shape == [extend_len, num_heads * head_dim]
        assert not paddle.any(paddle.isnan(res)).item(), "Decode output contains NaN"
        assert not paddle.any(paddle.isinf(res)).item(), "Decode output contains Inf"

        # Build reference: read all KV from cache and do naive attention
        # Collect all K/V from cache for this sequence
        total_kv_len = prefix_len + extend_len
        k_all = paddle.zeros([total_kv_len, kv_num_heads, head_dim], dtype="float32")
        v_all = paddle.zeros([total_kv_len, kv_num_heads, head_dim], dtype="float32")
        for t in range(total_kv_len):
            bid = int(block_tables[0, t // block_size].item())
            off = t % block_size
            k_all[t] = cache_k[bid, :, off, :].astype("float32")
            v_all[t] = cache_v[bid, :, off, :].astype("float32")

        # GQA expand for reference
        group_size = num_heads // kv_num_heads
        k_exp = k_all.unsqueeze(2).expand([-1, -1, group_size, -1]).reshape([total_kv_len, num_heads, head_dim])
        v_exp = v_all.unsqueeze(2).expand([-1, -1, group_size, -1]).reshape([total_kv_len, num_heads, head_dim])

        q_f32 = q_roped.astype("float32").squeeze(0) if q_roped.shape[0] == 1 else q_roped[0].astype("float32")
        # q_f32: [num_heads, head_dim]
        scale = 1.0 / (head_dim**0.5)
        # scores: [num_heads, total_kv_len]
        scores = paddle.einsum("hd,khd->hk", q_f32, k_exp) * scale
        # All KV visible (prefix=prefix_len, decode token is at extend position 0)
        attn = paddle.nn.functional.softmax(scores, axis=-1)
        ref_out = paddle.einsum("hk,khd->hd", attn, v_exp)  # [num_heads, head_dim]

        res_f32 = res.astype("float32").reshape([num_heads, head_dim])
        diff = float(paddle.max(paddle.abs(res_f32 - ref_out)).item())
        print(f"\n[E2] Decode single token max_diff={diff:.6e}")
        assert diff < ATOL, f"Decode correctness failed: max_diff={diff}"

    # ---- E3: RoPE position consistency (cache miss vs cache hit) ----
    def test_e3_rope_position_consistency(self):
        """
        Same raw QKV at same logical position must get same RoPE result,
        regardless of whether it's a cache miss (prefix=0) or hit (prefix>0).
        This tests through the full pipeline.
        """
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        total_len = 400
        split_point = 384

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv_all = paddle.randn([total_len, total_dim]).astype(COMPUTE_DTYPE)

        # -- Case A: cache miss, all 400 tokens as prefill --
        block_tables_a, total_blocks_a, _ = allocate_blocks(1, max_seq_len, block_size)
        cache_k_a = paddle.zeros([total_blocks_a, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_a = paddle.zeros([total_blocks_a, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        _, q_roped_a = run_full_pipeline(
            qkv_all,
            cache_k_a,
            cache_v_a,
            block_tables_a,
            paddle.to_tensor([total_len], dtype="int32"),
            paddle.to_tensor([0], dtype="int32"),
            paddle.to_tensor([total_len], dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # -- Case B: cache hit, last 16 tokens as prefill with prefix=384 --
        extend_len = total_len - split_point
        block_tables_b, total_blocks_b, _ = allocate_blocks(1, max_seq_len, block_size)
        cache_k_b = paddle.zeros([total_blocks_b, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_b = paddle.zeros([total_blocks_b, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        _, q_roped_b = run_full_pipeline(
            qkv_all[split_point:],
            cache_k_b,
            cache_v_b,
            block_tables_b,
            paddle.to_tensor([extend_len], dtype="int32"),
            paddle.to_tensor([split_point], dtype="int32"),
            paddle.to_tensor([extend_len], dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # Compare: q_roped_a[384:] should be bit-identical to q_roped_b
        q_a_tail = q_roped_a[split_point:].astype("float32")
        q_b_all = q_roped_b.astype("float32")

        diff = paddle.abs(q_a_tail - q_b_all)
        max_diff = float(diff.max().item())
        print(f"\n[E3] RoPE position consistency max_diff={max_diff:.6e}")
        assert max_diff < 1e-6, f"RoPE consistency FAILED: max_diff={max_diff}"

    # ---- E4: All decode batch ----
    def test_e4_all_decode_batch(self):
        """
        All sequences are decode (encoder=0). The deterministic path
        should still work via the unified kernel.
        """
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        bs = 4
        kv_lens = [100, 200, 150, 80]

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        token_nums = bs  # each decode seq contributes 1 token
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        block_tables, total_blocks, _ = allocate_blocks(bs, max_seq_len, block_size)
        # Pre-fill cache with random data (simulating previously cached KV)
        cache_k = paddle.randn([total_blocks, kv_num_heads, block_size, head_dim]).astype(COMPUTE_DTYPE)
        cache_v = paddle.randn([total_blocks, kv_num_heads, block_size, head_dim]).astype(COMPUTE_DTYPE)

        seq_lens_encoder = paddle.zeros([bs], dtype="int32")  # all decode
        seq_lens_decoder = paddle.to_tensor(kv_lens, dtype="int32")
        seq_lens_this_time = paddle.ones([bs], dtype="int32")  # each decode = 1 token

        res, q_roped = run_full_pipeline(
            qkv,
            cache_k,
            cache_v,
            block_tables,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        assert res.shape == [bs, num_heads * head_dim]
        assert not paddle.any(paddle.isnan(res)).item(), "All-decode output contains NaN"
        assert not paddle.any(paddle.isinf(res)).item(), "All-decode output contains Inf"

        # Verify each decode seq output is non-trivial (not all zeros)
        for b in range(bs):
            seq_out = res[b].astype("float32")
            assert float(seq_out.abs().max().item()) > 1e-6, f"Seq {b} output is all zeros"

        print(f"\n[E4] All decode batch (bs={bs}) passed")

    # ---- E5: Multi-sequence different prefix split-invariance ----
    def test_e5_split_invariance_pipeline(self):
        """
        Pipeline-level split invariance: for 2 sequences with different prefixes,
        the attention output for the extend part should be the same regardless
        of cache hit/miss.

        Runs the full pipeline twice with different prefix splits for the same
        logical sequences, then compares the extend-part outputs.
        """
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512

        # Sequence 0: 256 total tokens, split at 192 vs 0
        # Sequence 1: 128 total tokens, split at 96 vs 0
        total_lens = [256, 128]
        split_points = [192, 96]
        extend_lens_hit = [total_lens[i] - split_points[i] for i in range(2)]  # [64, 32]

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim

        # Generate full QKV for each sequence
        qkv_full = [paddle.randn([tl, total_dim]).astype(COMPUTE_DTYPE) for tl in total_lens]

        # -- Case A: cache miss (prefix=0, all tokens as extend) --
        qkv_a = paddle.concat(qkv_full, axis=0)
        block_tables_a, total_blocks_a, _ = allocate_blocks(2, max_seq_len, block_size)
        cache_k_a = paddle.zeros([total_blocks_a, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_a = paddle.zeros([total_blocks_a, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        res_a, _ = run_full_pipeline(
            qkv_a,
            cache_k_a,
            cache_v_a,
            block_tables_a,
            paddle.to_tensor(total_lens, dtype="int32"),  # encoder = full length
            paddle.zeros([2], dtype="int32"),  # decoder = 0 (no cache)
            paddle.to_tensor(total_lens, dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )
        res_a_f32 = res_a.astype("float32")

        # Extract extend-part outputs from Case A
        # seq0: tokens [192:256] = indices [192:256] in flat output
        # seq1: tokens [96:128] = indices [256+96 : 256+128] in flat output
        out_a_seq0 = res_a_f32[split_points[0] : total_lens[0]]
        out_a_seq1 = res_a_f32[total_lens[0] + split_points[1] : total_lens[0] + total_lens[1]]

        # -- Case B: cache hit (prefix populated, only extend tokens as input) --
        # First populate caches with prefill for the prefix parts
        block_tables_b, total_blocks_b, _ = allocate_blocks(2, max_seq_len, block_size)
        cache_k_b = paddle.zeros([total_blocks_b, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_b = paddle.zeros([total_blocks_b, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        # Write prefix KV for each sequence via prefill (using run_full_pipeline)
        for seq_idx in range(2):
            plen = split_points[seq_idx]
            if plen == 0:
                continue
            single_bt = block_tables_b[seq_idx : seq_idx + 1]
            run_full_pipeline(
                qkv_full[seq_idx][:plen],
                cache_k_b,
                cache_v_b,
                single_bt,
                paddle.to_tensor([plen], dtype="int32"),
                paddle.to_tensor([0], dtype="int32"),
                paddle.to_tensor([plen], dtype="int32"),
                rotary_embs,
                num_heads,
                kv_num_heads,
                head_dim,
                block_size,
                max_seq_len,
            )

        # Now run with extend-only QKV, prefix from cache
        qkv_b = paddle.concat([qkv_full[i][split_points[i] :] for i in range(2)], axis=0)

        res_b, _ = run_full_pipeline(
            qkv_b,
            cache_k_b,
            cache_v_b,
            block_tables_b,
            paddle.to_tensor(extend_lens_hit, dtype="int32"),  # encoder = extend len
            paddle.to_tensor(split_points, dtype="int32"),  # decoder = prefix len
            paddle.to_tensor(extend_lens_hit, dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )
        res_b_f32 = res_b.astype("float32")

        # Compare
        out_b_seq0 = res_b_f32[: extend_lens_hit[0]]
        out_b_seq1 = res_b_f32[extend_lens_hit[0] :]

        diff_seq0 = float(paddle.max(paddle.abs(out_a_seq0 - out_b_seq0)).item())
        diff_seq1 = float(paddle.max(paddle.abs(out_a_seq1 - out_b_seq1)).item())

        print(f"\n[E5] Split invariance seq0 max_diff={diff_seq0:.6e}, " f"seq1 max_diff={diff_seq1:.6e}")
        assert diff_seq0 < ATOL, f"Split invariance seq0 FAILED: {diff_seq0}"
        assert diff_seq1 < ATOL, f"Split invariance seq1 FAILED: {diff_seq1}"

    # ---- E6: Multi-round conversation determinism ----
    def test_e6_multi_round_determinism(self):
        """
        Simulate a multi-round conversation:
          Round 1: prefill 64 tokens (prefix=0)
          Round 2: decode 1 token (prefix=64)
          Round 3: decode 1 token (prefix=65)

        Run this sequence twice and verify outputs are bit-identical.
        """
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        prefill_len = 64

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim

        results_per_run = []

        for run_id in range(2):
            paddle.seed(42)  # same seed each run
            qkv_prefill = paddle.randn([prefill_len, total_dim]).astype(COMPUTE_DTYPE)
            qkv_decode1 = paddle.randn([1, total_dim]).astype(COMPUTE_DTYPE)
            qkv_decode2 = paddle.randn([1, total_dim]).astype(COMPUTE_DTYPE)

            block_tables, total_blocks, _ = allocate_blocks(1, max_seq_len, block_size)
            cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
            cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

            run_results = []

            # Round 1: prefill
            res1, _ = run_full_pipeline(
                qkv_prefill,
                cache_k,
                cache_v,
                block_tables,
                paddle.to_tensor([prefill_len], dtype="int32"),
                paddle.to_tensor([0], dtype="int32"),
                paddle.to_tensor([prefill_len], dtype="int32"),
                rotary_embs,
                num_heads,
                kv_num_heads,
                head_dim,
                block_size,
                max_seq_len,
            )
            run_results.append(res1.astype("float32").numpy())

            # Round 2: decode (prefix=64, extend=1)
            res2, _ = run_full_pipeline(
                qkv_decode1,
                cache_k,
                cache_v,
                block_tables,
                paddle.to_tensor([0], dtype="int32"),
                paddle.to_tensor([prefill_len], dtype="int32"),
                paddle.to_tensor([1], dtype="int32"),
                rotary_embs,
                num_heads,
                kv_num_heads,
                head_dim,
                block_size,
                max_seq_len,
            )
            run_results.append(res2.astype("float32").numpy())

            # Round 3: decode (prefix=65, extend=1)
            res3, _ = run_full_pipeline(
                qkv_decode2,
                cache_k,
                cache_v,
                block_tables,
                paddle.to_tensor([0], dtype="int32"),
                paddle.to_tensor([prefill_len + 1], dtype="int32"),
                paddle.to_tensor([1], dtype="int32"),
                rotary_embs,
                num_heads,
                kv_num_heads,
                head_dim,
                block_size,
                max_seq_len,
            )
            run_results.append(res3.astype("float32").numpy())

            results_per_run.append(run_results)

        # Compare run 0 vs run 1
        for round_idx in range(3):
            r0 = results_per_run[0][round_idx]
            r1 = results_per_run[1][round_idx]
            assert np.array_equal(r0, r1), (
                f"Round {round_idx + 1} not bit-identical: " f"max_diff={np.max(np.abs(r0 - r1)):.6e}"
            )

        print("\n[E6] Multi-round determinism verified (3 rounds, 2 runs)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
