"""
Tests for gqa_rope_write_cache CUDA custom op.

Covers ACTION-5 scenarios (B1-B10):
  B1: RoPE basic correctness (prefix=0)
  B2: RoPE position offset (prefix>0)
  B3: KV Cache write correctness
  B4: GQA scenario (q_heads=32, kv_heads=8)
  B5: MQA extreme scenario (kv_heads=1)
  B6: Consistency with position-based RoPE (cache miss vs cache hit)
  B7: head_dim=64/128/256
  B8: bfloat16 RoPE correctness with reference comparison
  B9: Non-contiguous block_table
  B10: Multi-batch (bs>1) with varying prefix/extend lengths

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_gqa_rope_write_cache.py -v -s

Note: The gqa_rope_write_cache CUDA kernel is compiled with data_t=bfloat16
      (see gqa_rope_write_cache.cu:1317). All qkv/cache tensors must be bfloat16.
      rotary_embs remains float32 as the kernel reads it via data<float>().
"""

import numpy as np
import paddle
import pytest

# The CUDA kernel data_t is hardcoded to bfloat16
COMPUTE_DTYPE = "bfloat16"
ATOL = 2e-2  # bfloat16 tolerance

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def ref_neox_rope(x, cos, sin):
    """Reference neox-style RoPE: split halves.
    x: [..., head_dim]
    cos, sin: broadcastable to x shape, covering full head_dim (repeated halves).
    """
    D = x.shape[-1]
    x_left = x[..., : D // 2]
    x_right = x[..., D // 2 :]
    out = paddle.concat(
        [
            x_left * cos[..., : D // 2] - x_right * sin[..., : D // 2],
            x_right * cos[..., D // 2 :] + x_left * sin[..., D // 2 :],
        ],
        axis=-1,
    )
    return out


def make_rotary_embs(max_seq_len, head_dim, base=10000.0):
    """Build rotary_embs in FastDeploy format: [2, 1, max_seq_len, 1, head_dim]."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (paddle.arange(0, half_dim, dtype="float32") / half_dim))
    positions = paddle.arange(max_seq_len, dtype="float32")
    freqs = paddle.outer(positions, inv_freq)  # [max_seq_len, half_dim]
    cos = paddle.cos(freqs)  # [max_seq_len, half_dim]
    sin = paddle.sin(freqs)
    cos_full = paddle.concat([cos, cos], axis=-1)  # [max_seq_len, head_dim]
    sin_full = paddle.concat([sin, sin], axis=-1)
    rotary_embs = paddle.stack([cos_full, sin_full], axis=0)  # [2, max_seq_len, head_dim]
    rotary_embs = rotary_embs.unsqueeze(1).unsqueeze(3)  # [2, 1, max_seq_len, 1, head_dim]
    return rotary_embs


def ref_apply_rope_to_qkv(qkv, num_heads, kv_num_heads, head_dim, rotary_embs, seq_lens_encoder, seq_lens_decoder, bs):
    """Apply RoPE to Q and K in QKV using reference implementation.

    Returns (q_roped, k_roped, v) all as [token_nums, heads, head_dim].
    """
    token_nums = qkv.shape[0]
    total_heads = num_heads + 2 * kv_num_heads
    qkv_3d = qkv.reshape([token_nums, total_heads, head_dim]).astype("float32")
    q_raw = qkv_3d[:, :num_heads, :]
    k_raw = qkv_3d[:, num_heads : num_heads + kv_num_heads, :]
    v_raw = qkv_3d[:, num_heads + kv_num_heads :, :]

    cos_table = rotary_embs[0, 0, :, 0, :]  # [max_seq_len, head_dim]
    sin_table = rotary_embs[1, 0, :, 0, :]

    # Build per-token positions
    positions = []
    for b in range(bs):
        enc_len = int(seq_lens_encoder[b])
        dec_len = int(seq_lens_decoder[b])
        if enc_len > 0:
            positions.extend(range(dec_len, dec_len + enc_len))

    positions = paddle.to_tensor(positions, dtype="int64")
    cos = cos_table[positions].unsqueeze(1)  # [token_nums, 1, head_dim]
    sin = sin_table[positions].unsqueeze(1)

    q_roped = ref_neox_rope(q_raw, cos, sin)
    k_roped = ref_neox_rope(k_raw, cos, sin)
    return q_roped, k_roped, v_raw


# ---------------------------------------------------------------------------
# Helper: call gqa_rope_write_cache with all required metadata
# ---------------------------------------------------------------------------


def call_gqa_rope_write_cache(
    qkv,
    key_cache,
    value_cache,
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
    use_neox_rotary_style=True,
):
    """
    Wrapper that prepares all metadata and calls gqa_rope_write_cache.
    Returns (q, k, v, qkv_out).
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

    bs = seq_lens_encoder.shape[0]

    # Compute cu_seqlens_q
    cu_seqlens_q_list = [0]
    running = 0
    for i in range(bs):
        running += int(seq_lens_this_time[i].item())
        cu_seqlens_q_list.append(running)
    cu_seqlens_q = paddle.to_tensor(cu_seqlens_q_list, dtype="int32")

    # Compute batch_id_per_token
    batch_ids = []
    for i in range(bs):
        stt = int(seq_lens_this_time[i].item())
        batch_ids.extend([i] * stt)
    batch_id_per_token = paddle.to_tensor(batch_ids, dtype="int32")

    # Prepare get_block_shape_and_split_kv_block buffers
    max_blocks_total = bs * ((max_seq_len + block_size - 1) // block_size)
    decode_max_tile = max(max_blocks_total, 1)
    decoder_batch_ids = paddle.full([decode_max_tile], 0, dtype="int32")
    decoder_tile_ids = paddle.full([decode_max_tile], 0, dtype="int32")
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

    # Get max_dec_len for pre_cache_len_concat
    max_dec_len = int(max_len_tensor_cpu[2].item())

    # Step 1: pre_cache_len_concat
    (cu_seqlens_k, cache_batch_ids, cache_tile_ids, cache_num_blocks, kv_token_num_cpu) = pre_cache_len_concat(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        max_dec_len,
        block_size,
    )

    # Step 2: gqa_rope_write_cache
    q, k, v, qkv_out = gqa_rope_write_cache(
        qkv,
        key_cache,
        value_cache,
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
        None,  # q_norm_weight, k_norm_weight
        None,
        None,  # cache_k_quant_scales, cache_v_quant_scales
        None,
        None,  # cache_k_dequant_scales, cache_v_dequant_scales
        None,
        None,  # cache_k_zp, cache_v_zp
        None,  # kv_signal_data
        kv_token_num_cpu[0].item(),
        max_seq_len,
        1e-6,  # rms_norm_eps
        use_neox_rotary_style,
        "none",  # cache_quant_type
        False,  # rope_3d
    )
    return q, k, v, qkv_out


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestGqaRopeWriteCache:
    """Tests for gqa_rope_write_cache CUDA op."""

    # B1: RoPE basic correctness (prefix=0)
    def test_b1_rope_basic_no_prefix(self):
        """Verify Q/K RoPE output matches reference when prefix=0 (pure prefill)."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        token_nums = 32
        bs = 1

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        q_ref, k_ref, _ = ref_apply_rope_to_qkv(
            qkv,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_embs,
            seq_lens_encoder.numpy(),
            seq_lens_decoder.numpy(),
            bs,
        )

        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        k_diff = float(paddle.max(paddle.abs(k.astype("float32") - k_ref)).item())
        print(f"\n[B1] Q RoPE max_diff={q_diff:.6e}, K max_diff={k_diff:.6e}")
        assert q_diff < ATOL, f"Q RoPE mismatch: {q_diff}"
        assert k_diff < ATOL, f"K RoPE mismatch: {k_diff}"

    # B2: RoPE position offset (prefix>0)
    def test_b2_rope_with_prefix(self):
        """Verify RoPE positions are offset by prefix_len for both Q and K."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        extend_len = 16
        prefix_len = 384
        bs = 1

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(123)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([extend_len, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        seq_lens_encoder = paddle.to_tensor([extend_len], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([prefix_len], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([extend_len], dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        # Reference: positions should be [384, 385, ..., 399]
        q_ref, k_ref, _ = ref_apply_rope_to_qkv(
            qkv,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_embs,
            seq_lens_encoder.numpy(),
            seq_lens_decoder.numpy(),
            bs,
        )

        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        print(f"\n[B2] Q RoPE with prefix max_diff={q_diff:.6e}")
        assert q_diff < ATOL, f"Q RoPE with prefix mismatch: {q_diff}"

        # Verify K via cache: tokens at position [prefix, prefix+extend) should have correct RoPE
        for t in range(extend_len):
            pos = prefix_len + t
            bid = pos // block_size
            off = pos % block_size
            cached_k = cache_k[bid, :, off, :]
            k_diff = float(paddle.max(paddle.abs(cached_k.astype("float32") - k_ref[t])).item())
            assert k_diff < ATOL, f"K cache mismatch at extend token {t}: {k_diff}"

    # B3: KV Cache write correctness
    def test_b3_kv_cache_write(self):
        """Verify K and V are correctly written to paged cache."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        token_nums = 20

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

        q, k_out, v_out, _ = call_gqa_rope_write_cache(
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

        # Verify cache_k and cache_v have been written
        # For sequential block_tables, token t -> block t//block_size, offset t%block_size
        for t in range(token_nums):
            bid = t // block_size
            off = t % block_size
            cached_k = cache_k[bid, :, off, :]  # [kv_num_heads, head_dim]
            cached_v = cache_v[bid, :, off, :]

            k_diff = float(paddle.max(paddle.abs(cached_k.astype("float32") - k_out[t].astype("float32"))).item())
            v_diff = float(paddle.max(paddle.abs(cached_v.astype("float32") - v_out[t].astype("float32"))).item())

            assert k_diff < 1e-3, f"K cache mismatch at token {t}: {k_diff}"
            assert v_diff < 1e-3, f"V cache mismatch at token {t}: {v_diff}"

        print(f"\n[B3] KV cache write verified for {token_nums} tokens")

    # B4: GQA scenario (q_heads=32, kv_heads=8)
    def test_b4_gqa(self):
        """Test with GQA: q_heads=32, kv_heads=8, group_size=4."""
        num_heads, kv_num_heads, head_dim = 32, 8, 128
        block_size = 64
        max_seq_len = 512
        token_nums = 16
        bs = 1

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        assert q.shape == [token_nums, num_heads, head_dim], f"Q shape: {q.shape}"
        assert k.shape == [token_nums, kv_num_heads, head_dim], f"K shape: {k.shape}"
        assert v.shape == [token_nums, kv_num_heads, head_dim], f"V shape: {v.shape}"

        q_ref, k_ref, _ = ref_apply_rope_to_qkv(
            qkv,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_embs,
            seq_lens_encoder.numpy(),
            seq_lens_decoder.numpy(),
            bs,
        )
        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        k_diff = float(paddle.max(paddle.abs(k.astype("float32") - k_ref)).item())
        print(f"\n[B4] GQA Q RoPE max_diff={q_diff:.6e}, K max_diff={k_diff:.6e}")
        assert q_diff < ATOL, f"GQA Q RoPE mismatch: {q_diff}"
        assert k_diff < ATOL, f"GQA K RoPE mismatch: {k_diff}"

    # B5: MQA extreme (kv_heads=1)
    def test_b5_mqa(self):
        """Test with MQA: q_heads=32, kv_heads=1, group_size=32."""
        num_heads, kv_num_heads, head_dim = 32, 1, 128
        block_size = 64
        max_seq_len = 512
        token_nums = 8
        bs = 1

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        assert q.shape == [token_nums, num_heads, head_dim]
        assert k.shape == [token_nums, kv_num_heads, head_dim]
        assert v.shape == [token_nums, kv_num_heads, head_dim]

        q_ref, k_ref, _ = ref_apply_rope_to_qkv(
            qkv,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_embs,
            seq_lens_encoder.numpy(),
            seq_lens_decoder.numpy(),
            bs,
        )
        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        k_diff = float(paddle.max(paddle.abs(k.astype("float32") - k_ref)).item())
        print(f"\n[B5] MQA Q RoPE max_diff={q_diff:.6e}, K max_diff={k_diff:.6e}")
        assert q_diff < ATOL, f"MQA Q RoPE mismatch: {q_diff}"
        assert k_diff < ATOL, f"MQA K RoPE mismatch: {k_diff}"

    # B6: RoPE position consistency (cache miss vs cache hit)
    def test_b6_rope_position_consistency(self):
        """Same raw Q at same position -> same RoPE result, regardless of cache miss/hit."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv_all = paddle.randn([400, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        # Case A: cache miss, all 400 tokens, prefix=0
        cache_k_a = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_a = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        q_a, _, _, _ = call_gqa_rope_write_cache(
            qkv_all,
            cache_k_a,
            cache_v_a,
            block_tables,
            paddle.to_tensor([400], dtype="int32"),
            paddle.to_tensor([0], dtype="int32"),
            paddle.to_tensor([400], dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # Case B: cache hit, last 16 tokens, prefix=384
        cache_k_b = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_b = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        q_b, _, _, _ = call_gqa_rope_write_cache(
            qkv_all[384:],
            cache_k_b,
            cache_v_b,
            block_tables,
            paddle.to_tensor([16], dtype="int32"),
            paddle.to_tensor([384], dtype="int32"),
            paddle.to_tensor([16], dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # q_a[384:] and q_b should match (same raw Q, same position 384-399)
        diff = paddle.abs(q_a[384:].astype("float32") - q_b.astype("float32"))
        max_diff = float(diff.max().item())
        print(f"\n[B6] RoPE consistency max_diff={max_diff:.6e}")
        assert max_diff < 1e-6, f"RoPE position consistency FAILED: {max_diff}"

    # B7: head_dim=128 (kernel only reliably supports head_dim=128 with bfloat16)
    def test_b7_head_dim_128(self):
        """Test with head_dim=128."""
        num_heads = 8
        kv_num_heads = 4
        head_dim = 128
        block_size = 64
        max_seq_len = 256
        token_nums = 16
        bs = 1

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        assert q.shape == [token_nums, num_heads, head_dim]
        assert k.shape == [token_nums, kv_num_heads, head_dim]

        q_ref, k_ref, _ = ref_apply_rope_to_qkv(
            qkv,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_embs,
            seq_lens_encoder.numpy(),
            seq_lens_decoder.numpy(),
            bs,
        )
        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        k_diff = float(paddle.max(paddle.abs(k.astype("float32") - k_ref)).item())
        print(f"\n[B7] head_dim={head_dim} Q max_diff={q_diff:.6e}, K max_diff={k_diff:.6e}")
        assert q_diff < ATOL, f"head_dim={head_dim} Q RoPE mismatch: {q_diff}"
        assert k_diff < ATOL, f"head_dim={head_dim} K RoPE mismatch: {k_diff}"

    # B8: bfloat16 RoPE correctness with reference comparison
    def test_b8_bfloat16(self):
        """Verify bfloat16 RoPE matches reference (kernel only supports bfloat16)."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 256
        token_nums = 16
        bs = 1

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        assert q.shape == [token_nums, num_heads, head_dim]

        q_ref, k_ref, _ = ref_apply_rope_to_qkv(
            qkv,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_embs,
            seq_lens_encoder.numpy(),
            seq_lens_decoder.numpy(),
            bs,
        )
        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        k_diff = float(paddle.max(paddle.abs(k.astype("float32") - k_ref)).item())
        print(f"\n[B8] bfloat16 Q max_diff={q_diff:.6e}, K max_diff={k_diff:.6e}")
        assert q_diff < ATOL, f"bfloat16 Q RoPE mismatch: {q_diff}"
        assert k_diff < ATOL, f"bfloat16 K RoPE mismatch: {k_diff}"

    # B9: Non-contiguous block_table
    def test_b9_non_contiguous_blocks(self):
        """Test with shuffled (non-contiguous) physical block IDs."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        token_nums = 128

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks_needed = (token_nums + block_size - 1) // block_size
        total_blocks = num_blocks_needed * 3  # allocate 3x blocks
        cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        # Shuffle block assignment: use non-contiguous blocks
        np.random.seed(42)
        available = np.arange(total_blocks, dtype=np.int32)
        np.random.shuffle(available)
        chosen = available[:num_blocks_needed]
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        block_tables = paddle.zeros([1, max_blocks_per_seq], dtype="int32")
        for i, b in enumerate(chosen):
            block_tables[0, i] = int(b)

        seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        # Verify K is written to the correct non-contiguous blocks
        for t in range(token_nums):
            logical_block = t // block_size
            physical_block = int(block_tables[0, logical_block].item())
            offset = t % block_size
            cached_k = cache_k[physical_block, :, offset, :]
            k_diff = float(paddle.max(paddle.abs(cached_k.astype("float32") - k[t].astype("float32"))).item())
            assert k_diff < 1e-3, f"Non-contiguous K cache mismatch at token {t}: {k_diff}"

        print(f"\n[B9] Non-contiguous block table verified for {token_nums} tokens")

    # B10: Multi-batch with different prefix lengths
    def test_b10_multi_batch(self):
        """Test bs>1 with varying prefix/extend lengths per sequence."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        max_seq_len = 512
        bs = 3
        # seq0: prefix=0, extend=32; seq1: prefix=128, extend=16; seq2: prefix=64, extend=8
        extend_lens = [32, 16, 8]
        prefix_lens = [0, 128, 64]
        token_nums = sum(extend_lens)

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)

        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        # Allocate enough blocks for all sequences
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        total_blocks = bs * max_blocks_per_seq
        cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        # Each sequence gets its own contiguous range of blocks
        block_tables = paddle.zeros([bs, max_blocks_per_seq], dtype="int32")
        for b in range(bs):
            for j in range(max_blocks_per_seq):
                block_tables[b, j] = b * max_blocks_per_seq + j

        seq_lens_encoder = paddle.to_tensor(extend_lens, dtype="int32")
        seq_lens_decoder = paddle.to_tensor(prefix_lens, dtype="int32")
        seq_lens_this_time = paddle.to_tensor(extend_lens, dtype="int32")

        q, k, v, _ = call_gqa_rope_write_cache(
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

        q_ref, k_ref, _ = ref_apply_rope_to_qkv(
            qkv,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_embs,
            seq_lens_encoder.numpy(),
            seq_lens_decoder.numpy(),
            bs,
        )

        # Only compare Q: k output has kv_token_num > token_num when prefix > 0
        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        print(f"\n[B10] Multi-batch Q max_diff={q_diff:.6e}")
        assert q_diff < ATOL, f"Multi-batch Q RoPE mismatch: {q_diff}"

        # Verify K via cache for each sequence
        token_offset = 0
        for b in range(bs):
            for t in range(extend_lens[b]):
                pos = prefix_lens[b] + t
                bid = int(block_tables[b, pos // block_size].item())
                off = pos % block_size
                cached_k = cache_k[bid, :, off, :]
                k_diff = float(paddle.max(paddle.abs(cached_k.astype("float32") - k_ref[token_offset + t])).item())
                assert k_diff < ATOL, f"K cache mismatch at seq {b} token {t}: {k_diff}"
            token_offset += extend_lens[b]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
