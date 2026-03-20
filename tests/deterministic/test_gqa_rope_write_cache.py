"""
Tests for gqa_rope_write_cache CUDA custom op.

Covers:
  - RoPE correctness with MHA/GQA/MQA head configurations (parametrized)
  - RoPE position offset with prefix
  - KV cache write correctness
  - Cache miss vs cache hit RoPE consistency
  - Non-contiguous block_table
  - Multi-batch (bs>1) with varying prefix/extend lengths

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_gqa_rope_write_cache.py -v -s

Note: The CUDA kernel data_t is hardcoded to bfloat16. All qkv/cache tensors must be bfloat16.
      rotary_embs remains float32 as the kernel reads it via data<float>().
"""

import numpy as np
import paddle
import pytest

COMPUTE_DTYPE = "bfloat16"
ATOL = 2e-2  # bfloat16 tolerance

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def ref_neox_rope(x, cos, sin):
    """Reference neox-style RoPE: split halves."""
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


def make_rotary_embs(max_seq_len, head_dim, base=10000.0):
    """Build rotary_embs in FastDeploy format: [2, 1, max_seq_len, 1, head_dim]."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (paddle.arange(0, half_dim, dtype="float32") / half_dim))
    positions = paddle.arange(max_seq_len, dtype="float32")
    freqs = paddle.outer(positions, inv_freq)
    cos_full = paddle.concat([paddle.cos(freqs), paddle.cos(freqs)], axis=-1)
    sin_full = paddle.concat([paddle.sin(freqs), paddle.sin(freqs)], axis=-1)
    rotary_embs = paddle.stack([cos_full, sin_full], axis=0)
    return rotary_embs.unsqueeze(1).unsqueeze(3)


def ref_apply_rope_to_qkv(qkv, num_heads, kv_num_heads, head_dim, rotary_embs, seq_lens_encoder, seq_lens_decoder, bs):
    """Apply RoPE to Q and K using reference implementation.
    Returns (q_roped, k_roped, v) all as [token_nums, heads, head_dim].
    """
    token_nums = qkv.shape[0]
    total_heads = num_heads + 2 * kv_num_heads
    qkv_3d = qkv.reshape([token_nums, total_heads, head_dim]).astype("float32")
    q_raw = qkv_3d[:, :num_heads, :]
    k_raw = qkv_3d[:, num_heads : num_heads + kv_num_heads, :]
    v_raw = qkv_3d[:, num_heads + kv_num_heads :, :]

    cos_table = rotary_embs[0, 0, :, 0, :]
    sin_table = rotary_embs[1, 0, :, 0, :]

    positions = []
    for b in range(bs):
        enc_len = int(seq_lens_encoder[b])
        dec_len = int(seq_lens_decoder[b])
        if enc_len > 0:
            positions.extend(range(dec_len, dec_len + enc_len))

    positions = paddle.to_tensor(positions, dtype="int64")
    cos = cos_table[positions].unsqueeze(1)
    sin = sin_table[positions].unsqueeze(1)
    return ref_neox_rope(q_raw, cos, sin), ref_neox_rope(k_raw, cos, sin), v_raw


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
    """Wrapper that prepares all metadata and calls gqa_rope_write_cache.
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

    cu_seqlens_q_list = [0]
    running = 0
    for i in range(bs):
        running += int(seq_lens_this_time[i].item())
        cu_seqlens_q_list.append(running)
    cu_seqlens_q = paddle.to_tensor(cu_seqlens_q_list, dtype="int32")

    batch_ids = []
    for i in range(bs):
        stt = int(seq_lens_this_time[i].item())
        batch_ids.extend([i] * stt)
    batch_id_per_token = paddle.to_tensor(batch_ids, dtype="int32")

    max_blocks_total = bs * ((max_seq_len + block_size - 1) // block_size)
    decode_max_tile = max(max_blocks_total, 1)
    decoder_batch_ids = paddle.full([decode_max_tile], 0, dtype="int32")
    decoder_tile_ids = paddle.full([decode_max_tile], 0, dtype="int32")
    decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
    decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
    decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
    max_len_tensor_cpu = paddle.full([9], 0, dtype="int32").cpu()

    group_size = num_heads // kv_num_heads

    # Encoder tiles: C++ kernel memsets bsz * div_up(max_enc_dec_len * group_size, encoder_block_shape_q)
    # elements, so we must allocate at least that many.
    encoder_block_shape_q = 64
    max_enc_dec_len = max(int(seq_lens_encoder[i].item()) + int(seq_lens_decoder[i].item()) for i in range(bs))
    max_encoder_tiles = max(
        bs * ((max_enc_dec_len * group_size + encoder_block_shape_q - 1) // encoder_block_shape_q), 1
    )
    encoder_batch_ids = paddle.full([max_encoder_tiles], 0, dtype="int32")
    encoder_tile_ids = paddle.full([max_encoder_tiles], 0, dtype="int32")
    encoder_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()

    kv_batch_ids = paddle.full([max(max_blocks_total, 1)], 0, dtype="int32")
    kv_tile_ids = paddle.full([max(max_blocks_total, 1)], 0, dtype="int32")
    kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()

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

    (cu_seqlens_k, cache_batch_ids, cache_tile_ids, cache_num_blocks, kv_token_num_cpu) = pre_cache_len_concat(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        max_dec_len,
        block_size,
    )

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
        1e-6,
        use_neox_rotary_style,
        "none",
        False,
    )
    return q, k, v, qkv_out


# ---------------------------------------------------------------------------
# Helper: generate common single-sequence test data
# ---------------------------------------------------------------------------


def _make_single_seq_data(
    num_heads, kv_num_heads, head_dim, token_nums, block_size=64, max_seq_len=512, prefix_len=0, seed=42
):
    """Generate test tensors for a single-sequence gqa_rope_write_cache test."""
    rotary_embs = make_rotary_embs(max_seq_len, head_dim)
    paddle.seed(seed)
    total_dim = (num_heads + 2 * kv_num_heads) * head_dim
    qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

    num_blocks = (max_seq_len + block_size - 1) // block_size
    cache_k = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
    cache_v = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
    block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

    seq_lens_encoder = paddle.to_tensor([token_nums], dtype="int32")
    seq_lens_decoder = paddle.to_tensor([prefix_len], dtype="int32")
    seq_lens_this_time = paddle.to_tensor([token_nums], dtype="int32")

    return (qkv, cache_k, cache_v, block_tables, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, rotary_embs)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestGqaRopeWriteCache:
    """Tests for gqa_rope_write_cache CUDA op."""

    # B1/B4/B5/B7: RoPE correctness across head configurations (prefix=0)
    @pytest.mark.parametrize(
        "num_heads,kv_num_heads,head_dim,token_nums,max_seq_len",
        [
            (8, 8, 128, 32, 512),  # B1: basic MHA
            (32, 8, 128, 16, 512),  # B4: GQA (group_size=4)
            (32, 1, 128, 8, 512),  # B5: MQA (group_size=32)
            (8, 4, 128, 16, 256),  # B7: mixed head ratio
        ],
        ids=["MHA-8h", "GQA-32q8kv", "MQA-32q1kv", "Mixed-8q4kv"],
    )
    def test_rope_basic_correctness(self, num_heads, kv_num_heads, head_dim, token_nums, max_seq_len):
        """Verify Q/K RoPE output matches reference (prefix=0, single batch)."""
        block_size = 64
        (qkv, cache_k, cache_v, block_tables, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, rotary_embs) = (
            _make_single_seq_data(num_heads, kv_num_heads, head_dim, token_nums, block_size, max_seq_len)
        )

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
            1,
        )

        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        k_diff = float(paddle.max(paddle.abs(k.astype("float32") - k_ref)).item())
        assert q_diff < ATOL, f"Q RoPE mismatch: {q_diff}"
        assert k_diff < ATOL, f"K RoPE mismatch: {k_diff}"

    # B2: RoPE position offset (prefix>0)
    def test_rope_with_prefix(self):
        """Verify RoPE positions are offset by prefix_len for both Q and K."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size, max_seq_len = 64, 512
        extend_len, prefix_len = 16, 384

        (qkv, cache_k, cache_v, block_tables, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, rotary_embs) = (
            _make_single_seq_data(
                num_heads, kv_num_heads, head_dim, extend_len, block_size, max_seq_len, prefix_len, seed=123
            )
        )

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
            1,
        )

        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
        assert q_diff < ATOL, f"Q RoPE with prefix mismatch: {q_diff}"

        # Verify K via cache at positions [prefix, prefix+extend)
        for t in range(extend_len):
            pos = prefix_len + t
            cached_k = cache_k[pos // block_size, :, pos % block_size, :]
            k_diff = float(paddle.max(paddle.abs(cached_k.astype("float32") - k_ref[t])).item())
            assert k_diff < ATOL, f"K cache mismatch at extend token {t}: {k_diff}"

    # B3: KV Cache write correctness
    def test_kv_cache_write(self):
        """Verify K and V are correctly written to paged cache."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size, max_seq_len = 64, 512
        token_nums = 20

        (qkv, cache_k, cache_v, block_tables, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, rotary_embs) = (
            _make_single_seq_data(num_heads, kv_num_heads, head_dim, token_nums, block_size, max_seq_len)
        )

        _, k_out, v_out, _ = call_gqa_rope_write_cache(
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

        for t in range(token_nums):
            bid, off = t // block_size, t % block_size
            k_diff = float(
                paddle.max(paddle.abs(cache_k[bid, :, off, :].astype("float32") - k_out[t].astype("float32"))).item()
            )
            v_diff = float(
                paddle.max(paddle.abs(cache_v[bid, :, off, :].astype("float32") - v_out[t].astype("float32"))).item()
            )
            assert k_diff < 1e-3, f"K cache mismatch at token {t}: {k_diff}"
            assert v_diff < 1e-3, f"V cache mismatch at token {t}: {v_diff}"

    # B6: Cache miss vs cache hit RoPE consistency
    def test_rope_position_consistency(self):
        """Same QKV at same position -> same Q RoPE and cache_k,
        regardless of cache miss (all tokens) vs cache hit (extend only + prefix).
        Reproduces: total_len=825, prefix_len=768, extend_len=57, block_size=64.
        """
        num_heads, kv_num_heads, head_dim = 28, 4, 128
        block_size, max_seq_len = 64, 4096
        total_len, prefix_len = 825, 768
        extend_len = total_len - prefix_len  # 57

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)
        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv_all = paddle.randn([total_len, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks = (max_seq_len + block_size - 1) // block_size
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)

        # Case A: cache miss — all 825 tokens, prefix=0
        cache_k_a = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_a = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        q_a, _, _, _ = call_gqa_rope_write_cache(
            qkv_all,
            cache_k_a,
            cache_v_a,
            block_tables,
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

        # Case B: cache hit — last 57 tokens only, prefix=768
        cache_k_b = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_b = paddle.zeros([num_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        q_b, _, _, _ = call_gqa_rope_write_cache(
            qkv_all[prefix_len:],
            cache_k_b,
            cache_v_b,
            block_tables,
            paddle.to_tensor([extend_len], dtype="int32"),
            paddle.to_tensor([prefix_len], dtype="int32"),
            paddle.to_tensor([extend_len], dtype="int32"),
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # Q comparison: last 57 tokens should match
        q_diff = float(paddle.max(paddle.abs(q_a[prefix_len:].astype("float32") - q_b.astype("float32"))).item())
        assert q_diff < 1e-6, f"Q RoPE position consistency FAILED: {q_diff}"

        # cache_k comparison: blocks covering positions 768-824
        blk_start = prefix_len // block_size
        for blk_idx in range(blk_start, (total_len + block_size - 1) // block_size):
            pos_start = blk_idx * block_size
            pos_end = min(pos_start + block_size, total_len)
            if pos_start >= prefix_len:
                offset_end = pos_end - pos_start
                a_slice = cache_k_a[blk_idx, :, :offset_end, :]
                b_slice = cache_k_b[blk_idx, :, :offset_end, :]
                k_diff = float(paddle.max(paddle.abs(a_slice.astype("float32") - b_slice.astype("float32"))).item())
                assert k_diff == 0.0, (
                    f"cache_k block {blk_idx} MISMATCH: max_diff={k_diff:.6e}. "
                    f"cache miss vs cache hit wrote different K values!"
                )

    # B9: Non-contiguous block_table
    def test_non_contiguous_blocks(self):
        """Test with shuffled (non-contiguous) physical block IDs."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size, max_seq_len = 64, 512
        token_nums = 128

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)
        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        num_blocks_needed = (token_nums + block_size - 1) // block_size
        total_blocks = num_blocks_needed * 3  # allocate 3x blocks
        cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

        # Shuffle block assignment
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

        _, k, _, _ = call_gqa_rope_write_cache(
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

        for t in range(token_nums):
            physical_block = int(block_tables[0, t // block_size].item())
            cached_k = cache_k[physical_block, :, t % block_size, :]
            k_diff = float(paddle.max(paddle.abs(cached_k.astype("float32") - k[t].astype("float32"))).item())
            assert k_diff < 1e-3, f"Non-contiguous K cache mismatch at token {t}: {k_diff}"

    # B10: Multi-batch with different prefix lengths
    def test_multi_batch(self):
        """Test bs>1 with varying prefix/extend lengths per sequence."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size, max_seq_len = 64, 512
        bs = 3
        extend_lens = [32, 16, 8]
        prefix_lens = [0, 128, 64]
        token_nums = sum(extend_lens)

        rotary_embs = make_rotary_embs(max_seq_len, head_dim)
        paddle.seed(42)
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim
        qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        total_blocks = bs * max_blocks_per_seq
        cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

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

        q_diff = float(paddle.max(paddle.abs(q.astype("float32") - q_ref)).item())
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
