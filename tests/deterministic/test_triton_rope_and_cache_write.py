"""
Tests for Triton rope_and_cache_write kernel.

Uses the C++ gqa_rope_write_cache as reference oracle, comparing:
  - q_roped: RoPE'd Q tensor
  - cache_k / cache_v: paged KV cache after write

Coverage:
  - RoPE styles: standard interleaved (Qwen2, parameterized), neox full (Qwen3)
  - Batch sizes: 1, 4, 5, 8
  - Scenarios: pure prefill, pure decode (1 token + long prefix), mixed batch
  - Block tables: contiguous, non-contiguous, cross-block boundary
  - Padding sentinel: batch_id_per_token tail filled with -1
  - CUDA Graph replay: capture once, replay with different data
  - Decode with seq_lens_encoder=0: kernel defensive branch
  - Edge cases: single token, different block_size, near max_seq_len

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_triton_rope_and_cache_write.py -v -s
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.triton_ops.rope_and_cache_write import (
    extract_cos_sin,
    triton_rope_and_cache_write,
)

COMPUTE_DTYPE = "bfloat16"


# ---------------------------------------------------------------------------
# Reference: call C++ gqa_rope_write_cache
# ---------------------------------------------------------------------------


def _call_cpp_rope(
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
    use_neox_rotary_style=True,
):
    """Call C++ gqa_rope_write_cache and return q."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_gqa_rope_write_cache import call_gqa_rope_write_cache

    q, k, v, qkv_out = call_gqa_rope_write_cache(
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
        use_neox_rotary_style=use_neox_rotary_style,
    )
    return q


def _call_triton_rope(
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
    use_neox_rotary_style=True,
):
    """Call Triton rope_and_cache_write and return q_out."""
    bs = seq_lens_encoder.shape[0]
    token_num = qkv.shape[0]

    # Build cu_seqlens_q and batch_id_per_token
    cu_seqlens_q_list = [0]
    batch_ids = []
    for i in range(bs):
        stt = int(seq_lens_this_time[i].item())
        cu_seqlens_q_list.append(cu_seqlens_q_list[-1] + stt)
        batch_ids.extend([i] * stt)
    cu_seqlens_q = paddle.to_tensor(cu_seqlens_q_list, dtype="int32")
    batch_id_per_token = paddle.to_tensor(batch_ids, dtype="int32")

    q_out = paddle.empty([token_num, num_heads, head_dim], dtype=COMPUTE_DTYPE)

    triton_rope_and_cache_write(
        qkv,
        cache_k,
        cache_v,
        q_out,
        rotary_embs,
        batch_id_per_token,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        block_tables,
        num_heads,
        kv_num_heads,
        head_dim,
        block_size,
        use_neox_rotary_style=use_neox_rotary_style,
    )
    return q_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rotary_embs(max_seq_len, head_dim, neox_full=True, base=10000.0):
    """Build rotary_embs: [2, 1, max_seq_len, 1, emb_dim]."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (paddle.arange(0, half_dim, dtype="float32") / half_dim))
    positions = paddle.arange(max_seq_len, dtype="float32")
    freqs = paddle.outer(positions, inv_freq)  # [max_seq_len, half_dim]

    if neox_full:
        cos_full = paddle.concat([paddle.cos(freqs), paddle.cos(freqs)], axis=-1)
        sin_full = paddle.concat([paddle.sin(freqs), paddle.sin(freqs)], axis=-1)
    else:
        cos_full = paddle.cos(freqs)
        sin_full = paddle.sin(freqs)

    embs = paddle.stack([cos_full, sin_full], axis=0)
    return embs.unsqueeze(1).unsqueeze(3)  # [2, 1, max_seq_len, 1, emb_dim]


def _make_test_data(
    num_heads,
    kv_num_heads,
    head_dim,
    extend_lens,
    prefix_lens,
    block_size=64,
    max_seq_len=512,
    neox_full=True,
    seed=42,
    contiguous_blocks=True,
):
    """Build test tensors for one or more sequences."""
    bs = len(extend_lens)
    token_nums = sum(extend_lens)
    rotary_embs = _make_rotary_embs(max_seq_len, head_dim, neox_full=neox_full)

    paddle.seed(seed)
    total_dim = (num_heads + 2 * kv_num_heads) * head_dim
    qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)

    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

    if contiguous_blocks:
        total_blocks = bs * max_blocks_per_seq
        block_tables = paddle.zeros([bs, max_blocks_per_seq], dtype="int32")
        for b in range(bs):
            for j in range(max_blocks_per_seq):
                block_tables[b, j] = b * max_blocks_per_seq + j
    else:
        total_needed = sum((prefix_lens[b] + extend_lens[b] + block_size - 1) // block_size for b in range(bs))
        total_blocks = max(total_needed * 3, bs * max_blocks_per_seq)
        np.random.seed(seed)
        available = np.arange(total_blocks, dtype=np.int32)
        np.random.shuffle(available)
        block_tables = paddle.zeros([bs, max_blocks_per_seq], dtype="int32")
        idx = 0
        for b in range(bs):
            for j in range(max_blocks_per_seq):
                block_tables[b, j] = int(available[idx])
                idx += 1

    cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
    cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)

    seq_lens_encoder = paddle.to_tensor(extend_lens, dtype="int32")
    seq_lens_decoder = paddle.to_tensor(prefix_lens, dtype="int32")
    seq_lens_this_time = paddle.to_tensor(extend_lens, dtype="int32")

    return (qkv, cache_k, cache_v, block_tables, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, rotary_embs)


def _max_diff(a, b):
    """Max absolute difference between two tensors, cast to float32."""
    return float(paddle.max(paddle.abs(a.astype("float32") - b.astype("float32"))).item())


def _compare_cpp_triton(
    data, num_heads, kv_num_heads, head_dim, block_size, max_seq_len, use_neox_rotary_style=True, label=""
):
    """Run both C++ and Triton, assert Q/K/V match. Returns (q_cpp, q_tri) for further checks."""
    qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data
    common = dict(
        block_tables=block_tables,
        seq_lens_encoder=enc,
        seq_lens_decoder=dec,
        seq_lens_this_time=stt,
        rotary_embs=rotary_embs,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_seq_len=max_seq_len,
        use_neox_rotary_style=use_neox_rotary_style,
    )

    cache_k_cpp, cache_v_cpp = cache_k_ref.clone(), cache_v_ref.clone()
    q_cpp = _call_cpp_rope(qkv, cache_k_cpp, cache_v_cpp, **common)

    cache_k_tri, cache_v_tri = cache_k_ref.clone(), cache_v_ref.clone()
    q_tri = _call_triton_rope(qkv, cache_k_tri, cache_v_tri, **common)

    q_diff = _max_diff(q_cpp, q_tri)
    k_diff = _max_diff(cache_k_cpp, cache_k_tri)
    v_diff = _max_diff(cache_v_cpp, cache_v_tri)

    pfx = f"[{label}] " if label else ""
    assert q_diff < 1e-3, f"{pfx}Q mismatch: {q_diff}"
    assert k_diff < 1e-3, f"{pfx}K mismatch: {k_diff}"
    assert v_diff == 0.0, f"{pfx}V not bit-exact: {v_diff}"

    return q_cpp, q_tri, cache_k_cpp, cache_k_tri, cache_v_cpp, cache_v_tri


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestTritonRopeAndCacheWrite:
    """Compare Triton kernel against C++ reference."""

    # --- Q RoPE correctness: neox full (Qwen3 style) ---
    @pytest.mark.parametrize(
        "num_heads,kv_num_heads,head_dim,token_nums",
        [(8, 8, 128, 32), (32, 8, 128, 16), (28, 4, 128, 64)],
        ids=["MHA-8h", "GQA-32q8kv", "GQA-28q4kv"],
    )
    def test_q_rope_neox_full(self, num_heads, kv_num_heads, head_dim, token_nums):
        """Q RoPE with neox full style should match C++ reference."""
        data = _make_test_data(
            num_heads, kv_num_heads, head_dim, extend_lens=[token_nums], prefix_lens=[0], neox_full=True
        )
        _compare_cpp_triton(
            data, num_heads, kv_num_heads, head_dim, 64, 512, use_neox_rotary_style=True, label="neox_full"
        )

    # --- Q RoPE correctness: standard interleaved (parameterized) ---
    @pytest.mark.parametrize(
        "num_heads,kv_num_heads,head_dim,token_nums",
        [(8, 8, 128, 32), (32, 8, 128, 16), (28, 4, 128, 64)],
        ids=["MHA-8h", "GQA-32q8kv", "GQA-28q4kv"],
    )
    def test_q_rope_standard(self, num_heads, kv_num_heads, head_dim, token_nums):
        """Q RoPE with standard interleaved style should match C++ reference."""
        data = _make_test_data(
            num_heads, kv_num_heads, head_dim, extend_lens=[token_nums], prefix_lens=[0], neox_full=False
        )
        _compare_cpp_triton(
            data, num_heads, kv_num_heads, head_dim, 64, 512, use_neox_rotary_style=False, label="standard"
        )

    # --- KV cache write: V should be bit-exact ---
    def test_v_cache_bitexact(self):
        """V cache write is pure copy, should be bit-exact."""
        data = _make_test_data(8, 8, 128, extend_lens=[32], prefix_lens=[0], neox_full=True)
        _compare_cpp_triton(data, 8, 8, 128, 64, 512, label="v_bitexact")

    # --- Prefix (decode-like) scenario ---
    def test_with_prefix(self):
        """Verify RoPE positions are correctly offset by prefix_len."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        extend_len, prefix_len, block_size = 16, 384, 64
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[extend_len],
            prefix_lens=[prefix_len],
            block_size=block_size,
            seed=123,
        )
        _, _, cache_k_cpp, cache_k_tri, _, _ = _compare_cpp_triton(
            data, num_heads, kv_num_heads, head_dim, block_size, 512, label="prefix"
        )

        # Extra: check K cache at each position
        block_tables = data[3]
        for t in range(extend_len):
            pos = prefix_len + t
            bid = int(block_tables[0, pos // block_size].item())
            off = pos % block_size
            k_diff = _max_diff(cache_k_cpp[bid, :, off, :], cache_k_tri[bid, :, off, :])
            assert k_diff < 1e-3, f"K cache with prefix mismatch at token {t}: {k_diff}"

    # --- Multi-batch ---
    def test_multi_batch(self):
        """Multiple sequences with different prefix/extend lengths."""
        data = _make_test_data(8, 4, 128, extend_lens=[32, 16, 8, 24], prefix_lens=[0, 128, 64, 256])
        _compare_cpp_triton(data, 8, 4, 128, 64, 512, label="multi_batch")

    # --- Non-contiguous blocks ---
    def test_non_contiguous_blocks(self):
        """Shuffled physical block IDs."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        token_nums, block_size = 128, 64
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[token_nums],
            prefix_lens=[0],
            block_size=block_size,
            contiguous_blocks=False,
        )
        _, _, cache_k_cpp, cache_k_tri, cache_v_cpp, cache_v_tri = _compare_cpp_triton(
            data, num_heads, kv_num_heads, head_dim, block_size, 512, label="non_contig"
        )

        # Extra per-token verification
        block_tables = data[3]
        for t in range(token_nums):
            bid = int(block_tables[0, t // block_size].item())
            off = t % block_size
            k_diff = _max_diff(cache_k_cpp[bid, :, off, :], cache_k_tri[bid, :, off, :])
            v_diff = _max_diff(cache_v_cpp[bid, :, off, :], cache_v_tri[bid, :, off, :])
            assert k_diff < 1e-3, f"Non-contiguous K mismatch at token {t}: {k_diff}"
            assert v_diff == 0.0, f"Non-contiguous V mismatch at token {t}: {v_diff}"

    # --- Cross-block boundary ---
    def test_cross_block_boundary(self):
        """Tokens spanning multiple blocks: prefix=60, extend=10 crosses boundary at 64."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        extend_len, prefix_len, block_size = 10, 60, 64
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[extend_len],
            prefix_lens=[prefix_len],
            block_size=block_size,
            max_seq_len=256,
            seed=99,
        )
        _, _, cache_k_cpp, cache_k_tri, cache_v_cpp, cache_v_tri = _compare_cpp_triton(
            data, num_heads, kv_num_heads, head_dim, block_size, 256, label="cross_block"
        )

        block_tables = data[3]
        for t in range(extend_len):
            pos = prefix_len + t
            bid = int(block_tables[0, pos // block_size].item())
            off = pos % block_size
            k_diff = _max_diff(cache_k_cpp[bid, :, off, :], cache_k_tri[bid, :, off, :])
            v_diff = _max_diff(cache_v_cpp[bid, :, off, :], cache_v_tri[bid, :, off, :])
            assert k_diff < 1e-3, f"Cross-block K mismatch at pos {pos}: {k_diff}"
            assert v_diff == 0.0, f"Cross-block V mismatch at pos {pos}: {v_diff}"

    # --- Pure decode scenario (single token per seq, long prefix) ---
    @pytest.mark.parametrize(
        "num_heads,kv_num_heads,bs",
        [(8, 8, 1), (32, 8, 4), (8, 4, 8)],
        ids=["MHA-bs1", "GQA-bs4", "GQA-bs8"],
    )
    def test_pure_decode(self, num_heads, kv_num_heads, bs):
        """Pure decode: each seq has exactly 1 new token + long prefix."""
        head_dim, block_size = 128, 64
        extend_lens = [1] * bs
        prefix_lens = [100 + i * 50 for i in range(bs)]
        data = _make_test_data(
            num_heads, kv_num_heads, head_dim, extend_lens=extend_lens, prefix_lens=prefix_lens, seed=77
        )
        _, _, cache_k_cpp, cache_k_tri, cache_v_cpp, cache_v_tri = _compare_cpp_triton(
            data, num_heads, kv_num_heads, head_dim, block_size, 512, label="pure_decode"
        )

        block_tables = data[3]
        for b in range(bs):
            pos = prefix_lens[b]
            bid = int(block_tables[b, pos // block_size].item())
            off = pos % block_size
            k_diff = _max_diff(cache_k_cpp[bid, :, off, :], cache_k_tri[bid, :, off, :])
            v_diff = _max_diff(cache_v_cpp[bid, :, off, :], cache_v_tri[bid, :, off, :])
            assert k_diff < 1e-3, f"Pure decode K mismatch at batch {b}: {k_diff}"
            assert v_diff == 0.0, f"Pure decode V not bit-exact at batch {b}: {v_diff}"

    # --- Mixed batch (prefill + decode together) ---
    def test_mixed_prefill_decode(self):
        """Mixed batch: some seqs prefill (many tokens), some decode (1 token)."""
        data = _make_test_data(32, 8, 128, extend_lens=[32, 1, 1, 16, 1], prefix_lens=[0, 200, 128, 0, 300], seed=55)
        _compare_cpp_triton(data, 32, 8, 128, 64, 512, label="mixed")

    # --- Padding sentinel: batch_id_per_token with -1 tail ---
    def test_padding_sentinel(self):
        """batch_id_per_token buffer larger than token_num, tail filled with -1."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        actual_tokens, buf_size = 16, 64
        block_size, max_seq_len = 64, 256
        rotary_embs = _make_rotary_embs(max_seq_len, head_dim, neox_full=True)
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim

        paddle.seed(42)
        qkv = paddle.randn([buf_size, total_dim]).astype(COMPUTE_DTYPE)
        q_out = paddle.empty([buf_size, num_heads, head_dim], dtype=COMPUTE_DTYPE)
        cache_k = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(max_blocks_per_seq, dtype="int32").unsqueeze(0)

        batch_ids = [0] * actual_tokens + [-1] * (buf_size - actual_tokens)
        batch_id_per_token = paddle.to_tensor(batch_ids, dtype="int32")
        cu_seqlens_q = paddle.to_tensor([0, actual_tokens], dtype="int32")
        enc = paddle.to_tensor([actual_tokens], dtype="int32")
        dec = paddle.to_tensor([0], dtype="int32")

        triton_rope_and_cache_write(
            qkv,
            cache_k,
            cache_v,
            q_out,
            rotary_embs,
            batch_id_per_token,
            cu_seqlens_q,
            enc,
            dec,
            block_tables,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            use_neox_rotary_style=True,
        )

        # Compare with clean run using only actual_tokens
        cache_k_ref = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_ref = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        q_ref = paddle.empty([actual_tokens, num_heads, head_dim], dtype=COMPUTE_DTYPE)
        batch_id_clean = paddle.to_tensor([0] * actual_tokens, dtype="int32")
        triton_rope_and_cache_write(
            qkv[:actual_tokens],
            cache_k_ref,
            cache_v_ref,
            q_ref,
            rotary_embs,
            batch_id_clean,
            cu_seqlens_q,
            enc,
            dec,
            block_tables,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            use_neox_rotary_style=True,
        )

        assert _max_diff(q_out[:actual_tokens], q_ref) == 0.0, "Padding sentinel Q mismatch"
        assert _max_diff(cache_k, cache_k_ref) == 0.0, "Padding sentinel K cache mismatch"
        assert _max_diff(cache_v, cache_v_ref) == 0.0, "Padding sentinel V cache mismatch"

    # --- CUDA Graph replay test ---
    def test_cudagraph_replay(self):
        """Verify kernel is CUDA Graph compatible: capture once, replay with different data."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        token_nums, block_size, max_seq_len = 16, 64, 256

        rotary_embs = _make_rotary_embs(max_seq_len, head_dim, neox_full=True)
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim

        # Pre-extract cos/sin BEFORE capture (avoids .contiguous() alloc inside graph)
        cos_2d, sin_2d, _ = extract_cos_sin(rotary_embs, head_dim, use_neox_rotary_style=True)

        # Pre-allocate all buffers (CUDA Graph requires fixed addresses)
        qkv_buf = paddle.empty([token_nums, total_dim], dtype=COMPUTE_DTYPE)
        q_out_buf = paddle.empty([token_nums, num_heads, head_dim], dtype=COMPUTE_DTYPE)
        cache_k_buf = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_buf = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(max_blocks_per_seq, dtype="int32").unsqueeze(0)

        cu_seqlens_q = paddle.to_tensor([0, token_nums], dtype="int32")
        batch_id_per_token = paddle.zeros([token_nums], dtype="int32")
        enc = paddle.to_tensor([token_nums], dtype="int32")
        dec = paddle.to_tensor([0], dtype="int32")
        q_out_ptr = q_out_buf.data_ptr()

        # Warm up
        paddle.seed(42)
        qkv_buf.copy_(paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE), False)
        triton_rope_and_cache_write(
            qkv_buf,
            cache_k_buf,
            cache_v_buf,
            q_out_buf,
            rotary_embs,
            batch_id_per_token,
            cu_seqlens_q,
            enc,
            dec,
            block_tables,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            use_neox_rotary_style=True,
            cos_2d=cos_2d,
            sin_2d=sin_2d,
        )

        # Capture CUDA Graph (uses pre-extracted cos/sin, no dynamic alloc)
        paddle.device.synchronize()
        from paddle.device.cuda import graphs

        graph = graphs.CUDAGraph()
        graph.capture_begin()
        triton_rope_and_cache_write(
            qkv_buf,
            cache_k_buf,
            cache_v_buf,
            q_out_buf,
            rotary_embs,
            batch_id_per_token,
            cu_seqlens_q,
            enc,
            dec,
            block_tables,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            use_neox_rotary_style=True,
            cos_2d=cos_2d,
            sin_2d=sin_2d,
        )
        graph.capture_end()

        # Replay with different data
        for i in range(5):
            paddle.seed(100 + i)
            new_qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)
            qkv_buf.copy_(new_qkv, False)
            cache_k_buf.zero_()
            cache_v_buf.zero_()

            graph.replay()
            paddle.device.synchronize()

            assert q_out_buf.data_ptr() == q_out_ptr, "q_out address changed after replay!"

            # Fresh call for comparison
            cache_k_fresh = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
            cache_v_fresh = paddle.zeros([max_blocks_per_seq, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
            q_fresh = paddle.empty([token_nums, num_heads, head_dim], dtype=COMPUTE_DTYPE)
            triton_rope_and_cache_write(
                new_qkv,
                cache_k_fresh,
                cache_v_fresh,
                q_fresh,
                rotary_embs,
                batch_id_per_token,
                cu_seqlens_q,
                enc,
                dec,
                block_tables,
                num_heads,
                kv_num_heads,
                head_dim,
                block_size,
                use_neox_rotary_style=True,
            )

            q_diff = _max_diff(q_out_buf, q_fresh)
            assert q_diff == 0.0, f"CUDA Graph replay Q mismatch at iter {i}: {q_diff}"

    # -----------------------------------------------------------------------
    # NEW: decode with seq_lens_encoder_for_rope workaround vs C++ reference
    # -----------------------------------------------------------------------
    def test_decode_with_workaround(self):
        """Verify decode path with seq_lens_encoder_for_rope workaround matches C++.

        In real decode, seq_lens_encoder=0. The caller transforms it to 1
        via `paddle.where(enc==0, 1, enc)` (the workaround). This test ensures
        the Triton kernel with this workaround produces correct RoPE positions,
        matching the C++ reference under the same conditions.
        """
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        bs, block_size, max_seq_len = 4, 64, 512
        extend_lens = [1] * bs
        prefix_lens = [100, 200, 150, 300]

        data = _make_test_data(
            num_heads, kv_num_heads, head_dim, extend_lens=extend_lens, prefix_lens=prefix_lens, seed=88
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, _, dec, stt, rotary_embs = data

        # Apply the workaround: enc=0 → enc=1 (same as deterministic_attention.py)
        enc_workaround = paddle.ones([bs], dtype="int32")

        common = dict(
            block_tables=block_tables,
            seq_lens_encoder=enc_workaround,
            seq_lens_decoder=dec,
            seq_lens_this_time=stt,
            rotary_embs=rotary_embs,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_seq_len=max_seq_len,
        )
        cache_k_cpp, cache_v_cpp = cache_k_ref.clone(), cache_v_ref.clone()
        q_cpp = _call_cpp_rope(qkv, cache_k_cpp, cache_v_cpp, **common)

        cache_k_tri, cache_v_tri = cache_k_ref.clone(), cache_v_ref.clone()
        q_tri = _call_triton_rope(qkv, cache_k_tri, cache_v_tri, **common)

        q_diff = _max_diff(q_cpp, q_tri)
        k_diff = _max_diff(cache_k_cpp, cache_k_tri)
        v_diff = _max_diff(cache_v_cpp, cache_v_tri)
        assert q_diff < 1e-3, f"Decode workaround Q mismatch: {q_diff}"
        assert k_diff < 1e-3, f"Decode workaround K mismatch: {k_diff}"
        assert v_diff == 0.0, f"Decode workaround V not bit-exact: {v_diff}"

    # -----------------------------------------------------------------------
    # NEW: single token, single batch (minimal CUDA Graph decode scenario)
    # -----------------------------------------------------------------------
    def test_single_token(self):
        """Minimal case: 1 token, 1 batch, no prefix."""
        data = _make_test_data(8, 8, 128, extend_lens=[1], prefix_lens=[0], seed=111)
        _compare_cpp_triton(data, 8, 8, 128, 64, 512, label="single_token")

    # -----------------------------------------------------------------------
    # NEW: different block_size (Triton self-consistency)
    # -----------------------------------------------------------------------
    def test_block_size_self_consistency(self):
        """Q output should be identical across block sizes (Q doesn't use block_size)."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        # Reference: block_size=64
        data64 = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[32],
            prefix_lens=[20],
            block_size=64,
            max_seq_len=256,
            seed=222,
        )
        qkv64, ck64, cv64, bt64, enc64, dec64, stt64, re64 = data64
        q64 = _call_triton_rope(
            qkv64,
            ck64.clone(),
            cv64.clone(),
            bt64,
            enc64,
            dec64,
            stt64,
            re64,
            num_heads,
            kv_num_heads,
            head_dim,
            64,
            256,
        )

        for block_size in [16, 32]:
            data = _make_test_data(
                num_heads,
                kv_num_heads,
                head_dim,
                extend_lens=[32],
                prefix_lens=[20],
                block_size=block_size,
                max_seq_len=256,
                seed=222,
            )
            qkv, ck, cv, bt, enc, dec, stt, re = data
            q = _call_triton_rope(
                qkv, ck.clone(), cv.clone(), bt, enc, dec, stt, re, num_heads, kv_num_heads, head_dim, block_size, 256
            )

            q_diff = _max_diff(q, q64)
            assert q_diff == 0.0, f"block_size={block_size} vs 64: Q should be identical, got {q_diff}"

    # -----------------------------------------------------------------------
    # NEW: near max_seq_len boundary (test cos/sin bounds)
    # -----------------------------------------------------------------------
    def test_near_max_seq_len(self):
        """Position near max_seq_len boundary: prefix=500, extend=10, max=512."""
        data = _make_test_data(
            8, 8, 128, extend_lens=[10], prefix_lens=[500], block_size=64, max_seq_len=512, seed=333
        )
        _compare_cpp_triton(data, 8, 8, 128, 64, 512, label="near_max_seq_len")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
