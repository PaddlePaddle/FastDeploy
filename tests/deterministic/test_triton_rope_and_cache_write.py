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

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_triton_rope_and_cache_write.py -v -s
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.triton_ops.rope_and_cache_write import (
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

    # Pre-allocate q_out
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
# Helper: build rotary embeddings
# ---------------------------------------------------------------------------


def _make_rotary_embs(max_seq_len, head_dim, neox_full=True, base=10000.0):
    """Build rotary_embs: [2, 1, max_seq_len, 1, emb_dim].
    neox_full=True: emb_dim=head_dim (Qwen3 style, cos/sin duplicated)
    neox_full=False: emb_dim=head_dim/2 (standard interleaved, Qwen2 style)
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (paddle.arange(0, half_dim, dtype="float32") / half_dim))
    positions = paddle.arange(max_seq_len, dtype="float32")
    freqs = paddle.outer(positions, inv_freq)  # [max_seq_len, half_dim]

    if neox_full:
        # Qwen3: full head_dim, duplicated
        cos_full = paddle.concat([paddle.cos(freqs), paddle.cos(freqs)], axis=-1)
        sin_full = paddle.concat([paddle.sin(freqs), paddle.sin(freqs)], axis=-1)
    else:
        # Standard interleaved: head_dim/2
        cos_full = paddle.cos(freqs)
        sin_full = paddle.sin(freqs)

    embs = paddle.stack([cos_full, sin_full], axis=0)  # [2, max_seq_len, emb_dim]
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
    """Build test tensors for one or more sequences.

    Args:
        extend_lens: list of ints, new tokens per sequence
        prefix_lens: list of ints, prefix (cached) length per sequence
    """
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
        # Non-contiguous: shuffle
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


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestTritonRopeAndCacheWrite:
    """Compare Triton kernel against C++ reference."""

    # --- Q RoPE correctness: neox full (Qwen3 style) ---
    @pytest.mark.parametrize(
        "num_heads,kv_num_heads,head_dim,token_nums",
        [
            (8, 8, 128, 32),
            (32, 8, 128, 16),
            (28, 4, 128, 64),
        ],
        ids=["MHA-8h", "GQA-32q8kv", "GQA-28q4kv"],
    )
    def test_q_rope_neox_full(self, num_heads, kv_num_heads, head_dim, token_nums):
        """Q RoPE with neox full style should match C++ reference."""
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[token_nums],
            prefix_lens=[0],
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        # C++ reference
        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
            use_neox_rotary_style=True,
        )

        # Triton
        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
            use_neox_rotary_style=True,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Q RoPE neox full mismatch: {q_diff}"

    # --- Q RoPE correctness: standard interleaved (parameterized) ---
    @pytest.mark.parametrize(
        "num_heads,kv_num_heads,head_dim,token_nums",
        [
            (8, 8, 128, 32),
            (32, 8, 128, 16),
            (28, 4, 128, 64),
        ],
        ids=["MHA-8h", "GQA-32q8kv", "GQA-28q4kv"],
    )
    def test_q_rope_standard(self, num_heads, kv_num_heads, head_dim, token_nums):
        """Q RoPE with standard interleaved style should match C++ reference."""
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[token_nums],
            prefix_lens=[0],
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=False,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
            use_neox_rotary_style=False,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
            use_neox_rotary_style=False,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Q RoPE standard mismatch: {q_diff}"

    # --- KV cache write: V should be bit-exact ---
    def test_v_cache_bitexact(self):
        """V cache write is pure copy, should be bit-exact."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        token_nums = 32
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[token_nums],
            prefix_lens=[0],
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        # V should be identical (pure copy, no math)
        v_diff = float(paddle.max(paddle.abs(cache_v_cpp.astype("float32") - cache_v_tri.astype("float32"))).item())
        assert v_diff == 0.0, f"V cache not bit-exact: {v_diff}"

        # K should also be very close (RoPE math)
        k_diff = float(paddle.max(paddle.abs(cache_k_cpp.astype("float32") - cache_k_tri.astype("float32"))).item())
        assert k_diff < 1e-3, f"K cache mismatch: {k_diff}"

    # --- Prefix (decode-like) scenario ---
    def test_with_prefix(self):
        """Verify RoPE positions are correctly offset by prefix_len."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        extend_len, prefix_len = 16, 384
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[extend_len],
            prefix_lens=[prefix_len],
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
            seed=123,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Q with prefix mismatch: {q_diff}"

        # Check K cache at positions [prefix, prefix+extend)
        for t in range(extend_len):
            pos = prefix_len + t
            bid = int(block_tables[0, pos // block_size].item())
            off = pos % block_size
            k_diff = float(
                paddle.max(
                    paddle.abs(
                        cache_k_cpp[bid, :, off, :].astype("float32") - cache_k_tri[bid, :, off, :].astype("float32")
                    )
                ).item()
            )
            assert k_diff < 1e-3, f"K cache with prefix mismatch at token {t}: {k_diff}"

    # --- Multi-batch ---
    def test_multi_batch(self):
        """Multiple sequences with different prefix/extend lengths."""
        num_heads, kv_num_heads, head_dim = 8, 4, 128
        extend_lens = [32, 16, 8, 24]
        prefix_lens = [0, 128, 64, 256]
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=extend_lens,
            prefix_lens=prefix_lens,
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Multi-batch Q mismatch: {q_diff}"

        v_diff = float(paddle.max(paddle.abs(cache_v_cpp.astype("float32") - cache_v_tri.astype("float32"))).item())
        assert v_diff == 0.0, f"Multi-batch V cache not bit-exact: {v_diff}"

        k_diff = float(paddle.max(paddle.abs(cache_k_cpp.astype("float32") - cache_k_tri.astype("float32"))).item())
        assert k_diff < 1e-3, f"Multi-batch K cache mismatch: {k_diff}"

    # --- Non-contiguous blocks ---
    def test_non_contiguous_blocks(self):
        """Shuffled physical block IDs."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        token_nums = 128
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[token_nums],
            prefix_lens=[0],
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
            contiguous_blocks=False,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Non-contiguous Q mismatch: {q_diff}"

        # Check K/V at each token via block_tables
        for t in range(token_nums):
            bid = int(block_tables[0, t // block_size].item())
            off = t % block_size
            k_diff = float(
                paddle.max(
                    paddle.abs(
                        cache_k_cpp[bid, :, off, :].astype("float32") - cache_k_tri[bid, :, off, :].astype("float32")
                    )
                ).item()
            )
            v_diff = float(
                paddle.max(
                    paddle.abs(
                        cache_v_cpp[bid, :, off, :].astype("float32") - cache_v_tri[bid, :, off, :].astype("float32")
                    )
                ).item()
            )
            assert k_diff < 1e-3, f"Non-contiguous K mismatch at token {t}: {k_diff}"
            assert v_diff == 0.0, f"Non-contiguous V mismatch at token {t}: {v_diff}"

    # --- Cross-block boundary ---
    def test_cross_block_boundary(self):
        """Tokens spanning multiple blocks."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        block_size = 64
        # prefix=60, extend=10 => positions 60..69, crossing block boundary at 64
        extend_len, prefix_len = 10, 60
        max_seq_len = 256
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=[extend_len],
            prefix_lens=[prefix_len],
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
            seed=99,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Cross-block Q mismatch: {q_diff}"

        for t in range(extend_len):
            pos = prefix_len + t
            bid = int(block_tables[0, pos // block_size].item())
            off = pos % block_size
            k_diff = float(
                paddle.max(
                    paddle.abs(
                        cache_k_cpp[bid, :, off, :].astype("float32") - cache_k_tri[bid, :, off, :].astype("float32")
                    )
                ).item()
            )
            v_diff = float(
                paddle.max(
                    paddle.abs(
                        cache_v_cpp[bid, :, off, :].astype("float32") - cache_v_tri[bid, :, off, :].astype("float32")
                    )
                ).item()
            )
            assert k_diff < 1e-3, f"Cross-block K mismatch at pos {pos}: {k_diff}"
            assert v_diff == 0.0, f"Cross-block V mismatch at pos {pos}: {v_diff}"

    # --- Pure decode scenario (single token per seq, long prefix) ---
    @pytest.mark.parametrize(
        "num_heads,kv_num_heads,bs",
        [
            (8, 8, 1),
            (32, 8, 4),
            (8, 4, 8),
        ],
        ids=["MHA-bs1", "GQA-bs4", "GQA-bs8"],
    )
    def test_pure_decode(self, num_heads, kv_num_heads, bs):
        """Pure decode: each seq has exactly 1 new token + long prefix."""
        head_dim = 128
        extend_lens = [1] * bs
        prefix_lens = [100 + i * 50 for i in range(bs)]
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=extend_lens,
            prefix_lens=prefix_lens,
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
            seed=77,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Pure decode Q mismatch: {q_diff}"

        # Check K/V at each decode position
        for b in range(bs):
            pos = prefix_lens[b]  # single decode token at this position
            bid = int(block_tables[b, pos // block_size].item())
            off = pos % block_size
            k_diff = float(
                paddle.max(
                    paddle.abs(
                        cache_k_cpp[bid, :, off, :].astype("float32") - cache_k_tri[bid, :, off, :].astype("float32")
                    )
                ).item()
            )
            v_diff = float(
                paddle.max(
                    paddle.abs(
                        cache_v_cpp[bid, :, off, :].astype("float32") - cache_v_tri[bid, :, off, :].astype("float32")
                    )
                ).item()
            )
            assert k_diff < 1e-3, f"Pure decode K mismatch at batch {b}: {k_diff}"
            assert v_diff == 0.0, f"Pure decode V not bit-exact at batch {b}: {v_diff}"

    # --- Mixed batch (prefill + decode together) ---
    def test_mixed_prefill_decode(self):
        """Mixed batch: some seqs prefill (many tokens), some decode (1 token)."""
        num_heads, kv_num_heads, head_dim = 32, 8, 128
        extend_lens = [32, 1, 1, 16]
        prefix_lens = [0, 192, 128, 0]
        block_size, max_seq_len = 64, 512
        data = _make_test_data(
            num_heads,
            kv_num_heads,
            head_dim,
            extend_lens=extend_lens,
            prefix_lens=prefix_lens,
            block_size=block_size,
            max_seq_len=max_seq_len,
            neox_full=True,
            seed=55,
        )
        qkv, cache_k_ref, cache_v_ref, block_tables, enc, dec, stt, rotary_embs = data

        cache_k_cpp = cache_k_ref.clone()
        cache_v_cpp = cache_v_ref.clone()
        q_cpp = _call_cpp_rope(
            qkv,
            cache_k_cpp,
            cache_v_cpp,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        cache_k_tri = cache_k_ref.clone()
        cache_v_tri = cache_v_ref.clone()
        q_tri = _call_triton_rope(
            qkv,
            cache_k_tri,
            cache_v_tri,
            block_tables,
            enc,
            dec,
            stt,
            rotary_embs,
            num_heads,
            kv_num_heads,
            head_dim,
            block_size,
            max_seq_len,
        )

        q_diff = float(paddle.max(paddle.abs(q_cpp.astype("float32") - q_tri.astype("float32"))).item())
        assert q_diff < 1e-3, f"Mixed batch Q mismatch: {q_diff}"

        v_diff = float(paddle.max(paddle.abs(cache_v_cpp.astype("float32") - cache_v_tri.astype("float32"))).item())
        assert v_diff == 0.0, f"Mixed batch V not bit-exact: {v_diff}"

        k_diff = float(paddle.max(paddle.abs(cache_k_cpp.astype("float32") - cache_k_tri.astype("float32"))).item())
        assert k_diff < 1e-3, f"Mixed batch K mismatch: {k_diff}"

    # --- Padding sentinel: batch_id_per_token with -1 tail ---
    def test_padding_sentinel(self):
        """batch_id_per_token buffer larger than token_num, tail filled with -1."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        actual_tokens = 16
        buf_size = 64  # larger buffer, tail is -1
        block_size, max_seq_len = 64, 256
        rotary_embs = _make_rotary_embs(max_seq_len, head_dim, neox_full=True)
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        total_blocks = max_blocks_per_seq
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim

        paddle.seed(42)
        # Create qkv with buf_size rows (padded)
        qkv = paddle.randn([buf_size, total_dim]).astype(COMPUTE_DTYPE)
        q_out = paddle.empty([buf_size, num_heads, head_dim], dtype=COMPUTE_DTYPE)
        cache_k = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(max_blocks_per_seq, dtype="int32").unsqueeze(0)

        # batch_id_per_token: valid for first actual_tokens, -1 for the rest
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

        # Compare with a clean run using only actual_tokens (no padding)
        cache_k_ref = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_ref = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
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

        # Q for valid tokens should match exactly
        q_diff = float(
            paddle.max(paddle.abs(q_out[:actual_tokens].astype("float32") - q_ref.astype("float32"))).item()
        )
        assert q_diff == 0.0, f"Padding sentinel Q mismatch: {q_diff}"

        # KV cache should match exactly
        k_diff = float(paddle.max(paddle.abs(cache_k.astype("float32") - cache_k_ref.astype("float32"))).item())
        v_diff = float(paddle.max(paddle.abs(cache_v.astype("float32") - cache_v_ref.astype("float32"))).item())
        assert k_diff == 0.0, f"Padding sentinel K cache mismatch: {k_diff}"
        assert v_diff == 0.0, f"Padding sentinel V cache mismatch: {v_diff}"

    # --- CUDA Graph replay test ---
    def test_cudagraph_replay(self):
        """Verify kernel is CUDA Graph compatible: capture once, replay with different data."""
        num_heads, kv_num_heads, head_dim = 8, 8, 128
        token_nums = 16
        block_size, max_seq_len = 64, 256

        rotary_embs = _make_rotary_embs(max_seq_len, head_dim, neox_full=True)
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        total_blocks = max_blocks_per_seq
        total_dim = (num_heads + 2 * kv_num_heads) * head_dim

        # Pre-allocate all buffers (CUDA Graph requires fixed addresses)
        qkv_buf = paddle.empty([token_nums, total_dim], dtype=COMPUTE_DTYPE)
        q_out_buf = paddle.empty([token_nums, num_heads, head_dim], dtype=COMPUTE_DTYPE)
        cache_k_buf = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        cache_v_buf = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
        block_tables = paddle.arange(max_blocks_per_seq, dtype="int32").unsqueeze(0)

        cu_seqlens_q = paddle.to_tensor([0, token_nums], dtype="int32")
        batch_id_per_token = paddle.zeros([token_nums], dtype="int32")
        enc = paddle.to_tensor([token_nums], dtype="int32")
        dec = paddle.to_tensor([0], dtype="int32")

        # Record captured q_out pointer
        q_out_ptr = q_out_buf.data_ptr()

        # Fill initial data and warm up
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
        )

        # Capture CUDA Graph
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
        )
        graph.capture_end()

        # Replay with different data multiple times
        for i in range(5):
            # Fill new random data
            paddle.seed(100 + i)
            new_qkv = paddle.randn([token_nums, total_dim]).astype(COMPUTE_DTYPE)
            qkv_buf.copy_(new_qkv, False)
            cache_k_buf.zero_()
            cache_v_buf.zero_()

            # Replay captured graph
            graph.replay()
            paddle.device.synchronize()

            # Verify address stability
            assert q_out_buf.data_ptr() == q_out_ptr, "q_out address changed after replay!"

            # Verify correctness by comparing with fresh call
            cache_k_fresh = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
            cache_v_fresh = paddle.zeros([total_blocks, kv_num_heads, block_size, head_dim], dtype=COMPUTE_DTYPE)
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

            q_diff = float(paddle.max(paddle.abs(q_out_buf.astype("float32") - q_fresh.astype("float32"))).item())
            assert q_diff == 0.0, f"CUDA Graph replay Q mismatch at iter {i}: {q_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
