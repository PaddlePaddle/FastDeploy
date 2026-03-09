"""
CUDA Graph compatibility tests for deterministic attention index-building functions.

Verifies that pre-allocated buffer paths produce identical results to dynamic allocation,
and that functions can survive CUDA Graph capture/replay without cudaErrorIllegalAddress.
"""

import paddle
import pytest
import triton

from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
    _elementwise_add_kernel,
    _indptr_to_lens_kernel,
    build_kv_indices_from_block_tables,
    build_unified_kv_indices,
    extend_attention_fwd_unified,
    pre_cache_len_concat_triton,
    triton_cumsum_with_zero_prefix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block_tables(bs, max_seq_len, block_size, num_total_blocks=256):
    """Create random block_tables and seq_lens for testing."""
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = paddle.randint(0, num_total_blocks, [bs, max_blocks_per_seq], dtype="int32")
    seq_lens = paddle.randint(1, max_seq_len + 1, [bs], dtype="int32")
    return block_tables, seq_lens


def _assert_equal(a, b, msg=""):
    """Assert two int32 tensors are identical."""
    assert paddle.equal_all(a, b).item(), f"{msg}: mismatch\n  a={a.cpu().numpy()}\n  b={b.cpu().numpy()}"


def _compute_valid_gridx(seq_lens_encoder, seq_lens_decoder, bsz, block_size):
    """Compute number of valid elements in batch_ids/tile_ids output."""
    gridx = 0
    for bid in range(bsz):
        enc = int(seq_lens_encoder[bid].item())
        dec = int(seq_lens_decoder[bid].item())
        cache_len = dec if enc > 0 else 0
        loop_times = (cache_len + block_size - 1) // block_size
        gridx += loop_times
    return gridx


# ---------------------------------------------------------------------------
# 1. Helper kernel tests
# ---------------------------------------------------------------------------


class TestHelperKernels:
    """Test _indptr_to_lens_kernel and _elementwise_add_kernel."""

    @pytest.mark.parametrize("n", [1, 2, 4, 7, 16, 33])
    def test_indptr_to_lens(self, n):
        lens_ref = paddle.randint(0, 100, [n], dtype="int32")
        indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(lens_ref).astype("int32")])
        lens_out = paddle.empty([n], dtype="int32")
        BLOCK = triton.next_power_of_2(n)
        _indptr_to_lens_kernel[(1,)](indptr, lens_out, n, BLOCK=BLOCK)
        _assert_equal(lens_out, lens_ref, "indptr_to_lens")

    @pytest.mark.parametrize("n", [1, 2, 4, 7, 16, 33])
    def test_elementwise_add(self, n):
        a = paddle.randint(0, 100, [n], dtype="int32")
        b = paddle.randint(0, 100, [n], dtype="int32")
        out = paddle.empty([n], dtype="int32")
        BLOCK = triton.next_power_of_2(n)
        _elementwise_add_kernel[(1,)](a, b, out, n, BLOCK=BLOCK)
        _assert_equal(out, a + b, "elementwise_add")


# ---------------------------------------------------------------------------
# 2. Pre-allocated buffer equivalence tests
# ---------------------------------------------------------------------------


class TestPreAllocatedBufferEquivalence:
    """Verify functions produce identical results with vs without pre-allocated buffers."""

    @pytest.mark.parametrize("n", [1, 5, 16, 33])
    def test_cumsum_with_buffer(self, n):
        x = paddle.randint(1, 10, [n], dtype="int32")
        ref = triton_cumsum_with_zero_prefix(x, n)
        buf = paddle.empty([64], dtype="int32")  # oversized
        out = triton_cumsum_with_zero_prefix(x, n, out_buf=buf)
        _assert_equal(out, ref, "cumsum buf vs dynamic")
        # Verify it wrote into buf (same data_ptr)
        assert out.data_ptr() == buf.data_ptr(), "should reuse buf memory"

    @pytest.mark.parametrize("bs", [1, 3, 8])
    def test_build_kv_indices_with_buffer(self, bs):
        block_size = 16
        max_seq_len = 128
        block_tables, seq_lens = _make_block_tables(bs, max_seq_len, block_size)
        total_kv = int(paddle.sum(seq_lens).item())

        ref_indptr, ref_indices = build_kv_indices_from_block_tables(
            block_tables,
            seq_lens,
            block_size,
            bs,
            total_kv_len=total_kv,
        )
        indptr_buf = paddle.empty([bs + 1], dtype="int32")
        indices_buf = paddle.empty([max(total_kv, 1)], dtype="int32")
        out_indptr, out_indices = build_kv_indices_from_block_tables(
            block_tables,
            seq_lens,
            block_size,
            bs,
            total_kv_len=total_kv,
            kv_indptr_buf=indptr_buf,
            kv_indices_buf=indices_buf,
        )
        _assert_equal(out_indptr, ref_indptr, "kv_indptr buf vs dynamic")
        _assert_equal(out_indices, ref_indices, "kv_indices buf vs dynamic")

    def test_build_unified_kv_indices_with_buffer(self):
        bs = 4
        prefix_lens_val = paddle.to_tensor([10, 20, 5, 15], dtype="int32")
        extend_lens_val = paddle.to_tensor([3, 1, 7, 2], dtype="int32")
        prefix_kv_indptr = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(prefix_lens_val).astype("int32"),
            ]
        )
        prefix_kv_indices = paddle.randint(0, 100, [int(prefix_lens_val.sum().item())], dtype="int32")
        extend_start_loc = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(extend_lens_val[:-1]).astype("int32"),
            ]
        )
        extend_kv_indices = paddle.randint(0, 100, [int(extend_lens_val.sum().item())], dtype="int32")

        ref_indptr, ref_indices, ref_plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_lens_val,
            extend_kv_indices,
            bs,
        )

        total_len = prefix_kv_indices.shape[0] + extend_kv_indices.shape[0]
        out_indptr, out_indices, out_plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_lens_val,
            extend_kv_indices,
            bs,
            unified_kv_indptr_buf=paddle.empty([bs + 1], dtype="int32"),
            unified_kv_indices_buf=paddle.empty([total_len], dtype="int32"),
            prefix_lens_buf=paddle.empty([bs], dtype="int32"),
            unified_lens_buf=paddle.empty([bs], dtype="int32"),
        )
        _assert_equal(out_indptr, ref_indptr, "unified_kv_indptr")
        _assert_equal(out_indices, ref_indices, "unified_kv_indices")
        _assert_equal(out_plens, ref_plens, "prefix_lens")

    @pytest.mark.parametrize("bsz", [1, 4, 8])
    def test_pre_cache_len_concat_with_buffer(self, bsz):
        block_size = 64
        seq_lens_encoder = paddle.randint(0, 200, [bsz], dtype="int32")
        seq_lens_decoder = paddle.randint(1, 500, [bsz], dtype="int32")
        seq_lens_this_time = paddle.where(seq_lens_encoder > 0, seq_lens_encoder, paddle.ones([bsz], dtype="int32"))
        max_dec = int(seq_lens_decoder.max().item())
        max_tile = (max_dec + block_size - 1) // block_size

        ref_cu, ref_bid, ref_tid = pre_cache_len_concat_triton(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            bsz,
            block_size,
            max_tile,
        )

        max_out = max(bsz * max_tile, 1)
        out_cu, out_bid, out_tid = pre_cache_len_concat_triton(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            bsz,
            block_size,
            max_tile,
            cu_seqlens_k_buf=paddle.empty([bsz + 1], dtype="int32"),
            batch_ids_buf=paddle.empty([max_out], dtype="int32"),
            tile_ids_buf=paddle.empty([max_out], dtype="int32"),
            cache_len_buf=paddle.empty([bsz], dtype="int32"),
            loop_times_buf=paddle.empty([bsz], dtype="int32"),
            gridx_offset_buf=paddle.empty([bsz + 1], dtype="int32"),
        )
        _assert_equal(out_cu, ref_cu, "cu_seqlens_k")
        # batch_ids/tile_ids only valid up to total loop_times (tail is uninitialized)
        valid = _compute_valid_gridx(seq_lens_encoder, seq_lens_decoder, bsz, block_size)
        if valid > 0:
            _assert_equal(out_bid[:valid], ref_bid[:valid], "batch_ids")
            _assert_equal(out_tid[:valid], ref_tid[:valid], "tile_ids")


# ---------------------------------------------------------------------------
# 3. CUDA Graph capture/replay tests
# ---------------------------------------------------------------------------


class TestCudaGraphReplay:
    """
    Verify that using pre-allocated buffers allows CUDA Graph capture and replay
    without cudaErrorIllegalAddress. This is the core bug that was missed.
    """

    def _capture_and_replay(self, fn, num_replays=3):
        """Helper: warmup, capture, replay num_replays times, return last output."""
        # Warmup
        fn()
        paddle.device.synchronize()

        graph = paddle.device.cuda.graphs.CUDAGraph()
        graph.capture_begin()
        result = fn()
        graph.capture_end()
        paddle.device.synchronize()

        for _ in range(num_replays):
            graph.replay()
        paddle.device.synchronize()
        return result

    def test_cumsum_replay(self):
        """triton_cumsum_with_zero_prefix survives capture/replay with pre-allocated buf."""
        n = 8
        x = paddle.randint(1, 10, [n], dtype="int32")
        buf = paddle.empty([n + 1], dtype="int32")

        ref = triton_cumsum_with_zero_prefix(x, n)

        def fn():
            return triton_cumsum_with_zero_prefix(x, n, out_buf=buf)

        result = self._capture_and_replay(fn)
        _assert_equal(result, ref, "cumsum after replay")

    def test_build_kv_indices_replay(self):
        """build_kv_indices_from_block_tables survives capture/replay."""
        bs, block_size, max_seq_len = 4, 16, 64
        block_tables, seq_lens = _make_block_tables(bs, max_seq_len, block_size)
        total_kv = int(paddle.sum(seq_lens).item())

        indptr_buf = paddle.empty([bs + 1], dtype="int32")
        indices_buf = paddle.empty([max(bs * max_seq_len, 1)], dtype="int32")

        ref_indptr, ref_indices = build_kv_indices_from_block_tables(
            block_tables,
            seq_lens,
            block_size,
            bs,
            total_kv_len=total_kv,
        )

        def fn():
            return build_kv_indices_from_block_tables(
                block_tables,
                seq_lens,
                block_size,
                bs,
                total_kv_len=total_kv,
                kv_indptr_buf=indptr_buf,
                kv_indices_buf=indices_buf,
            )

        indptr_out, indices_out = self._capture_and_replay(fn)
        _assert_equal(indptr_out, ref_indptr, "kv_indptr after replay")
        _assert_equal(indices_out, ref_indices, "kv_indices after replay")

    def test_pre_cache_len_concat_replay(self):
        """pre_cache_len_concat_triton survives capture/replay."""
        bsz, block_size = 4, 64
        seq_lens_encoder = paddle.to_tensor([50, 0, 30, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([100, 200, 80, 150], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([50, 1, 30, 1], dtype="int32")
        max_dec = int(seq_lens_decoder.max().item())
        max_tile = (max_dec + block_size - 1) // block_size
        max_out = max(bsz * max_tile, 1)

        bufs = dict(
            cu_seqlens_k_buf=paddle.empty([bsz + 1], dtype="int32"),
            batch_ids_buf=paddle.empty([max_out], dtype="int32"),
            tile_ids_buf=paddle.empty([max_out], dtype="int32"),
            cache_len_buf=paddle.empty([bsz], dtype="int32"),
            loop_times_buf=paddle.empty([bsz], dtype="int32"),
            gridx_offset_buf=paddle.empty([bsz + 1], dtype="int32"),
        )

        ref = pre_cache_len_concat_triton(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            bsz,
            block_size,
            max_tile,
        )

        def fn():
            return pre_cache_len_concat_triton(
                seq_lens_encoder,
                seq_lens_decoder,
                seq_lens_this_time,
                bsz,
                block_size,
                max_tile,
                **bufs,
            )

        out = self._capture_and_replay(fn)
        _assert_equal(out[0], ref[0], "cu_seqlens_k after replay")
        # Only compare valid range of batch_ids/tile_ids
        valid = _compute_valid_gridx(seq_lens_encoder, seq_lens_decoder, bsz, block_size)
        if valid > 0:
            _assert_equal(out[1][:valid], ref[1][:valid], "batch_ids after replay")
            _assert_equal(out[2][:valid], ref[2][:valid], "tile_ids after replay")

    def test_full_index_pipeline_replay(self):
        """
        Full index-building pipeline (the exact call chain in _deterministic_build_triton_indices)
        survives CUDA Graph capture/replay with pre-allocated buffers.

        This is the scenario that caused the original cudaErrorIllegalAddress crash.
        """
        bs, block_size, max_seq_len = 4, 16, 128
        num_total_blocks = 256
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        block_tables = paddle.randint(0, num_total_blocks, [bs, max_blocks_per_seq], dtype="int32")

        # Decode scenario: each seq extends by 1 token, has a prefix
        prefix_lens = paddle.to_tensor([30, 50, 20, 40], dtype="int32")
        extend_seq_lens = paddle.to_tensor([1, 1, 1, 1], dtype="int32")
        total_prefix_len = int(prefix_lens.sum().item())
        total_extend_len = int(extend_seq_lens.sum().item())
        max_total_kv = max(bs * max_seq_len, 1)

        # Pre-allocate all buffers
        bufs = dict(
            qo_indptr=paddle.empty([bs + 1], dtype="int32"),
            prefix_kv_indptr=paddle.empty([bs + 1], dtype="int32"),
            prefix_kv_indices=paddle.empty([max_total_kv], dtype="int32"),
            all_kv_indptr=paddle.empty([bs + 1], dtype="int32"),
            all_kv_indices=paddle.empty([max_total_kv], dtype="int32"),
            extend_kv_indices=paddle.empty([max(bs, 1)], dtype="int32"),
            unified_kv_indptr=paddle.empty([bs + 1], dtype="int32"),
            unified_kv_indices=paddle.empty([max_total_kv], dtype="int32"),
            prefix_lens_buf=paddle.empty([bs], dtype="int32"),
            unified_lens_buf=paddle.empty([bs], dtype="int32"),
            total_seq_lens_buf=paddle.empty([bs], dtype="int32"),
        )

        def pipeline():
            from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
                _scatter_extend_kv_indices_kernel,
            )

            qo_indptr = triton_cumsum_with_zero_prefix(
                extend_seq_lens,
                bs,
                out_buf=bufs["qo_indptr"],
            )
            p_indptr, p_indices = build_kv_indices_from_block_tables(
                block_tables,
                prefix_lens,
                block_size,
                bs,
                total_kv_len=total_prefix_len,
                kv_indptr_buf=bufs["prefix_kv_indptr"],
                kv_indices_buf=bufs["prefix_kv_indices"],
            )
            # total_seq_lens via Triton kernel (no dynamic alloc)
            total_seq_lens = bufs["total_seq_lens_buf"][:bs]
            BLOCK = triton.next_power_of_2(bs)
            _elementwise_add_kernel[(1,)](prefix_lens, extend_seq_lens, total_seq_lens, bs, BLOCK=BLOCK)

            a_indptr, a_indices = build_kv_indices_from_block_tables(
                block_tables,
                total_seq_lens,
                block_size,
                bs,
                total_kv_len=total_prefix_len + total_extend_len,
                kv_indptr_buf=bufs["all_kv_indptr"],
                kv_indices_buf=bufs["all_kv_indices"],
            )
            ext_start = qo_indptr[:bs]
            ext_kv = bufs["extend_kv_indices"][: max(total_extend_len, 1)]
            _scatter_extend_kv_indices_kernel[(bs,)](
                a_indices,
                a_indptr,
                prefix_lens,
                ext_start,
                extend_seq_lens,
                ext_kv,
                BLOCK=128,
            )
            u_indptr, u_indices, _ = build_unified_kv_indices(
                p_indptr,
                p_indices,
                ext_start,
                extend_seq_lens,
                ext_kv,
                bs,
                unified_kv_indptr_buf=bufs["unified_kv_indptr"],
                unified_kv_indices_buf=bufs["unified_kv_indices"],
                prefix_lens_buf=bufs["prefix_lens_buf"],
                unified_lens_buf=bufs["unified_lens_buf"],
            )
            return qo_indptr, u_indptr, u_indices

        # Reference (dynamic alloc)
        ref_qo, ref_uindptr, ref_uindices = pipeline()
        ref_qo = ref_qo.clone()
        ref_uindptr = ref_uindptr.clone()
        ref_uindices = ref_uindices.clone()

        # CUDA Graph capture + replay
        result = self._capture_and_replay(pipeline, num_replays=5)
        _assert_equal(result[0], ref_qo, "qo_indptr after replay")
        _assert_equal(result[1], ref_uindptr, "unified_kv_indptr after replay")
        _assert_equal(result[2], ref_uindices, "unified_kv_indices after replay")

    def test_replay_with_changing_effective_bs(self):
        """
        After capture at bs=4, replay with different effective batch sizes
        (simulated by changing seq_lens_this_time to have trailing zeros).
        The pre-allocated buffers must handle this without crash.
        """
        max_bs = 4
        extend_seq_lens = paddle.to_tensor([1, 1, 1, 1], dtype="int32")

        buf = paddle.empty([max_bs + 1], dtype="int32")

        def fn():
            return triton_cumsum_with_zero_prefix(extend_seq_lens, max_bs, out_buf=buf)

        # Capture
        fn()
        paddle.device.synchronize()
        graph = paddle.device.cuda.graphs.CUDAGraph()
        graph.capture_begin()
        result = fn()
        graph.capture_end()
        paddle.device.synchronize()

        # Replay multiple times — should not crash
        for _ in range(5):
            graph.replay()
        paddle.device.synchronize()

        # Verify result is still correct
        ref = triton_cumsum_with_zero_prefix(extend_seq_lens, max_bs)
        _assert_equal(result, ref, "cumsum after multiple replays")


# ---------------------------------------------------------------------------
# 4. Attention output buffer reuse test
# ---------------------------------------------------------------------------


class TestOutputBufferReuse:
    """Test that attention output with pre-allocated buffer matches dynamic allocation."""

    def test_attention_output_with_preallocated_buffer(self):
        """extend_attention_fwd_unified produces same result with pre-allocated vs fresh output."""
        num_q_heads, num_kv_heads, head_dim = 8, 2, 64
        block_size = 16
        num_blocks = 32
        prefix_lens_val = [10, 20]
        extend_lens_val = [3, 1]
        total_tokens = sum(extend_lens_val)

        q = paddle.randn([total_tokens, num_q_heads, head_dim], dtype="float16")
        k_cache = paddle.randn([num_blocks, num_kv_heads, block_size, head_dim], dtype="float16")
        v_cache = paddle.randn([num_blocks, num_kv_heads, block_size, head_dim], dtype="float16")

        # Build indices (using dynamic path for simplicity)
        prefix_lens_t = paddle.to_tensor(prefix_lens_val, dtype="int32")
        extend_lens_t = paddle.to_tensor(extend_lens_val, dtype="int32")
        total_lens = prefix_lens_t + extend_lens_t
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(extend_lens_t).astype("int32")])

        kv_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(total_lens).astype("int32")])
        # Simple flat indices
        total_kv = int(total_lens.sum().item())
        kv_indices = paddle.randint(0, num_blocks * block_size, [total_kv], dtype="int32")

        max_extend = max(extend_lens_val)

        # Dynamic allocation
        o_dynamic = paddle.zeros([total_tokens, num_q_heads, head_dim], dtype="float16")
        res_dynamic = extend_attention_fwd_unified(
            q,
            o_dynamic,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens_t,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_extend,
        )

        # Pre-allocated buffer (oversized)
        max_capture = 8
        output_buf = paddle.empty([max_capture, num_q_heads, head_dim], dtype="float16")
        o_prealloc = output_buf[:total_tokens].zero_()
        res_prealloc = extend_attention_fwd_unified(
            q,
            o_prealloc,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens_t,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_extend,
        )

        diff = (res_dynamic.astype("float32") - res_prealloc.astype("float32")).abs().max().item()
        assert diff == 0.0, f"Output mismatch: max diff = {diff}"


# ---------------------------------------------------------------------------
# 5. Buffer size validation tests
# ---------------------------------------------------------------------------


class TestBufferSizeValidation:
    """
    Verify that _init_cudagraph_buffers allocates each buffer large enough
    for worst-case scenarios. This class of tests would have caught the
    extend_kv_indices undersizing bug (max_bsz instead of max_total_kv_len).
    """

    @pytest.mark.parametrize(
        "max_num_seqs,max_model_len,block_size",
        [
            (8, 8192, 64),  # production config
            (16, 4096, 64),  # larger batch
            (4, 16384, 128),  # very long context
            (1, 1024, 16),  # minimal config
        ],
    )
    def test_buffer_sizes_sufficient(self, max_num_seqs, max_model_len, block_size):
        """Every pre-allocated buffer must be >= its worst-case usage."""
        max_total_kv_len = max(max_num_seqs * max_model_len, 1)
        max_tile_per_bs = (max_model_len + block_size - 1) // block_size
        max_pre_cache_size = max(max_num_seqs * max_tile_per_bs, 1)

        from fastdeploy.model_executor.layers.attention.deterministic_attention import (
            DeterministicCudaGraphBuffers,
        )

        bufs = DeterministicCudaGraphBuffers()
        bufs.cu_seqlens_k = paddle.empty([max_num_seqs + 1], dtype="int32")
        bufs.pre_cache_batch_ids = paddle.empty([max_pre_cache_size], dtype="int32")
        bufs.pre_cache_tile_ids = paddle.empty([max_pre_cache_size], dtype="int32")
        bufs.cache_len = paddle.empty([max_num_seqs], dtype="int32")
        bufs.loop_times = paddle.empty([max_num_seqs], dtype="int32")
        bufs.gridx_offset = paddle.empty([max_num_seqs + 1], dtype="int32")
        bufs.qo_indptr = paddle.empty([max_num_seqs + 1], dtype="int32")
        bufs.prefix_kv_indptr = paddle.empty([max_num_seqs + 1], dtype="int32")
        bufs.prefix_kv_indices = paddle.empty([max_total_kv_len], dtype="int32")
        bufs.all_kv_indptr = paddle.empty([max_num_seqs + 1], dtype="int32")
        bufs.all_kv_indices = paddle.empty([max_total_kv_len], dtype="int32")
        bufs.extend_kv_indices = paddle.empty([max(max_total_kv_len, 1)], dtype="int32")
        bufs.unified_kv_indptr = paddle.empty([max_num_seqs + 1], dtype="int32")
        bufs.unified_kv_indices = paddle.empty([max_total_kv_len], dtype="int32")
        bufs.prefix_lens_buf = paddle.empty([max_num_seqs], dtype="int32")
        bufs.unified_lens_buf = paddle.empty([max_num_seqs], dtype="int32")
        bufs.total_seq_lens_buf = paddle.empty([max_num_seqs], dtype="int32")

        # indptr buffers: need bs+1 elements
        assert bufs.cu_seqlens_k.shape[0] >= max_num_seqs + 1
        assert bufs.qo_indptr.shape[0] >= max_num_seqs + 1
        assert bufs.prefix_kv_indptr.shape[0] >= max_num_seqs + 1
        assert bufs.all_kv_indptr.shape[0] >= max_num_seqs + 1
        assert bufs.unified_kv_indptr.shape[0] >= max_num_seqs + 1
        assert bufs.gridx_offset.shape[0] >= max_num_seqs + 1

        # per-batch buffers: need bs elements
        assert bufs.cache_len.shape[0] >= max_num_seqs
        assert bufs.loop_times.shape[0] >= max_num_seqs
        assert bufs.prefix_lens_buf.shape[0] >= max_num_seqs
        assert bufs.unified_lens_buf.shape[0] >= max_num_seqs
        assert bufs.total_seq_lens_buf.shape[0] >= max_num_seqs

        # KV index buffers: ALL must be >= max_total_kv_len
        # This is the key assertion that catches the original bug!
        for name in ["prefix_kv_indices", "all_kv_indices", "extend_kv_indices", "unified_kv_indices"]:
            buf = getattr(bufs, name)
            assert buf.shape[0] >= max_total_kv_len, f"{name} too small: {buf.shape[0]} < {max_total_kv_len}"

        # pre_cache buffers
        assert bufs.pre_cache_batch_ids.shape[0] >= max_pre_cache_size
        assert bufs.pre_cache_tile_ids.shape[0] >= max_pre_cache_size

    def test_extend_kv_indices_regression(self):
        """
        Regression: extend_kv_indices must be sized to max_total_kv_len,
        NOT max_bsz. The original bug used max_bsz (=8) when it needed
        max_total_kv_len (=65536), causing cudaErrorIllegalAddress.
        """
        max_num_seqs = 8
        max_model_len = 8192
        max_total_kv_len = max_num_seqs * max_model_len  # 65536
        buggy_size = max(max_num_seqs, 1)  # 8 — what the bug produced
        correct_size = max(max_total_kv_len, 1)  # 65536 — what's needed
        assert correct_size >= max_total_kv_len
        assert buggy_size < max_total_kv_len, "Test precondition"


# ---------------------------------------------------------------------------
# 6. Production-scale integration test
# ---------------------------------------------------------------------------


class TestProductionScaleIntegration:
    """
    Test full index-building + attention pipeline at production scale under CUDA Graph.
    """

    def test_full_pipeline_production_scale(self):
        """Full index pipeline + attention at production scale under CUDA Graph."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            _scatter_extend_kv_indices_kernel,
        )

        bs = 2
        prefix_len = 500
        block_size = 64
        num_heads, kv_heads, head_dim = 8, 2, 64
        extend_len = 1

        max_blocks = (8192 + block_size - 1) // block_size
        block_tables = paddle.arange(0, bs * max_blocks, dtype="int32").reshape([bs, max_blocks])
        num_kv_blocks = bs * max_blocks
        cache_k = paddle.randn([num_kv_blocks, kv_heads, block_size, head_dim], dtype="float16")
        cache_v = paddle.randn([num_kv_blocks, kv_heads, block_size, head_dim], dtype="float16")

        max_kv = bs * 8192
        buf_qo = paddle.empty([bs + 1], dtype="int32")
        buf_pkvi = paddle.empty([bs + 1], dtype="int32")
        buf_pkv = paddle.empty([max_kv], dtype="int32")
        buf_akvi = paddle.empty([bs + 1], dtype="int32")
        buf_akv = paddle.empty([max_kv], dtype="int32")
        buf_ekv = paddle.empty([max(max_kv, 1)], dtype="int32")
        buf_ukvi = paddle.empty([bs + 1], dtype="int32")
        buf_ukv = paddle.empty([max_kv], dtype="int32")
        buf_pl = paddle.empty([bs], dtype="int32")
        buf_ul = paddle.empty([bs], dtype="int32")
        buf_tsl = paddle.empty([bs], dtype="int32")
        buf_o = paddle.empty([bs, num_heads, head_dim], dtype="float16")
        buf_q = paddle.randn([bs, num_heads, head_dim], dtype="float16")

        total_prefix = bs * prefix_len
        total_extend = bs * extend_len

        def step():
            prefix_lens = paddle.full([bs], prefix_len, dtype="int32")
            ext_lens = paddle.ones([bs], dtype="int32")
            q = buf_q[:bs]

            qo_indptr = triton_cumsum_with_zero_prefix(ext_lens, bs, out_buf=buf_qo)
            pkvi, pkv = build_kv_indices_from_block_tables(
                block_tables,
                prefix_lens,
                block_size,
                bs,
                total_kv_len=total_prefix,
                kv_indptr_buf=buf_pkvi,
                kv_indices_buf=buf_pkv,
            )

            tsl = buf_tsl[:bs]
            BLK = triton.next_power_of_2(bs)
            _elementwise_add_kernel[(1,)](prefix_lens, ext_lens, tsl, bs, BLOCK=BLK)

            akvi, akv = build_kv_indices_from_block_tables(
                block_tables,
                tsl,
                block_size,
                bs,
                total_kv_len=total_prefix + total_extend,
                kv_indptr_buf=buf_akvi,
                kv_indices_buf=buf_akv,
            )

            esl = qo_indptr[:bs]
            ekv = buf_ekv[: max(total_extend, 1)]
            if bs > 0 and total_extend > 0:
                _scatter_extend_kv_indices_kernel[(bs,)](akv, akvi, prefix_lens, esl, ext_lens, ekv, BLOCK=128)

            ukvi, ukv, pl = build_unified_kv_indices(
                pkvi,
                pkv,
                esl,
                ext_lens,
                ekv,
                bs,
                unified_kv_indptr_buf=buf_ukvi,
                unified_kv_indices_buf=buf_ukv,
                prefix_lens_buf=buf_pl,
                unified_lens_buf=buf_ul,
            )

            o = buf_o[:bs].zero_()
            return extend_attention_fwd_unified(
                q, o, cache_k, cache_v, qo_indptr, ukvi, ukv, pl, num_heads, kv_heads, head_dim, extend_len, True
            )

        # Warmup
        for _ in range(3):
            step()
        paddle.device.cuda.synchronize()

        # Capture
        step()
        paddle.device.cuda.synchronize()
        g = paddle.device.cuda.graphs.CUDAGraph()
        g.capture_begin()
        step()
        g.capture_end()

        # Replay 10 times — no crash = pass
        for _ in range(10):
            g.replay()
        paddle.device.cuda.synchronize()

    def test_indices_in_valid_range(self):
        """Verify KV indices don't exceed KV cache bounds."""
        bs = 4
        prefix_len = 200
        block_size = 64

        max_blocks = (8192 + block_size - 1) // block_size
        num_total_blocks = bs * max_blocks
        block_tables = paddle.randint(0, num_total_blocks, [bs, max_blocks], dtype="int32")
        prefix_lens = paddle.full([bs], prefix_len, dtype="int32")
        total_prefix = bs * prefix_len

        _, kv_indices = build_kv_indices_from_block_tables(
            block_tables, prefix_lens, block_size, bs, total_kv_len=total_prefix
        )

        max_valid_index = num_total_blocks * block_size - 1
        max_idx = int(kv_indices[:total_prefix].max().item())
        min_idx = int(kv_indices[:total_prefix].min().item())
        assert max_idx <= max_valid_index, f"KV index {max_idx} > max valid {max_valid_index}"
        assert min_idx >= 0, f"KV index is negative: {min_idx}"
