"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
DeterministicAttentionMixin: split-invariant Triton unified attention path.

Extracted from AppendAttentionBackend to keep the original forward_mixed path untouched.
Mixed into AppendAttentionBackend via multiple inheritance; methods access host attributes
(block_size, num_heads, kv_num_heads, head_dim, max_seq_len, causal, rope_3d) through self.
"""

import os
from dataclasses import dataclass
from typing import Optional

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    gqa_rope_write_cache,
    pre_cache_len_concat,
)
from fastdeploy.model_executor.layers.attention.triton_ops.rope_and_cache_write import (
    triton_rope_and_cache_write,
)
from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
    _elementwise_add_kernel,
    _scatter_extend_kv_indices_kernel,
    build_kv_indices_from_block_tables,
    build_unified_kv_indices,
    extend_attention_fwd_unified,
    triton_cumsum_with_zero_prefix,
)
from fastdeploy.utils import get_logger

logger = get_logger("deterministic_attention", "deterministic_attention.log")


@dataclass
class DeterministicCudaGraphBuffers:
    """Pre-allocated GPU buffers for CUDA Graph-compatible deterministic attention."""

    # index building
    qo_indptr: Optional[paddle.Tensor] = None
    prefix_kv_indptr: Optional[paddle.Tensor] = None
    prefix_kv_indices: Optional[paddle.Tensor] = None
    all_kv_indptr: Optional[paddle.Tensor] = None
    all_kv_indices: Optional[paddle.Tensor] = None
    extend_kv_indices: Optional[paddle.Tensor] = None
    unified_kv_indptr: Optional[paddle.Tensor] = None
    unified_kv_indices: Optional[paddle.Tensor] = None
    prefix_lens_buf: Optional[paddle.Tensor] = None
    unified_lens_buf: Optional[paddle.Tensor] = None
    total_seq_lens_buf: Optional[paddle.Tensor] = None
    # attention output
    output: Optional[paddle.Tensor] = None
    # q_roped buffer: Triton rope writes here at fixed address for CUDA Graph
    q_roped: Optional[paddle.Tensor] = None


class DeterministicAttentionMixin:
    """
    Mixin providing deterministic attention for AppendAttentionBackend.

    Expects host class to provide via self:
        block_size, max_seq_len, num_heads, kv_num_heads, head_dim, causal, rope_3d, fd_config
    """

    def _init_cudagraph_buffers(self):
        """Pre-allocate GPU buffers at max sizes for CUDA Graph compatibility.
        Called lazily on first step_use_cudagraph=True forward."""

        max_bsz = self.fd_config.scheduler_config.max_num_seqs
        max_model_len = self.max_seq_len
        max_total_kv_len = max(max_bsz * max_model_len, 1)
        max_capture_size = self.fd_config.graph_opt_config.max_capture_size

        bufs = DeterministicCudaGraphBuffers()
        # Index building buffers (always needed)
        bufs.qo_indptr = paddle.empty([max_bsz + 1], dtype="int32")
        bufs.prefix_kv_indptr = paddle.empty([max_bsz + 1], dtype="int32")
        bufs.prefix_kv_indices = paddle.empty([max_total_kv_len], dtype="int32")
        bufs.all_kv_indptr = paddle.empty([max_bsz + 1], dtype="int32")
        bufs.all_kv_indices = paddle.empty([max_total_kv_len], dtype="int32")
        bufs.extend_kv_indices = paddle.empty([max(max_total_kv_len, 1)], dtype="int32")
        bufs.unified_kv_indptr = paddle.empty([max_bsz + 1], dtype="int32")
        bufs.unified_kv_indices = paddle.empty([max_total_kv_len], dtype="int32")
        bufs.prefix_lens_buf = paddle.empty([max_bsz], dtype="int32")
        bufs.unified_lens_buf = paddle.empty([max_bsz], dtype="int32")
        bufs.total_seq_lens_buf = paddle.empty([max_bsz], dtype="int32")
        # q_roped and output buffers (always needed)
        compute_dtype = paddle.get_default_dtype()
        bufs.q_roped = paddle.empty([max_capture_size, self.num_heads, self.head_dim], dtype=compute_dtype)
        bufs.output = paddle.empty([max_capture_size, self.num_heads, self.head_dim], dtype=compute_dtype)

        # Estimate total allocated bytes
        elem_size = 2  # bfloat16
        total_bytes = (
            7 * (max_bsz + 1) * 4
            + 5 * max_bsz * 4
            + 5 * max_total_kv_len * 4
            + 2 * max_capture_size * self.num_heads * self.head_dim * elem_size
        )
        logger.info(
            f"[DeterministicAttention] Pre-allocated CUDA Graph buffers: "
            f"{total_bytes / 1024 / 1024:.1f} MB "
            f"(max_bsz={max_bsz}, max_kv_len={max_total_kv_len}, "
            f"max_capture={max_capture_size})"
        )
        self._cudagraph_bufs = bufs

    def _deterministic_build_triton_indices(self, forward_meta, bufs=None):
        """
        Build unified KV indices for Triton attention from block_tables.
        CUDA Graph compatible: all .item() replaced by pre-computed CPU scalars + Triton kernels.

        Args:
            bufs: Optional DeterministicCudaGraphBuffers for CUDA Graph compatibility.
                  When None, tensors are allocated dynamically.

        Returns: (qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len)
        """
        import triton

        bs = forward_meta.deter_bs
        total_extend_len = forward_meta.deter_total_extend_len
        max_extend_len = forward_meta.deter_max_extend_len
        total_prefix_len = forward_meta.deter_total_prefix_len

        # Active slots may not be contiguous (e.g. slot 0 done, slot 1 still decoding).
        # Use integer indices to compact active slots (avoids boolean mask shape mismatch).
        slt = forward_meta.seq_lens_this_time
        if bufs is not None:
            # CUDA Graph path: slots are padded to max_num_seqs, always contiguous
            extend_seq_lens = slt[:bs]
            prefix_lens = forward_meta.prefix_lens[:bs].astype("int32")
            block_tables = forward_meta.block_tables
        else:
            active_idx = paddle.nonzero(slt > 0).flatten()[:bs]
            extend_seq_lens = slt[active_idx]
            prefix_lens = forward_meta.prefix_lens[active_idx].astype("int32")
            block_tables = forward_meta.block_tables[active_idx]
        qo_indptr = triton_cumsum_with_zero_prefix(
            extend_seq_lens,
            bs,
            out_buf=bufs.qo_indptr if bufs else None,
        )

        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            block_tables,
            prefix_lens,
            self.block_size,
            bs,
            total_kv_len=total_prefix_len,
            kv_indptr_buf=bufs.prefix_kv_indptr if bufs else None,
            kv_indices_buf=bufs.prefix_kv_indices if bufs else None,
        )

        # Compute total_seq_lens = prefix_lens + extend_seq_lens (allocation-free when bufs)
        if bufs is not None and bs > 0:
            total_seq_lens = bufs.total_seq_lens_buf[:bs]
            BLOCK = triton.next_power_of_2(bs)
            _elementwise_add_kernel[(1,)](prefix_lens, extend_seq_lens, total_seq_lens, bs, BLOCK=BLOCK)
        else:
            total_seq_lens = prefix_lens + extend_seq_lens

        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(
            block_tables,
            total_seq_lens,
            self.block_size,
            bs,
            total_kv_len=total_prefix_len + total_extend_len,
            kv_indptr_buf=bufs.all_kv_indptr if bufs else None,
            kv_indices_buf=bufs.all_kv_indices if bufs else None,
        )

        extend_start_loc = qo_indptr[:bs]

        if bufs is not None:
            extend_kv_indices = bufs.extend_kv_indices[: max(total_extend_len, 1)]
        else:
            extend_kv_indices = paddle.empty([max(total_extend_len, 1)], dtype="int32")

        if bs > 0 and total_extend_len > 0:
            _scatter_extend_kv_indices_kernel[(bs,)](
                all_kv_indices,
                all_kv_indptr,
                prefix_lens,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                BLOCK=128,
            )

        unified_kv_indptr, unified_kv_indices, _ = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
            unified_kv_indptr_buf=bufs.unified_kv_indptr if bufs else None,
            unified_kv_indices_buf=bufs.unified_kv_indices if bufs else None,
            prefix_lens_buf=bufs.prefix_lens_buf if bufs else None,
            unified_lens_buf=bufs.unified_lens_buf if bufs else None,
        )

        return qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len

    def _deterministic_build_triton_indices_ref(self, forward_meta):
        """
        Reference implementation (Python for-loop + .item()).
        NOT compatible with CUDA Graph capture — kept for correctness validation.
        """
        seq_lens_this_time = forward_meta.seq_lens_this_time
        active_idx = paddle.nonzero(seq_lens_this_time > 0).flatten()
        bs = active_idx.shape[0]

        # Compact active slots (may not be contiguous)
        prefix_lens = forward_meta.prefix_lens[active_idx].astype("int32")
        extend_seq_lens = seq_lens_this_time[active_idx]
        block_tables = forward_meta.block_tables[active_idx]
        qo_indptr = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(extend_seq_lens).astype("int32"),
            ]
        )

        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables_ref,
        )

        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables_ref(
            block_tables,
            prefix_lens,
            self.block_size,
            bs,
        )
        total_seq_lens = prefix_lens + extend_seq_lens
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables_ref(
            block_tables,
            total_seq_lens,
            self.block_size,
            bs,
        )

        extend_start_loc = (
            paddle.concat(
                [
                    paddle.zeros([1], dtype="int32"),
                    paddle.cumsum(extend_seq_lens[:-1]).astype("int32"),
                ]
            )
            if bs > 1
            else paddle.zeros([1], dtype="int32")
        )

        total_extend_len = int(paddle.sum(extend_seq_lens).item())
        extend_kv_indices = paddle.empty([max(total_extend_len, 1)], dtype="int32")
        for s in range(bs):
            plen = int(prefix_lens[s].item())
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
            bs,
        )

        max_extend_len = int(paddle.max(extend_seq_lens).item())
        return qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len

    @staticmethod
    def _diag_md5(tensor, n=16):
        """Quick MD5 for diagnostics (first n hex chars)."""
        import hashlib

        data = tensor.cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()[:n]

    @staticmethod
    def _diag_cache_block_md5(cache_k, block_tables, seq_idx, block_idx, block_size):
        """MD5 of a specific KV cache block."""
        import hashlib

        blk_id = int(block_tables[seq_idx][block_idx].item())
        data = cache_k[blk_id].cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()[:16]

    def _deterministic_rope_kv_write(self, qkv, cache_k, cache_v, layer, forward_meta, metadata):
        """Bisect helper: only pre_cache_len_concat + gqa_rope_write_cache (KV cache write), no attention."""
        norm_after_rope_in_kernel = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope_in_kernel else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope_in_kernel else None
        cache_k_scales = getattr(layer, "cache_k_scale", None)
        cache_v_scales = getattr(layer, "cache_v_scale", None)

        (
            cu_seqlens_k,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            kv_token_num_cpu,
        ) = pre_cache_len_concat(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.max_len_tensor_cpu[2],
            self.block_size,
        )
        kv_token_num = kv_token_num_cpu[0].item()

        gqa_rope_write_cache(
            qkv,
            cache_k,
            cache_v,
            forward_meta.cu_seqlens_q,
            cu_seqlens_k,
            forward_meta.rotary_embs,
            forward_meta.seq_lens_this_time,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.batch_id_per_token,
            forward_meta.block_tables,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            q_norm_weight,
            k_norm_weight,
            cache_k_scales,
            cache_v_scales,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            metadata.kv_signal_data_list[layer.layer_id],
            kv_token_num,
            self.max_seq_len,
            getattr(layer, "rms_norm_eps", 1e-6),
            layer.use_neox_rotary_style,
            getattr(layer, "cache_quant_type_str", "none"),
            self.rope_3d,
        )

        # DIAG: snapshot KV cache after gqa_rope_write_cache, compare after append_attention overwrites
        if os.environ.get("FD_DIAG_KV_CMP", "0") == "1" and layer.layer_id == 0:
            bt = forward_meta.block_tables
            blk0 = int(bt[0][0].item())
            # Snapshot block 0, head 0 of cache_k after gqa_rope_write_cache
            self._kv_snapshot = cache_k[blk0, 0, :, :4].clone().cpu().float().numpy()
            self._kv_snapshot_blk = blk0

    def _deterministic_forward(self, qkv, cache_k, cache_v, layer, forward_meta, metadata):
        """
        Unified deterministic path for ALL tokens (prefill + decode).

        Uses the same pattern as FlashAttentionBackend:
          1. pre_cache_len_concat + gqa_rope_write_cache -> RoPE + KV cache write (all tokens)
          2. Triton unified attention -> split-invariant attention over paged cache (all tokens)

        All tokens (prefill and decode) go through the same Triton kernel,
        eliminating path divergence and the need for merge_prefill_decode_output.
        """
        norm_after_rope_in_kernel = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope_in_kernel else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope_in_kernel else None

        cache_k_scales = getattr(layer, "cache_k_scale", None)
        cache_v_scales = getattr(layer, "cache_v_scale", None)

        # --- Step 1: Prepare metadata for gqa_rope_write_cache ---
        # BUG FIX: gqa_rope_write_cache C++ kernel skips tokens where seq_lens_encoder==0
        # (designed for the split prefill/decode path in append_attention).
        # In deterministic mode, ALL tokens go through this unified path, so decode tokens
        # (enc==0) must appear as 1-token prefill to receive correct RoPE and KV cache writes.
        enc = forward_meta.seq_lens_encoder
        seq_lens_encoder_for_rope = paddle.where(enc == 0, paddle.ones_like(enc), enc)

        # Check Triton rope eligibility early to skip unnecessary pre_cache_len_concat
        _use_triton_rope = os.environ.get("FD_USE_TRITON_ROPE", "1") == "1"
        _triton_rope_eligible = (
            _use_triton_rope
            and forward_meta.step_use_cudagraph
            and q_norm_weight is None  # no QK-norm
            and getattr(layer, "cache_quant_type_str", "none") == "none"  # no cache quant
            and not self.rope_3d  # no rope_3d
        )

        # Use Triton GPU-only version when CUDA Graph is active (no D2H copy).
        # CPU scalars come from forward_meta (pre-computed outside capture region).
        if forward_meta.step_use_cudagraph:
            # Lazy init CUDA Graph buffers on first use
            if not hasattr(self, "_cudagraph_bufs"):
                self._init_cudagraph_buffers()
            bufs = self._cudagraph_bufs

            if not _triton_rope_eligible:
                # cudagraph + Triton not eligible: C++ inplace crashes at replay.
                # Fail fast here before wasting compute on pre_cache metadata.
                raise AssertionError(
                    "Deterministic + CUDA Graph requires Triton RoPE, but current config "
                    "is not eligible. Possible causes: QK-norm enabled, cache quantization "
                    "enabled, rope_3d enabled, or FD_USE_TRITON_ROPE=0. "
                    "Fix: disable cudagraph (--cudagraph 0) or disable deterministic mode."
                )
        else:
            (
                cu_seqlens_k,
                pre_cache_batch_ids,
                pre_cache_tile_ids_per_batch,
                pre_cache_num_blocks_cpu,
                kv_token_num_cpu,
            ) = pre_cache_len_concat(
                seq_lens_encoder_for_rope,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.max_len_tensor_cpu[2],
                self.block_size,
            )
            kv_token_num = kv_token_num_cpu[0].item()

        # --- Step 2: RoPE + KV cache write ---
        # DIAG: log gqa_rope_write_cache metadata for Layer 0 (non-cudagraph only)
        # Enable with: export FD_DIAG_ATTN=1
        if os.environ.get("FD_DIAG_ATTN", "0") == "1" and layer.layer_id == 0 and not forward_meta.step_use_cudagraph:
            from fastdeploy.utils import get_logger as _gl

            _wlog = _gl("worker_process", "worker_process.log")
            _wlog.info(f"[DIAG-ROPE] cu_seqlens_q={forward_meta.cu_seqlens_q.cpu().numpy().tolist()}")
            _wlog.info(f"[DIAG-ROPE] cu_seqlens_k={cu_seqlens_k.cpu().numpy().tolist()}")
            _wlog.info(f"[DIAG-ROPE] seq_lens_this_time={forward_meta.seq_lens_this_time.cpu().numpy().tolist()}")
            _wlog.info(f"[DIAG-ROPE] seq_lens_encoder={forward_meta.seq_lens_encoder.cpu().numpy().tolist()}")
            _wlog.info(f"[DIAG-ROPE] seq_lens_decoder={forward_meta.seq_lens_decoder.cpu().numpy().tolist()}")
            _wlog.info(
                f"[DIAG-ROPE] rotary_embs shape={list(forward_meta.rotary_embs.shape)} md5={self._diag_md5(forward_meta.rotary_embs[-57:])}"
            )
            _wlog.info(f"[DIAG-ROPE] block_tables[0]={forward_meta.block_tables[0, :15].cpu().numpy().tolist()}")

        # --- Bisect skip for cudagraph debugging ---
        # Usage: FD_DETER_GRAPH_SKIP=rope+index+attn (any combination)
        _graph_skip = os.environ.get("FD_DETER_GRAPH_SKIP", "")
        _skip_rope = "rope" in _graph_skip
        _skip_index = "index" in _graph_skip
        _skip_attn = "attn" in _graph_skip

        if _skip_rope:
            # Dummy output for bisect: no rope, no KV cache write
            token_nums = qkv.shape[0]
            q_roped = paddle.zeros([token_nums, self.num_heads, self.head_dim], dtype=qkv.dtype)
        elif _triton_rope_eligible:
            # Triton path: fused RoPE + paged cache write, no k_buf/v_buf/qkv_out_buf needed
            token_nums = qkv.shape[0]
            assert (
                token_nums <= bufs.q_roped.shape[0]
            ), f"token_nums={token_nums} > q_roped buf={bufs.q_roped.shape[0]}"
            triton_rope_and_cache_write(
                qkv,
                cache_k,
                cache_v,
                bufs.q_roped,
                forward_meta.rotary_embs,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                seq_lens_encoder_for_rope,
                forward_meta.seq_lens_decoder,
                forward_meta.block_tables,
                self.num_heads,
                self.kv_num_heads,
                self.head_dim,
                self.block_size,
                use_neox_rotary_style=layer.use_neox_rotary_style,
            )
            q_roped = bufs.q_roped[:token_nums]
        elif forward_meta.step_use_cudagraph:
            # gqa_rope_write_cache_inplace crashes at CUDA Graph replay
            # (temporary tensor GC, cudaLaunchHostFunc incompatibility, etc.
            #  see docs/cudagraph_rope_inplace_fix.md Section 4).
            # Triton rope is the only safe path for cudagraph. If not eligible,
            # fail fast instead of silently crashing at replay time.
            raise AssertionError(
                "Deterministic + CUDA Graph requires Triton RoPE, but current config "
                "is not eligible. Possible causes: QK-norm enabled, cache quantization "
                "enabled, rope_3d enabled, or FD_USE_TRITON_ROPE=0. "
                "Fix: disable cudagraph (--cudagraph 0) or disable deterministic mode."
            )
        else:
            # Non-cudagraph: C++ gqa_rope_write_cache (dynamic alloc, safe)
            q_roped, _k_flat, _v_flat, _qkv_out = gqa_rope_write_cache(
                qkv=qkv,
                key_cache=cache_k,
                value_cache=cache_v,
                cu_seqlens_q=forward_meta.cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                rotary_embs=forward_meta.rotary_embs,
                seq_lens_this_time=forward_meta.seq_lens_this_time,
                seq_lens_encoder=seq_lens_encoder_for_rope,
                seq_lens_decoder=forward_meta.seq_lens_decoder,
                batch_id_per_token=forward_meta.batch_id_per_token,
                block_tables=forward_meta.block_tables,
                kv_batch_ids=forward_meta.kv_batch_ids,
                kv_tile_ids_per_batch=forward_meta.kv_tile_ids_per_batch,
                kv_num_blocks=forward_meta.kv_num_blocks_x_cpu,
                cache_batch_ids=pre_cache_batch_ids,
                cache_tile_ids_per_batch=pre_cache_tile_ids_per_batch,
                cache_num_blocks=pre_cache_num_blocks_cpu,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                cache_k_quant_scales=cache_k_scales,
                cache_v_quant_scales=cache_v_scales,
                cache_k_dequant_scales=getattr(layer, "cache_k_out_scale", None),
                cache_v_dequant_scales=getattr(layer, "cache_v_out_scale", None),
                cache_k_zp=getattr(layer, "cache_k_zp", None),
                cache_v_zp=getattr(layer, "cache_v_zp", None),
                kv_signal_data=metadata.kv_signal_data_list[layer.layer_id],
                kv_token_num=kv_token_num,
                max_seq_len=self.max_seq_len,
                rms_norm_eps=getattr(layer, "rms_norm_eps", 1e-6),
                use_neox_rotary_style=layer.use_neox_rotary_style,
                cache_quant_type=getattr(layer, "cache_quant_type_str", "none"),
                rope_3d=self.rope_3d,
            )

        # --- DIAG: Layer 0 — 4 return values + paged cache (C++ path only) ---
        # Enable with: export FD_DIAG_ATTN=1
        _is_cpp_path = not (_skip_rope or _triton_rope_eligible or forward_meta.step_use_cudagraph)
        if os.environ.get("FD_DIAG_ATTN", "0") == "1" and layer.layer_id == 0 and _is_cpp_path:
            from fastdeploy.utils import get_logger as _gl

            _wlog = _gl("worker_process", "worker_process.log")
            enc = int(forward_meta.seq_lens_encoder[0].item())
            dec = int(forward_meta.seq_lens_decoder[0].item())
            _n = min(57, qkv.shape[0])
            _wlog.info(f"[DIAG-L0] enc={enc} dec={dec} token_num={qkv.shape[0]} kv_token_num={_k_flat.shape[0]}")
            # Input QKV: all new tokens
            _wlog.info(f"[DIAG-L0] qkv[-1] md5={self._diag_md5(qkv[-1:])}")
            _wlog.info(f"[DIAG-L0] qkv[-{_n}:] md5={self._diag_md5(qkv[-_n:])}")
            # Return value 1: q_roped (RoPE'd Q)
            _wlog.info(f"[DIAG-L0] q_roped[-1] md5={self._diag_md5(q_roped[-1:])}")
            _wlog.info(f"[DIAG-L0] q_roped[-{_n}:] md5={self._diag_md5(q_roped[-_n:])}")
            # Return value 2: _k_flat (flat K buffer, shape [kv_token_num, kv_heads, head_dim])
            _wlog.info(f"[DIAG-L0] _k_flat shape={list(_k_flat.shape)} _k_flat[-1] md5={self._diag_md5(_k_flat[-1:])}")
            _wlog.info(f"[DIAG-L0] _k_flat[-{_n}:] md5={self._diag_md5(_k_flat[-_n:])}")
            # Return value 4: _qkv_out (RoPE'd full QKV, same shape as input qkv)
            _wlog.info(
                f"[DIAG-L0] _qkv_out shape={list(_qkv_out.shape)} _qkv_out[-1] md5={self._diag_md5(_qkv_out[-1:])}"
            )
            _wlog.info(f"[DIAG-L0] _qkv_out[-{_n}:] md5={self._diag_md5(_qkv_out[-_n:])}")
            # Paged cache: block 12 valid entries
            bt = forward_meta.block_tables
            total_kv = enc + dec
            num_blocks_used = (total_kv + self.block_size - 1) // self.block_size
            last_bi = num_blocks_used - 1
            last_blk = int(bt[0][last_bi].item())
            valid_n = total_kv - last_bi * self.block_size
            _wlog.info(
                f"[DIAG-L0] cache_k last_block blk_idx={last_bi} blk_id={last_blk} "
                f"valid={valid_n}/{self.block_size} "
                f"valid_md5={self._diag_md5(cache_k[last_blk, :, :valid_n, :])}"
            )
            # Also check block 0 and 11 for prefix consistency
            _wlog.info(f"[DIAG-L0] cache_k blk0 md5={self._diag_md5(cache_k[int(bt[0][0].item())])}")
            if num_blocks_used > 11:
                _wlog.info(f"[DIAG-L0] cache_k blk11 md5={self._diag_md5(cache_k[int(bt[0][11].item())])}")

        # --- Step 3: Triton unified attention for all tokens (prefill + decode) ---
        if not _skip_index:
            if forward_meta.step_use_cudagraph:
                (qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len) = (
                    self._deterministic_build_triton_indices(forward_meta, bufs=bufs)
                )
                token_nums = q_roped.shape[0]
                # Lazy-allocate output buffer with correct dtype on first use
                if bufs.output is None or bufs.output.dtype != q_roped.dtype:
                    max_capture_size = self.fd_config.graph_opt_config.max_capture_size
                    bufs.output = paddle.empty([max_capture_size, self.num_heads, self.head_dim], dtype=q_roped.dtype)
                o = bufs.output[:token_nums].zero_()
            else:
                if "use_ref_indices" in os.environ.get("FD_OVERLAP_DIAG", ""):
                    (qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len) = (
                        self._deterministic_build_triton_indices_ref(forward_meta)
                    )
                else:
                    (qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len) = (
                        self._deterministic_build_triton_indices(forward_meta)
                    )
            # DIAG: compare indices with reference implementation (Layer 0 only)
            if os.environ.get("FD_DIAG_INDEX", "0") == "1" and layer.layer_id == 0:
                (qo_ref, kv_indptr_ref, kv_indices_ref, plen_ref, bs_ref, mel_ref) = (
                    self._deterministic_build_triton_indices_ref(forward_meta)
                )
                import numpy as np

                _eq = lambda a, b, name: logger.info(
                    f"[DIAG-IDX] {name}: match={bool(np.array_equal(a.cpu().numpy(), b.cpu().numpy()))} "
                    f"len={len(a)}/{len(b)} "
                    f"triton={a[:min(10,len(a))].cpu().numpy().tolist()} ref={b[:min(10,len(b))].cpu().numpy().tolist()}"
                )
                _eq(qo_indptr[: bs + 1], qo_ref[: bs_ref + 1], "qo_indptr")
                _eq(unified_kv_indptr[: bs + 1], kv_indptr_ref[: bs_ref + 1], "unified_kv_indptr")
                # Show exact mismatch position for kv_indices
                tri_np = unified_kv_indices.cpu().numpy()
                ref_np = kv_indices_ref.cpu().numpy()
                n = min(len(tri_np), len(ref_np))
                diff_mask = tri_np[:n] != ref_np[:n]
                if diff_mask.any():
                    first_diff = int(np.argmax(diff_mask))
                    logger.info(
                        f"[DIAG-IDX] unified_kv_indices: MISMATCH at pos={first_diff} "
                        f"triton_len={len(tri_np)} ref_len={len(ref_np)} "
                        f"triton[{first_diff}]={tri_np[first_diff]} ref[{first_diff}]={ref_np[first_diff]} "
                        f"total_diff={int(diff_mask.sum())}"
                    )
                elif len(tri_np) != len(ref_np):
                    logger.info(
                        f"[DIAG-IDX] unified_kv_indices: prefix match but LEN DIFF "
                        f"triton_len={len(tri_np)} ref_len={len(ref_np)}"
                    )
                else:
                    logger.info(f"[DIAG-IDX] unified_kv_indices: EXACT MATCH len={len(tri_np)}")
                _eq(prefix_lens[:bs], plen_ref[:bs_ref], "prefix_lens")
                logger.info(f"[DIAG-IDX] bs={bs}/{bs_ref} max_extend_len={max_extend_len}/{mel_ref}")

            token_nums = q_roped.shape[0]
            o = paddle.zeros([token_nums, self.num_heads, self.head_dim], dtype=q_roped.dtype)

        if _skip_index or _skip_attn:
            # Bisect: skip index/attention, return dummy output
            token_nums = q_roped.shape[0]
            res = paddle.zeros([token_nums, self.num_heads * self.head_dim], dtype=q_roped.dtype)
        else:
            # DIAG: lightweight metadata log for Layer 0
            if os.environ.get("FD_DIAG_ATTN", "0") == "1" and layer.layer_id == 0:
                _pnp = prefix_lens[:bs].cpu().numpy().tolist() if bs > 0 else []
                _kv_lens = (
                    (unified_kv_indptr[1 : bs + 1] - unified_kv_indptr[:bs]).cpu().numpy().tolist() if bs > 0 else []
                )
                _wlog.info(
                    f"[DIAG-L0] bs={bs} mel={max_extend_len} plen={_pnp} kv_len={_kv_lens} tokens={q_roped.shape[0]}"
                )
            res = extend_attention_fwd_unified(
                q_roped,
                o,
                cache_k,
                cache_v,
                qo_indptr,
                unified_kv_indptr,
                unified_kv_indices,
                prefix_lens,
                self.num_heads,
                self.kv_num_heads,
                self.head_dim,
                max_extend_len,
                self.causal,
            ).reshape([-1, self.num_heads * self.head_dim])

        # DIAG: naive attention comparison (Layer 0, first seq only)
        if os.environ.get("FD_DIAG_ATTN_CMP", "0") == "1" and layer.layer_id == 0 and bs > 0:
            import numpy as np

            q_start = int(qo_indptr[0].item())
            q_len = int(qo_indptr[1].item()) - q_start
            kv_start = int(unified_kv_indptr[0].item())
            kv_len = int(unified_kv_indptr[1].item()) - kv_start
            plen = int(prefix_lens[0].item())
            kv_idx = unified_kv_indices[kv_start : kv_start + kv_len].cpu().numpy()
            sm_scale = 1.0 / (self.head_dim**0.5)
            # Pick head 0 for comparison
            h = 0
            kv_h = h // (self.num_heads // self.kv_num_heads)
            q_vec = q_roped[q_start : q_start + q_len, h, :].cpu().float().numpy()  # [q_len, D]
            # Gather K,V from paged cache using kv_indices
            block_size = cache_k.shape[2]
            k_list, v_list = [], []
            for idx in kv_idx:
                blk_id = idx // block_size
                off = idx % block_size
                k_list.append(cache_k[blk_id, kv_h, off, :].cpu().float().numpy())
                v_list.append(cache_v[blk_id, kv_h, off, :].cpu().float().numpy())
            if not k_list:
                logger.info("[DIAG-ATTN-CMP] seq0 head0: SKIP (kv_len=0)")
            else:
                k_mat = np.stack(k_list)  # [kv_len, D]
                v_mat = np.stack(v_list)  # [kv_len, D]
                # QK^T
                scores = q_vec @ k_mat.T * sm_scale  # [q_len, kv_len]
                # Causal mask
                if self.causal:
                    for qi in range(q_len):
                        for ki in range(kv_len):
                            if ki >= plen and (ki - plen) > qi:
                                scores[qi, ki] = -1e20
                # Softmax
                scores_max = scores.max(axis=-1, keepdims=True)
                scores_exp = np.exp(scores - scores_max)
                scores_sum = scores_exp.sum(axis=-1, keepdims=True)
                attn_weights = scores_exp / scores_sum
                naive_out = attn_weights @ v_mat  # [q_len, D]
                # Compare with Triton output
                triton_out = (
                    res[q_start : q_start + q_len, h * self.head_dim : (h + 1) * self.head_dim].cpu().float().numpy()
                )
                diff = np.abs(naive_out - triton_out)
                logger.info(
                    f"[DIAG-ATTN-CMP] seq0 head0: q_len={q_len} kv_len={kv_len} plen={plen} "
                    f"max_diff={diff.max():.6f} mean_diff={diff.mean():.6f} "
                    f"naive_norm={np.linalg.norm(naive_out):.4f} triton_norm={np.linalg.norm(triton_out):.4f}"
                )
                if diff.max() > 0.01:
                    logger.info(f"[DIAG-ATTN-CMP] naive[0,:4]={naive_out[0,:4]} triton[0,:4]={triton_out[0,:4]}")
                    logger.info(f"[DIAG-ATTN-CMP] naive[-1,:4]={naive_out[-1,:4]} triton[-1,:4]={triton_out[-1,:4]}")
        # --- DIAG: Layer 0 attention output ---
        # Enable with: export FD_DIAG_ATTN=1
        if os.environ.get("FD_DIAG_ATTN", "0") == "1" and layer.layer_id == 0:
            _wlog.info(f"[DIAG-L0] prefix_lens={prefix_lens.cpu().numpy().tolist()} extend_len={max_extend_len}")
            _wlog.info(f"[DIAG-L0] attn_out[-1] md5={self._diag_md5(res[-1:])}")
            _wlog.info(f"[DIAG-L0] attn_out_all md5={self._diag_md5(res)}")

        return res
