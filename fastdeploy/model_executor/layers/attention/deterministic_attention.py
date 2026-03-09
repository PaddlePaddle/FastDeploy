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

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    gqa_rope_write_cache,
    pre_cache_len_concat,
)
from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
    _scatter_extend_kv_indices_kernel,
    build_kv_indices_from_block_tables,
    build_unified_kv_indices,
    extend_attention_fwd_unified,
    pre_cache_len_concat_triton,
    triton_cumsum_with_zero_prefix,
)


class DeterministicAttentionMixin:
    """
    Mixin providing deterministic attention for AppendAttentionBackend.

    Expects host class to provide via self:
        block_size, max_seq_len, num_heads, kv_num_heads, head_dim, causal, rope_3d
    """

    def _deterministic_build_triton_indices(self, forward_meta):
        """
        Build unified KV indices for Triton attention from block_tables.
        CUDA Graph compatible: all .item() replaced by pre-computed CPU scalars + Triton kernels.

        Handles ALL sequences in the batch (both prefill and decode):
        - Prefill sequences: extend_len = seq_lens_this_time (= seq_lens_encoder)
        - Decode sequences: extend_len = seq_lens_this_time (= 1)

        Returns: (qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len)
        """
        bs = forward_meta.deter_bs
        total_extend_len = forward_meta.deter_total_extend_len
        max_extend_len = forward_meta.deter_max_extend_len
        total_prefix_len = forward_meta.deter_total_prefix_len

        seq_lens_this_time = forward_meta.seq_lens_this_time
        prefix_lens = forward_meta.prefix_lens[:bs].astype("int32")
        extend_seq_lens = seq_lens_this_time[:bs]
        qo_indptr = triton_cumsum_with_zero_prefix(extend_seq_lens, bs)

        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            forward_meta.block_tables,
            prefix_lens,
            self.block_size,
            bs,
            total_kv_len=total_prefix_len,
        )
        total_seq_lens = prefix_lens + extend_seq_lens
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(
            forward_meta.block_tables,
            total_seq_lens,
            self.block_size,
            bs,
            total_kv_len=total_prefix_len + total_extend_len,
        )

        # extend_start_loc is the exclusive prefix sum = qo_indptr[:bs]
        extend_start_loc = qo_indptr[:bs]

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
        )

        return qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len

    def _deterministic_build_triton_indices_ref(self, forward_meta):
        """
        Reference implementation (Python for-loop + .item()).
        NOT compatible with CUDA Graph capture — kept for correctness validation.
        """
        seq_lens_this_time = forward_meta.seq_lens_this_time
        bs = int((seq_lens_this_time > 0).sum().item())

        prefix_lens = forward_meta.prefix_lens[:bs].astype("int32")
        extend_seq_lens = seq_lens_this_time[:bs]
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
            forward_meta.block_tables,
            prefix_lens,
            self.block_size,
            bs,
        )
        total_seq_lens = prefix_lens + extend_seq_lens
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables_ref(
            forward_meta.block_tables,
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
        # Use Triton GPU-only version when CUDA Graph is active (no D2H copy).
        # CPU scalars come from forward_meta (pre-computed outside capture region).
        if forward_meta.step_use_cudagraph:
            bsz = forward_meta.seq_lens_this_time.shape[0]
            max_dec_len = int(forward_meta.max_len_tensor_cpu[2])
            max_tile_per_bs = (max_dec_len + self.block_size - 1) // self.block_size
            cu_seqlens_k, pre_cache_batch_ids, pre_cache_tile_ids_per_batch = pre_cache_len_concat_triton(
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                bsz,
                self.block_size,
                max_tile_per_bs,
            )
            # Build CPU tensors directly (no D2H copy), needed by gqa_rope_write_cache C++ op
            pre_cache_num_blocks_cpu = paddle.to_tensor(
                [forward_meta.deter_pre_cache_num_blocks], dtype="int32", place=paddle.CPUPlace()
            )
            kv_token_num = forward_meta.deter_kv_token_num
        else:
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

        # --- Step 2: RoPE + KV cache write (no attention) ---
        # DIAG: log gqa_rope_write_cache metadata for Layer 0
        # Enable with: export FD_DIAG_ATTN=1
        if os.environ.get("FD_DIAG_ATTN", "0") == "1" and layer.layer_id == 0:
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
        q_roped, _k_flat, _v_flat, _qkv_out = gqa_rope_write_cache(
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
        # q_roped: [token_nums, num_heads, head_dim] with RoPE already applied

        # --- DIAG: Layer 0 — 4 return values + paged cache ---
        # Enable with: export FD_DIAG_ATTN=1
        if os.environ.get("FD_DIAG_ATTN", "0") == "1" and layer.layer_id == 0:
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
        (qo_indptr, unified_kv_indptr, unified_kv_indices, prefix_lens, bs, max_extend_len) = (
            self._deterministic_build_triton_indices(forward_meta)
        )

        token_nums = q_roped.shape[0]
        o = paddle.zeros([token_nums, self.num_heads, self.head_dim], dtype=q_roped.dtype)
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

        # --- DIAG: Layer 0 attention output ---
        # Enable with: export FD_DIAG_ATTN=1
        if os.environ.get("FD_DIAG_ATTN", "0") == "1" and layer.layer_id == 0:
            _wlog.info(f"[DIAG-L0] prefix_lens={prefix_lens.cpu().numpy().tolist()} extend_len={max_extend_len}")
            _wlog.info(f"[DIAG-L0] attn_out[-1] md5={self._diag_md5(res[-1:])}")
            _wlog.info(f"[DIAG-L0] attn_out_all md5={self._diag_md5(res)}")

        return res
