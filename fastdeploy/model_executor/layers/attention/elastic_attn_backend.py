"""``Qwen3ElasticAttentionBackend`` -- prefill goes through Block-Sparse-Attention,
decode keeps ``append_attention``.

This is a thin subclass of :class:`FlashAttentionBackend` that replaces the
prefill leg of :py:meth:`forward_mixed` with the Elastic-Attention path:

  ``gqa_rope_write_cache`` → router decision (per layer) → repeat_interleave →
  ``Xattention_prefill_dim4`` (Triton + BSA).

Decode leg, ``merge_prefill_decode_output`` and all of ``init_attention_metadata``
are reused unchanged from the parent, so chunked prefill / continuous batching
work with no extra wiring.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import paddle

from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.flash_attn_backend import (
    FLASH_ATTN_VERSION,
    FlashAttentionBackend,
)
from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    get_attn_mask_q,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_signal_layerwise,
    pre_cache_len_concat,
)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import merge_prefill_decode_output
else:  # pragma: no cover
    merge_prefill_decode_output = None

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


class Qwen3ElasticAttentionBackend(FlashAttentionBackend):
    """Elastic-Attention backend for PawQwen3.

    The prefill leg is overridden to call ``Xattention_prefill_dim4`` (Triton
    estimator + Block-Sparse-Attention CUDA kernel) rather than dense
    ``flash_attn_func``.  Decode leg, KV-cache shape and PD signals are inherited.
    """

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: "ForwardMeta",
    ):
        # Lazy imports to avoid circular import + to keep BSA-build optional
        # for the rest of the package. Note: `kernels/` and `utils.py` live
        # under `fastdeploy.model_executor.models.qwen3_elastic`, NOT next to
        # this backend (which sits under `layers/attention/`), so we must use
        # absolute imports here.
        from fastdeploy.model_executor.models.qwen3_elastic.kernels import (
            Xattention_prefill_dim4,
        )
        from fastdeploy.model_executor.models.qwen3_elastic.utils import (
            ctx_q_pool,
            derive_head_mask_type,
        )

        metadata = self.attention_metadata

        # ---- Same as parent: PD signals + cache addr + layer-0 metadata ----
        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        if int(os.getenv("USE_TBO", "0")) == 1:
            if hasattr(forward_meta, "tbo_microbatch_id"):
                if forward_meta.tbo_microbatch_id == 0:
                    os.environ["FLAGS_fmt_write_cache_completed_signal"] = "0"
                elif forward_meta.tbo_microbatch_id == 1:
                    os.environ["FLAGS_fmt_write_cache_completed_signal"] = "1"

        norm_after_rope_in_kernel = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope_in_kernel else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope_in_kernel else None

        cache_quant_type_str = getattr(layer, "cache_quant_type_str", "none")
        if cache_quant_type_str == "block_wise_fp8":
            cache_k = forward_meta.caches[4 * layer.layer_id]
            cache_v = forward_meta.caches[4 * layer.layer_id + 1]
            cache_k_scales = forward_meta.caches[4 * layer.layer_id + 2]
            cache_v_scales = forward_meta.caches[4 * layer.layer_id + 3]
        else:
            cache_k = forward_meta.caches[2 * layer.layer_id]
            cache_v = forward_meta.caches[2 * layer.layer_id + 1]
            cache_k_scales = getattr(layer, "cache_k_scale", None)
            cache_v_scales = getattr(layer, "cache_v_scale", None)

        if layer.layer_id == 0:
            get_block_shape_and_split_kv_block(
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.decoder_batch_ids,
                forward_meta.decoder_tile_ids_per_batch,
                forward_meta.decoder_num_blocks_cpu,
                forward_meta.decoder_num_blocks_device,
                forward_meta.decoder_chunk_size_device,
                forward_meta.max_len_tensor_cpu,
                forward_meta.encoder_batch_ids,
                forward_meta.encoder_tile_ids_per_batch,
                forward_meta.encoder_num_blocks_x_cpu,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                self.encoder_block_shape_q,
                self.decoder_block_shape_q,
                self.group_size,
                self.block_size,
            )

            if forward_meta.max_len_tensor_cpu[1].item() > 0:
                forward_meta.max_len_tensor_cpu_decoder = paddle.clone(forward_meta.max_len_tensor_cpu)
                forward_meta.max_len_tensor_cpu_decoder[1] = 0

                (
                    forward_meta.cu_seqlens_k,
                    forward_meta.pre_cache_batch_ids,
                    forward_meta.pre_cache_tile_ids_per_batch,
                    forward_meta.pre_cache_num_blocks_cpu,
                    forward_meta.kv_token_num_cpu,
                ) = pre_cache_len_concat(
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.max_len_tensor_cpu[2],
                    self.block_size,
                )
                # Elastic prefill path doesn't use FA4 / attn_mask_q.
                forward_meta.attn_mask_q = None

        use_fa_do_prefill = forward_meta.max_len_tensor_cpu[1].item() > 0

        # ----------------------- prefill leg : BSA -----------------------
        if use_fa_do_prefill:
            # ---- Extract pre-RoPE K for router (BEFORE gqa_rope_write_cache) ----
            # The router MLP (mask_allocator) was trained on K post-q_norm/k_norm
            # but PRE-RoPE (see reference modeling_flash_qwen.py L1582-1650:
            # q_norm/k_norm -> router(k) -> RoPE). Here ``qkv`` is already
            # post-norm because Qwen3ElasticAttention.forward applies
            # ``self.qk_norm(qkv_out)`` before calling ``self.attn``. We slice
            # K out of the fused QKV tensor while it is still pre-RoPE.
            #   qkv layout: [T, q_size + 2*kv_size] with Q | K | V contiguous.
            T_pre = qkv.shape[0]
            q_size_local = layer.num_heads * layer.head_dim
            kv_size_local = layer.kv_num_heads * layer.head_dim
            k_pre_rope = qkv[:, q_size_local : q_size_local + kv_size_local].reshape(
                [T_pre, layer.kv_num_heads, layer.head_dim]
            )

            q, k, v, _ = gqa_rope_write_cache(
                qkv,
                cache_k,
                cache_v,
                forward_meta.cu_seqlens_q,
                forward_meta.cu_seqlens_k,
                forward_meta.rotary_embs,
                forward_meta.seq_lens_this_time,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.block_tables,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                forward_meta.pre_cache_batch_ids,
                forward_meta.pre_cache_tile_ids_per_batch,
                forward_meta.pre_cache_num_blocks_cpu,
                q_norm_weight,
                k_norm_weight,
                cache_k_scales,
                cache_v_scales,
                getattr(layer, "cache_k_out_scale", None),
                getattr(layer, "cache_v_out_scale", None),
                getattr(layer, "cache_k_zp", None),
                getattr(layer, "cache_v_zp", None),
                metadata.kv_signal_data_list[layer.layer_id],
                forward_meta.kv_token_num_cpu[0].item(),
                self.max_seq_len,
                getattr(layer, "rms_norm_eps", 1e-6),
                layer.use_neox_rotary_style,
                getattr(layer, "cache_quant_type_str", "none"),
                self.rope_3d,
            )
            # q: [total_T, num_heads, head_dim]; k, v: [total_T, kv_num_heads, head_dim]
            T_total = q.shape[0]
            assert T_total == T_pre, (T_total, T_pre)

            # GQA expand to match Q heads (BSA requires H_q == H_kv)
            k_full = paddle.repeat_interleave(k, self.group_size, axis=1)
            v_full = paddle.repeat_interleave(v, self.group_size, axis=1)

            # Number of prefill segments. ``cu_seqlens_k`` has shape [B_prefill+1].
            B_prefill = int(forward_meta.cu_seqlens_k.shape[0]) - 1
            prefill_cu = forward_meta.cu_seqlens_q[: B_prefill + 1]

            # Process each prefill segment independently: per-segment router +
            # BSA call. This is required for varlen profile-runs and multi-seq
            # batching; the BS=1 production path collapses to a single iteration.
            seg_outs = []
            for i in range(B_prefill):
                s = int(prefill_cu[i].item())
                e = int(prefill_cu[i + 1].item())
                Ti = e - s
                if Ti <= 0:
                    continue

                seg_k_pre = k_pre_rope[s:e]                       # [Ti, H_kv, D]
                seg_pool = ctx_q_pool(seg_k_pre)                  # [1, H_kv, D]
                seg_z = layer.mask_allocator(seg_pool).reshape([-1])
                seg_mask = derive_head_mask_type(
                    seg_z,
                    retrieval_mode=layer.retrieval_mode,
                    toggle_type=layer.toggle_type,
                    group_size=self.group_size,
                )

                seg_q = q[s:e].transpose([1, 0, 2]).unsqueeze(0)       # [1, H, Ti, D]
                seg_k = k_full[s:e].transpose([1, 0, 2]).unsqueeze(0)
                seg_v = v_full[s:e].transpose([1, 0, 2]).unsqueeze(0)

                seg_out = Xattention_prefill_dim4(
                    seg_q, seg_k, seg_v,
                    stride=layer.xattn_stride,
                    cu_seq_lens=paddle.to_tensor([0, Ti], dtype="int32"),
                    norm=layer.xattn_norm,
                    threshold=layer.xattn_threshold,
                    block_size=layer.block_size,
                    use_triton=True,
                    head_mask_type=seg_mask,
                    sink_num=layer.sink_blocks,
                    local_num=layer.local_blocks,
                    causal=True,
                )  # [1, H, Ti, D]
                seg_outs.append(
                    seg_out.transpose([0, 2, 1, 3]).reshape([Ti, self.attn_outputsize_tp])
                )

                # Cache the *last* segment's router decision for downstream
                # use (debug / metric). Sufficient for BS=1 production.
                if hasattr(layer, "_z_kv_cache"):
                    layer._z_kv_cache.set_value(seg_z)
                if hasattr(layer, "_head_mask_type_cache"):
                    layer._head_mask_type_cache.set_value(seg_mask)

                # # ---------- BEGIN TEMP: dump per-layer router decisions ----------
                # # Triggered only when FD_ELASTIC_DUMP_ROUTER points to a JSONL
                # # path. Each prefill segment appends one record. Remove this
                # # block (and the END TEMP marker below) when no longer needed.
                # _dump_path = os.getenv("FD_ELASTIC_DUMP_ROUTER", "")
                # if _dump_path:
                #     # Skip FD's profile-run / warmup prefill (which has seg_len
                #     # ~= max_num_batched_tokens-1, e.g. 65534 for max_model_len
                #     # =65536). The runner sets FD_ELASTIC_DUMP_SKIP_SEGLEN_GE
                #     # to a threshold above the real prompt but below the
                #     # warmup length.
                #     _skip_ge = os.getenv("FD_ELASTIC_DUMP_SKIP_SEGLEN_GE", "")
                #     _skip = bool(_skip_ge) and int(Ti) >= int(_skip_ge)
                #     if not _skip:
                #         import json as _json
                #         _record = {
                #             "layer_id": int(getattr(layer, "layer_id", -1)),
                #             "seg_idx": int(i),
                #             "seg_len": int(Ti),
                #             "z_kv": seg_z.tolist(),
                #             "head_mask_type": seg_mask.tolist(),
                #             "retrieval_mode": layer.retrieval_mode,
                #             "toggle_type": layer.toggle_type,
                #         }
                #         with open(_dump_path, "a", encoding="utf-8") as _df:
                #             _df.write(_json.dumps(_record) + "\n")
                # # ---------- END TEMP: dump per-layer router decisions ----------

            res_encoder = paddle.concat(seg_outs, axis=0) if seg_outs else paddle.zeros(
                [0, self.attn_outputsize_tp], dtype=q.dtype
            )

        # ----------------------- decode leg : append_attention -----------------------
        res_decoder = append_attention(
            qkv, cache_k, cache_v,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            forward_meta.block_tables,
            forward_meta.encoder_batch_ids,
            forward_meta.encoder_tile_ids_per_batch,
            forward_meta.encoder_num_blocks_x_cpu,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            forward_meta.decoder_num_blocks_cpu,
            forward_meta.max_len_tensor_cpu_decoder if use_fa_do_prefill else forward_meta.max_len_tensor_cpu,
            forward_meta.rotary_embs,
            forward_meta.attn_mask,
            layer.qkv_bias,
            layer.qkv_scale,
            cache_k_scales,
            cache_v_scales,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            layer.linear_shift,
            layer.linear_smooth,
            forward_meta.attn_mask_offsets,
            metadata.kv_signal_data_list[layer.layer_id],
            q_norm_weight,
            k_norm_weight,
            getattr(layer, "sinks", None),
            getattr(layer, "rms_norm_eps", 1e-6),
            metadata._fuse_kernel_compute_dtype,
            getattr(layer, "cache_quant_type_str", "none"),
            layer.use_neox_rotary_style,
            self.rope_3d,
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            getattr(layer, "out_scale", -1.0),
            self.encoder_block_shape_q,
            self.decoder_block_shape_q,
            self.max_partition_size,
            self.max_seq_len,
            self.speculate_max_draft_token_num + 1,
            self.causal,
            self.speculative_method is not None,
        )

        if use_fa_do_prefill:
            merge_prefill_decode_output(
                res_encoder, res_decoder,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.cu_seqlens_q,
                self.num_heads,
                self.head_dim,
                self.speculate_max_draft_token_num + 1,
            )
            return res_encoder
        return res_decoder
