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

"""Generic PaddleFormers modeling backend base class."""

import logging

from fastdeploy.model_executor.utils import is_paddlefleet_available

if not is_paddlefleet_available():
    logging.warning("paddlefleet is not installed, skipping base_fleet module")
else:
    import math
    from collections.abc import Iterable
    from typing import TYPE_CHECKING, Dict

    import paddle
    from paddle import nn
    from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
    from paddlefleet.models.gpt.lm_head import GPTLMHead
    from paddlefleet.transformer.layer import FleetLayer
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddleformers.utils.log import logger

    from fastdeploy.model_executor.forward_meta import ForwardMeta  # noqa: F401
    from fastdeploy.model_executor.graph_optimization.decorator import (
        support_graph_optimization,
    )

    if TYPE_CHECKING:
        from fastdeploy.config import FDConfig

    from fastdeploy.model_executor.layers.attention.attention import Attention

    USE_ERNIE = False

    class FastDeployAttention(FleetLayer):
        """
        FastDeploy version of DotProductAttention, holding an internal FastDeploy Attention module.

        This class can be used to replace PaddleFleet's DotProductAttention,
        using FastDeploy's attention backend for computation.
        """

        def __init__(
            self,
            config: TransformerConfig,
            fd_attention: Attention,
            num_attention_heads: int,
            num_key_value_heads: int,
            softmax_scale: float,
            hidden_size_per_attention_head: int,
            hidden_size_per_partition: int,
            layer_id: int,
            window_attn_skip_freq=None,
            sliding_window: int = 0,
        ):
            """
            Initialize FastDeployAttention.

            Args:
                fd_attention: FastDeploy Attention instance
                num_attention_heads: Number of attention heads
                num_key_value_heads: Number of KV heads
                softmax_scale: Softmax scaling factor
                hidden_size_per_attention_head: Hidden dimension per attention head
                hidden_size_per_partition: Hidden size per partition
                layer_id: Current layer ID
            """
            super().__init__(config)
            self.fd_attention = fd_attention
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.softmax_scale = softmax_scale
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
            self.hidden_size_per_partition = hidden_size_per_partition
            self.layer_id = layer_id
            self.window_attn_skip_freq = window_attn_skip_freq
            self.sliding_window = sliding_window

        def forward(
            self,
            query: paddle.Tensor,
            key: paddle.Tensor,
            value: paddle.Tensor,
            attention_mask: paddle.Tensor,
            attn_mask_startend_row_indices: paddle.Tensor = None,
            attn_mask_type=None,
            attention_bias: paddle.Tensor = None,
            packed_seq_params=None,
            use_rr_flash_attention: bool = False,
            past_key_values=None,
            layer_idx=None,
            use_cache=False,
            x: paddle.Tensor = None,
            qr: paddle.Tensor = None,
            kv_compressed: paddle.Tensor = None,
            k_pos_emb: paddle.Tensor = None,
            q_absorbed: paddle.Tensor = None,
            v_b_proj_weight: paddle.Tensor = None,
        ):
            """
            Forward pass.

            Args:
                query: Query tensor, supported formats:
                    - 4D BSHD: [b, sq, np, hn] (PaddleFleet default)
                    - 3D HSD: [np, sq, hn]
                    - 3D SHD: [sq, np, hn]
                key: Key tensor, same format as above, but head count may differ (GQA)
                value: Value tensor, same format as above
                attention_mask: Attention mask
                attn_mask_startend_row_indices: FlashMask start-end row indices
                attn_mask_type: Attention mask type
                attention_bias: Attention bias
                packed_seq_params: Packed sequence parameters
                use_rr_flash_attention: Whether to use RR Flash Attention
                kv_compressed: Compressed KV tensor for MLA (Multi-Latent Attention)

            Returns:
                Attention output tensor
            """
            # Try to get forward_meta from config (PaddleFleet does not pass this parameter when calling)
            forward_meta = getattr(self.config, "forward_meta", None)
            assert forward_meta is not None, "forward_meta must be provided"

            # Set scaling factor
            original_scale = getattr(self.fd_attention, "scale", None)
            if original_scale is None:
                self.fd_attention.scale = self.softmax_scale

            # Check if MLA mode is enabled
            is_mla = getattr(self.config, "multi_latent_attention", False)

            try:
                # Refer to the processing logic of fastdeploy_append_attention_forward
                # Support 3D (SHD) and 4D (BSHD) input

                # 4D input: squeeze to 3D (only supports batch=1)
                def squeeze_to_3d(t: paddle.Tensor, name: str) -> paddle.Tensor:
                    if t is None:
                        return None
                    if t.ndim == 4:
                        if int(t.shape[0]) != 1:
                            raise ValueError(
                                f"{name} batch size {int(t.shape[0])} not supported, only batch=1 is supported"
                            )
                        return t.squeeze(0)
                    if t.ndim == 3:
                        return t
                    raise ValueError(f"{name} has unexpected dims {t.ndim}, expect 3 or 4")

                q = squeeze_to_3d(query, "query")
                k = squeeze_to_3d(key, "key")
                v = squeeze_to_3d(value, "value")

                if is_mla:
                    need_do_prefill = forward_meta.max_len_tensor_cpu[1] > 0
                    need_do_decode = forward_meta.max_len_tensor_cpu[2] > 0

                    assert kv_compressed is not None, "kv_compressed must be provided when use"
                    compressed_kv = kv_compressed.squeeze(0)
                    k_pos_emb_sq = k_pos_emb.squeeze(0)

                    if self.window_attn_skip_freq is not None and self.window_attn_skip_freq[self.layer_id] == 1:
                        kv_lora_rank = self.config.kv_lora_rank

                        q_input = squeeze_to_3d(q_absorbed, "q_absorbed") if q_absorbed.ndim == 4 else q_absorbed
                        num_attention_heads_tp = q_input.shape[1]

                        """DSA sliding-window attention path, mirroring DeepseekV3MLAAttention.forward_swa_static."""
                        from fastdeploy.model_executor.layers.attention import (
                            DSAAttentionBackend,
                        )
                        from fastdeploy.model_executor.models.deepseek_v3 import (
                            get_swa_indexer_top_k,
                        )

                        indexer_top_k = paddle.full([q_input.shape[0], 1, self.sliding_window[0]], -1, dtype="int32")
                        get_swa_indexer_top_k(
                            indexer_top_k,
                            forward_meta.block_tables,
                            forward_meta.cu_seqlens_q,
                            forward_meta.seq_lens_encoder,
                            forward_meta.seq_lens_decoder,
                            forward_meta.batch_id_per_token,
                        )
                        fmqa_out = DSAAttentionBackend.forward_static(
                            q=q_input.contiguous(),
                            indexer_topk=indexer_top_k,
                            compressed_kv=compressed_kv,
                            k_pe=k_pos_emb_sq,
                            latent_cache=forward_meta.caches[self.layer_id],
                            forward_meta=forward_meta,
                            attn_softmax_scale=self.softmax_scale,
                        )

                        fmqa_out = fmqa_out.reshape_([-1, num_attention_heads_tp, kv_lora_rank]).transpose([1, 0, 2])
                        fmqa_out = paddle.bmm(fmqa_out, v_b_proj_weight)
                        output = fmqa_out.transpose([1, 0, 2]).reshape(
                            [-1, num_attention_heads_tp * self.config.v_head_dim]
                        )

                    else:
                        output = None
                        fmqa_out = None
                        if need_do_prefill:
                            # Prefill: keep 3D tensors for flash_attn_func
                            output = self.fd_attention.forward(
                                q=q,
                                k=k,
                                v=v,
                                qkv=None,
                                compressed_kv=compressed_kv,
                                k_pe=k_pos_emb_sq,
                                forward_meta=forward_meta,
                            )
                            output.reshape_([output.shape[0], output.shape[1] * output.shape[2]])

                        if need_do_decode:
                            # Decode: use absorbed q for multi_head_latent_attention C++ kernel
                            # q_absorbed: [s, heads, kv_lora_rank + qk_rope_head_dim] (after squeeze_to_3d)
                            # C++ kernel expects: [token_num, heads * (kv_lora_rank + qk_rope_head_dim)]
                            q_abs = squeeze_to_3d(q_absorbed, "q_absorbed") if q_absorbed.ndim == 4 else q_absorbed
                            seq_len = int(q_abs.shape[0])
                            q_input = q_abs.reshape([seq_len, -1])

                            fmqa_out = self.fd_attention.forward(
                                q=q_input,
                                k=None,
                                v=None,
                                qkv=None,
                                compressed_kv=compressed_kv,
                                k_pe=k_pos_emb_sq,
                                forward_meta=forward_meta,
                            )

                            # V de-absorption: kernel output [token, heads * kv_lora_rank]
                            # -> [heads, token, kv_lora_rank] @ wv_b [heads, kv_lora_rank, v_head_dim]
                            # -> [token, heads * v_head_dim]
                            kv_lora_rank = self.config.kv_lora_rank
                            v_head_dim = self.config.v_head_dim
                            num_heads = fmqa_out.shape[-1] // kv_lora_rank
                            fmqa_out = fmqa_out.reshape([-1, num_heads, kv_lora_rank]).transpose([1, 0, 2])
                            fmqa_out = paddle.bmm(fmqa_out, v_b_proj_weight)
                            fmqa_out = fmqa_out.transpose([1, 0, 2]).reshape([-1, num_heads * v_head_dim])
                            # Merge prefill and decode outputs if both are present
                            if need_do_prefill:
                                try:
                                    from fastdeploy.model_executor.ops.gpu import (
                                        merge_prefill_decode_output,
                                    )

                                    merge_prefill_decode_output(
                                        output,
                                        fmqa_out,
                                        forward_meta.seq_lens_encoder,
                                        forward_meta.seq_lens_decoder,
                                        forward_meta.seq_lens_this_time,
                                        forward_meta.cu_seqlens_q,
                                        num_heads,
                                        v_head_dim,
                                        1,
                                    )
                                except (ImportError, AttributeError):
                                    logger.warning(
                                        "merge_prefill_decode_output not available, using decode output only"
                                    )
                                    output = fmqa_out
                            else:
                                output = fmqa_out
                else:
                    # Standard mode: concatenate QKV
                    seq_len = int(q.shape[0])

                    # SHD: [seq, heads, dim] -> flatten to [seq, heads*dim]
                    q_flat = q.reshape([seq_len, -1])
                    k_flat = k.reshape([seq_len, -1])
                    v_flat = v.reshape([seq_len, -1])

                    # Concatenate QKV: [seq, (q_heads + kv_heads + kv_heads) * head_dim]
                    qkv = paddle.concat([q_flat, k_flat, v_flat], axis=-1)

                    output = self.fd_attention.forward(qkv=qkv, forward_meta=forward_meta)

                # Restore batch dimension: [seq, hidden] -> [b, seq, hidden]
                # PaddleFleet expects 3D output format
                output = output.unsqueeze(0)

                return output
            finally:
                # Restore original scale
                if original_scale is None:
                    if hasattr(self.fd_attention, "scale"):
                        delattr(self.fd_attention, "scale")
                else:
                    self.fd_attention.scale = original_scale

    @support_graph_optimization
    class PaddleFleetModelBase(nn.Layer):
        """
        A mixin-style base class to provide PaddleFormers backend logic on top of nn.Layer.
        This class subclasses nn.Layer and provides common methods to
        initialize and manage a PaddleFormers model.
        """

        def __init__(self, fd_config: "FDConfig", **kwargs):
            super().__init__(fd_config)
            logger.info("Initializing PaddleFormers backend.")
            self.fd_config = fd_config  # FastDeploy's top-level FDConfig
            self.model_config = fd_config.model_config  # FastDeploy's ModelConfig
            if USE_ERNIE:
                from paddleformers.transformers.configuration_utils import (
                    PretrainedConfig,
                )

                _config_dict, _ = PretrainedConfig.get_config_dict(
                    self.model_config.model, _configuration_file="model_config.json"
                )
                from ernie5.pretrain import Ernie5V2Config

                self.paddleformers_config = Ernie5V2Config.from_dict(_config_dict)
                self.paddleformers_config.moe_dequant_input = True
            else:
                from paddleformers.transformers import AutoConfig

                self.paddleformers_config = AutoConfig.from_pretrained(self.model_config.model)

            # Assign parallel config from fd_config.parallel_config to paddleformers_config
            parallel_config = fd_config.parallel_config
            self.paddleformers_config.data_parallel_size = parallel_config.data_parallel_size
            self.paddleformers_config.tensor_model_parallel_size = parallel_config.tensor_parallel_size
            self.paddleformers_config.sequence_parallel = parallel_config.sequence_parallel
            self.paddleformers_config.expert_model_parallel_size = parallel_config.expert_parallel_size
            # if parallel_config.expert_parallel_size > 1 and parallel_config.sequence_parallel == False:
            #     self.paddleformers_config.tensor_model_parallel_size = 1
            #     logger.warning("When using expert parallelism and tensor parallelism, sequence parallelism must be used in fleet set tp=1 .")
            self.paddleformers_config.parallel_output = self.paddleformers_config.tensor_model_parallel_size == 1
            self.paddleformers_config.max_seq_len = self.model_config.max_model_len
            self.paddleformers_config.params_dtype = self.model_config.dtype or "bfloat16"
            # self.paddleformers_config.moe_grouped_gemm = True
            self.paddleformers_config.moe_token_dispatcher_type = "deepep"
            # self.paddleformers_config.use_cpu_initialization = True
            self.paddleformers_config.use_cpu_initialization = True
            self.paddleformers_config.perform_initialization = False
            self.paddleformers_config.gated_attention = getattr(self.paddleformers_config, "use_gated_attn", False)
            self.paddleformers_config.moe_layer_interval = getattr(self.paddleformers_config, "moe_layer_freq", 1)
            if getattr(self.paddleformers_config, "multi_latent_attention", False):
                self.paddleformers_config.qk_head_dim = (
                    self.paddleformers_config.qk_rope_head_dim + self.paddleformers_config.qk_nope_head_dim
                )
            # Initialize PaddleFleet parallel_state so that its TP group is consistent with FastDeploy.
            # PaddleFleet's ColumnParallelLinear/RowParallelLinear obtains TP world_size/rank
            # via parallel_state. Without initialization, it defaults to 1, causing weights
            # to not be TP-sharded, which mismatches FastDeploy's KV cache (allocated per TP).
            # if parallel_config.tensor_parallel_size > 1:
            self._init_paddlefleet_parallel_state(fd_config)

            # The specific text model config
            # Sync important config values from text_config to model_config
            # This ensures fallback models use their actual config values instead of FD defaults
            self._sync_config_from_text_config()
            # For convenience, keep direct access to some FD configs
            self.quant_config = self.fd_config.quant_config

            # Load model using from_pretrained to support weight loading
            # Pass dtype, config and other options from kwargs

            model_load_kwargs = {
                "dtype": self.model_config.dtype,
                "config": self.paddleformers_config,
                "convert_from_hf": True,
                "load_via_cpu": True,
                "load_checkpoint_format": "flex_checkpoint",
            }
            if USE_ERNIE:
                from fleet_bridge import AutoModelForCausalLM

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model,
                    config=self.paddleformers_config,
                    dtype=self.model_config.dtype,
                )
            else:
                from paddleformers.transformers.auto.modeling import (
                    AutoModelForCausalLM,
                )

                # Set random seed before model construction for reproducibility
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model,
                    **model_load_kwargs,
                )

            self.model.eval()
            # Patch PaddleFleet core_attention with FastDeploy attention
            patched_count = patch_paddlefleet_core_attention(
                model=self.model,
                fd_config=self.fd_config,
            )
            logger.info(f"Patched {patched_count} attention layers with FastDeploy")

        def compute_logits(self, hidden_state, forward_meta=None):
            """Compute logits from hidden states using lm_head."""
            lm_head = self.model.get_lm_head()
            # ColumnParallelLinear expects input [s, b, h]
            hidden_state = hidden_state.unsqueeze(1)  # [num_tokens, h] -> [num_tokens, 1, h]
            logits = lm_head({"hidden_states": hidden_state})
            # Output [num_tokens, 1, vocab], squeeze back to [num_tokens, vocab]
            if logits.ndim == 3:
                logits = logits.squeeze(1)
            logits = logits.astype(paddle.float32)
            logits[:, self.model_config.ori_vocab_size :] = -float("inf")
            return logits

        def _init_paddlefleet_parallel_state(self, fd_config) -> None:
            """
            Initialize PaddleFleet's parallel_state so that ColumnParallelLinear/RowParallelLinear
            can correctly obtain TP world_size and rank, and thus correctly shard weights
            and build sharded_state_dict.

            References the initialization logic in PaddleFormers' training_args.py,
            using the official initialize_fleet API instead of directly manipulating
            parallel_state internal variables.
            """
            from paddle.distributed import fleet

            parallel_config = fd_config.parallel_config

            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": parallel_config.tensor_parallel_size,
                "pp_degree": 1,
                "sep_degree": 1,
                "sharding_degree": parallel_config.data_parallel_size,
                "ep_degree": parallel_config.expert_parallel_size,
                "cp_degree": 1,
                "moe_sharding_degree": 1,
                "order": [
                    "pp",
                    "moe_sharding",
                    "ep",
                    "dp",
                    "sharding",
                    "sep",
                    "cp",
                    "mp",
                ],
            }
            # Reset parallel state so that PaddleFleet's fleet.init can reinitialize
            # with the correct EP topology instead of reusing FastDeploy's.
            import paddle.distributed.fleet.base.topology as tp_mod
            import paddle.distributed.parallel_helper as ph

            # 1) Reset hybrid parallel group so _init_hybrid_parallel_env runs again
            tp_mod._HYBRID_PARALLEL_GROUP = None
            # 2) Reset parallel context so init_parallel_env runs again
            ph.__parallel_ctx__clz__ = None

            fleet.init(is_collective=True, strategy=strategy)
            logger.info(
                f"Initialized PaddleFleet parallel_state via initialize_fleet "
                f"(sharddp={parallel_config.data_parallel_size}, "
                f"mp={parallel_config.tensor_parallel_size}, "
                f"ep={parallel_config.expert_parallel_size}, "
                f"sp={parallel_config.sequence_parallel})"
            )
            import paddle.distributed as dist
            from paddlefleet import parallel_state

            tp_group = parallel_state._TENSOR_MODEL_PARALLEL_GROUP
            current_tp_size = None
            if tp_group is not None:
                current_tp_size = getattr(tp_group, "nranks", None)
                if current_tp_size is None:
                    current_tp_size = getattr(tp_group, "world_size", None)

            expected_tp_size = parallel_config.tensor_parallel_size
            need_init = tp_group is None or current_tp_size != expected_tp_size
            if need_init:
                if expected_tp_size == 1:
                    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = dist.new_group(ranks=[dist.get_rank()])
                else:
                    hcg = fleet.get_hybrid_communicate_group()
                    parallel_state.initialize_model_parallel(hcg)

            from paddlefleet.tensor_parallel.random import (
                model_parallel_cuda_manual_seed,
            )

            try:
                model_parallel_cuda_manual_seed(seed=42)
            except AssertionError:
                pass

        def _sync_config_from_text_config(self) -> None:
            """
            Sync important config values from text_config (PaddleFormers/HF config)
            to model_config. This ensures fallback models use their actual config
            values instead of FD's defaults.

            This is crucial for models with unique configs like:
            - Gemma3: tie_word_embeddings=True, layer_types, sliding_window
            - Mistral: sliding_window
            - etc.
            """
            mc = self.model_config
            tc = self.paddleformers_config

            sync_fields = [
                "tie_word_embeddings",
                "sliding_window",
                "sliding_window_pattern",
                "layer_types",  # May be computed as property
                "rope_theta",
                "rope_scaling",
                "head_dim",
                "v_head_dim",  # For MLA (Multi-Latent Attention) support
                "qk_head_dim",
                "rms_norm_eps",
                "rope_local_base_freq",  # Gemma3 specific
                "query_pre_attn_scalar",  # Gemma3 specific
            ]

            synced = []
            for field in sync_fields:
                text_value = getattr(tc, field, None)
                if text_value is not None:
                    # Only sync if not already set or if FD default differs
                    current_value = getattr(mc, field, None) if hasattr(mc, field) else None
                    if current_value is None or current_value != text_value:
                        setattr(mc, field, text_value)
                        synced.append(f"{field}={text_value}")

        def embed_input_ids(self, input_ids: paddle.Tensor) -> paddle.Tensor:
            """Embed input_ids using the model's embedding layer."""
            embedding_layer = self.model.get_input_embeddings()

            original_ndim = input_ids.ndim
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)  # [num_tokens] -> [1, num_tokens]

            inputs_embeds = embedding_layer(input_ids)

            # Embedding output is [batch, seq, h], squeeze back to [num_tokens, h]
            if original_ndim == 1 and inputs_embeds.ndim == 3:
                inputs_embeds = inputs_embeds.squeeze(0)

            if hasattr(self, "embed_scale") and self.embed_scale is not None:
                inputs_embeds *= self.embed_scale
            return inputs_embeds

        @paddle.no_grad()
        def forward(
            self,
            inputs: Dict,
            forward_meta: ForwardMeta,
            **kwargs,
        ):
            """Full transformer forward: input_ids -> hidden_states.

            This method is the primary forward pass for the model, computing:
            1. Position IDs based on seq_lens_decoder (absolute positions for RoPE)
            2. Token embeddings via embed_input_ids
            3. Transformer layers via self.model()

            Returns:
                hidden_states: [TotalTokens, HiddenDim]
            """
            # Handle empty batch case (e.g., DP worker with no data in EP mode)
            if getattr(forward_meta, "is_zero_size", False) or inputs["ids_remove_padding"].shape[0] == 0:
                # Return zero tensor with correct shape: [0, hidden_size]
                hidden_size = self.model_config.hidden_size
                dtype = self.model_config.dtype
                return paddle.empty([0, hidden_size], dtype=dtype)

            ids_remove_padding = inputs["ids_remove_padding"]
            num_tokens = ids_remove_padding.shape[0]
            batch_id_per_token = forward_meta.batch_id_per_token  # [num_tokens]
            seq_lens_decoder = forward_meta.seq_lens_decoder  # [batch_size, 1]

            if batch_id_per_token is not None and seq_lens_decoder is not None:
                decoder_offsets = seq_lens_decoder.squeeze(-1)  # [batch_size]
                # Ensure decoder_offsets is at least 1D tensor
                if decoder_offsets.ndim == 0:
                    decoder_offsets = decoder_offsets.reshape([1])
                token_decoder_offsets = paddle.index_select(
                    decoder_offsets, batch_id_per_token, axis=0
                )  # [num_tokens]

                cu_seqlens = forward_meta.cu_seqlens_q  # [batch_size + 1]
                if cu_seqlens is not None:
                    token_global_idx = paddle.arange(num_tokens, dtype="int64")
                    request_start_idx = paddle.index_select(cu_seqlens[:-1], batch_id_per_token, axis=0)
                    relative_positions = token_global_idx - request_start_idx.astype("int64")
                else:
                    relative_positions = paddle.zeros([num_tokens], dtype="int64")
                position_ids = token_decoder_offsets.astype("int64") + relative_positions
            else:
                position_ids = paddle.arange(num_tokens, dtype="int64")
                if seq_lens_decoder is not None:
                    position_ids = position_ids + seq_lens_decoder[0, 0].astype("int64")
            forward_meta.rope_already_applied = True
            # Also set forward_meta on each TransformerLayer's config
            # so that FastDeployAttention can retrieve it from core_attn.config
            if hasattr(self.model, "run_function"):
                for layer in self.model.run_function:
                    if not isinstance(layer, (GPTEmbedding, GPTLMHead)):
                        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "core_attention"):
                            core_attn = layer.self_attn.core_attention
                            if hasattr(core_attn, "config"):
                                core_attn.config.forward_meta = forward_meta

            inputs_embeds = self.embed_input_ids(ids_remove_padding).unsqueeze(0)

            # Build input dict, PipelineLayer passes data between layers via dict
            model_input = {
                "input_ids": None,
                "position_ids": position_ids,
            }
            # Add other parameters from kwargs
            for k, v in kwargs.items():
                if v is not None:
                    model_input[k] = v

            # Iterate over run_function, skip GPTLMHead
            # Only call TransformerLayer
            i = -1
            for layer in self.model.run_function:
                if isinstance(layer, GPTLMHead):
                    continue
                if isinstance(layer, (GPTEmbedding)):
                    model_input = layer(model_input, decoder_input=inputs_embeds)
                else:
                    model_input = layer(model_input)
                i += 1
            hidden_states = model_input["hidden_states"]
            # [b, s, h] -> [s, h] (b=1)
            hidden_states = hidden_states.squeeze(0)

            return hidden_states

        @paddle.no_grad()
        def load_weights(self, weights: Iterable[tuple[str, paddle.Tensor]]):
            # use model.from_pretrained to load weight
            logger.debug("load_weights called but skipped: weights already loaded via from_pretrained")
            pass

        def set_state_dict(self, state_dict):
            self.model.set_state_dict(state_dict)

    # ============================================================================
    # PaddleFleet Attention Patch Functions
    # ============================================================================

    def patch_paddlefleet_core_attention(
        model,
        fd_config: "FDConfig",
        layers_to_patch: list[int] | None = None,
    ):
        """
        Replace core_attention in all TransformerLayers of a PaddleFleet model with FastDeployAttention.

        Args:
            model: PaddleFleet model instance (inheriting from PipelineLayer)
            fd_config: FastDeploy FDConfig object, used to create Attention instances
            layers_to_patch: List of layer indices to patch, None means patch all layers

        Returns:
            int: Number of layers successfully patched

        Raises:
            ValueError: If the model structure is unexpected or parameters are incorrect
        """
        if fd_config is None:
            raise ValueError("fd_config must be provided")

        from fastdeploy.model_executor.layers.attention.attention import Attention

        # Iterate over run_function to find TransformerLayers
        patched_count = 0
        transformer_layers = []

        # Collect all TransformerLayers
        if hasattr(model, "run_function"):
            for layer in model.run_function:
                # Try to identify TransformerLayer
                layer_type = type(layer).__name__
                if "TransformerLayer" in layer_type or "transformer" in str(type(layer)):
                    transformer_layers.append(layer)

        if not transformer_layers:
            # Try alternative ways to find layers
            for name, module in model.named_sublayers():
                if "TransformerLayer" in type(module).__name__:
                    transformer_layers.append(module)

        if not transformer_layers:
            raise ValueError("No TransformerLayer found in model")

        # Patch core_attention for each TransformerLayer
        for layer in transformer_layers:
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None:
                # Try to get from other attributes
                layer_number = getattr(layer, "layer_id", None)

            if layer_number is None:
                logger.warning("layer_number not found, skip patching...")
                continue  # Skip layers where layer_id cannot be obtained

            # Check if this layer needs to be patched
            if layers_to_patch is not None and (layer_number) not in layers_to_patch:
                continue

            # Get core_attention
            if not hasattr(layer, "self_attn"):
                logger.warning(f"self_attn not found in layer {layer_number}, skip patching...")
                continue

            core_attn = layer.self_attn.core_attention
            if core_attn is None:
                logger.warning(f"core_attn not found in layer {layer_number}, skip patching...")
                continue

            # Get configuration info
            # Prefer per-partition values (values after TP sharding),
            # because PaddleFleet's QKV output is already per-partition when TP>1
            num_attention_heads = getattr(
                core_attn, "num_attention_heads_per_partition", getattr(core_attn.config, "num_attention_heads", None)
            )
            num_key_value_heads = getattr(
                core_attn,
                "num_query_groups_per_partition",
                getattr(core_attn.config, "num_key_value_heads", num_attention_heads),
            )
            hidden_size_per_attention_head = getattr(core_attn, "hidden_size_per_attention_head", None)
            if hidden_size_per_attention_head is not None:
                softmax_scale = getattr(core_attn, "softmax_scale", 1.0 / math.sqrt(hidden_size_per_attention_head))
            else:
                softmax_scale = 1.0

            hidden_size_per_partition = getattr(core_attn, "hidden_size_per_partition", None)
            if hidden_size_per_partition is None:
                head_dim = getattr(core_attn, "hidden_size_per_attention_head", hidden_size_per_attention_head)
                hidden_size_per_partition = num_attention_heads * head_dim

            fd_layer_id = layer_number

            # Create Attention instance inside FastDeployAttention
            fd_attn_instance = Attention(
                fd_config=fd_config,
                layer_id=fd_layer_id,
            )

            # Override Attention instance's head config to match PaddleFleet model
            # This is necessary because fd_config.model_config may differ from PaddleFleet model config
            fd_attn_instance.num_heads = num_attention_heads
            fd_attn_instance.kv_num_heads = num_key_value_heads
            fd_attn_instance.head_dim = hidden_size_per_attention_head
            logger.info(
                f"Overriding Attention config: num_heads={num_attention_heads}, kv_num_heads={num_key_value_heads}, head_dim={hidden_size_per_attention_head}"
            )

            # Create FastDeployAttention object and directly replace core_attention
            fast_deploy_core_attn = FastDeployAttention(
                config=core_attn.config,
                fd_attention=fd_attn_instance,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                softmax_scale=softmax_scale,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                hidden_size_per_partition=hidden_size_per_partition,
                layer_id=fd_layer_id,
                window_attn_skip_freq=getattr(fd_config.model_config, "window_attn_skip_freq", None),
                sliding_window=getattr(fd_config.model_config, "sliding_window", 0),
            )

            # Replace core_attention object
            layer.self_attn.core_attention = fast_deploy_core_attn

            patched_count += 1
            logger.info(f"Replaced core_attention with FastDeployAttention for layer {fd_layer_id}")

        logger.info(f"Successfully replaced {patched_count} core_attention layers with FastDeployAttention")

        return patched_count
