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

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import paddle
from paddle import nn
from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from paddleformers.transformers import AutoModel, PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.model_executor.forward_meta import ForwardMeta  # noqa: F401
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig

from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.utils import WeightsMapper


class PaddleFormersRMSNormWrapper(nn.Layer):
    """
    Wrapper for FD's RMSNorm to make it compatible with PaddleFormers.

    FD's RMSNorm always returns (output, residual_out) tuple,
    but PaddleFormers expects a single tensor.
    This wrapper extracts only the normalized output.
    """

    def __init__(self, fd_rmsnorm: RMSNorm):
        super().__init__()
        self._fd_rmsnorm = fd_rmsnorm
        # Expose weight for weight loading and other access
        self.weight = fd_rmsnorm.weight

    def forward(self, x):
        # FD RMSNorm returns (out, residual_out), we only need out
        out, _ = self._fd_rmsnorm(x)
        return out


def getattr_iter(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def maybe_prefix(prefix, name):
    if prefix:
        return f"{prefix}.{name}"
    return name


def fastdeploy_append_attention_forward(
    module: paddle.nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: paddle.Tensor,
    scaling: float | None = None,
    **kwargs,
):
    config = getattr(module, "config", None)
    if config is None:
        raise ValueError(f"Module {module} does not have 'config' attribute.")

    attention_instances = getattr(config, "attention_instances", None)
    forward_meta = getattr(config, "forward_meta", None)

    if attention_instances is None:
        raise ValueError("attention_instances not found in module.config")
    if forward_meta is None:
        raise ValueError("forward_meta not found in module.config")

    layer_idx = getattr(module, "layer_idx", getattr(module, "layer_id", None))
    if layer_idx is None:
        raise ValueError("layer_idx not found on attention module")

    self_attn = attention_instances[int(layer_idx)]
    if scaling is not None:
        self_attn.scale = float(scaling)

    # 统一获取 heads 信息
    num_heads = (
        getattr(module, "num_heads", None)
        or getattr(config, "num_attention_heads", None)
        or getattr(self_attn, "num_heads", None)
    )
    num_kv_heads = (
        getattr(module, "num_key_value_heads", None)
        or getattr(config, "num_key_value_heads", None)
        or getattr(self_attn, "num_key_value_heads", None)
        or getattr(self_attn, "kv_num_heads", None)
        or num_heads
    )
    num_heads = int(num_heads) if num_heads is not None else None
    num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else None

    # 仅支持 3D(HSD/SHD) 或 4D(BHSD/BSHD, 且 B=1) 输入
    def squeeze_to_3d(t: paddle.Tensor, name: str) -> paddle.Tensor:
        if t.ndim == 4:
            if int(t.shape[0]) != 1:
                raise ValueError(f"{name} batch size {int(t.shape[0])} not supported")
            return t.squeeze(0)
        if t.ndim == 3:
            return t
        raise ValueError(f"{name} has unexpected dims {t.ndim}, expect 3 or 4")

    q = squeeze_to_3d(query, "query")
    k = squeeze_to_3d(key, "key")
    v = squeeze_to_3d(value, "value")

    def heads_match(actual_heads: int, expected_heads: int | None) -> bool:
        if expected_heads is None:
            return False
        if actual_heads == expected_heads:
            return True
        return actual_heads > 0 and expected_heads % actual_heads == 0

    # 使用 Q/K 共同判断布局；歧义时默认 hsd（兼容 Paddle 常见路径）
    is_hsd = (
        heads_match(int(q.shape[0]), num_heads)
        and heads_match(int(k.shape[0]), num_kv_heads)
        and heads_match(int(v.shape[0]), num_kv_heads)
    )
    is_shd = (
        heads_match(int(q.shape[1]), num_heads)
        and heads_match(int(k.shape[1]), num_kv_heads)
        and heads_match(int(v.shape[1]), num_kv_heads)
    )

    if is_hsd:
        q_flat = q.transpose([1, 0, 2]).reshape([int(q.shape[1]), -1])
        k_flat = k.transpose([1, 0, 2]).reshape([int(k.shape[1]), -1])
        v_flat = v.transpose([1, 0, 2]).reshape([int(v.shape[1]), -1])
    elif is_shd:
        q_flat = q.reshape([int(q.shape[0]), -1])
        k_flat = k.reshape([int(k.shape[0]), -1])
        v_flat = v.reshape([int(v.shape[0]), -1])
    else:
        raise ValueError(
            f"Invalid attention layout: q={list(q.shape)}, k={list(k.shape)}, v={list(v.shape)}, "
            f"heads={num_heads}/{num_kv_heads}"
        )

    # Q/K/V flatten 后序列长度必须一致
    q_seq, k_seq, v_seq = int(q_flat.shape[0]), int(k_flat.shape[0]), int(v_flat.shape[0])
    if not (q_seq == k_seq == v_seq):
        raise ValueError(
            f"Sequence length mismatch after flattening: Q={q_seq}, K={k_seq}, V={v_seq}, "
            f"raw query={list(query.shape)}, key={list(key.shape)}, value={list(value.shape)}."
        )

    # 若 forward_meta 带了 ids_remove_padding，则强校验 Q 序列长度
    ids_remove_padding = getattr(forward_meta, "ids_remove_padding", None)
    if ids_remove_padding is not None:
        expected_seq = int(ids_remove_padding.shape[0])
        if q_seq != expected_seq:
            raise ValueError(f"Seq len mismatch: got {q_seq}, expect {expected_seq}")

    qkv = paddle.concat([q_flat, k_flat, v_flat], axis=-1)
    output = self_attn.forward(qkv=qkv, forward_meta=forward_meta)

    return output, None


ALL_ATTENTION_FUNCTIONS._global_mapping["fastdeploy_append"] = fastdeploy_append_attention_forward


@support_graph_optimization
class PaddleFormersModelBase(nn.Layer):
    """
    A mixin-style base class to provide PaddleFormers backend logic on top of nn.Layer.
    This class subclasses nn.Layer and provides common methods to
    initialize and manage a PaddleFormers model.
    """

    pf_to_fd_mapper = WeightsMapper(
        orig_to_new_prefix={
            "": "model.",
            "model.model.": "model.",
            "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
            "embed_tokens.weight": "model.embed_tokens.embeddings.weight",
            "model.lm_head.weight": "lm_head.linear.weight",
            "model.score.": "classifier.",
            "model.classifier.": "classifier.",
        }
    )

    def __init_subclass__(cls, *args, **kwargs):
        """Merge pf_to_fd_mapper in MRO from most specific to least specific."""
        super().__init_subclass__(*args, **kwargs)

        # Collect all mappings from base classes
        merged_mappings = {}
        for base in reversed(cls.__mro__):  # Reverse to go from least to most specific
            if base_pf_to_fd_mapper := getattr(base, "pf_to_fd_mapper", None):
                if hasattr(base_pf_to_fd_mapper, "orig_to_new_prefix"):
                    merged_mappings.update(base_pf_to_fd_mapper.orig_to_new_prefix)

        # Create new mapper with merged mappings
        cls.pf_to_fd_mapper = WeightsMapper(orig_to_new_prefix=merged_mappings)

    def __init__(self, fd_config: "FDConfig", **kwargs):
        super().__init__(fd_config)
        logger.info("Initializing PaddleFormers backend.")
        self.fd_config = fd_config  # FastDeploy's top-level FDConfig
        self.model_config = fd_config.model_config  # FastDeploy's ModelConfig

        from paddleformers.transformers import AutoConfig

        self.paddleformers_config = AutoConfig.from_pretrained(self.model_config.model)

        # PaddleFormers fused optimize option
        self.paddleformers_config.fuse_rms_norm = True
        model_type = getattr(self.paddleformers_config, "model_type", "").lower()
        supported_fused_qkv_models = ["qwen3", "qwen2"]

        tp_size = fd_config.parallel_config.tensor_parallel_size
        if tp_size > 1:
            self._use_fused_qkv = False
            logger.info(f"Fusion disabled for TP={tp_size} due to shape incompatibility")
        else:
            self._use_fused_qkv = model_type in supported_fused_qkv_models
            if self._use_fused_qkv:
                self.paddleformers_config.fuse_attention_qkv = True
                logger.info(f"Enabled fuse_attention_qkv for model_type={model_type}")
            else:
                logger.debug(f"QKV fusion not enabled for model_type={model_type}")

        # PaddleFormers fused optimize option
        self._use_fused_ffn = model_type in supported_fused_qkv_models
        if self._use_fused_ffn:
            self.paddleformers_config.fuse_attention_ffn = True
            self.paddleformers_config.fuse_swiglu = True
            logger.info(f"Enabled fuse_attention_ffn and fuse_swiglu for model_type={model_type}")

        self.text_config = self.paddleformers_config  # The specific text model config
        # Sync important config values from text_config to model_config
        # This ensures fallback models use their actual config values instead of FD defaults
        self._sync_config_from_text_config()
        # For convenience, keep direct access to some FD configs
        self.quant_config = self.fd_config.quant_config
        self.parallel_config = self.fd_config.parallel_config
        self.tp_group = self.parallel_config.tp_group
        self.tp_rank = self.parallel_config.tensor_parallel_rank
        self.paddleformers_config._attn_implementation = "fastdeploy_append"

        self.model: PretrainedModel = AutoModel.from_config(
            self.paddleformers_config,
            dtype=self.model_config.dtype,
        )
        self.model.eval()

        # Linear and Norm replace for FD optimized versions and TP support
        self.recursive_replace()
        # Attention instances for FD Attention backend
        self.attention_instances = self.create_attention_instances()
        self.paddleformers_config.attention_instances = self.attention_instances
        # Embedding replace for TP support
        input_embeddings = self.model.get_input_embeddings()
        self.embed_scale = getattr(input_embeddings, "embed_scale", None)
        embedding_dim = getattr_iter(self.text_config, ("embedding_size", "hidden_size"))
        if embedding_dim is None:
            raise ValueError(
                "Failed to determine embedding dimension from text_config: "
                "neither 'embedding_size' nor 'hidden_size' is set. "
                f"text_config type={type(self.text_config).__name__}."
            )
        self.model.set_input_embeddings(
            VocabParallelEmbedding(
                fd_config=self.fd_config,
                num_embeddings=self.text_config.vocab_size,
                embedding_dim=embedding_dim,
            )
        )

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
        tc = self.text_config

        sync_fields = [
            "tie_word_embeddings",
            "sliding_window",
            "sliding_window_pattern",
            "layer_types",  # May be computed as property
            "rope_theta",
            "rope_scaling",
            "head_dim",
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

    def recursive_replace(self):
        """Recursively replace modules in the model as needed.

        Replaces:
        - nn.Linear with FD's tensor parallel linear classes (based on naming rules)
        - *RMSNorm with FD's RMSNorm
        """
        tp_plan = self._get_tp_plan()

        def _get_linear_style(qual_name: str) -> str:
            """Determine linear style based on layer name patterns."""
            for pattern, style in tp_plan.items():
                if re.search(pattern, qual_name):
                    return style
            return "replicate"

        def _recursive_replace(module: nn.Layer, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                new_module = child_module

                if isinstance(child_module, nn.Linear):
                    style = _get_linear_style(qual_name)

                    # PaddlePaddle nn.Linear: weight shape is [in_features, out_features]
                    # PyTorch nn.Linear: has in_features/out_features attributes
                    if hasattr(child_module, "weight") and child_module.weight is not None:
                        weight_shape = child_module.weight.shape
                        in_features = weight_shape[0]
                        out_features = weight_shape[1]
                    else:
                        in_features = getattr(child_module, "in_features", None)
                        out_features = getattr(child_module, "out_features", None)

                    with_bias = hasattr(child_module, "bias") and child_module.bias is not None

                    if style == "colwise":
                        # For qkv_proj when fused QKV is enabled:
                        # Use ColumnParallelLinear (not QKVParallelLinear) because we fuse weights
                        # into PaddleFormers' per-KV-head interleaved format in load_weights()
                        if "qkv_proj" in qual_name and self._use_fused_qkv:
                            new_module = ColumnParallelLinear(
                                self.fd_config,
                                prefix=qual_name,
                                input_size=in_features,
                                output_size=out_features,
                                with_bias=with_bias,
                            )
                        # For up_gate_proj when fused FFN is enabled:
                        # Use MergedColumnParallelLinear which handles gate+up weight loading
                        elif "up_gate_proj" in qual_name and self._use_fused_ffn:
                            new_module = MergedColumnParallelLinear(
                                self.fd_config,
                                prefix=qual_name,
                                input_size=in_features,
                                output_size=out_features,
                                with_bias=with_bias,
                            )
                        else:
                            new_module = ColumnParallelLinear(
                                self.fd_config,
                                prefix=qual_name,
                                input_size=in_features,
                                output_size=out_features,
                                with_bias=with_bias,
                            )
                    elif style == "rowwise":
                        new_module = RowParallelLinear(
                            self.fd_config,
                            prefix=qual_name,
                            input_size=in_features,
                            output_size=out_features,
                            with_bias=with_bias,
                        )
                    else:  # replicate
                        new_module = ReplicatedLinear(
                            self.fd_config,
                            prefix=qual_name,
                            input_size=in_features,
                            output_size=out_features,
                            with_bias=with_bias,
                        )

                # RMSNorm replacement: use wrapper to adapt FD's tuple return to single tensor
                elif child_module.__class__.__name__.endswith("RMSNorm"):
                    if hasattr(child_module, "weight") and child_module.weight is not None:
                        hidden_size = child_module.weight.shape[0]
                    else:
                        hidden_size = getattr(self.text_config, "hidden_size", None)
                    eps = getattr(child_module, "epsilon", getattr(child_module, "variance_epsilon", 1e-6))
                    fd_rmsnorm = RMSNorm(
                        self.fd_config,
                        hidden_size=hidden_size,
                        eps=eps,
                        prefix=qual_name,
                        begin_norm_axis=-1,  # Normalize only last dim (hidden), not entire flattened tensor
                    )
                    # Wrap with PaddleFormersRMSNormWrapper for interface compatibility
                    new_module = PaddleFormersRMSNormWrapper(fd_rmsnorm)
                else:
                    _recursive_replace(child_module, prefix=qual_name)

                if new_module is not child_module:
                    setattr(module, child_name, new_module)

        _recursive_replace(self.model, prefix="model")

    def _get_tp_plan(self) -> dict[str, str]:
        """Get TP plan for linear layer replacement.

        Priority:
        1. Try to get from PaddleFormers model's _get_tensor_parallel_mappings classmethod
        2. Fall back to default naming-based rules

        Returns:
            Dict mapping regex patterns to style ("colwise", "rowwise", "replicate")
        """
        # Try to get TP mappings from PaddleFormers model class
        model_cls = type(self.model)
        if hasattr(model_cls, "_get_tensor_parallel_mappings"):
            try:
                # Call the classmethod with config
                mappings = model_cls._get_tensor_parallel_mappings(self.text_config, is_split=True)
                if mappings:
                    # Convert PaddleFormers mappings to our format
                    # mappings is like: {"model.layers.0.self_attn.q_proj.weight": partial(fn, is_column=True)}
                    # Extract layer name patterns and determine colwise/rowwise
                    colwise_layers = set()
                    rowwise_layers = set()

                    for key, func in mappings.items():
                        # Extract the layer suffix (e.g., "self_attn.q_proj.weight" -> "q_proj")
                        parts = key.split(".")
                        if len(parts) >= 2:
                            # Find the layer name (second to last before .weight/.bias)
                            for i, part in enumerate(parts):
                                if part.endswith("_proj") or part in (
                                    "up_proj",
                                    "gate_proj",
                                    "down_proj",
                                    "o_proj",
                                    "q_proj",
                                    "k_proj",
                                    "v_proj",
                                    "qkv_proj",
                                ):
                                    # Check is_column from partial func
                                    if hasattr(func, "keywords") and func.keywords.get("is_column", False):
                                        colwise_layers.add(part)
                                    else:
                                        rowwise_layers.add(part)

                    if colwise_layers or rowwise_layers:
                        # Handle QKV fusion: adjust layer names based on fusion setting
                        if self._use_fused_qkv:
                            # Using fused QKV: add qkv_proj, remove separate q/k/v_proj
                            colwise_layers.add("qkv_proj")
                            colwise_layers.discard("q_proj")
                            colwise_layers.discard("k_proj")
                            colwise_layers.discard("v_proj")
                        else:
                            # Not using fused QKV: ensure separate projections
                            colwise_layers.discard("qkv_proj")
                            colwise_layers.update(["q_proj", "k_proj", "v_proj"])

                        # Handle Gate+Up fusion: adjust layer names based on fusion setting
                        if self._use_fused_ffn:
                            # Using fused FFN: add up_gate_proj, remove separate gate/up_proj
                            colwise_layers.add("up_gate_proj")
                            colwise_layers.discard("gate_proj")
                            colwise_layers.discard("up_proj")
                        else:
                            # Not using fused FFN: ensure separate projections
                            colwise_layers.discard("up_gate_proj")
                            colwise_layers.update(["gate_proj", "up_proj"])

                        converted_plan = {}
                        for layer in colwise_layers:
                            converted_plan[rf"\.{layer}$"] = "colwise"
                        for layer in rowwise_layers:
                            converted_plan[rf"\.{layer}$"] = "rowwise"
                        return converted_plan
            except Exception as e:
                logger.warning(f"Failed to get PaddleFormers TP mappings: {e}, using default")

        # Default naming-based TP plan
        return {
            # Column Parallel (output dimension split)
            r"\.qkv_proj$": "colwise",  # Fused QKV projection
            r"\.up_gate_proj$": "colwise",  # Fused FFN projection
            r"\.q_proj$": "colwise",
            r"\.k_proj$": "colwise",
            r"\.v_proj$": "colwise",
            r"\.gate_proj$": "colwise",
            r"\.up_proj$": "colwise",
            # Row Parallel (input dimension split)
            r"\.o_proj$": "rowwise",
            r"\.down_proj$": "rowwise",
        }

    def create_attention_instances(self) -> dict[int, Attention]:
        """Create FastDeploy attention instances for all layers.

        These instances replace PaddleFormers' attention and are passed to model.forward().
        For centralized deployment, create instances for all layers.
        """
        num_layers = self.text_config.num_hidden_layers

        layer_types = getattr(self.text_config, "layer_types", None)
        sliding_window = getattr(self.text_config, "sliding_window", None)

        if layer_types is None:
            sliding_window_pattern = getattr(self.text_config, "sliding_window_pattern", None)
            if sliding_window_pattern is not None and sliding_window is not None:
                layer_types = [
                    "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                    for i in range(num_layers)
                ]

        if layer_types is not None:
            if not hasattr(self.fd_config.model_config, "layer_types"):
                self.fd_config.model_config.layer_types = layer_types
            if not hasattr(self.fd_config.model_config, "sliding_window") and sliding_window is not None:
                self.fd_config.model_config.sliding_window = sliding_window

        attention_instances = {}
        for i in range(num_layers):
            attention_instances[i] = Attention(
                fd_config=self.fd_config,
                layer_id=i,
            )

        return attention_instances

    def embed_input_ids(self, input_ids: paddle.Tensor) -> paddle.Tensor:
        """Embed input_ids using the model's embedding layer."""
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        if hasattr(self, "embed_scale") and self.embed_scale is not None:
            inputs_embeds *= self.embed_scale
        return inputs_embeds

    @paddle.no_grad()
    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
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
        num_tokens = ids_remove_padding.shape[0]

        batch_id_per_token = forward_meta.batch_id_per_token  # [num_tokens]
        seq_lens_decoder = forward_meta.seq_lens_decoder  # [batch_size, 1]

        if batch_id_per_token is not None and seq_lens_decoder is not None:
            decoder_offsets = seq_lens_decoder.squeeze(-1)  # [batch_size]
            token_decoder_offsets = paddle.index_select(decoder_offsets, batch_id_per_token, axis=0)  # [num_tokens]

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

        inputs_embeds = self.embed_input_ids(ids_remove_padding).unsqueeze(0)

        if getattr(self.text_config, "uses_mrope", False):
            position_ids = position_ids.unsqueeze(1)
        else:
            position_ids = position_ids.unsqueeze(0)

        forward_meta.rope_already_applied = True
        self.paddleformers_config.forward_meta = forward_meta

        model_output = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=position_ids,
            return_dict=False,
            **kwargs,
        )

        hidden_states = model_output[0][0, ...]  # Remove batch dim

        return hidden_states

    @paddle.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, paddle.Tensor]]):
        """Load weights from checkpoint into model parameters."""
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        sublayers_dict = dict(self.named_sublayers())
        process_fn = process_weights_after_loading(sublayers_dict, self.fd_config)
        params_dict = dict(self.named_parameters())

        # === 前缀别名处理 ===
        model_type = str(getattr(self.paddleformers_config, "model_type", "") or "").lower()
        ckpt_prefix_aliases = {model_type, model_type.replace("-", "_"), model_type.replace("_", "")} - {""}
        ckpt_alias_markers = (".layers.", ".embed_tokens.", ".lm_head.", ".norm.", ".final_layernorm.", ".rotary_emb.")

        def resolve_param_name(weight_name: str) -> str | None:
            # 动态收集前缀别名
            if "." in weight_name:
                prefix = weight_name.split(".", 1)[0]
                if prefix not in {"model", "lm_head"} and any(m in weight_name for m in ckpt_alias_markers):
                    ckpt_prefix_aliases.add(prefix)

            # 生成候选名称
            candidates = [weight_name]
            candidates.append(weight_name[6:] if weight_name.startswith("model.") else "model." + weight_name)
            if "." in weight_name:
                prefix, rest = weight_name.split(".", 1)
                if prefix in ckpt_prefix_aliases:
                    candidates.extend([rest, "model." + rest])

            return next((c for c in candidates if c in params_dict), None)

        # === 权重映射配置 ===
        stacked_params_mapping = [
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]
        if self._use_fused_ffn:
            stacked_params_mapping += [("up_gate_proj", "gate_proj", "gate"), ("up_gate_proj", "up_proj", "up")]

        # === QKV 融合相关 ===
        mc = self.fd_config.model_config
        model_format = str(getattr(mc, "model_format", "") or "").lower()
        qkv_buffer, qkv_bias_buffer = {}, {}

        def parse_qkv_name(name: str) -> tuple[str, str, str] | None:
            for proj, ptype in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
                if proj in name:
                    layer_key = name.replace(f".{proj}.weight", "").replace(f".{proj}.bias", "")
                    return layer_key, ptype, name.replace(proj, "qkv_proj")
            return None

        def fuse_qkv(q, k, v, is_bias: bool) -> paddle.Tensor:
            num_heads, num_kv_heads, head_dim = mc.num_attention_heads, mc.num_key_value_heads, mc.head_dim
            hidden_size, num_kv_groups = mc.hidden_size, num_heads // num_kv_heads
            q_out, kv_out = num_heads * head_dim, num_kv_heads * head_dim

            if is_bias:
                # 校验 bias 维度和形状
                if q.ndim != 1 or k.ndim != 1 or v.ndim != 1:
                    raise ValueError(f"Unexpected qkv bias dims: q={q.shape}, k={k.shape}, v={v.shape}; expected 1D.")
                if q.shape[0] != q_out or k.shape[0] != kv_out or v.shape[0] != kv_out:
                    raise ValueError(
                        f"Unexpected qkv bias shapes: q={q.shape}, k={k.shape}, v={v.shape}; "
                        f"expected q=[{q_out}], k/v=[{kv_out}]."
                    )
                return paddle.concat(
                    [
                        q.reshape([num_kv_heads, num_kv_groups, head_dim]),
                        k.reshape([num_kv_heads, 1, head_dim]),
                        v.reshape([num_kv_heads, 1, head_dim]),
                    ],
                    axis=1,
                ).reshape([-1])

            # 校验 weight 形状和 model_format
            q_shape, k_shape, v_shape = [int(x) for x in q.shape], [int(x) for x in k.shape], [int(x) for x in v.shape]
            torch_layout = (
                q_shape == [q_out, hidden_size]
                and k_shape == [kv_out, hidden_size]
                and v_shape == [kv_out, hidden_size]
            )
            paddle_layout = (
                q_shape == [hidden_size, q_out]
                and k_shape == [hidden_size, kv_out]
                and v_shape == [hidden_size, kv_out]
            )

            if model_format == "torch":
                if not torch_layout:
                    raise ValueError(
                        f"model_format=torch requires torch layout, got q={q_shape}, k={k_shape}, v={v_shape}."
                    )
                q, k, v = q.T, k.T, v.T
            elif model_format == "paddle":
                if not paddle_layout:
                    raise ValueError(
                        f"model_format=paddle requires paddle layout, got q={q_shape}, k={k_shape}, v={v_shape}."
                    )
            else:
                raise ValueError(f"Unsupported model_format: {model_format}. Expect 'torch' or 'paddle'.")

            # 转置后校验
            if q.shape[0] != hidden_size or k.shape[0] != hidden_size or v.shape[0] != hidden_size:
                raise ValueError(
                    f"QKV shape mismatch after normalization: q={list(q.shape)}, k={list(k.shape)}, v={list(v.shape)}."
                )

            fused = paddle.concat(
                [
                    q.reshape([hidden_size, num_kv_heads, num_kv_groups, head_dim]),
                    k.reshape([hidden_size, num_kv_heads, 1, head_dim]),
                    v.reshape([hidden_size, num_kv_heads, 1, head_dim]),
                ],
                axis=2,
            ).reshape([hidden_size, -1])

            return fused.T if model_format == "torch" else fused

        # === 辅助函数 ===
        def load_param(name: str, tensor: paddle.Tensor, shard_id=None, no_transpose: bool = False):
            param = params_dict[name]
            if no_transpose and hasattr(param, "weight_need_transpose"):
                param.weight_need_transpose = False
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            weight_loader(param, tensor, shard_id)
            process_fn(re.sub(r"\.(weight|bias)$", "", name), param)

        # === 主循环 ===
        loaded_count = skipped_count = 0

        for weight_name, weight in weights:
            # 1. QKV 融合处理
            if self._use_fused_qkv and (qkv_info := parse_qkv_name(weight_name)):
                layer_key, proj_type, qkv_param_name = qkv_info
                is_bias = ".bias" in weight_name
                buf = qkv_bias_buffer if is_bias else qkv_buffer
                buf.setdefault(layer_key, {})[proj_type] = weight

                if len(buf[layer_key]) == 3:
                    resolved = resolve_param_name(qkv_param_name)
                    if resolved:
                        fused = fuse_qkv(buf[layer_key]["q"], buf[layer_key]["k"], buf[layer_key]["v"], is_bias)
                        load_param(resolved, fused, no_transpose=not is_bias)
                        loaded_count += 3
                    else:
                        logger.warning(f"QKV {'bias ' if is_bias else ''}param {qkv_param_name} not found")
                        skipped_count += 3
                    del buf[layer_key]
                continue

            # 2. Stacked params mapping
            for param_name, src_name, shard_id in stacked_params_mapping:
                if src_name in weight_name:
                    resolved = resolve_param_name(weight_name.replace(src_name, param_name))
                    if resolved:
                        load_param(resolved, weight, shard_id)
                        loaded_count += 1
                    else:
                        logger.warning(f"Stacked mapping: {weight_name} -> NOT FOUND")
                    break
            else:
                # 3. 直接加载
                resolved = resolve_param_name(weight_name)
                if resolved:
                    try:
                        load_param(resolved, weight)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {resolved}: {e}")
                        skipped_count += 1
                else:
                    skipped_count += 1

        logger.info(f"Weight loading: {loaded_count} loaded, {skipped_count} skipped")

        # === tie_word_embeddings 处理 ===
        if hasattr(self, "lm_head") and getattr(self, "tie_word_embeddings", False):
            embed = self.model.get_input_embeddings()
            if hasattr(embed, "embeddings") and hasattr(embed.embeddings, "weight"):
                self.lm_head.linear.weight.set_value(embed.embeddings.weight.T)
            else:
                logger.warning("tie_word_embeddings=True but embed_tokens.embeddings.weight not found!")
