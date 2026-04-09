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
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.utils import WeightsMapper, slice_fn


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


class PaddleFormersQKVParallelLinear(QKVParallelLinear):
    """PF-specific QKV loader that packs local shards in PF interleaved order."""

    def __init__(self, fd_config, prefix: str, with_bias: bool = False):
        super().__init__(fd_config=fd_config, prefix=prefix, with_bias=with_bias)
        self._pending_local_shards: dict[int, dict[str, paddle.Tensor]] = {}
        self._model_format = str(getattr(fd_config.model_config, "model_format", "") or "").lower()

    @staticmethod
    def _to_tensor(t: paddle.Tensor | object) -> paddle.Tensor:
        return t if isinstance(t, paddle.Tensor) else paddle.to_tensor(t)

    def _extract_local_shard(self, param: paddle.Tensor, loaded_weight: paddle.Tensor, loaded_shard_id: str):
        output_dim = getattr(param, "output_dim", None)
        if output_dim is None:
            raise ValueError("Missing output_dim for QKV parameter.")

        dim = -1 if output_dim else 0
        denom = self.num_heads_per_rank + 2 * self.kv_num_heads_per_rank
        head_dim = int(param.shape[dim]) // int(denom)

        weight = self._to_tensor(loaded_weight)
        if getattr(param, "weight_need_transpose", False):
            if weight.ndim != 2:
                raise ValueError(f"Expected 2D tensor for transpose, got shape={list(weight.shape)}")
            weight = weight.transpose([1, 0])

        if self.tp_size > 1 and output_dim is not None and not self.fd_config.load_config.is_pre_sharded:
            block_size = self._get_shard_size_mapping(loaded_shard_id, head_dim)
            shard_id = self.local_rank if loaded_shard_id == "q" else self.local_rank // self.num_kv_head_replicas
            shard_offset = shard_id * block_size
            weight = slice_fn(weight, output_dim, start=shard_offset, end=shard_offset + block_size)

        return weight

    @staticmethod
    def _to_hidden_major(weight: paddle.Tensor, expected_out: int, name: str) -> paddle.Tensor:
        if weight.ndim != 2:
            raise ValueError(f"Expected 2D {name} shard, got shape={list(weight.shape)}")

        s0, s1 = int(weight.shape[0]), int(weight.shape[1])
        if s1 == expected_out:
            return weight
        if s0 == expected_out:
            return weight.transpose([1, 0])
        raise ValueError(
            f"Cannot normalize {name} shard shape={list(weight.shape)} to hidden-major with expected_out={expected_out}."
        )

    def _pack_pf_interleaved_local(
        self,
        q_local: paddle.Tensor,
        k_local: paddle.Tensor,
        v_local: paddle.Tensor,
        output_dim: bool,
    ):
        kv_local = int(self.kv_num_heads_per_rank)
        if kv_local <= 0:
            raise ValueError("Invalid kv_num_heads_per_rank, must be > 0.")
        if self.num_heads_per_rank % kv_local != 0:
            raise ValueError(
                f"num_heads_per_rank={self.num_heads_per_rank} is not divisible by kv_num_heads_per_rank={kv_local}"
            )
        q_groups_local = self.num_heads_per_rank // kv_local

        if q_local.ndim == 1:
            q = q_local.reshape([kv_local, q_groups_local, self.head_dim])
            k = k_local.reshape([kv_local, 1, self.head_dim])
            v = v_local.reshape([kv_local, 1, self.head_dim])
            return paddle.concat([q, k, v], axis=1).reshape([-1])

        q_out = kv_local * q_groups_local * self.head_dim
        kv_out = kv_local * self.head_dim

        q_hm = self._to_hidden_major(q_local, q_out, "q")
        k_hm = self._to_hidden_major(k_local, kv_out, "k")
        v_hm = self._to_hidden_major(v_local, kv_out, "v")

        hidden_size = int(q_hm.shape[0])
        if int(k_hm.shape[0]) != hidden_size or int(v_hm.shape[0]) != hidden_size:
            raise ValueError(
                "Q/K/V hidden dimension mismatch after normalization: "
                f"q={list(q_hm.shape)}, k={list(k_hm.shape)}, v={list(v_hm.shape)}"
            )

        q = q_hm.reshape([hidden_size, kv_local, q_groups_local, self.head_dim])
        k = k_hm.reshape([hidden_size, kv_local, 1, self.head_dim])
        v = v_hm.reshape([hidden_size, kv_local, 1, self.head_dim])
        packed_hidden_major = paddle.concat([q, k, v], axis=2).reshape([hidden_size, -1])

        if output_dim:
            return packed_hidden_major
        return packed_hidden_major.transpose([1, 0])

    def _split_pf_fused_qkv(self, loaded_weight: paddle.Tensor, is_bias: bool):
        if self._model_format != "paddle":
            raise ValueError(
                "Direct qkv_proj loading is only supported for model_format='paddle'. "
                "Use split q_proj/k_proj/v_proj weights for other formats."
            )

        weight = self._to_tensor(loaded_weight)
        if is_bias:
            if weight.ndim != 1:
                raise ValueError(f"Unexpected fused qkv bias dims: {list(weight.shape)}, expected 1D.")
            width = int(weight.shape[0])
        else:
            if weight.ndim != 2:
                raise ValueError(f"Unexpected fused qkv weight dims: {list(weight.shape)}, expected 2D.")
            width = int(weight.shape[1])

        global_width = int((self.num_heads + 2 * self.kv_num_heads) * self.head_dim)
        local_width = int((self.num_heads_per_rank + 2 * self.kv_num_heads_per_rank) * self.head_dim)

        if width == global_width:
            num_heads, num_kv_heads = self.num_heads, self.kv_num_heads
        elif width == local_width:
            num_heads, num_kv_heads = self.num_heads_per_rank, self.kv_num_heads_per_rank
        else:
            raise ValueError(
                f"Cannot validate fused qkv_proj width={width}. "
                f"Expect global={global_width} or local={local_width} for PF interleaved layout."
            )

        if num_heads % num_kv_heads != 0:
            raise ValueError(f"Invalid head config: num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        q_groups = num_heads // num_kv_heads

        if is_bias:
            fused = weight.reshape([num_kv_heads, q_groups + 2, self.head_dim])
            q = fused[:, :q_groups, :].reshape([-1])
            k = fused[:, q_groups : q_groups + 1, :].reshape([-1])
            v = fused[:, q_groups + 1 :, :].reshape([-1])
            return q, k, v

        hidden_size = int(weight.shape[0])
        fused = weight.reshape([hidden_size, num_kv_heads, q_groups + 2, self.head_dim])
        q = fused[:, :, :q_groups, :].reshape([hidden_size, -1])
        k = fused[:, :, q_groups : q_groups + 1, :].reshape([hidden_size, -1])
        v = fused[:, :, q_groups + 1 :, :].reshape([hidden_size, -1])
        return q, k, v

    def weight_loader(self, param, loaded_weight, loaded_shard_id: str | None = None):
        if loaded_shard_id is None:
            is_bias = len(param.shape) == 1
            q_shard, k_shard, v_shard = self._split_pf_fused_qkv(loaded_weight, is_bias=is_bias)
            self.weight_loader(param, q_shard, "q")
            self.weight_loader(param, k_shard, "k")
            self.weight_loader(param, v_shard, "v")
            return

        if loaded_shard_id not in {"q", "k", "v"}:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
            return

        local_shard = self._extract_local_shard(param, loaded_weight, loaded_shard_id)
        key = id(param)
        pending = self._pending_local_shards.setdefault(key, {})
        pending[loaded_shard_id] = local_shard

        if len(pending) < 3:
            setattr(param, "_pf_qkv_pending", True)
            return

        packed = self._pack_pf_interleaved_local(
            pending["q"],
            pending["k"],
            pending["v"],
            output_dim=bool(getattr(param, "output_dim", True)),
        )
        if not param._is_initialized():
            param.initialize()
        if packed.dtype != param.dtype:
            packed = packed.cast(param.dtype)
        if list(param.shape) != list(packed.shape):
            raise ValueError(f"Packed qkv shape mismatch: packed={list(packed.shape)} param={list(param.shape)}")

        param.set_value(packed)
        del self._pending_local_shards[key]
        setattr(param, "_pf_qkv_pending", False)


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

    tp_size = 1
    if hasattr(self_attn, "fd_config") and hasattr(self_attn.fd_config, "parallel_config"):
        tp_size = int(getattr(self_attn.fd_config.parallel_config, "tensor_parallel_size", 1) or 1)

    # Resolve head-related metadata.
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

    # Support only 3D (HSD/SHD) or 4D (BHSD/BSHD with B=1) inputs.
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
        if tp_size > 1 and expected_heads % tp_size == 0:
            expected_heads //= tp_size
        return actual_heads == expected_heads

    # Determine layout from Q/K/V head axes; keep default behavior on ambiguity.
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

    # Sequence lengths must match after flattening Q/K/V.
    q_seq, k_seq, v_seq = int(q_flat.shape[0]), int(k_flat.shape[0]), int(v_flat.shape[0])
    if not (q_seq == k_seq == v_seq):
        raise ValueError(
            f"Sequence length mismatch after flattening: Q={q_seq}, K={k_seq}, V={v_seq}, "
            f"raw query={list(query.shape)}, key={list(key.shape)}, value={list(value.shape)}."
        )

    # If forward_meta provides ids_remove_padding, strictly validate Q sequence length.
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
        self._use_fused_qkv = model_type in supported_fused_qkv_models
        if self._use_fused_qkv:
            self.paddleformers_config.fuse_attention_qkv = True
            logger.info(f"Enabled fuse_attention_qkv for model_type={model_type}, tp={tp_size}")
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
        # Patch PF attention head counts to TP-local values for fused qkv reshape
        self._localize_pf_attention_heads()
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
                        # qkv_proj uses PF-specific TP-aware loader to support
                        # unified split-QKV loading across TP1/TP>1.
                        if "qkv_proj" in qual_name and self._use_fused_qkv:
                            new_module = PaddleFormersQKVParallelLinear(
                                self.fd_config,
                                prefix=qual_name,
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

    def _localize_pf_attention_heads(self):
        """Patch PF attention modules' head counts to TP-local values.

        PF Attention.__init__ reads global head counts from config and stores
        them as instance attrs (num_heads, num_key_value_heads, etc.).
        Since we cannot set config.tensor_model_parallel_size > 1 (it would
        trigger PF's own TP linears, conflicting with recursive_replace),
        we patch the instance attrs directly after model creation.

        Only needed when fused qkv is enabled, because the PF forward path
        reshapes qkv_proj output using these head counts.
        """
        tp_size = self.fd_config.parallel_config.tensor_parallel_size
        if tp_size <= 1 or not self._use_fused_qkv:
            return

        g_heads = int(self.text_config.num_attention_heads)
        g_kv = int(getattr(self.text_config, "num_key_value_heads", g_heads))
        local_heads = g_heads // tp_size
        local_kv = max(1, g_kv // tp_size)
        local_groups = local_heads // local_kv

        patched = 0
        for name, module in self.model.named_sublayers():
            # PF attention modules store head counts as instance attrs used in forward reshape
            if not hasattr(module, "num_key_value_groups"):
                continue
            module.num_heads = local_heads
            module.num_key_value_heads = local_kv
            module.num_key_value_groups = local_groups
            patched += 1

        if patched:
            logger.info(
                f"Localized {patched} PF attention modules: "
                f"heads {g_heads}->{local_heads}, kv {g_kv}->{local_kv}, tp={tp_size}"
            )

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

        # === Checkpoint prefix alias handling ===
        model_type = str(getattr(self.paddleformers_config, "model_type", "") or "").lower()
        ckpt_prefix_aliases = {model_type, model_type.replace("-", "_"), model_type.replace("_", "")} - {""}
        ckpt_alias_markers = (".layers.", ".embed_tokens.", ".lm_head.", ".norm.", ".final_layernorm.", ".rotary_emb.")

        def resolve_param_name(weight_name: str) -> str | None:
            # Collect prefix aliases dynamically.
            if "." in weight_name:
                prefix = weight_name.split(".", 1)[0]
                if prefix not in {"model", "lm_head"} and any(m in weight_name for m in ckpt_alias_markers):
                    ckpt_prefix_aliases.add(prefix)

            # Generate candidate parameter names.
            candidates = [weight_name]
            candidates.append(weight_name[6:] if weight_name.startswith("model.") else "model." + weight_name)
            if "." in weight_name:
                prefix, rest = weight_name.split(".", 1)
                if prefix in ckpt_prefix_aliases:
                    candidates.extend([rest, "model." + rest])

            return next((c for c in candidates if c in params_dict), None)

        # === Stacked parameter mapping config ===
        stacked_params_mapping = [
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]
        if self._use_fused_ffn:
            stacked_params_mapping += [("up_gate_proj", "gate_proj", "gate"), ("up_gate_proj", "up_proj", "up")]

        # === QKV loading helpers ===
        qkv_split_layers: set[str] = set()
        qkv_direct_pending: dict[tuple[str, bool], tuple[str, paddle.Tensor]] = {}

        def parse_qkv_shard_name(name: str) -> tuple[str, str, str] | None:
            shard_suffixes = (
                (".q_proj.weight", "q"),
                (".k_proj.weight", "k"),
                (".v_proj.weight", "v"),
                (".q_proj.bias", "q"),
                (".k_proj.bias", "k"),
                (".v_proj.bias", "v"),
            )
            for suffix, shard_id in shard_suffixes:
                if name.endswith(suffix):
                    layer_key = name.replace(suffix, "")
                    qkv_param_name = name.replace(".q_proj.", ".qkv_proj.")
                    qkv_param_name = qkv_param_name.replace(".k_proj.", ".qkv_proj.")
                    qkv_param_name = qkv_param_name.replace(".v_proj.", ".qkv_proj.")
                    return layer_key, shard_id, qkv_param_name
            return None

        def parse_direct_qkv_name(name: str) -> tuple[str, bool] | None:
            if name.endswith(".qkv_proj.weight"):
                return name.replace(".qkv_proj.weight", ""), False
            if name.endswith(".qkv_proj.bias"):
                return name.replace(".qkv_proj.bias", ""), True
            return None

        # === Helper functions ===
        def load_param(name: str, tensor: paddle.Tensor, shard_id=None):
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            weight_loader(param, tensor, shard_id)
            if shard_id in {"q", "k", "v"} and bool(getattr(param, "_pf_qkv_pending", False)):
                return False
            process_fn(re.sub(r"\.(weight|bias)$", "", name), param)
            return True

        # === Main loading loop ===
        loaded_count = skipped_count = 0

        for weight_name, weight in weights:
            # 1. Handle fused QKV path in a unified split-shard style.
            if self._use_fused_qkv:
                if qkv_info := parse_qkv_shard_name(weight_name):
                    layer_key, proj_type, qkv_param_name = qkv_info
                    qkv_split_layers.add(layer_key)
                    resolved = resolve_param_name(qkv_param_name)
                    if resolved:
                        try:
                            load_param(resolved, weight, shard_id=proj_type)
                            loaded_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to load qkv shard {weight_name} -> {resolved}: {e}")
                            skipped_count += 1
                    else:
                        logger.warning(f"QKV shard mapping not found: {weight_name} -> {qkv_param_name}")
                        skipped_count += 1
                    continue

                if direct_qkv_info := parse_direct_qkv_name(weight_name):
                    layer_key, is_bias = direct_qkv_info
                    qkv_direct_pending[(layer_key, is_bias)] = (weight_name, weight)
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
                # 3. Direct load.
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

        # 4. Handle direct qkv_proj.* only when split q/k/v is absent for that layer.
        if self._use_fused_qkv and qkv_direct_pending:
            for (layer_key, is_bias), (weight_name, weight) in qkv_direct_pending.items():
                if layer_key in qkv_split_layers:
                    logger.info(
                        f"Skip direct qkv {'bias' if is_bias else 'weight'} for {layer_key}: "
                        "split q/k/v shards are present."
                    )
                    continue

                resolved = resolve_param_name(weight_name)
                if resolved:
                    try:
                        load_param(resolved, weight)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load direct fused qkv {weight_name} -> {resolved}: {e}")
                        skipped_count += 1
                else:
                    logger.warning(f"Direct fused qkv param not found: {weight_name}")
                    skipped_count += 1

        if self._use_fused_qkv:
            pending_qkv_params = [
                name for name, param in params_dict.items() if bool(getattr(param, "_pf_qkv_pending", False))
            ]
            if pending_qkv_params:
                raise RuntimeError(
                    "Incomplete QKV shard loading detected for parameters: " + ", ".join(sorted(pending_qkv_params))
                )

        logger.info(f"Weight loading: {loaded_count} loaded, {skipped_count} skipped")

        # === tie_word_embeddings handling ===
        if hasattr(self, "lm_head") and getattr(self, "tie_word_embeddings", False):
            embed = self.model.get_input_embeddings()
            if hasattr(embed, "embeddings") and hasattr(embed.embeddings, "weight"):
                self.lm_head.linear.weight.set_value(embed.embeddings.weight.T)
            else:
                logger.warning("tie_word_embeddings=True but embed_tokens.embeddings.weight not found!")
