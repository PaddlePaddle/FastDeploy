"""Elastic-Attention port of Qwen3 (PawQwen3ForCausalLM).

Reuses Qwen3 weight-loading / TP mappings; replaces ``Qwen3Attention`` with
:class:`Qwen3ElasticAttention` which (a) hosts the per-layer
:class:`AttentionRouter` MLP and config knobs, and (b) routes the actual
attention compute through :class:`Qwen3ElasticAttentionBackend` (see
``fastdeploy/model_executor/layers/attention/elastic_attn_backend.py``).

The integration spec lives in ``ELASTIC_FASTDEPLOY_INTEGRATION.md``; in
particular §4 (model registration), §5 (router + utils) and §8 (config).
"""

from __future__ import annotations

import re
from functools import partial
from typing import Dict

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.elastic_attn_backend import (
    Qwen3ElasticAttentionBackend,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import QKRMSNorm, RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2MLP

from .config_elastic import populate_elastic_fields
from .utils import AttentionRouter


class Qwen3ElasticMLP(Qwen2MLP):
    pass


class Qwen3ElasticAttention(nn.Layer):
    """Qwen3 attention with the Elastic-Attention router head.

    Backend-side ``Qwen3ElasticAttentionBackend.forward_mixed`` reads the
    per-layer config knobs and the ``mask_allocator`` MLP to decide
    head_mask_type and dispatch to BSA on the prefill leg; on decode it
    behaves identically to vanilla Qwen3 (``append_attention``).
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        populate_elastic_fields(fd_config.model_config)

        self.fd_config = fd_config
        self.layer_id = layer_id
        self.head_dim = fd_config.model_config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        num_kv_heads_replicas = max(1, tp_size // fd_config.model_config.num_key_value_heads)
        self.num_heads_local = fd_config.model_config.num_attention_heads // tp_size
        self.num_kv_heads_local = max(
            1, fd_config.model_config.num_key_value_heads * num_kv_heads_replicas // tp_size
        )
        self.q_size = self.num_heads_local * self.head_dim
        self.kv_size = self.num_kv_heads_local * self.head_dim

        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.head_dim * fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
            layer_id=layer_id,
        )
        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )
        self.qk_norm = QKRMSNorm(
            fd_config,
            head_dim=self.head_dim,
            q_size=self.q_size,
            kv_size=self.kv_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=prefix,
            begin_norm_axis=2,
        )

        # ---- Elastic-Attention specific ----
        # Router MLP (loaded from ckpt weights ``mask_allocator.*``)
        self.mask_allocator = AttentionRouter(
            num_kv_heads=self.num_kv_heads_local,
            d_feature=self.head_dim,
        )
        # Trained scalar bias (kept for strict-load compat; not used at inference).
        self.attn_mask_log_alphas = self.create_parameter(
            shape=[self.num_kv_heads_local],
            default_initializer=nn.initializer.Constant(0.0),
        )

        mc = fd_config.model_config
        self.sink_size = int(mc.sink_size)
        self.local_window_size = int(mc.local_window_size)
        self.toggle_type = str(mc.toggle_type)
        self.retrieval_mode = str(mc.retrieval_mode)
        self.enable_ada_sparsity = bool(mc.enable_ada_sparsity)
        self.pooling_mode = str(mc.pooling_mode)
        # IMPORTANT: read elastic ``block_size`` (xattn / BSA granularity, default
        # 128, matching PyTorch reference's ``self.granularity = getattr(config,
        # "block_size", 128)``) directly from the ckpt's pretrained_config.
        # We MUST NOT read from ``model_config`` here because FastDeploy's
        # ``cache_config.block_size = 64`` (KV-cache block size) leaks onto
        # ``model_config`` via attribute proxying in some configs, which
        # silently corrupts the xattn/BSA block grid (sink_blocks/local_blocks
        # halve, BSA mask grid no longer aligns with token blocks -> garbled
        # output). The ckpt's config.json has no ``block_size`` field, so we
        # fall back to 128.
        _pc = getattr(mc, "pretrained_config", None) or mc
        self.block_size = int(getattr(_pc, "block_size", 128))
        self.sink_blocks = (self.sink_size + self.block_size - 1) // self.block_size
        self.local_blocks = (self.local_window_size + self.block_size - 1) // self.block_size
        self.xattn_stride = int(mc.xattn_stride)
        self.xattn_threshold = float(mc.xattn_threshold)
        self.xattn_norm = float(mc.xattn_norm)

        # router decision cache (filled by backend on prefill)
        self._z_kv_cache = paddle.zeros([self.num_kv_heads_local], dtype="int32")
        self._head_mask_type_cache = paddle.zeros([self.num_heads_local], dtype="int32")

        # ---- Inject elastic attrs onto self.attn ----
        # The attention backend's ``forward_mixed`` receives
        # ``layer = self.attn`` (the inner ``Attention`` instance, see
        # ``layers/attention/attention.py:280-289``), NOT this parent. The
        # elastic backend reads ``layer.mask_allocator`` / ``layer.toggle_type``
        # etc., so we mirror these handles onto the ``Attention`` instance.
        # Using object.__setattr__ to avoid triggering nn.Layer's
        # parameter/sublayer registration twice.
        for _name in (
            "mask_allocator",
            "toggle_type",
            "retrieval_mode",
            "enable_ada_sparsity",
            "pooling_mode",
            "block_size",
            "sink_blocks",
            "local_blocks",
            "xattn_stride",
            "xattn_threshold",
            "xattn_norm",
            "_z_kv_cache",
            "_head_mask_type_cache",
        ):
            object.__setattr__(self.attn, _name, getattr(self, _name))

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        self.qk_norm.load_state_dict(state_dict)
        self.attn.load_state_dict(state_dict)

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = self.qkv_proj(hidden_states)
        qkv_out = self.qk_norm(qkv_out, forward_meta)
        atten_out = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        return self.o_proj(atten_out)


class Qwen3ElasticDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, fd_config: FDConfig, prefix: str = "") -> None:
        super().__init__(fd_config, prefix)
        layer_id = int(prefix.split(sep=".")[-1])
        self.self_attn = Qwen3ElasticAttention(
            fd_config=fd_config, layer_id=layer_id, prefix=f"{prefix}.self_attn"
        )


@support_graph_optimization
class Qwen3ElasticModel(nn.Layer):
    def __init__(self, fd_config: FDConfig | None = None):
        super().__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )
        self.layers = nn.LayerList(
            [
                Qwen3ElasticDecoderLayer(
                    fd_config=fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)
        return self.norm(hidden_states, residual)[0]


@ModelRegistry.register_model_class(
    architecture="PawQwen3ForCausalLM",
    module_name="qwen3_elastic",
    category=[ModelCategory.TEXT_GENERATION],
    primary_use=ModelCategory.TEXT_GENERATION,
)
class PawQwen3ForCausalLM(ModelForCasualLM):
    """Elastic-Attention Qwen3 (full_xattn / streaming) for inference."""

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        self.fd_config = fd_config
        populate_elastic_fields(fd_config.model_config)

        # Force ``architectures[0]`` to start with "Qwen" so that
        # ``rotary_embedding.get_rope_impl()`` picks ``QwenRotaryEmbedding``
        # (rotary_dim=128, neox-style) instead of falling through to
        # ``ErnieRotaryEmbedding`` (rotary_dim//2=64). Mismatched RoPE shape
        # silently corrupts every layer and the model collapses to emitting
        # token id 0 ("!" repeatedly). The original architecture name is kept
        # in ``ModelRegistry`` because dispatch already happened before this.
        archs = list(getattr(fd_config.model_config, "architectures", []) or [])
        if archs and not archs[0].startswith("Qwen"):
            archs[0] = "Qwen3" + archs[0]
            fd_config.model_config.architectures = archs

        self.model = Qwen3ElasticModel(fd_config=fd_config)
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(cls):
        return "PawQwen3ForCausalLM"

    @classmethod
    def _get_attn_backend_cls(cls, *args, **kwargs):
        return Qwen3ElasticAttentionBackend

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        is_pooling_model = hasattr(self, "is_pooling_model") and self.is_pooling_model
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("qk_norm.q_norm", "q_norm", None),
            ("qk_norm.k_norm", "k_norm", None),
        ]

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(
            dict(self.named_sublayers()), self.fd_config
        )

        # Training-only keys we silently drop.
        skip_substrings = (
            ".mask_allocator.log_temp",
        )

        for loaded_weight_name, loaded_weight in weights_iterator:
            if any(s in loaded_weight_name for s in skip_substrings):
                continue
            logger.debug(f"Loading weight: {loaded_weight_name}")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        if self.tie_word_embeddings and not is_pooling_model:
            self.lm_head.linear.weight.set_value(
                self.model.embed_tokens.embeddings.weight.transpose([1, 0]).astype(
                    self.lm_head.linear.weight.dtype
                )
            )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.load_state_dict(
                {self.lm_head.weight_key: self.model.embed_tokens.embeddings.weight}
            )
        else:
            self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        ids_remove_padding = inputs["ids_remove_padding"]
        return self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

    def clear_graph_opt_backend(self):
        self.model.clear_graph_opt_backend(fd_config=self.fd_config)


class PawQwen3PretrainedModel(PretrainedModel):
    """TP mapping: identical to Qwen3 + extra mask_allocator router (replicated)."""

    config_class = FDConfig

    def _init_weight(self, layer):
        return None

    @classmethod
    def arch_name(cls):
        return "PawQwen3ForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}
            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                "layers.0.self_attn.q_proj.weight": partial(fn, is_column=True),
                "layers.0.self_attn.q_proj.bias": partial(fn, is_column=True),
                "layers.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.up_proj.weight": partial(fn, is_column=True),
            }
            if config.num_key_value_heads % config.tensor_model_parallel_size == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action
            # Router MLP is small -- replicate (no TP split) by simply not adding mappings.
            return final_actions

        return get_tensor_parallel_split_mappings(config.num_hidden_layers)
