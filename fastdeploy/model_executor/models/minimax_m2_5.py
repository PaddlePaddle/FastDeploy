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

from __future__ import annotations

import os
import re
import time
from typing import Dict

import numpy as np
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
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    process_weights_after_loading,
)


def _dequant_fp8_blockwise_to_bf16(fp8_weight, scale):
    """Dequant FP8 weight to BF16 using block-wise scale (numpy on CPU)."""
    BLOCK = 128
    N, K = fp8_weight.shape
    n_blocks_r = (N + BLOCK - 1) // BLOCK
    n_blocks_c = (K + BLOCK - 1) // BLOCK
    wt_f32 = fp8_weight.cast("float32").numpy()
    sc = scale.numpy()
    pad_r = n_blocks_r * BLOCK - N
    pad_c = n_blocks_c * BLOCK - K
    if pad_r > 0 or pad_c > 0:
        wt_f32 = np.pad(wt_f32, ((0, pad_r), (0, pad_c)))
    wt_blocked = wt_f32.reshape([n_blocks_r, BLOCK, n_blocks_c, BLOCK])
    sc_expanded = sc.reshape([n_blocks_r, n_blocks_c])[:, np.newaxis, :, np.newaxis]
    wt_dequant = (wt_blocked * sc_expanded).reshape([n_blocks_r * BLOCK, n_blocks_c * BLOCK])[:N, :K]
    return paddle.to_tensor(wt_dequant, dtype="bfloat16")


def _marlin_permute_scales(s, size_k, size_n, group_size):
    """Permute scales for Marlin format."""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    if group_size < size_k and group_size != -1:
        s = s.reshape([-1, len(scale_perm)])[:, scale_perm]
    else:
        s = s.reshape([-1, len(scale_perm_single)])[:, scale_perm_single]
    return s.reshape([-1, size_n]).contiguous()


def _process_fp8_marlin_weights(moe_layer, up_gate_info_list, down_info_list, up_gate_scales, down_scales, block_size):
    """Process FP8 MoE weights for Marlin kernel and load into layer.

    Writes each expert directly into the pre-allocated target tensor to avoid
    accumulating all experts in GPU memory simultaneously (OOM fix).
    """
    from fastdeploy.model_executor.ops.gpu import gptq_marlin_repack

    num_experts = len(up_gate_info_list)
    num_bits = 8
    perm = paddle.empty([0], dtype="int32")

    # Determine output shapes from expert 0
    gate_w0 = up_gate_info_list[0]["gate"]
    N_exp, K = gate_w0.shape
    N_combined = N_exp * 2
    out_rows_ug = K // 16
    out_cols_ug = N_combined * 16 // 4

    down_w0 = down_info_list[0]
    N_down, K_down = down_w0.shape
    out_rows_d = K_down // 16
    out_cols_d = N_down * 16 // 4

    # Create target tensors (CPU for large EP, will be copied to GPU by layer)
    target_ug_weight = moe_layer.up_gate_proj_weight
    target_ug_scale = moe_layer.up_gate_proj_weight_scale
    target_d_weight = moe_layer.down_proj_weight
    target_d_scale = moe_layer.down_proj_weight_scale

    for i in range(num_experts):
        gate_w = up_gate_info_list[i]["gate"]
        up_w = up_gate_info_list[i]["up"]

        stacked = paddle.stack([gate_w, up_w], axis=0).view("uint8")
        combined = stacked.reshape([N_combined, K])
        transposed = combined.T.contiguous()
        reshaped = transposed.reshape([K // 4, 4, N_combined])
        b0 = reshaped[:, 0, :].cast("int32")
        b1 = reshaped[:, 1, :].cast("int32") << 8
        b2 = reshaped[:, 2, :].cast("int32") << 16
        b3 = reshaped[:, 3, :].cast("int32") << 24
        packed = b0 | b1 | b2 | b3
        marlin_qw_flat = gptq_marlin_repack(packed, perm, K, N_combined, num_bits)[0]
        marlin_qw = marlin_qw_flat.reshape([out_rows_ug, out_cols_ug])
        target_ug_weight[i].set_value(marlin_qw)
        del stacked, combined, transposed, reshaped, b0, b1, b2, b3, packed, marlin_qw_flat, marlin_qw

        s = up_gate_scales[i].T
        n_blocks_n = s.shape[1]
        s_expanded = (
            s.unsqueeze(2).expand([s.shape[0], n_blocks_n, block_size]).reshape([s.shape[0], n_blocks_n * block_size])
        )
        s_expanded = s_expanded[:, :N_combined]
        marlin_s = _marlin_permute_scales(s_expanded, K, N_combined, block_size)
        # FP8 Marlin kernel expects scales in a specific fixed-point encoding;
        # 2**120 is the alignment factor between E4M3FN float8 and Marlin's
        # internal scale representation (cf. vLLM Marlin FP8 implementation).
        marlin_s = marlin_s * (2**120)
        target_ug_scale[i].set_value(marlin_s.cast(target_ug_scale.dtype))
        del s, s_expanded, marlin_s

        down_w = down_info_list[i]
        down_u8 = down_w.view("uint8")
        transposed_d = down_u8.T.contiguous()
        reshaped_d = transposed_d.reshape([K_down // 4, 4, N_down])
        b0_d = reshaped_d[:, 0, :].cast("int32")
        b1_d = reshaped_d[:, 1, :].cast("int32") << 8
        b2_d = reshaped_d[:, 2, :].cast("int32") << 16
        b3_d = reshaped_d[:, 3, :].cast("int32") << 24
        packed_d = b0_d | b1_d | b2_d | b3_d
        marlin_qw_d_flat = gptq_marlin_repack(packed_d, perm, K_down, N_down, num_bits)[0]
        marlin_qw_d = marlin_qw_d_flat.reshape([out_rows_d, out_cols_d])
        target_d_weight[i].set_value(marlin_qw_d)
        del down_u8, transposed_d, reshaped_d, b0_d, b1_d, b2_d, b3_d, packed_d, marlin_qw_d_flat, marlin_qw_d

        s_d = down_scales[i].T
        n_blocks_n_d = s_d.shape[1]
        s_d_expanded = (
            s_d.unsqueeze(2)
            .expand([s_d.shape[0], n_blocks_n_d, block_size])
            .reshape([s_d.shape[0], n_blocks_n_d * block_size])
        )
        s_d_expanded = s_d_expanded[:, :N_down]
        marlin_s_d = _marlin_permute_scales(s_d_expanded, K_down, N_down, block_size)
        marlin_s_d = marlin_s_d * (2**120)
        target_d_scale[i].set_value(marlin_s_d.cast(target_d_scale.dtype))
        del s_d, s_d_expanded, marlin_s_d

        # Periodic cache flush
        if i % 16 == 0:
            paddle.device.cuda.empty_cache()

    logger.info(
        f"Marlin FP8: Loaded up_gate={moe_layer.up_gate_proj_weight.shape}, down={moe_layer.down_proj_weight.shape}"
    )


class MiniMaxRMSNorm(paddle.nn.Layer):
    """
    MiniMax-M2.5 QK-RMSNorm: per-token normalization across the FULL Q or K vector.

    This matches vLLM's MiniMaxText01RMSNormTP behavior:
      variance = mean(x^2, axis=-1)  # across all features (all heads concatenated)
      x_normed = x * rsqrt(variance + eps) * weight
    """

    def __init__(self, hidden_size: int, tp_size: int, eps: float = 1e-6, weight_key: str = ""):
        super().__init__()
        self.tp_size = tp_size
        self.shard_size = hidden_size // tp_size
        self.eps = eps
        self.weight_key = weight_key
        self.weight = self.create_parameter(
            shape=[self.shard_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )

    def weight_loader(self, param, loaded_weight):
        """Load the TP shard of this weight."""
        from fastdeploy.model_executor.layers.utils import get_tensor

        w = get_tensor(loaded_weight).cast("float32")
        shard_size = w.shape[0] // self.tp_size
        tp_rank = paddle.distributed.get_rank() if self.tp_size > 1 else 0
        shard = w[tp_rank * shard_size : (tp_rank + 1) * shard_size]
        param.set_value(shard)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """x: [..., shard_size]"""
        orig_dtype = x.dtype
        x = x.cast("float32")
        if self.tp_size > 1:
            # All-reduce variance across TP ranks
            var_local = x.pow(2).mean(axis=-1, keepdim=True)
            from fastdeploy.distributed.communication import (
                tensor_model_parallel_all_reduce,
            )

            var = tensor_model_parallel_all_reduce(var_local) / self.tp_size
        else:
            var = x.pow(2).mean(axis=-1, keepdim=True)
        x = x * paddle.rsqrt(var + self.eps) * self.weight
        return x.cast(orig_dtype)


class MiniMaxM2_5Attention(nn.Layer):
    """
    MiniMax-M2.5 Attention with GQA and QK-Norm.

    Architecture:
    - GQA: 48 query heads, 8 KV heads, head_dim=128
    - Partial RoPE: rotary_dim=64 (partial_rotary_factor=0.5)
    - QK Norm: per-token full-vector RMSNorm on Q and K (MiniMaxText01RMSNormTP style)
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        self.fd_config = fd_config
        self.layer_id = layer_id
        self.head_dim = fd_config.model_config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        num_kv_heads_replicas = max(1, tp_size // fd_config.model_config.num_key_value_heads)
        self.q_size = fd_config.model_config.num_attention_heads * self.head_dim // tp_size
        self.kv_size = fd_config.model_config.num_key_value_heads * self.head_dim * num_kv_heads_replicas // tp_size

        # QKV projection
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)

        # Output projection
        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim * fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
            layer_id=layer_id,
        )

        # Attention backend (handles RoPE internally)
        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

        # QK Norm: per-token full-vector RMSNorm, matching MiniMaxText01RMSNormTP
        # Total Q features = num_attention_heads * head_dim (before TP split)
        total_q = fd_config.model_config.num_attention_heads * self.head_dim
        total_k = fd_config.model_config.num_key_value_heads * self.head_dim
        self.q_norm = MiniMaxRMSNorm(
            hidden_size=total_q,
            tp_size=tp_size,
            eps=fd_config.model_config.rms_norm_eps,
            weight_key=f"{prefix}.q_norm.weight",
        )
        self.k_norm = MiniMaxRMSNorm(
            hidden_size=total_k,
            tp_size=tp_size if fd_config.model_config.num_key_value_heads >= tp_size else 1,
            eps=fd_config.model_config.rms_norm_eps,
            weight_key=f"{prefix}.k_norm.weight",
        )

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        # q_norm and k_norm loaded via load_weights in MiniMaxM2ForCausalLM
        self.attn.load_state_dict(state_dict)

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = self.qkv_proj(hidden_states)
        # Split QKV and apply per-token QK norm
        q = qkv_out[:, : self.q_size]
        k = qkv_out[:, self.q_size : self.q_size + self.kv_size]
        v = qkv_out[:, self.q_size + self.kv_size :]
        q = self.q_norm(q)
        k = self.k_norm(k)
        qkv_normed = paddle.concat([q, k, v], axis=-1)
        attn_out = self.attn(qkv=qkv_normed, forward_meta=forward_meta)
        output = self.o_proj(attn_out)
        return output


class MiniMaxM2_5MoE(nn.Layer):
    """
    MiniMax-M2.5 MoE Block.

    All 62 layers use MoE (no dense FFN layers).
    256 experts, top-8 routing, sigmoid scoring.
    Has e_score_correction_bias for routing bias correction.
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        num_experts = fd_config.model_config.num_local_experts

        # Weight key map: MiniMax uses w1/w2/w3 naming in checkpoint + correction bias
        weight_key_map = {
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            "gate_correction_bias_key": f"{prefix}.gate.e_score_correction_bias",
        }

        # Gate projects hidden_size -> num_experts (float32, no quant)
        # Must create gate BEFORE FusedMoE so e_score_correction_bias is available
        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=num_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )
        self.gate._dump_gate = True
        self.gate._gate_layer_idx = layer_id

        # MiniMax has e_score_correction_bias for routing bias correction
        # (used by noaux_tc routing: topk by score+bias, weight by score)
        if getattr(fd_config.model_config, "use_routing_bias", True):
            self.gate.e_score_correction_bias = self.create_parameter(
                shape=[1, num_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
        else:
            self.gate.e_score_correction_bias = None

        # Create FusedMoE with the correction bias reference (only once)
        # renormalize=True: MiniMax-M2.5 normalizes top-k weights by their sum
        self.experts = FusedMoE(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            renormalize=True,
            moe_intermediate_size=fd_config.model_config.intermediate_size,
            num_experts=num_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method="noaux_tc",  # MiniMax uses sigmoid routing with correction bias
            n_group=1,  # No grouping in MiniMax (unlike DeepSeek)
            topk_group=1,
            routed_scaling_factor=1.0,
            layer_idx=layer_id,
            gate_correction_bias=self.gate.e_score_correction_bias,
            weight_key_map=weight_key_map,
        )

    def forward(self, x, forward_meta):
        return self.experts(x, self.gate, forward_meta)

    def load_state_dict(self, state_dict):
        self.gate.load_state_dict(state_dict)
        self.experts.load_state_dict(state_dict)


class MiniMaxM2_5DecoderLayer(nn.Layer):
    """
    MiniMax-M2.5 Decoder Layer.

    All layers are MoE layers (no dense FFN).
    """

    def __init__(self, fd_config: FDConfig, prefix: str = "") -> None:
        super().__init__()

        layer_id = int(prefix.split(".")[-1])
        self.layer_id = layer_id

        self.self_attn = MiniMaxM2_5Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = MiniMaxM2_5MoE(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            layer_id=layer_id,
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        hidden_states = self.self_attn(
            forward_meta=forward_meta,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states, forward_meta)

        return hidden_states, residual


@support_graph_optimization
class MiniMaxM2_5Model(nn.Layer):
    """
    MiniMax-M2.5 Transformer Model (62 decoder layers, all MoE).
    """

    def __init__(self, fd_config: FDConfig = None):
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        # Use "model" as the prefix (matches HF checkpoint structure)
        fd_config.model_config.pretrained_config.prefix_name = "model"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                MiniMaxM2_5DecoderLayer(
                    fd_config,
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
            logger.info(f"Loading layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        if self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out


@ModelRegistry.register_model_class(
    architecture="MiniMaxM2ForCausalLM",
    module_name="minimax_m2_5",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniMaxM2ForCausalLM(ModelForCasualLM):
    """
    MiniMax-M2.5 Causal Language Model.

    Registered as "MiniMaxM2ForCausalLM" to match the architecture name
    in the model's config.json.
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        # Ensure num_local_experts is accessible as num_experts for FusedMoE
        if not hasattr(fd_config.model_config, "num_experts") or fd_config.model_config.num_experts is None:
            fd_config.model_config.num_experts = fd_config.model_config.num_local_experts

        # Set partial_rotary_factor from rotary_dim / head_dim (MiniMax: 64/128 = 0.5)
        if hasattr(fd_config.model_config, "rotary_dim") and fd_config.model_config.partial_rotary_factor == 1.0:
            fd_config.model_config.partial_rotary_factor = (
                fd_config.model_config.rotary_dim / fd_config.model_config.head_dim
            )

        # moe_intermediate_size field: FD uses this name, but MiniMax config has intermediate_size
        # for expert FFN (different from a dense model's intermediate_size).
        # They are the same here: expert intermediate_size=1536.
        # FusedMoE reads fd_config.model_config.moe_intermediate_size if set,
        # but we pass it explicitly, so this is just informational.

        self.model = MiniMaxM2_5Model(fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(cls):
        return "MiniMaxM2ForCausalLM"

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """
        Map MiniMax checkpoint weight names to FD internal names.

        MiniMax checkpoint uses:
          model.layers.{i}.mlp.experts.{j}.w1.weight  -> gate_proj (shard_id="gate")
          model.layers.{i}.mlp.experts.{j}.w2.weight  -> down_proj (shard_id="down")
          model.layers.{i}.mlp.experts.{j}.w3.weight  -> up_proj   (shard_id="up")
        """
        return FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.num_local_experts,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )

    @staticmethod
    def _extract_layer_idx(weight_name: str, num_main_layers: int) -> int:
        """Extract decoder layer index from weight name. Returns -1 for non-layer weights."""
        if "model.layers." in weight_name:
            parts = weight_name.split(".")
            try:
                idx = int(parts[parts.index("layers") + 1])
                return idx
            except (ValueError, IndexError):
                pass
        return -1

    def _dequant_fp8_weights(
        self,
        fp8_weights: dict,
        fp8_scales: dict,
        params_dict: dict,
        stacked_params_mapping: list,
        expert_params_mapping: list,
        process_weights_after_loading_fn,
        block_size: int,
        sm80_keep_fp8: bool = False,
    ):
        """Dequantize a set of FP8 weights and load them into model parameters.

        If marlin_fp8 is enabled, skip expert weight dequant (they stay FP8
        and are passed to the Marlin backend).

        If sm80_keep_fp8 is True, load FP8 weights directly without dequanting.
        Also loads weight_scale_inv onto the layer for on-the-fly dequant in
        BlockWiseFP8LinearMethod.apply(). This saves ~4x memory per layer on SM80.

        NOTE: process_weights_after_loading_fn is called ONCE per unique sublayer
        AFTER all weights are loaded, to avoid repeated transpose/re-quantize
        for stacked params (qkv_proj gets Q, K, V separately).
        """
        # Track which sublayers need process_weights_after_loading (deduplicated)
        pending_process: set = set()

        _enable_marlin_fp8 = os.environ.get("FD_MARLIN_FP8", "0") == "1"

        for _wi, (wname, wt) in enumerate(fp8_weights.items()):
            # Skip expert weights if Marlin FP8 mode is active
            if _enable_marlin_fp8 and "mlp.experts" in wname:
                continue

            # Periodically flush the GPU memory pool to prevent OOM from accumulated
            # pool-held tensors. MUST synchronize first to ensure all async GPU copies
            # (from weight_loader) complete before pool memory is freed.
            if _wi > 0 and _wi % 32 == 0:
                paddle.device.synchronize()
                paddle.device.cuda.empty_cache()

            scale_name = wname.replace(".weight", ".weight_scale_inv")
            scale = fp8_scales.get(scale_name)

            if sm80_keep_fp8:
                # SM80: load FP8 weight directly, skip dequant.
                # BlockWiseFP8LinearMethod.apply() handles on-the-fly dequant.
                wt_tensor = get_tensor(wt)
                sc_tensor = get_tensor(scale) if scale is not None else None
                matched = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in wname or "mlp.experts" in wname:
                        continue
                    model_param_name = wname.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    # Load FP8 weight via weight_loader (handles stacking + TP shard)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, wt_tensor, shard_id)
                    # Load scale_inv: find the parent sublayer and set directly.
                    # NOTE: weight_loader transposes the weight from [out,in] to [in,out],
                    # but the scale_inv stays in torch layout [n_blocks_out,n_blocks_in].
                    # We'll fix the layout after all shards are loaded.
                    if sc_tensor is not None:
                        sublayer_name = model_param_name.rsplit(".", 1)[0]
                        sublayers_dict = dict(self.named_sublayers())
                        if sublayer_name in sublayers_dict:
                            parent = sublayers_dict[sublayer_name]
                            if hasattr(parent, "weight_scale_inv") and parent.weight_scale_inv is not None:
                                si = parent.weight_scale_inv
                                # For stacked params (qkv), the scale_inv param has stacked
                                # shape but we load one shard at a time. Accumulate shards
                                # in a dict and copy the full tensor at the end.
                                if shard_id is not None:
                                    if not hasattr(parent, "_scale_shards"):
                                        parent._scale_shards = {}
                                    # Shard the scale to match weight's TP shard.
                                    # Scale dim0 corresponds to weight's output_dim.
                                    # weight_loader already sharded weight along output_dim.
                                    # We need to shard scale along dim0 with the same offset/size.
                                    tp_size = self.fd_config.parallel_config.tensor_parallel_size
                                    tp_rank = self.fd_config.parallel_config.tensor_parallel_rank
                                    head_dim = getattr(parent, "head_dim", 128)
                                    num_heads_per_rank = getattr(parent, "num_heads_per_rank", 0)
                                    kv_num_heads_per_rank = getattr(parent, "kv_num_heads_per_rank", 0)
                                    if tp_size > 1:
                                        if shard_id == "q":
                                            # q scale: [num_heads_full, n_blocks_in]
                                            # TP shard: [num_heads_per_rank, n_blocks_in]
                                            # Each head = head_dim // block_size blocks
                                            blocks_per_head = head_dim // block_size
                                            shard_blocks = num_heads_per_rank * blocks_per_head
                                            shard_offset = tp_rank * shard_blocks
                                            sc_shard = sc_tensor[shard_offset : shard_offset + shard_blocks, :]
                                        elif shard_id == "k":
                                            blocks_per_head = head_dim // block_size
                                            kv_replicas = max(1, tp_size // getattr(parent, "kv_num_heads", 8))
                                            kv_shard_id = tp_rank // kv_replicas
                                            shard_blocks = kv_num_heads_per_rank * blocks_per_head
                                            shard_offset = kv_shard_id * shard_blocks
                                            sc_shard = sc_tensor[shard_offset : shard_offset + shard_blocks, :]
                                        elif shard_id == "v":
                                            blocks_per_head = head_dim // block_size
                                            kv_replicas = max(1, tp_size // getattr(parent, "kv_num_heads", 8))
                                            kv_shard_id = tp_rank // kv_replicas
                                            shard_blocks = kv_num_heads_per_rank * blocks_per_head
                                            shard_offset = kv_shard_id * shard_blocks
                                            sc_shard = sc_tensor[shard_offset : shard_offset + shard_blocks, :]
                                        else:
                                            sc_shard = sc_tensor
                                    else:
                                        sc_shard = sc_tensor
                                    parent._scale_shards[shard_id] = sc_shard
                                    # When all 3 shards are loaded, concat and copy full tensor
                                    if len(parent._scale_shards) == 3:
                                        shards = parent._scale_shards
                                        # Concat along the first dimension (n_blocks_out)
                                        full_scale_np = np.concatenate(
                                            [
                                                shards["q"].numpy(),
                                                shards["k"].numpy(),
                                                shards["v"].numpy(),
                                            ],
                                            axis=0,
                                        )
                                        full_scale = paddle.to_tensor(full_scale_np, dtype=si.dtype)
                                        # si shape may differ from full_scale (e.g. [24,64] vs [64,24]).
                                        # Transpose full_scale to match si's shape before copy_.
                                        if full_scale.shape != si.shape:
                                            full_scale = full_scale.transpose([1, 0])
                                        si.copy_(full_scale, False)
                                        del parent._scale_shards
                                else:
                                    # si shape may differ from sc_tensor (e.g. [24,48] vs [48,24])
                                    if sc_tensor.shape != si.shape:
                                        sc_tensor = sc_tensor.transpose([1, 0])
                                    si.copy_(sc_tensor, False)
                    msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
                    pending_process.add(msn)
                    matched = True
                    break
                if not matched and wname in params_dict:
                    param = params_dict[wname]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, wt_tensor)
                    if sc_tensor is not None:
                        sublayer_name = wname.rsplit(".", 1)[0]
                        sublayers_dict = dict(self.named_sublayers())
                        if sublayer_name in sublayers_dict:
                            parent = sublayers_dict[sublayer_name]
                            if hasattr(parent, "weight_scale_inv") and parent.weight_scale_inv is not None:
                                si = parent.weight_scale_inv
                                if sc_tensor.shape != si.shape:
                                    sc_tensor = sc_tensor.transpose([1, 0])
                                si.copy_(sc_tensor, False)
                    msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", wname)
                    pending_process.add(msn)
                continue  # skip the normal dequant path

            if scale is None:
                logger.warning(f"No scale for {wname}, loading raw fp8 as bf16")
                wt_dq = get_tensor(wt).cast("bfloat16")
            else:
                wt_f32_t = get_tensor(wt).cast("float32")
                sc_np = get_tensor(scale).numpy()
                wt_np = wt_f32_t.numpy()
                del wt_f32_t

                out_dim, in_dim = wt_np.shape
                n_blocks_r = (out_dim + block_size - 1) // block_size
                n_blocks_c = (in_dim + block_size - 1) // block_size
                pad_r = n_blocks_r * block_size - out_dim
                pad_c = n_blocks_c * block_size - in_dim
                if pad_r > 0 or pad_c > 0:
                    wt_np = np.pad(wt_np, ((0, pad_r), (0, pad_c)))
                wt_blocked = wt_np.reshape([n_blocks_r, block_size, n_blocks_c, block_size])
                sc_expanded = sc_np.reshape([n_blocks_r, n_blocks_c])[:, np.newaxis, :, np.newaxis]
                wt_dequant = (wt_blocked * sc_expanded).reshape([n_blocks_r * block_size, n_blocks_c * block_size])[
                    :out_dim, :in_dim
                ]
                del wt_np, sc_np, wt_blocked, sc_expanded

                wt_dq = paddle.to_tensor(wt_dequant, dtype="bfloat16")
                del wt_dequant

            # Load into model parameter
            matched = False
            for mapping in expert_params_mapping:
                param_name_e, weight_name_e, expert_id, shard_id = mapping
                if weight_name_e not in wname:
                    continue
                model_param_name = wname.replace(weight_name_e, param_name_e)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = param.weight_loader
                weight_loader(param, wt_dq, shard_id=shard_id, expert_id=expert_id)
                msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
                pending_process.add(msn)
                matched = True
                break

            if not matched:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in wname or "mlp.experts" in wname:
                        continue
                    model_param_name = wname.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, wt_dq, shard_id)
                    msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
                    pending_process.add(msn)
                    matched = True
                    break

            if not matched:
                if wname in params_dict:
                    param = params_dict[wname]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, wt_dq)
                    msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", wname)
                    pending_process.add(msn)

            del wt_dq

        # Call process_weights_after_loading ONCE per unique sublayer
        sublayers_dict = dict(self.named_sublayers())
        for msn in pending_process:
            if msn in sublayers_dict:
                process_weights_after_loading_fn(msn)

    def _load_fp8_marlin_layer(
        self, layer_idx, fp8_weights, fp8_scales, params_dict, expert_params_mapping, block_size, moe_layer=None
    ):
        """Load FP8 expert weights directly to Marlin backend for a single layer."""
        # Use cached moe_layer if provided, otherwise search
        if moe_layer is None:
            for name, sublayer in self.named_sublayers():
                if f"layers.{layer_idx}." in name and hasattr(sublayer, "up_gate_proj_weight"):
                    moe_layer = sublayer
                    break

        if moe_layer is None:
            logger.warning(f"Marlin FP8: No MoE layer found for layer {layer_idx}")
            return

        num_experts = moe_layer.num_local_experts

        # EP filter: with EP=8, each GPU only loads its 32 local experts
        expert_id_offset = moe_layer.expert_id_offset  # e.g., rank * 32
        ep_expert_end = expert_id_offset + num_experts  # e.g., rank * 32 + 32

        # Group weights by expert (only local experts when EP is enabled)
        # Note: fp8_weights keys are already renamed (block_sparse_moe -> mlp)
        expert_up_gate = {}
        expert_down = {}
        expert_up_gate_scales = {}
        expert_down_scales = {}

        for wname, wt in fp8_weights.items():
            if "mlp.experts" not in wname:
                continue
            wt_tensor = get_tensor(wt)
            parts = wname.split(".")
            try:
                exp_idx = parts.index("experts") + 1
                expert_id = int(parts[exp_idx])
            except (ValueError, IndexError):
                continue

            # Skip non-local experts (EP filter)
            if not (expert_id_offset <= expert_id < ep_expert_end):
                continue
            local_expert_id = expert_id - expert_id_offset  # 0-based local ID

            if "w1" in wname or "w3" in wname:
                # w1 = gate_proj, w3 = up_proj -> combined as up_gate_proj
                if local_expert_id not in expert_up_gate:
                    expert_up_gate[local_expert_id] = {}
                if "w1" in wname:
                    expert_up_gate[local_expert_id]["gate"] = wt_tensor
                else:
                    expert_up_gate[local_expert_id]["up"] = wt_tensor

                scale_name = wname.replace(".weight", ".weight_scale_inv")
                if scale_name in fp8_scales:
                    if local_expert_id not in expert_up_gate_scales:
                        expert_up_gate_scales[local_expert_id] = {}
                    if "w1" in wname:
                        expert_up_gate_scales[local_expert_id]["gate"] = get_tensor(fp8_scales[scale_name])
                    else:
                        expert_up_gate_scales[local_expert_id]["up"] = get_tensor(fp8_scales[scale_name])
            elif "w2" in wname:
                expert_down[local_expert_id] = wt_tensor
                scale_name = wname.replace(".weight", ".weight_scale_inv")
                if scale_name in fp8_scales:
                    expert_down_scales[local_expert_id] = get_tensor(fp8_scales[scale_name])

        if not expert_up_gate or not expert_down:
            logger.warning(
                f"Marlin FP8: No expert weights found for layer {layer_idx}, "
                f"up_gate={len(expert_up_gate)}, down={len(expert_down)}"
            )
            return

        # Prepare per-expert data lists (avoid paddle.concat/stack on FP8)
        up_gate_info_list = []
        down_info_list = []
        up_gate_scales = []
        down_scales = []

        for i in range(num_experts):
            gate_w = expert_up_gate[i]["gate"]  # [1536, 3072] FP8
            up_w = expert_up_gate[i]["up"]  # [1536, 3072] FP8
            up_gate_info_list.append({"gate": gate_w, "up": up_w})

            gate_s = expert_up_gate_scales[i]["gate"]  # [12, 24]
            up_s = expert_up_gate_scales[i]["up"]  # [12, 24]
            combined_s = paddle.concat([gate_s.cast("float32"), up_s.cast("float32")], axis=0)
            up_gate_scales.append(combined_s)

            down_info_list.append(expert_down[i])  # FP8
            down_scales.append(expert_down_scales[i].cast("float32"))

        # SM80: dequant all local experts FP8→BF16 at load time, store on GPU.
        # Forward uses BF16 directly (no dequant, no CPU→GPU copy).
        # Memory: 64 experts × 27MB × 30 layers ≈ 50GB/GPU (fits 80GB A100).
        from fastdeploy.model_executor.utils import get_sm_version
        from fastdeploy.platforms import current_platform

        if get_sm_version() < 90 and current_platform.is_cuda():
            _t_start = time.time()

            # Get dimensions from expert weight shapes
            # gate weight: [moe_intermediate_size, hidden_size] = [1536, 3072]
            # down weight: [hidden_size, moe_intermediate_size] = [3072, 1536]
            first_gate_w = expert_up_gate[0]["gate"]
            first_down_w = expert_down[0]
            _moe_intermediate_size = first_gate_w.shape[0]  # 1536
            _hidden_size = first_gate_w.shape[1]  # 3072

            # Pre-allocate stacked output tensors (match weight shapes)
            # gate+up combined: concat([1536,3072], [1536,3072]) = [3072, 3072]
            # down: [3072, 1536]
            stacked_ug = paddle.zeros([num_experts, _moe_intermediate_size * 2, _hidden_size], dtype=paddle.bfloat16)
            stacked_down = paddle.zeros([num_experts] + list(first_down_w.shape), dtype=paddle.bfloat16)

            for i in range(num_experts):
                gate_w = expert_up_gate[i]["gate"]  # [1536, 3072] FP8
                up_w = expert_up_gate[i]["up"]  # [1536, 3072] FP8

                # Dequant up_gate combined - use slice assignment to avoid temp allocation
                gate_s = expert_up_gate_scales[i]["gate"].cast("float32")
                up_s = expert_up_gate_scales[i]["up"].cast("float32")
                combined_w = paddle.concat([gate_w, up_w], axis=0)  # [3072, 3072] FP8
                combined_s = paddle.concat([gate_s, up_s], axis=0)  # [24, 24] f32
                ug_bf16 = _dequant_fp8_blockwise_to_bf16(combined_w, combined_s)
                stacked_ug[i] = ug_bf16
                del combined_w, combined_s, ug_bf16, gate_s, up_s

                # Dequant down
                down_w = expert_down[i]  # [3072, 1536] FP8 (hidden_size, moe_intermediate_size)
                down_s = expert_down_scales[i].cast("float32")
                down_bf16 = _dequant_fp8_blockwise_to_bf16(down_w, down_s)
                stacked_down[i] = down_bf16
                del down_w, down_s, down_bf16

                # Free FP8 weights for this expert to save memory
                del expert_up_gate[i]
                del expert_down[i]

                if (i + 1) % 16 == 0:
                    paddle.device.cuda.empty_cache()

            # Split stacked_ug into gate and up
            moe_layer._sm80_gate = stacked_ug[:, :_moe_intermediate_size, :]  # [E, 1536, 3072]
            moe_layer._sm80_up = stacked_ug[:, _moe_intermediate_size:, :]  # [E, 1536, 3072]
            moe_layer._sm80_down = stacked_down  # [E, 3072, 1536]
            del stacked_ug, stacked_down
            paddle.device.cuda.empty_cache()

            _dt = time.time() - _t_start
            _mem = paddle.device.cuda.memory_allocated() / (1024**3)
            logger.info(
                f"SM80: Dequant {num_experts} experts for layer {layer_idx} "
                f"took {_dt:.1f}s, GPU mem: {_mem:.1f} GB. Stacked on GPU."
            )
            return  # Skip Marlin packing

        # Process through Marlin backend (packs weights for Marlin kernel)
        _process_fp8_marlin_weights(
            moe_layer,
            up_gate_info_list,
            down_info_list,
            up_gate_scales,
            down_scales,
            block_size,
        )

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from weights_iterator.

        Handles:
        - q_proj/k_proj/v_proj  -> qkv_proj  (stacked)
        - gate_proj(w1)/up_proj(w3) -> up_gate_proj  (expert, stacked)
        - down_proj(w2)             -> down_proj      (expert)
        - embed_tokens / lm_head   -> direct load
        - q_norm / k_norm          -> qk_norm weights
        - MTP layers (model.layers.62+) -> skipped
        """
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            # MiniMax e_score_correction_bias: checkpoint has ".gate.e_score_correction_bias"
            # (after block_sparse_moe→mlp rename), FD FusedMoE has "experts.gate_correction_bias"
            ("experts.gate_correction_bias", "gate.e_score_correction_bias", None),
        ]

        expert_params_mapping = self.get_expert_mapping()
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        # Cache layer_idx -> moe_sublayer mapping (avoids named_sublayers() traversal per layer)
        _enable_marlin_fp8 = os.environ.get("FD_MARLIN_FP8", "0") == "1"
        moe_layers = {}
        if _enable_marlin_fp8:
            for name, sublayer in self.named_sublayers():
                m = re.match(r".*layers\.(\d+)\.", name)
                if m and hasattr(sublayer, "up_gate_proj_weight"):
                    moe_layers[int(m.group(1))] = sublayer

        num_main_layers = self.fd_config.model_config.num_hidden_layers  # 62

        # Collect FP8 weights and scales grouped by layer index for streaming dequant.
        # Layer -1 = non-layer weights (embed, norm, lm_head)
        fp8_by_layer: Dict[int, dict] = {}  # layer_idx -> {wname: fp8_tensor}
        scales_by_layer: Dict[int, dict] = {}  # layer_idx -> {scale_name: scale_tensor}

        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")

            # Skip MTP layers (model.layers.62, 63, 64, ...)
            # MTP layer indices start at num_hidden_layers
            skip = False
            if "model.layers." in loaded_weight_name:
                # Extract layer index
                parts = loaded_weight_name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                    if layer_idx >= num_main_layers:
                        skip = True
                except (ValueError, IndexError):
                    pass
            if skip:
                continue

            # MiniMax checkpoint uses "block_sparse_moe" but FD model uses "mlp"
            loaded_weight_name = loaded_weight_name.replace(".block_sparse_moe.", ".mlp.")

            # MiniMax checkpoint has e_score_correction_bias at ".mlp.e_score_correction_bias"
            # but FD FusedMoE stores it as ".mlp.gate.e_score_correction_bias"
            # (since gate_correction_bias_key = "{prefix}.gate.e_score_correction_bias")
            loaded_weight_name = loaded_weight_name.replace(
                ".mlp.e_score_correction_bias",
                ".mlp.gate.e_score_correction_bias",
            )

            # MiniMax FP8 checkpoint stores per-block scales as "*.weight_scale_inv".
            # Collect scale for later dequantization, grouped by layer index.
            if ".weight_scale_inv" in loaded_weight_name:
                li = self._extract_layer_idx(loaded_weight_name, num_main_layers)
                scales_by_layer.setdefault(li, {})[loaded_weight_name] = loaded_weight
                continue

            # For ALL FP8 weights (both expert and linear), collect for streaming dequant.
            if (
                loaded_weight_name.endswith(".weight")
                and hasattr(loaded_weight, "dtype")
                and "float8" in str(loaded_weight.dtype).lower()
            ):
                li = self._extract_layer_idx(loaded_weight_name, num_main_layers)
                fp8_by_layer.setdefault(li, {})[loaded_weight_name] = loaded_weight
                continue

            # Special handling for q_norm / k_norm weights.
            # Now using MiniMaxRMSNorm with shard-wise weight [total_size/tp].
            # Use the norm's weight_loader to properly TP-shard the weight.
            if ".q_norm.weight" in loaded_weight_name:
                # e.g. "model.layers.0.self_attn.q_norm.weight"
                # prefix = "model.layers.0.self_attn"
                prefix = loaded_weight_name.replace(".q_norm.weight", "")
                q_norm_key = f"{prefix}.q_norm.weight"  # = "model.layers.0.self_attn.q_norm.weight"
                if q_norm_key in params_dict:
                    param = params_dict[q_norm_key]
                    wl = getattr(param, "weight_loader", None)
                    if wl is not None:
                        wl(param, loaded_weight)
                    else:
                        w = get_tensor(loaded_weight).cast("float32")
                        tp = self.fd_config.parallel_config.tensor_parallel_size
                        shard = w.shape[0] // tp
                        rank = paddle.distributed.get_rank() if tp > 1 else 0
                        param.set_value(w[rank * shard : (rank + 1) * shard])
                continue
            if ".k_norm.weight" in loaded_weight_name:
                prefix = loaded_weight_name.replace(".k_norm.weight", "")
                k_norm_key = f"{prefix}.k_norm.weight"
                if k_norm_key in params_dict:
                    param = params_dict[k_norm_key]
                    wl = getattr(param, "weight_loader", None)
                    if wl is not None:
                        wl(param, loaded_weight)
                    else:
                        w = get_tensor(loaded_weight).cast("float32")
                        if w.shape[0] == param.shape[0]:
                            param.set_value(w)
                        else:
                            tp = self.fd_config.parallel_config.tensor_parallel_size
                            shard = max(1, w.shape[0] // tp)
                            rank = paddle.distributed.get_rank() if tp > 1 else 0
                            end = min((rank + 1) * shard, w.shape[0])
                            param.set_value(w[rank * shard : end])
                continue

            # Try stacked parameter mappings
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                # Expert weights are handled separately
                if "mlp.experts" in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                matched = True
                break

            if matched:
                model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
                process_weights_after_loading_fn(model_sublayer_name, param)
                continue

            # Try expert parameter mappings
            matched_expert = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                matched_expert = True
                break

            if matched_expert:
                model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
                process_weights_after_loading_fn(model_sublayer_name, param)
                continue

            # Direct load for remaining parameters
            model_param_name = loaded_weight_name
            if model_param_name not in params_dict:
                continue
            param = params_dict[model_param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))

            weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        # ---- Batch FP8 dequant (all layers at once) ----
        # Process non-layer weights first (embed, norm, lm_head, etc.)
        # Then process ALL decoder layers' FP8 weights in one batch,
        # to avoid repeated per-layer overhead.
        BLOCK_SIZE = 128

        # Process non-layer FP8 weights (layer_idx = -1)
        if -1 in fp8_by_layer:
            logger.info(f"Dequantizing non-layer FP8 weights ({len(fp8_by_layer[-1])} tensors) ...")
            self._dequant_fp8_weights(
                fp8_by_layer[-1],
                scales_by_layer.get(-1, {}),
                params_dict,
                stacked_params_mapping,
                expert_params_mapping,
                process_weights_after_loading_fn,
                BLOCK_SIZE,
            )
            del fp8_by_layer[-1]
            if -1 in scales_by_layer:
                del scales_by_layer[-1]
            paddle.device.cuda.empty_cache()

        # Process ALL decoder layers' FP8 weights in one batch
        layer_indices = sorted(k for k in fp8_by_layer.keys() if k >= 0)
        mem_before_all = paddle.device.cuda.memory_allocated() / (1024**3)
        logger.info(
            f"Batch FP8 dequant: processing {len(layer_indices)} layers "
            f"({sum(len(fp8_by_layer[li]) for li in layer_indices)} tensors total, "
            f"GPU: {mem_before_all:.1f} GB) ..."
        )

        for li in layer_indices:
            logger.info(f"Start load layer {li}")  # Engine progress detection
            if _enable_marlin_fp8:
                # Marlin FP8 mode: load FP8 expert weights directly to Marlin backend
                self._load_fp8_marlin_layer(
                    li,
                    fp8_by_layer[li],
                    scales_by_layer.get(li, {}),
                    params_dict,
                    expert_params_mapping,
                    BLOCK_SIZE,
                    moe_layer=moe_layers.get(li),
                )
                # Non-expert FP8 weights (attention qkv/o_proj)
                non_expert_fp8 = {k: v for k, v in fp8_by_layer[li].items() if "mlp.experts" not in k}
                non_expert_scales = {k: v for k, v in scales_by_layer.get(li, {}).items() if "mlp.experts" not in k}
                if non_expert_fp8:
                    self._dequant_fp8_weights(
                        non_expert_fp8,
                        non_expert_scales,
                        params_dict,
                        stacked_params_mapping,
                        expert_params_mapping,
                        process_weights_after_loading_fn,
                        BLOCK_SIZE,
                    )
            else:
                self._dequant_fp8_weights(
                    fp8_by_layer[li],
                    scales_by_layer.get(li, {}),
                    params_dict,
                    stacked_params_mapping,
                    expert_params_mapping,
                    process_weights_after_loading_fn,
                    BLOCK_SIZE,
                )
            # Free layer data
            del fp8_by_layer[li]
            if li in scales_by_layer:
                del scales_by_layer[li]

        paddle.device.cuda.empty_cache()
        mem_after_all = paddle.device.cuda.memory_allocated() / (1024**3)
        logger.info(
            f"Batch FP8 dequant done. GPU: {mem_after_all:.1f} GB " f"(freed {mem_before_all - mem_after_all:.1f} GB)"
        )

        del fp8_by_layer, scales_by_layer

        # Transpose all Linear weights after loading.
        # ColumnParallelLinear/RowParallelLinear: BlockWiseFP8LinearMethod does NOT
        # transpose for torch format + FP8 checkpoint, so we must do it here.
        # ReplicatedLinear (e.g. MoE gate): UnquantizedLinearMethod would also
        # transpose, but that happens in process_final_after_loading. We include it
        # here so that direct callers (e.g. run_fd_gen.py) work. process_final_after_loading
        # is aware and will skip re-transposing via a guard.
        if self.fd_config.model_config.model_format == "torch":

            from fastdeploy.model_executor.layers.linear import (
                ColumnParallelLinear,
                ReplicatedLinear,
                RowParallelLinear,
            )
            from fastdeploy.model_executor.utils import process_weight_transpose

            for name, sublayer in self.named_sublayers():
                if isinstance(sublayer, (ColumnParallelLinear, RowParallelLinear, ReplicatedLinear)):
                    if getattr(sublayer, "_torch_weight_transposed", False):
                        continue  # Already transposed
                    if hasattr(sublayer, "weight") and sublayer.weight is not None:
                        if sublayer.weight.ndim == 2:
                            process_weight_transpose(sublayer, "weight")
                            sublayer._torch_weight_transposed = True

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        # Cast to float32 before lm_head to avoid BF16 precision issues
        hidden_states_f32 = hidden_states.cast("float32")
        lm_head_w_f32 = self.lm_head.linear.weight.cast("float32")
        logits = paddle.matmul(hidden_states_f32, lm_head_w_f32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def empty_input_forward(self, forward_meta):
        """Warm-up empty forward for MoE expert routing initialization."""
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(self.fd_config.model_config.num_hidden_layers):
            self.model.layers[i].mlp.experts(fake_hidden_states, self.model.layers[i].mlp.gate, forward_meta)

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        ids_remove_padding = inputs["ids_remove_padding"]
        hidden_states = self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend."""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)


class MiniMaxM2PretrainedModel(PretrainedModel):
    """Pretrained model wrapper for MiniMax-M2.5."""

    config_class = FDConfig

    def _init_weight(self, layer):
        return None

    @classmethod
    def arch_name(cls):
        return "MiniMaxM2ForCausalLM"
