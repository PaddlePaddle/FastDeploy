# FastDeploy 模型代码模板

本文件包含三种常见情形的完整代码模板，覆盖 FastDeploy 中新增模型支持时 90% 以上的场景。

---

## 目录

1. [模板 A：继承现有模型（最简情形）](#template-a)
2. [模板 B：全新标准 Transformer（Dense 模型）](#template-b)
3. [模板 C：MoE 模型完整模板](#template-c)
4. [stacked_params_mapping 参考表](#params-mapping)
5. [常用 Import 清单](#imports)

---

## 模板 A：继承现有模型 {#template-a}

适用：新模型架构与某个已有模型高度相似，只是换了名字或有微小差异。

**典型案例**：GLM-MoE-DSA（继承 DeepSeekV32，PR #6863）、GLM4.7 Lite（继承 DeepSeekV3，PR #7139）

```python
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

"""<ModelName> model - extends <BaseModelName> with <brief diff description>."""

from __future__ import annotations

from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry

# 继承最相似的已有实现
from fastdeploy.model_executor.models.<base_module> import (
    <BaseForCausalLM>,
    <BasePretrainedModel>,  # 如果需要不同的 TP mapping 才需要重载
)


@ModelRegistry.register_model_class(
    architecture="<NewArch>ForCausalLM",   # 与 config.json architectures[0] 完全一致
    module_name="<new_module_name>",       # 通常是本文件名（不含 .py）
    category=ModelCategory.TEXT_GENERATION,
)
class <NewArch>ForCausalLM(<BaseForCausalLM>):
    """<NewModelName> causal language model.

    Reuses <BaseModelName> implementation. Key differences:
    - <difference 1>
    - <difference 2>
    """

    @classmethod
    def name(cls) -> str:
        return "<NewArch>ForCausalLM"

    # 如果 attention 路由方式不同，重载 __init__ 并修改 model_type 判断
    # 通常不需要重载任何方法


# 如果 tensor parallel mapping 与父类完全相同，直接复用：
# <NewArch>PretrainedModel = <BasePretrainedModel>

# 如果需要不同的 TP mapping，则重载：
class <NewArch>PretrainedModel(<BasePretrainedModel>):
    @classmethod
    def arch_name(cls) -> str:
        return "<NewArch>ForCausalLM"

    # 仅重载有差异的方法
```

---

## 模板 B：全新标准 Dense Transformer {#template-b}

适用：Dense 模型，GQA attention + SwiGLU MLP，与 Qwen2 架构相似。

```python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# [Apache 2.0 License header]

"""<ModelName> model implementation for FastDeploy."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import paddle
import paddle.nn as nn

from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.utils import support_graph_optimization
from fastdeploy.model_executor.layers.attention import Attention


class <Model>MLP(nn.Layer):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        fd_config,
    ):
        super().__init__()
        self.up_gate_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            fd_config=fd_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            fd_config=fd_config,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.up_gate_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class <Model>Attention(nn.Layer):
    """Multi-head attention with GQA support."""

    def __init__(self, layer_idx: int, fd_config):
        super().__init__()
        config = fd_config.model_config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            fd_config=fd_config,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            fd_config=fd_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 131072),
            base=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            layer_idx=layer_idx,
            fd_config=fd_config,
        )

    def forward(self, positions, hidden_states, forward_meta: ForwardMeta):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ], axis=-1)

        cos, sin = self.rotary_emb(positions, forward_meta)
        q, k = self.rotary_emb.apply(q, k, cos, sin, forward_meta)

        attn_output = self.attn(q, k, v, forward_meta)
        output = self.o_proj(attn_output)
        return output


class <Model>DecoderLayer(nn.Layer):
    """Single transformer decoder layer."""

    def __init__(self, layer_idx: int, fd_config):
        super().__init__()
        config = fd_config.model_config
        self.self_attn = <Model>Attention(layer_idx, fd_config)
        self.mlp = <Model>MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            fd_config=fd_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, positions, hidden_states, forward_meta: ForwardMeta):
        # Pre-norm → attention → residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_meta)
        hidden_states = residual + hidden_states

        # Pre-norm → MLP → residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@support_graph_optimization
class <Model>Model(nn.Layer):
    """The core <ModelName> transformer."""

    def __init__(self, fd_config):
        super().__init__()
        config = fd_config.model_config

        self.embed_tokens = VocabParallelEmbedding(
            fd_config.vocab_size,
            config.hidden_size,
            fd_config=fd_config,
        )
        self.layers = nn.LayerList([
            <Model>DecoderLayer(i, fd_config)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        input_ids = inputs["ids_remove_padding"]
        positions = forward_meta.position_ids

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_meta)

        hidden_states = self.norm(hidden_states)
        return hidden_states


@ModelRegistry.register_model_class(
    architecture="<ModelArch>ForCausalLM",     # 与 config.json architectures[0] 完全一致
    module_name="<model_module_name>",          # 本文件名（不含 .py）
    category=[ModelCategory.TEXT_GENERATION],
    primary_use=ModelCategory.TEXT_GENERATION,
)
class <ModelArch>ForCausalLM(ModelForCasualLM):
    """<ModelName> causal language model for FastDeploy inference."""

    def __init__(self, fd_config):
        super().__init__(fd_config)
        self.model = <Model>Model(fd_config)
        self.lm_head = ParallelLMHead(
            fd_config.vocab_size,
            fd_config.model_config.hidden_size,
            fd_config=fd_config,
        )

    @classmethod
    def name(cls) -> str:
        return "<ModelArch>ForCausalLM"

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        hidden_states = self.model(inputs, forward_meta)
        return hidden_states

    def compute_logits(self, hidden_state, **kwargs):
        logits = self.lm_head(hidden_state)
        return logits

    def load_weights(self, weights):
        """Load model weights with parameter name mapping."""
        stacked_params_mapping = [
            # (fused_param_name, original_param_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params = set()

        for name, param in weights:
            # 处理 lm_head 权重共享
            if "lm_head.weight" in name:
                continue  # 通常与 embed_tokens 共享

            # 处理 fused 参数映射
            for fused_name, orig_name, shard_id in stacked_params_mapping:
                if orig_name in name:
                    name = name.replace(orig_name, fused_name)
                    break

            if name in params_dict:
                param_data = params_dict[name]
                param_data.set_value(param)
                loaded_params.add(name)

        return loaded_params


# Pretrained model class for tensor parallelism configuration
from paddleformers.transformers import PretrainedModel as PaddlePretrainedModel

class <ModelArch>PretrainedModel(PaddlePretrainedModel):

    @classmethod
    def arch_name(cls) -> str:
        return "<ModelArch>ForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        from paddleformers.transformers.convert_slow_tokenizer import split_or_merge_func
        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        # 列并行（Column Parallel）：沿最后一维切分
        column_parallel = [
            "self_attn.qkv_proj.weight",
            "mlp.up_gate_proj.weight",
        ]
        # 行并行（Row Parallel）：沿第一维切分
        row_parallel = [
            "self_attn.o_proj.weight",
            "mlp.down_proj.weight",
        ]

        mappings = {}
        for key in column_parallel:
            mappings[key] = fn(key, "col")
        for key in row_parallel:
            mappings[key] = fn(key, "row")

        return mappings
```

---

## 模板 C：MoE 模型完整模板 {#template-c}

适用：含 MoE 路由的模型（如 Qwen3-MoE、DeepSeek-MoE 变体）。

关键差异点：
- 使用 `FusedMoE` 层替代普通 MLP
- 使用 `FusedMoE.make_expert_params_mapping()` 进行权重映射
- Expert Parallelism 配置

```python
# [License header]

"""<ModelName> MoE model implementation."""

from __future__ import annotations
from typing import Dict, Optional

import paddle
import paddle.nn as nn

from fastdeploy.model_executor.layers.fused_moe import FusedMoE
from fastdeploy.model_executor.models.model_base import (
    ModelCategory, ModelForCasualLM, ModelRegistry,
)
# ... (其他 imports 同模板 B)


class <Model>MoE(nn.Layer):
    """Mixture of Experts routing layer."""

    def __init__(self, config, fd_config):
        super().__init__()

        self.num_experts = config.num_experts          # 总专家数
        self.num_experts_per_tok = config.num_experts_per_tok  # 每 token 激活的专家数

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            fd_config=fd_config,
        )

        # 可选的 shared experts（如 DeepSeek/GLM 风格）
        if hasattr(config, "num_shared_experts") and config.num_shared_experts > 0:
            self.shared_experts = <Model>MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                fd_config=fd_config,
            )
        else:
            self.shared_experts = None

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias_attr=False)

    def forward(self, hidden_states):
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        router_logits = self.gate(hidden_states)
        moe_output = self.experts(hidden_states, router_logits)

        if shared_output is not None:
            return moe_output + shared_output
        return moe_output


# ... (Attention 层同模板 B，仅 MLP 层替换为 MoE)


@ModelRegistry.register_model_class(
    architecture="<ModelArch>ForCausalLM",
    module_name="<model_module_name>",
    category=ModelCategory.TEXT_GENERATION,
)
class <ModelArch>ForCausalLM(ModelForCasualLM):

    def load_weights(self, weights):
        """MoE weight loading with expert parameter mapping."""

        # 专家权重映射（MoE 特有）
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # 标准 stacked 参数映射
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, param in weights:
            # 先处理专家权重
            for mapping in expert_params_mapping:
                if mapping.is_expert_weights and mapping.weight_name in name:
                    # Expert weight handling
                    expert_id = int(name.split(".experts.")[1].split(".")[0])
                    new_name = name.replace(
                        f"experts.{expert_id}.{mapping.weight_name}",
                        f"experts.{mapping.param_name}"
                    )
                    if new_name in params_dict:
                        # Shard into expert slot
                        params_dict[new_name].data[expert_id] = param
                    break
            else:
                # 非专家权重走标准路径
                for fused_name, orig_name, _ in stacked_params_mapping:
                    if orig_name in name:
                        name = name.replace(orig_name, fused_name)
                        break
                if name in params_dict:
                    params_dict[name].set_value(param)
```

---

## stacked_params_mapping 参考表 {#params-mapping}

不同架构的权重映射规律：

| 架构类型 | 原始权重名 | FastDeploy 融合名 |
|---------|-----------|-----------------|
| 所有模型 | `q_proj`, `k_proj`, `v_proj` | `qkv_proj` |
| SwiGLU（Qwen/LLaMA） | `gate_proj`, `up_proj` | `up_gate_proj` |
| GeGLU | `gate_proj`, `up_proj` | `up_gate_proj` |
| MLA（DeepSeek） | `kv_a_proj_with_mqa` | 保持原名 |
| 标准 FFN | `fc1`, `fc2` | 保持原名 |

---

## 常用 Import 清单 {#imports}

```python
# 基础框架
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.models.model_base import (
    ModelCategory, ModelForCasualLM, ModelRegistry,
)
from fastdeploy.model_executor.utils import support_graph_optimization

# 线性层
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,  # FFN gate+up fused
    QKVParallelLinear,           # QKV fused
    RowParallelLinear,           # o_proj, down_proj
    ColumnParallelLinear,        # single column-parallel
)

# Attention
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.attention.mla_attention_backend import MLAAttention
from fastdeploy.model_executor.layers.attention.dsa_attention_backend import DSAAttention

# Normalization
from fastdeploy.model_executor.layers.normalization import RMSNorm, LayerNorm

# Embedding & LM Head
from fastdeploy.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead

# RoPE
from fastdeploy.model_executor.layers.rotary_embedding import get_rope

# MoE
from fastdeploy.model_executor.layers.fused_moe import FusedMoE

# Activation
from fastdeploy.model_executor.layers.activation import SiluAndMul, GeluAndMul
```
