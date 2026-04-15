# FastDeploy 架构选型指南

本文件帮助你在给定 `config.json` 后，快速决定新模型应该如何集成到 FastDeploy。

---

## 架构识别速查表

根据 `config.json` 的关键字段快速识别模型类型：

### 1. 识别 Attention 类型

| config.json 特征 | Attention 类型 | FastDeploy 实现 |
|----------------|--------------|----------------|
| `kv_lora_rank` 存在 | MLA（Multi-head Latent Attention） | `mla_attention_backend.py` |
| `q_lora_rank` + `kv_lora_rank` | MLA | `mla_attention_backend.py` |
| `sparse_indexer` / `num_indexer_heads` | DSA（Dynamic Sparse Attention） | `dsa_attention_backend.py` |
| `num_key_value_heads < num_attention_heads` | GQA（Grouped Query Attention） | 标准 `Attention` with GQA |
| `num_key_value_heads == 1` | MQA（Multi-Query Attention） | 标准 `Attention` with MQA |
| `num_key_value_heads == num_attention_heads` | MHA（Multi-Head Attention） | 标准 `Attention` |
| `qk_norm: true` | 带 QK Normalization 的 MHA | `Glm4MoeAttention` 模式 |

### 2. 识别 FFN/MoE 类型

| config.json 特征 | FFN 类型 | FastDeploy 实现 |
|----------------|---------|----------------|
| `num_experts` / `n_routed_experts` 存在 | MoE | `FusedMoE` |
| `num_shared_experts` 存在 | MoE + Shared Experts | `FusedMoE` + 额外 MLP |
| 只有 `intermediate_size` | Dense FFN | `MergedColumnParallelLinear` |
| `hidden_act: "silu"` / `"swiglu"` | SwiGLU | `SiluAndMul` |
| `hidden_act: "gelu"` | GELU | `GeluAndMul` |

### 3. 识别继承目标

```
识别到 MLA/DSA attention?
├── YES: 是否与 DeepSeekV3 完全相同的 MLA？
│   ├── YES: 直接继承 DeepseekV3ForCausalLM
│   ├── 是 DSA (Dynamic Sparse Attention)？
│   │   └── 继承 DeepseekV32ForCausalLM（例：glm_moe_dsa.py）
│   └── 有微小差异（如不同的 rope_scaling）？
│       └── 继承并重载 __init__ 中的对应部分
│
└── NO: 标准 Transformer
    ├── 是否 MoE？
    │   ├── YES: 参考 glm4_moe.py 或 qwen3moe.py
    │   └── NO: 参考 qwen3.py 或 qwen2.py
    └── 是否有 QK Normalization？
        ├── YES: 参考 glm4_moe.py（含 qk_norm 支持）
        └── NO: 参考 qwen2.py（最标准的实现）
```

---

## 参考模型对照表

| 你的新模型类似于 | 推荐参考文件 | 继承方式 |
|---------------|------------|---------|
| DeepSeek V3/R1 变体 | `deepseek_v3.py` | 继承 `DeepseekV3ForCausalLM` |
| DeepSeek V3.2（含 DSA） | `deepseek_v3.py` | 继承 `DeepseekV32ForCausalLM` |
| GLM4.5/4.6/4.7 MoE | `glm4_moe.py` | 继承或新建同结构 |
| GLM MoE + DSA | `glm_moe_dsa.py` + `deepseek_v3.py` | 继承 `DeepseekV32ForCausalLM` |
| Qwen3 / Qwen2.5 | `qwen3.py` / `qwen2.py` | 继承对应类 |
| Qwen3 MoE | `qwen3moe.py` | 继承或新建 |
| ERNIE 4.5 | `ernie4_5_moe.py` | 参考 |
| 全新架构 | `qwen2.py` | 从 `ModelForCasualLM` 新建 |

---

## 关键配置字段速查

### ForwardMeta 可用字段

`ForwardMeta` 对象在 `forward()` 调用时传入，包含：

```python
forward_meta.position_ids      # 位置编码 ids
forward_meta.is_prefill        # True=prefill, False=decode
forward_meta.seq_lens          # 每个序列的长度
forward_meta.block_tables      # KV cache 的 block table
forward_meta.attn_mask         # attention mask（可选）
```

**注意**：PR #7139 将 `position_ids` 和 `encoder_mask` 从显式参数移入 `ForwardMeta`，使调用链更简洁。新模型应采用这种方式。

### FDConfig 结构

```python
fd_config.model_config          # 原始模型配置（来自 config.json）
fd_config.vocab_size            # 经过 padding 的词表大小
fd_config.tp_size               # tensor parallel size
fd_config.ep_size               # expert parallel size（MoE 模型）
fd_config.quant_config          # 量化配置
```

---

## 多卡部署关键配置

### Tensor Parallelism (TP)

- **Column Parallel**：`QKVParallelLinear`、`MergedColumnParallelLinear` — 沿输出维度切分
- **Row Parallel**：`RowParallelLinear` — 沿输入维度切分，自动做 AllReduce
- 注意：`num_kv_heads` 必须能被 `tp_size` 整除，否则需要 head padding

### Expert Parallelism (EP，仅 MoE）

- `FusedMoE` 自动处理专家的 EP 分配
- 配置：`fd_config.ep_size`
- EP 推荐配置：ep_size = num_experts（每 GPU 一个专家）

### Head Padding 处理（TP > num_kv_heads 时）

```python
# 参考 PR #7139 中的模式：
if self.num_kv_heads % tp_size != 0:
    # Pad to multiple of tp_size
    padded_kv_heads = ((self.num_kv_heads + tp_size - 1) // tp_size) * tp_size
    # ... 在 attention forward 中 pad q/output，然后 trim
```

---

## 量化兼容性

| 量化格式 | 支持情况 | 注意事项 |
|---------|---------|---------|
| BF16 | ✅ 所有模型 | 默认精度 |
| W8A16 | ✅ Dense + MoE | Linear 层自动替换 |
| W8A8 | ✅ Dense | MoE 需要特殊处理 |
| W4A16 | ✅ Dense + MoE | 需要 GPTQ/AWQ 格式 |
| FP8 | ✅ H100/H800+ | 需要 FP8 calibration |
| W2A16 | ⚠️ 实验性 | 仅部分模型 |

量化权重的加载由 `fd_config.quant_config` 自动处理，模型实现代码通常无需特殊处理。
