# 模型文档模板

用于在 `docs/` 目录下生成标准的模型部署文档。

---

## 模板：`docs/<model_name>.md`

````markdown
# <ModelName> 部署指南

## 模型介绍

<ModelName> 是由 <组织名> 发布的 <参数量> 大语言模型，主要特点：

- **架构**：<Transformer/MoE/混合架构>，<Dense/Sparse> 激活
- **Attention**：<MHA/GQA/MLA/DSA>，<num_kv_heads> KV heads
- **上下文长度**：<max_position_embeddings> tokens
- **支持功能**：<文本生成/多模态/工具调用/推理>

## 快速开始

### 环境要求

- FastDeploy >= 2.x
- CUDA 12.0+
- 显存：<建议显存配置>

### 单卡部署（<推荐 GPU 型号>）

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model <model_path_or_hub_id> \
    --tensor-parallel-size 1 \
    --max-model-len <recommended_ctx_len>
```

### 多卡部署（<N>×GPU）

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model <model_path_or_hub_id> \
    --tensor-parallel-size <N> \
    --max-model-len <max_context_length>
```

### API 调用测试

```bash
curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<model_name>",
        "prompt": "Hello, I am",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

## 性能参考

> 以下数据基于 <测试环境，如 8×H800 SXM5>，batch size = <N>

| 配置 | 吞吐量（tokens/s） | 首 Token 延迟 | 备注 |
|-----|------------------|-------------|-----|
| TP=<N>, BF16 | - | - | baseline |
| TP=<N>, W8A8 | - | - | - |
| TP=<N>, W4A16 | - | - | - |

*数据持续更新中*

## 量化支持

| 格式 | 支持 | 推荐场景 |
|-----|-----|---------|
| BF16 | ✅ | 精度优先 |
| W8A16 | ✅ | 平衡配置 |
| W8A8 | <✅/⚠️> | 吞吐优先 |
| W4A16 | <✅/⚠️> | 显存受限 |
| FP8 | <✅/⚠️> | H100/H800 |

## 注意事项

1. **Tensor Parallelism 限制**：`num_key_value_heads`（<N>）必须能被 TP size 整除，支持 TP = <list of valid values>。
2. **MoE 配置**：建议 `expert-parallel-size` = <推荐值>，实验显示 EP=<N> 配合 TP=<N> 性能最优。
3. **上下文长度**：完整 <max_len> 上下文需要约 <显存量> 显存（BF16，TP=<N>）。
4. **已知限制**：<任何已知问题>。

## 相关资源

- 模型权重：[HuggingFace](<hf_url>) | [ModelScope](<ms_url>) | [AIStudio](<ai_url>)
- 原始论文：<arxiv_url>
- 技术报告：<url>
````

---

## 文档命名约定

| 场景 | 文件路径 |
|-----|---------|
| 主要模型 | `docs/<model_family>.md` |
| 特定版本 | `docs/<model_family>_<version>.md` |
| 中文文档 | `docs/zh/<model_family>.md` |

---

## supported_models.md 表格更新

在 `docs/supported_models.md` 中添加新行，格式：

```markdown
| <ModelName> | <Size> | <BF16/INT8/INT4> | <✅/⚠️> | <✅/⚠️> | [链接](<model_url>) |
```

列说明（以现有表格为准）：
- 模型名称
- 参数量
- 支持的量化格式
- 多卡 TP 支持
- MoE/EP 支持（如适用）
- 官方权重链接
