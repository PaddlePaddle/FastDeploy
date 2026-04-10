---
name: fastdeploy-llm-integration
description: >
  Guides you through adding inference deployment support for a new open-source LLM to the FastDeploy repository.
  Given a model path (local or HuggingFace/ModelScope hub), this skill walks through analyzing the model architecture,
  choosing the right base class, generating the model implementation file, updating registries, writing docs, and
  producing a deployment test script.

  Use this skill whenever the user wants to: add a new model to FastDeploy, integrate an open-source LLM for inference,
  support a new model architecture in PaddlePaddle's inference framework, port a HuggingFace model to FastDeploy,
  or asks "如何在FastDeploy中支持XX模型" / "帮我给FastDeploy新增XX模型支持".

IMPORTANT: Always use this skill when the user mentions FastDeploy and a model name/path together, even if they
just ask "how do I add X to FastDeploy" — this skill has all the patterns and templates needed.
---

# FastDeploy LLM Integration Skill

你的任务是：给定一个开源大模型路径，完整实现该模型在 FastDeploy 仓库中的推理部署支持，包括模型实现代码、文档和测试脚本。

---

## 工作流程总览

```
步骤 1: 分析模型架构
步骤 2: 选择继承策略（复用 vs 新建）
步骤 3: 生成模型实现文件
步骤 4: 更新注册和配置
步骤 5: 补充文档
步骤 6: 生成部署测试脚本
```

---

## 步骤 1：分析模型架构

首先读取模型的 `config.json`：

```bash
cat /path/to/model/config.json
# 或从 HuggingFace 获取：
curl https://huggingface.co/<org>/<model>/raw/main/config.json
```

**关键字段提取清单：**

| 字段 | 用途 |
|------|------|
| `architectures` | 注册用的 architecture name，如 `["Qwen2ForCausalLM"]` |
| `model_type` | attention/MLP 路径选择的分支条件 |
| `hidden_size` | 模型宽度 |
| `num_hidden_layers` | 层数 |
| `num_attention_heads` | 注意力头数 |
| `num_key_value_heads` | GQA 头数（若 < num_attention_heads 则为 GQA） |
| `intermediate_size` | FFN 中间层大小 |
| `num_experts` / `num_routed_experts` | MoE 专家数（有则为 MoE 模型） |
| `rope_theta` / `rope_scaling` | 位置编码配置 |
| `attention_bias` | Attention 是否有 bias |
| `qk_norm` | 是否有 QK normalization（GLM4.5+ 特性） |

---

## 步骤 2：选择继承策略

根据分析结果，按以下决策树选择最优策略：

```
config.json 分析
    │
    ├── 与 DeepSeekV3 架构高度相似（MLA/DSA attention + MoE）？
    │       └── YES → 继承 DeepseekV3ForCausalLM 或 DeepseekV32ForCausalLM
    │                  参考：glm_moe_dsa.py（PR #6863）
    │
    ├── 与 GLM4 MoE 相似（标准 MHA + MoE + QK Norm）？
    │       └── YES → 继承 Glm4MoeForCausalLM 或从头实现，参考 glm4_moe.py
    │
    ├── 与 Qwen2/3 架构相似（GQA + RoPE + SwiGLU）？
    │       └── YES → 继承 Qwen2ForCausalLM / Qwen3ForCausalLM
    │                  参考：qwen3.py
    │
    └── 全新架构 → 从 ModelForCasualLM 基类开始
             参考：qwen2.py（最完整的参考实现）
```

**继承的好处**：
- 减少代码量 80%+
- 自动继承 tensor parallelism、weight sharding
- 只需重载差异部分（如不同的 attention 实现、不同的 MoE 路由）

---

## 步骤 3：生成模型实现文件

文件路径：`fastdeploy/model_executor/models/<model_name>.py`

参考 `references/model_templates.md` 中的完整代码模板。根据步骤 2 的继承策略选择对应模板：

### 模板 A：继承现有模型（推荐，适合 90% 的情况）

```python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# ... (standard Apache 2.0 header)

"""<ModelName> model implementation for FastDeploy."""

from __future__ import annotations

from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry

# 从最相似的已有模型继承
from fastdeploy.model_executor.models.<base_model> import <BaseForCausalLM>


@ModelRegistry.register_model_class(
    architecture="<NewModelArchitecture>ForCausalLM",   # 必须与 config.json 的 architectures[0] 完全一致
    module_name="<model_name>",
    category=ModelCategory.TEXT_GENERATION,
)
class <NewModel>ForCausalLM(<BaseForCausalLM>):
    """<NewModel> causal language model.

    Reuses <BaseModel> infrastructure with <description of differences>.
    """

    @classmethod
    def name(cls) -> str:
        return "<NewModelArchitecture>ForCausalLM"

    # 只重载与 base model 有差异的部分
    # 例如：不同的 attention 类型、不同的 MLP 结构、额外的 normalization
```

### 模板 B：全新模型实现

完整代码结构见 `references/model_templates.md`，包含：
- 标准 MLP 层（SwiGLU / GeGLU / ReLU 变体）
- Attention 层（MHA / GQA / MLA）
- DecoderLayer
- 主 Model 类（带 `@support_graph_optimization`）
- ForCausalLM 注册类
- PretrainedModel 类（tensor parallel 配置）

---

## 步骤 4：更新注册和配置

### 4a. 验证自动注册

FastDeploy 使用 `__init__.py` 自动扫描 `models/` 目录，**无需手动注册**。只要文件放在 `fastdeploy/model_executor/models/` 下，装饰器就会被自动加载。

验证命令：
```python
from fastdeploy.model_executor.models import ModelRegistry
print(ModelRegistry.get_supported_archs())
# 应该能看到你的新 architecture name
```

### 4b. 更新 model_type 条件分支（如有需要）

如果你的新模型与某个已有模型共享同一个 Python 文件（如 glm_moe_dsa 复用 deepseek_v3.py），需要更新对应文件中的 model_type 判断：

```python
# 在 deepseek_v3.py 或其他共享文件中
if model_type in ["deepseek_v3", "deepseek_v32", "<your_new_model_type>"]:
    self.attn = DeepseekV3MLAAttention(...)
```

### 4c. 更新 supported_models.md

在 `docs/supported_models.md` 表格中添加新行：

```markdown
| <ModelName> | <ModelSize> | BF16 | ✅ | ✅ | - |
```

---

## 步骤 5：补充文档

在 `docs/` 目录下创建或更新模型文档。参考 `references/doc_template.md` 生成标准文档，包含：

1. 模型简介（架构特点）
2. 部署命令（最小可运行示例）
3. 性能指标（如已有 benchmark）
4. 注意事项（量化兼容性、TP 限制等）

---

## 步骤 6：生成部署测试脚本

生成两种测试脚本：

### 快速验证脚本（本地调试用）

```python
# test_<model_name>_inference.py
"""Quick sanity check for <ModelName> integration in FastDeploy."""
import subprocess, sys

MODEL_PATH = "<model_path>"  # 用户提供的路径

def test_model_loads():
    """Test that the model architecture is correctly registered."""
    from fastdeploy.model_executor.models import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    assert "<NewModelArchitecture>ForCausalLM" in archs, \
        f"Model not registered! Available: {archs}"
    print("✅ Model registration: PASS")

def test_basic_inference():
    """Run a simple single-GPU inference test."""
    result = subprocess.run([
        "python", "-m", "fastdeploy.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--max-model-len", "1024",
        "--tensor-parallel-size", "1",
        # Add --dry-run or short test here if supported
    ], capture_output=True, text=True, timeout=120)
    print(result.stdout[-2000:])  # Last 2000 chars
    print("✅ Server startup: PASS" if result.returncode == 0 else "❌ FAIL")

if __name__ == "__main__":
    test_model_loads()
    test_basic_inference()
```

### 完整部署命令（生产用）

```bash
# Single GPU
python -m fastdeploy.entrypoints.openai.api_server \
    --model <model_path> \
    --tensor-parallel-size 1 \
    --max-model-len 32768

# Multi-GPU (8-way TP)
python -m fastdeploy.entrypoints.openai.api_server \
    --model <model_path> \
    --tensor-parallel-size 8 \
    --max-model-len 131072

# MoE with Expert Parallelism
python -m fastdeploy.entrypoints.openai.api_server \
    --model <model_path> \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --max-model-len 131072

# curl 测试
curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "<model_name>", "prompt": "Hello, I am", "max_tokens": 50}'
```

---

## 参考资料

- **代码模板**：读取 `references/model_templates.md` 获取完整代码样板
- **架构决策树**：`references/architecture_guide.md` — 更详细的架构选型指南
- **PR 参考**：
  - [PR #6863](https://github.com/PaddlePaddle/FastDeploy/pull/6863) — GLM-MoE-DSA（继承 DeepSeekV3，最简继承示例）
  - [PR #7139](https://github.com/PaddlePaddle/FastDeploy/pull/7139) — GLM4.7 Flash（ForwardMeta 参数化模式）
  - [PR #6689](https://github.com/PaddlePaddle/FastDeploy/pull/6689) — DeepSeek-v3.2（自定义 CUDA kernel 集成）

---

## 输出物清单

完成后，向用户提供以下文件：

1. `fastdeploy/model_executor/models/<model_name>.py` — 模型实现
2. `docs/<model_name>_deployment.md` — 部署文档
3. `test_<model_name>_inference.py` — 测试脚本
4. （如需要）修改说明：`deepseek_v3.py` 或其他共享文件中新增的 model_type 分支

---

## 常见陷阱

**陷阱 1：architecture name 不匹配**
`@ModelRegistry.register_model_class(architecture=...)` 中的字符串必须与模型 `config.json` 中 `architectures[0]` 完全一致，大小写敏感。

**陷阱 2：忘记 model_type 条件**
如果你的模型继承了 DeepSeekV3 但 attention 类型不同，需要在父类中添加 `model_type` 判断，否则会走错 attention 路径。

**陷阱 3：Tensor Parallelism 配置**
`num_key_value_heads` 必须能被 `tensor_parallel_size` 整除，否则需要使用 head padding（参考 PR #7139 中的 padding 逻辑）。

**陷阱 4：MoE 专家权重格式**
MoE 模型的专家权重需要用 `FusedMoE.make_expert_params_mapping()` 做参数映射，不能直接用标准的 `stacked_params_mapping`。

**陷阱 5：PretrainedModel 未注册**
如果你没有创建 `PretrainedModel` 子类，tensor parallelism mapping 会缺失，多卡推理可能出现权重切分错误。
