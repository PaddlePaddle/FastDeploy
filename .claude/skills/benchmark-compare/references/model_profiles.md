# 模型部署配置表 (Model Profiles)

本文件记录各模型在 FastDeploy 和 SGLang 上的推荐部署参数。Agent 在步骤 1 中根据 `MODEL_PATH` 匹配模型名称，获取推荐配置。

**匹配规则：** 取 `MODEL_PATH` 的最后一级目录名，模糊匹配下表中的 "匹配关键词" 列。

---

## 配置表

| 模型名 | 匹配关键词 | 最小 TP | 推荐量化 | max-model-len | FD Attention Backend | FD 环境变量 | SG Attention Backend | gpu-memory-utilization | Hyperparameter YAML | 备注 |
|--------|-----------|---------|----------|---------------|---------------------|-------------|---------------------|----------------------|-------------------|------|
| GLM-4.7-Flash | `GLM-4.7`, `GLM4.7`, `glm-4.7` | 1 | none | 65536 | MLA_ATTN | `USE_FLASH_MLA=1`, `FLAGS_flash_attn_version=3`, `FD_SAMPLING_CLASS=rejection` | flashmla | 0.97 | `GLM-32k.yaml` | MoE 模型 ~59GB，单卡 H800 刚好放下 |
| DeepSeek-V2 | `DeepSeek-V2`, `deepseek-v2` | 1 | none/fp8 | 65536 | MLA_ATTN | `USE_FLASH_MLA=1`, `FLAGS_flash_attn_version=3`, `FD_SAMPLING_CLASS=rejection` | flashmla | 0.97 | `deepseek-32k.yaml` | MoE + MLA |
| DeepSeek-V3 | `DeepSeek-V3`, `deepseek-v3` | 8 | fp8 | 32768 | MLA_ATTN | `USE_FLASH_MLA=1`, `FLAGS_flash_attn_version=3`, `FD_SAMPLING_CLASS=rejection` | flashmla | 0.95 | `deepseek-32k.yaml` | 671B MoE，需多机或 PD 分离 |
| DeepSeek-R1 | `DeepSeek-R1`, `deepseek-r1` | 8 | fp8 | 32768 | MLA_ATTN | `USE_FLASH_MLA=1`, `FLAGS_flash_attn_version=3`, `FD_SAMPLING_CLASS=rejection` | flashmla | 0.95 | `deepseek-32k.yaml` | 671B，同 V3 架构 |
| Qwen2.5-7B | `Qwen2.5-7B`, `qwen2.5-7b` | 1 | none | 32768 | - | `FD_SAMPLING_CLASS=rejection` | - | 0.90 | `qwen2-32k.yaml` | Dense 7B，单卡轻松 |
| Qwen2.5-72B | `Qwen2.5-72B`, `qwen2.5-72b` | 4 | wint4/fp8 | 32768 | - | `FD_SAMPLING_CLASS=rejection` | - | 0.95 | `qwen2-32k.yaml` | Dense 72B |
| Qwen3-235B | `Qwen3-235B`, `qwen3-235b` | 4 | fp8 | 32768 | - | `FD_SAMPLING_CLASS=rejection` | - | 0.95 | `qwen3-32k.yaml` | MoE 235B，EP 可选 |
| Qwen3-30B-A3B | `Qwen3-30B`, `qwen3-30b` | 1 | none/fp8 | 32768 | - | `FD_SAMPLING_CLASS=rejection` | - | 0.95 | `qwen3-32k.yaml` | MoE 30B (3B active) |
| ERNIE-4.5 | `ernie-4.5`, `ERNIE-4.5`, `eb45` | 1 | none | 32768 | - | `FD_SAMPLING_CLASS=rejection` | - | 0.90 | `eb45-32k.yaml` | 快速验证用 |
| Llama-3.1-70B | `Llama-3.1-70B`, `llama-3.1-70b` | 4 | fp8 | 32768 | - | `FD_SAMPLING_CLASS=rejection` | - | 0.95 | `request.yaml` | Dense 70B |
| Llama-3.1-8B | `Llama-3.1-8B`, `llama-3.1-8b` | 1 | none | 32768 | - | `FD_SAMPLING_CLASS=rejection` | - | 0.90 | `request.yaml` | Dense 8B |

---

## 字段说明

| 字段 | 说明 |
|------|------|
| **最小 TP** | 模型权重能装入 GPU 所需的最小 tensor-parallel 数。80GB H800 为基准 |
| **推荐量化** | `none`=BF16 全精度；`fp8`/`block_wise_fp8`=FP8 量化；`wint4`/`wint8`=权重量化 |
| **max-model-len** | FD 的 `--max-model-len` / SG 的 `--context-length` |
| **FD Attention Backend** | FD 的 `FD_ATTENTION_BACKEND` 环境变量。MLA 架构模型用 `MLA_ATTN` |
| **FD 环境变量** | 启动 FD 时需额外设置的环境变量 |
| **SG Attention Backend** | SG 的 `--attention-backend` 参数。MLA 架构用 `flashmla`；默认 `-` 表示不指定 |
| **gpu-memory-utilization** | MoE 大模型建议 0.97；Dense 模型 0.90-0.95 |
| **Hyperparameter YAML** | benchmark 使用的 request hyperparameter 文件（相对于 `FastDeploy/benchmarks/yaml/request_yaml/`） |

---

## 如何添加新模型

1. 在上表中新增一行
2. 如果模型需要新的 hyperparameter 配置，在 `FastDeploy/benchmarks/yaml/request_yaml/` 创建对应 YAML
3. 如果模型使用新的 attention 机制，确认 FD 和 SG 的 attention backend 参数
4. 如果是 MoE 模型，注意 `gpu-memory-utilization` 需设高（0.95-0.97）

---

## 部署模式参考

### 单卡 (TP=1)
- 适用：≤80GB 的模型（如 GLM-4.7-Flash ~59GB, Qwen2.5-7B ~14GB）
- GPU 需求：2 张（FD 1张 + SG 1张）

### 多卡 TP (TP>1)
- 适用：>80GB 的 Dense 模型（如 Qwen2.5-72B 需 TP=4）
- GPU 需求：2×TP 张
- FD: `--tensor-parallel-size N`
- SG: `--tp N`

### PD 分离 (Prefill-Decode Separation)
- 适用：超大模型需极致吞吐（如 DeepSeek-V3）
- 仅 FD 支持：
  - Prefill 实例：`--splitwise-role prefill`
  - Decode 实例：`--splitwise-role decode`
  - 需配置 RDMA/IPC 通信
- SG 作为标准模式对比基线

### 多机 (Multi-Node)
- 适用：单机放不下的模型（如 DeepSeek-V3 671B 需 8×H800）
- 需用户提供各节点 IP 和 SSH 配置
- Benchmark 使用 `--ip-list` 分发请求
