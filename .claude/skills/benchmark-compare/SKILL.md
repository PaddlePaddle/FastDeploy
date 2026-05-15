---
name: benchmark
description: >
FastDeploy vs SGLang 推理框架性能对比测试工具。自动完成环境安装、服务启动、
性能测试、结果可视化全流程。支持单卡/多卡 TP/多机 PD 分离部署模式。
也支持仅从用户提供的日志/数据生成多模式可视化 HTML 报告（无需启动服务）。
触发方式：/benchmark 或 "帮我跑 benchmark"、"对比测试 FD 和 SG"、"性能对比"、"生成报告"
user_invocable: true
---

# Benchmark 推理框架对比测试

自动完成 FastDeploy 与 SGLang 推理框架的性能对比测试，生成可视化 HTML 报告。

---

## 两种工作模式

### 模式 A：全自动测试（默认）

完整 12 步流程：环境安装 → 服务部署 → 性能测试 → 提取指标 → 生成报告。

### 模式 B：仅生成报告（Report-Only）

用户已有测试数据（日志文件 / 手动结果），只需生成多模式可视化 HTML 报告。

**触发条件**（满足任一）：
- 用户说 "生成报告"、"出报告"、"整理结果到 HTML"
- 用户提供了日志文件路径或贴出了 benchmark 结果数据
- 用户明确说不需要重新跑测试

**模式 B 流程**：跳转到 → [仅生成报告流程](#仅生成报告流程report-only)

---

## 参数表

用户只需提供以下参数（未提供则使用默认值）：

| 参数 | 键名 | 默认值 | 说明 |
|------|------|--------|------|
| 模型路径 | `MODEL_PATH` | `<path_to_model>` | 模型权重目录 |
| 并发数 | `CONCURRENCY` | `32` | 最大并发请求数 |
| 是否量化 | `QUANTIZATION` | `none` | `none` / `block_wise_fp8`(FD) + `fp8`(SG) / `wint4` / `wint8`。注意：用户说"FP8"时，FD 实际使用 Block-Wise FP8（`--quantization block_wise_fp8`），SG 使用 per-tensor FP8（`--quantization fp8`），两者量化粒度不同，报告中需明确标注 |
| 数据集路径 | `DATASET_PATH` | `<path_to_dataset>` | JSONL 格式 |
| TP 大小 | `TP_SIZE` | `1` | tensor-parallel-size |
| DP 大小 | `DP_SIZE` | `1` | data-parallel-size |
| EP 大小 | `EP_SIZE` | `0` | expert-parallel-size，MoE 模型专用。FD 映射为 `--enable-expert-parallel`（EP size 隐式=TP×DP），SG 映射为 `--ep-size N` |
| 部署模式 | `DEPLOY_MODE` | `single` | `single` / `tp` / `tp_dp_ep` / `pd` / `multi-node` |
| FD 端口 | `FD_PORT` | `8180` | FastDeploy 服务端口 |
| SG 端口 | `SG_PORT` | `8280` | SGLang 服务端口 |
| GPU 列表 | `GPU_LIST` | 自动选择空闲卡 | 逗号分隔，如 `0,1,2,3` |
| 最大序列长度 | `MAX_MODEL_LEN` | `65536` | max-model-len / context-length |
| 请求数 | `NUM_PROMPTS` | `1024` | benchmark 发送的总请求数 |
| 输出目录 | `OUTPUT_DIR` | `<SKILL_ROOT>/..` (Ducc 根目录) | 结果文件存放位置 |
| Hyperparameter YAML | `HYPER_YAML` | 自动匹配（参考 model_profiles） | 请求采样参数配置 |

**示例 prompt：**
- `/benchmark` — 全部使用默认值
- `帮我跑 benchmark，模型用 /path/to/Qwen2.5-72B，TP=4，并发 64`
- `对比测试 GLM-4.7-Flash，开启 fp8 量化`

---

## 部署模式决策树

根据参数自动选择部署策略：

```
用户指定 DEPLOY_MODE?
├── YES → 使用指定模式
└── NO → 根据 TP_SIZE 推断
    ├── TP_SIZE = 1 → single 模式
    │   └── 选两张空闲 GPU，FD 和 SG 各占一张
    ├── TP_SIZE > 1 → tp 模式
    │   └── 需要 2 × TP_SIZE 张空闲 GPU
    │       ├── FD: CUDA_VISIBLE_DEVICES=前 TP 张
    │       └── SG: CUDA_VISIBLE_DEVICES=后 TP 张
    └── 用户指定 pd → pd 模式
        └── FD 启动 prefill + decode 两个进程
            SG 使用标准模式作为对比基线

multi-node 模式:
  └── 用户需提供各节点 IP
      ├── FD: 每节点启动一个 worker
      └── SG: --tp N (跨节点)
      └── benchmark: --ip-list 参数分发请求
```

**GPU 选择逻辑：**
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 选择 memory.used = 0 MiB 的卡
# 需要的 GPU 数量 = 2 × TP_SIZE（single/tp 模式）
```

---

## 自动执行流程

```
步骤 1: 参数解析与模型 Profile 匹配
步骤 2: 检查环境（FastDeploy / SGLang 是否已安装）
步骤 3: 如未安装，执行环境安装
步骤 4: 检查 GPU 空闲情况，分配 GPU
步骤 5: 启动 FastDeploy 服务（调用 scripts/launch_service.sh）
步骤 6: 启动 SGLang 服务（调用 scripts/launch_service.sh）
步骤 7: 等待服务就绪（调用 scripts/health_check.sh）
步骤 8: 运行 FastDeploy benchmark（调用 scripts/run_benchmark.sh）
步骤 9: 运行 SGLang benchmark（调用 scripts/run_benchmark.sh）
步骤 10: 提取指标（调用 scripts/extract_metrics.py）
步骤 11: 生成 HTML 可视化报告（参考 references/html_template.md）
步骤 12: 展示对比摘要，清理服务进程
```

---

## 步骤 1：参数解析与 Profile 匹配

1. 从用户 prompt 中提取参数，未提供的使用默认值
2. 读取 `references/model_profiles.md`，匹配模型路径对应的推荐配置
3. 如果 model_profiles 中有该模型，使用推荐的 TP_SIZE / QUANTIZATION / MAX_MODEL_LEN / HYPER_YAML
4. 用户显式指定的参数优先级高于 profile 推荐值

---

## 步骤 2-3：环境检查与安装

### 检查已安装

```bash
SKILL_ROOT="$(cd "$(dirname "$0")" && pwd)"
DUCC_ROOT="$(dirname "$(dirname "$(dirname "$SKILL_ROOT")")")"

# FastDeploy
ls "$DUCC_ROOT/FastDeploy/.venv/bin/activate" 2>/dev/null && echo "FD_INSTALLED"
# SGLang
ls "$DUCC_ROOT/sglang_env/.venv/bin/activate" 2>/dev/null && echo "SG_INSTALLED"
```

如果已安装则跳过。

### 安装 FastDeploy

**关键：必须使用 Python 3.10**（PaddlePaddle wheel 只有 cp310）。

```bash
cd "$DUCC_ROOT"
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
uv venv --python /usr/bin/python3.10
source .venv/bin/activate
uv pip install pip
python -m pip install <paddle_gpu_whl_url>
pip install -r requirements.txt
```

设置 LD_LIBRARY_PATH 并编译：
```bash
export LD_LIBRARY_PATH=$(python3 -c "
import site, os
sp = site.getsitepackages()[0]
nvidia_dir = os.path.join(sp, 'nvidia')
libs = [os.path.join(nvidia_dir, d, 'lib') for d in os.listdir(nvidia_dir) if os.path.isdir(os.path.join(nvidia_dir, d, 'lib'))]
print(':'.join(libs))
"):$LD_LIBRARY_PATH

bash build.sh 0 python false '[90]'
pip install -e . --no-build-isolation
```

安装 FlashMLA：
```bash
cd "$DUCC_ROOT"
git clone https://github.com/PFCCLab/FlashMLA.git
cd FlashMLA
git submodule update --init --recursive
source ../FastDeploy/.venv/bin/activate
pip install -v . --no-build-isolation
cd ..
```

### 代码修改（两个文件都要改）

在 `FastDeploy/benchmarks/backend_request_func.py` 和 `FastDeploy/benchmarks/backend_request_func_swe.py` 中，找到：
```python
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos

    headers = {
```

替换为：
```python
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos

    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and "min_tokens" in metadata and "min_tokens" not in payload:
        payload["min_tokens"] = metadata["min_tokens"]

    fixed_len_mode = (
        isinstance(payload.get("min_tokens"), int)
        and isinstance(payload.get("max_tokens"), int)
        and payload["min_tokens"] == payload["max_tokens"]
    )
    if fixed_len_mode:
        payload.setdefault("ignore_eos", True)
        payload.setdefault("stop", [])

    headers = {
```

### 安装 SGLang

```bash
cd "$DUCC_ROOT"
mkdir -p sglang_env
python3.10 -m venv --without-pip sglang_env/.venv
source sglang_env/.venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
pip install "sglang[all]==0.5.10.post1"
```

---

## 步骤 4：GPU 分配

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

根据部署模式分配：
- **single (TP=1)**: 选 2 张空闲卡 → `FD_GPUS=<卡1>`, `SG_GPUS=<卡2>`
- **tp (TP>1)**: 选 `2×TP` 张空闲卡 → `FD_GPUS=<前TP张>`, `SG_GPUS=<后TP张>`
- **pd**: 选 `TP+1+TP` 张（FD prefill TP张 + FD decode 1张 + SG TP张）

如果用户通过 `GPU_LIST` 指定了具体卡号，直接使用。

---

## 步骤 5-6：启动服务

调用 `scripts/launch_service.sh`：

```bash
# 启动 FastDeploy
bash scripts/launch_service.sh \
  --framework fd \
  --model "$MODEL_PATH" \
  --port "$FD_PORT" \
  --gpus "$FD_GPUS" \
  --tp "$TP_SIZE" \
  --concurrency "$CONCURRENCY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --quantization "$QUANTIZATION" \
  --log-file "$OUTPUT_DIR/fd_server.log" \
  --venv "$DUCC_ROOT/FastDeploy/.venv"

# 启动 SGLang
bash scripts/launch_service.sh \
  --framework sg \
  --model "$MODEL_PATH" \
  --port "$SG_PORT" \
  --gpus "$SG_GPUS" \
  --tp "$TP_SIZE" \
  --concurrency "$CONCURRENCY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --quantization "$QUANTIZATION" \
  --log-file "$OUTPUT_DIR/sg_server.log" \
  --venv "$DUCC_ROOT/sglang_env/.venv"
```

PD 分离模式额外参数：
```bash
bash scripts/launch_service.sh \
  --framework fd \
  --pd-role prefill \
  --port "$FD_PORT" \
  ...

bash scripts/launch_service.sh \
  --framework fd \
  --pd-role decode \
  --port "$((FD_PORT+1))" \
  ...
```

---

## 步骤 7：等待服务就绪

```bash
bash scripts/health_check.sh --host 127.0.0.1 --port "$FD_PORT" --timeout 300 --log-file "$OUTPUT_DIR/fd_server.log"
bash scripts/health_check.sh --host 127.0.0.1 --port "$SG_PORT" --timeout 300 --log-file "$OUTPUT_DIR/sg_server.log"
```

---

## 步骤 8-9：运行 Benchmark

```bash
# 测试 FastDeploy
bash scripts/run_benchmark.sh \
  --label fd \
  --model "$MODEL_PATH" \
  --port "$FD_PORT" \
  --dataset "$DATASET_PATH" \
  --hyperparams "$HYPER_YAML" \
  --concurrency "$CONCURRENCY" \
  --num-prompts "$NUM_PROMPTS" \
  --output "$OUTPUT_DIR/$RESULT_FD"

# 测试 SGLang
bash scripts/run_benchmark.sh \
  --label sg \
  --model "$MODEL_PATH" \
  --port "$SG_PORT" \
  --dataset "$DATASET_PATH" \
  --hyperparams "$HYPER_YAML" \
  --concurrency "$CONCURRENCY" \
  --num-prompts "$NUM_PROMPTS" \
  --output "$OUTPUT_DIR/$RESULT_SG"
```

**结果文件命名规则：** `<ModelShortName>_long_bs<CONCURRENCY>_[<quant>_]<fd|sg>.txt`

---

## 步骤 10：提取指标

```bash
python3 scripts/extract_metrics.py \
  --fd-result "$OUTPUT_DIR/$RESULT_FD" \
  --sg-result "$OUTPUT_DIR/$RESULT_SG" \
  --model-path "$MODEL_PATH" \
  --fd-config '{"gpu":"H800","tp":'$TP_SIZE',"dp":'$DP_SIZE',"ep":'$EP_SIZE',"concurrency":'$CONCURRENCY',"quantization":"'$QUANTIZATION'"}' \
  --sg-config '{"gpu":"H800","tp":'$TP_SIZE',"dp":'$DP_SIZE',"ep":'$EP_SIZE',"concurrency":'$CONCURRENCY',"quantization":"'$QUANTIZATION'"}' \
  --output "$OUTPUT_DIR/metrics.json"
```

---

## 步骤 11：生成 HTML 报告

使用 `scripts/generate_report.py` 生成多模式可视化报告：

```bash
python3 scripts/generate_report.py \
  --data-json "$OUTPUT_DIR/metrics.json" \
  --output "$OUTPUT_DIR/benchmark_report.html" \
  --model-name "$MODEL_NAME" \
  --gpu-type "H800" \
  --tp $TP_SIZE \
  --dp $DP_SIZE \
  --ep $EP_SIZE \
  --default-quant "$QUANTIZATION" \
  --default-bs "$CONCURRENCY"
```

如果只跑了单个场景（单种量化 + 单种并发），报告仍然正确——选择器只有一个选项。

如果跑了多个场景（如多种并发或同时 BF16+FP8），先将各场景指标合并为一个 JSON（格式见下文），再调用 generate_report.py。

**合并 JSON 格式**：
```json
{
  "bf16_bs32": {"fd": {...metrics...}, "sg": {...metrics...}},
  "fp8_bs64": {"fd": {...metrics...}, "sg": {...metrics...}}
}
```

**报告设计规范详见** `references/html_template.md`。

---

## 步骤 12：展示结果与清理

展示 Markdown 对比表格：

```
| 指标 | FastDeploy | SGLang | 差异 | 胜出 |
|------|-----------|--------|------|------|
| Total Token Throughput (tok/s) | xxx | xxx | ±x% | ... |
| Mean TTFT (ms) | xxx | xxx | ±x% | ... |
| Mean TPOT (ms) | xxx | xxx | ±x% | ... |
| Mean ITL (ms) | xxx | xxx | ±x% | ... |
| Mean E2EL (ms) | xxx | xxx | ±x% | ... |
```

清理：
```bash
# 停止服务进程
kill $(lsof -t -i :$FD_PORT) 2>/dev/null
kill $(lsof -t -i :$SG_PORT) 2>/dev/null
```

告知用户 HTML 报告路径。

---

## 关键注意事项

| # | 问题 | 解决方案 |
|---|------|----------|
| 1 | Python 版本 | **必须用 3.10**，`uv venv --python /usr/bin/python3.10` |
| 2 | OOM | **必须加 `--gpu-memory-utilization 0.97`**（MoE 模型权重大） |
| 3 | uv venv 无 pip | 创建后先 `uv pip install pip` |
| 4 | FlashMLA 编译 | 需先设置 `LD_LIBRARY_PATH`（NVIDIA 库路径） |
| 5 | 两框架不能共用 GPU | 各占独立卡，否则显存不够 |
| 6 | 端口冲突 | 启动前检查并 kill 占用进程 |
| 7 | 模型加载慢 | MoE 约 2-4 分钟，耐心轮询 `/v1/models` |
| 8 | SM 架构号 | H800/H100 用 `[90]`，A100 用 `[80]` |
| 9 | benchmark 耗时 | 1024 prompts × 并发 32 约 5-8 分钟 |
| 10 | 代码修改位置 | `backend_request_func.py` 和 `_swe.py` 都要改 |
| 11 | 多卡 TP GPU 数 | 需要 `2×TP` 张空闲 GPU（FD + SG 各 TP 张） |
| 12 | PD 分离 | 仅 FD 支持，SG 作为标准模式基线 |
| 13 | FP8 量化类型差异 | FD 使用 `block_wise_fp8`（分块量化，粒度更细），SG 使用 `fp8`（per-tensor）。报告中需明确标注为 "Block-Wise FP8"，避免用户误解为同一种 FP8 实现 |
| 14 | FP8 并发限制 | FD 的 FP8 模式下 `--max-num-seqs` 建议设为 32（设 64 会导致 MoE 模型 worker crash）。benchmark 的 `--max-concurrency` 可以更高（请求在服务端排队） |
| 15 | **CUDA Graph 必须开启** | **两个框架都必须开启 CUDA Graph**（各自默认行为），这是测试最优性能的前提。FD 默认开启（不要设 `FLAGS_use_cuda_graph=0`）；SG 默认开启（不要加 `--disable-cuda-graph`）。如果 OOM，应通过降低 `max-num-seqs` 或 `gpu-memory-utilization` 来解决，而不是禁用 CUDA Graph |
| 16 | SGLang DP 端口冲突 | SGLang 在 DP>1 时，torch.distributed 初始化可能与系统服务（18xxx 端口范围）冲突。解决方案：启动前 `export MASTER_PORT=45000`（`launch_service.sh` 已自动处理）|
| 17 | 报告展示部署方式 | HTML 报告中必须显示 TP/DP/EP 配置。使用 `generate_report.py --tp N --dp N --ep N` 参数传入 |

---

## 仅生成报告流程（Report-Only）

当用户已有测试数据，无需重新跑测试，只需生成 HTML 报告时使用。

### 数据输入方式

Agent 需要从以下任一来源获取数据：

**方式 1：用户提供日志文件路径**
- 日志文件是 `benchmark_serving.py` 的标准输出
- Agent 使用 `extract_metrics.py` 或直接正则解析
- 文件命名约定：`*_bs<N>_[<quant>_]<fd|sg>.txt`

**方式 2：用户直接贴出数据/表格**
- Agent 从用户消息中提取关键指标
- 构建 JSON 数据对象

**方式 3：用户提供 JSON 文件**
- 直接使用已整理好的 metrics JSON

### 执行步骤

```
步骤 R1: 确认数据来源和场景维度（哪些量化 × 哪些并发）
步骤 R2: 从日志/数据中提取各场景的 FD 和 SG 指标
步骤 R3: 构建合并 JSON（格式：{quant}_bs{concurrency}）
步骤 R4: 确认模型信息和部署配置（向用户确认或从日志推断）
步骤 R5: 调用 generate_report.py 或直接内联生成 HTML
步骤 R6: 输出 HTML 报告路径
```

### 步骤 R2 详细：从日志文件解析指标

```bash
# 方法 A：使用 extract_metrics.py 逐场景提取
python3 scripts/extract_metrics.py \
  --fd-result /path/to/fd_result.txt \
  --sg-result /path/to/sg_result.txt \
  --model-path /path/to/model \
  --output /tmp/metrics_scene1.json

# 方法 B：使用 generate_report.py 自动扫描目录
python3 scripts/generate_report.py \
  --log-dir /path/to/log_directory \
  --model-name "GLM-4.7-Flash" \
  --output benchmark_report.html
```

日志文件扫描规则（`generate_report.py --log-dir`）：
- 递归扫描目录下所有 `.txt` 文件
- 从文件名正则匹配 `_bs<N>_[<quant>_]<fd|sg>.txt`
- 自动归类到对应场景

### 步骤 R4 详细：向用户确认的信息

如果无法从数据/日志中推断，需要向用户确认：
- 模型名称
- GPU 型号和数量
- TP 大小
- FD/SG 的 Attention Backend
- SGLang 版本
- 默认展示的量化和并发

### 步骤 R5 详细：生成报告

```bash
python3 scripts/generate_report.py \
  --data-json all_metrics.json \
  --output benchmark_report.html \
  --model-name "GLM-4.7-Flash" \
  --model-type "MoE (glm4_moe_lite)" \
  --model-size "~58.2 GB" \
  --model-experts "64R + 1S (Active: 4)" \
  --model-layers-hidden "47 / 2048" \
  --gpu-type H800 --tp 1 --dp 1 --ep 0 \
  --max-model-len 65536 \
  --fd-attention "MLA_ATTN (FlashAttn v3)" \
  --sg-attention "flashmla" \
  --sg-version "0.5.10.post1" \
  --default-quant bf16 --default-bs 512 \
  --dataset-url "https://..." \
  --dataset-desc "browsecomp_plus (830 samples)" \
  --test-date "2026-05-07"
```

### 注意事项

- 如果用户只提供了部分场景的数据，报告选择器中只会出现有数据的选项
- 如果某个场景只有 FD 或只有 SG 的数据，该场景会被跳过（需要配对）
- Agent 也可以不调用脚本，直接根据 `references/html_template.md` 的规范生成完整 HTML 内联到文件中（适合数据已在对话上下文中的情况）

---

## 扩展指南

### 添加新模型

1. 在 `references/model_profiles.md` 添加一行模型配置
2. 如需特殊 hyperparameter，在 `FastDeploy/benchmarks/yaml/request_yaml/` 新建 YAML
3. 运行 `/benchmark --model <新模型路径>` 即可

### 添加新框架

1. 在 `scripts/launch_service.sh` 添加新的 `--framework` 分支
2. 在 `scripts/run_benchmark.sh` 添加对应的 backend 类型
3. 在 `scripts/extract_metrics.py` 添加结果解析逻辑
4. 更新 `references/html_template.md` 支持多框架对比

### 多机部署

1. 用户需提供 `--node-list <ip1,ip2,...>`
2. `launch_service.sh` 通过 SSH 在各节点启动 worker
3. `run_benchmark.sh` 使用 `--ip-list` 分发到多个 endpoint
4. 需要 RDMA/IPC 配置（参考 FD service YAML 中的 `cache_transfer_protocol`）
