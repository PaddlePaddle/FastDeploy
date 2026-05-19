# Benchmark Skill — FastDeploy vs SGLang 性能对比测试

## 概述

自动完成 FastDeploy 与 SGLang 两个推理框架的性能对比测试，包括环境安装、服务启动、benchmark 执行、指标提取和可视化 HTML 报告生成。最终输出带有量化/并发选择器的交互式 HTML 报告。

## 使用方法

### 触发方式

在 Ducc（Claude Code）中使用 `/benchmark` 命令，或直接用自然语言描述需求：

```
根据benchmark_compare_skill，完成FastDeploy和SGLang性能测试对比：
模型：<path_to_model>
数据集：<path_to_dataset>
并发：64，512
量化：不量化（BF16），FP8（Block-Wise）
使用GPU5和GPU6
```

### 参数说明

| 参数 | 是否必需 | 说明 | 示例 |
|------|---------|------|------|
| 模型路径 | 是 | 模型权重目录的完整路径 | `/path/to/GLM-4.7-Flash` |
| 数据集 | 否（有默认值） | JSONL 格式的测试数据集 | `/path/to/data.jsonl` |
| 并发 | 否（默认 32） | 一个或多个并发数，逗号分隔 | `64，512` |
| 量化 | 否（默认 BF16） | 一种或多种量化方式，FD 使用 Block-Wise FP8，SG 使用 Per-Tensor FP8 | `不量化（BF16），FP8` |
| GPU | 否（自动选空闲卡） | 指定使用哪些 GPU | `使用GPU0-7` |
| TP | 否（默认 1） | tensor-parallel 大小 | `TP=4` |
| DP | 否（默认 1） | data-parallel 大小 | `DP=2` |
| EP | 否（默认不启用） | expert-parallel 大小，MoE 模型专用 | `EP=8` |

### 使用示例

**最简用法**（使用全部默认值）：
```
/benchmark
```

**指定模型和并发**：
```
帮我跑 benchmark，模型用 /path/to/Qwen2.5-72B，TP=4，并发 64
```

**TP+DP+EP 组合**（MoE 模型 8 卡全并行）：
```
对比测试 GLM-4.7-Flash，TP=4，DP=2，EP=8，并发 64 和 512，量化 BF16 和 FP8
```

**多场景对比**（多种量化 × 多种并发）：
```
对比测试 GLM-4.7-Flash，并发 64 和 512，量化 BF16 和 FP8（Block-Wise）
```

**仅生成报告**（已有测试数据）：
```
帮我用 benchmark_results 目录下的日志文件生成 HTML 报告
```

## 工作流程

Agent 会自动执行以下步骤：

1. **参数解析** — 从用户 prompt 提取模型、并发、量化等参数
2. **环境检查** — 检查 FastDeploy / SGLang 是否已安装
3. **环境安装** — 如未安装，自动完成编译安装（Python 3.10）
4. **GPU 分配** — 查找空闲 GPU 或使用用户指定的卡
5. **启动 FastDeploy** — 部署 FD 推理服务
6. **启动 SGLang** — 部署 SG 推理服务
7. **健康检查** — 轮询等待服务就绪
8. **运行 FD Benchmark** — 对 FD 执行压测
9. **运行 SG Benchmark** — 对 SG 执行压测
10. **提取指标** — 从结果文件解析性能数据
11. **生成 HTML 报告** — 输出可视化交互报告
12. **展示摘要** — 输出 Markdown 对比表格，清理服务进程

多种并发 × 多种量化时，Agent 会逐场景执行步骤 5-9，最终合并所有结果生成一份统一报告。

## 输出结果

- **HTML 报告**：`benchmark_results/benchmark_report.html`
  - 支持量化方式切换（BF16 / FP8 Block-Wise）
  - 支持并发数切换（64 / 512 等）
  - 明暗主题切换
  - Chart.js 可视化图表
  - 完整指标对比表格
- **原始日志**：`benchmark_results/GLM-4.7-Flash_long_bs<N>_[<quant>_]<fd|sg>.txt`
- **指标 JSON**：`benchmark_results/metrics_<quant>_bs<N>.json`

## 目录结构

```
benchmark-compare/
├── SKILL.md                    # 主技能定义（工作流编排 + 参数表 + 决策树）
├── README.md                   # 本文件（使用指南）
├── scripts/
│   ├── launch_service.sh       # 通用服务启动脚本（支持 FD/SG, TP/DP/PD）
│   ├── health_check.sh         # 服务健康检查（轮询 /v1/models）
│   ├── run_benchmark.sh        # Benchmark 执行封装
│   ├── extract_metrics.py      # 从结果文件提取指标 → JSON
│   └── generate_report.py      # 生成多模式 HTML 可视化报告
└── references/
    ├── html_template.md        # HTML 报告设计规范
    └── model_profiles.md       # 模型推荐部署参数表
```

## 支持的部署模式

| 模式 | 说明 | GPU 需求 | 触发条件 |
|------|------|----------|----------|
| single | 单卡部署，FD 和 SG 各一张 | 2 张 | TP=1（默认） |
| tp | 多卡 Tensor Parallel | 2 × TP 张 | TP > 1 |
| tp_dp_ep | TP + DP + EP 全并行（MoE 模型） | TP × DP 张（两框架共用同批卡） | TP > 1 且 DP > 1 且 EP > 0 |
| pd | PD 分离（仅 FD），SG 标准模式 | TP + 1 + TP 张 | 用户指定 pd |
| multi-node | 多机部署 | 用户指定 | 用户提供节点 IP |

## 环境依赖

- Python 3.10（PaddlePaddle cp310 wheel 要求）
- NVIDIA GPU（H800/H100 推荐，SM 架构 [90]）
- `uv`（Python 包管理器）
- `curl`, `lsof`, `nvidia-smi`

## 注意事项

- **EP 并行映射差异**：
  - FastDeploy：`--enable-expert-parallel` 为 flag，EP size 隐式等于 TP × DP
  - SGLang：`--ep-size N` 为显式数值参数
  - 典型配置：TP=4, DP=2, EP=8 表示 8 卡全部参与 expert 并行
- **FP8 量化类型**：用户说"FP8"时，实际对应两种不同实现：
  - FastDeploy 使用 `--quantization block_wise_fp8`（Block-Wise FP8，按 block 粒度量化，精度损失更小）
  - SGLang 使用 `--quantization fp8`（Per-Tensor FP8，粗粒度量化）
  - 报告中会明确标注为 "Block-Wise FP8" 以区分，避免误解为相同量化方式
- **GPU 显存**：MoE 模型必须使用 `--gpu-memory-utilization 0.97`，否则加载失败
- **GPU 隔离**：两个框架不能共用同一张 GPU，需各占独立卡
- **FP8 并发限制**：FD 的 FP8 模式下 `--max-num-seqs` 建议设为 32（设 64 会导致 worker crash），benchmark 的 `--max-concurrency` 可以更高（请求排队）
- **服务重启**：每个 benchmark 场景前建议重启服务，避免前一轮残留状态导致 crash
- **加载时间**：MoE 模型加载约需 2-4 分钟，请耐心等待
- **测试时长**：830 条请求 × 并发 64 约需 5-6 分钟

## 扩展指南

- **添加新模型**：编辑 `references/model_profiles.md` 增加模型配置
- **修改报告样式**：编辑 `references/html_template.md` 或 `scripts/generate_report.py`
- **添加新框架**：在各 script 中添加新的 `--framework` 分支
