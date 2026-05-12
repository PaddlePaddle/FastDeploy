# Benchmark Skill — FastDeploy vs SGLang 性能对比测试

## 概述

自动完成 FastDeploy 与 SGLang 两个推理框架的性能对比测试，包括环境安装、服务启动、benchmark 执行、指标提取和可视化 HTML 报告生成。

## 目录结构

```
benchmark/
├── SKILL.md                    # 主技能定义（工作流编排 + 参数表 + 决策树）
├── README.md                   # 本文件
├── scripts/
│   ├── launch_service.sh       # 通用服务启动脚本（支持 FD/SG, TP/DP/PD）
│   ├── health_check.sh         # 服务健康检查（轮询 /v1/models）
│   ├── run_benchmark.sh        # Benchmark 执行封装
│   └── extract_metrics.py      # 从结果文件提取指标 → JSON
└── references/
    ├── html_template.md        # 完整 HTML 报告模板（含 CSS/JS + 占位符）
    └── model_profiles.md       # 模型推荐部署参数表
```

## 快速使用

### 作为 Agent Skill 使用（推荐）

在 Claude Code / Ducc 中输入：

```
/benchmark
```

或自然语言：

```
帮我跑一下 benchmark，模型用 /path/to/model，并发 64，开启 fp8 量化
```

Agent 会自动读取 SKILL.md 并按流程执行全部 12 个步骤。

### 手动使用脚本

```bash
SKILL_DIR=".claude/skills/benchmark"

# 1. 启动服务
bash $SKILL_DIR/scripts/launch_service.sh \
  --framework fd --model /path/to/model --port 8180 \
  --gpus 0 --tp 1 --concurrency 32 --venv ./FastDeploy/.venv

bash $SKILL_DIR/scripts/launch_service.sh \
  --framework sg --model /path/to/model --port 8280 \
  --gpus 1 --tp 1 --concurrency 32 --venv ./sglang_env/.venv

# 2. 等待就绪
bash $SKILL_DIR/scripts/health_check.sh --port 8180 --timeout 300
bash $SKILL_DIR/scripts/health_check.sh --port 8280 --timeout 300

# 3. 运行 benchmark
bash $SKILL_DIR/scripts/run_benchmark.sh \
  --label fd --model /path/to/model --port 8180 \
  --dataset /path/to/data.jsonl \
  --hyperparams ./FastDeploy/benchmarks/yaml/request_yaml/GLM-32k.yaml \
  --output ./result_fd.txt \
  --venv ./FastDeploy/.venv \
  --benchmark-dir ./FastDeploy/benchmarks

bash $SKILL_DIR/scripts/run_benchmark.sh \
  --label sg --model /path/to/model --port 8280 \
  --dataset /path/to/data.jsonl \
  --hyperparams ./FastDeploy/benchmarks/yaml/request_yaml/GLM-32k.yaml \
  --output ./result_sg.txt \
  --venv ./FastDeploy/.venv \
  --benchmark-dir ./FastDeploy/benchmarks

# 4. 提取指标
python3 $SKILL_DIR/scripts/extract_metrics.py \
  --fd-result ./result_fd.txt \
  --sg-result ./result_sg.txt \
  --model-path /path/to/model \
  --output ./metrics.json
```

## 支持的部署模式

| 模式 | 说明 | GPU 需求 |
|------|------|----------|
| single | 单卡部署，FD 和 SG 各一张 | 2 张 |
| tp | 多卡 Tensor Parallel | 2 × TP 张 |
| pd | PD 分离（仅 FD），SG 标准模式 | TP + 1 + TP 张 |
| multi-node | 多机部署 | 用户指定 |

## 依赖

- Python 3.10（PaddlePaddle cp310 wheel）
- NVIDIA GPU (H800/H100 推荐)
- `uv` (Python 包管理器)
- `curl`, `lsof`, `nvidia-smi`

## 注意事项

- MoE 模型必须使用 `--gpu-memory-utilization 0.97`
- 两个框架不能共用同一张 GPU
- 模型加载约需 2-4 分钟（MoE），请耐心等待
- benchmark 1024 条请求约需 5-8 分钟

## 扩展

- 添加新模型：编辑 `references/model_profiles.md`
- 修改报告样式：编辑 `references/html_template.md`
- 添加新框架：在各 script 中添加新的 `--framework` 分支
