# write-fastdeploy-unittest

[English](README.md) | 简体中文

用于为 FastDeploy 生成符合 CI 规范的单元测试的 Skill。

## 功能

- 根据被测代码类型自动选择测试模式（纯逻辑 / GPU Kernel / 离线推理 / E2E 服务）
- 遵循 FastDeploy CI 分类规则（multi-GPU 串行 vs single-GPU 并行）
- 满足 80% diff coverage PR 门槛要求
- 正确使用端口变量、日志隔离、资源清理等 CI 约定

## 使用方式

### 基础用法 — 指定源文件

```
使用 write-fastdeploy-unittest skill，为 fastdeploy/cache_manager/transfer_factory/file_store/file_store.py 补充单测
```

### 从覆盖率报告 — 直接粘贴覆盖率行

```
使用 write-fastdeploy-unittest skill，为以下文件补充单测：

fastdeploy/model_executor/model_loader/default_loader.py    48  32  14  0  26%  37-38, 42, 46-52, 56-66, 69-97
```

覆盖率报告格式为：`文件路径  语句总数  未覆盖语句数  分支总数  未覆盖分支数  覆盖率%  未覆盖行号`。Agent 会聚焦未覆盖的行，针对性地编写测试。

### 从增量覆盖率 JSON — PR diff coverage 检查数据

```
使用 write-fastdeploy-unittest skill，为以下文件补充单测：

"fastdeploy/worker/gpu_model_runner.py": {"percent_covered": 0.0, "violation_lines": [1398], "covered_lines": [], "violations": [[1398, null]]}
```

JSON 格式字段说明：
- `percent_covered`：增量行覆盖率百分比
- `violation_lines`：未覆盖的行号列表（需要补测的目标行）
- `covered_lines`：已覆盖的行号列表
- `violations`：详情，格式为 `[[行号, 分支信息]]`

Agent 会聚焦 `violation_lines` 中的行号，针对性地编写测试覆盖这些分支。

### 工作流程

Agent 会自动完成：
1. 读取目标源文件，分析未覆盖行的逻辑
2. **检查是否已有对应的测试文件**（优先在已有文件上追加 test case，避免重复建文件）
3. 选择合适的测试 Pattern（Pattern 1-4）
4. 在已有测试文件中追加，或在 `tests/` 对应子目录下新建测试文件
5. 运行测试并验证覆盖率

## 测试 Pattern 速查

| Pattern | 适用场景 | 依赖 |
|---------|----------|------|
| 1 — Pure Logic | config、utils、scheduler、router 等 | 无 GPU，mock 外部依赖 |
| 2 — GPU Kernel | ops、layers、数值计算 | 需要 GPU，`@pytest.mark.gpu` |
| 3 — Offline Inference | LLM API、模型加载 | 需要 MODEL_PATH |
| 4 — E2E Serving | HTTP 服务端到端 | subprocess + 端口 |

## 关键约定

- 测试文件命名：`test_<module>.py`
- 测试类命名：`Test<Module>`
- 覆盖率验证：`python -m coverage run --source=<目录路径> -m pytest <测试文件> && coverage report -m`
- `--source` 参数接受目录路径（如 `fastdeploy/engine`）或顶层包名（如 `fastdeploy`），不接受点分模块路径（如 `fastdeploy.engine.module`）或 `.py` 文件路径

## 相关文件

- [SKILL.md](SKILL.md) — 完整的 skill 指令文档
