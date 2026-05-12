# Nsys 性能抓取 Skill

## 概述

通用 GPU 推理服务 nsys 性能抓取工具。根据用户的启动脚本自动注入 nsys 命令、向 FastDeploy 代码注入 profiling 埋点、构建带 nsys 的启动脚本，完成完整的 GPU profiling 抓取流程，输出 .nsys-rep 文件。

## 工作流程

完整流程由 SKILL.md 定义，共 5 步：

1. **信息收集** — 从用户启动脚本和描述中提取模型路径、端口、nsys 版本等参数
2. **注入 profiling** — 检查并向 FastDeploy `gpu_model_runner.py` 注入 `nvprof_start`/`nvprof_stop`
3. **生成脚本** — 构建 `start_nsys.sh`（注入 nsys 命令到启动脚本）
4. **用户确认** — 展示生成的脚本和注入的代码，等用户确认
5. **执行抓取** — 启动服务 → 等待就绪 → 发送请求 → 等待文件 → 重命名

## 文件结构

```
~/.claude/skills/nsys-capture/
├── SKILL.md                # Skill 定义文件（完整工作流规范）
├── nsys_utils.sh           # 辅助执行脚本（timeit 等工具）
├── nsys_capture.sh          # 核心函数（wait_service / wait_and_rename_file）
├── nsys_default_client.py   # 默认测试请求客户端
└── README.md               # 本文档
```

## 环境变量

所有参数均通过 Step 1 信息收集从用户启动脚本中自动提取，无需手动设置环境变量。以下为可选覆盖项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NSYS_OUTPUT_DIR` | `/tmp/nsys_record` | nsys 输出目录 |

## 依赖

- nsys (Nsight Systems)
- bash 4.0+
- curl
- python 3.10+
- openai python 库（默认请求客户端使用）

## 常见问题

**Q: nsys 文件未生成？**
1. 检查埋点：`nvprof_start` 和 `nvprof_stop` 各一处，顺序正确
2. 检查请求 token 数是否足够（需 > 埋点触发 iter 数）
3. 用 `nsys sessions list` 查看是否进入过 `RangeCollection` 状态

**Q: 服务启动失败？**
检查 `/tmp/nsys_serve.log` 最后 50 行，常见原因：端口占用、模型路径不存在、GPU 显存不足

**Q: 如何分析 .nsys-rep 文件？**
使用 `nsys-ui <file>.nsys-rep` 打开分析。
