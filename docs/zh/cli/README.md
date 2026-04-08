# FastDeploy CLI 使用说明文档

## 简介
**FastDeploy CLI** 是 FastDeploy 推理框架提供的命令行工具，用于**运行、部署和测试 AI 模型的推理任务**。它帮助开发者在命令行中快速完成模型加载、接口调用、服务启动、性能评测以及环境信息收集等常见工作。

通过 FastDeploy CLI，您可以：

* 🚀 **运行与验证模型推理**：直接在命令行中进行对话生成或文本补全（`chat`、`complete`）
* 🧩 **服务化部署模型**：一键启动与 OpenAI 协议兼容的 API 服务（`serve`）
* 📊 **执行性能与效果评测**：进行延迟、吞吐、任务评估等基准测试（`bench`）
* ⚙️ **收集运行环境信息**：输出系统、框架、GPU 及 FastDeploy 版本配置（`collect-env`）
* 📁 **批量运行推理任务**：支持文件或 URL 输入输出的批处理模式（`run-batch`）
* 🔡 **管理模型的 Tokenizer**：执行文本与 token 的编码、解码及词表导出（`tokenizer`）

### 查看帮助信息
```
fastdeploy --help
```
### 可用命令
```
fastdeploy {chat, complete, serve, bench, collect-env, run-batch, tokenizer}
```

| 命令名称          | 主要功能说明                     | 详细说明链接                              |
| ------------- | -------------------------- | ----------------------------------- |
| `chat`        | 在命令行中进行对话生成任务，用于验证聊天模型推理效果 | [查看 chat 命令说明](chat.md)               |
| `complete`    | 进行文本补全任务，支持多种语言模型输出测试      | [查看 complete 命令说明](complete.md)       |
| `serve`       | 启动与 OpenAI 协议兼容的本地推理服务     | [查看 serve 命令说明](serve.md)             |
| `bench`       | 对模型进行性能（延迟、吞吐）或精度评测        | [查看 bench 命令说明](bench.md)             |
| `collect-env` | 收集并打印系统、GPU、依赖等运行环境信息      | [查看 collect-env 命令说明](collect-env.md) |
| `run-batch`   | 批量执行推理任务，支持文件/URL输入输出      | [查看 run-batch 命令说明](run-batch.md)     |
| `tokenizer`   | 执行文本与 token 的编码、解码及词表导出    | [查看 tokenizer 命令说明](tokenizer.md)     |
