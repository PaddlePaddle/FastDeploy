# FastDeploy CLI User Guide

## Introduction

**FastDeploy CLI** is a command-line tool provided by the FastDeploy inference framework, designed for **running, deploying, and testing AI model inference tasks**. It allows developers to quickly perform model loading, API calls, service deployment, performance benchmarking, and environment information collection directly from the command line.

With FastDeploy CLI, you can:

* 🚀 **Run and validate model inference**: Generate chat responses or text completions directly in the command line (`chat`, `complete`).
* 🧩 **Deploy models as services**: Start an OpenAI-compatible API service with a single command (`serve`).
* 📊 **Perform performance and evaluation tests**: Conduct latency, throughput, and task benchmarks (`bench`).
* ⚙️ **Collect environment information**: Output system, framework, GPU, and FastDeploy version information (`collect-env`).
* 📁 **Run batch inference tasks**: Supports batch input/output from files or URLs (`run-batch`).
* 🔡 **Manage model tokenizers**: Encode/decode text and tokens, or export vocabulary (`tokenizer`).

---

### View Help Information

```bash
fastdeploy --help
```

### Available Commands

```bash
fastdeploy {chat, complete, serve, bench, collect-env, run-batch, tokenizer}
```

---

| Command Name  | Description                                                                                      | Detailed Documentation                             |
| ------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `chat`        | Run interactive chat generation tasks in the command line to verify chat model inference results | [View chat command details](chat.md)               |
| `complete`    | Perform text completion tasks and test various language model outputs                            | [View complete command details](complete.md)       |
| `serve`       | Launch a local inference service compatible with the OpenAI API protocol                         | [View serve command details](serve.md)             |
| `bench`       | Evaluate model performance (latency, throughput) and accuracy                                    | [View bench command details](bench.md)             |
| `collect-env` | Collect and print system, GPU, dependency, and FastDeploy environment information                | [View collect-env command details](collect-env.md) |
| `run-batch`   | Run batch inference tasks with file or URL input/output                                          | [View run-batch command details](run-batch.md)     |
| `tokenizer`   | Encode/decode text and tokens, and export vocabulary                                             | [View tokenizer command details](tokenizer.md)     |
