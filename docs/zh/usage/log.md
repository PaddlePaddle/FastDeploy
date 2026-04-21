[English](../../usage/log.md)

# 日志说明

FastDeploy 在部署过程中，会产生如下日志文件，各日志含义说明。
默认日志目录为执行目录下的 `log` 文件夹，若需要指定可设置环境变量 `FD_LOG_DIR`。

## 日志通道划分

FastDeploy 将日志分为三个通道：

| 通道 | Logger 名称 | 输出文件 | 说明 |
|------|-------------|----------|------|
| main | `fastdeploy.main.*` | `fastdeploy.log`, `console.log` | 主日志，记录系统配置、启动信息等 |
| request | `fastdeploy.request.*` | `request.log` | 请求日志，记录请求生命周期和处理细节 |
| console | `fastdeploy.console.*` | `console.log` | 控制台日志，输出到终端和 console.log |

## 请求日志级别

请求日志 (`request.log`) 支持 4 个级别，通过环境变量 `FD_LOG_REQUESTS_LEVEL` 控制：

| 级别 | 枚举名 | 说明 | 示例内容 |
|------|--------|------|----------|
| 0 | LIFECYCLE | 生命周期起止 | 请求创建/初始化、完成统计（InputToken/OutputToken/耗时）、流式响应首次和最后发送、请求中止 |
| 1 | STAGES | 处理阶段 | 信号量获取/释放、首 token 时间记录、信号处理（preemption/abortion/recovery）、缓存任务、预处理耗时、参数调整警告 |
| 2 | CONTENT | 内容和调度 | 请求参数、处理后的请求、调度信息（入队/拉取/完成）、响应内容（超长内容会被截断） |
| 3 | FULL | 完整数据 | 完整的请求和响应数据、原始接收请求 |

默认级别为 2 (CONTENT)，记录请求参数、调度信息和响应内容。较低级别 (0-1) 只记录关键事件，级别 3 则包含完整原始数据。

## 日志相关环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `FD_LOG_DIR` | `log` | 日志文件存储目录 |
| `FD_LOG_LEVEL` | `INFO` | 日志级别，支持 `INFO` 或 `DEBUG` |
| `FD_LOG_REQUESTS` | `1` | 是否启用请求日志，`0` 禁用，`1` 启用 |
| `FD_LOG_REQUESTS_LEVEL` | `2` | 请求日志级别，范围 0-3 |
| `FD_LOG_MAX_LEN` | `2048` | L2 级别日志内容的最大长度（超出部分会被截断） |
| `FD_LOG_BACKUP_COUNT` | `7` | 日志文件保留数量 |
| `FD_DEBUG` | `0` | 调试模式，`1` 启用时日志级别设为 `DEBUG` |

## 推理服务日志

* `fastdeploy.log` : 主日志文件，记录系统配置、启动信息、运行状态等
* `request.log` : 请求日志文件，记录用户请求的生命周期和处理细节
* `console.log` : 控制台日志，记录模型启动耗时等信息，该日志信息会被打印到控制台
* `error.log` : 错误日志文件，记录所有 ERROR 及以上级别的日志
* `backup_env.*.json` : 记录当前实例启动时设置的环境变量，文件个数与卡数相同
* `workerlog.*` : 记录模型启动加载进度及推理算子报错信息，每个卡对应一个文件
* `worker_process.log` : 记录引擎每一轮推理的数据
* `cache_manager.log` : 记录每一个请求分配 KV Cache 的逻辑索引，以及当前请求的命中情况
* `launch_worker.log` : 记录模型启动信息及报错信息
* `gpu_worker.log` : 记录 profile 时计算 KV Cache block 数目的信息
* `gpu_model_runner.log` : 当前的模型信息及加载时间

## 调度器日志
* `scheduler.log` : 记录调度器的信息包含当前结点的信息，每条请求分配的信息

## 投机解码日志
* `speculate.log` : 投机解码相关信息

## Prefix Caching 相关日志

* `cache_queue_manager.log` : 记录启动参数，及接收到的请求信息
* `cache_transfer_manager.log` : 记录启动参数，及接收到的请求信息
* `launch_cache_manager.log` : 启动 cache transfer 记录启动参数，报错信息

## PD 分离相关日志

* `cache_messager.log` : 记录 P 实例使用的传输协议及传输信息
* `splitwise_connector.log` : 记录收到 P/D 发送的数据，及建联信息

## CudaGraph 相关日志

* `cudagraph_piecewise_backend.log` : 记录 cuda graph 启动及报错信息
