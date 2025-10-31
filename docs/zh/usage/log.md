[English](../../usage/log.md)

# 日志说明

FastDeploy 在部署过程中，会产生如下日志文件，各日志含义说明
默认日志目录为执行目录下的 `log` 文件夹，若需要指定可设置环境变量 `FD_LOG_DIR`。

## 推理服务日志
* `backup_env.*.json` : 记录当前实例启动时设置的环境变量，文件个数与卡数相同
* `envlog.*` : 记录当前实例启动时设置的环境变量，文件个数与卡数相同
* `console.log` : 记录模型启动耗时等信息，该日志信息会被打印到控制台
* `data_processor.log` : 记录输入数据及输出输出编码解码的内容
* `fastdeploy.log` : 记录当前实例启动的各个 config 的信息，运行中记录用户请求的 request 及 response 信息
* `workerlog.*` : 记录模型启动加载进度及推理算子报错信息，每个卡对应一个文件
* `worker_process.log` : 记录引擎每一轮推理的数据
* `cache_manager.log` : 记录每一个请求分配 KV Cache 的逻辑索引，以及当前请求的命中情况
* `launch_worker.log` : 记录模型启动信息及报错信息
* `gpu_worker.log` : 记录 profile 时计算 KV Cache block 数目的信息
* `gpu_model_runner.log` : 当前的模型信息及加载时间

## 在线推理客户端日志
* `api_server.log` : 记录启动参数，及接收到的请求信息

## 调度器日志
* `scheduler.log` : 记录调度器的信息包含当前结点的信息，每条请求分配的信息

## 投机解码日志
* `speculate.log` : 投机解码相关信息

## Prefix Caching 相关日志

* `cache_queue_manager.log` : 记录启动参数，及接收到的请求信息
* `cache_transfer_manager.log` : 记录启动参数，及接收到的请求信息
* `cache_queue_manager.log` : 记录启动参数，及接收到的请求信息
* `launch_cache_manager.log` : 启动 cache transfer 记录启动参数，报错信息

## PD 分离相关日志

* `cache_messager.log` : 记录P 实例使用的传输协议及传输信息
* `splitwise_connector.log` : 记录收到 P/D 发送的数据，及建联信息

## CudaGraph 相关日志

* `cudagraph_piecewise_backend.log` : 记录 cuda graph 启动及报错信息
