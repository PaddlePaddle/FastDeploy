[简体中文](../zh/usage/log.md)

# Log Description

FastDeploy generates the following log files during deployment. Below is an explanation of each log's purpose.
By default, logs are stored in the `log` directory under the execution path. To specify a custom directory, set the environment variable `FD_LOG_DIR`.

## Log Channel Separation

FastDeploy separates logs into three channels:

| Channel | Logger Name | Output Files | Description |
|---------|-------------|--------------|-------------|
| main | `fastdeploy.main.*` | `fastdeploy.log`, `console.log` | Main logs for system configuration, startup info, etc. |
| request | `fastdeploy.request.*` | `request.log` | Request logs for request lifecycle and processing details |
| console | `fastdeploy.console.*` | `console.log` | Console logs, output to terminal and console.log |

## Request Log Levels

Request logs (`request.log`) support 4 levels, controlled by the environment variable `FD_LOG_REQUESTS_LEVEL`:

| Level | Enum Name | Description | Example Content |
|-------|-----------|-------------|-----------------|
| 0 | LIFECYCLE | Lifecycle start/end | Request creation/initialization, completion stats (InputToken/OutputToken/latency), first and last streaming response, request abort |
| 1 | STAGES | Processing stages | Semaphore acquire/release, first token time recording, signal handling (preemption/abortion/recovery), cache task, preprocess time, parameter adjustment warnings |
| 2 | CONTENT | Content and scheduling | Request parameters, processed request, scheduling info (enqueue/pull/finish), response content (long content is truncated) |
| 3 | FULL | Complete raw data | Complete request and response data, raw received request |

Default level is 2 (CONTENT), which logs request parameters, scheduling info, and response content. Lower levels (0-1) only log critical events, while level 3 includes complete raw data.

## Log-Related Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FD_LOG_DIR` | `log` | Log file storage directory |
| `FD_LOG_LEVEL` | `INFO` | Log level, supports `INFO` or `DEBUG` |
| `FD_LOG_REQUESTS` | `1` | Enable request logging, `0` to disable, `1` to enable |
| `FD_LOG_REQUESTS_LEVEL` | `2` | Request log level, range 0-3 |
| `FD_LOG_MAX_LEN` | `2048` | Maximum length for L2 level log content (excess is truncated) |
| `FD_LOG_BACKUP_COUNT` | `7` | Number of log files to retain |
| `FD_DEBUG` | `0` | Debug mode, `1` enables DEBUG log level |

## Inference Service Logs

* `fastdeploy.log` : Main log file, records system configuration, startup information, runtime status, etc.
* `request.log` : Request log file, records user request lifecycle and processing details
* `console.log` : Console log, records model startup time and other information. This log is also printed to the console.
* `error.log` : Error log file, records all ERROR and above level logs
* `backup_env.*.json` : Records environment variables set during instance startup. The number of files matches the number of GPU cards.
* `workerlog.*` : Tracks model loading progress and inference operator errors. Each GPU card has a corresponding file.
* `worker_process.log` : Logs engine inference data for each iteration.
* `cache_manager.log` : Records KV Cache logical index allocation for each request and cache hit status.
* `launch_worker.log` : Logs model startup information and error messages.
* `gpu_worker.log` : Records KV Cache block count information during profiling.
* `gpu_model_runner.log` : Contains model details and loading time.

## Scheduler Logs
* `scheduler.log` : Records scheduler information, including node status and request allocation details.

## Speculative Decoding Logs
* `speculate.log` : Contains speculative decoding-related information.

## Prefix Caching Logs
* `cache_queue_manager.log` : Logs startup parameters and received request information.
* `cache_transfer_manager.log` : Logs startup parameters and received request information.
* `launch_cache_manager.log` : Records cache transfer startup parameters and error messages.

## PD Disaggregation Logs
* `cache_messager.log` : Logs transmission protocols and messages used by the P instance.
* `splitwise_connector.log` : Records data received from P/D instances and connection establishment details.

## CudaGraph Logs
* `cudagraph_piecewise_backend.log` : Logs CudaGraph startup and error information.
