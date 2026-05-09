[简体中文](../zh/usage/log.md)

# Log Description

FastDeploy generates the following log files during deployment. Below is an explanation of each log's purpose.
By default, logs are stored in the `log` directory under the execution path. To specify a custom directory, set the environment variable `FD_LOG_DIR`.

## Log Channel Separation

FastDeploy separates logs into three channels:

| Channel | Logger Name | Output Files | Description |
|---------|-------------|--------------|-------------|
| main | `fastdeploy.main.*` | `fastdeploy.log` | Main logs for system configuration, startup info, etc. |
| request | `fastdeploy.request.*` | `request.log` | Request logs for request lifecycle and processing details |
| console | `fastdeploy.console.*` | `fastdeploy.log` + terminal | Console logs for startup info, etc. Written to fastdeploy.log and also printed to terminal |

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
| `FD_LOG_BACKUP_COUNT` | `7` | Number of log files to retain |
| `FD_DEBUG` | `0` | Debug mode, `1` enables DEBUG log level |
| `FD_TRACE` | `off` | Trace mode: `off` disabled, `local` writes trace.log only, `otel` reports to OpenTelemetry only, `all` enables both |

## Inference Service Logs

* `fastdeploy.log` : Main log file, records system configuration, startup information, runtime status, and console output (console_logger)
* `request.log` : Request log file, records user request lifecycle and processing details
* `trace.log` : Trace log file, records events and timestamps for each stage of request processing, used for performance analysis (requires `FD_TRACE=local` or `all`)
* `error.log` : Error log file, records all ERROR and above level logs
* `worker_process.log` : Consolidated worker logs including engine inference data, model runner info, GPU worker profiling, and CudaGraph status.
* `cache_manager.log` : Consolidated cache logs including KV Cache allocation, cache hit status, and cache transfer manager info.

## Speculative Decoding Logs
* `speculate.log` : Contains speculative decoding-related information.

## Prefix Caching Logs
* `cache_manager_*.log` : Logs cache transfer manager startup parameters and received request information (one file per GPU).

## PD Disaggregation Logs
* `cache_messager_*.log` : Logs transmission protocols and messages used by the P instance (one file per GPU).
* `splitwise_connector.log` : Records data received from P/D instances and connection establishment details.

## Paddle Logs
* `paddle/workerlog.*` : Paddle distributed launch logs, one file per GPU card.
* `paddle/backup_env.*.json` : Records environment variables set during instance startup. The number of files matches the number of GPU cards.
