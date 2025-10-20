[简体中文](../zh/usage/log.md)

# Log Description

FastDeploy generates the following log files during deployment. Below is an explanation of each log's purpose.
By default, logs are stored in the `log` directory under the execution path. To specify a custom directory, set the environment variable `FD_LOG_DIR`.

## Inference Service Logs
* `backup_env.*.json` : Records environment variables set during instance startup. The number of files matches the number of GPU cards.
* `envlog.*` : Logs environment variables set during instance startup. The number of files matches the number of GPU cards.
* `console.log` : Records model startup time and other information. This log is also printed to the console.
* `data_processor.log` : Logs input/output data encoding and decoding details.
* `fastdeploy.log` : Records configuration information during instance startup, as well as request and response details during runtime.
* `workerlog.*` : Tracks model loading progress and inference operator errors. Each GPU card has a corresponding file.
* `worker_process.log` : Logs engine inference data for each iteration.
* `cache_manager.log` : Records KV Cache logical index allocation for each request and cache hit status.
* `launch_worker.log` : Logs model startup information and error messages.
* `gpu_worker.log` : Records KV Cache block count information during profiling.
* `gpu_model_runner.log` : Contains model details and loading time.

## Online Inference Client Logs
* `api_server.log` : Logs startup parameters and received request information.

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
