[简体中文](../zh/online_serving/metrics.md)

# Monitoring Metrics

After FastDeploy is launched, it supports continuous monitoring of the FastDeploy service status through Metrics. When starting FastDeploy, you can specify the port for the Metrics service by configuring the `metrics-port` parameter.

| Category | Metric Name | Type | Description | Unit |
| :---: | ----------------------------------------- | --------- | ------------------------------ | ------ |
| Request | `fastdeploy:requests_number` | Counter | Total number of received requests | count |
| Request | `fastdeploy:request_success_total` | Counter | Number of successfully processed requests | count |
| Request | `fastdeploy:num_requests_running` | Gauge | Number of requests currently running | count |
| Request | `fastdeploy:num_requests_waiting` | Gauge | Number of requests currently waiting | count |
| Latency | `fastdeploy:time_to_first_token_seconds` | Histogram | Time to generate the first token (TTFT) | s |
| Latency | `fastdeploy:time_per_output_token_seconds` | Histogram | Time interval between generated tokens (TPOT) | s |
| Latency | `fastdeploy:e2e_request_latency_seconds` | Histogram | End-to-end request latency distribution | s |
| Latency | `fastdeploy:request_inference_time_seconds` | Histogram | Time spent in the RUNNING phase | s |
| Latency | `fastdeploy:request_queue_time_seconds` | Histogram | Time spent in the WAITING phase | s |
| Latency | `fastdeploy:request_prefill_time_seconds` | Histogram | Time spent in the Prefill phase | s |
| Latency | `fastdeploy:request_decode_time_seconds` | Histogram | Time spent in the Decode phase | s |
| Token | `fastdeploy:prompt_tokens_total` | Counter | Total number of processed prompt tokens | count |
| Token | `fastdeploy:generation_tokens_total` | Counter | Total number of generated tokens | count |
| Token | `fastdeploy:request_prompt_tokens` | Histogram | Prompt token count per request | count |
| Token | `fastdeploy:request_generation_tokens` | Histogram | Generation token count per request | count |
| Token | `fastdeploy:request_params_max_tokens` | Histogram | Distribution of `max_tokens` per request | count |
| Batch | `fastdeploy:available_batch_size` | Gauge | Number of additional requests that can be inserted during Decode | count |
| Batch | `fastdeploy:batch_size` | Gauge | Actual batch size during inference | count |
| Batch | `fastdeploy:max_batch_size` | Gauge | Maximum batch size configured at service startup | count |
| KV Cache | `fastdeploy:cache_config_info` | Gauge | Cache configuration info of the inference engine | count |
| KV Cache | `fastdeploy:hit_req_rate` | Gauge | Prefix cache hit rate at the request level | % |
| KV Cache | `fastdeploy:hit_token_rate` | Gauge | Prefix cache hit rate at the token level | % |
| KV Cache | `fastdeploy:cpu_hit_token_rate` | Gauge | CPU-side token-level prefix cache hit rate | % |
| KV Cache | `fastdeploy:gpu_hit_token_rate` | Gauge | GPU-side token-level prefix cache hit rate | % |
| KV Cache | `fastdeploy:prefix_cache_token_num` | Counter | Total number of tokens in prefix cache | count |
| KV Cache | `fastdeploy:prefix_gpu_cache_token_num` | Counter | Total number of prefix cache tokens on GPU | count |
| KV Cache | `fastdeploy:prefix_cpu_cache_token_num` | Counter | Total number of prefix cache tokens on CPU | count |
| KV Cache | `fastdeploy:available_gpu_block_num` | Gauge | Available GPU blocks in cache (including unreleased prefix blocks) | count |
| KV Cache | `fastdeploy:free_gpu_block_num` | Gauge | Number of free GPU blocks in cache | count |
| KV Cache | `fastdeploy:max_gpu_block_num` | Gauge | Total number of GPU blocks initialized at startup | count |
| KV Cache | `fastdeploy:available_gpu_resource` | Gauge | Ratio of available GPU blocks to total GPU blocks | % |
| KV Cache | `fastdeploy:gpu_cache_usage_perc` | Gauge | GPU KV cache utilization | % |
| KV Cache | `fastdeploy:send_cache_failed_num` | Counter | Total number of cache send failures | count |

## Accessing Metrics

- Access URL: `http://localhost:8000/metrics`
- Metric Type: Prometheus format
