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
| Token | `fastdeploy:request_token_ratio`   | Histogram | Token generation rate per Request | count |
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
| KV Cache | `fastdeploy:max_cpu_block_num` | Gauge | Total number of CPU blocks initialized at startup | count |
| KV Cache | `fastdeploy:available_gpu_resource` | Gauge | Ratio of available GPU blocks to total GPU blocks | % |
| KV Cache | `fastdeploy:gpu_cache_usage_perc` | Gauge | GPU KV cache utilization | % |
| KV Cache | `fastdeploy:send_cache_failed_num` | Counter | Total number of cache send failures | count |

## Accessing Metrics

- Access URL: `http://localhost:8000/metrics`
- Metric Type: Prometheus format

## Trace Events

FastDeploy outputs structured trace events to `trace.log` at key stages of request processing, useful for diagnosing per-request latency bottlenecks. Each trace log entry contains fields such as `timestamp` (milliseconds), `request_id`, `event`, and `stage`.

### Common Events (Mixed / All Instances)

| Stage | Event | Description |
| :---: | --- | --- |
| PREPROCESSING | `PREPROCESSING_START` | API Server begins preprocessing the request |
| PREPROCESSING | `PREPROCESSING_END` | Engine receives the request, preprocessing complete |
| SCHEDULE | `REQUEST_SCHEDULE_START` | Request enters the scheduling flow |
| SCHEDULE | `REQUEST_QUEUE_START` | Request enters the scheduling queue |
| SCHEDULE | `REQUEST_QUEUE_END` | Request dequeued from the scheduling queue |
| SCHEDULE | `RESOURCE_ALLOCATE_START` | Resource allocation begins for the request |
| SCHEDULE | `PREPARE_PREFIX_CACHE_START` | Prefix cache block matching begins |
| SCHEDULE | `PREPARE_PREFIX_CACHE_END` | Prefix cache block matching complete |
| SCHEDULE | `RESOURCE_ALLOCATE_END` | Resource allocation complete |
| SCHEDULE | `REQUEST_SCHEDULE_END` | Scheduling flow complete |
| PREFILL | `INFERENCE_START` | Request sent to GPU for inference |
| PREFILL | `FIRST_TOKEN_GENERATED` | First token generated |
| DECODE | `DECODE_START` | Enters Decode phase |
| DECODE | `INFERENCE_END` | Inference complete (all tokens generated) |
| DECODE | `PREEMPTED` | Request preempted |
| DECODE | `RESCHEDULED_INFERENCE_START` | Preempted request rescheduled for execution |
| POSTPROCESSING | `WRITE_CACHE_TO_STORAGE_START` | Begins writing KV Cache to external storage |
| POSTPROCESSING | `WRITE_CACHE_TO_STORAGE_END` | KV Cache written to external storage |
| POSTPROCESSING | `POSTPROCESSING_START` | Post-processing begins |
| POSTPROCESSING | `POSTPROCESSING_END` | Post-processing complete, response sent |

### PD Disaggregation — Prefill (P) Instance Events

| Stage | Event | Description |
| :---: | --- | --- |
| SCHEDULE | `ASK_DECODE_RESOURCE_START` | P begins requesting resources from D (sends ZMQ request) |
| SCHEDULE | `ASK_DECODE_RESOURCE_END` | P receives resource allocation confirmation from D (with dest_block_ids) |
| PREFILL | `PREFILL_INFERENCE_END` | P instance Prefill inference complete |
| POSTPROCESSING | `CHECK_CACHE_TRANSFER_START` | P begins waiting for KV Cache transfer to complete |
| POSTPROCESSING | `CHECK_CACHE_TRANSFER_END` | KV Cache transfer confirmed, ready to send first token to D |

### PD Disaggregation — Decode (D) Instance Events

| Stage | Event | Description |
| :---: | --- | --- |
| DECODE | `DECODE_PROCESS_PREALLOCATE_REQUEST_START` | D begins processing resource allocation request from P |
| DECODE | `DECODE_PROCESS_PREALLOCATE_REQUEST_END` | D completes resource allocation and returns dest_block_ids to P |
| DECODE | `DECODE_PROCESS_PREFILLED_REQUEST_START` | D receives first token from P, begins processing Prefilled request |
| DECODE | `DECODE_PROCESS_PREFILLED_REQUEST_END` | D adds Prefilled request to running queue |
| DECODE | `DECODE_INFERENCE_END` | D instance Decode inference complete |

### Request Lifecycle Sequence

**Mixed mode** (single instance, full inference):
```
PREPROCESSING_START → PREPROCESSING_END → REQUEST_QUEUE_START → REQUEST_QUEUE_END
→ RESOURCE_ALLOCATE_START → RESOURCE_ALLOCATE_END → INFERENCE_START
→ FIRST_TOKEN_GENERATED → DECODE_START → INFERENCE_END
→ POSTPROCESSING_START → POSTPROCESSING_END
```

**PD Disaggregation — Prefill (P) Instance**:
```
PREPROCESSING_START → PREPROCESSING_END → REQUEST_QUEUE_START → REQUEST_QUEUE_END
→ ASK_DECODE_RESOURCE_START → ASK_DECODE_RESOURCE_END
→ RESOURCE_ALLOCATE_START → RESOURCE_ALLOCATE_END
→ INFERENCE_START → PREFILL_INFERENCE_END
→ CHECK_CACHE_TRANSFER_START → CHECK_CACHE_TRANSFER_END → [send first token to D]
```

**PD Disaggregation — Decode (D) Instance**:
```
PREPROCESSING_START → PREPROCESSING_END → REQUEST_QUEUE_START → REQUEST_QUEUE_END
→ DECODE_PROCESS_PREALLOCATE_REQUEST_START → DECODE_PROCESS_PREALLOCATE_REQUEST_END
→ [wait for P to complete prefill and transfer KV Cache]
→ DECODE_PROCESS_PREFILLED_REQUEST_START → DECODE_PROCESS_PREFILLED_REQUEST_END
→ INFERENCE_START → DECODE_INFERENCE_END
→ POSTPROCESSING_START → POSTPROCESSING_END
```
