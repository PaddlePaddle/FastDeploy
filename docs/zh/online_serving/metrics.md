[English](../../online_serving/metrics.md)

# 监控 Metrics

在 FastDeploy 启动后，支持通过 Metrics 持续监控的 FastDeploy 的服务状态。启动 FastDeploy 时，可以通过配置 `metrics-port` 参数指定 Metircs 服务的端口。

| 指标分类 | 指标名称                                  | 数据类型      | 说明                         | 单位 |
| :---: | ----------------------------------------- | --------- |------------------------------|------|
| 请求| `fastdeploy:requests_number`              | Counter   | 已接收的请求总数             | 个   |
| 请求 | `fastdeploy:request_success_total`        | Counter   | 成功处理的请求个数           | 个   |
| 请求 | `fastdeploy:num_requests_running`         | Gauge     | 当前正在运行的请求数量       | 个   |
| 请求 | `fastdeploy:num_requests_waiting`         | Gauge     | 当前等待中的请求数量         | 个   |
| 时延 | `fastdeploy:time_to_first_token_seconds`  | Histogram | 首 token 生成所需时间        | 秒   |
| 时延 | `fastdeploy:time_per_output_token_seconds`| Histogram | 间隔输出 token 的生成时间    | 秒   |
| 时延 | `fastdeploy:e2e_request_latency_seconds`  | Histogram | 请求的端到端延迟分布         | 秒   |
| 时延 | `fastdeploy:request_inference_time_seconds`| Histogram | 请求在 RUNNING 阶段耗时      | 秒   |
| 时延 | `fastdeploy:request_queue_time_seconds`   | Histogram | 请求在 WAITING 阶段耗时      | 秒   |
| 时延 | `fastdeploy:request_prefill_time_seconds` | Histogram | 请求的 Prefill 阶段耗时      | 秒   |
| 时延 | `fastdeploy:request_decode_time_seconds`  | Histogram | 请求的 Decode 阶段耗时       | 秒   |
| Token | `fastdeploy:prompt_tokens_total`          | Counter   | 已处理的 prompt token 总数   | 个   |
| Token | `fastdeploy:generation_tokens_total`      | Counter   | 已生成的 token 总数          | 个   |
| Token | `fastdeploy:request_prompt_tokens`        | Histogram | 每个请求的 prompt token 数量 | 个   |
| Token | `fastdeploy:request_token_ratio`          | Histogram | 每个请求的token生成速率       | 个   |
| Token | `fastdeploy:request_generation_tokens`    | Histogram | 每个请求的 generation token 数量    | 个   |
| Token | `fastdeploy:request_params_max_tokens`    | Histogram | 请求的 max_tokens 分布       | 个   |
| 批处理 | `fastdeploy:available_batch_size`         | Gauge     | Decode 阶段还可以插入的请求数量 | 个   |
| 批处理 | `fastdeploy:batch_size`                   | Gauge     | 推理时的真实批处理大小        | 个   |
| 批处理 | `fastdeploy:max_batch_size`               | Gauge     | 服务启动时确定的最大批处理大小  | 个   |
| KV缓存 | `fastdeploy:cache_config_info`            | Gauge     | 推理引擎的缓存配置信息        | 个   |
| KV缓存 | `fastdeploy:hit_req_rate`                 | Gauge     | 请求级别前缀缓存命中率        | 百分比   |
| KV缓存 | `fastdeploy:hit_token_rate`               | Gauge     | token 级别前缀缓存命中率      | 百分比   |
| KV缓存 | `fastdeploy:cpu_hit_token_rate`           | Gauge     | token 级别 CPU 前缀缓存命中率   | 百分比   |
| KV缓存 | `fastdeploy:gpu_hit_token_rate`           | Gauge     | token 级别 GPU 前缀缓存命中率   | 百分比   |
| KV缓存 | `fastdeploy:prefix_cache_token_num`       | Counter   | 前缀缓存token总数           | 个   |
| KV缓存 | `fastdeploy:prefix_gpu_cache_token_num`   | Counter   | 位于 GPU 上的前缀缓存 token 总数  | 个   |
| KV缓存 | `fastdeploy:prefix_cpu_cache_token_num`   | Counter   | 位于 CPU 上的前缀缓存 token 总数  | 个   |
| KV缓存 | `fastdeploy:available_gpu_block_num`      | Gauge     | 缓存中可用的 GPU 块数量（包含尚未正式释放的前缀缓存块）| 个   |
| KV缓存 | `fastdeploy:free_gpu_block_num`           | Gauge     | 缓存中的可用块数             | 个   |
| KV缓存 | `fastdeploy:max_gpu_block_num`            | Gauge     | 服务启动时确定的 GPU 总块数        | 个   |
| KV缓存 | `fastdeploy:max_cpu_block_num`            | Gauge     | 服务启动时确定的 CPU 总块数        | 个   |
| KV缓存 | `fastdeploy:available_gpu_resource`       | Gauge     | 可用块占比，即可用 GPU 块数量 / 最大GPU块数量| 百分比   |
| KV缓存 | `fastdeploy:gpu_cache_usage_perc`         | Gauge     | GPU 上的 KV 缓存使用率          | 百分比    |
| KV缓存 | `fastdeploy:send_cache_failed_num`        | Counter   | 发送缓存失败的总次数          | 个   |

## 指标访问

- 访问地址：`http://localhost:8000/metrics`
- 指标类型：Prometheus 格式

## Trace 事件

FastDeploy 在请求处理的关键阶段输出结构化 trace 事件到 `trace.log`，用于定位请求级别的延迟瓶颈。每条 trace 日志包含 `timestamp`（毫秒）、`request_id`、`event`、`stage` 等字段。

### 通用事件（Mixed / 所有实例）

| 阶段 | 事件 | 说明 |
| :---: | --- | --- |
| PREPROCESSING | `PREPROCESSING_START` | API Server 开始预处理请求 |
| PREPROCESSING | `PREPROCESSING_END` | Engine 收到请求，预处理完成 |
| SCHEDULE | `REQUEST_SCHEDULE_START` | 请求进入调度流程 |
| SCHEDULE | `REQUEST_QUEUE_START` | 请求进入调度队列等待 |
| SCHEDULE | `REQUEST_QUEUE_END` | 请求从调度队列取出 |
| SCHEDULE | `RESOURCE_ALLOCATE_START` | 开始为请求分配资源 |
| SCHEDULE | `PREPARE_PREFIX_CACHE_START` | 开始匹配前缀缓存块 |
| SCHEDULE | `PREPARE_PREFIX_CACHE_END` | 前缀缓存块匹配完成 |
| SCHEDULE | `RESOURCE_ALLOCATE_END` | 资源分配完成 |
| SCHEDULE | `REQUEST_SCHEDULE_END` | 调度流程结束 |
| PREFILL | `INFERENCE_START` | 请求送入 GPU 执行推理 |
| PREFILL | `FIRST_TOKEN_GENERATED` | 首 token 生成 |
| DECODE | `DECODE_START` | 进入 Decode 阶段 |
| DECODE | `INFERENCE_END` | 推理完成（所有 token 生成完毕） |
| DECODE | `PREEMPTED` | 请求被抢占 |
| DECODE | `RESCHEDULED_INFERENCE_START` | 被抢占的请求重新调度执行 |
| POSTPROCESSING | `WRITE_CACHE_TO_STORAGE_START` | 开始将 KV Cache 写入外部存储 |
| POSTPROCESSING | `WRITE_CACHE_TO_STORAGE_END` | KV Cache 写入外部存储完成 |
| POSTPROCESSING | `POSTPROCESSING_START` | 开始后处理 |
| POSTPROCESSING | `POSTPROCESSING_END` | 后处理完成，响应发送完毕 |

### PD 分离 — Prefill (P) 实例专属事件

| 阶段 | 事件 | 说明 |
| :---: | --- | --- |
| SCHEDULE | `ASK_DECODE_RESOURCE_START` | P 开始向 D 申请资源（发送 ZMQ 请求） |
| SCHEDULE | `ASK_DECODE_RESOURCE_END` | P 收到 D 的资源分配确认（含 dest_block_ids） |
| PREFILL | `PREFILL_INFERENCE_END` | P 实例 Prefill 推理完成 |
| POSTPROCESSING | `CHECK_CACHE_TRANSFER_START` | P 开始等待 KV Cache 传输完成 |
| POSTPROCESSING | `CHECK_CACHE_TRANSFER_END` | KV Cache 传输完成确认，准备发送 first token 到 D |

### PD 分离 — Decode (D) 实例专属事件

| 阶段 | 事件 | 说明 |
| :---: | --- | --- |
| DECODE | `DECODE_PROCESS_PREALLOCATE_REQUEST_START` | D 开始处理 P 发来的资源分配请求 |
| DECODE | `DECODE_PROCESS_PREALLOCATE_REQUEST_END` | D 完成资源分配并返回 dest_block_ids 给 P |
| DECODE | `DECODE_PROCESS_PREFILLED_REQUEST_START` | D 收到 P 的 first token，开始处理 Prefilled 请求 |
| DECODE | `DECODE_PROCESS_PREFILLED_REQUEST_END` | D 将 Prefilled 请求加入 running queue |
| DECODE | `DECODE_INFERENCE_END` | D 实例 Decode 推理完成 |

### 请求生命周期时序图

**Mixed 模式**（单实例完整推理）：
```
PREPROCESSING_START → PREPROCESSING_END → REQUEST_QUEUE_START → REQUEST_QUEUE_END
→ RESOURCE_ALLOCATE_START → RESOURCE_ALLOCATE_END → INFERENCE_START
→ FIRST_TOKEN_GENERATED → DECODE_START → INFERENCE_END
→ POSTPROCESSING_START → POSTPROCESSING_END
```

**PD 分离 — Prefill (P) 实例**：
```
PREPROCESSING_START → PREPROCESSING_END → REQUEST_QUEUE_START → REQUEST_QUEUE_END
→ ASK_DECODE_RESOURCE_START → ASK_DECODE_RESOURCE_END
→ RESOURCE_ALLOCATE_START → RESOURCE_ALLOCATE_END
→ INFERENCE_START → PREFILL_INFERENCE_END
→ CHECK_CACHE_TRANSFER_START → CHECK_CACHE_TRANSFER_END → [发送 first token 到 D]
```

**PD 分离 — Decode (D) 实例**：
```
PREPROCESSING_START → PREPROCESSING_END → REQUEST_QUEUE_START → REQUEST_QUEUE_END
→ DECODE_PROCESS_PREALLOCATE_REQUEST_START → DECODE_PROCESS_PREALLOCATE_REQUEST_END
→ [等待 P 完成 prefill 并传输 KV Cache]
→ DECODE_PROCESS_PREFILLED_REQUEST_START → DECODE_PROCESS_PREFILLED_REQUEST_END
→ INFERENCE_START → DECODE_INFERENCE_END
→ POSTPROCESSING_START → POSTPROCESSING_END
```
