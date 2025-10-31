[English](../../online_serving/metrics.md)

# 监控 Metrics

在 FastDeploy 启动后，支持通过 Metrics 持续监控的 FastDeploy 的服务状态。启动 FastDeploy 时，可以通过配置 `metrics-port` 参数指定 Metircs 服务的端口。

| 指标名称                                  | 类型      | 说明                         | 单位 |
| ----------------------------------------- | --------- |------------------------------|------|
| `fastdeploy:num_requests_running`         | Gauge     | 当前正在运行的请求数量       | 个   |
| `fastdeploy:num_requests_waiting`         | Gauge     | 当前等待中的请求数量         | 个   |
| `fastdeploy:time_to_first_token_seconds`  | Histogram | 首 token 生成所需时间        | 秒   |
| `fastdeploy:time_per_output_token_seconds`| Histogram | 间隔输出 token 的生成时间    | 秒   |
| `fastdeploy:e2e_request_latency_seconds`  | Histogram | 请求的端到端延迟分布         | 秒   |
| `fastdeploy:request_inference_time_seconds`| Histogram | 请求在 RUNNING 阶段耗时      | 秒   |
| `fastdeploy:request_queue_time_seconds`   | Histogram | 请求在 WAITING 阶段耗时      | 秒   |
| `fastdeploy:request_prefill_time_seconds` | Histogram | 请求的 prefill 阶段耗时      | 秒   |
| `fastdeploy:request_decode_time_seconds`  | Histogram | 请求的 decode 阶段耗时       | 秒   |
| `fastdeploy:prompt_tokens_total`          | Counter   | 已处理的 prompt token 总数   | 个   |
| `fastdeploy:generation_tokens_total`      | Counter   | 已生成的 token 总数          | 个   |
| `fastdeploy:request_prompt_tokens`        | Histogram | 每个请求的 prompt token 数量 | 个   |
| `fastdeploy:request_generation_tokens`    | Histogram | 每个请求生成的 token 数量    | 个   |
| `fastdeploy:gpu_cache_usage_perc`         | Gauge     | GPU KV-cache 使用率          | 百分比    |
| `fastdeploy:request_params_max_tokens`    | Histogram | 请求的 max_tokens 分布       | 个   |
| `fastdeploy:request_success_total`        | Counter   | 成功处理的请求个数           | 个   |
| `fastdeploy:cache_config_info`            | Gauge     | 推理引擎的缓存配置信息        | 个   |
| `fastdeploy:available_batch_size`         | Gauge     | Decode阶段还可以插入的请求数量 | 个   |
| `fastdeploy:hit_req_rate`                 | Gauge     | 请求级别前缀缓存命中率        | 百分比   |
| `fastdeploy:hit_token_rate`               | Gauge     | token级别前缀缓存命中率      | 百分比   |
| `fastdeploy:cpu_hit_token_rate`           | Gauge     | token级别CPU前缀缓存命中率   | 百分比   |
| `fastdeploy:gpu_hit_token_rate`           | Gauge     | token级别GPU前缀缓存命中率   | 百分比   |
| `fastdeploy:prefix_cache_token_num`       | Counter   | 前缀缓存token总数           | 个   |
| `fastdeploy:prefix_gpu_cache_token_num`   | Counter   | 位于GPU上的前缀缓存token总数  | 个   |
| `fastdeploy:prefix_cpu_cache_token_num`   | Counter   | 位于GPU上的前缀缓存token总数  | 个   |
| `fastdeploy:batch_size`                   | Gauge     | 推理时的真实批处理大小        | 个   |
| `fastdeploy:max_batch_size`               | Gauge     | 服务启动时确定的最大批处理大小  | 个   |
| `fastdeploy:available_gpu_block_num`      | Gauge     | 缓存中可用的GPU块数量（包含尚未正式释放的前缀缓存块）| 个   |
| `fastdeploy:free_gpu_block_num`           | Gauge     | 缓存中的可用块数             | 个   |
| `fastdeploy:max_gpu_block_num`            | Gauge     | 服务启动时确定的总块数        | 个   |
| `fastdeploy:available_gpu_resource`       | Gauge     | 可用块占比，即可用GPU块数量 / 最大GPU块数量| 个   |
| `fastdeploy:requests_number`              | Counter   | 已接收的请求总数             | 个   |
| `fastdeploy:send_cache_failed_num`        | Counter   | 发送缓存失败的总次数          | 个   |
| `fastdeploy:first_token_latency`          | Gauge     | 最近一次生成首token耗时       | 秒   |
| `fastdeploy:infer_latency`                | Gauge     | 最近一次生成单个token的耗时   | 秒   |
## 指标访问

- 访问地址：`http://localhost:8000/metrics`
- 指标类型：Prometheus 格式
