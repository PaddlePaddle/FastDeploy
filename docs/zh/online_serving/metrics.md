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

## 指标访问

- 访问地址：`http://localhost:8000/metrics`
- 指标类型：Prometheus 格式
