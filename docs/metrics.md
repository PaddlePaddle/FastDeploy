# FastDeploy 指标说明




| 指标名称                                    | 类型      | 说明             | 单位 |
| ------------------------------------------- | --------- |----------------| ---- |
| `fastdeploy:num_requests_running`           | Gauge     | 当前正在运行的请求数量    | 个   |
| `fastdeploy:num_requests_waiting`           | Gauge     | 当前等待中的请求数量     | 个   |
| `fastdeploy:time_to_first_token_seconds`    | Histogram | 首 token 生成所需时间 | 秒   |
| `fastdeploy:time_per_output_token_seconds`  | Histogram | 间隔输出 token 的生成时间 | 秒   |
| `fastdeploy:e2e_request_latency_seconds`  | Histogram | 请求的端到端延迟分布 | 秒   |
| `fastdeploy:request_inference_time_seconds`  | Histogram | 请求在RUNNING阶段耗时 | 秒   |
| `fastdeploy:request_queue_time_seconds`  | Histogram | 请求在WAITING阶段耗时 | 秒   |


## 指标访问
- 访问地址：`http://localhost:8000/metrics`
- 指标类型：Prometheus 格式

