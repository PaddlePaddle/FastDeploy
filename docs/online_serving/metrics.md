[简体中文](../zh/online_serving/metrics.md)

# Monitoring Metrics

After FastDeploy is launched, it supports continuous monitoring of the FastDeploy service status through Metrics. When starting FastDeploy, you can specify the port for the Metrics service by configuring the `metrics-port` parameter.

| Metric Name                                  | Type      | Description                         | Unit |
| --------------------------------------------- | --------- |-------------------------------------|------|
| `fastdeploy:num_requests_running`            | Gauge     | Number of currently running requests       | Count   |
| `fastdeploy:num_requests_waiting`            | Gauge     | Number of currently waiting requests         | Count   |
| `fastdeploy:time_to_first_token_seconds`     | Histogram | Time required to generate the first token        | Seconds   |
| `fastdeploy:time_per_output_token_seconds`   | Histogram | Generation time for interval output tokens    | Seconds   |
| `fastdeploy:e2e_request_latency_seconds`     | Histogram | Distribution of end-to-end request latency         | Seconds   |
| `fastdeploy:request_inference_time_seconds`  | Histogram | Time consumed by requests in the RUNNING phase      | Seconds   |
| `fastdeploy:request_queue_time_seconds`      | Histogram | Time consumed by requests in the WAITING phase      | Seconds   |
| `fastdeploy:request_prefill_time_seconds`    | Histogram | Time consumed by requests in the prefill phase      | Seconds   |
| `fastdeploy:request_decode_time_seconds`     | Histogram | Time consumed by requests in the decode phase       | Seconds   |
| `fastdeploy:prompt_tokens_total`             | Counter   | Total number of processed prompt tokens   | Count   |
| `fastdeploy:generation_tokens_total`         | Counter   | Total number of generated tokens          | Count   |
| `fastdeploy:request_prompt_tokens`           | Histogram | Number of prompt tokens per request | Count   |
| `fastdeploy:request_generation_tokens`       | Histogram | Number of tokens generated per request    | Count   |
| `fastdeploy:gpu_cache_usage_perc`            | Gauge     | GPU KV-cache usage rate          | Percentage    |
| `fastdeploy:request_params_max_tokens`       | Histogram | Distribution of max_tokens for requests       | Count   |
| `fastdeploy:request_success_total`           | Counter   | Number of successfully processed requests           | Count   |
| `fastdeploy:cache_config_info`               | Gauge     | Information of the engine's CacheConfig             | Count   |
| `fastdeploy:available_batch_size`            | Gauge     | Number of requests that can still be inserted during the Decode phase| Count   |
| `fastdeploy:hit_req_rate`                    | Gauge     | Request-level prefix cache hit rate                 | Percentage   |
| `fastdeploy:hit_token_rate`                  | Gauge     | Token-level prefix cache hit rate                   | Percentage   |
| `fastdeploy:cpu_hit_token_rate`              | Gauge     | Token-level CPU prefix cache hit rate               | Percentage   |
| `fastdeploy:gpu_hit_token_rate`              | Gauge     | Token-level GPU prefix cache hit rate               | Percentage   |
| `fastdeploy:prefix_cache_token_num`          | Counter   | Total number of cached tokens                       | Count   |
| `fastdeploy:prefix_gpu_cache_token_num`      | Counter   | Total number of cached tokens on GPU                | Count   |
| `fastdeploy:prefix_cpu_cache_token_num`      | Counter   | Total number of cached tokens on CPU                | Count   |
| `fastdeploy:batch_size`                      | Gauge     | Real batch size during inference                    | Count   |
| `fastdeploy:max_batch_size`                  | Gauge     | Maximum batch size determined when service started  | Count   |
| `fastdeploy:available_gpu_block_num`         | Gauge     | Number of available gpu blocks in cache, including prefix caching blocks that are not officially released               | Count   |
| `fastdeploy:free_gpu_block_num`              | Gauge     | Number of free blocks in cache                      | Count   |
| `fastdeploy:max_gpu_block_num`               | Gauge     | Number of total blocks determined when service started| Count   |
| `fastdeploy:available_gpu_resource`          | Gauge     | Available blocks percentage, i.e. available_gpu_block_num / max_gpu_block_num               | Count   |
| `fastdeploy:requests_number`                 | Counter   | Total number of requests received                   | Count   |
| `fastdeploy:send_cache_failed_num`           | Counter   | Total number of failures of sending cache           | Count   |
| `fastdeploy:first_token_latency`             | Gauge     | Latest time to generate first token in seconds      | Seconds   |
| `fastdeploy:infer_latency`                   | Gauge     | Latest time to generate one token in seconds        | Seconds   |
## Accessing Metrics

- Access URL: `http://localhost:8000/metrics`
- Metric Type: Prometheus format
