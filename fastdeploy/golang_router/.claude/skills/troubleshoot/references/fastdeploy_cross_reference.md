# FastDeploy 后端交叉引用

从 Router 日志推断 FastDeploy 后端问题时的排查指引。

---

## 症状 → 后端排查

### 1. 后端不可达 (502)

**Router 日志特征**：
```
[ERROR] Failed to connect to backend service: dial tcp {ip}:{port}: connect: connection refused
```

**排查步骤**：
1. `curl http://{worker_url}/health` — 确认后端是否存活
2. `curl http://{worker_url}/v1/models` — 确认模型是否加载完成
3. 检查后端日志 `logs/workerlog.0`
4. `netstat -tlnp | grep {port}` — 确认端口监听
5. 检查网络连通性（防火墙、安全组）

### 2. 后端 OOM / 频繁重启

**Router 日志特征**：
- Worker 频繁 REMOVED → RE-REGISTERED（短周期内多次）
- 健康检查间歇性失败

**排查步骤**：
1. `dmesg | grep -i oom` — 检查 OOM killer
2. `nvidia-smi` — 检查 GPU 内存
3. 后端日志搜索 `CUDA out of memory`
4. 检查 `max_num_seqs`、`max_model_len` 配置

### 3. 高推理延迟

**Router 日志特征**：
- 请求 p99 高（>10s）但调度耗时仅 ms 级
- 确认延迟不在 Router 层（调度耗时 << 总延迟）

**排查步骤**：
1. 检查后端 Prometheus metrics：`http://{worker_url}:{metrics_port}/metrics`
   - `fastdeploy_llm_running_queue_size` — 推理队列
   - `fastdeploy_llm_waiting_queue_size` — 等待队列
   - `fastdeploy_llm_generation_tokens_per_second` — 吞吐量
2. 确认 GPU 利用率：`nvidia-smi --query-gpu=utilization.gpu --format=csv`
3. 检查是否有长 prompt 请求拖慢整体

### 4. 流式响应异常

**Router 日志特征**：
```
[ERROR] scanner error: {err}  (非 context canceled)
[ERROR] copy error: {err}  (非 context canceled)
```

**排查步骤**：
1. 后端日志搜索对应 request_id
2. 检查后端是否产生格式错误的 SSE
3. 检查网络是否有中间代理超时切断

### 5. 请求超时/卡住

**Router 日志特征**：
- 有 select worker 但长时间无 release/completed
- [stats] 中 running 持续不降

**根因**：Router 的 `http.Client{}` 没有设置超时，后端不响应则阻塞到客户端断连或 TCP 超时。

**排查步骤**：
1. 检查后端是否还在处理请求
2. 检查后端是否出现死锁
3. `ss -tnp | grep {port}` — 检查 TCP 连接状态

---

## 通用 FastDeploy 排查工具

### collect-env

收集环境信息：
```bash
python -m fastdeploy.utils.collect_env
```

### 后端日志位置

- 默认：`logs/workerlog.0`
- 多 Worker：`logs/workerlog.{N}`

### Prometheus Metrics

后端 metrics 端口（从注册信息获取 `metrics_port`）：
```
http://{worker_ip}:{metrics_port}/metrics
```

关键指标：
- `fastdeploy_llm_running_queue_size` — 当前推理中的请求数
- `fastdeploy_llm_waiting_queue_size` — 等待队列长度
- `fastdeploy_llm_generation_tokens_per_second` — 生成吞吐
- `fastdeploy_llm_request_total` — 总请求数
