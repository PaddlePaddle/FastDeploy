**FastDeploy** 基于**OpenTelemetry Collector** 导出请求追踪数据。
可通过在启动服务器时添加 `--trace-enable` 来开启追踪，并使用 `--otlp-traces-endpoint` 配置 OpenTelemetry Collector 接收端点。

## 配置指南（Setup Guide）

### 1. 安装依赖和工具

```bash
# 手动安装
pip install opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc
```

### 2. 启动 OpenTelemetry Collector 和 Jaeger

```bash
docker compose -f examples/observability/tracing/tracing_compose.yaml up -d
```

### 3. 启动带追踪功能的 FastDeploy 服务器

- FastDeploy设置环境变量

```shell
# 开启Trace
"TRACES_ENABLE": "true",
# 服务名称
"FD_SERVICE_NAME": "FastDeploy",
# 实例名称
"FD_HOST_NAME": "trace_test",
"TRACES_EXPORTER": "otlp",
# grpc方式导出端口为4317， http方式导出端口为4318
"EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
"EXPORTER_OTLP_HEADERS": "Authentication=Txxxxx",
# 导出方式
"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "grpc",
```

- 启动FastDeploy

### 4. 发送请求并观察追踪数据

- 在浏览器访问 Jaeger UI（端口 `16686`）可视化请求追踪。

- Collector 同时会将追踪数据导出为 `/tmp/otel_trace.json`。

## 如何为自己的代码添加追踪

FastDeploy 已在主要节点插入了追踪点。开发者可使用 `trace.py` 提供的 API 进行更精细的追踪。

### 4.1 初始化追踪

每个涉及追踪的**进程**执行：

```python
process_tracing_init()
```

请求涉及到的每个**线程**执行：

```python
trace_set_thread_info("thread_label", tp_rank, dp_rank)
```

- `thread_label` 用于线程区分，可视化显示
- `tp_rank`/`dp_rank` 可选，标记张量并行或数据并行 rank

### 4.2 标记请求开始和结束

```python
trace_req_start(rid, bootstrap_room, ts, role)
trace_req_finish(rid, ts, attrs)
```

- 会创建 Bootstrap Room Span与 Root Span
- 支持 FastAPI Instrumentor 已创建 Span 的继承（context copy）
- `attrs` 可添加额外属性

### 4.3 为 Slice 添加追踪

普通 Slice：

```python
trace_slice_start("slice_name", rid)
trace_slice_end("slice_name", rid)
```

- 最后一个 Slice 可标记线程结束：

```python
trace_slice_end("slice_name", rid, thread_finish_flag=True)
```

### 4.4 请求跨线程 Trace Context 传播

发送端（ZMQ）：

```python
trace_context = trace_get_proc_propagate_context(rid)
req.trace_context = trace_context
```

接收端（ZMQ）：

```python
trace_set_proc_propagate_context(rid, req.trace_context)
```

### 4.5 添加事件和属性

事件（记录到当前 Slice）：

```python
trace_event("event_name", rid, ts, attrs)
```

属性（添加到当前 Slice）：

```python
trace_slice_add_attr(rid, attrs)
```

## 扩展追踪框架

### 5.1 Trace Context 层级

- 两级 Trace Context：
  - `TraceReqContext` → 请求级上下文
  - `TraceThreadContext` → 线程级上下文
- 三级 Span 结构：
  - `req_root_span`
  - `thread_span`
  - `slice_span`

### 5.2 可用的 Span 名枚举（`TraceSpanName`）

```python
FASTDEPLOY
PREPROCESS
SCHEDULE
PREFILL
DECODE
POSTPROCESS
```

- 在创建 slice 时可使用枚举，保证命名规范化

### 5.3 注意事项

1. 每个线程 Span 必须在最后一个 Slice 结束时关闭。
2. FastAPI Instrumentor 已创建的 Span 会被继承到内部追踪上下文。
