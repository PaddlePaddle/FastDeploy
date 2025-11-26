# FastDeploy Tracing with OpenTelemetry

**FastDeploy** exports request tracing data through the **OpenTelemetry Collector**.
Tracing can be enabled when starting the server using the `--trace-enable` flag, and the OpenTelemetry Collector endpoint can be configured via `--otlp-traces-endpoint`.

---

## Setup Guide

### 1. Install Dependencies

```bash
# Manual installation
pip install opentelemetry-sdk \
            opentelemetry-api \
            opentelemetry-exporter-otlp \
            opentelemetry-exporter-otlp-proto-grpc
```

---

### 2. Start OpenTelemetry Collector and Jaeger

```bash
docker compose -f examples/observability/tracing/tracing_compose.yaml up -d
```

---

### 3. Start FastDeploy Server with Tracing Enabled

#### Configure FastDeploy Environment Variables

```shell
# Enable tracing
"TRACES_ENABLE": "true",

# Service name
"FD_SERVICE_NAME": "FastDeploy",

# Instance name
"FD_HOST_NAME": "trace_test",

# Exporter type
"TRACES_EXPORTER": "otlp",

# OTLP endpoint:
#   gRPC: 4317
#   HTTP: 4318
"EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",

# Optional headers
"EXPORTER_OTLP_HEADERS": "Authentication=Txxxxx",

# Export protocol
"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "grpc",
```

#### Start FastDeploy

Start the FastDeploy server with the above configuration and ensure that tracing is enabled.

---

### 4. Send Requests and View Traces

* Open the **Jaeger UI** in your browser (port `16686`) to visualize request traces.
* The OpenTelemetry Collector will also export the trace data to a local file:

```plain
/tmp/otel_trace.json
```

---

## Adding Tracing to Your Own Code

FastDeploy already inserts tracing points at most critical execution stages.
Developers can use the APIs provided in `trace.py` to add more fine-grained tracing.

---

### 4.1 Initialize Tracing

Each **process** involved in tracing must call:

```python
process_tracing_init()
```

Each **thread** that participates in a traced request must call:

```python
trace_set_thread_info("thread_label", tp_rank, dp_rank)
```

* `thread_label`: identifier used for visual distinction of threads.
* `tp_rank` / `dp_rank`: optional values to label tensor parallelism or data parallelism ranks.

---

### 4.2 Mark Request Start and Finish

```python
trace_req_start(rid, bootstrap_room, ts, role)
trace_req_finish(rid, ts, attrs)
```

* Creates both a **Bootstrap Room Span** and a **Root Span**.
* Supports inheritance from spans created by the **FastAPI Instrumentor** (context copying).
* `attrs` can be used to attach additional attributes to the request span.

---

### 4.3 Add Tracing for Slices

#### Standard Slice

```python
trace_slice_start("slice_name", rid)
trace_slice_end("slice_name", rid)
```

#### Mark Thread Completion

The last slice in a thread can mark the thread span as finished:

```python
trace_slice_end("slice_name", rid, thread_finish_flag=True)
```

---

### 4.4 Trace Context Propagation Across Threads

#### Sender Side (ZMQ)

```python
trace_context = trace_get_proc_propagate_context(rid)
req.trace_context = trace_context
```

#### Receiver Side (ZMQ)

```python
trace_set_proc_propagate_context(rid, req.trace_context)
```

---

### 4.5 Trace Context Propagation Across Nodes

#### Sender Side (HTTP)

```python
trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
headers = {"trace_context": trace_context}
session.post(url, headers=headers)
```

#### Receiver Side (HTTP)

```python
trace_set_remote_propagate_context(request.headers['trace_context'])
```

---

### 4.6 Add Events and Attributes

#### Events (recorded on the current slice)

```python
trace_event("event_name", rid, ts, attrs)
```

#### Attributes (attached to the current slice)

```python
trace_slice_add_attr(rid, attrs)
```

---

## Extending the Tracing Framework

### 5.1 Trace Context Hierarchy

* Two levels of Trace Context:

  * **`TraceReqContext`** – request-level context
  * **`TraceThreadContext`** – thread-level context

* Four-level Span hierarchy:

  * `bootstrap_room_span`
  * `req_root_span`
  * `thread_span`
  * `slice_span`

---

### 5.2 Span Hierarchy Example

```plain
Span: POST /v1/chat/completions              # API entry point
│
├── Span: API Server(host: | pid:84037)
│   ├── Span: preprocess                     # preprocessing stage
│   │   └── Span: encode                     # encoding stage
│   │
│   └── Span: Insert Task to Scheduler(host: | pid:83851)
│       ├── Span: submit_to_infer_engine     # request → engine communication
│       └── Span: insert_task_to_scheduler   # task scheduling
│
├── Span: Scheduler Task to Work(host: | pid:82760)
│   ├── Span: schedule_queue_wait            # queue waiting latency
│   └── Span: schedule_dispatch              # GPU/memory resource allocation
│
├── Span: Token Processor(host: | pid:82764)
│   ├── Span: prefill                        # prefill phase
│   └── Span: decode_loop                   # decode phase
```

---

### 5.3 Available Span Name Enum (`TraceSpanName`)

```python
FASTDEPLOY
PREPROCESS
ENCODE
SUBMIT_TO_INFER_ENGINE
SCHEDULE
SCHEDULE_QUEUE_WAIT
SCHEDULE_DISPATCH
SCHEDULER_ALLOCATE_RESOURCE
PREFILL
DECODE_LOOP
```

* These enums can be used when creating slices to ensure consistent naming.

---

### 5.4 Important Notes

1. Each **thread span must be closed** when the final slice of that thread finishes.
2. Spans created by **FastAPI Instrumentor** are automatically inherited by the internal tracing context.
