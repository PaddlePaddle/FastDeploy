# Router 错误目录

按 HTTP 状态码和日志级别分类的 Router 错误快速索引。每条含严重程度、根因、影响、排查命令、问题来源层。

---

## 按 HTTP 状态码索引

注意：HTTP 响应体中的错误消息与 logger 输出的 ERROR 消息**可能不同**。
例如：HTTP 502 响应 `Failed to select worker: {err}` 对应的日志 ERROR 是 `Failed to select mixed worker: {err}`。
分析时需将两者关联而非简单去重。

### 400 Bad Request

| 错误消息 | 根因 | 来源层 | 排查 |
|---------|------|-------|------|
| `Invalid request body: {err}` | 请求体读取失败 | 客户端 | 检查客户端请求格式 |
| `Invalid JSON format: {err}` | JSON 解析失败 | 客户端 | 检查 JSON 格式 |
| `DefaultManager is nil` | Manager 未初始化 | Router | 检查 Router 启动日志 |

### 500 Internal Server Error

| 错误消息 | 根因 | 来源层 | 排查 |
|---------|------|-------|------|
| `Failed to build disaggregate_info: {err}` | PD 模式配置错误 | Router | 检查 register.yaml 参数 |
| `Failed to encode modified request: {err}` | 请求编码失败 | Router | 检查请求参数特殊字符 |
| `Internal server error` (Panic) | Router 代码 bug | Router | 检查 Panic recovered 日志 |

### 502 Bad Gateway

| 错误消息 | 根因 | 来源层 | 排查 |
|---------|------|-------|------|
| `Failed to select worker: {err}` | 无可用 Mixed Worker | FD 后端 | `curl /health` 检查后端 |
| `Failed to select worker pair: {err}` | 无可用 PD Worker | FD 后端 | 检查 prefill/decode 注册状态 |
| `Failed to connect to backend service: {err}` | 后端不可达 | FD 后端 | `curl {worker_url}/health` |

### 503 Service Unavailable

| 错误消息 | 根因 | 来源层 | 排查 |
|---------|------|-------|------|
| `No available prefill/decode workers` | 全部 Worker 不健康 | FD 后端 | 检查部署状态 |

---

## 按日志级别索引

### ERROR 级别

| 消息模板 | 严重程度 | 来源层 | 影响 |
|---------|---------|-------|------|
| `Failed to select mixed worker: {err}` | HIGH | FD 后端 | 请求返回 502 |
| `Failed to select prefill worker: {err}` | HIGH | FD 后端 | 请求返回 502 |
| `Failed to read register request body: {err}` | MEDIUM | Router | 注册失败 |
| `Failed to unmarshal register request JSON: {err}` | MEDIUM | Router | 注册失败 |
| `Failed to create decode request for {url}: {err}` | HIGH | FD 后端 | PD 请求失败 |
| `Failed to create prefill request for {url}: {err}` | HIGH | FD 后端 | PD 请求失败 |
| `Decode request failed for {url}: {err}` | HIGH | FD 后端 | PD 请求失败 |
| `Prefill request failed for {url}: {err}` | HIGH | FD 后端 | PD 请求失败 |
| `Failed to read request body: {err}` | LOW | 客户端 | 单请求失败 |
| `Failed to unmarshal request JSON: {err}` | LOW | 客户端 | 单请求失败 |
| `Failed to select worker pair: {err}` | HIGH | FD 后端 | 请求返回 502 |
| `Failed to build disaggregate_info: {err}` | HIGH | Router | 请求返回 500 |
| `Failed to encode modified request: {err}` | HIGH | Router | 请求返回 500 |
| `Failed to select worker: {err}` | HIGH | FD 后端 | 请求返回 502 |
| `Failed to connect to backend service: {err}` | HIGH | FD 后端 | 请求返回 502 |
| `Request failed (attempt {n}/{max}): {err}` | MEDIUM | FD 后端 | 重试中 |
| `Failed to create backend request for {url}: {err}` | HIGH | FD 后端 | 请求失败 |
| `Backend request failed for {url}: {err}` | HIGH | FD 后端 | 请求失败 |
| `scanner error: {err}` | MEDIUM | FD 后端/客户端 | 流式响应中断（gateway redirect 函数） |
| `[prefill] scanner error: {err}, message={msg}` | MEDIUM | FD 后端/客户端 | PD 模式 prefill 流式错误 |
| `copy error: {err}` | MEDIUM | FD 后端/客户端 | 非流式响应中断 |
| `[prefill] copy error: {err}, message={msg}` | MEDIUM | FD 后端/客户端 | PD 模式 prefill 非流式错误 |
| `Removed unhealthy prefill/decode/mixed instance: {url}` | HIGH | FD 后端 | Worker 被移除（注意：这是 ERROR 级别） |

### WARN 级别

| 消息模板 | 严重程度 | 来源层 | 影响 |
|---------|---------|-------|------|
| `GetRemoteMetrics failed for {url}, falling back to local counter` | LOW | FD 后端 | 调度精度降低 |
| `release worker: {url} skipped, counter already cleaned up` | LOW | Router | 计数器异常 |
| `release worker: {url} skipped, counter already zero (possible double-release)` | MEDIUM | Router | 计数器逻辑 bug |
| `cache-aware prefill: tokenizer failed, fallback to char tokens: {err}` | LOW | Router | cache-aware 精度降低 |
| `Instance {url} role is unknown` | LOW | Router | 注册角色不识别 |

### INFO 级别（异常相关）

| 消息模板 | 含义 | 关注场景 |
|---------|------|---------|
| `unhealthy worker counter preserved (inflight requests): {url}, count: {N}` | 不健康 Worker 仍有 inflight 请求 | 频繁出现说明 Worker 不稳定 |
| `unhealthy worker token counter preserved (inflight requests): {url}, tokens: {N}` | 不健康 Worker 仍有 token 计数 | 同上 |
| `cleanup unhealthy worker counter: {url}` | 清理不健康 Worker 的请求计数 | 正常清理 |
| `cleanup unhealthy worker token counter: {url}` | 清理不健康 Worker 的 token 计数 | 正常清理 |
| `preserved counters for {N} workers with inflight requests: [...]` | 保留了 N 个 Worker 的计数器 | N 大说明多 Worker 不稳定 |
| `removed counters for {N} unhealthy workers: [...]` | 移除了 N 个 Worker 的计数器 | 正常清理 |
| `Server {url} is healthy` | 健康检查恢复 | Worker 恢复（来自 HealthGenerate 端点） |

注意：以下事件是 **ERROR 级别**，不是 INFO：
- `Removed unhealthy prefill/decode/mixed instance: {url}` — Worker 被移除

注意：以下内容是 **HTTP 响应体**，不是 logger 输出（不会出现在日志行中）：
- `Register success` — 注册成功的 HTTP 200 响应体
- Worker 注册检测应通过 H1 行的 `POST /register 200` 判断

---

## 注册参数校验错误

| 错误消息 | 根因 | 排查 |
|---------|------|------|
| `invalid connector_port: {value}` | connector_port 非数字或范围错误 | 检查 register.yaml |
| `invalid engine_worker_queue_port: {value}` | engine_worker_queue_port 非数字或范围错误 | 检查 register.yaml |
| `invalid metrics_port: {value}` | metrics_port 非数字或范围错误 | 检查 register.yaml |
| `rdma_ports[{i}] invalid port: {value}` | RDMA 端口配置错误 | 检查 register.yaml |

---

## scanner error / copy error 区分

| error 内容 | 来源层 | 含义 |
|-----------|-------|------|
| `context canceled` | 客户端 | 客户端主动断连（超时或取消） |
| 其他 | FD 后端 | 后端流式响应异常 |
