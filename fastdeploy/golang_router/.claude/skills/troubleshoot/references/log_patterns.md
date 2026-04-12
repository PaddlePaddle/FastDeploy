# 日志格式与提取规则

本文档定义 Router 日志的所有类别、Grep 匹配模式、精确正则，供各子 skill 参考。

---

## 日志基本格式

```
[LEVEL] YYYY/MM/DD HH:MM:SS logger.go:<line>: [context_tags] message
```

### Context Tags（可选，顺序固定）

- `[trace_id:<value>]`
- `[req_id:<value>]`
- `[session_id:<value>]`
- `[request_id:<value>]`

所有 tag 可能同时出现，也可能只有部分或没有。顺序固定为：`trace_id → req_id → session_id → request_id`。

### ID 匹配正则

搜索某个 ID 时，同时匹配四种 tag：
```
session_id:<ID>|trace_id:<ID>|request_id:<ID>|req_id:<ID>
```

---

## 日志分类提取

| 类别 | Grep 模式 | 用途 | 典型内容 |
|------|----------|------|---------|
| E1 — ERROR | `\[ERROR\]` | 错误分类 | 各类 Failed to ... 错误 |
| E2 — WARN | `\[WARN\]` | 警告分类 | counter 异常、tokenizer 退化 |
| H1 — HTTP 请求 | `\] \[(POST\|GET)\] /` | 延迟/状态码/吞吐量 | HTTP middleware 日志行 |
| H2 — 健康事件 | `Removed unhealthy\|is not healthy\|is healthy` | Worker 健康时间线 | 上下线事件 |
| H2b — 注册事件 | `\] \[POST\] /register.*200` | Worker 注册 | 从 H1 HTTP 行中匹配 POST /register 返回 200 |
| H3 — 调度事件 | `select worker\|release worker\|Failed to select\|SelectWorkerPair` | 调度/计数器分析 | Worker 选择和释放 |
| H4 — 后端问题 | `Failed to connect\|request failed\|scanner error\|copy error\|Panic recovered` | 后端问题 | 连接/流式/Panic（注意：`scanner error`/`copy error` 与 H9 有重叠，带 `[prefill]` 前缀的行同时属于 H9） |
| H5 — Counter | `counter preserved\|cleanup unhealthy\|removed counters\|counter already\|double-release\|preserved counters` | 计数器异常 | 计数器生命周期 |
| H6 — Cache-aware | `cache-aware prefill: final strategy:` | Cache 调度诊断 | 策略选择 + hitRatios |
| H7 — Stats | `\[stats\]` | 负载/命中率 | 周期性统计行 |
| H8 — ts_ms | `ts_ms=` | 调度耗时 | 调度开始结束时间戳 |
| H9 — Prefill 生命周期 | `\[prefill\]` | PD 模式 prefill 追踪 | 首包/释放/错误 |
| H10 — 请求标记 | `Parsing completed\|Request completed successfully` | 请求生命周期 | 调度开始/请求结束标记 |
| H11 — Token 释放 | `release prefill tokens` | Token 计数器生命周期 | Token 释放事件 |

---

## H1 — HTTP 请求行格式

```
[INFO] 2025/01/15 18:25:33 logger.go:45: [POST] /v1/chat/completions HTTP/1.1 200 1.234567s 10.0.0.1
```

字段：`[METHOD] /path HTTP/1.1 STATUS LATENCY CLIENT_IP`

### 延迟单位归一化

Go `time.Duration.String()` 输出格式不固定，需归一化为毫秒：

| 原始格式 | 含义 | 转换为 ms |
|---------|------|----------|
| `1.5s` | 秒 | × 1000 |
| `150ms` | 毫秒 | 直接使用 |
| `150.5ms` | 毫秒 | 直接使用 |
| `500µs` | 微秒 | ÷ 1000 |
| `500us` | 微秒（ASCII） | ÷ 1000 |
| `500ns` | 纳秒 | ÷ 1000000 |
| `1m30s` | 分+秒 | 分×60000 + 秒×1000 |
| `1h2m3s` | 时+分+秒 | 时×3600000 + 分×60000 + 秒×1000 |

正则提取延迟值：`(\d+(?:\.\d+)?(?:h|m(?!s)|s|ms|µs|us|ns))+`

### 仅推理请求

延迟分析只统计推理请求路径：
- `/v1/chat/completions`
- `/v1/completions`

排除健康检查 `/health`、注册 `/register` 等管理路径。

---

## H6 — Cache-aware 策略行格式

```
[INFO] 2025/01/15 18:25:33 logger.go:87: [trace_id:xxx] [session_id:xxx] cache-aware prefill: final strategy: cache_aware_scoring, selected=http://10.0.0.1:9965, loads=map[http://10.0.0.1:9965:2 http://10.0.0.2:9965:5], hitRatios=map[http://10.0.0.1:9965:0.85 http://10.0.0.2:9965:0.42]. ts_ms=2025-01-15 18:25:33.123
```

```
[INFO] ... cache-aware prefill: final strategy: process_tokens, reason: load imbalanced, loads=map[...]. ts_ms=2025-01-15 18:25:33.123
```

注意：日志中**没有** `scores=map[...]` 字段。scores 仅在 DEBUG 级别的 `chooseByScore` 中逐条打印。
如需分析非最优选择，需从 hitRatios + loads 使用公式重新计算：
`score = (100-hitRatio)/100 * hitRatioWeight + loadRatio * loadBalanceWeight`

### Go map 解析

`hitRatios=map[key1:val1 key2:val2]`

- 空 map：`hitRatios=map[]` — 表示冷启动
- 正则提取 map 内容：`map\[(.*?)\]`
- 每对 key:value 用空格分隔：`(\S+):(\S+)`
- key 是 worker URL，value 是 float64

### selected worker 的 hitRatio

从 hitRatios map 中查找 selected URL 的值：
- 在 map 中找到 → 使用该值
- 不在 map 中 → hitRatio = 0
- map 为空 → 冷启动，hitRatio = 0

### ts_ms 格式

`ts_ms=2025-01-15 18:25:33.123`

格式：`2006-01-02 15:04:05.000`（Go reference time）

用于计算调度耗时（两个 ts_ms 之间的差值）。

---

## H7 — Stats 行格式

```
[INFO] 2025/01/15 18:25:33 logger.go:87: [stats] total_running=5, workers: [http://10.0.0.1:9965: running=2, http://10.0.0.2:9965: running=3], cache_hit_rate=85.71% (hits=6/total=7)
```

注意：由于 Go `log.Lshortfile` 打印的是 `Printf` 调用处，stats 行的源文件始终为 `logger.go:NN:`（行号随编译变化），而非 `handler.go`。

注意：stats 行**不包含**任何 context tag（trace_id 等），因为由后台 goroutine 周期输出。

### 关键：per-interval 计数器

`hits` 和 `total` 是 **per-interval** 的值（每 5s 通过 `atomic.Swap(0)` 重置为 0）。

计算累计值必须 **sum 所有行**：
- 累计 Session Hit Rate = `sum(hits) / sum(total) * 100`

### Worker 负载提取

`workers: [url1: running=N, url2: running=N]`

- 注意格式：`workers:` 带冒号+空格，每个 worker 格式为 `url: running=N`，逗号+空格分隔
- **不包含 token 数据**（reportStats 只读取 running 计数）

正则：`(http://[^:]+:\d+): running=(\d+)`

### cache_hit_rate 提取

`cache_hit_rate=85.71% (hits=6/total=7)`

正则：`cache_hit_rate=([\d.]+)% \(hits=(\d+)/total=(\d+)\)`

---

## 模板归一化

ERROR/WARN 消息分组时，需将变量替换为占位符：

| 变量类型 | 正则 | 替换为 |
|---------|------|-------|
| URL | `https?://[\w.:]+` | `{url}` |
| UUID | `[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}` | `{uuid}` |
| 数字 | `\d+` (仅在特定位置) | `{N}` |
| IP:Port | `\d+\.\d+\.\d+\.\d+:\d+` | `{ip:port}` |

---

## Fallback 策略行识别

| final strategy | reason 关键词 | 含义 |
|---------------|--------------|------|
| `cache_aware_scoring` | (无 reason) | 正常 cache-aware 调度 |
| `process_tokens` | `tokenize failed` | 退化 B：字符级 tokenize 也失败 |
| `process_tokens` | `load imbalanced` | 退化 C：负载不均衡 |
| `process_tokens` | (其他) | 退化 D：策略未初始化等 |

退化 A（Tokenizer 服务→字符级）在 WARN 行识别：
```
[WARN] ... cache-aware prefill: tokenizer failed, fallback to char tokens: {err}
```
注意完整前缀 `cache-aware prefill: tokenizer failed`。
退化 A 后仍可走 cache_aware_scoring（精度降低），与 B/C/D 不互斥。

---

## H4 — 后端问题匹配说明

H4 的 `request failed` 模式会匹配多个消息模板：
- `Request failed (attempt {n}/{max}): {err}` — 重试日志
- `Decode request failed for {url}: {err}` — PD 模式 decode 失败
- `Prefill request failed for {url}: {err}` — PD 模式 prefill 失败
- `Backend request failed for {url}: {err}` — 后端请求失败

分析时需通过模板归一化去重。

---

## H9 — Prefill 生命周期事件

PD（Prefill/Decode 分离）模式下，`completions.go` 产生的 `[prefill]` 前缀日志：

| 消息模板 | 含义 |
|---------|------|
| `[prefill] first chunk received, release counter url=%s` | Prefill 首包到达，释放计数器 |
| `[prefill] non-stream prefill response done, release counter url=%s` | 非流式 prefill 完成 |
| `[prefill] release in defer (fallback) url=%s, isStream=%v` | defer 兜底释放 |
| `[prefill] release in CommonCompletions defer (error path) url=%s` | 错误路径释放 |
| `[prefill] backendResp is nil or backendResp.Body is nil, url=%s` | 后端响应异常 |
| `[prefill] scanner error: %v, message=%s` | 流式读取错误（ERROR 级别） |
| `[prefill] copy error: %v, message=%s` | 非流式复制错误（ERROR 级别） |

---

## H10 — 请求生命周期标记

| 消息 | 含义 | 级别 |
|------|------|------|
| `Parsing completed; starting worker selection.` | 请求解析完成，开始调度 | INFO |
| `Request completed successfully.` | 请求成功完成 | INFO |

---

## H11 — Token 释放

`release prefill tokens: %s, tokens: %d` — 释放 prefill token 计数。
数据源：`handler.go:333`。用于 troubleshoot-load 的 token 计数器分析。

---

## Select/Release 日志细节（与代码一致）

- `select worker (prefill): <url>, tokens: <n>`
- `select worker (decode|mixed): <url>, count: <n>`
- `release worker: <url>, count: <n>`（request counter 释放）
- `release prefill tokens: <url>, tokens: <n>`（token counter 释放；可能来自 prefill 或 mixed 请求路径）

重点：release 只有上面这两种。`release worker` 不带 worker type，`release prefill tokens` 的文本也不能直接断定是 prefill（mixed 也可能调用）。因此按 `prefill/decode/mixed` 统计时，需要从 select 侧做归类；确实无法归类时才记为 `unknown`。

---

## 使用脚本工具

各 skill 的脚本位于各自的 `scripts/` 目录下，自动处理上述所有日志解析和计算。

### 快速参考

| 任务 | 脚本 |
|------|------|
| 解析 H1 HTTP 行 | `log_parser.py parse-http [--inference-only]` |
| 解析 H6 cache 策略行 | `log_parser.py parse-cache-strategy` |
| 解析 H7 stats 行 | `log_parser.py parse-stats` |
| 检测非支持请求 | `log_parser.py unsupported-requests [--summary-only]` |
| ASCII 折线图 | `chart.py` |
| Unicode 柱状图 | `chart.py` |
| Markdown 表格 | `chart.py` |
| Worker 时间线 | `chart.py` |

所有工具从 stdin 读取，输出到 stdout。中间数据使用 JSON Lines 格式。

---

## 已知路由列表

Router 支持的全部路由（来自 `internal/router/router.go`）：

| Method | Path | 类型 |
|--------|------|------|
| POST | `/v1/chat/completions` | 推理 |
| POST | `/v1/completions` | 推理 |
| POST | `/register` | 实例注册 |
| GET | `/registered_number` | 注册数量查询 |
| GET | `/registered` | 注册列表查询 |
| GET | `/health_generate` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |

### 非支持请求排查

客户端可能发送不属于已知路由的请求（如 `/v1/models`），会收到 404 但仍记录在 H1 HTTP 日志中。

使用 `log_parser.py unsupported-requests` 子命令检测：
```bash
# 完整输出（详细列表 + 汇总）
grep -E '\] \[(POST|GET|PUT|DELETE|PATCH|HEAD|OPTIONS)\] /' logfile | python3 log_parser.py unsupported-requests

# 仅汇总
grep -E '\] \[(POST|GET|PUT|DELETE|PATCH|HEAD|OPTIONS)\] /' logfile | python3 log_parser.py unsupported-requests --summary-only
```
