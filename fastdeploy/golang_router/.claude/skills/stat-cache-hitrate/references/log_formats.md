# 日志格式参考

本文件描述 FastDeploy Go Router 的日志格式和解析规则。统计 cache 命中率前必须阅读。

---

## 通用日志行格式

```
[LEVEL] YYYY/MM/DD HH:MM:SS logger.go:<line>: <optional_context_prefixes> <message>
```

- **Level**：`[INFO]`、`[ERROR]`、`[WARN]`、`[DEBUG]`
- **Timestamp**：`YYYY/MM/DD HH:MM:SS`
- **可选 context 前缀**：`[trace_id:...]`、`[req_id:...]`、`[session_id:...]`、`[request_id:...]` 可能出现在 `logger.go:XX:` 和实际消息之间，顺序固定（trace_id → req_id → session_id → request_id），但不一定全部出现

---

## 类别 A：Cache-Aware 策略行

### A1. cache_aware_scoring（正常走 cache-aware 路径）

```
[INFO] 2026/03/30 20:16:57 logger.go:79: [session_id:slimshetty/swebench-verified:sweb.eval.x86_64.psf__requests-1766] [request_id:565a594c-...] cache-aware prefill: final strategy: cache_aware_scoring, selected=http://10.52.95.17:9263, loads=map[http://10.52.95.146:9263:20 http://10.52.95.17:9263:20 ...], hitRatios=map[http://10.52.95.17:9263:100]. ts_ms=2026-03-30 20:16:57.021
```

**提取字段**：
- `selected=<url>` — 被选中的 worker URL，格式 `http://IP:PORT`
- `hitRatios=map[...]` — Go map 格式，详见下方解析规则
- `loads=map[...]` — 各 worker 的负载

### A2. process_tokens fallback（未走 cache-aware 路径）

```
cache-aware prefill: final strategy: process_tokens, reason: load imbalanced, loads=map[...]
cache-aware prefill: final strategy: process_tokens, reason: tokenize failed: <error>
cache-aware prefill: final strategy: process_tokens, reason: strategy not initialized
```

---

## 类别 B：Stats 行

```
[INFO] 2026/03/30 20:14:38 logger.go:79: [stats] total_running=14, workers: [http://10.52.96.143:9867: running=0, http://10.52.95.26:9867: running=1, ...], cache_hit_rate=0.00% (hits=0/total=7)
```

**提取字段**：
- `total_running=<N>` — 所有 worker 的运行请求总数
- `workers: [...]` — 各 worker 的 `running=N`
- `cache_hit_rate=<X.XX>%` — 该窗口的命中率百分比
- `(hits=<N>/total=<M>)` — 该 5s 窗口的命中次数和总次数

**关键**：`hits` 和 `total` 是 **per-interval** 的，代码使用 `atomic.Swap(0)` 每 5s 重置为 0。

---

## 类别 C：推理请求行

```
[INFO] 2026/03/30 18:25:49 logger.go:79: [POST] /v1/chat/completions HTTP/1.1 200 2.798235ms 10.52.95.139
```

格式：`[METHOD] /path HTTP/1.1 <status_code> <duration> <client_ip>`

延迟单位可能是 `s`、`ms`、`µs`/`us`。

**注意**：仅 `POST /v1/chat/completions` 和 `POST /v1/completions` 为推理请求。其余路径（`/register`、`/registered_number`、`/registered`、`/health_generate`、`/metrics`）为管理/监控请求，统计推理吞吐量时应排除。

---

## Go Map 解析规则

Go 的 `fmt.Sprintf("%v", map)` 输出格式：`map[key1:val1 key2:val2 ...]`

### hitRatios 的特殊挑战

Worker URL 包含 `:`（如 `http://10.52.95.17:9263`），而 Go map 的 key-value 分隔符也是 `:`。
因此 `hitRatios=map[http://10.52.95.17:9263:100]` 中：
- URL = `http://10.52.95.17:9263`
- Ratio = `100`

### 推荐解析方法

**方法 1：正则匹配**（推荐）

提取 `hitRatios=map[` 和 `]` 之间的内容，然后用正则匹配每个 entry：

```
正则：(http://[^\s:]+:\d+):(\d+)
```

示例：
```
输入：http://10.52.95.17:9263:100 http://10.52.96.143:9867:50
匹配1：group1=http://10.52.95.17:9263, group2=100
匹配2：group1=http://10.52.96.143:9867, group2=50
```

**方法 2：从右分割**

对 map 内容按空格分割每个 token，然后对每个 token 找最后一个 `:` 分割：
```
token = "http://10.52.95.17:9263:100"
lastColon = 最后一个 ":" 的位置
url = token[:lastColon]  → "http://10.52.95.17:9263"
ratio = token[lastColon+1:]  → "100"
```

### 空 map

`hitRatios=map[]` 表示冷启动，没有任何 worker 有匹配的前缀缓存。

### loads map 解析

同样的规则适用于 `loads=map[...]`，value 是负载数：
```
loads=map[http://10.52.95.146:9263:20 http://10.52.95.17:9263:20]
```

### workers 列表解析（stats 行）

`workers: [http://10.52.96.143:9867: running=0, ...]` 格式不同：
- 用 `,` 分割每个 entry
- 每个 entry 格式：`http://IP:PORT: running=N`
- 注意 URL 后面跟的是 `: running=`（带空格），不是 Go map 的 `:val`

---

## 时间戳解析

日志时间戳格式：`YYYY/MM/DD HH:MM:SS`

提取正则：`(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})`

用于：
- 确定日志时间跨度
- 按时间分窗口（5s、1min 等）
- 按 quartile 分段统计趋势
