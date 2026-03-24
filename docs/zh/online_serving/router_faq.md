[English](../../online_serving/router_faq.md)

# Router 常见问题排查

本文档基于 [Golang Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/golang_router) 的代码实现，汇总了 Router 在使用过程中常见的日志信息、返回输出及问题排查方法，帮助用户快速定位和解决问题。

Router 的基本使用方式请参考 [负载均衡调度 Router](router.md)。

## 常见日志分析

> **说明**：`{}` 表示变量，实际日志中会替换为具体的值。

### Error 级别日志

| 日志表现 | 日志含义 | 导致结果 | 可排查内容 |
| :--- | :--- | :--- | :--- |
| `Removed unhealthy prefill instance: {url}` | Prefill 实例健康检查失败，已被移除 | 该 Prefill 实例不再参与调度 | 健康状况 |
| `Removed unhealthy decode instance: {url}` | Decode 实例健康检查失败，已被移除 | 该 Decode 实例不再参与调度 | 健康状况 |
| `Removed unhealthy mixed instance: {url}` | Mixed 实例健康检查失败，已被移除 | 该 Mixed 实例不再参与调度 | 健康状况 |
| `Failed to register instance: {error}` | 实例注册失败 | Router 无法注册该实例 | 健康状况、注册参数 |
| `Failed to read YAML file {path}: {error}` | 启动时无法读取注册配置文件 | 配置文件中的实例无法被注册 | 文件路径、文件权限 |
| `Failed to unmarshal YAML file {path}: {error}` | 注册配置文件格式解析错误 | 配置文件中的实例无法被注册 | YAML 格式 |
| `Failed to register instance from index {index}: {error}` | 配置文件中第 {index} 个实例注册失败 | 该实例未能成功注册 | 健康状况、注册参数 |
| `failed to send request to {url} with error: {error}` | 健康检查请求发送失败 | 该实例可能被判定为不健康 | 网络连通性、代理设置 |
| `scanner error: {error}` | 读取后端流式响应时发生错误 | 当前请求可能失败 | 后端实例状态 |
| `[prefill] scanner error: {error}, message={message}` | 读取 Prefill 后端流式响应时发生错误 | 当前 Prefill 请求可能失败 | 后端实例状态 |
| `[prefill] copy error: {error}, message={message}` | 复制 Prefill 响应数据时发生错误 | 当前 Prefill 请求可能失败 | 后端实例状态 |
| `Panic recovered: {error}` | 请求处理过程中发生 panic 并被恢复 | 当前请求失败，但服务继续运行 | 后端实例状态、请求内容 |
| `empty baseURL provided` | 健康检查时传入了空的基础 URL | 健康检查无法执行 | 注册参数 |
| `failed to create request: {error}` | 创建健康检查请求失败 | 该实例可能被判定为不健康 | 网络环境 |
| `failed to read response body: {error}` | 读取健康检查响应体失败 | 该实例可能被判定为不健康 | 后端实例状态 |

### Warn 级别日志

| 日志表现 | 日志含义 | 导致结果 | 可排查内容 |
| :--- | :--- | :--- | :--- |
| `Server {url} is not healthy` | 该 URL 对应的实例未通过健康检查 | Router 无法注册该实例，或将该实例从已注册列表中移除 | 健康状况 |
| `Instance {url} role is unknown` | 实例角色无法识别 | 该实例不会被加入调度列表 | 注册参数 |
| `cache-aware prefill: tokenizer failed, fallback to char tokens: {error}` | Tokenizer 服务调用失败，已自动回退至字符级分词 | cache_aware 策略仍然生效，使用字符级分词代替 Tokenizer 进行缓存匹配，不影响正常请求处理 | Tokenizer 服务状态 |
| `cache-aware prefill: tokenize failed, fallback to process_tokens: {error}` | 分词彻底失败（如输入为空），回退至 process_tokens 策略 | Prefill 调度暂时不使用 cache_aware 策略，不影响正常请求处理 | 请求内容、Tokenizer 服务状态 |
| `cache-aware prefill: final strategy: process_tokens, reason: tokenize failed: {error}. ts_ms={ts}` | 分词失败（新格式），回退至 process_tokens 策略 | Prefill 调度暂时不使用 cache_aware 策略，不影响正常请求处理 | 请求内容、Tokenizer 服务状态 |

### Info 级别日志

| 日志表现 | 日志含义 | 说明 |
| :--- | :--- | :--- |
| `Starting server on {host:port}` | Router 服务正在启动 | 正常启动日志 |
| `Server {url} is healthy` | 实例健康检查通过 | 正常运行日志 |
| `Successfully registered instance from index {index}` | 配置文件中的实例注册成功 | 正常启动日志 |
| `No instances found in config file {path}` | 注册配置文件中未找到实例信息 | 请检查 register.yaml 内容是否为空 |
| `Request completed successfully.` | 请求处理完成 | 正常运行日志 |
| `Request failed, retrying...` | 请求失败，正在进行重试 | Router 最多重试 3 次 |
| `select worker (prefill): {url}, tokens: {tokens}` | Prefill 调度选中 Worker，显示当前 token 处理量 | 正常运行日志 |
| `select worker ({type}): {url}, count: {count}` | Decode/Mixed 调度选中 Worker，显示当前请求并发数 | 正常运行日志 |
| `release worker: {url}, count: {count}` | 请求结束，释放 Worker 计数器 | 正常运行日志 |
| `release prefill tokens: {url}, tokens: {tokens}` | Prefill 请求结束，释放 token 负载 | 正常运行日志 |
| `cleanup unhealthy worker counter: {url}` | 清理不健康 Worker 的计数器 | 正常运行日志 |
| `removed counters for {count} unhealthy workers: {urls}` | 批量清理不健康 Worker 的计数器 | 正常运行日志 |
| `[stats] total_running={n}, workers: [{loads}], cache_hit_rate={rate}% (hits={hits}/total={total})` | 周期性统计：总请求数、各 Worker 负载、缓存命中率 | 正常运行日志，用于监控调优 |
| `Parsing completed; starting worker selection.` | 请求解析完成，开始选择 Worker | 正常运行日志 |
| `Request completed with an error.` | 请求处理完成但发生错误 | 请排查后端实例状态 |
| `[SelectWorkerPair] decode selection failed, releasing prefill counter url={url}` | PD 分离模式下 Decode 选择失败，释放 Prefill 计数器 | 异常处理日志 |
| `[prefill] first chunk received, release counter url={url}` | Prefill 流式响应收到首个数据块，释放计数器 | 正常运行日志 |
| `[prefill] non-stream prefill response done, release counter url={url}` | Prefill 非流式响应完成，释放计数器 | 正常运行日志 |
| `[prefill] backendResp is nil or backendResp.Body is nil, url={url}` | Prefill 后端响应为空 | 可能表示后端连接异常 |
| `[prefill] release in defer (fallback) url={url}, isStream={bool}` | Prefill 请求异常退出时的兜底资源释放 | 异常处理日志 |
| `[prefill] release in CommonCompletions defer (error path) url={url}` | 请求出错时释放 Prefill 资源 | 异常处理日志 |
| `cache-aware prefill: final strategy: process_tokens, reason: strategy not initialized` | cache_aware 策略未初始化，回退至 process_tokens | 请检查 cache_aware 相关配置 |
| `cache-aware prefill: final strategy: process_tokens, reason: load imbalanced, loads={loads}. ts_ms={ts}` | 实例间负载不均衡，回退至 process_tokens 策略 | 正常运行日志，负载均衡自动切换 |
| `cache-aware prefill: final strategy: cache_aware_scoring, selected={url}, loads={loads}, hitRatios={ratios}. ts_ms={ts}` | cache_aware 打分策略选中 Worker | 正常运行日志，显示负载和命中率 |
| `[{method}] {path} {proto} {status} {latency} {clientIP}` | HTTP 请求访问日志 | 正常运行日志，记录每个请求的基本信息 |
| `before SelectWorker prefill. ts_ms={ts}` | PD 分离模式下开始选择 Prefill Worker | 正常运行日志，用于性能追踪 |
| `before SelectWorker decode, after prefill. ts_ms={ts}` | Prefill 选择完成后开始选择 Decode Worker | 正常运行日志，用于性能追踪 |
| `after SelectWorker decode, before return. ts_ms={ts}` | Decode Worker 选择完成 | 正常运行日志，用于性能追踪 |

### Debug 级别日志

> Debug 级别日志需要将日志级别设置为 `debug` 才会输出，通常用于开发调试。

| 日志表现 | 日志含义 | 说明 |
| :--- | :--- | :--- |
| `Healthy instances: prefill={urls}, decode={urls}, mixed={urls}` | 列出当前各角色的健康实例列表 | 用于确认实例发现是否正常 |
| `cache-aware prefill: hashes={n} workers={n} load={loads} hit={hits}` | cache_aware 策略的 hash 数量、Worker 数量及负载信息 | 用于调试缓存命中 |
| `cache-aware prefill: tokenizer tokens={tokens}` | Tokenizer 分词结果 | 用于调试分词结果 |
| `cache-aware score: worker={url} hit={hit} loadRatio={ratio} score={score}` | cache_aware 策略的打分详情 | 用于调试调度决策 |
| `radix match: hashes={n} matched_len={n} node_children={n}` | 前缀树匹配详情 | 用于调试缓存匹配 |
| `radix record: worker={url} hashes={n} node_depth={n}` | 前缀树记录详情 | 用于调试缓存记录 |
| `radix eviction: removed={n} nodeCount={n}` | 前缀树淘汰详情 | 用于调试缓存淘汰 |

## 常见返回输出分析

### 推理请求错误（/v1/chat/completions、/v1/completions）

| 输出表现 | HTTP 状态码 | 含义 | 可排查内容 |
| :--- | :---: | :--- | :--- |
| `{"error": "No available prefill/decode workers"}` | 503 | Prefill 或 Decode 实例全部不可用，没有已注册的健康实例 | 健康状况 |
| `{"error": "Failed to select worker pair"}` | 502 | PD 分离模式下选择 Worker 对失败 | 健康状况、调度策略 |
| `{"error": "Failed to select worker"}` | 502 | 集中式模式下选择 Worker 失败 | 健康状况、调度策略 |
| `{"error": "Failed to connect to backend service"}` | 502 | 连接后端推理实例失败（已重试 3 次仍失败） | 后端实例状态、网络连通性 |
| `{"error": "Failed to build disaggregate_info"}` | 500 | 构建 PD 分离通信信息失败 | 注册参数（connector_port、device_ids 等） |
| `{"error": "Invalid request body"}` | 400 | 请求体读取失败 | 请求格式 |
| `{"error": "Invalid JSON format"}` | 400 | 请求体 JSON 解析失败 | 请求格式 |

### 注册请求错误（/register）

| 输出表现 | HTTP 状态码 | 含义 | 可排查内容 |
| :--- | :---: | :--- | :--- |
| `{"code": 503, "msg": "{url} service is not healthy"}` | 503 | 实例健康检查未通过，无法注册 | 健康状况 |
| `{"code": 400, "msg": "Invalid request body"}` | 400 | 注册请求体读取失败 | 请求格式 |
| `{"code": 400, "msg": "Invalid InstanceInfo JSON format: {error}"}` | 400 | 注册请求 JSON 格式错误 | 请求格式 |
| `{"code": 400, "msg": "splitwise mode only supports PREFILL/DECODE instances"}` | 400 | PD 分离模式下不允许注册 MIXED 实例 | 部署模式、实例角色 |
| `{"code": 400, "msg": "only MIXED instances are allowed"}` | 400 | 集中式模式下只允许注册 MIXED 实例 | 部署模式、实例角色 |
| `{"code": 400, "msg": "invalid InstanceInfo format: {error}"}` | 400 | 实例注册信息校验失败 | 注册参数 |
| `{"code": 200, "msg": "Register success"}` | 200 | 注册成功 | — |

### 常见注册参数校验错误

| 错误信息 | 含义 | 解决方法 |
| :--- | :--- | :--- |
| `role is required` | 缺少 role 字段 | 添加 role 字段，可选值：prefill / decode / mixed |
| `invalid role: {role}` | role 值不合法 | 使用合法的 role 值：prefill / decode / mixed |
| `host_ip is required` | 缺少 host_ip 字段 | 添加 host_ip 字段 |
| `invalid host_ip: {ip}` | host_ip 不是合法的 IP 地址 | 填写正确的 IP 地址 |
| `port is required` | 缺少 port 字段 | 添加 port 字段 |
| `invalid port: {port}` | port 不是合法的端口号 | 填写 1-65535 范围内的端口号 |
| `invalid protocol: {protocol}` | 传输协议不合法 | 使用合法的协议值：ipc / rdma |

## 常见问题排查方式

### 健康状况

实例健康检查是 Router 正常运行的基础。当出现实例无法注册或被移除的情况时，请按以下步骤排查：

**1. 查看实例注册情况**

查看 Router 当前已注册的实例列表和数量：
```bash
# 查看已注册实例列表
curl -X GET http://{router_url}/registered

# 查看已注册实例数量
curl -X GET http://{router_url}/registered_number
```

确认所有预期实例是否都已注册成功。如果实例数量不符，说明部分实例注册失败或已被健康检查移除。

**2. 检查实例健康状态与网络连通性**

从 Router 所在机器直接访问推理实例的健康接口：
```bash
curl -X GET http://{server_url}/health
```

- 返回 HTTP 200 表示实例健康且网络连通
- 无法访问或返回非 200 状态码，请进一步排查：
  - 实例是否已启动并监听在正确端口上
  - Router 与实例之间是否存在代理干扰（尝试关闭代理：`unset http_proxy && unset https_proxy`）
  - 是否存在防火墙限制

**常见解决方式：**
- 关闭网络代理：`unset http_proxy && unset https_proxy`
- gunicorn 版本兼容问题：如果出现注册实例数不全的情况，可能是 gunicorn 与 FastDeploy 版本不兼容，回退至 `gunicorn==25.0.3` 可解决

### 调度策略

当出现 `Failed to select worker` 或 `Failed to select worker pair` 错误时：

**1. 确认已注册实例数**
```bash
curl -X GET http://{router_url}/registered_number
```

如果返回的实例数为 0，说明没有可用的健康实例，请先参考[健康状况](#健康状况)排查。

**2. 检查调度策略配置**

确认 config.yaml 中的调度策略与部署模式匹配。Router 的默认调度策略如下：

| 部署模式 | 配置项 | 默认策略 |
| :--- | :--- | :--- |
| 集中式模式 | `policy` | `request_num` |
| PD 分离模式（Prefill 节点） | `prefill-policy` | `process_tokens` |
| PD 分离模式（Decode 节点） | `decode-policy` | `request_num` |

如未通过配置文件指定策略，Router 将使用上述默认策略进行调度。如需使用 `cache_aware`、`fd_metrics_score` 等高级策略，请通过配置文件显式指定。各策略的详细说明请参考 [调度策略说明](router.md#调度策略说明)。

**3. 检查 fd_metrics_score 策略的依赖**

使用 `fd_metrics_score` 策略时，Router 会实时从推理实例的 `/metrics` 接口获取 running/waiting 请求数进行打分调度。当 `/metrics` 接口不可用时（如实例未配置 `metrics_port` 或 metrics 服务异常），Router 会自动回退至内部请求计数器进行调度，不影响正常的请求处理，但调度精度会有所下降。

如需确保 `fd_metrics_score` 策略获得最佳调度效果，请确认推理实例的 metrics 接口正常返回：
```bash
curl -X GET http://{server_url}/metrics
```

### 注册参数

当注册失败且返回参数校验错误时：

**1. 检查部署模式与实例角色是否匹配**
- PD 分离模式（`--splitwise`）：只能注册 `prefill` 和 `decode` 角色的实例
- 集中式模式（默认）：只能注册 `mixed` 角色的实例

**2. 检查必填参数**

注册请求必须包含以下字段：
- `role`：实例角色（prefill / decode / mixed）
- `host_ip`：实例 IP 地址
- `port`：实例端口号

**3. 检查 PD 分离模式下的可选参数**

PD 分离模式下建议完整配置以下参数，以确保 KV Cache 传输正常：
- `connector_port`：PD 通信端口
- `transfer_protocol`：传输协议（ipc / rdma）
- `device_ids`：GPU 设备 ID
- `rdma_ports`：RDMA 端口（使用 rdma 协议时必填）

### 启动失败

**1. 配置文件加载失败**

启动时日志出现 `Failed to load config`，请检查：
- `--config_path` 指向的文件路径是否正确
- 配置文件是否为合法的 YAML 格式
- 配置参数值是否合法

**2. 端口被占用**

启动时日志出现 `Failed to start server`，请检查：
- `--port` 指定的端口是否已被其他进程占用
- 可使用 `lsof -i:{port}` 或 `netstat -tlnp | grep {port}` 确认端口使用情况

### Tokenizer 服务（cache_aware 策略）

使用 `cache_aware` 调度策略时，Router 会调用 Tokenizer 服务对请求进行分词以计算缓存命中率。当 Tokenizer 服务不可用时，Router 内置了两级退化机制：

1. **回退至字符级分词**（常见情况）：日志出现 `tokenizer failed, fallback to char tokens`。此时 cache_aware 策略仍然生效，只是使用字符级分词代替 Tokenizer 进行缓存匹配，缓存命中精度会有所下降，但不影响正常请求处理。
2. **回退至 process_tokens 策略**（极端情况）：当分词彻底失败（如请求内容为空）时，日志出现 `tokenize failed, fallback to process_tokens`。此时 cache_aware 策略暂时不生效，改为按 token 处理量进行调度，同样不影响正常请求处理。

如需恢复 cache_aware 策略的完整功能：

**1. 检查 Tokenizer 服务是否正常运行**
```bash
curl -X POST http://{tokenizer_url}/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

**2. 检查相关配置**
- 确认 config.yaml 中 `tokenizer-url` 配置正确
- 如果 Tokenizer 服务响应较慢，可适当增大 `tokenizer-timeout-secs`（默认 2 秒）
