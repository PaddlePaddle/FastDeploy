# Router 问题排查实战手册（日志定位 + troubleshoot skill）

本文档结合以下两部分信息整理：
- Router 常见问题与日志语义：[`docs/zh/online_serving/router_faq.md`](router_faq.md)
- `fastdeploy/golang_router/.claude/skills/troubleshoot` 的脚本能力与使用方式

目标：给出一套可落地的排查流程，帮助你从“现象”快速定位到“日志证据”和“处理建议”。

---

## 1. 先定范围：全量 / 尾部 / 指定时间段

建议先根据问题发生时间选择分析范围（这是和分析模式并列的维度）：

- **全量分析**：适合历史慢性问题、趋势问题。
- **尾部分析（`--tail`）**：适合刚发生的故障，优先看最近 N 行或 N 分钟。
- **指定时间段（`--start/--end`）**：适合已知故障窗口（例如 14:05~14:20）。

> 说明：`--tail` 与 `--start/--end` 互斥，二选一。

---

## 2. 先看健康与注册，再看调度与请求

根据 `router_faq.md` 的建议，先确认“有没有可用实例”，再看“请求是否调度成功”。

### 2.1 健康与注册检查（必做）

```bash
# 已注册实例列表
curl -X GET http://{router_url}/registered

# 已注册实例数量
curl -X GET http://{router_url}/registered_number

# 从 Router 机器检查后端健康
curl -X GET http://{server_url}/health
```

重点日志关键词：
- 健康移除：`Removed unhealthy ... instance`
- 注册失败：`Failed to register instance`
- 健康检查失败：`failed to send request to ...` / `Server ... is not healthy`

若实例都不健康或未注册，后续 502/503 多数是结果，不是根因。

### 2.2 调度失败检查

常见错误：
- `Failed to select worker`
- `Failed to select worker pair`
- `No available prefill/decode workers`

这类问题先确认：
1) 注册数量是否为 0；
2) 调度策略与部署模式是否匹配；
3) `fd_metrics_score` 依赖的 `/metrics` 是否可访问。

### 2.3 请求链路与后端请求失败

常见日志：
- `Failed to connect to backend service`
- `Request failed (attempt n/max)`
- `Decode/Prefill/Backend request failed for {url}`
- `Panic recovered`

这类问题通常需要结合 trace（ID 级别）看完整链路。

---

## 3. 使用 troubleshoot skill 的标准方式

脚本入口（在 `fastdeploy/golang_router/` 下）：

```bash
SCRIPTS=.claude/skills/troubleshoot/scripts
python3 $SCRIPTS/troubleshoot.py <log_file> [options]
```

### 3.1 全量体检（默认推荐首轮）

```bash
python3 $SCRIPTS/troubleshoot.py <log_file>
```

会同时输出：errors / latency / health / cache / load 的综合结果。

### 3.2 指定维度分析（精准打点）

```bash
python3 $SCRIPTS/troubleshoot.py <log_file> --errors
python3 $SCRIPTS/troubleshoot.py <log_file> --latency
python3 $SCRIPTS/troubleshoot.py <log_file> --health
python3 $SCRIPTS/troubleshoot.py <log_file> --cache
python3 $SCRIPTS/troubleshoot.py <log_file> --load
```

### 3.3 请求追踪（ID 级排查）

```bash
# 单个 ID
python3 $SCRIPTS/troubleshoot.py <log_file> --trace <ID>

# 多个 ID
python3 $SCRIPTS/troubleshoot.py <log_file> --trace "id1,id2,id3"
```

trace 会展示：
- 匹配到的 tag 类型（request_id / trace_id / session_id / req_id）
- 生命周期完整性
- 事件链（含原始日志 RAW）
- 仅 request_id / 仅 session_id / 仅 trace_id 的统计
- 各标签组合形式（detail 中给出组合与对应 ID）

### 3.4 范围过滤与 trace 组合

当你要“在某个时间窗内追踪某个 ID”时，使用范围参数和 trace 组合：

```bash
python3 $SCRIPTS/troubleshoot.py <log_file> --start "2026/04/13 14:05:00" --end "2026/04/13 14:20:00" --trace "<ID>"
```

这符合“范围维度（全量/尾部/时间段）”与“模式维度（含 trace）”分离的使用方式。

---

## 4. 一套可复制的故障定位流程

### 步骤 A：确认故障窗口与错误现象
- 收集用户报错时间、HTTP 状态码（502/503/500/400）和请求路径。

### 步骤 B：先跑时间窗综合分析
```bash
python3 $SCRIPTS/troubleshoot.py <log_file> --start "HH:MM:SS" --end "HH:MM:SS"
```
- 看 STATUS（HEALTHY / DEGRADED / CRITICAL）。
- 优先看 errors、health 章节，判断是否是后端健康/注册问题。

### 步骤 C：按症状进入专项
- 502/503：`--errors --health --load`
- 延迟突增：`--latency --load --cache`
- 单请求失败：`--trace <ID>`（可叠加步骤 B 的时间窗）

### 步骤 D：在 detail 文件中取证
报告目录默认：
`skill_output/troubleshoot/<YYYYMMDD_HHMMSS>/`

重点文件：
- `summary/troubleshoot_report.md`
- `detail/trace_<ID>.md`
- `detail/health_events.md`
- `detail/load_select_release.md`

---

## 5. 现象到日志的快速映射

| 现象 | 优先看日志/关键词 | 推荐命令 |
|---|---|---|
| 503 无可用 worker | `No available prefill/decode workers`, `Removed unhealthy ...` | `--health --errors` |
| 502 调度失败 | `Failed to select worker`, `Failed to select worker pair` | `--errors --health --load` |
| 502 后端连接失败 | `Failed to connect to backend service`, `Request failed (attempt ...)` | `--errors --trace <ID>` |
| 请求卡住/链路不完整 | 有 select 无 release、无 `Request completed successfully.` | `--trace <ID>` |
| 延迟抖动 | HTTP latency、`[stats] total_running...` | `--latency --load --cache` |

---

## 6. 常见误区

1. **只看 502/503 响应，不看健康与注册日志**：容易把“结果”当“根因”。
2. **不限定时间窗口**：日志噪音大，容易误判。
3. **trace 只看结构化事件，不看 RAW**：可能漏掉关键上下文（例如同一秒的 WARN/ERROR 细节）。
4. **把范围维度和模式维度混在一起**：建议先定范围（全量/尾部/时间段），再定模式（完整/多维/trace）。

---

## 7. 推荐排查命令模板

```bash
# 模板 1：故障窗口综合体检
python3 $SCRIPTS/troubleshoot.py <log_file> --start "YYYY/MM/DD HH:MM:SS" --end "YYYY/MM/DD HH:MM:SS"

# 模板 2：最近 30 分钟快速巡检
python3 $SCRIPTS/troubleshoot.py <log_file> --tail 30m

# 模板 3：单请求深挖（配合时间窗）
python3 $SCRIPTS/troubleshoot.py <log_file> --start "HH:MM:SS" --end "HH:MM:SS" --trace "<request_or_trace_or_session_id>"
```

如果你已经知道故障集中在特定 ID，优先从模板 3 入手，然后回到模板 1 看全局背景。
