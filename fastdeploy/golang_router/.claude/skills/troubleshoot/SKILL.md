---
name: troubleshoot
description: >
  FastDeploy Go Router 综合问题排查 skill。覆盖错误分类、延迟分析、请求追踪、Worker 健康时间线、
  Cache 调度诊断、负载与计数器分析六个维度。输出按三层问题来源分类：Router 自身、FastDeploy 后端、客户端。

  当用户要求以下操作时触发此 skill：排查 router 问题、分析 router 日志、router 排查、
  查看 router 状态、综合排查、全量扫描、troubleshoot router、/troubleshoot、
  分析错误日志、502/503 排查、延迟分析、Worker 健康、负载分析、cache 调度诊断、
  请求追踪、trace 请求。
  关键词：troubleshoot、排查、router 问题、全量扫描、综合分析、error、502、latency、
  health、load、cache、trace、/troubleshoot。

---

# Router Troubleshooting

综合排查 FastDeploy Go Router 问题，输出完整诊断报告。

> IMPORTANT: 执行前务必先读取 `references/log_patterns.md` 了解日志格式和提取规则。错误分类时参考 `references/error_catalog.md`。涉及后端问题时参考 `references/fastdeploy_cross_reference.md`。

## 执行前交互

运行脚本前，Claude 必须按以下顺序向用户确认参数：

### 1. 日志文件路径
使用 AskUserQuestion 工具向用户询问日志文件路径。提供两个常用快捷选项（客户端会自动提供 Other 自定义输入）：
- 选项 1: `logs/router.log`（默认）
- 选项 2: `fd-router.log`（golang_router 根目录）

**重要规则**：
- 如果用户已经在消息中明确指定了日志路径，直接使用该路径，跳过询问步骤
- 用户指定路径后不要质疑、推荐替代文件、或以任何理由尝试切换到其他文件
- 支持绝对路径（如 `/home/user/logs/xxx.log`）和相对路径（如 `logs/fd-router (2).log`）

如果用户直接确认或未指定路径，使用脚本的自动发现逻辑。

### 2. 分析范围
必须使用 **AskUserQuestion 的离散选项**（不要只发纯文本编号）：
- 选项 1: `全量分析（默认）` — 分析整个日志文件
- 选项 2: `尾部分析` — 只分析最近数据（可指定行数或时间如 `--tail 5000` 或 `--tail 30m`）
- 选项 3: `指定时间段` — 分析特定时间范围内的日志

如果用户未选择，默认使用全量分析。

#### 指定时间段的处理

脚本原生支持 `--start` 和 `--end` 参数，无需手动预过滤。两者可单独或同时指定。

时间格式灵活：支持 `YYYY/MM/DD HH:MM:SS`、`HH:MM:SS`、`HH:MM`、`MM/DD`、`MM/DD HH:MM`。
缺失部分自动从日志首末行推断（缺年份取首行，缺日期取末行）。
`--start/--end` 与 `--tail` 互斥。

当用户选择“指定时间段”时，必须再发起一次 **AskUserQuestion**（离散选项）引导时间输入：
- 选项 1: `当天（00:00:00 到当前）`（推荐）
- 选项 2: `最近半小时`（自动换算为 `--start now-30m --end now` 语义）

用户若通过客户端默认 `Other` 输入时间，则将该输入直接作为时间范围参数解析。
可补充一条简短示例引导：
- 示例 1：`16:00-16:30`
- 示例 2：`03/31 16:00 ~ 03/31 18:00`
- 示例 3：`2026/03/31 16:00:00`（仅起始）

### 3. 分析模式
必须使用 **AskUserQuestion 的离散选项**（不要只发纯文本编号）：
- 选项 1: `完整分析（默认）` — 运行所有维度（errors + latency + health + cache + load）
- 选项 2: `单维度/多维度分析` — 选择特定维度（errors / latency / health / cache / load），可选多个
- 选项 3: `请求追踪` — 追踪特定请求 ID

如果用户未选择，默认使用完整分析。

当用户选择“请求追踪”后，**不要再发 AskUserQuestion** 收集 trace ID。
直接发一条提示并等待用户输入完成后再继续执行即可。

提示文案建议：
- `请输入要追踪的 ID（支持 trace_id / request_id / session_id，多个用逗号分隔；输入 all 可全量追踪）`
- 示例：`a1b2c3d4` / `trace-001,trace-002` / `session-abc-123` / `all`

### 4. 输出目录
诊断报告默认保存到 `skill_output/troubleshoot/<YYYYMMDD_HHMMSS>/`（自动按运行时间创建子目录）。
用户可通过 `--output` 指定**基目录**，脚本会继续在其下创建 `<YYYYMMDD_HHMMSS>/summary` 与 `<YYYYMMDD_HHMMSS>/detail`，避免覆盖历史明细。

## 用法

脚本路径（相对于 `fastdeploy/golang_router/`）：`.claude/skills/troubleshoot/scripts/`

```bash
SCRIPTS=.claude/skills/troubleshoot/scripts

# 全量扫描（errors + latency + health + cache + load）
python3 $SCRIPTS/troubleshoot.py <log_file>

# 单维度分析
python3 $SCRIPTS/troubleshoot.py <log_file> --errors
python3 $SCRIPTS/troubleshoot.py <log_file> --latency
python3 $SCRIPTS/troubleshoot.py <log_file> --health
python3 $SCRIPTS/troubleshoot.py <log_file> --cache
python3 $SCRIPTS/troubleshoot.py <log_file> --load

# 请求追踪（需指定 ID，支持逗号分隔多 ID）
python3 $SCRIPTS/troubleshoot.py <log_file> --trace <ID>
python3 $SCRIPTS/troubleshoot.py <log_file> --trace "id1,id2"
python3 $SCRIPTS/troubleshoot.py <log_file> --trace all

# 尾部分析
python3 $SCRIPTS/troubleshoot.py <log_file> --tail 5000
python3 $SCRIPTS/troubleshoot.py <log_file> --tail 30m

# 指定时间段（--start 和 --end 可单独或同时使用）
python3 $SCRIPTS/troubleshoot.py <log_file> --start "16:00:00" --end "17:00:00"
python3 $SCRIPTS/troubleshoot.py <log_file> --start "2026/03/31 16:00:00"
python3 $SCRIPTS/troubleshoot.py <log_file> --start "03/31" --end "03/31 18:00"

# 组合模式
python3 $SCRIPTS/troubleshoot.py <log_file> --errors --latency
python3 $SCRIPTS/troubleshoot.py <log_file> --errors --tail 5000
python3 $SCRIPTS/troubleshoot.py <log_file> --start "16:00" --end "17:00" --errors --latency
```

默认日志路径：`logs/router.log` → `fd-router.log`

## 输出

- **终端**：简洁三层汇总（Router / FD 后端 / 客户端），含状态码分布、错误 Top N、趋势图
- **文件**：详细报告导出到 `skill_output/troubleshoot/<YYYYMMDD_HHMMSS>/summary/troubleshoot_report.md`
  - 逐分钟事件详情拆分到 `detail/health_events.md`
  - 请求追踪事件链拆分到 `detail/trace/trace_<ID>.md`
- **Cache 明细要求**：`cache_session_stickiness.md` / `cache_suboptimal.md` / `cache_eviction.md` / `cache_fallback.md` / `cache_cross.md`
  必须始终生成（即使无异常也写“未发现/样本不足”总结，避免链接缺失）
- **状态行**：`STATUS: HEALTHY / DEGRADED / CRITICAL`

## 三层诊断框架

| 层 | 典型问题 | 日志特征 |
|----|---------|---------|
| Router | Panic、500、Counter 异常、调度瓶颈、Cache 策略不优 | `Panic recovered`、`Failed to encode`、`double-release` |
| FD 后端 | 502、Worker 下线、高推理延迟、请求卡住 | `Failed to connect`、`Removed unhealthy`、p99 高 |
| 客户端 | 断连、请求格式错误 | `context canceled`、400 |

## 脚本架构

```
scripts/
  log_parser.py    — 日志解析原语（HTTP/Cache/Stats/错误归一化/事件匹配）
  stats.py         — 通用统计计算（百分位数/时间窗口/分组）
  chart.py         — 终端可视化（sparkline/柱状图/表格/时间线）
  troubleshoot.py  — 主编排器
  analyzers/
    errors.py      — 错误分类分析
    latency.py     — 延迟分析
    health.py      — Worker 健康时间线
    cache.py       — Cache 调度诊断
    load.py        — 负载与计数器分析
    trace.py       — 请求追踪
```

## 重要规则

1. 大文件 (>5000 行) 用 grep 分类提取，不一次性读取
2. 每个问题标注来源层（Router / FD 后端 / 客户端）
3. Cache 命中率数值分析用 `/stat-cache-hitrate`，本 skill 做策略诊断
4. 分析前读取 `references/log_patterns.md`
5. 错误查询参考 `references/error_catalog.md`
6. 后端问题排查参考 `references/fastdeploy_cross_reference.md`
7. 输出格式参考 `references/report_templates.md`
