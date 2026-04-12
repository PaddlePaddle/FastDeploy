---
name: stat-cache-hitrate
description: >
  统计 FastDeploy Go Router 日志中的三层 cache 命中率指标，生成可视化报告。
  三层指标：Prefix Hit Ratio（KV Cache 内容复用度）、Session Hit Rate（请求级路由粘性）、
  Per-Worker Cache Stats（各 prefill worker 的缓存利用排名）。支持全量统计、tail 快速查看、
  持续监控模式。

  当用户提到以下内容时触发此 skill：统计/查看 cache 命中率、查看 cache-aware 调度效果、
  查看缓存预热情况、统计 hitRatio、查看 prefix 命中率、session hit rate。
  关键词：cache 命中率、hitRatio、cache-aware、prefix hit、session hit rate、
  缓存预热、/stat-cache-hitrate。

IMPORTANT: 执行前阅读 references/log_formats.md 了解日志格式和解析规则。
---

# Cache Hit Rate Statistics

统计 FastDeploy Go Router 的三层 cache 命中率，生成可视化报告。

## 执行前交互

运行脚本前，Claude 必须先向用户确认以下参数：

### 1. 日志文件路径
使用 AskUserQuestion 工具向用户询问日志文件路径。提供两个常用快捷选项（客户端会自动提供 Other 自定义输入）：
- 选项 1: `logs/router.log`（默认）
- 选项 2: `fd-router.log`（golang_router 根目录常用文件名）

**重要规则**：
- 如果用户已经在消息中明确指定了日志路径，直接使用该路径，跳过询问步骤
- 用户指定路径后不要质疑、推荐替代文件、或以任何理由尝试切换到其他文件
- 支持绝对路径（如 `/home/user/logs/xxx.log`）和相对路径（如 `logs/fd-router (2).log`）

如果用户直接确认或未指定路径，使用默认值 `logs/router.log`。

### 2. 分析模式
向用户询问分析模式：
> "请选择分析模式：
> 1. **全量统计**（默认）— 扫描完整日志
> 2. **快速查看尾部** — 只看最近的数据（可指定行数如 2000 或时间如 30m）
> 3. **持续监控** — 全量分析后提示监控命令
> 4. **指定时间段** — 分析特定时间范围（如 `--start "16:00" --end "17:00"`）"

如果用户未选择，默认使用全量统计。

`--start/--end` 与 `--tail` 互斥。`--start` 和 `--end` 可单独或同时指定。
时间格式灵活：支持 `YYYY/MM/DD HH:MM:SS`、`HH:MM:SS`、`HH:MM`、`MM/DD`、`MM/DD HH:MM`。
缺失部分自动从日志首末行推断。

### 3. 输出目录
分析结果默认保存到 `skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/`（自动按运行时间创建子目录）。
用户可通过 `--output` 指定自定义目录。

## 使用方式

运行统计脚本（相对于 `fastdeploy/golang_router/` 目录）：

```bash
# 全量统计
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --output skill_output/stat-cache-hitrate/

# 快速查看尾部数据
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --tail       # 默认最后 2000 行
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --tail 5000   # 指定行数
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --tail 30m    # 指定时间

# 持续监控
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --watch

# 指定时间段（--start 和 --end 可单独或同时使用）
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --start "16:00:00" --end "17:00:00"
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --start "2026/03/31 16:00:00"
python3 .claude/skills/stat-cache-hitrate/scripts/stat_cache_hitrate.py <日志文件> --start "03/31" --end "03/31 18:00"
```

默认日志路径：`logs/router.log`（相对于 `fastdeploy/golang_router/`）。常用备选：`fd-router.log`（根目录）。不传 `--output` 时自动输出到 `skill_output/stat-cache-hitrate/<timestamp>/`。

脚本会自动根据文件大小选择解析策略：小文件（<5000 行）在内存中处理，大文件用 grep + 管道流式处理。

## 输出说明

### 三层指标

| 层级 | 指标 | 含义 |
|------|------|------|
| 第一层 | Prefix Hit Ratio | 被选中 worker 的 KV cache 命中率，反映内容级复用度 |
| 第二层 | Session Hit Rate | 带 session_id 的请求被路由到同一 worker 的比例 |
| 第三层 | Per-Worker Stats | 每个 prefill worker 被选中的次数和平均命中率排名 |

### 输出文件位置

详细报告和图表输出到 `skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/` 目录，每次运行自动创建带时间戳的子目录。

- 主报告 `cache_hitrate_report_*.md` — Per-Worker 统计 + Fallback 明细
- `details/per_window_data.md` — 每5s窗口明细（连续空窗口自动合并为 3 行：起始/合并说明/结束）
- `details/session_hit_details.md` — 每个 session 的命中明细（TSV 单行格式，便于横向滚动查看），包含 `session / req_count / first_hit / avg_hit(excl_first) / max_hit / min_hit / all_hits / prefill_urls / switch_req_pairs / sharp_drop_request_ids`

### 交叉诊断矩阵

| Session HR | Prefix HR | 诊断 |
|------------|-----------|------|
| 高 | 高 | cache-aware 策略运行良好 |
| 高 | 低 | session 粘性好但 prompt 内容变化大，KV cache 实际复用低 |
| 低 | 高 | 换 worker 了但新 worker 也有类似前缀缓存 |
| 低 | 低 | 负载均衡强制分散或缓存未预热 |

## 重要规则

1. **`[stats]` 计数器 per-interval**：每 5s `atomic.Swap(0)` 重置，必须 sum 所有行计算累计值
2. **Session HR 只统计带 session_id 的请求**
3. **Prefix HR 取 selected worker 的值**：不在 hitRatios map 中则为 0
4. **此 skill 只关注 cache 命中率**：延迟/错误/健康等排查由 troubleshoot skill 负责
5. **与 troubleshoot-cache 互补**：本 skill 做数值统计，troubleshoot-cache 做调度策略诊断

## 参考文件

- `references/log_formats.md` — 日志格式和解析规则
- `references/report_templates.md` — 终端报告和详细导出的模板
