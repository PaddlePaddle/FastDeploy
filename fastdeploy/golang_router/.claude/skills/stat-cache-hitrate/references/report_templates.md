# 报告输出模板

本文件包含 cache 命中率分析报告的终端输出模板和详细数据导出模板。

---

## 终端概览报告模板

```
## Cache Hit Rate Analysis Report
**File**: <path> | **Lines**: N | **Span**: <start> ~ <end> (<duration>)

### 1. Prefix Hit Ratio (KV Cache 内容复用度)
  累计平均: XX.X% (被选中 worker)
  分布:
    0-20%  ██░░░░░░░░░░░░░░░░░░  X%   (N=xxx)
   20-40%  ███░░░░░░░░░░░░░░░░░  X%   (N=xxx)
   40-60%  █████░░░░░░░░░░░░░░░  X%   (N=xxx)
   60-80%  ████████████░░░░░░░░  X%   (N=xxx)
  80-100%  ████████████████████  X%   (N=xxx)
  冷启动率: X.X%
  趋势: Q1=X% → Q2=X% → Q3=X% → Q4=X% ↑/↓/→

  Prefix Hit Ratio (5s 窗口):
  100%|                              ·····················
   80%|                     ····· ···
   60%|               ·····
   40%|          ·····
   20%|    ······
    0%|····
      +---+---+---+---+---+---+---+---+---+---→ time
       18:25 18:26 18:27 18:28 18:29 18:30

### 2. Session Hit Rate (请求级路由粘性)
  累计: XX.X% (hits=N / total=N)
  覆盖率: X.X% 的推理请求带 session_id
  趋势: Q1=X% → Q2=X% → Q3=X% → Q4=X%

  Session Hit Rate (5s 窗口):
  100%|                                    ····················
   80%|                          ··········
   60%|               ···········
   40%|
   20%|
    0%|·······
      +---+---+---+---+---+---+---+---+---+---→ time

### 3. Per-Worker Cache Stats
  ┌───────────────────────────┬──────────┬──────────┬─────────────────┐
  │ Prefill Worker            │ Selected │ Select % │ Avg Hit(Select) │
  ├───────────────────────────┼──────────┼──────────┼─────────────────┤
  │ http://10.52.95.17:9263   │   1,234  │  15.2%   │      82%        │
  │ http://10.52.96.143:9867  │     890  │  11.0%   │      74%        │
  │ ...                       │    ...   │   ...    │      ...        │
  └───────────────────────────┴──────────┴──────────┴─────────────────┘

### 4. Scheduling Strategy
  cache_aware_scoring: N (X%) | fallback: N (X%)
    fallback reasons: load_imbalanced=N, tokenize_failed=N, not_initialized=N
  非最优命中选择: X% (负载均衡优先于命中率的比例)

### 5. Diagnosis
  ✅/⚠/❌ <综合诊断>

### 图表说明（Legend）
  - Unicode 柱状图：每个区间的请求占比，条越长占比越高
  - ASCII 折线图：横轴是时间窗口，纵轴是命中率（0-100%）
  - Q1→Q4 趋势：按时间四等分后的均值变化（↑/↓/→）

📄 详细数据见:
  - 报告文件: /abs/path/to/skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/cache_hitrate_report_<timestamp>.md
    URI: file:///abs/path/to/skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/cache_hitrate_report_<timestamp>.md
  - 窗口明细: /abs/path/to/skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/details/per_window_data.md
    URI: file:///abs/path/to/skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/details/per_window_data.md
  - Session 命中详情: /abs/path/to/skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/details/session_hit_details.md
    URI: file:///abs/path/to/skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/details/session_hit_details.md
    (含 prefill_urls、worker 切换前后 request_id，以及命中率突降 request_id)
```

---

## 格式规则

### Unicode 柱状图

- 总宽度 20 个字符
- `█` 表示已填充部分，`░` 表示空白部分
- 后跟百分比和绝对数量

```
计算方法：
filled = round(percentage / 100 * 20)
bar = "█" * filled + "░" * (20 - filled)
output = f"{bar}  {percentage}%   (N={count})"
```

示例：
```
████████████░░░░░░░░  60%   (N=1200)
██████████████████░░  90%   (N=1800)
██░░░░░░░░░░░░░░░░░░  10%   (N=200)
```

### ASCII 折线图

- Y 轴：0-100% 范围，6 行（0%, 20%, 40%, 60%, 80%, 100%）
- X 轴：时间，标注关键时间点
- 数据点用 `·` 绘制
- 坐标轴用 `|` `+` `─` `→`

```
时间粒度自动调整：
- 日志跨度 <30min → 5s 原始粒度
- 日志跨度 <3h → 1min 粒度
- 日志跨度 >3h → 5min 粒度
```

图表宽度约 60 列。数据点太多时自动聚合到更粗的粒度。

### 表格

使用 Unicode box-drawing 字符：

```
┌ ─ ┬ ─ ┐    顶部
│   │   │    数据行
├ ─ ┼ ─ ┤    分隔行
│   │   │    数据行
└ ─ ┴ ─ ┘    底部
```

### 趋势箭头

- `↑` — 上升趋势（Q4 > Q1 + 10%）
- `↓` — 下降趋势（Q4 < Q1 - 10%）
- `→` — 稳定（变化 < 10%）

---

## 详细数据导出模板

主报告：`skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/cache_hitrate_report_<YYYYMMDD_HHMMSS>.md`
每窗口明细：`skill_output/stat-cache-hitrate/<YYYYMMDD_HHMMSS>/details/per_window_data.md`

### 主报告

```markdown
# Cache Hit Rate Detailed Report

**Generated**: <timestamp>
**Source**: <log_file_path>

## 1. Per-Worker 完整统计

| Worker | Selected | Select % | Avg Hit (Selected) | Avg Hit (All) | Max Hit |
|--------|----------|----------|--------------------|----- ---------|---------|
| http://10.52.95.17:9263 | 1,234 | 15.2% | 82% | 68% | 100% |
| ... | ... | ... | ... | ... | ... |

## 2. Fallback 明细

### 3.1 load imbalanced (N 次)
| Time | Loads |
|------|-------|
| 20:15:03 | map[...] |

### 3.2 tokenize failed (N 次)
| Time | Error |
|------|-------|
| ... | ... |

## 4. 非最优命中选择明细

| Time | Selected | Selected HR | Best Worker | Best HR | Load Diff |
|------|----------|-------------|-------------|---------|-----------|
| 20:15:10 | w1:9263 | 60% | w2:9867 | 85% | w1=5, w2=18 |
| ... | ... | ... | ... | ... | ... |
```

---

## --tail 快速查看模板

`--tail` 模式下只输出核心指标：

```
## Cache Hit Rate (Recent)
**File**: <path> | **tail <N> lines** | **Span**: <start> ~ <end>

  Prefix Hit Ratio:  XX.X% (avg) | Cold start: X.X%
  Session Hit Rate:  XX.X% (hits=N/total=N) | Coverage: X.X%
  Strategy: scoring N (X%) | fallback N (X%)

  Recent trend (1min buckets):
  100%|          ·····
   80%|     ·····
   60%|·····
      +---+---+---+---+---→
       -5m  -4m  -3m  -2m  -1m

💡 持续跟踪: /loop 30s /analyze-cache-hitrate --tail
```

## --watch 持续监控模板

`--watch` 模式先输出完整报告（同终端概览报告模板），末尾额外提示：

```
💡 全量分析完成。持续跟踪后续变化:
   /loop 30s /analyze-cache-hitrate --tail
```
