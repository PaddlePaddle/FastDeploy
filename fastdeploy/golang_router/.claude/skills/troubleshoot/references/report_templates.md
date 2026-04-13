# 报告输出规范

所有 troubleshoot 分析维度共享的可视化和格式规范。

---

## 通用可视化组件

### Unicode 柱状图
- 填充块：`█`（U+2588），空块：`░`（U+2591）
- 总宽度：20 字符，右侧标注百分比和计数
- 块数 = round(percentage / 100 * 20)，最小 1 块（>0% 时）

### Sparkline 折线图
- 字符集：`▁▂▃▄▅▆▇█`（8 级高度）
- 图表宽度：60 字符，自动降采样
- X 轴标注时间（首/尾 + 中间 2-3 个刻度）
- Y 轴自适应：百分比类 0-100%，计数类 0-max

### Markdown 表格
- 标准 Markdown 表格格式
- 数值列右对齐

### Worker 可用性时间线
- `█` = 在线，`░` = 下线
- 右侧标注在线率百分比

---

## 严重程度标记

| 标记 | 含义 | 使用场景 |
|------|------|---------|
| CRITICAL | 服务不可用 | Panic、全部 Worker 不健康、错误率 >20% |
| HIGH | 部分请求失败 | 502/503、Worker 频繁下线 |
| MEDIUM | 性能下降 | 高延迟、cache 命中率低 |
| LOW | 需关注 | 计数器异常、tokenizer 退化 |
| INFO | 正常 | 统计信息 |

---

## 报告格式

### 简洁版（终端输出）

- 第一行：`STATUS: HEALTHY / DEGRADED / CRITICAL — 简要说明`
- 状态定义：`HEALTHY`=无明显异常；`DEGRADED`=服务可用但性能/稳定性下降（需关注）；`CRITICAL`=服务不可用或高风险故障
- 按三层分类（Router / FD 后端 / 客户端）
- 每个问题一行摘要 + 关键指标
- 末尾提示详细版文件路径

### 详细版（文件导出）

- 路径：`skill_output/troubleshoot/<YYYYMMDD_HHMMSS>/troubleshoot_report_<timestamp>.md`
- 主报告包含各维度总结 + 可视化图表（sparkline/柱状图/时间线等）
- 详情拆分到 `details/` 子目录：
  - `detail/health_events.md` — Worker 逐分钟健康事件 + 健康诊断
  - `detail/load_select_release.md` — 负载诊断 + select/release 明细
  - `detail/load_diagnoses.md` — load 诊断列表
  - `detail/load_counter_state.md` — request/token counter 末状态
  - `detail/latency_diagnoses.md` — 延迟诊断详情
  - `detail/cache_diagnosis.md` — cache 六维诊断详情（session 粘性/非最优/驱逐/Fallback/冷启动/交叉诊断）
  - `detail/cache_session_stickiness.md` / `detail/cache_suboptimal.md` / `detail/cache_eviction.md` / `detail/cache_fallback.md` / `detail/cache_cross.md` — cache 分职责拆分明细
  - `detail/trace_<ID>.md` — 请求追踪事件链

---

## 状态判定规则

- **CRITICAL**：存在 Panic、全部 Worker 不健康、或错误率 >20%
- **DEGRADED**：存在 502/503、Worker 不稳定、或错误率 >5%
- **HEALTHY**：无严重问题

---

## 各维度报告结构

### Errors（错误分析）

```
HTTP 状态码分布（柱状图）
错误率趋势（折线图）
ERROR/WARN Top N（柱状图 + 表格，标注来源层）
Panic 列表
```

### Latency（延迟分析）— 待实现

```
延迟百分位数 (p50/p90/p95/p99)
延迟分布（柱状图）
吞吐量趋势（折线图）
慢请求 Top 10
```

### Health（Worker 健康）— 待实现

```
Worker 可用性时间线
健康事件汇总表
可用性统计
```

### Cache（调度诊断）— 待实现

```
调度策略分布
Session 粘性分析
非最优选择分析
Fallback 原因分类
```

### Load（负载分析）— 待实现

```
Worker 负载分布
计数器异常检测
Token 计数器统计
```

### Trace（请求追踪）— 待实现

```
单请求事件链
生命周期完整性检查
Session 多请求汇总
```
