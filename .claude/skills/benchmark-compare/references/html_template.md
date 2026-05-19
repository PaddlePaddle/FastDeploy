# HTML 报告模板（多模式版）

本文件包含 Benchmark 对比报告的 HTML 模板规范。报告支持多种测试场景（量化方式 × 并发数）切换。

## 架构概述

报告采用 **数据驱动 + 动态渲染** 模式：
1. 所有场景的指标数据嵌入为 JavaScript 对象 `benchmarkData`
2. 用户通过 Segmented Control 选择量化方式和并发数
3. 选择变化时，JS 动态更新所有 UI 组件（指标卡片、图表、表格、结论）
4. 选择状态持久化到 `localStorage`

**不再使用 `{{PLACEHOLDER}}` 占位符模式。** Agent 生成报告时直接从数据源（日志文件 / metrics.json）读取指标并嵌入为 JSON。

---

## 规则

- **百分比符号规则**：正数+绿色=FD 领先，负数+红色=FD 落后
  - 吞吐类指标（越高越好）：`diff = (fd - sg) / sg * 100`
  - 延迟类指标（越低越好）：`diff = (sg - fd) / fd * 100`
  - 正值用 `m-better` class（绿色），负值用 `m-worse` class（红色）
- **配置卡片对齐**：FD 和 SG 卡片同一行位置放置对等概念（GPU/TP、并发/长度、Attention/量化、版本信息）
- **数据集链接**：Params Bar 中数据集需带下载链接和「（点击可下载）」后缀
- **默认主题**：明亮模式 (`data-theme="light"`)
- **默认选择**：bf16 + 并发512（或由数据源决定的最大并发），持久化到 localStorage
- **中文界面**，技术术语保留英文（TTFT、TPOT、ITL、E2EL、Throughput、Decode、MoE）

---

## 数据结构

### benchmarkData 对象

```javascript
const benchmarkData = {
  "bf16_bs1": {
    "fd": { "successful_requests": 830, "benchmark_duration": 4568.59, "total_token_throughput": 177.79, ... },
    "sg": { "successful_requests": 830, "benchmark_duration": 3228.47, "total_token_throughput": 251.59, ... }
  },
  "bf16_bs32": { "fd": {...}, "sg": {...} },
  "bf16_bs64": { "fd": {...}, "sg": {...} },
  "bf16_bs512": { "fd": {...}, "sg": {...} },
  "fp8_bs1": { "fd": {...}, "sg": {...} },
  "fp8_bs32": { "fd": {...}, "sg": {...} },
  "fp8_bs64": { "fd": {...}, "sg": {...} },
  "fp8_bs512": { "fd": {...}, "sg": {...} }
};
```

Key 格式：`{quant}_bs{concurrency}`

每个 framework 对象包含的完整指标字段：
```
successful_requests, benchmark_duration,
total_input_tokens, total_generated_tokens,
request_throughput, output_token_throughput, total_token_throughput,
mean_ttft, median_ttft, p80_ttft, p95_ttft, p99_ttft,
mean_tpot, median_tpot, p80_tpot, p95_tpot, p99_tpot,
mean_itl, median_itl, p80_itl, p95_itl, p99_itl,
mean_e2el, median_e2el, p80_e2el, p95_e2el, p99_e2el,
mean_decode, median_decode
```

---

## UI 结构

### 1. Segmented Control（选择器）

```html
<div class="selector-bar">
    <div class="selector-group">
        <span class="selector-label">量化</span>
        <div class="seg-control" id="quant-selector">
            <div class="seg-btn" data-val="bf16" onclick="setQuant('bf16')">BF16</div>
            <div class="seg-btn" data-val="fp8" onclick="setQuant('fp8')">FP8</div>
        </div>
    </div>
    <div class="selector-group">
        <span class="selector-label">并发</span>
        <div class="seg-control" id="bs-selector">
            <div class="seg-btn" data-val="1" onclick="setBS('1')">1</div>
            <div class="seg-btn" data-val="32" onclick="setBS('32')">32</div>
            <div class="seg-btn" data-val="64" onclick="setBS('64')">64</div>
            <div class="seg-btn" data-val="512" onclick="setBS('512')">512</div>
        </div>
    </div>
</div>
```

### 2. 动态组件

以下组件需在选择变化时动态更新：
- **Badge 行**：量化方式 & 并发数 badge
- **配置卡片**：FD/SG 量化方式 & 并发数字段
- **Params Bar**：请求数（根据数据变化）
- **指标卡片** (8 cards)：值 + diff 百分比 + diff class
- **图表** (4 charts)：全部 destroy + rebuild
- **表格行**：全部 innerHTML 替换
- **结论段落**：根据数据动态生成

### 3. 指标卡片定义

```javascript
const metricDefs = [
    { key: 'total_token_throughput', title: 'Total Token Throughput', unit: 'tok/s', hint: '越高越好', higher: true },
    { key: 'output_token_throughput', title: 'Output Token Throughput', unit: 'tok/s', hint: '越高越好', higher: true },
    { key: 'mean_ttft', title: 'Mean TTFT (首 Token 延迟)', unit: 'ms', hint: '越低越好', higher: false },
    { key: 'mean_tpot', title: 'Mean TPOT (Token 间延迟)', unit: 'ms', hint: '越低越好', higher: false },
    { key: 'mean_itl', title: 'Mean ITL (Inter-Token Latency)', unit: 'ms', hint: '越低越好', higher: false },
    { key: 'mean_e2el', title: 'Mean E2EL (端到端延迟)', unit: 'ms', hint: '越低越好', higher: false },
    { key: 'mean_decode', title: 'Decode Speed', unit: 'tok/s', hint: '越高越好', higher: true },
    { key: 'request_throughput', title: 'Request Throughput', unit: 'req/s', hint: '越高越好', higher: true },
];
```

### 4. 表格指标列表

```javascript
const tableMetrics = [
    { key: 'successful_requests', label: '成功请求数', higher: true },
    { key: 'benchmark_duration', label: '测试总时长 (s)', higher: false },
    { key: 'total_token_throughput', label: '总 Token Throughput (tok/s)', higher: true },
    { key: 'output_token_throughput', label: '输出 Token Throughput (tok/s)', higher: true },
    { key: 'request_throughput', label: 'Request Throughput (req/s)', higher: true },
    { key: 'mean_ttft', label: 'Mean TTFT (ms)', higher: false },
    { key: 'median_ttft', label: 'Median TTFT (ms)', higher: false },
    { key: 'p99_ttft', label: 'P99 TTFT (ms)', higher: false },
    { key: 'mean_tpot', label: 'Mean TPOT (ms)', higher: false },
    { key: 'p99_tpot', label: 'P99 TPOT (ms)', higher: false },
    { key: 'mean_itl', label: 'Mean ITL (ms)', higher: false },
    { key: 'p99_itl', label: 'P99 ITL (ms)', higher: false },
    { key: 'mean_e2el', label: 'Mean E2EL (ms)', higher: false },
    { key: 'p99_e2el', label: 'P99 E2EL (ms)', higher: false },
    { key: 'mean_decode', label: 'Decode Speed (tok/s)', higher: true },
];
```

---

## Diff 计算逻辑

```javascript
function computeDiff(fdVal, sgVal, higherIsBetter) {
    if (higherIsBetter) {
        return sgVal !== 0 ? ((fdVal - sgVal) / sgVal * 100) : 0;
    } else {
        return fdVal !== 0 ? ((sgVal - fdVal) / fdVal * 100) : 0;
    }
}
// 正值 → FD 领先 → 绿色 (m-better / diff-good)
// 负值 → FD 落后 → 红色 (m-worse / diff-bad)
```

---

## 图表规范

4 个 Chart.js 图表：

| 图表 | 类型 | X 轴 | 说明 |
|------|------|------|------|
| 吞吐量对比 | bar | Total Token, Output Token, Request (×scale) | scale 根据 request_throughput 大小决定 |
| 延迟对比 | bar | TTFT, TPOT (×10), ITL (×10), E2EL (/10) | 归一化到同一量级展示 |
| TTFT 分位 | line | Mean, Median, P80, P95, P99 | fill area |
| ITL 分位 | line | Mean, Median, P80, P95, P99 | fill area |

切换主题或切换数据时均需 `destroy()` 旧图表后重建。

---

## CSS：Segmented Control 样式

```css
.selector-bar {
    display: flex; justify-content: center; align-items: center; gap: 24px;
    margin-bottom: 40px; flex-wrap: wrap;
}
.selector-group { display: flex; align-items: center; gap: 10px; }
.selector-label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
.seg-control {
    display: flex; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 4px; gap: 2px;
    transition: background var(--transition), border-color var(--transition);
}
.seg-btn {
    padding: 8px 18px; border-radius: 9px; font-size: 0.82rem; font-weight: 600;
    cursor: pointer; border: none; background: transparent;
    color: var(--text-muted); transition: all 0.2s ease; user-select: none;
}
.seg-btn:hover { color: var(--text-secondary); }
.seg-btn.active {
    background: var(--fd-primary); color: #fff;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
```

---

## 结论动态生成

结论段落由 JS 根据当前数据动态生成，包含：
1. **测试条件**：模型名、GPU、量化方式、并发数
2. **整体表现**：统计 tableMetrics 中各指标胜出方，报告胜出比例
3. **吞吐量**：Total Token Throughput 数值对比和比率
4. **成功率**：FD 成功请求数 / 最大请求数

---

## localStorage 持久化

| Key | 说明 | 默认值 |
|-----|------|--------|
| `bench-quant` | 当前量化选择 | `bf16` |
| `bench-bs` | 当前并发选择 | `512` |
| `benchmark-theme` | 主题 | `light` |

---

## 数据来源

Agent 从以下来源获取 benchmark 指标：
1. **日志文件**：`benchmark_serving.py` 的标准输出文件，使用正则提取数值
2. **metrics.json**：由 `extract_metrics.py` 脚本生成的结构化 JSON
3. **手动提供**：用户直接给出数值

日志文件解析正则模式参考 `extract_metrics.py` 中的 `patterns` 字典。

---

## 设计规范

- **配色方案：**
  - FastDeploy: Indigo 系 (#6366f1 / #4f46e5)
  - SGLang: Amber 系 (#f59e0b / #d97706)
  - FD 领先: Emerald (#10b981)
  - FD 落后: Red (#ef4444)
- **图表：** 切换主题或数据时自动重建（destroy + rebuild）
- **过渡动画：** 0.35s cubic-bezier(0.4,0,0.2,1) 平滑切换
- **响应式：** `@media (max-width: 1000px)` config-grid 单列，`@media (max-width: 900px)` charts-grid 单列
- **字体：** 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'PingFang SC', sans-serif

---

## 完整模板结构（伪代码）

```html
<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
    <!-- Chart.js CDN -->
    <style>/* 全部 CSS 内联，包含 theme variables、selector、cards、table 等 */</style>
</head>
<body>
    <!-- Theme Toggle (fixed, top-right) -->
    <div class="container">
        <!-- Header: h1 + subtitle + badge-row (dynamic badges for quant/bs) -->
        <!-- Selector Bar: quant segmented control + bs segmented control -->
        <!-- Config Cards: Model (static) + FD (dynamic quant/bs) + SG (dynamic quant/bs) -->
        <!-- Params Bar: request count (dynamic) + dataset link + hyperparams -->
        <!-- Legend: FD dot + SG dot -->
        <!-- Metric Cards: 8 cards, innerHTML dynamically generated -->
        <!-- Charts Grid: 4 canvas elements, rebuilt on data/theme change -->
        <!-- Detail Table: thead static, tbody dynamically generated -->
        <!-- Conclusion: dynamically generated -->
        <!-- Footer -->
    </div>
    <script>
        const benchmarkData = { /* ALL scenario data embedded */ };
        let currentQuant = localStorage.getItem('bench-quant') || 'bf16';
        let currentBS = localStorage.getItem('bench-bs') || '512';
        // setQuant(), setBS(), updateAll()
        // updateSelectors(), updateMetricCards(), updateTable(), updateConclusion()
        // Chart functions: getChartColors(), rebuildCharts(), buildCharts()
        // Theme: toggleTheme(), restore from localStorage
        updateAll(); // initial render
    </script>
</body>
</html>
```

---

## 生成报告的步骤

1. 收集所有场景的 benchmark 日志/数据
2. 解析每个场景的 FD + SG 指标（使用 `extract_metrics.py` 或手动正则）
3. 构建 `benchmarkData` JSON 对象
4. 将 JSON 嵌入 HTML 模板的 `<script>` 部分
5. 确认模型信息、配置信息（GPU、TP、Attention、版本等）
6. 输出为 `benchmark_report.html`
