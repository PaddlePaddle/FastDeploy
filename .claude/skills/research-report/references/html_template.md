# HTML Report Template Reference

This file contains the complete CSS, JavaScript, and markup patterns for generating research reports.
Copy the structure below and fill in the content.

---

## Full Report Template

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{REPORT_TITLE}}</title>
  <!-- RESEARCH_META
  title: {{REPORT_TITLE}}
  date: {{YYYY-MM-DD}}
  tags: {{Tag1, Tag2, Tag3}}
  summary: {{One sentence summary for the index page.}}
  -->
  <!-- Mermaid for diagrams -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    /* ── Reset & Base ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:         #0d1117;
      --bg2:        #161b22;
      --bg3:        #1c2128;
      --border:     #30363d;
      --text:       #e6edf3;
      --text2:      #8b949e;
      --text3:      #656d76;
      --cyan:       #39d0d8;
      --blue:       #58a6ff;
      --green:      #3fb950;
      --yellow:     #e3b341;
      --orange:     #f78166;
      --purple:     #bc8cff;
      --red:        #f85149;
      --font-mono:  'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    }
    html { scroll-behavior: smooth; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.75;
      font-size: 16px;
    }

    /* ── Layout ── */
    .page-wrap {
      display: grid;
      grid-template-columns: 240px 1fr;
      min-height: 100vh;
    }
    @media (max-width: 900px) {
      .page-wrap { grid-template-columns: 1fr; }
      .sidebar { display: none; }
    }

    /* ── Sidebar / TOC ── */
    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
      background: var(--bg2);
      border-right: 1px solid var(--border);
      padding: 24px 16px;
    }
    .sidebar-brand {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text3);
      margin-bottom: 16px;
    }
    .toc-list { list-style: none; }
    .toc-item a {
      display: block;
      padding: 5px 8px;
      font-size: 13px;
      color: var(--text2);
      text-decoration: none;
      border-radius: 6px;
      transition: all .15s;
      border-left: 2px solid transparent;
    }
    .toc-item a:hover {
      color: var(--text);
      background: var(--bg3);
      border-left-color: var(--cyan);
    }
    .toc-item.h3 a { padding-left: 20px; font-size: 12px; }

    /* ── Main Content ── */
    .content {
      max-width: 860px;
      margin: 0 auto;
      padding: 48px 32px 80px;
    }
    @media (max-width: 640px) { .content { padding: 24px 16px 60px; } }

    /* ── Cover / Header ── */
    .cover {
      padding: 40px 0 32px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 40px;
    }
    .cover-meta {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 20px;
      letter-spacing: 0.4px;
    }
    .badge-cyan   { background: rgba(57,208,216,.15); color: var(--cyan); }
    .badge-blue   { background: rgba(88,166,255,.15); color: var(--blue); }
    .badge-green  { background: rgba(63,185,80,.15);  color: var(--green); }
    .badge-yellow { background: rgba(227,179,65,.15); color: var(--yellow); }
    .badge-purple { background: rgba(188,140,255,.15);color: var(--purple); }
    .badge-orange { background: rgba(247,129,102,.15);color: var(--orange); }
    .cover h1 {
      font-size: clamp(24px, 4vw, 40px);
      font-weight: 800;
      line-height: 1.2;
      margin-bottom: 12px;
      background: linear-gradient(135deg, #e6edf3 20%, var(--cyan) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .cover-summary {
      font-size: 16px;
      color: var(--text2);
      max-width: 640px;
      line-height: 1.6;
    }
    .cover-date {
      font-size: 12px;
      color: var(--text3);
      margin-top: 12px;
    }

    /* ── Typography ── */
    h2 {
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
      margin: 48px 0 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
      scroll-margin-top: 24px;
    }
    h3 {
      font-size: 17px;
      font-weight: 600;
      color: var(--text);
      margin: 28px 0 12px;
      scroll-margin-top: 24px;
    }
    h4 { font-size: 15px; font-weight: 600; margin: 20px 0 8px; color: var(--text2); }
    p { margin-bottom: 14px; }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
    strong { color: var(--text); font-weight: 600; }
    em { color: var(--yellow); font-style: italic; }

    /* ── TL;DR Box ── */
    .tldr {
      background: linear-gradient(135deg, rgba(57,208,216,.08), rgba(188,140,255,.08));
      border: 1px solid rgba(57,208,216,.25);
      border-radius: 12px;
      padding: 24px 28px;
      margin: 0 0 40px;
    }
    .tldr-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: var(--cyan);
      margin-bottom: 12px;
    }
    .tldr ul { list-style: none; display: flex; flex-direction: column; gap: 8px; }
    .tldr li {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 15px;
      color: var(--text);
    }
    .tldr li::before { content: "→"; color: var(--cyan); flex-shrink: 0; font-weight: 700; }

    /* ── Callout Boxes ── */
    .callout {
      display: flex;
      gap: 14px;
      padding: 16px 20px;
      border-radius: 10px;
      margin: 20px 0;
      border: 1px solid;
    }
    .callout-icon { font-size: 20px; flex-shrink: 0; line-height: 1.4; }
    .callout-body { flex: 1; }
    .callout-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
    .callout-text { font-size: 14px; line-height: 1.65; }

    .callout-insight { background: rgba(57,208,216,.07); border-color: rgba(57,208,216,.3); }
    .callout-insight .callout-title { color: var(--cyan); }
    .callout-warning { background: rgba(227,179,65,.07); border-color: rgba(227,179,65,.3); }
    .callout-warning .callout-title { color: var(--yellow); }
    .callout-danger  { background: rgba(248,81,73,.07);  border-color: rgba(248,81,73,.3); }
    .callout-danger  .callout-title { color: var(--red); }
    .callout-tip     { background: rgba(63,185,80,.07);  border-color: rgba(63,185,80,.3); }
    .callout-tip     .callout-title { color: var(--green); }

    /* ── Metric Cards ── */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }
    .metric-card {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      text-align: center;
    }
    .metric-num {
      font-size: 28px;
      font-weight: 800;
      display: block;
      line-height: 1.1;
    }
    .metric-label { font-size: 12px; color: var(--text2); margin-top: 4px; }
    .metric-card.cyan .metric-num  { color: var(--cyan); }
    .metric-card.blue .metric-num  { color: var(--blue); }
    .metric-card.green .metric-num { color: var(--green); }
    .metric-card.yellow .metric-num{ color: var(--yellow); }

    /* ── Tables ── */
    .table-wrap { overflow-x: auto; margin: 20px 0; border-radius: 10px; border: 1px solid var(--border); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    thead tr { background: var(--bg2); }
    th {
      padding: 12px 16px;
      text-align: left;
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text2);
      border-bottom: 1px solid var(--border);
    }
    td { padding: 12px 16px; border-bottom: 1px solid var(--border); color: var(--text); }
    tbody tr:last-child td { border-bottom: none; }
    tbody tr:hover { background: var(--bg3); }
    .cell-good  { color: var(--green); font-weight: 600; }
    .cell-bad   { color: var(--red);   font-weight: 600; }
    .cell-mid   { color: var(--yellow);font-weight: 600; }
    .cell-mono  { font-family: var(--font-mono); font-size: 12px; }

    /* ── Code ── */
    pre {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      overflow-x: auto;
      margin: 16px 0;
      position: relative;
    }
    code {
      font-family: var(--font-mono);
      font-size: 13px;
      line-height: 1.6;
      color: var(--text);
    }
    p code, li code {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 13px;
      color: var(--cyan);
    }
    .code-label {
      position: absolute;
      top: 10px; right: 14px;
      font-size: 11px;
      color: var(--text3);
      font-family: var(--font-mono);
    }
    /* Simple keyword colors */
    .kw  { color: var(--purple); }
    .fn  { color: var(--blue); }
    .str { color: var(--green); }
    .num { color: var(--orange); }
    .cm  { color: var(--text3); font-style: italic; }
    .tp  { color: var(--yellow); }

    /* ── Mermaid diagrams ── */
    .mermaid-wrap {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 24px;
      margin: 20px 0;
      overflow-x: auto;
    }
    .mermaid { text-align: center; }

    /* ── Lists ── */
    ul, ol { padding-left: 24px; margin-bottom: 14px; }
    li { margin-bottom: 6px; }
    ul li::marker { color: var(--cyan); }
    ol li::marker { color: var(--cyan); font-weight: 600; }

    /* ── Comparison bars ── */
    .bar-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
    .bar-label { width: 140px; font-size: 13px; color: var(--text2); flex-shrink: 0; }
    .bar-track { flex: 1; background: var(--bg3); border-radius: 4px; height: 8px; }
    .bar-fill  { height: 100%; border-radius: 4px; transition: width .4s; }
    .bar-value { width: 48px; text-align: right; font-size: 12px; color: var(--text2); flex-shrink: 0; }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

    /* ── References ── */
    .ref-list { list-style: none; padding: 0; counter-reset: refs; }
    .ref-item {
      counter-increment: refs;
      display: flex;
      gap: 10px;
      padding: 10px 0;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }
    .ref-item::before {
      content: "[" counter(refs) "]";
      color: var(--cyan);
      font-weight: 700;
      font-size: 12px;
      flex-shrink: 0;
      margin-top: 2px;
    }

    /* ── Back to top ── */
    .back-top {
      position: fixed;
      bottom: 24px; right: 24px;
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: 50%;
      width: 40px; height: 40px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
      cursor: pointer;
      text-decoration: none;
      color: var(--text2);
      transition: all .2s;
      opacity: 0.7;
    }
    .back-top:hover { opacity: 1; border-color: var(--cyan); color: var(--cyan); }
  </style>
</head>
<body>
<script>mermaid.initialize({ theme: 'dark', startOnLoad: true });</script>

<div class="page-wrap">

  <!-- ── Sidebar TOC ── -->
  <nav class="sidebar">
    <div class="sidebar-brand">📋 目录</div>
    <ul class="toc-list">
      <!-- Generate one <li class="toc-item"> per section heading -->
      <li class="toc-item"><a href="#tldr">TL;DR</a></li>
      <li class="toc-item"><a href="#background">背景</a></li>
      <li class="toc-item h3"><a href="#background-sub1">子节标题</a></li>
      <li class="toc-item"><a href="#concepts">核心知识点</a></li>
      <li class="toc-item"><a href="#comparison">对比分析</a></li>
      <li class="toc-item"><a href="#conclusions">结论与建议</a></li>
      <li class="toc-item"><a href="#references">参考资料</a></li>
    </ul>
  </nav>

  <!-- ── Main Content ── -->
  <main class="content">

    <!-- Cover -->
    <div class="cover">
      <div class="cover-meta">
        <span class="badge badge-cyan">{{Tag1}}</span>
        <span class="badge badge-blue">{{Tag2}}</span>
        <span class="badge badge-purple">{{Tag3}}</span>
      </div>
      <h1>{{Report Title Here}}</h1>
      <p class="cover-summary">{{One to two sentence overview of what this report covers and why it matters.}}</p>
      <div class="cover-date">📅 {{YYYY-MM-DD}} &nbsp;·&nbsp; Ibin! Research Notes</div>
    </div>

    <!-- TL;DR -->
    <div class="tldr" id="tldr">
      <div class="tldr-title">⚡ TL;DR — 核心要点</div>
      <ul>
        <li>{{Key takeaway 1 — concrete and specific}}</li>
        <li>{{Key takeaway 2 — the most surprising or important finding}}</li>
        <li>{{Key takeaway 3 — actionable recommendation}}</li>
        <li>{{Key takeaway 4 (optional)}}</li>
      </ul>
    </div>

    <!-- ═══════════════════════════════════════════ -->
    <!-- Section: Background -->
    <h2 id="background">📖 背景 / Background</h2>
    <p>{{Introductory paragraph explaining why this topic matters and what problem it solves.}}</p>

    <h3 id="background-sub1">{{Sub-section Title}}</h3>
    <p>{{Content...}}</p>

    <!-- Example: Callout box -->
    <div class="callout callout-insight">
      <div class="callout-icon">💡</div>
      <div class="callout-body">
        <div class="callout-title">Key Insight</div>
        <div class="callout-text">{{Highlight a particularly important or non-obvious point here.}}</div>
      </div>
    </div>

    <!-- ═══════════════════════════════════════════ -->
    <!-- Section: Key Concepts -->
    <h2 id="concepts">🧠 核心知识点 / Key Concepts</h2>

    <!-- Example: Metric cards -->
    <div class="metric-grid">
      <div class="metric-card cyan">
        <span class="metric-num">{{Value}}</span>
        <div class="metric-label">{{Metric Name}}</div>
      </div>
      <div class="metric-card green">
        <span class="metric-num">{{Value}}</span>
        <div class="metric-label">{{Metric Name}}</div>
      </div>
      <div class="metric-card yellow">
        <span class="metric-num">{{Value}}</span>
        <div class="metric-label">{{Metric Name}}</div>
      </div>
    </div>

    <!-- Example: Mermaid diagram -->
    <div class="mermaid-wrap">
      <div class="mermaid">
flowchart LR
    A[Input Q/K/V] --> B[Split into Blocks]
    B --> C[Compute Attention per Block]
    C --> D[Online Softmax]
    D --> E[Accumulate Output]
      </div>
    </div>

    <!-- Example: Code block -->
    <pre><code><span class="cm"># Example code block</span>
<span class="kw">def</span> <span class="fn">flash_attention</span>(Q, K, V):
    <span class="cm"># Block size tuned for SRAM</span>
    Bc = <span class="num">64</span>
    <span class="kw">for</span> block <span class="kw">in</span> split(K, V, Bc):
        <span class="kw">return</span> online_softmax_accumulate(Q, block)
<span class="code-label">Python</span></code></pre>

    <!-- Example: Warning callout -->
    <div class="callout callout-warning">
      <div class="callout-icon">⚠️</div>
      <div class="callout-body">
        <div class="callout-title">注意事项</div>
        <div class="callout-text">{{Gotcha, limitation, or common mistake to avoid.}}</div>
      </div>
    </div>

    <!-- ═══════════════════════════════════════════ -->
    <!-- Section: Comparison -->
    <h2 id="comparison">📊 对比分析 / Comparison</h2>

    <!-- Example: Comparison table -->
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>方案 / Method</th>
            <th>内存占用</th>
            <th>计算复杂度</th>
            <th>实现难度</th>
            <th>适用场景</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>{{Method A}}</strong></td>
            <td class="cell-good">低</td>
            <td class="cell-mid">O(n)</td>
            <td class="cell-bad">高</td>
            <td>长序列解码</td>
          </tr>
          <tr>
            <td><strong>{{Method B}}</strong></td>
            <td class="cell-bad">高</td>
            <td class="cell-good">O(n²)</td>
            <td class="cell-good">低</td>
            <td>短序列 Prefill</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Example: Bar chart comparison -->
    <h3>性能对比</h3>
    <div class="bar-row">
      <div class="bar-label">{{Method A}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:85%; background:var(--cyan)"></div></div>
      <div class="bar-value">85%</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">{{Method B}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:60%; background:var(--blue)"></div></div>
      <div class="bar-value">60%</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">{{Baseline}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:40%; background:var(--text3)"></div></div>
      <div class="bar-value">40%</div>
    </div>

    <!-- ═══════════════════════════════════════════ -->
    <!-- Section: Conclusions -->
    <h2 id="conclusions">✅ 结论与建议 / Conclusions</h2>
    <p>{{Summary of the main findings and what they mean for the reader.}}</p>

    <div class="callout callout-tip">
      <div class="callout-icon">🚀</div>
      <div class="callout-body">
        <div class="callout-title">推荐行动</div>
        <div class="callout-text">{{What should the reader DO based on these findings? Be specific.}}</div>
      </div>
    </div>

    <hr />

    <!-- References -->
    <h2 id="references">📚 参考资料 / References</h2>
    <ol class="ref-list">
      <li class="ref-item"><a href="{{URL1}}" target="_blank">{{Paper/Article Title 1}}</a> — {{Brief note}}</li>
      <li class="ref-item"><a href="{{URL2}}" target="_blank">{{Paper/Article Title 2}}</a> — {{Brief note}}</li>
      <li class="ref-item"><a href="{{URL3}}" target="_blank">{{Paper/Article Title 3}}</a> — {{Brief note}}</li>
    </ol>

  </main>
</div>

<a class="back-top" href="#" title="回到顶部">↑</a>

</body>
</html>
```

---

## Callout Variants Quick Reference

| Class | Icon | Use for |
|---|---|---|
| `callout-insight` | 💡 | Key insights, non-obvious findings |
| `callout-warning` | ⚠️ | Gotchas, limitations, caveats |
| `callout-danger`  | 🚨 | Critical errors, security issues |
| `callout-tip`     | 🚀 | Action items, recommendations |

## Badge Colors Quick Reference

Use badge colors to categorize the topic domain:
- `badge-cyan` — Core concept / main topic
- `badge-blue` — Related technology / framework
- `badge-green` — Performance / optimization
- `badge-yellow` — Caution / experimental
- `badge-purple` — Research / theory
- `badge-orange` — Hardware / systems

## Mermaid Diagram Examples

**Flowchart:**
```
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Result A]
    B -->|No| D[Result B]
```

**Sequence diagram:**
```
sequenceDiagram
    User->>Model: Prompt
    Model->>KVCache: Lookup prefix
    KVCache-->>Model: Cache hit
    Model-->>User: Fast response
```

**Architecture diagram:**
```
graph LR
    subgraph GPU["GPU (A100)"]
        SM[Streaming Multiprocessor]
        SRAM[Shared Memory 192KB]
    end
    HBM[HBM2e 80GB] --> SM
    SM <--> SRAM
```

## Notes for Report Generation

1. **Always fill ALL `{{placeholders}}`** — never leave template placeholders in the output
2. **Sidebar TOC** — update `toc-list` to match actual section IDs in the document
3. **Mermaid** — only include the `<script>` tag and `mermaid.initialize()` if you're using diagrams; otherwise omit for faster loading
4. **Long reports** — for reports with many sections, add a progress indicator or section count to the cover
5. **Self-contained** — the only external dependency allowed is Mermaid CDN; all CSS is inline
