---
name: research-report
description: >
Generate a beautiful, self-contained HTML research report from any research question or topic. Use this skill whenever the user wants to: research a technical concept, compare technologies, summarize findings on a topic, create a visual knowledge document, or compile study notes into a polished HTML page. Also handles organizing past reports into a browsable research index website. Trigger this skill for prompts like "研究一下X", "调研Y", "帮我生成一份关于Z的HTML报告", "把我的研究记录整理成网站", "research X and make a report", or any request to investigate a topic and produce a visual summary document.
---

# Research Report Generator

Turn any research question into a polished, visually rich **self-contained HTML report**, and maintain a browsable **research index** of all past reports.

---

## Workflow

Follow these four phases for every research request:

### Phase 1 — Clarify & Scope (30 seconds)

Before diving in, confirm two things (can be inferred from context, no need to ask if obvious):
- **Topic**: What exactly to research (narrow or broad)
- **Save location**: Where to store the report and index — default is `~/Research/` unless the user specifies. Always ask if unclear.

If the user just wants the HTML in the conversation without saving, that's fine too.

### Phase 2 — Research

Use all available tools:
- **Web search** for up-to-date information, papers, benchmarks, comparisons
- **Prior knowledge** for foundational concepts
- **Uploaded files** if the user provides source material

Aim for depth over breadth. Gather enough to write 4–6 substantive sections. Track your sources — you'll cite them.

### Phase 3 — Generate HTML Report

Write a single, self-contained `.html` file using the template and style guide in `references/html_template.md`.

**Naming convention**: `YYYY-MM-DD_topic-slug.html`
Example: `2026-04-09_flash-attention-v2-vs-v3.html`

Embed report metadata in the file header (the index script reads this):
```html
<!-- RESEARCH_META
title: Flash Attention v2 vs v3: Deep Comparison
date: 2026-04-09
tags: CUDA, Attention, GPU, Performance
summary: A deep dive comparing Flash Attention v2 and v3 across throughput, memory footprint, and implementation complexity.
-->
```

**Required sections** (adapt headings to the topic):
1. **封面/Header** — Title, date, topic badges, one-line summary
2. **TL;DR** — 3–5 bullet key takeaways (this is the most important section for busy readers)
3. **背景 / Background** — Necessary context; assume reader is smart but may not know this domain
4. **核心知识点 / Key Concepts** — The meaty technical content; use subsections, tables, diagrams
5. **对比分析 / Comparison** (if applicable) — Side-by-side tables, pros/cons
6. **结论与建议 / Conclusions** — Actionable recommendations; what the reader should do with this knowledge
7. **参考资料 / References** — Numbered list with links

**Visual elements to use liberally** (see `references/html_template.md` for markup):
- 💡 Callout boxes for key insights
- ⚠️ Warning/caution boxes for gotchas
- 📊 Comparison tables with clear headers
- 🔢 Metric highlight cards (big number + label)
- Code blocks with syntax highlighting
- Mermaid diagrams for flows and architectures (loaded from CDN)
- Progress bars or visual scales for performance comparisons

Good research reports feel like a skilled colleague who spent a week on a topic explaining it to you in an afternoon. Prioritize clarity and insight over exhaustiveness.

### Phase 4 — Update Research Index

After saving the report, update (or create) `index.html` in the same directory using `scripts/update_index.py`:

```bash
python3 scripts/update_index.py /path/to/research/directory
```

This scans all `.html` reports, extracts their metadata, and regenerates the index page. Tell the user where both files are saved.

---

## Handling "Update Index Only" Requests

If the user says something like "帮我整理一下我的研究记录" or "更新一下研究索引":
1. Ask for (or confirm) the research directory path
2. Run `python3 scripts/update_index.py <path>` directly
3. Report back how many reports were found and indexed

---

## Quality bar

A great report:
- Has an instantly useful TL;DR — someone skimming it in 30 seconds learns something
- Uses concrete examples, numbers, and comparisons rather than vague statements
- Looks beautiful and professional (follow the HTML template closely)
- Is completely self-contained — works offline, no broken links to external CSS
- Cites real sources with working URLs

A bad report:
- Is just a wall of text with no visual structure
- Has a TL;DR that just restates the title
- Has placeholder diagrams or "coming soon" sections
- Doesn't actually answer the user's question

Read `references/html_template.md` before generating any HTML — it contains the full CSS and markup patterns to copy.
Read `references/usage_guide.md` for prompting tips and worked examples to share with the user.
