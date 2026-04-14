#!/usr/bin/env python3
"""
update_index.py — Research Report Index Generator
===================================================
用法：
    python3 update_index.py /path/to/research/directory
    python3 update_index.py .                             # 当前目录
    python3 update_index.py ~/Research --output ~/Research/index.html

功能：
    扫描目录中所有 .html 研究报告，提取元数据（标题、日期、标签、摘要），
    生成一个美观的可浏览索引页面 index.html。
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ─── Metadata extractor ──────────────────────────────────────
def extract_meta(filepath: Path) -> dict:
    """Extract RESEARCH_META block from HTML comment."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")[:6000]
    except Exception:
        return {}

    # Try structured meta block
    m = re.search(r"<!--\s*RESEARCH_META\s*(.*?)\s*-->", content, re.DOTALL)
    if m:
        meta = {}
        for line in m.group(1).strip().splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip().lower()] = val.strip()
        if meta.get("title"):
            return meta

    # Fallback: extract from <title> tag
    t = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    title = re.sub(r"<[^>]+>", "", t.group(1)).strip() if t else filepath.stem

    # Try to extract date from filename (YYYY-MM-DD_slug.html)
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", filepath.name)
    date = date_match.group(1) if date_match else ""

    return {"title": title, "date": date, "tags": "", "summary": ""}


def parse_tags(tags_str: str) -> list[str]:
    if not tags_str:
        return []
    return [t.strip() for t in re.split(r"[,，]", tags_str) if t.strip()]


# ─── Index page generator ────────────────────────────────────
CSS = """
  :root {
    --bg:#0d1117; --bg2:#161b22; --bg3:#1c2128; --border:#30363d;
    --text:#e6edf3; --text2:#8b949e; --text3:#656d76;
    --cyan:#39d0d8; --blue:#58a6ff; --green:#3fb950; --yellow:#e3b341;
    --purple:#bc8cff; --orange:#f78166;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',sans-serif;
       background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh;}
  header{background:var(--bg2);border-bottom:1px solid var(--border);
         padding:0 24px;position:sticky;top:0;z-index:100;}
  .header-inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;
                justify-content:space-between;height:56px;gap:16px;}
  .logo{font-size:16px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:8px;}
  .logo-icon{background:linear-gradient(135deg,var(--cyan),var(--purple));
             border-radius:8px;width:28px;height:28px;display:flex;
             align-items:center;justify-content:center;font-size:14px;}
  .search{flex:1;max-width:400px;position:relative;}
  .search input{width:100%;background:var(--bg3);border:1px solid var(--border);
                border-radius:8px;padding:7px 12px 7px 34px;color:var(--text);
                font-size:14px;outline:none;transition:border-color .2s;}
  .search input:focus{border-color:var(--cyan);}
  .search input::placeholder{color:var(--text3);}
  .search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);
               color:var(--text3);font-size:13px;}
  .hero{padding:48px 24px 32px;text-align:center;background:radial-gradient(ellipse 80% 50% at 50% 0%,rgba(57,208,216,.07),transparent 70%);}
  .hero h1{font-size:clamp(24px,4vw,40px);font-weight:800;
           background:linear-gradient(135deg,#e6edf3 30%,var(--cyan));
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
           margin-bottom:10px;}
  .hero p{color:var(--text2);font-size:15px;}
  .stats{display:flex;gap:24px;justify-content:center;margin-top:20px;flex-wrap:wrap;}
  .stat{text-align:center;}
  .stat-n{font-size:26px;font-weight:800;color:var(--cyan);display:block;}
  .stat-l{font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;}
  main{max-width:1100px;margin:0 auto;padding:32px 24px 80px;}
  /* Tags filter */
  .tag-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:24px;align-items:center;}
  .tag-btn{background:var(--bg2);border:1px solid var(--border);border-radius:20px;
           padding:4px 14px;font-size:12px;color:var(--text2);cursor:pointer;
           transition:all .15s;white-space:nowrap;}
  .tag-btn:hover,.tag-btn.active{background:rgba(57,208,216,.15);
                                  border-color:rgba(57,208,216,.4);color:var(--cyan);}
  .tag-label{font-size:12px;color:var(--text3);margin-right:4px;}
  /* Cards */
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;}
  .card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
        padding:20px;text-decoration:none;color:inherit;display:flex;flex-direction:column;
        gap:10px;transition:all .2s;position:relative;overflow:hidden;}
  .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,var(--cyan),var(--purple));
                opacity:0;transition:opacity .2s;}
  .card:hover{border-color:#6e7681;background:var(--bg3);transform:translateY(-2px);
              box-shadow:0 8px 24px rgba(0,0,0,.3);}
  .card:hover::before{opacity:1;}
  .card-date{font-size:11px;color:var(--text3);}
  .card-title{font-size:15px;font-weight:700;color:var(--text);line-height:1.35;}
  .card-summary{font-size:13px;color:var(--text2);line-height:1.6;flex:1;}
  .card-footer{display:flex;align-items:center;justify-content:space-between;margin-top:4px;}
  .card-tags{display:flex;gap:6px;flex-wrap:wrap;}
  .tag{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:500;
       background:rgba(57,208,216,.12);color:var(--cyan);}
  .card-arrow{color:var(--text3);font-size:16px;transition:transform .2s;}
  .card:hover .card-arrow{transform:translateX(4px);color:var(--cyan);}
  .no-results{text-align:center;padding:64px;color:var(--text3);display:none;font-size:15px;}
  footer{border-top:1px solid var(--border);padding:20px;text-align:center;
         font-size:12px;color:var(--text3);}
  footer a{color:var(--blue);text-decoration:none;}
"""

CARD_TEMPLATE = """
    <a class="card" href="{href}" target="_blank" data-tags="{tags_data}" data-title="{title_lower}">
      <div class="card-date">📅 {date}</div>
      <div class="card-title">{title}</div>
      <div class="card-summary">{summary}</div>
      <div class="card-footer">
        <div class="card-tags">{tags_html}</div>
        <span class="card-arrow">→</span>
      </div>
    </a>"""

JS = """
  const cards = document.querySelectorAll('.card');
  let activeTag = 'all';
  function render() {
    const q = document.getElementById('search').value.toLowerCase();
    let vis = 0;
    cards.forEach(c => {
      const tagMatch = activeTag === 'all' || c.dataset.tags.includes(activeTag);
      const textMatch = !q || c.dataset.title.includes(q) || c.dataset.tags.includes(q);
      const show = tagMatch && textMatch;
      c.style.display = show ? '' : 'none';
      if (show) vis++;
    });
    document.getElementById('noResults').style.display = vis === 0 ? 'block' : 'none';
  }
  document.getElementById('search').addEventListener('input', render);
  document.querySelectorAll('.tag-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tag-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeTag = btn.dataset.tag;
      render();
    });
  });
"""


def generate_index(reports: list[dict], output_path: Path, dir_path: Path):
    total = len(reports)
    all_tags = sorted({t for r in reports for t in r["tags_list"]})

    # Build tag filter buttons
    tag_btns = '<button class="tag-btn active" data-tag="all">全部</button>\n'
    for t in all_tags:
        tag_btns += f'    <button class="tag-btn" data-tag="{t}">{t}</button>\n'

    # Build cards
    cards_html = ""
    for r in reports:
        href = os.path.relpath(r["path"], output_path.parent)
        tags_html = "".join(f'<span class="tag">{t}</span>' for t in r["tags_list"])
        cards_html += CARD_TEMPLATE.format(
            href=href,
            tags_data=" ".join(r["tags_list"]).lower(),
            title_lower=r["title"].lower(),
            date=r["date"] or "未知日期",
            title=r["title"],
            summary=r["summary"] or "点击查看完整报告 →",
            tags_html=tags_html,
        )

    # Build stats
    latest = reports[0]["date"] if reports else "—"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Ibin! 研究笔记索引</title>
  <style>{CSS}</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="logo"><div class="logo-icon">🔬</div>Research Notes</div>
    <div class="search">
      <span class="search-icon">🔍</span>
      <input id="search" type="text" placeholder="搜索报告…" />
    </div>
  </div>
</header>

<div class="hero">
  <h1>📚 Ibin! 研究笔记库</h1>
  <p>AI 推理、GPU 优化、大模型加速领域的深度调研报告合集</p>
  <div class="stats">
    <div class="stat"><span class="stat-n">{total}</span><span class="stat-l">篇报告</span></div>
    <div class="stat"><span class="stat-n">{len(all_tags)}</span><span class="stat-l">技术标签</span></div>
    <div class="stat"><span class="stat-n">{latest}</span><span class="stat-l">最近更新</span></div>
  </div>
</div>

<main>
  <div class="tag-bar">
    <span class="tag-label">筛选：</span>
    {tag_btns}
  </div>
  <div class="cards">
{cards_html}
  </div>
  <div class="no-results" id="noResults">🔍 没有找到匹配的报告</div>
</main>

<footer>
  <p>Built by <strong>Ibin!</strong> · 最后生成时间：{datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')} CST</p>
</footer>

<script>{JS}</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"✅ 索引已生成: {output_path}  ({total} 篇报告)")


# ─── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="生成研究报告索引页面")
    parser.add_argument("directory", help="研究报告所在目录", nargs="?", default=".")
    parser.add_argument("--output", "-o", help="输出 index.html 路径（默认在目录内）")
    args = parser.parse_args()

    dir_path = Path(args.directory).expanduser().resolve()
    if not dir_path.is_dir():
        print(f"❌ 目录不存在：{dir_path}")
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve() if args.output else dir_path / "index.html"

    # Scan for HTML files
    html_files = sorted(
        [f for f in dir_path.glob("*.html") if f.name != "index.html"],
        reverse=True,  # newest first (assumes YYYY-MM-DD prefix)
    )

    if not html_files:
        print(f"⚠️  目录 {dir_path} 中没有找到 HTML 报告文件")
        sys.exit(0)

    print(f"📁 扫描目录: {dir_path}")
    print(f"📄 找到 {len(html_files)} 个报告文件\n")

    reports = []
    for f in html_files:
        meta = extract_meta(f)
        tags_list = parse_tags(meta.get("tags", ""))
        reports.append(
            {
                "path": f,
                "title": meta.get("title", f.stem),
                "date": meta.get("date", ""),
                "tags": meta.get("tags", ""),
                "tags_list": tags_list,
                "summary": meta.get("summary", ""),
            }
        )
        print(f"  ✓ {f.name}")
        print(f"    标题: {meta.get('title', '—')}")
        print(f"    日期: {meta.get('date', '—')}")
        print(f"    标签: {meta.get('tags', '—')}\n")

    generate_index(reports, output_path, dir_path)
    print(f"\n🌐 在浏览器中打开: file://{output_path}")


if __name__ == "__main__":
    main()
