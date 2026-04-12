#!/usr/bin/env python3
"""
stat_cache_hitrate — FastDeploy Go Router Cache 命中率统计工具

统计三层 cache 命中率指标：
  1. Prefix Hit Ratio  — KV Cache 内容复用度
  2. Session Hit Rate   — 请求级路由粘性
  3. Per-Worker Stats   — 各 worker 缓存利用排名

用法：
  python3 stat_cache_hitrate.py <log_file> [--tail N|Nm] [--watch] [--output DIR]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote
from collections import defaultdict
from datetime import datetime

# 同目录模块导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart import render_bar, render_sparkline, render_table
from log_parser import (
    complete_time_arg,
    extract_ts,
    filter_file_by_time_range,
    parse_cache_strategy_line,
    parse_stats_line,
    parse_ts,
)
from session_analysis import compute_session_details, summarize_session_details
from stats import compute_statistics, count_by, time_bucket
from window_utils import merge_blank_window_rows


def _strip_scheme(url):
    return re.sub(r"^https?://", "", url)


def _build_path_links(path):
    """返回绝对路径与 file URI，兼容空格/中文路径。"""
    abs_path = str(Path(path).resolve())
    file_uri = "file://" + quote(abs_path, safe="/:-._~")
    return abs_path, file_uri

# ════════════════════════════════════════════════════════════════
# Phase 1: 日志读取
# ════════════════════════════════════════════════════════════════


def count_lines(filepath):
    """快速统计文件行数。"""
    result = subprocess.run(["wc", "-l", filepath], capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.strip().split()[0])
    return 0


def read_lines(filepath, tail=None):
    """读取日志文件，支持 tail 模式。"""
    if tail:
        if isinstance(tail, str) and tail.endswith("m"):
            # 按时间 tail：读取全部行，过滤最近 N 分钟
            minutes = int(tail[:-1])
            all_lines = _read_file_lines(filepath)
            return _filter_by_time(all_lines, minutes)
        else:
            # 按行数 tail
            n = int(tail)
            result = subprocess.run(["tail", "-n", str(n), filepath], capture_output=True, text=True)
            return result.stdout.splitlines() if result.returncode == 0 else []
    return _read_file_lines(filepath)


def _read_file_lines(filepath):
    with open(filepath, "r", errors="replace") as f:
        return f.readlines()


def _filter_by_time(lines, minutes):
    """过滤最近 N 分钟的日志行。"""
    # 找最后一行的时间戳作为基准
    last_ts = None
    for line in reversed(lines):
        ts = extract_ts(line)
        if ts:
            last_ts = parse_ts(ts)
            break
    if not last_ts:
        return lines

    from datetime import timedelta

    cutoff = last_ts - timedelta(minutes=minutes)
    result = []
    for line in lines:
        ts = extract_ts(line)
        if ts:
            try:
                if parse_ts(ts) >= cutoff:
                    result.append(line)
            except ValueError:
                result.append(line)
        else:
            result.append(line)
    return result


# ════════════════════════════════════════════════════════════════
# Phase 2: 日志提取与解析
# ════════════════════════════════════════════════════════════════

STRATEGY_PATTERN = "cache-aware prefill: final strategy:"
STATS_PATTERN = "[stats]"
INFERENCE_PATTERNS = ["] [POST] /v1/chat/completions ", "] [POST] /v1/completions "]


def _shell_quote(s):
    """Shell 引号转义，安全处理含空格、括号、单引号的路径。"""
    return "'" + s.replace("'", "'\\''") + "'"


def grep_and_parse(filepath, grep_pattern, parse_cmd, tail=None):
    """大文件模式：grep 过滤 + log_parser.py CLI 管道解析。"""
    parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_parser.py")

    if tail and not (isinstance(tail, str) and tail.endswith("m")):
        grep_cmd = f"tail -n {tail} {_shell_quote(filepath)} | grep -F {_shell_quote(grep_pattern)} | python3 {_shell_quote(parser_path)} {parse_cmd}"
    else:
        grep_cmd = f"grep -F {_shell_quote(grep_pattern)} {_shell_quote(filepath)} | python3 {_shell_quote(parser_path)} {parse_cmd}"

    result = subprocess.run(grep_cmd, shell=True, capture_output=True, text=True)
    records = []
    for line in result.stdout.strip().splitlines():
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def grep_count(filepath, grep_pattern, tail=None):
    """大文件模式：grep 计数。"""
    if tail and not (isinstance(tail, str) and tail.endswith("m")):
        cmd = f"tail -n {tail} {_shell_quote(filepath)} | grep -cE {_shell_quote(grep_pattern)}"
    else:
        cmd = f"grep -cE {_shell_quote(grep_pattern)} {_shell_quote(filepath)}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def extract_data(filepath, tail=None):
    """提取并解析日志数据，根据文件大小自动选择策略。"""
    total = count_lines(filepath)

    if total < 5000:
        # 小文件：内存中处理
        lines = read_lines(filepath, tail)
        strategy_recs = [r for l in lines if (r := parse_cache_strategy_line(l)) is not None]
        stats_recs = [r for l in lines if (r := parse_stats_line(l)) is not None]
        inference_count = sum(1 for l in lines if any(p in l for p in INFERENCE_PATTERNS))
        return strategy_recs, stats_recs, inference_count, len(lines)
    else:
        # 大文件：grep + subprocess
        strategy_recs = grep_and_parse(filepath, STRATEGY_PATTERN, "parse-cache-strategy", tail)
        stats_recs = grep_and_parse(filepath, STATS_PATTERN, "parse-stats", tail)
        inference_count = grep_count(filepath, r"\] \[POST\] /v1/chat/completions |\] \[POST\] /v1/completions ", tail)
        line_count = int(tail) if tail and not (isinstance(tail, str) and tail.endswith("m")) else total
        return strategy_recs, stats_recs, inference_count, line_count


# ════════════════════════════════════════════════════════════════
# Phase 3: 三层指标计算
# ════════════════════════════════════════════════════════════════


def compute_prefix_hitrate(strategies):
    """计算第一层：Prefix Hit Ratio。"""
    scoring_recs = [r for r in strategies if r.get("strategy") == "cache_aware_scoring"]
    if not scoring_recs:
        return {"mean": 0, "stats": None, "distribution": [], "cold_start_rate": 0, "trend": [], "count": 0}

    hit_ratios = [r.get("selected_hitRatio", 0) for r in scoring_recs]
    cold_starts = sum(1 for r in scoring_recs if not r.get("hitRatios"))

    stats = compute_statistics(hit_ratios, distribution_spec="0-20,20-40,40-60,60-80,80-100")
    trend = time_bucket(scoring_recs, "auto", [("selected_hitRatio", "mean")])

    return {
        "mean": stats["mean"],
        "stats": stats,
        "distribution": stats.get("distribution", []),
        "cold_start_rate": round(cold_starts / len(scoring_recs) * 100, 1) if scoring_recs else 0,
        "trend": trend,
        "count": len(scoring_recs),
    }


def compute_session_hitrate(stats_recs, inference_count):
    """计算第二层：Session Hit Rate。"""
    total_hits = sum(r.get("hits", 0) for r in stats_recs)
    total_total = sum(r.get("total", 0) for r in stats_recs)

    session_hr = round(total_hits / total_total * 100, 1) if total_total else 0
    coverage = round(total_total / inference_count * 100, 1) if inference_count else 0

    # 趋势：每个窗口的 hits/total
    trend = time_bucket(stats_recs, "auto", [("hits", "sum"), ("total", "sum")])
    for t in trend:
        h = t.get("hits_sum", 0)
        tot = t.get("total_sum", 0)
        t["value"] = round(h / tot * 100, 1) if tot else 0

    return {
        "rate": session_hr,
        "hits": total_hits,
        "total": total_total,
        "coverage": coverage,
        "inference_count": inference_count,
        "trend": trend,
    }


def compute_per_worker_stats(strategies):
    """计算第三层：Per-Worker Cache Stats。"""
    scoring_recs = [r for r in strategies if r.get("strategy") == "cache_aware_scoring"]
    if not scoring_recs:
        return []

    worker_data = defaultdict(lambda: {"selected_count": 0, "hit_ratios": []})
    total_scoring = len(scoring_recs)

    for r in scoring_recs:
        selected = r.get("selected", "")
        if selected:
            worker_data[selected]["selected_count"] += 1
            worker_data[selected]["hit_ratios"].append(r.get("selected_hitRatio", 0))

    result = []
    for worker, data in worker_data.items():
        avg_hr = round(sum(data["hit_ratios"]) / len(data["hit_ratios"]), 1) if data["hit_ratios"] else 0
        result.append(
            {
                "Worker": _strip_scheme(worker),
                "Selected": data["selected_count"],
                "Select%": f"{round(data['selected_count'] / total_scoring * 100, 1)}%",
                "AvgHitRatio": f"{avg_hr}%",
            }
        )

    result.sort(key=lambda x: x["Selected"], reverse=True)
    return result


def compute_scheduling_stats(strategies):
    """计算调度策略概况。"""
    if not strategies:
        return {"scoring_count": 0, "fallback_count": 0, "scoring_pct": 0, "fallback_reasons": [], "suboptimal_pct": 0}

    scoring = [r for r in strategies if r.get("strategy") == "cache_aware_scoring"]
    fallback = [r for r in strategies if r.get("strategy") == "process_tokens"]

    # Fallback 原因分类
    fallback_reasons = count_by(fallback, "reason") if fallback else []

    # 非最优命中选择比例
    suboptimal = 0
    for r in scoring:
        hit_ratios = r.get("hitRatios", {})
        if not hit_ratios:
            continue
        selected_hr = r.get("selected_hitRatio", 0)
        max_hr = max(hit_ratios.values()) if hit_ratios else 0
        if selected_hr < max_hr:
            suboptimal += 1

    total = len(strategies)
    return {
        "scoring_count": len(scoring),
        "fallback_count": len(fallback),
        "scoring_pct": round(len(scoring) / total * 100, 1) if total else 0,
        "fallback_reasons": fallback_reasons,
        "suboptimal_count": suboptimal,
        "suboptimal_pct": round(suboptimal / len(scoring) * 100, 1) if scoring else 0,
    }


def cross_diagnose(prefix_hr, session_hr):
    """交叉诊断矩阵。"""
    p_high = prefix_hr["mean"] >= 60
    s_high = session_hr["rate"] >= 60

    if s_high and p_high:
        return {
            "icon": "\u2705",
            "summary": "cache-aware 策略运行良好",
            "detail": "Session 粘性好，KV cache 实际复用度高",
        }
    elif s_high and not p_high:
        return {
            "icon": "\u26a0\ufe0f",
            "summary": "Session 粘性好但 Prefix HR 低",
            "detail": "prompt 内容变化大，同 worker 的 KV cache 实际复用低",
        }
    elif not s_high and p_high:
        return {
            "icon": "\u26a0\ufe0f",
            "summary": "换 worker 频繁但 Prefix HR 尚可",
            "detail": "负载均衡分散了请求，但新 worker 也有类似前缀缓存",
        }
    else:
        return {
            "icon": "\u274c",
            "summary": "命中率全面偏低",
            "detail": "负载均衡强制分散或缓存未预热，建议检查 worker 数量和 session 分配策略",
        }


# ════════════════════════════════════════════════════════════════
# Phase 4: 报告格式化
# ════════════════════════════════════════════════════════════════


def _quartile_trend(trend, value_field):
    """将趋势数据分为 4 个 quartile，计算每段均值。"""
    if not trend:
        return ""
    n = len(trend)
    if n < 4:
        values = [t.get(value_field, 0) for t in trend]
        avg = round(sum(values) / len(values), 1) if values else 0
        return f"{avg}%"

    q_size = n // 4
    quartiles = []
    for i in range(4):
        start = i * q_size
        end = start + q_size if i < 3 else n
        vals = [t.get(value_field, 0) for t in trend[start:end]]
        quartiles.append(round(sum(vals) / len(vals), 1) if vals else 0)

    arrow = (
        "\u2191" if quartiles[3] > quartiles[0] + 10 else "\u2193" if quartiles[3] < quartiles[0] - 10 else "\u2192"
    )
    return f"Q1={quartiles[0]}% \u2192 Q2={quartiles[1]}% \u2192 Q3={quartiles[2]}% \u2192 Q4={quartiles[3]}% {arrow}"


def format_full_report(filepath, line_count, prefix_hr, session_hr, per_worker, scheduling, diagnosis, time_span=None, window_rows=None):
    """格式化完整终端报告。"""
    parts = []

    # 标题
    span_str = time_span or ""
    parts.append("## Cache Hit Rate Report")
    parts.append(f"**File**: {filepath} | **Lines**: {line_count:,}")
    if span_str:
        parts.append(f"**Span**: {span_str}")
    parts.append("")

    # 图表说明
    parts.append("### 图表说明（如何解读）")
    parts.append("  - Unicode 柱状图：每行代表一个 Prefix HR 区间（如 60-80%），条越长表示该区间请求占比越高。")
    parts.append("  - ASCII 折线图：横轴是时间窗口，纵轴是命中率（0-100%）；越靠上表示命中率越高。")
    parts.append("  - 趋势 Q1→Q4：把时间均分为四段，比较首尾；↑ 上升，↓ 下降，→ 基本稳定。")
    parts.append("")

    # 1. Prefix Hit Ratio
    parts.append("### 1. Prefix Hit Ratio (KV Cache 内容复用度)")
    if prefix_hr["stats"]:
        _ = prefix_hr["stats"]
        parts.append(f'  累计平均: {prefix_hr["mean"]}% (被选中 worker, N={prefix_hr["count"]})')
        parts.append("  分布:")

        dist_data = [
            {"label": d["range"] + "%", "value": d["pct"], "count": d["count"]} for d in prefix_hr["distribution"]
        ]
        parts.append("  Unicode 柱状图（Prefix HR 分布）:")
        parts.append(render_bar(dist_data, show_count=True))

        parts.append(f'  冷启动率: {prefix_hr["cold_start_rate"]}%')

        trend_str = _quartile_trend(prefix_hr["trend"], "selected_hitRatio_mean")
        if trend_str:
            parts.append(f"  趋势: {trend_str}")

        # Sparkline
        if prefix_hr["trend"]:
            sparkline_data = [
                {"bucket": t["bucket"], "value": t.get("selected_hitRatio_mean", 0)} for t in prefix_hr["trend"]
            ]
            parts.append("")
            parts.append("  ASCII 折线图（Prefix HR 趋势）:")
            parts.append(render_sparkline(sparkline_data, title="Prefix HR Trend", y_label="%", y_range=(0, 100)))
    else:
        parts.append("  (无 cache_aware_scoring 数据)")
    parts.append("")

    # 2. Session Hit Rate
    parts.append("### 2. Session Hit Rate (请求级路由粘性)")
    parts.append(f'  累计: {session_hr["rate"]}% (hits={session_hr["hits"]} / total={session_hr["total"]})')
    parts.append(f'  覆盖率: {session_hr["coverage"]}% 的推理请求带 session_id')

    trend_str = _quartile_trend(session_hr["trend"], "value")
    if trend_str:
        parts.append(f"  趋势: {trend_str}")

    if session_hr["trend"]:
        parts.append("")
        parts.append("  ASCII 折线图（Session HR 趋势）:")
        parts.append(render_sparkline(session_hr["trend"], title="Session HR Trend", y_label="%", y_range=(0, 100)))
    parts.append("")

    # 3. Per-Worker
    parts.append("### 3. Per-Worker Cache Stats")
    if per_worker:
        parts.append(
            render_table(
                per_worker,
                columns=["Worker", "Selected", "Select%", "AvgHitRatio"],
                right_align={"Selected", "Select%", "AvgHitRatio"},
            )
        )
    else:
        parts.append("  (无数据)")
    parts.append("")

    # 4. Scheduling Strategy
    parts.append("### 4. Scheduling Strategy")
    parts.append(
        f'  cache_aware_scoring: {scheduling["scoring_count"]} ({scheduling["scoring_pct"]}%)'
        f' | fallback: {scheduling["fallback_count"]}'
    )
    if scheduling["fallback_reasons"]:
        reasons = ", ".join(f'{r["value"]}={r["count"]}' for r in scheduling["fallback_reasons"])
        parts.append(f"    fallback reasons: {reasons}")
    parts.append(
        f'  非最优命中选择: {scheduling["suboptimal_pct"]}%'
        f' ({scheduling.get("suboptimal_count", 0)} 次, 负载均衡优先于命中率)'
    )
    parts.append("")

    # 5. Diagnosis
    parts.append("### 5. Diagnosis")
    parts.append(f'  {diagnosis["icon"]} {diagnosis["summary"]}')
    parts.append(f'  {diagnosis["detail"]}')

    # 6. 每窗口明细预览
    if window_rows:
        parts.append("")
        parts.append("### 6. 每5s窗口明细预览（前10行）")
        parts.append(
            render_table(
                window_rows[:10],
                columns=["Time", "Prefix HR", "Session HR", "Scoring", "Fallback", "Total Running"],
                right_align={"Scoring", "Fallback", "Total Running"},
            )
        )

    return "\n".join(parts)


def format_tail_report(filepath, line_count, prefix_hr, session_hr, scheduling):
    """格式化 --tail 精简报告。"""
    parts = []
    parts.append("## Cache Hit Rate (Recent)")
    parts.append(f"**File**: {filepath} | **tail {line_count} lines**")
    parts.append("")
    parts.append(f'  Prefix Hit Ratio:  {prefix_hr["mean"]}% (avg) | Cold start: {prefix_hr["cold_start_rate"]}%')
    parts.append(
        f'  Session Hit Rate:  {session_hr["rate"]}% (hits={session_hr["hits"]}/total={session_hr["total"]})'
        f' | Coverage: {session_hr["coverage"]}%'
    )
    parts.append(
        f'  Strategy: scoring {scheduling["scoring_count"]} ({scheduling["scoring_pct"]}%)'
        f' | fallback {scheduling["fallback_count"]}'
    )

    # Sparkline
    if prefix_hr["trend"]:
        parts.append("")
        sparkline_data = [
            {"bucket": t["bucket"], "value": t.get("selected_hitRatio_mean", 0)} for t in prefix_hr["trend"]
        ]
        parts.append(render_sparkline(sparkline_data, title="Recent Prefix HR", y_label="%", y_range=(0, 100)))
        parts.append("  说明: 折线越靠上表示对应时间窗口 Prefix HR 越高。")

    return "\n".join(parts)


def build_per_window_rows(strategies, stats_recs):
    """构建每窗口明细行，用于终端预览和 details 导出。"""
    time_data = defaultdict(
        lambda: {
            "prefix_vals": [],
            "hits": 0,
            "total": 0,
            "scoring": 0,
            "fallback": 0,
            "running": 0,
            "has_running": False,
        }
    )
    for r in strategies:
        ts = r.get("ts", "")
        if r.get("strategy") == "cache_aware_scoring":
            time_data[ts]["scoring"] += 1
            time_data[ts]["prefix_vals"].append(r.get("selected_hitRatio", 0))
        else:
            time_data[ts]["fallback"] += 1

    for r in stats_recs:
        ts = r.get("ts", "")
        time_data[ts]["hits"] += r.get("hits", 0)
        time_data[ts]["total"] += r.get("total", 0)
        if "total_running" in r:
            time_data[ts]["running"] += r.get("total_running", 0)
            time_data[ts]["has_running"] = True

    rows = []
    for ts in sorted(time_data.keys()):
        d = time_data[ts]
        short_ts = ts.split(" ")[-1] if " " in ts else ts
        if d["prefix_vals"]:
            prefix_mean = round(sum(d["prefix_vals"]) / len(d["prefix_vals"]), 1)
            prefix_hr = f"{prefix_mean}%"
        else:
            prefix_hr = "-"

        if d["total"] > 0:
            session_val = round(d["hits"] / d["total"] * 100, 1)
            session_hr = f'{session_val}% ({d["hits"]}/{d["total"]})'
        else:
            session_hr = "-"

        running = str(d["running"]) if d["has_running"] else "-"
        rows.append(
            {
                "Time": short_ts,
                "Prefix HR": prefix_hr,
                "Session HR": session_hr,
                "Scoring": str(d["scoring"]),
                "Fallback": str(d["fallback"]),
                "Total Running": running,
            }
        )
    return rows


def save_detailed_report(
    filepath,
    strategies,
    stats_recs,
    prefix_hr,
    session_hr,
    per_worker,
    scheduling,
    diagnosis,
    output_dir,
    time_span=None,
):
    """导出详细数据 Markdown 文件。

    主报告包含 Per-Worker 统计和 Fallback 明细。
    每窗口明细数据拆分到 details/per_window_data.md。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"cache_hitrate_report_{timestamp}.md")

    parts = []
    parts.append("# Cache Hit Rate Detailed Report")
    parts.append(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    parts.append(f"**Source**: {filepath}")
    if time_span:
        parts.append(f"**Span**: {time_span}")
    parts.append("")

    parts.append("## 图表说明（Legend）")
    parts.append("- **Unicode 柱状图**: 展示 Prefix HR 分布，`█` 越多说明该命中率区间占比越高。")
    parts.append("- **ASCII 折线图**: 展示命中率随时间变化，横轴为时间窗口，纵轴为命中率（0-100%）。")
    parts.append("- **Q1~Q4 趋势**: 将观察区间均分四段，反映整体走向（↑/↓/→）。")
    parts.append("")

    # 1) 主指标摘要（与终端一致，避免“只在终端可见”）
    parts.append("## 1. Key Metrics Summary")
    parts.append("")
    parts.append("### Prefix Hit Ratio")
    if prefix_hr["stats"]:
        parts.append(f'- 累计平均: **{prefix_hr["mean"]}%** (N={prefix_hr["count"]})')
        parts.append(f'- 冷启动率: **{prefix_hr["cold_start_rate"]}%**')
        trend_str = _quartile_trend(prefix_hr["trend"], "selected_hitRatio_mean")
        if trend_str:
            parts.append(f"- 趋势: {trend_str}")
        dist_data = [{"label": d["range"] + "%", "value": d["pct"], "count": d["count"]} for d in prefix_hr["distribution"]]
        parts.append("")
        parts.append("```text")
        parts.append("Unicode 柱状图（Prefix HR 分布）")
        parts.append(render_bar(dist_data, show_count=True))
        if prefix_hr["trend"]:
            sparkline_data = [{"bucket": t["bucket"], "value": t.get("selected_hitRatio_mean", 0)} for t in prefix_hr["trend"]]
            parts.append("")
            parts.append("ASCII 折线图（Prefix HR 趋势）")
            parts.append(render_sparkline(sparkline_data, title="Prefix HR Trend", y_label="%", y_range=(0, 100)))
        parts.append("```")
    else:
        parts.append("- (无 cache_aware_scoring 数据)")
    parts.append("")

    parts.append("### Session Hit Rate")
    parts.append(f'- 累计: **{session_hr["rate"]}%** (hits={session_hr["hits"]}/total={session_hr["total"]})')
    parts.append(f'- 覆盖率: **{session_hr["coverage"]}%**')
    trend_str = _quartile_trend(session_hr["trend"], "value")
    if trend_str:
        parts.append(f"- 趋势: {trend_str}")
    if session_hr["trend"]:
        parts.append("")
        parts.append("```text")
        parts.append("ASCII 折线图（Session HR 趋势）")
        parts.append(render_sparkline(session_hr["trend"], title="Session HR Trend", y_label="%", y_range=(0, 100)))
        parts.append("```")
    parts.append("")

    parts.append("### Scheduling Strategy")
    parts.append(
        f'- cache_aware_scoring: **{scheduling["scoring_count"]} ({scheduling["scoring_pct"]}%)**'
        f' | fallback: **{scheduling["fallback_count"]}**'
    )
    parts.append(
        f'- 非最优命中选择: **{scheduling["suboptimal_pct"]}%**'
        f' ({scheduling.get("suboptimal_count", 0)} 次, 负载均衡优先于命中率)'
    )
    parts.append(f'- Diagnosis: {diagnosis["icon"]} {diagnosis["summary"]}；{diagnosis["detail"]}')
    parts.append("")

    # 2) Per-Worker 完整统计
    parts.append("## 2. Per-Worker 完整统计")
    parts.append("")
    if per_worker:
        parts.append(
            render_table(
                per_worker,
                columns=["Worker", "Selected", "Select%", "AvgHitRatio"],
                right_align={"Selected", "Select%", "AvgHitRatio"},
            )
        )
    parts.append("")

    # 3) Fallback 明细
    if scheduling["fallback_reasons"]:
        parts.append("## 3. Fallback 明细")
        for reason in scheduling["fallback_reasons"]:
            parts.append(f'- **{reason["value"]}**: {reason["count"]} 次 ({reason["pct"]}%)')
        parts.append("")

    # 每窗口明细 → 拆分到 details/
    window_rows = build_per_window_rows(strategies, stats_recs)
    window_rows_merged = merge_blank_window_rows(window_rows)
    session_rows = compute_session_details(strategies, _strip_scheme)
    session_summary = summarize_session_details(session_rows)

    if window_rows:
        # 主报告中添加引用
        parts.append(
            f"> 每5s窗口明细数据（原始 {len(window_rows)} 条，合并后 {len(window_rows_merged)} 条）:"
            " [details/per_window_data.md](details/per_window_data.md)"
        )
        parts.append("")

        # 写入 details 子目录
        details_dir = os.path.join(output_dir, "details")
        os.makedirs(details_dir, exist_ok=True)
        detail_parts = ["# 每5s窗口明细数据", ""]
        detail_parts.append(
            "> 注：连续空窗口（Prefix/Session 都为空、且 Scoring/Fallback=0）已按 3 行格式合并展示（起始/合并说明/结束）。"
        )
        detail_parts.append("")
        detail_parts.append(
            render_table(
                window_rows_merged,
                columns=["Time", "Prefix HR", "Session HR", "Scoring", "Fallback", "Total Running"],
                right_align={"Scoring", "Fallback", "Total Running"},
            )
        )
        detail_parts.append("")

        detail_path = os.path.join(details_dir, "per_window_data.md")
        with open(detail_path, "w") as f:
            f.write("\n".join(detail_parts))

        if session_rows:
            parts.append(
                f"> Session 命中详情 ({len(session_rows)} sessions): [details/session_hit_details.md](details/session_hit_details.md)"
            )
            parts.append("")

            session_parts = ["# Session 命中详情", ""]
            session_parts.append("## 概览")
            session_parts.append(f'- Total sessions: **{session_summary["total_sessions"]}**')
            session_parts.append(
                f'- Sessions with >1 request: **{session_summary["multi_req"]}**'
                f' | single request: **{session_summary["single_req"]}**'
            )
            if session_summary["multi_req"] > 0:
                sticky_pct = round(session_summary["sticky_multi"] / session_summary["multi_req"] * 100, 1)
                session_parts.append(
                    f'- Sticky (multi-request): **{session_summary["sticky_multi"]} ({sticky_pct}%)**'
                    f' | non-sticky: **{session_summary["non_sticky_multi"]}**'
                )
            session_parts.append(
                f'- Non-first request avg hit: **{session_summary["non_first_avg"]}%**'
                f' (N={session_summary["non_first_total"]})'
            )
            session_parts.append("")
            session_parts.append("## 明细表")
            session_parts.append(
                render_table(
                    session_rows,
                    columns=[
                        "session",
                        "req_count",
                        "first_hit",
                        "avg_hit(excl_first)",
                        "max_hit",
                        "min_hit",
                        "all_hits",
                        "prefill_urls",
                        "switch_req_pairs",
                        "sharp_drop_request_ids",
                        "sticky",
                        "unique_workers",
                    ],
                    right_align={"req_count", "first_hit", "avg_hit(excl_first)", "max_hit", "min_hit", "unique_workers"},
                )
            )
            session_parts.append("")

            session_path = os.path.join(details_dir, "session_hit_details.md")
            with open(session_path, "w") as f:
                f.write("\n".join(session_parts))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(parts))

    return output_path


# ════════════════════════════════════════════════════════════════
# 时间跨度计算
# ════════════════════════════════════════════════════════════════


def compute_time_span(strategies, stats_recs):
    """从数据中计算时间跨度字符串。"""
    all_ts = []
    for r in strategies + stats_recs:
        ts = r.get("ts", "")
        if ts:
            try:
                all_ts.append(parse_ts(ts))
            except ValueError:
                pass
    if len(all_ts) < 2:
        return None
    t_min = min(all_ts)
    t_max = max(all_ts)
    duration = t_max - t_min
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    start = t_min.strftime("%Y-%m-%d %H:%M:%S")
    end = t_max.strftime("%Y-%m-%d %H:%M:%S")
    if hours > 0:
        return f"{start} ~ {end} ({hours}h{minutes}m)"
    return f"{start} ~ {end} ({minutes}m)"


# ════════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="FastDeploy Go Router Cache 命中率统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("log_file", help="日志文件路径")
    parser.add_argument("--tail", nargs="?", const="2000", help="只分析尾部数据（行数如 2000，或时间如 30m）")
    parser.add_argument("--watch", action="store_true", help="全量分析后提示持续监控命令")
    parser.add_argument(
        "--output", default=None, help="详细报告输出目录（默认：skill_output/stat-cache-hitrate/<timestamp>/）"
    )
    parser.add_argument(
        "--start", default=None, help='起始时间（如 "16:00:00"、"03/31 16:00"、"2026/03/31 16:00:00"）'
    )
    parser.add_argument("--end", default=None, help='结束时间（如 "17:00:00"、"03/31 17:00"、"2026/03/31 17:00:00"）')
    return parser.parse_args()


def main():
    args = parse_args()

    # 验证文件存在
    if not os.path.isfile(args.log_file):
        print(f"Error: 文件不存在: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    # --tail 与 --start/--end 不能混用（两者是不同的范围选择方式）
    if args.tail and (args.start or args.end):
        print("Error: --tail 与 --start/--end 不能同时使用，请选择其一", file=sys.stderr)
        sys.exit(1)

    # 时间范围预过滤（--start 和 --end 可单独或同时指定）
    import atexit

    log_file = args.log_file
    if args.start or args.end:
        start_ts = complete_time_arg(args.start, log_file, is_end=False) if args.start else None
        end_ts = complete_time_arg(args.end, log_file, is_end=True) if args.end else None
        filtered_path, is_temp = filter_file_by_time_range(log_file, start_ts, end_ts)
        if is_temp:
            atexit.register(lambda p=filtered_path: os.unlink(p) if os.path.exists(p) else None)
        log_file = filtered_path
        print(f'时间范围过滤: {start_ts or "..."} ~ {end_ts or "..."}', file=sys.stderr)

    # Phase 2: 提取 + 解析
    strategy_recs, stats_recs, inference_count, line_count = extract_data(log_file, args.tail)

    if not strategy_recs and not stats_recs:
        print(
            "Warning: 未找到 cache-aware 策略行或 [stats] 行。" "请确认日志文件包含 Go Router 日志。", file=sys.stderr
        )
        sys.exit(0)

    # Phase 3: 计算三层指标
    prefix_hr = compute_prefix_hitrate(strategy_recs)
    session_hr = compute_session_hitrate(stats_recs, inference_count)
    per_worker = compute_per_worker_stats(strategy_recs)
    scheduling = compute_scheduling_stats(strategy_recs)
    diagnosis = cross_diagnose(prefix_hr, session_hr)

    # Phase 4: 输出
    if args.tail:
        print(format_tail_report(args.log_file, line_count, prefix_hr, session_hr, scheduling))
    else:
        time_span = compute_time_span(strategy_recs, stats_recs)
        window_rows = build_per_window_rows(strategy_recs, stats_recs)
        print(
            format_full_report(
                args.log_file,
                line_count,
                prefix_hr,
                session_hr,
                per_worker,
                scheduling,
                diagnosis,
                time_span,
                window_rows=window_rows,
            )
        )

        # 导出详细报告
        if args.output:
            output_dir = args.output
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            golang_router_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", ".."))
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(golang_router_root, "skill_output", "stat-cache-hitrate", run_timestamp)
        report_path = save_detailed_report(
            args.log_file,
            strategy_recs,
            stats_recs,
            prefix_hr,
            session_hr,
            per_worker,
            scheduling,
            diagnosis,
            output_dir,
            time_span=time_span,
        )
        print("\n\U0001f4c4 详细数据见:")
        report_abs, report_uri = _build_path_links(report_path)
        print(f"  - 报告文件: {report_abs}")
        print(f"    URI: {report_uri}")
        details_path = os.path.join(os.path.dirname(report_path), "details", "per_window_data.md")
        if os.path.exists(details_path):
            details_abs, details_uri = _build_path_links(details_path)
            print(f"  - 窗口明细: {details_abs}")
            print(f"    URI: {details_uri}")

    if args.watch:
        print("\n\U0001f4a1 持续跟踪: /loop 30s /stat-cache-hitrate --tail")


if __name__ == "__main__":
    main()
