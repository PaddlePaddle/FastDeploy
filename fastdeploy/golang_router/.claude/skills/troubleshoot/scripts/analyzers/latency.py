#!/usr/bin/env python3
"""
Latency Analyzer — 延迟分析

分析 Router 日志中的请求延迟百分位数、延迟分布、吞吐量趋势、调度耗时、慢请求。
仅统计推理请求路径（/v1/chat/completions, /v1/completions）。
"""

import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chart import render_bar, render_sparkline, render_table
from log_parser import TS_MS_RE, extract_tags, parse_http_line
from stats import compute_statistics, time_bucket

# ════════════════════════════════════════════════════════════════
# 调度耗时解析
# ════════════════════════════════════════════════════════════════


def _parse_scheduling_ms(ts_ms_lines):
    """从 ts_ms 行计算调度耗时（同一请求两个 ts_ms 之间的差值）。

    同一 request_id 的两条 ts_ms 行之间的时间差即为调度耗时。
    返回 ms 列表。
    """
    from datetime import datetime

    # 按 request_id 分组
    by_reqid = defaultdict(list)
    for line in ts_ms_lines:
        m = TS_MS_RE.search(line)
        if not m:
            continue
        ts_ms_str = m.group(1)
        tags = extract_tags(line)
        rid = tags.get("request_id", "")
        if rid:
            try:
                dt = datetime.strptime(ts_ms_str, "%Y-%m-%d %H:%M:%S.%f")
                by_reqid[rid].append(dt)
            except ValueError:
                pass

    # 计算每个 request_id 的 max - min 差值
    durations = []
    for rid, timestamps in by_reqid.items():
        if len(timestamps) >= 2:
            timestamps.sort()
            delta_ms = (timestamps[-1] - timestamps[0]).total_seconds() * 1000
            durations.append(round(delta_ms, 3))

    return durations


# ════════════════════════════════════════════════════════════════
# 主分析函数
# ════════════════════════════════════════════════════════════════

LATENCY_DIST_SPEC = "<100,100-500,500-1000,1000-5000,5000-10000,>10000"


def analyze_latency(log_file, tail=None):
    """分析日志中的请求延迟。

    Args:
        log_file: 日志文件路径
        tail: 尾部行数限制

    Returns:
        dict: {
            stats: {count, p50, p90, p95, p99, max, mean, stddev, distribution},
            latency_trend: [{bucket, latency_ms_p50}],
            throughput_trend: [{bucket, count}],
            slow_top10: [{ts, path, status, latency_ms, client_ip}],
            scheduling_stats: {p50, p90, p99} | None,
            diagnoses: [{message, severity}],
        }
    """
    # Phase 1: Grep 提取
    http_lines = _grep_lines(log_file, r"\[(POST|GET)\] /", tail)
    ts_ms_lines = _grep_lines(log_file, "ts_ms=", tail)

    # Phase 2: 解析 HTTP 行（仅推理路径）
    http_records = []
    for line in http_lines:
        r = parse_http_line(line, inference_only=True)
        if r:
            http_records.append(r)

    # Phase 3: 分析

    # 3.1 延迟统计
    latency_values = [r["latency_ms"] for r in http_records]
    stats = compute_statistics(
        latency_values,
        percentiles_list=[50, 90, 95, 99],
        distribution_spec=LATENCY_DIST_SPEC,
    )

    # 3.2 延迟趋势 (p50)
    latency_trend = time_bucket(
        http_records,
        window="auto",
        agg_specs=[("latency_ms", "p50")],
    )

    # 3.3 吞吐量趋势
    throughput_trend = time_bucket(http_records, window="auto")

    # 3.4 慢请求 Top 10
    sorted_by_latency = sorted(http_records, key=lambda r: -r["latency_ms"])
    slow_top10 = []
    for r in sorted_by_latency[:10]:
        slow_top10.append(
            {
                "ts": r["ts"],
                "path": r["path"],
                "status": r["status"],
                "latency_ms": r["latency_ms"],
                "client_ip": r["client_ip"],
            }
        )

    # 3.5 调度耗时
    scheduling_stats = None
    if ts_ms_lines:
        sched_durations = _parse_scheduling_ms(ts_ms_lines)
        if sched_durations:
            sched_raw = compute_statistics(sched_durations, percentiles_list=[50, 90, 99])
            scheduling_stats = {
                "p50": sched_raw["p50"],
                "p90": sched_raw["p90"],
                "p99": sched_raw["p99"],
                "count": sched_raw["count"],
            }

    # 3.6 诊断规则
    diagnoses = _run_diagnostics(stats, scheduling_stats)

    return {
        "stats": stats,
        "latency_trend": latency_trend,
        "throughput_trend": throughput_trend,
        "slow_top10": slow_top10,
        "scheduling_stats": scheduling_stats,
        "diagnoses": diagnoses,
    }


def _run_diagnostics(stats, scheduling_stats):
    """应用诊断规则。"""
    diagnoses = []

    if stats["count"] == 0:
        diagnoses.append({"message": "未找到推理请求", "severity": "INFO"})
        return diagnoses

    p99 = stats.get("p99", 0)
    p50 = stats.get("p50", 0)

    # p99 > 10s
    if p99 > 10000:
        if scheduling_stats and scheduling_stats["p99"] < 100:
            diagnoses.append(
                {
                    "message": f'p99={p99:.0f}ms 但调度仅 {scheduling_stats["p99"]:.0f}ms → 延迟在后端推理层',
                    "severity": "HIGH",
                }
            )
        elif scheduling_stats and scheduling_stats["p99"] >= 100:
            diagnoses.append(
                {
                    "message": f'p99={p99:.0f}ms 且调度 p99={scheduling_stats["p99"]:.0f}ms → 调度层瓶颈',
                    "severity": "CRITICAL",
                }
            )
        else:
            diagnoses.append(
                {
                    "message": f"p99={p99:.0f}ms (>10s)，后端推理延迟高",
                    "severity": "HIGH",
                }
            )

    # 尾延迟
    if p50 > 0 and p99 / p50 > 10:
        diagnoses.append(
            {
                "message": f"p99/p50={p99/p50:.1f}x → 尾延迟严重",
                "severity": "MEDIUM",
            }
        )

    if not diagnoses:
        diagnoses.append(
            {
                "message": f"延迟正常 (p50={p50:.0f}ms, p99={p99:.0f}ms)",
                "severity": "INFO",
            }
        )

    return diagnoses


# ════════════════════════════════════════════════════════════════
# Grep 工具
# ════════════════════════════════════════════════════════════════


def _grep_lines(log_file, pattern, tail=None):
    """用 grep 从日志文件提取匹配行。"""
    try:
        if tail:
            cmd = f"tail -n {tail} {_shell_quote(log_file)} | grep -E {_shell_quote(pattern)}"
        else:
            cmd = f"grep -E {_shell_quote(pattern)} {_shell_quote(log_file)}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode > 1:
            return []
        return [line for line in result.stdout.split("\n") if line.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _shell_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


# ════════════════════════════════════════════════════════════════
# 报告格式化
# ════════════════════════════════════════════════════════════════


def format_latency_report(result):
    """将分析结果格式化为终端报告。"""
    sections = []
    stats = result["stats"]

    sections.append("## 延迟分析")
    sections.append("")

    if stats["count"] == 0:
        sections.append("  未找到推理请求 (/v1/chat/completions, /v1/completions)")
        return "\n".join(sections)

    # 百分位数概览
    sections.append(
        f'  推理请求: {stats["count"]}  |  '
        f'p50={_fmt_ms(stats["p50"])}  p90={_fmt_ms(stats["p90"])}  '
        f'p95={_fmt_ms(stats["p95"])}  p99={_fmt_ms(stats["p99"])}  '
        f'max={_fmt_ms(stats["max"])}'
    )
    sections.append("  指标口径: pXX=延迟分位数；吞吐量=每个时间桶内请求数(count)；调度耗时=同 request_id 的 ts_ms(max-min)。")
    sections.append("")

    # 延迟分布
    if stats.get("distribution"):
        sections.append("### 延迟分布")
        sections.append("")
        bar_data = []
        for d in stats["distribution"]:
            bar_data.append(
                {
                    "label": d["range"],
                    "value": d["pct"],
                    "count": d["count"],
                }
            )
        sections.append(render_bar(bar_data, show_count=True))
        sections.append("")

    # 延迟趋势
    if result["latency_trend"] and len(result["latency_trend"]) > 1:
        sections.append("### 延迟趋势 (p50)")
        sections.append("")
        sections.append(
            render_sparkline(
                result["latency_trend"],
                value_field="latency_ms_p50",
                title="p50 Latency",
                y_label="ms",
            )
        )
        sections.append("")

    # 吞吐量趋势
    if result["throughput_trend"] and len(result["throughput_trend"]) > 1:
        sections.append("### 吞吐量趋势")
        sections.append("")
        sections.append(
            render_sparkline(
                result["throughput_trend"],
                value_field="count",
                title="Throughput",
                y_label="req",
            )
        )
        sections.append("")

    # 调度耗时
    if result["scheduling_stats"]:
        ss = result["scheduling_stats"]
        sections.append(f'### 调度耗时 ({ss["count"]} samples)')
        sections.append(f'  p50={_fmt_ms(ss["p50"])}  p90={_fmt_ms(ss["p90"])}  p99={_fmt_ms(ss["p99"])}')
        sections.append("")

    # 慢请求 Top 10
    if result["slow_top10"]:
        sections.append("### 慢请求 Top 10")
        sections.append("")
        table_data = []
        for r in result["slow_top10"]:
            table_data.append(
                {
                    "时间": r["ts"][-8:] if len(r["ts"]) > 8 else r["ts"],
                    "延迟": _fmt_ms(r["latency_ms"]),
                    "状态": str(r["status"]),
                    "路径": r["path"],
                    "Client": r["client_ip"],
                }
            )
        sections.append(
            render_table(
                table_data,
                columns=["时间", "延迟", "状态", "路径", "Client"],
            )
        )
        sections.append("")

    # 诊断（仅在 detail 输出）
    if result["diagnoses"]:
        sections.append("### 诊断")
        sections.append("  诊断见详情: [detail/latency_diagnoses.md](../detail/latency_diagnoses.md)")
        sections.append("")

    return "\n".join(sections)


def _fmt_ms(ms):
    """格式化毫秒值为人类可读字符串。"""
    if ms >= 60000:
        return f"{ms/60000:.1f}min"
    elif ms >= 1000:
        return f"{ms/1000:.2f}s"
    elif ms >= 1:
        return f"{ms:.1f}ms"
    else:
        return f"{ms*1000:.0f}µs"
