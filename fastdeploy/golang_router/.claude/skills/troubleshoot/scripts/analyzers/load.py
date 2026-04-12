#!/usr/bin/env python3
"""
Load Analyzer — 负载与计数器分析

分析 Worker 负载分布、计数器异常、请求堆积检测、token 计数器。
"""

import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chart import render_bar, render_sparkline, render_table
from log_parser import extract_ts, match_select_release, parse_stats_line
from stats import compute_statistics, time_bucket

# ════════════════════════════════════════════════════════════════
# Counter 异常检测正则
# ════════════════════════════════════════════════════════════════

DOUBLE_RELEASE_RE = re.compile(r"release worker:\s*(http://\S+)\s+skipped.*?double-release")
COUNTER_CLEANED_RE = re.compile(r"release worker:\s*(http://\S+)\s+skipped.*?counter already cleaned up")
COUNTER_PRESERVED_RE = re.compile(r"counter preserved.*?(http://\S+)")
TOKEN_PRESERVED_RE = re.compile(r"token counter preserved.*?(http://\S+)")

# Token 事件
SELECT_TOKENS_RE = re.compile(r"select worker \(prefill\):\s*(http://\S+),\s*tokens:\s*(\d+)")
RELEASE_TOKENS_RE = re.compile(r"release prefill tokens:\s*(http://\S+),\s*tokens:\s*(\d+)")


def parse_counter_anomaly(line):
    """解析 H5 counter 异常行。"""
    ts = extract_ts(line)
    m = DOUBLE_RELEASE_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "anomaly_type": "double-release"}
    m = COUNTER_CLEANED_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "anomaly_type": "counter-cleaned-up"}
    m = COUNTER_PRESERVED_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "anomaly_type": "counter-preserved"}
    m = TOKEN_PRESERVED_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "anomaly_type": "token-preserved"}
    return None


# ════════════════════════════════════════════════════════════════
# 主分析函数
# ════════════════════════════════════════════════════════════════


def analyze_load(log_file, tail=None):
    """分析负载与计数器。

    Returns:
        dict: {load_stats, worker_load, load_trend, counter_anomalies,
               select_release, token_stats, diagnoses, summary}
    """
    h7_lines = _grep_lines(log_file, r"\[stats\]", tail)
    h3_lines = _grep_lines(log_file, r"select worker|release worker|Failed to select", tail)
    h5_lines = _grep_lines(
        log_file,
        r"counter preserved|cleanup unhealthy|removed counters|counter already|double-release|preserved counters",
        tail,
    )
    h11_lines = _grep_lines(log_file, r"release prefill tokens", tail)

    # 解析 stats 行
    stats_records = [r for line in h7_lines for r in [parse_stats_line(line)] if r]

    # 负载统计
    total_running_vals = [r["total_running"] for r in stats_records if "total_running" in r]
    load_stats = compute_statistics(total_running_vals) if total_running_vals else {}

    # Per-Worker 负载分布
    worker_running = defaultdict(list)
    for r in stats_records:
        for w_url, running in r.get("workers", {}).items():
            worker_running[w_url].append(running)

    worker_load = []
    for w_url in sorted(worker_running.keys()):
        vals = worker_running[w_url]
        avg = sum(vals) / len(vals) if vals else 0
        worker_load.append(
            {
                "worker": w_url.replace("http://", ""),
                "avg_running": round(avg, 1),
                "max_running": max(vals) if vals else 0,
                "samples": len(vals),
            }
        )

    # 负载趋势
    load_trend = (
        time_bucket(stats_records, window="auto", agg_specs=[("total_running", "mean")]) if stats_records else []
    )

    # Counter 异常
    counter_anomalies = defaultdict(lambda: defaultdict(int))
    for line in h5_lines:
        evt = parse_counter_anomaly(line)
        if evt:
            counter_anomalies[evt["anomaly_type"]][evt["worker"]] += 1

    anomaly_summary = []
    for atype, workers in counter_anomalies.items():
        total = sum(workers.values())
        anomaly_summary.append(
            {
                "type": atype,
                "total": total,
                "workers": dict(workers),
            }
        )

    # Select/Release 匹配
    sr_result = (
        match_select_release(h3_lines)
        if h3_lines
        else {"matched": [], "unmatched_selects": [], "failed_selects": [], "per_worker": {}}
    )

    # Token 统计
    token_stats = _analyze_tokens(h3_lines, h11_lines)

    # 请求堆积检测
    pileup = _detect_pileup(stats_records)

    # 诊断
    diagnoses = _diagnose(load_stats, worker_load, anomaly_summary, sr_result, pileup)

    return {
        "load_stats": load_stats,
        "worker_load": worker_load,
        "load_trend": load_trend,
        "counter_anomalies": anomaly_summary,
        "select_release": sr_result,
        "token_stats": token_stats,
        "pileup_detected": pileup,
        "diagnoses": diagnoses,
        "summary": f"{len(stats_records)} stats 采样, {len(worker_running)} Worker(s)",
    }


def _analyze_tokens(h3_lines, h11_lines):
    """分析 token 分配与释放。"""
    token_alloc = defaultdict(list)
    token_release = defaultdict(list)

    for line in h3_lines:
        m = SELECT_TOKENS_RE.search(line)
        if m:
            token_alloc[m.group(1)].append(int(m.group(2)))

    for line in h11_lines:
        m = RELEASE_TOKENS_RE.search(line)
        if m:
            token_release[m.group(1)].append(int(m.group(2)))

    result = []
    all_workers = set(token_alloc.keys()) | set(token_release.keys())
    for w in sorted(all_workers):
        allocs = token_alloc.get(w, [])
        releases = token_release.get(w, [])
        result.append(
            {
                "worker": w.replace("http://", ""),
                "alloc_count": len(allocs),
                "alloc_avg": round(sum(allocs) / len(allocs), 0) if allocs else 0,
                "release_count": len(releases),
            }
        )
    return result


def _detect_pileup(stats_records):
    """检测请求堆积：total_running 连续上升 >5 个采样点。"""
    if len(stats_records) < 5:
        return False
    vals = [r.get("total_running", 0) for r in stats_records]
    max_consecutive = 0
    current = 0
    for i in range(1, len(vals)):
        if vals[i] > vals[i - 1]:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0
    return max_consecutive >= 5


def _diagnose(load_stats, worker_load, anomaly_summary, sr_result, pileup):
    """生成负载诊断。"""
    diagnoses = []

    if pileup:
        diagnoses.append(
            {"severity": "HIGH", "message": "total_running 持续上升，疑似请求堆积", "source_layer": "FD 后端"}
        )

    # 空闲 Worker
    for w in worker_load:
        if w["avg_running"] == 0 and w["samples"] > 3:
            diagnoses.append(
                {
                    "severity": "MEDIUM",
                    "message": f'{w["worker"]} running 持续 =0（空闲或故障未移除）',
                    "source_layer": "Router",
                }
            )

    # 负载严重不均
    if load_stats.get("stddev", 0) > 3:
        diagnoses.append(
            {
                "severity": "MEDIUM",
                "message": f'负载标准差 {load_stats["stddev"]}，分布不均衡',
                "source_layer": "Router",
            }
        )

    # Counter 异常
    for a in anomaly_summary:
        if a["type"] == "double-release" and a["total"] > 0:
            diagnoses.append(
                {
                    "severity": "MEDIUM",
                    "message": f'double-release {a["total"]} 次（计数器逻辑 bug）',
                    "source_layer": "Router",
                }
            )

    # Select/Release 不一致
    for w_url, pw in sr_result.get("per_worker", {}).items():
        if pw.get("delta", 0) > 0:
            diagnoses.append(
                {
                    "severity": "HIGH",
                    "message": f'{w_url.replace("http://","")} select-release 差值 {pw["delta"]}（请求泄漏/卡住）',
                    "source_layer": "FD 后端",
                }
            )

    # 卡住的请求
    if sr_result.get("unmatched_selects"):
        diagnoses.append(
            {
                "severity": "HIGH",
                "message": f'{len(sr_result["unmatched_selects"])} 个 select 无对应 release（疑似卡住）',
                "source_layer": "FD 后端",
            }
        )

    return diagnoses


# ════════════════════════════════════════════════════════════════
# 报告格式化
# ════════════════════════════════════════════════════════════════


def format_load_report(result):
    """将分析结果格式化为终端报告。"""
    sections = ["## 负载与计数器分析", ""]
    sections.append(f'  {result["summary"]}')
    sections.append("")

    if result["diagnoses"]:
        sections.append("### 诊断")
        sections.append("")
        for d in result["diagnoses"]:
            sections.append(f'  [{d["severity"]}] [{d["source_layer"]}] {d["message"]}')
        sections.append("")

    # 负载概览
    ls = result.get("load_stats", {})
    if ls:
        sections.append("### 负载概览 (total_running)")
        sections.append("")
        sections.append(
            f'  mean={ls.get("mean",0)}  p50={ls.get("p50",0)}  p90={ls.get("p90",0)}  '
            f'p99={ls.get("p99",0)}  max={ls.get("max",0)}  stddev={ls.get("stddev",0)}'
        )
        sections.append("")

    # Per-Worker 负载
    if result["worker_load"]:
        sections.append("### Per-Worker 负载")
        sections.append("")
        bar_data = [
            {"label": w["worker"][:25], "value": min(100, w["avg_running"] * 5), "count": w["avg_running"]}
            for w in result["worker_load"]
        ]
        sections.append(render_bar(bar_data, show_count=True))
        sections.append("")

    # 负载趋势
    if result["load_trend"] and len(result["load_trend"]) > 1:
        sections.append("### 负载趋势")
        sections.append("")
        sections.append(
            render_sparkline(
                result["load_trend"], value_field="total_running_mean", title="Total Running", y_label="req"
            )
        )
        sections.append("")

    # Counter 异常
    if result["counter_anomalies"]:
        sections.append("### 计数器异常")
        sections.append("")
        for a in result["counter_anomalies"]:
            workers_str = ", ".join(f'{w.replace("http://","")}({c})' for w, c in a["workers"].items())
            sections.append(f'  {a["type"]}: {a["total"]} 次 [{workers_str}]')
        sections.append("")

    # Select/Release 匹配
    sr = result.get("select_release", {})
    if sr.get("per_worker"):
        sections.append("### Select/Release 匹配")
        sections.append("")
        table_data = []
        for w_url, pw in sorted(sr["per_worker"].items()):
            table_data.append(
                {
                    "Worker": w_url.replace("http://", ""),
                    "Select": str(pw["selects"]),
                    "Release": str(pw["releases"]),
                    "Delta": str(pw["delta"]),
                }
            )
        sections.append(
            render_table(
                table_data,
                columns=["Worker", "Select", "Release", "Delta"],
                right_align={"Select", "Release", "Delta"},
            )
        )
        sections.append("")

    if sr.get("unmatched_selects"):
        sections.append(f'  ⚠ {len(sr["unmatched_selects"])} 个未匹配 select（疑似请求卡住）')
        for u in sr["unmatched_selects"][:5]:
            sections.append(f'    [{u.get("select_ts","")}] {u["worker"].replace("http://","")} ({u["type"]})')
        sections.append("")

    # Token 统计
    if result.get("token_stats"):
        sections.append("### Token 计数器")
        sections.append("")
        sections.append(
            render_table(
                result["token_stats"],
                columns=["worker", "alloc_count", "alloc_avg", "release_count"],
                right_align={"alloc_count", "alloc_avg", "release_count"},
            )
        )
        sections.append("")

    return "\n".join(sections)


# ════════════════════════════════════════════════════════════════
# Grep 工具
# ════════════════════════════════════════════════════════════════


def _grep_lines(log_file, pattern, tail=None):
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
