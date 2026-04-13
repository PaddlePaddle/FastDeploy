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

from log_parser import extract_ts, match_select_release, parse_stats_line
from stats import compute_statistics, time_bucket
from analyzers.load_report import format_load_report

# ════════════════════════════════════════════════════════════════
# Counter 异常检测正则
# ════════════════════════════════════════════════════════════════

URL_RE = r"((?:https?://)?[A-Za-z0-9.-]+(?::\d+)?)"
DOUBLE_RELEASE_RE = re.compile(rf"release worker:\s*{URL_RE}\s+skipped.*?double-release")
COUNTER_CLEANED_RE = re.compile(rf"release worker:\s*{URL_RE}\s+skipped.*?counter already cleaned up")
COUNTER_PRESERVED_RE = re.compile(rf"counter preserved.*?{URL_RE}")
TOKEN_PRESERVED_RE = re.compile(rf"token counter preserved.*?{URL_RE}")

# Token 事件
SELECT_TOKENS_RE = re.compile(rf"select worker \((\w+)\):\s*{URL_RE},\s*tokens:\s*(\d+)")
RELEASE_TOKENS_RE = re.compile(rf"release (?:([a-zA-Z_]+)\s+)?tokens:\s*{URL_RE},\s*tokens:\s*(\d+)")
SELECT_REQ_COUNT_RE = re.compile(rf"select worker \((\w+)\):\s*{URL_RE},\s*count:\s*(\d+)")
RELEASE_REQ_COUNT_RE = re.compile(rf"release worker:\s*{URL_RE},\s*count:\s*(\d+)")


def _strip_scheme(url):
    return re.sub(r"^https?://", "", url)


def _normalize_worker_type(worker_type):
    t = (worker_type or "unknown").lower()
    if t in ("prefill", "decode", "mixed"):
        return t
    return "unknown"


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
    h11_lines = _grep_lines(log_file, r"release (?:[a-zA-Z_]+\s+)?tokens", tail)

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
                "worker": _strip_scheme(w_url),
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
        match_select_release(h3_lines + h11_lines)
        if h3_lines
        else {
            "matched": [],
            "unmatched_selects": [],
            "unmatched_releases": [],
            "untracked_selects": [],
            "failed_selects": [],
            "per_worker": {},
            "id_coverage": {},
            "type_summary": {},
            "worker_type_profile": {},
        }
    )

    # Token 统计
    token_stats = _analyze_tokens(h3_lines, h11_lines)
    counter_last_state = _analyze_counter_last_state(h3_lines + h11_lines)

    # 请求堆积检测
    pileup = _detect_pileup(stats_records)

    # 诊断
    diagnoses = _diagnose(load_stats, worker_load, anomaly_summary, sr_result, token_stats, pileup)

    return {
        "load_stats": load_stats,
        "worker_load": worker_load,
        "load_trend": load_trend,
        "counter_anomalies": anomaly_summary,
        "select_release": sr_result,
        "token_stats": token_stats,
        "counter_last_state": counter_last_state,
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
            token_alloc[m.group(2)].append(int(m.group(3)))

    for line in h11_lines:
        m = RELEASE_TOKENS_RE.search(line)
        if m:
            token_release[m.group(2)].append(int(m.group(3)))

    result = []
    all_workers = set(token_alloc.keys()) | set(token_release.keys())
    for w in sorted(all_workers):
        allocs = token_alloc.get(w, [])
        releases = token_release.get(w, [])
        result.append(
            {
                "worker": _strip_scheme(w),
                "alloc_count": len(allocs),
                "alloc_avg": round(sum(allocs) / len(allocs), 0) if allocs else 0,
                "release_count": len(releases),
            }
        )
    return result


def _analyze_counter_last_state(lines):
    """统计每个 worker 的 request/token counter 最后一条计数日志值与动作类型。"""
    state = defaultdict(
        lambda: {
            "req_last_action": "-",
            "req_last_value": "-",
            "token_last_action": "-",
            "token_last_value": "-",
            "last_ts": "",
        }
    )
    for line in lines:
        ts = extract_ts(line) or ""
        m = SELECT_REQ_COUNT_RE.search(line)
        if m:
            w = m.group(2)
            state[w]["req_last_action"] = "select"
            state[w]["req_last_value"] = m.group(3)
            state[w]["last_ts"] = ts
            continue
        m = RELEASE_REQ_COUNT_RE.search(line)
        if m:
            w = m.group(1)
            state[w]["req_last_action"] = "release"
            state[w]["req_last_value"] = m.group(2)
            state[w]["last_ts"] = ts
            continue
        m = SELECT_TOKENS_RE.search(line)
        if m:
            w = m.group(2)
            state[w]["token_last_action"] = "select"
            state[w]["token_last_value"] = m.group(3)
            state[w]["last_ts"] = ts
            continue
        m = RELEASE_TOKENS_RE.search(line)
        if m:
            w = m.group(2)
            state[w]["token_last_action"] = "release"
            state[w]["token_last_value"] = m.group(3)
            state[w]["last_ts"] = ts
            continue

    result = []
    for w in sorted(state.keys()):
        s = state[w]
        result.append({"worker": _strip_scheme(w), **s})
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


def _diagnose(load_stats, worker_load, anomaly_summary, sr_result, token_stats, pileup):
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

    id_cov = sr_result.get("id_coverage", {})
    has_correlatable_ids = (id_cov.get("with_request_id", 0) + id_cov.get("with_alt_id", 0)) > 0

    # Select/Release 不一致（仅在存在可关联 ID 时启用，避免无 ID 场景误报）
    if has_correlatable_ids:
        for w_url, pw in sr_result.get("per_worker", {}).items():
            if pw.get("delta", 0) > 0:
                diagnoses.append(
                    {
                        "severity": "HIGH",
                        "message": f'{_strip_scheme(w_url)} select-release 差值 {pw["delta"]}（请求泄漏/卡住）',
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

    # Token 计数器潜在泄漏
    for t in token_stats:
        if t.get("alloc_count", 0) > t.get("release_count", 0):
            diagnoses.append(
                {
                    "severity": "MEDIUM",
                    "message": f'{t["worker"]} token alloc/release 不平衡 ({t["alloc_count"]}/{t["release_count"]})',
                    "source_layer": "Router",
                }
            )

    return diagnoses


# ════════════════════════════════════════════════════════════════
# 报告格式化
# ════════════════════════════════════════════════════════════════




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
