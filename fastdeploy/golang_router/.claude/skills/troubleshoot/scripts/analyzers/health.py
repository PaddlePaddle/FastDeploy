#!/usr/bin/env python3
"""
Health Analyzer — Worker 健康时间线分析

追踪 Worker 上下线事件、恢复检测、可用性统计。
按 Worker URL 聚合事件，构建状态时间线。
"""

import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chart import render_table, render_timeline
from log_parser import extract_ts, parse_http_line, parse_ts

# ════════════════════════════════════════════════════════════════
# 健康事件解析
# ════════════════════════════════════════════════════════════════

WORKER_URL_RE = r"((?:https?://)?[A-Za-z0-9.-]+(?::\d+)?)"
NOT_HEALTHY_RE = re.compile(rf"{WORKER_URL_RE}\s+is not healthy")
REMOVED_RE = re.compile(rf"Removed unhealthy \w+ instance:\s*{WORKER_URL_RE}")
IS_HEALTHY_RE = re.compile(rf"{WORKER_URL_RE}\s+is healthy")
COUNTER_PRESERVED_RE = re.compile(rf"counter preserved.*?{WORKER_URL_RE}")
CLEANUP_UNHEALTHY_RE = re.compile(rf"cleanup unhealthy.*?{WORKER_URL_RE}")


def _strip_scheme(url):
    return re.sub(r"^https?://", "", url)


def parse_health_event(line):
    """解析 H2 健康事件行。返回 {ts, worker, event_type} 或 None。"""
    ts = extract_ts(line)
    m = REMOVED_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "event_type": "REMOVED"}
    m = NOT_HEALTHY_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "event_type": "NOT_HEALTHY"}
    m = IS_HEALTHY_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "event_type": "HEALTHY"}
    return None


def parse_counter_preserved(line):
    """解析 H5 counter preserved / cleanup 事件。"""
    ts = extract_ts(line)
    m = COUNTER_PRESERVED_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "event_type": "COUNTER_PRESERVED"}
    m = CLEANUP_UNHEALTHY_RE.search(line)
    if m:
        return {"ts": ts, "worker": m.group(1), "event_type": "CLEANUP_UNHEALTHY"}
    return None


# ════════════════════════════════════════════════════════════════
# 主分析函数
# ════════════════════════════════════════════════════════════════


def analyze_health(log_file, tail=None):
    """分析 Worker 健康状态。

    Returns:
        dict: {workers, diagnoses, time_range, summary}
    """
    h2_lines = _grep_lines(log_file, r"Removed unhealthy|is not healthy|is healthy", tail)
    h5_lines = _grep_lines(log_file, r"counter preserved|cleanup unhealthy", tail)
    register_lines = _grep_lines(log_file, r"\[POST\] /register", tail)

    health_events = [e for line in h2_lines for e in [parse_health_event(line)] if e]
    counter_events = [e for line in h5_lines for e in [parse_counter_preserved(line)] if e]

    register_events = []
    for line in register_lines:
        r = parse_http_line(line)
        if r and r["method"] == "POST" and r["path"] == "/register" and r["status"] == 200:
            register_events.append({"ts": r["ts"], "client_ip": r["client_ip"]})

    if not health_events and not register_events:
        return {
            "workers": {},
            "diagnoses": [],
            "time_range": {"start": "", "end": ""},
            "summary": "未检测到 Worker 健康事件",
        }

    workers = _build_worker_timelines(health_events, counter_events, register_events)

    all_ts = sorted([e["ts"] for e in health_events + register_events if e.get("ts")])
    time_range = {"start": all_ts[0] if all_ts else "", "end": all_ts[-1] if all_ts else ""}

    diagnoses = _diagnose(workers)
    down_workers = sum(1 for w in workers.values() if w["down_count"] > 0)

    return {
        "workers": workers,
        "diagnoses": diagnoses,
        "time_range": time_range,
        "summary": f"{len(workers)} Worker(s), {down_workers} 有下线事件",
    }


def _build_worker_timelines(health_events, counter_events, register_events):
    """构建每个 Worker 的状态时间线。"""
    worker_urls = {evt["worker"] for evt in health_events}

    # IP → worker URL 映射
    ip_to_urls = defaultdict(set)
    for url in worker_urls:
        ip_m = re.search(r"(?:https?://)?(\d+\.\d+\.\d+\.\d+)", url)
        if ip_m:
            ip_to_urls[ip_m.group(1)].add(url)

    worker_events = defaultdict(list)
    for evt in health_events:
        worker_events[evt["worker"]].append(evt)

    counter_counts = defaultdict(int)
    for evt in counter_events:
        if evt["event_type"] == "COUNTER_PRESERVED":
            counter_counts[evt["worker"]] += 1

    register_by_ip = defaultdict(list)
    for evt in register_events:
        register_by_ip[evt["client_ip"]].append(evt)

    workers = {}
    for url in sorted(worker_urls):
        events = sorted(worker_events[url], key=lambda e: e["ts"] or "")
        ip_m = re.search(r"(?:https?://)?(\d+\.\d+\.\d+\.\d+)", url)
        worker_ip = ip_m.group(1) if ip_m else ""

        # 恢复检测：REMOVED 后有 register
        recovered = False
        recovery_events = []
        for evt in events:
            if evt["event_type"] == "REMOVED" and worker_ip:
                for reg in register_by_ip.get(worker_ip, []):
                    if reg["ts"] and evt["ts"] and reg["ts"] > evt["ts"]:
                        recovered = True
                        recovery_events.append({"ts": reg["ts"], "type": "RE-REGISTERED"})
                        break

        all_events = [{"ts": e["ts"], "type": e["event_type"]} for e in events]
        for reg in register_by_ip.get(worker_ip, []):
            all_events.append({"ts": reg["ts"], "type": "REGISTERED"})
        all_events.extend(recovery_events)
        all_events.sort(key=lambda e: e["ts"] or "")

        down_periods = _compute_down_periods(all_events)
        down_count = len(down_periods)
        avg_down_s = (sum(p["duration_s"] for p in down_periods) / len(down_periods)) if down_periods else 0.0
        detect_latency = _compute_detect_latency(all_events)

        workers[url] = {
            "events": all_events,
            "uptime_pct": _compute_uptime_pct(all_events),
            "down_count": down_count,
            "avg_down_duration_s": round(avg_down_s, 1),
            "recovered": recovered,
            "inflight_preserved": counter_counts.get(url, 0),
            "down_periods": down_periods,
            "avg_detect_latency_s": detect_latency,
        }

    return workers


def _compute_down_periods(events):
    """从事件列表计算下线时段。"""
    down_periods = []
    down_start = None
    for evt in events:
        if evt["type"] in ("NOT_HEALTHY", "REMOVED"):
            if down_start is None and evt["ts"]:
                down_start = evt["ts"]
        elif evt["type"] in ("HEALTHY", "RE-REGISTERED"):
            if down_start is not None and evt["ts"]:
                try:
                    duration_s = (parse_ts(evt["ts"]) - parse_ts(down_start)).total_seconds()
                    down_periods.append({"start": down_start, "end": evt["ts"], "duration_s": max(0, duration_s)})
                except ValueError:
                    pass
                down_start = None
    if down_start is not None:
        down_periods.append({"start": down_start, "end": None, "duration_s": 0})
    return down_periods


def _compute_detect_latency(events):
    """计算 NOT_HEALTHY -> REMOVED 平均检测延迟（秒）。"""
    last_unhealthy = None
    latencies = []
    for evt in events:
        if evt["type"] == "NOT_HEALTHY" and evt.get("ts"):
            last_unhealthy = evt["ts"]
        elif evt["type"] == "REMOVED" and last_unhealthy and evt.get("ts"):
            try:
                latencies.append((parse_ts(evt["ts"]) - parse_ts(last_unhealthy)).total_seconds())
            except ValueError:
                pass
            last_unhealthy = None
    if not latencies:
        return "-"
    return round(sum(latencies) / len(latencies), 1)


def _compute_uptime_pct(events):
    """计算 Worker 可用性百分比。"""
    if not events:
        return 100.0
    ts_list = [e["ts"] for e in events if e["ts"]]
    if len(ts_list) < 2:
        return 0.0 if events[0]["type"] in ("NOT_HEALTHY", "REMOVED") else 100.0
    try:
        first_dt, last_dt = parse_ts(ts_list[0]), parse_ts(ts_list[-1])
        total_s = (last_dt - first_dt).total_seconds()
        if total_s <= 0:
            return 100.0
    except ValueError:
        return 100.0

    down_s, down_start = 0.0, None
    for evt in events:
        if evt["type"] in ("NOT_HEALTHY", "REMOVED") and down_start is None and evt["ts"]:
            try:
                down_start = parse_ts(evt["ts"])
            except ValueError:
                pass
        elif evt["type"] in ("HEALTHY", "RE-REGISTERED") and down_start is not None and evt["ts"]:
            try:
                down_s += (parse_ts(evt["ts"]) - down_start).total_seconds()
            except ValueError:
                pass
            down_start = None
    if down_start is not None:
        down_s += (last_dt - down_start).total_seconds()

    return round(max(0, total_s - down_s) / total_s * 100, 1)


def _diagnose(workers):
    """根据 Worker 健康数据生成诊断。"""
    diagnoses = []
    if not workers:
        return diagnoses

    all_down = all(w["events"] and w["events"][-1]["type"] in ("NOT_HEALTHY", "REMOVED") for w in workers.values())
    if all_down:
        diagnoses.append(
            {
                "severity": "CRITICAL",
                "message": f"所有 Worker ({len(workers)}) 当前均不可用",
                "source_layer": "FD 后端",
            }
        )

    for url, w in workers.items():
        s = _strip_scheme(url)
        if w["down_count"] > 3:
            diagnoses.append(
                {
                    "severity": "HIGH",
                    "message": f'{s} 下线 {w["down_count"]} 次，Worker 不稳定',
                    "source_layer": "FD 后端",
                }
            )
        for p in w.get("down_periods", []):
            if p["duration_s"] > 300:
                diagnoses.append(
                    {
                        "severity": "HIGH",
                        "message": f'{s} 下线 {p["duration_s"]/60:.1f}min（{p["start"]} ~ {p["end"] or "未恢复"}）',
                        "source_layer": "FD 后端",
                    }
                )
        if len(w["events"]) >= 3:
            ts_list = [e["ts"] for e in w["events"] if e["ts"]]
            if len(ts_list) >= 2:
                try:
                    hours = (parse_ts(ts_list[-1]) - parse_ts(ts_list[0])).total_seconds() / 3600
                    if hours > 0 and len(w["events"]) / hours > 3:
                        diagnoses.append(
                            {
                                "severity": "MEDIUM",
                                "message": f'{s} 状态变更频繁 ({len(w["events"])/hours:.1f} 次/小时)',
                                "source_layer": "FD 后端",
                            }
                        )
                except ValueError:
                    pass
        if w["inflight_preserved"] > 3:
            diagnoses.append(
                {
                    "severity": "MEDIUM",
                    "message": f'{s} counter preserved {w["inflight_preserved"]} 次（下线时仍有 inflight 请求）',
                    "source_layer": "FD 后端",
                }
            )

    return diagnoses


# ════════════════════════════════════════════════════════════════
# 报告格式化
# ════════════════════════════════════════════════════════════════


def format_health_report(result):
    """将分析结果格式化为终端报告。

    Returns:
        tuple: (summary_text, detail_text)
            summary_text: 总结部分（诊断 + 可用性表格 + 时间线）
            detail_text: 事件详情（逐条事件记录，可能很长）
    """
    sections = ["## Worker 健康分析", ""]
    if not result["workers"]:
        sections.append("  未检测到 Worker 健康事件（所有 Worker 状态正常或无健康日志）")
        return "\n".join(sections), ""

    sections.append(f'  {result["summary"]}')
    if result["time_range"]["start"]:
        sections.append(f'  时间范围: {result["time_range"]["start"]} ~ {result["time_range"]["end"]}')
    sections.append("")

    if result["diagnoses"]:
        sections.append("### 诊断")
        sections.append("")
        sections.append("  诊断见详情: [detail/health_events.md](detail/health_events.md)")
        sections.append("")

    # Worker 可用性表格
    sections.append("### Worker 可用性")
    sections.append("")
    table_data = []
    for url, w in sorted(result["workers"].items()):
        avg_down = ""
        if w["avg_down_duration_s"] > 0:
            avg_down = (
                f'{w["avg_down_duration_s"]/60:.1f}min'
                if w["avg_down_duration_s"] >= 60
                else f'{w["avg_down_duration_s"]:.0f}s'
            )
        table_data.append(
            {
                "Worker": _strip_scheme(url),
                "在线率": f'{w["uptime_pct"]}%',
                "下线次数": str(w["down_count"]),
                "平均下线时长": avg_down or "-",
                "检测延迟": (f'{w["avg_detect_latency_s"]}s' if w["avg_detect_latency_s"] != "-" else "-"),
                "恢复": "是" if w["recovered"] else ("否" if w["down_count"] > 0 else "-"),
                "inflight保留": str(w["inflight_preserved"]) if w["inflight_preserved"] > 0 else "-",
            }
        )
    sections.append(
        render_table(
            table_data,
            columns=["Worker", "在线率", "下线次数", "平均下线时长", "检测延迟", "恢复", "inflight保留"],
            right_align={"在线率", "下线次数", "平均下线时长", "检测延迟", "inflight保留"},
        )
    )
    sections.append("")

    # 时间线
    if result["time_range"]["start"] and result["time_range"]["end"]:
        sections.append("### Worker 时间线")
        sections.append("")
        timeline_data = _build_timeline_data(result)
        if timeline_data:
            sections.append(render_timeline(timeline_data, width=40))
            sections.append("")

    # 事件详情 → 拆分到 detail_text
    detail_parts = ["# Worker 健康事件详情", ""]
    has_events = False
    if result.get("diagnoses"):
        detail_parts.append("## 诊断")
        detail_parts.append("")
        for d in result["diagnoses"]:
            detail_parts.append(f'[{d["severity"]}] [{d["source_layer"]}] {d["message"]}')
        detail_parts.append("")
    for url, w in sorted(result["workers"].items()):
        if w["events"]:
            has_events = True
            detail_parts.append(f"## {_strip_scheme(url)}")
            detail_parts.append("")
            for evt in w["events"]:
                detail_parts.append(f'  [{evt["ts"]}] {evt["type"]}')
            detail_parts.append("")

    detail_text = "\n".join(detail_parts) if has_events else ""

    # 主报告中添加引用
    if has_events:
        sections.append("> 完整事件详情: [detail/health_events.md](detail/health_events.md)")
        sections.append("")

    return "\n".join(sections), detail_text


def _build_timeline_data(result):
    """构建 render_timeline 需要的数据格式。"""
    tr = result["time_range"]
    if not tr["start"] or not tr["end"]:
        return None
    workers_data = {}
    for url, w in result["workers"].items():
        periods = []
        status, start = "up", tr["start"]
        for evt in w["events"]:
            if not evt["ts"]:
                continue
            if evt["type"] in ("NOT_HEALTHY", "REMOVED") and status == "up":
                periods.append({"from": start, "to": evt["ts"], "status": "up"})
                status, start = "down", evt["ts"]
            elif evt["type"] in ("HEALTHY", "RE-REGISTERED") and status == "down":
                periods.append({"from": start, "to": evt["ts"], "status": "down"})
                status, start = "up", evt["ts"]
        periods.append({"from": start, "to": tr["end"], "status": status})
        workers_data[url] = periods
    return {"start": tr["start"], "end": tr["end"], "workers": workers_data}


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
