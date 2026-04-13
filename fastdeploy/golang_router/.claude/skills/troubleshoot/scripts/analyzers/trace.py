#!/usr/bin/env python3
"""
Trace Analyzer — 请求追踪

通过 session_id / trace_id / request_id / req_id 追踪单个或多个请求的
完整生命周期，重建事件链，检测不完整生命周期。
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from log_parser import (
    extract_tags,
    extract_ts,
    match_select_release,
    parse_cache_strategy_line,
    parse_http_line,
)

# ════════════════════════════════════════════════════════════════
# 事件识别正则
# ════════════════════════════════════════════════════════════════

PARSING_COMPLETE_RE = re.compile(r"Parsing completed.*worker selection")
URL_RE = r"((?:https?://)?[A-Za-z0-9.-]+(?::\d+)?)"
SELECT_WORKER_RE = re.compile(rf"select worker\s*(?:\((\w+)\))?:\s*{URL_RE}")
RELEASE_WORKER_RE = re.compile(rf"release worker\s*(?:\((\w+)\))?:\s*{URL_RE}")
RELEASE_TOKENS_RE = re.compile(rf"release prefill tokens:\s*{URL_RE},\s*tokens:\s*(\d+)")
REQUEST_COMPLETE_RE = re.compile(r"Request completed successfully")
TS_MS_RE = re.compile(r"ts_ms=(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")

# Prefill 事件
PREFILL_FIRST_CHUNK_RE = re.compile(rf"\[prefill\] first chunk received.*?{URL_RE}")
PREFILL_DONE_RE = re.compile(rf"\[prefill\] non-stream prefill response done.*?{URL_RE}")
PREFILL_ERROR_RE = re.compile(rf"\[prefill\] (scanner error|copy error).*?{URL_RE}")
PREFILL_DEFER_RE = re.compile(rf"\[prefill\] release in defer.*?{URL_RE}")
PREFILL_ERR_PATH_RE = re.compile(rf"\[prefill\] release in CommonCompletions defer \(error path\).*?{URL_RE}")
FAILED_SELECT_RE = re.compile(r"Failed to select")


def _strip_scheme(url):
    return re.sub(r"^https?://", "", url)


# ════════════════════════════════════════════════════════════════
# 主分析函数
# ════════════════════════════════════════════════════════════════


def analyze_trace(log_file, trace_ids, tail=None):
    """追踪指定 ID 的请求生命周期。

    Args:
        log_file: 日志文件路径
        trace_ids: ID 列表（逗号分隔的字符串或列表）
        tail: 尾部行数限制

    Returns:
        dict: {traces: {id: {events, lifecycle_complete, diagnoses}}, summary}
    """
    if isinstance(trace_ids, str):
        trace_ids = [tid.strip() for tid in trace_ids.split(",") if tid.strip()]

    if not trace_ids:
        return {"traces": {}, "summary": "未指定追踪 ID"}

    traces = {}
    for tid in trace_ids:
        # Grep 搜索四种 context tag
        pattern = f"session_id:{tid}|trace_id:{tid}|request_id:{tid}|req_id:{tid}"
        matching_lines = _grep_lines(log_file, pattern, tail)

        if not matching_lines:
            traces[tid] = {
                "events": [],
                "lifecycle_complete": False,
                "diagnoses": [{"severity": "INFO", "message": f"未找到 ID={tid} 的匹配行"}],
                "matched_tag": None,
                "related_ids": {},
            }
            continue

        # 识别匹配到的 tag 类型，并展开 session 下所有 request_id
        first_tags = extract_tags(matching_lines[0])
        is_session = tid in [first_tags.get("session_id", "")]

        # 如果是 session_id，收集所有关联的 request_id
        related_request_ids = set()
        if is_session:
            for line in matching_lines:
                tags = extract_tags(line)
                rid = tags.get("request_id", "")
                if rid:
                    related_request_ids.add(rid)

            # 为每个 request_id 额外搜索行
            extra_lines = []
            for rid in related_request_ids:
                rid_lines = _grep_lines(log_file, f"request_id:{rid}", tail)
                extra_lines.extend(rid_lines)
            all_lines = list(set(matching_lines + extra_lines))
        else:
            all_lines = matching_lines

        # 解析事件链
        events = _parse_event_chain(all_lines)
        lifecycle_complete = _check_lifecycle_complete(events)
        sr_check = match_select_release(all_lines)
        diagnoses = _diagnose_trace(events, lifecycle_complete, sr_check)

        tag_coverage = _build_id_coverage_stats(all_lines)
        tag_combos = _build_id_combo_stats(all_lines)
        matched_tags = _detect_matched_tags(all_lines, tid)
        traces[tid] = {
            "events": events,
            "lifecycle_complete": lifecycle_complete,
            "diagnoses": diagnoses,
            "sr_check": sr_check,
            "matched_tag": _format_matched_tag(matched_tags),
            "matched_tags": matched_tags,
            "related_ids": {
                "request_ids": sorted(related_request_ids) if is_session else [],
            },
            "id_coverage": tag_coverage,
            "id_combos": tag_combos,
        }

    total_traced = len(traces)
    complete = sum(1 for t in traces.values() if t["lifecycle_complete"])

    return {
        "traces": traces,
        "summary": f"{total_traced} ID(s) 追踪, {complete} 生命周期完整",
    }


def _parse_event_chain(lines):
    """从匹配行重建事件链，按时间排序。"""
    events = []

    for line in lines:
        ts = extract_ts(line)
        tags = extract_tags(line)

        # HTTP 请求进入/完成
        http = parse_http_line(line)
        if http:
            events.append(
                {
                    "ts": ts,
                    "type": "HTTP",
                    "tags": tags,
                    "method": http["method"],
                    "path": http["path"],
                    "status": http["status"],
                    "latency_ms": http["latency_ms"],
                    "raw": line.strip(),
                }
            )
            continue

        # Parsing completed
        if PARSING_COMPLETE_RE.search(line):
            events.append({"ts": ts, "type": "PARSING_COMPLETE", "tags": tags, "raw": line.strip()})
            continue

        # Cache-aware strategy
        strategy = parse_cache_strategy_line(line)
        if strategy:
            events.append(
                {
                    "ts": ts,
                    "type": "CACHE_STRATEGY",
                    "tags": tags,
                    "strategy": strategy.get("strategy"),
                    "selected": strategy.get("selected", ""),
                    "selected_hitRatio": strategy.get("selected_hitRatio", 0),
                    "raw": line.strip(),
                }
            )
            continue

        # Select worker
        m = SELECT_WORKER_RE.search(line)
        if m:
            events.append(
                {
                    "ts": ts,
                    "type": "SELECT_WORKER",
                    "tags": tags,
                    "worker_type": m.group(1) or "unknown",
                    "worker": m.group(2),
                    "raw": line.strip(),
                }
            )
            continue

        # Release worker
        m = RELEASE_WORKER_RE.search(line)
        if m:
            events.append(
                {
                    "ts": ts,
                    "type": "RELEASE_WORKER",
                    "tags": tags,
                    "worker_type": m.group(1) or "unknown",
                    "worker": m.group(2),
                    "raw": line.strip(),
                }
            )
            continue

        # Release tokens
        m = RELEASE_TOKENS_RE.search(line)
        if m:
            events.append(
                {
                    "ts": ts,
                    "type": "RELEASE_TOKENS",
                    "tags": tags,
                    "worker": m.group(1),
                    "tokens": int(m.group(2)),
                    "raw": line.strip(),
                }
            )
            continue

        # Prefill events
        m = PREFILL_FIRST_CHUNK_RE.search(line)
        if m:
            events.append({"ts": ts, "type": "PREFILL_FIRST_CHUNK", "tags": tags, "worker": m.group(1), "raw": line.strip()})
            continue
        m = PREFILL_DONE_RE.search(line)
        if m:
            events.append({"ts": ts, "type": "PREFILL_DONE", "tags": tags, "worker": m.group(1), "raw": line.strip()})
            continue
        m = PREFILL_ERROR_RE.search(line)
        if m:
            events.append(
                {"ts": ts, "type": "PREFILL_ERROR", "tags": tags, "error": m.group(1), "worker": m.group(2), "raw": line.strip()}
            )
            continue
        m = PREFILL_DEFER_RE.search(line)
        if m:
            events.append(
                {"ts": ts, "type": "PREFILL_DEFER_RELEASE", "tags": tags, "worker": m.group(1), "raw": line.strip()}
            )
            continue
        m = PREFILL_ERR_PATH_RE.search(line)
        if m:
            events.append(
                {"ts": ts, "type": "PREFILL_ERROR_PATH_RELEASE", "tags": tags, "worker": m.group(1), "raw": line.strip()}
            )
            continue

        # Request completed
        if REQUEST_COMPLETE_RE.search(line):
            events.append({"ts": ts, "type": "REQUEST_COMPLETE", "tags": tags, "raw": line.strip()})
            continue

        # ts_ms
        m = TS_MS_RE.search(line)
        if m:
            events.append({"ts": ts, "type": "TS_MS", "tags": tags, "ts_ms": m.group(1), "raw": line.strip()})
            continue

        # Failed to select
        if FAILED_SELECT_RE.search(line):
            events.append({"ts": ts, "type": "FAILED_SELECT", "tags": tags, "raw": line.strip()})
            continue

    # 按时间排序
    events.sort(key=lambda e: e.get("ts") or "")
    return events


def _check_lifecycle_complete(events):
    """检查生命周期是否完整。"""
    types = {e["type"] for e in events}
    has_entry = "HTTP" in types or "PARSING_COMPLETE" in types
    has_exit = "REQUEST_COMPLETE" in types or (
        "HTTP" in types and any(e["type"] == "HTTP" and e.get("status") for e in events)
    )
    has_select = "SELECT_WORKER" in types
    has_release = "RELEASE_WORKER" in types

    return has_entry and has_exit and (not has_select or has_release)


def _diagnose_trace(events, lifecycle_complete, sr_check=None):
    """生成追踪诊断。"""
    diagnoses = []
    types = [e["type"] for e in events]

    if not lifecycle_complete:
        if "SELECT_WORKER" in types and "RELEASE_WORKER" not in types:
            diagnoses.append({"severity": "HIGH", "message": "有 select 但无 release — 疑似请求卡住"})
        elif "HTTP" not in types and "PARSING_COMPLETE" not in types:
            diagnoses.append({"severity": "MEDIUM", "message": "未找到 HTTP 入口事件"})
        elif "REQUEST_COMPLETE" not in types:
            diagnoses.append({"severity": "MEDIUM", "message": "未检测到请求完成事件 — 疑似异常退出"})

    if "PREFILL_ERROR" in types:
        for e in events:
            if e["type"] == "PREFILL_ERROR":
                diagnoses.append(
                    {"severity": "HIGH", "message": f'Prefill 错误: {e.get("error","")} @ {e.get("worker","")}'}
                )

    if "FAILED_SELECT" in types:
        diagnoses.append({"severity": "HIGH", "message": "Failed to select worker — 无可用 Worker"})

    if sr_check:
        if sr_check.get("unmatched_selects"):
            diagnoses.append(
                {
                    "severity": "HIGH",
                    "message": f'match-select-release 检测到 {len(sr_check["unmatched_selects"])} 个 unmatched select',
                }
            )
        if sr_check.get("unmatched_releases"):
            diagnoses.append(
                {
                    "severity": "MEDIUM",
                    "message": f'match-select-release 检测到 {len(sr_check["unmatched_releases"])} 个 unmatched release',
                }
            )

    return diagnoses


# ════════════════════════════════════════════════════════════════
# 报告格式化
# ════════════════════════════════════════════════════════════════


def format_trace_report(result):
    """将追踪结果格式化为终端报告。

    Returns:
        tuple: (summary_text, detail_dict)
            summary_text: 总结部分（概览 + 诊断 + 生命周期状态）
            detail_dict: {trace_id: event_chain_text} 各 ID 的完整事件链
    """
    sections = ["## 请求追踪", ""]
    sections.append(f'  {result["summary"]}')
    sections.append("")

    detail_dict = {}

    for tid, trace in result["traces"].items():
        sections.append(f"### ID: {tid}")
        if trace.get("matched_tag"):
            sections.append(f'  匹配类型: {trace["matched_tag"]}')
        if trace.get("id_coverage"):
            c = trace["id_coverage"]
            sections.append(
                "  ID统计: "
                f'request_only={c["request_only"]}, session_only={c["session_only"]}, trace_only={c["trace_only"]}'
            )
        if trace.get("related_ids", {}).get("request_ids"):
            sections.append(f'  关联 request_ids: {", ".join(trace["related_ids"]["request_ids"])}')

        status = "完整" if trace["lifecycle_complete"] else "不完整"
        sections.append(f"  生命周期: {status}")
        sections.append("")

        # 诊断
        if trace["diagnoses"]:
            for d in trace["diagnoses"]:
                sections.append(f'  [{d["severity"]}] {d["message"]}')
            sections.append("")

        # 事件链 → 拆分到 detail_dict
        if trace["events"]:
            detail_lines = [f"# 请求追踪事件链: {tid}", ""]
            if trace.get("matched_tag"):
                detail_lines.append(f'匹配类型: {trace["matched_tag"]}')
            if trace.get("id_coverage"):
                c = trace["id_coverage"]
                detail_lines.append("ID覆盖统计:")
                detail_lines.append(
                    f'- only_request_id: {c["request_only"]} | only_session_id: {c["session_only"]} | only_trace_id: {c["trace_only"]}'
                )
            if trace.get("id_combos"):
                detail_lines.append("")
                detail_lines.append("标签组合明细（按唯一ID计数）:")
                for item in trace["id_combos"]:
                    detail_lines.append(
                        f'- combo={item["combo"]} | count={item["count"]} | ids={", ".join(item["ids"])}'
                    )
            if trace.get("related_ids", {}).get("request_ids"):
                detail_lines.append(f'关联 request_ids: {", ".join(trace["related_ids"]["request_ids"])}')
            detail_lines.append(f"生命周期: {status}")
            detail_lines.append("")
            detail_lines.append("## 事件链")
            detail_lines.append("")
            for evt in trace["events"]:
                line = f'  [{evt.get("ts","")}] {evt["type"]}'
                if evt.get("worker"):
                    line += f' → {_strip_scheme(evt["worker"])}'
                if evt.get("status"):
                    line += f' [{evt["status"]}]'
                if evt.get("latency_ms"):
                    line += f' {evt["latency_ms"]}ms'
                if evt.get("strategy"):
                    line += f' strategy={evt["strategy"]}'
                if evt.get("selected_hitRatio"):
                    line += f' hitRatio={evt["selected_hitRatio"]}'
                if evt.get("tokens"):
                    line += f' tokens={evt["tokens"]}'
                if evt.get("error"):
                    line += f' error={evt["error"]}'
                if evt.get("ts_ms"):
                    line += f' ts_ms={evt["ts_ms"]}'
                detail_lines.append(line)
                if evt.get("raw"):
                    detail_lines.append(f'    RAW: {evt["raw"]}')
            detail_lines.append("")
            detail_dict[tid] = "\n".join(detail_lines)

            # 主报告中添加引用和摘要
            safe_tid = tid.replace("/", "_")
            sections.append(f'  事件数: {len(trace["events"])}')
            sections.append(f"  > 完整事件链: [detail/trace_{safe_tid}.md](../detail/trace_{safe_tid}.md)")
            sections.append("")

    return "\n".join(sections), detail_dict


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


def _detect_matched_tags(lines, target_id):
    matched = set()
    for line in lines:
        tags = extract_tags(line)
        for key in ("request_id", "trace_id", "session_id", "req_id"):
            if tags.get(key) == target_id:
                matched.add(key)
    return sorted(matched)


def _format_matched_tag(matched_tags):
    if not matched_tags:
        return "unknown"
    if len(matched_tags) == 1:
        return matched_tags[0]
    return "+".join(matched_tags)


def _build_id_coverage_stats(lines):
    request_only_ids = set()
    session_only_ids = set()
    trace_only_ids = set()

    for line in lines:
        tags = extract_tags(line)
        req_val = tags.get("request_id") or tags.get("req_id")
        session_val = tags.get("session_id")
        trace_val = tags.get("trace_id")
        has_request = bool(req_val)
        has_session = bool(session_val)
        has_trace = bool(trace_val)

        if has_request and not has_session and not has_trace:
            request_only_ids.add(req_val)
        if has_session and not has_request and not has_trace:
            session_only_ids.add(session_val)
        if has_trace and not has_request and not has_session:
            trace_only_ids.add(trace_val)

    return {
        "request_only": len(request_only_ids),
        "session_only": len(session_only_ids),
        "trace_only": len(trace_only_ids),
    }


def _build_id_combo_stats(lines):
    combo_to_ids = {}
    for line in lines:
        tags = extract_tags(line)
        keys = []
        if tags.get("request_id"):
            keys.append("request_id")
        if tags.get("req_id"):
            keys.append("req_id")
        if tags.get("session_id"):
            keys.append("session_id")
        if tags.get("trace_id"):
            keys.append("trace_id")
        combo = "+".join(keys) if keys else "no_id_tag"

        ids = []
        if tags.get("request_id"):
            ids.append(tags["request_id"])
        if tags.get("req_id"):
            ids.append(tags["req_id"])
        if tags.get("session_id"):
            ids.append(tags["session_id"])
        if tags.get("trace_id"):
            ids.append(tags["trace_id"])
        id_key = "|".join(ids) if ids else "<none>"

        combo_to_ids.setdefault(combo, set()).add(id_key)

    rows = []
    for combo, ids in combo_to_ids.items():
        rows.append({"combo": combo, "count": len(ids), "ids": sorted(ids)})
    rows.sort(key=lambda x: x["count"], reverse=True)
    return rows
