#!/usr/bin/env python3
"""
Troubleshoot — FastDeploy Go Router 综合问题排查主编排器

Usage:
    python3 troubleshoot.py <log_file> [options]

Options:
    --errors            仅分析错误日志
    --latency           仅分析延迟
    --health            仅分析 Worker 健康
    --cache             仅分析 Cache 调度
    --load              仅分析负载与计数器
    --trace ID          追踪指定请求（支持逗号分隔多 ID）
    --tail N            仅分析尾部 N 行（支持 N 或 Nm 格式如 30m）
    --start TIME        起始时间（如 "16:00:00"、"03/31 16:00"）
    --end TIME          结束时间（如 "17:00:00"、"2026/03/31 17:00:00"）
    --output DIR        详细报告导出目录（默认: skill_output/troubleshoot/<timestamp>/）

支持维度：errors, latency, health, cache, load, trace
"""

import argparse
import os
import sys
from datetime import datetime

# 确保能 import 同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzers.cache import analyze_cache, format_cache_report
from analyzers.errors import analyze_errors, format_errors_report
from analyzers.health import analyze_health, format_health_report
from analyzers.latency import analyze_latency, format_latency_report
from analyzers.load import analyze_load, format_load_report
from analyzers.trace import analyze_trace, format_trace_report
from log_parser import complete_time_arg, filter_file_by_recent_minutes, filter_file_by_time_range


def determine_log_file(user_path=None):
    """确定日志文件路径。

    搜索顺序：
    1. 用户指定路径（直接使用，不质疑）
    2. logs/router.log
    3. fd-router.log（golang_router 根目录）
    """
    if user_path:
        if os.path.isfile(user_path):
            return user_path
        print(f"ERROR: 文件不存在: {user_path}", file=sys.stderr)
        sys.exit(1)

    # 尝试不同 CWD 下的候选路径
    candidates = [
        "logs/router.log",  # CWD = golang_router/
        "fd-router.log",  # CWD = golang_router/
        "fastdeploy/golang_router/logs/router.log",  # CWD = 项目根
        "fastdeploy/golang_router/fd-router.log",  # CWD = 项目根
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    print("ERROR: 未找到日志文件。请指定路径或检查 logs/ 目录。", file=sys.stderr)
    sys.exit(1)


def parse_tail_arg(tail_str):
    """解析 --tail 参数：支持纯数字(行数)或 Nm(分钟)格式。"""
    if tail_str is None:
        return None
    if tail_str.endswith("m"):
        return {"type": "minutes", "value": int(tail_str[:-1])}
    return {"type": "lines", "value": int(tail_str)}


def determine_status(results):
    """根据分析结果判定全局状态。"""
    reasons = []

    # Errors 维度
    errors_result = results.get("errors")
    if errors_result:
        if errors_result["panic_list"]:
            return "CRITICAL", f'{len(errors_result["panic_list"])} Panic 事件'
        if errors_result["error_rate"] > 20:
            return "CRITICAL", f'错误率 {errors_result["error_rate"]}%'
        if errors_result["error_rate"] > 5:
            reasons.append(f'错误率 {errors_result["error_rate"]}%')
        for s in errors_result["status_code_dist"]:
            code = str(s["value"])
            if code in ("502", "503") and s["count"] > 0:
                reasons.append(f'{code}: {s["count"]}')

    # Latency 维度
    latency_result = results.get("latency")
    if latency_result:
        for d in latency_result.get("diagnoses", []):
            if d["severity"] == "CRITICAL":
                return "CRITICAL", d["message"]
            if d["severity"] == "HIGH":
                reasons.append(d["message"])

    # Health 维度
    health_result = results.get("health")
    if health_result:
        for d in health_result.get("diagnoses", []):
            if d["severity"] == "CRITICAL":
                return "CRITICAL", d["message"]
            if d["severity"] == "HIGH":
                reasons.append(d["message"])

    # Load 维度
    load_result = results.get("load")
    if load_result:
        for d in load_result.get("diagnoses", []):
            if d["severity"] == "CRITICAL":
                return "CRITICAL", d["message"]
            if d["severity"] == "HIGH":
                reasons.append(d["message"])

    # Cache 维度
    cache_result = results.get("cache")
    if cache_result:
        for d in cache_result.get("diagnoses", []):
            if d["severity"] == "HIGH":
                reasons.append(d["message"])

    if reasons:
        # 去重并限制长度，避免状态行过长难读
        deduped = []
        seen = set()
        for r in reasons:
            if r not in seen:
                deduped.append(r)
                seen.add(r)
        max_reasons = 4
        shown = deduped[:max_reasons]
        extra = len(deduped) - len(shown)
        summary = "；".join(shown)
        if extra > 0:
            summary += f"；另有 {extra} 项诊断见各维度 detail 报告"
        return "DEGRADED", summary

    if not results:
        return "HEALTHY", "无分析数据"

    return "HEALTHY", "无严重问题"


def format_full_report(results, status, status_reason):
    """组装完整报告。

    Returns:
        tuple: (report_text, details)
            report_text: 主报告文本（总结 + 可视化）
            details: dict 包含需要拆分到独立文件的详情数据
                - 'health_events': str 或 None
                - 'load_select_release': str 或 None
                - 'trace_files': {trace_id: text} 或 {}
    """
    parts = []
    details = {
        "health_events": None,
        "load_select_release": None,
        "latency_diagnoses": None,
        "cache_diagnosis": None,
        "load_diagnoses": None,
        "load_counter_state": None,
        "cache_session_stickiness": None,
        "cache_suboptimal": None,
        "cache_eviction": None,
        "cache_fallback": None,
        "cache_cross": None,
        "trace_files": {},
    }

    # 状态行
    parts.append(f"STATUS: {status} — {status_reason}")
    parts.append(
        "状态定义: HEALTHY=无明显异常；DEGRADED=服务可用但存在性能/稳定性问题（需关注）；CRITICAL=服务不可用或高风险故障。"
    )
    parts.append("=" * 60)
    parts.append("")

    # 各维度报告
    if "errors" in results:
        parts.append(format_errors_report(results["errors"]))

    if "latency" in results:
        parts.append(format_latency_report(results["latency"]))
        if results["latency"].get("diagnoses"):
            lines = ["# 延迟诊断详情", ""]
            for d in results["latency"]["diagnoses"]:
                lines.append(f'[{d.get("severity","")}] {d.get("message","")}')
            lines.append("")
            details["latency_diagnoses"] = "\n".join(lines)

    if "health" in results:
        summary, detail = format_health_report(results["health"])
        parts.append(summary)
        if detail:
            details["health_events"] = detail

    if "load" in results:
        summary, detail = format_load_report(results["load"])
        parts.append(summary)
        if detail:
            details["load_select_release"] = detail
        if results["load"].get("diagnoses"):
            lines = ["# Load 诊断详情", ""]
            for d in results["load"]["diagnoses"]:
                lines.append(f'[{d.get("severity","")}] [{d.get("source_layer","")}] {d.get("message","")}')
            lines.append("")
            details["load_diagnoses"] = "\n".join(lines)
        if results["load"].get("counter_last_state"):
            rows = results["load"]["counter_last_state"]
            lines = ["# Load Counter 末状态", "", "| worker | req_last_action | req_last_value | token_last_action | token_last_value | last_ts |", "|:--|:--|--:|:--|--:|:--|"]
            for r in rows:
                lines.append(
                    f'| {r.get("worker","")} | {r.get("req_last_action","-")} | {r.get("req_last_value","-")} | {r.get("token_last_action","-")} | {r.get("token_last_value","-")} | {r.get("last_ts","")} |'
                )
            lines.append("")
            details["load_counter_state"] = "\n".join(lines)

    if "cache" in results:
        summary, detail = format_cache_report(results["cache"])
        parts.append(summary)
        if detail:
            details["cache_diagnosis"] = detail
        c = results["cache"]
        if c.get("session_stickiness"):
            lines = ["# Cache Session 粘性详情", ""]
            for sid, s in c["session_stickiness"].items():
                lines.append(f'- {sid}: req={s.get("total_requests",0)}, stickiness={s.get("stickiness_pct",0)}%, switches={s.get("switches",0)}')
            lines.append("")
            details["cache_session_stickiness"] = "\n".join(lines)
        if c.get("suboptimal_selections"):
            lines = ["# Cache 非最优选择详情", ""]
            for x in c["suboptimal_selections"][:200]:
                lines.append(f'- [{x.get("ts","")}] selected={x.get("selected","")} best={x.get("best_hr_worker","")} reason={x.get("reason","")}')
            lines.append("")
            details["cache_suboptimal"] = "\n".join(lines)
        if c.get("eviction_impact"):
            lines = ["# Cache 驱逐影响详情", ""]
            for x in c["eviction_impact"][:200]:
                lines.append(f'- session={x.get("session_id","")} interval={x.get("interval_mins",0)}m hitRatio_after={x.get("hitRatio_after",0)} evicted={x.get("evicted",False)}')
            lines.append("")
            details["cache_eviction"] = "\n".join(lines)
        if c.get("fallback_reasons"):
            lines = ["# Cache Fallback 原因详情", ""]
            for x in c["fallback_reasons"]:
                lines.append(f'- {x.get("value","")}: {x.get("count",0)} ({x.get("pct",0)}%)')
            lines.append("")
            details["cache_fallback"] = "\n".join(lines)
        if c.get("cross_diagnosis"):
            lines = ["# Cache 交叉诊断详情", ""]
            for x in c["cross_diagnosis"]:
                lines.append(f'- diagnosis={x.get("diagnosis","")}, action={x.get("action","")}, avg_stickiness={x.get("avg_stickiness_pct",0)}%')
            lines.append("")
            details["cache_cross"] = "\n".join(lines)

    if "trace" in results:
        summary, detail_dict = format_trace_report(results["trace"])
        parts.append(summary)
        if detail_dict:
            details["trace_files"] = detail_dict

    return "\n".join(parts), details


def save_detailed_report(report_text, output_dir, details=None):
    """保存报告到文件。

    Args:
        report_text: 主报告文本
        output_dir: 输出目录
        details: 详情数据 dict（来自 format_full_report）
    """
    summary_dir = os.path.join(output_dir, "summary")
    detail_dir = os.path.join(output_dir, "detail")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)
    filepath = os.path.join(summary_dir, "troubleshoot_report.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Router Troubleshooting Report\n")
        f.write(f'> Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(report_text)

    # 保存详情到 detail/ 子目录
    if details:
        if details.get("health_events"):
            health_path = os.path.join(detail_dir, "health_events.md")
            with open(health_path, "w", encoding="utf-8") as f:
                f.write(details["health_events"])

        if details.get("load_select_release"):
            load_path = os.path.join(detail_dir, "load_select_release.md")
            with open(load_path, "w", encoding="utf-8") as f:
                f.write(details["load_select_release"])

        if details.get("latency_diagnoses"):
            latency_path = os.path.join(detail_dir, "latency_diagnoses.md")
            with open(latency_path, "w", encoding="utf-8") as f:
                f.write(details["latency_diagnoses"])

        if details.get("cache_diagnosis"):
            cache_path = os.path.join(detail_dir, "cache_diagnosis.md")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(details["cache_diagnosis"])
        if details.get("load_diagnoses"):
            with open(os.path.join(detail_dir, "load_diagnoses.md"), "w", encoding="utf-8") as f:
                f.write(details["load_diagnoses"])
        if details.get("load_counter_state"):
            with open(os.path.join(detail_dir, "load_counter_state.md"), "w", encoding="utf-8") as f:
                f.write(details["load_counter_state"])
        if details.get("cache_session_stickiness"):
            with open(os.path.join(detail_dir, "cache_session_stickiness.md"), "w", encoding="utf-8") as f:
                f.write(details["cache_session_stickiness"])
        if details.get("cache_suboptimal"):
            with open(os.path.join(detail_dir, "cache_suboptimal.md"), "w", encoding="utf-8") as f:
                f.write(details["cache_suboptimal"])
        if details.get("cache_eviction"):
            with open(os.path.join(detail_dir, "cache_eviction.md"), "w", encoding="utf-8") as f:
                f.write(details["cache_eviction"])
        if details.get("cache_fallback"):
            with open(os.path.join(detail_dir, "cache_fallback.md"), "w", encoding="utf-8") as f:
                f.write(details["cache_fallback"])
        if details.get("cache_cross"):
            with open(os.path.join(detail_dir, "cache_cross.md"), "w", encoding="utf-8") as f:
                f.write(details["cache_cross"])

        for trace_id, trace_text in details.get("trace_files", {}).items():
            safe_id = trace_id.replace("/", "_")
            trace_path = os.path.join(detail_dir, f"trace_{safe_id}.md")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(trace_text)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="FastDeploy Go Router Troubleshooting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("log_file", nargs="?", help="日志文件路径")
    parser.add_argument("--errors", action="store_true", help="仅分析错误日志")
    parser.add_argument("--latency", action="store_true", help="仅分析延迟")
    parser.add_argument("--health", action="store_true", help="仅分析 Worker 健康")
    parser.add_argument("--cache", action="store_true", help="仅分析 Cache 调度")
    parser.add_argument("--load", action="store_true", help="仅分析负载与计数器")
    parser.add_argument("--trace", metavar="ID", help="追踪指定请求（逗号分隔多 ID）")
    parser.add_argument("--tail", help="尾部行数或分钟数 (如 5000 或 30m)")
    parser.add_argument(
        "--start", default=None, help='起始时间（如 "16:00:00"、"03/31 16:00"、"2026/03/31 16:00:00"）'
    )
    parser.add_argument("--end", default=None, help='结束时间（如 "17:00:00"、"03/31 17:00"、"2026/03/31 17:00:00"）')
    parser.add_argument("--output", help="详细报告导出目录（默认：skill_output/troubleshoot/<timestamp>/）")

    args = parser.parse_args()

    # 确定日志文件
    log_file = determine_log_file(args.log_file)
    print(f"日志文件: {log_file}", file=sys.stderr)

    # --tail 与 --start/--end 不能混用（两者是不同的范围选择方式）
    if args.tail and (args.start or args.end):
        print("Error: --tail 与 --start/--end 不能同时使用，请选择其一", file=sys.stderr)
        sys.exit(1)

    # 时间范围预过滤（--start 和 --end 可单独或同时指定）
    import atexit

    if args.start or args.end:
        start_ts = complete_time_arg(args.start, log_file, is_end=False) if args.start else None
        end_ts = complete_time_arg(args.end, log_file, is_end=True) if args.end else None
        filtered_path, is_temp = filter_file_by_time_range(log_file, start_ts, end_ts)
        if is_temp:
            atexit.register(lambda p=filtered_path: os.unlink(p) if os.path.exists(p) else None)
        log_file = filtered_path
        print(f'时间范围过滤: {start_ts or "..."} ~ {end_ts or "..."}', file=sys.stderr)

    tail_arg = parse_tail_arg(args.tail)
    tail = None
    # --tail Nm 采用真实时间窗口过滤，再全量分析过滤后的临时文件
    if tail_arg and tail_arg["type"] == "minutes":
        filtered_path, is_temp = filter_file_by_recent_minutes(log_file, tail_arg["value"])
        if is_temp:
            atexit.register(lambda p=filtered_path: os.unlink(p) if os.path.exists(p) else None)
        log_file = filtered_path
        print(f"--tail {tail_arg['value']}m: 使用日志时间戳过滤最近窗口", file=sys.stderr)
    elif tail_arg and tail_arg["type"] == "lines":
        tail = tail_arg["value"]

    # 确定分析模式
    any_mode = args.errors or args.latency or args.health or args.cache or args.load or args.trace
    run_errors = args.errors or (not any_mode)
    run_latency = args.latency or (not any_mode)
    run_health = args.health or (not any_mode)
    run_load = args.load or (not any_mode)
    run_cache = args.cache or (not any_mode)
    run_trace = bool(args.trace)  # trace 需要指定 ID，全量扫描不自动调用

    results = {}
    step = 0
    total_steps = sum([run_errors, run_latency, run_health, run_cache, run_load, run_trace])

    # 执行分析
    if run_errors:
        step += 1
        print(f"[{step}/{total_steps}] 分析错误日志...", file=sys.stderr)
        results["errors"] = analyze_errors(log_file, tail=tail)

    if run_latency:
        step += 1
        print(f"[{step}/{total_steps}] 分析请求延迟...", file=sys.stderr)
        results["latency"] = analyze_latency(log_file, tail=tail)

    if run_health:
        step += 1
        print(f"[{step}/{total_steps}] 分析 Worker 健康...", file=sys.stderr)
        results["health"] = analyze_health(log_file, tail=tail)

    if run_cache:
        step += 1
        print(f"[{step}/{total_steps}] 分析 Cache 调度...", file=sys.stderr)
        results["cache"] = analyze_cache(log_file, tail=tail)

    if run_load:
        step += 1
        print(f"[{step}/{total_steps}] 分析负载与计数器...", file=sys.stderr)
        results["load"] = analyze_load(log_file, tail=tail)

    if run_trace:
        step += 1
        print(f"[{step}/{total_steps}] 追踪请求...", file=sys.stderr)
        results["trace"] = analyze_trace(log_file, args.trace, tail=tail)

    # 判定状态
    status, status_reason = determine_status(results)

    # 输出报告
    report, details = format_full_report(results, status, status_reason)
    print(report)

    # 保存详细报告
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_base = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        golang_router_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", ".."))
        output_base = os.path.join(golang_router_root, "skill_output", "troubleshoot")
    output_dir = os.path.join(output_base, run_timestamp)
    filepath = save_detailed_report(report, output_dir, details=details)
    print(f"\n详细报告已保存到: {filepath}", file=sys.stderr)


if __name__ == "__main__":
    main()
