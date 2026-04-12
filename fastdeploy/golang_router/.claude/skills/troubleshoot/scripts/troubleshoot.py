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
from log_parser import complete_time_arg, filter_file_by_time_range


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
        # 分钟模式：转换为大致行数（假设 ~20 行/秒）
        minutes = int(tail_str[:-1])
        return minutes * 60 * 20
    return int(tail_str)


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
        return "DEGRADED", ", ".join(reasons)

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
                - 'trace_files': {trace_id: text} 或 {}
    """
    parts = []
    details = {"health_events": None, "trace_files": {}}

    # 状态行
    parts.append(f"STATUS: {status} — {status_reason}")
    parts.append("=" * 60)
    parts.append("")

    # 各维度报告
    if "errors" in results:
        parts.append(format_errors_report(results["errors"]))

    if "latency" in results:
        parts.append(format_latency_report(results["latency"]))

    if "health" in results:
        summary, detail = format_health_report(results["health"])
        parts.append(summary)
        if detail:
            details["health_events"] = detail

    if "load" in results:
        parts.append(format_load_report(results["load"]))

    if "cache" in results:
        parts.append(format_cache_report(results["cache"]))

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
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"troubleshoot_report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Router Troubleshooting Report\n")
        f.write(f'> Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(report_text)

    # 保存详情到 details/ 子目录
    if details:
        details_dir = os.path.join(output_dir, "details")

        if details.get("health_events"):
            os.makedirs(details_dir, exist_ok=True)
            health_path = os.path.join(details_dir, "health_events.md")
            with open(health_path, "w", encoding="utf-8") as f:
                f.write(details["health_events"])

        for trace_id, trace_text in details.get("trace_files", {}).items():
            os.makedirs(details_dir, exist_ok=True)
            safe_id = trace_id.replace("/", "_")
            trace_path = os.path.join(details_dir, f"trace_{safe_id}.md")
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

    # 确定分析模式
    any_mode = args.errors or args.latency or args.health or args.cache or args.load or args.trace
    run_errors = args.errors or (not any_mode)
    run_latency = args.latency or (not any_mode)
    run_health = args.health or (not any_mode)
    run_load = args.load or (not any_mode)
    run_cache = args.cache or (not any_mode)
    run_trace = bool(args.trace)  # trace 需要指定 ID，全量扫描不自动调用

    tail = parse_tail_arg(args.tail)

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
    if args.output:
        output_dir = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        golang_router_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", ".."))
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(golang_router_root, "skill_output", "troubleshoot", run_timestamp)
    filepath = save_detailed_report(report, output_dir, details=details)
    print(f"\n详细报告已保存到: {filepath}", file=sys.stderr)


if __name__ == "__main__":
    main()
