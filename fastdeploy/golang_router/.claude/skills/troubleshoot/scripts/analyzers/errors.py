#!/usr/bin/env python3
"""
Errors Analyzer — 错误分类分析

分析 Router 日志中的 ERROR/WARN 日志、HTTP 状态码分布、Panic 事件。
按问题来源层（Router / FastDeploy 后端 / 客户端）标注每类错误。
"""

import os
import subprocess
import sys

# 让 analyzers 能 import 同级 scripts 下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chart import render_bar, render_sparkline, render_table
from log_parser import extract_ts, parse_error_line, parse_http_line
from stats import count_by, time_bucket

# ════════════════════════════════════════════════════════════════
# 错误来源层映射（从 error_catalog.md 提取的核心规则）
# ════════════════════════════════════════════════════════════════

# 模板 → 来源层 映射（归一化后的模板匹配）
SOURCE_LAYER_RULES = [
    # Router 自身
    ("Failed to build disaggregate_info", "Router"),
    ("Failed to encode modified request", "Router"),
    ("Panic recovered", "Router"),
    ("DefaultManager is nil", "Router"),
    ("double-release", "Router"),
    ("counter already cleaned up", "Router"),
    ("counter already zero", "Router"),
    ("tokenizer failed", "Router"),
    ("Instance {url} role is unknown", "Router"),
    ("Failed to read YAML file config/register.yaml", "Router"),
    # 客户端
    ("Invalid request body", "客户端"),
    ("Invalid JSON format", "客户端"),
    ("Failed to read request body", "客户端"),
    ("Failed to unmarshal request JSON", "客户端"),
    # FD 后端（默认多数 ERROR 来自后端）
    ("Failed to select", "FD 后端"),
    ("Failed to connect to backend", "FD 后端"),
    ("No available", "FD 后端"),
    ("request failed", "FD 后端"),
    ("Removed unhealthy", "FD 后端"),
    ("is not healthy", "FD 后端"),
    ("is healthy", "FD 后端"),
    ("Backend request failed", "FD 后端"),
    ("Decode request failed", "FD 后端"),
    ("Prefill request failed", "FD 后端"),
    ("Failed to create decode request", "FD 后端"),
    ("Failed to create prefill request", "FD 后端"),
    ("Failed to create backend request", "FD 后端"),
    ("GetRemoteMetrics failed", "FD 后端"),
]

# scanner error / copy error 特殊处理：context canceled → 客户端，其他 → FD 后端
SCANNER_COPY_PATTERNS = ("scanner error", "copy error")


def classify_source_layer(template, original=""):
    """根据错误模板判断来源层。"""
    # scanner error / copy error 特殊判断
    for pat in SCANNER_COPY_PATTERNS:
        if pat in template or pat in original:
            if "context canceled" in original:
                return "客户端"
            return "FD 后端"

    for pattern, layer in SOURCE_LAYER_RULES:
        if pattern in template:
            return layer

    return "未知"


# ════════════════════════════════════════════════════════════════
# 主分析函数
# ════════════════════════════════════════════════════════════════


def analyze_errors(log_file, tail=None, top_n=20):
    """分析日志中的错误。

    Args:
        log_file: 日志文件路径
        tail: 尾部行数限制（None 则全量）
        top_n: 错误 Top N

    Returns:
        dict: {
            error_top_n: [{template, count, pct, source_layer, level, urls}],
            status_code_dist: [{value, count, pct}],
            panic_list: [{ts, context}],
            error_rate: float,
            error_trend: [{bucket, count}],
            total_errors: int,
            total_warns: int,
            total_requests: int,
            summary: str,
        }
    """
    # Phase 1: Grep 提取各类日志
    error_lines = _grep_lines(log_file, r"\[ERROR\]", tail)
    warn_lines = _grep_lines(log_file, r"\[WARN\]", tail)
    http_lines = _grep_lines(log_file, r"\[(POST|GET)\] /", tail)
    panic_lines = _grep_lines(log_file, "Panic recovered", tail)

    # Phase 2: 解析
    # 2.1 ERROR + WARN 归一化
    error_records = [parse_error_line(line) for line in error_lines]
    warn_records = [parse_error_line(line) for line in warn_lines]
    all_error_records = error_records + warn_records

    # 2.2 HTTP 请求解析
    http_records = []
    for line in http_lines:
        r = parse_http_line(line)
        if r:
            http_records.append(r)

    # 2.3 Panic 提取
    panic_list = []
    for line in panic_lines:
        ts = extract_ts(line)
        panic_list.append({"ts": ts or "", "context": line.strip()})

    # Phase 3: 分析
    # 3.1 按模板分组 Top N
    error_top = _compute_error_top_n(all_error_records, top_n)

    # 3.2 HTTP 状态码分布
    status_dist = count_by(http_records, "status")

    # 3.3 错误率
    total_requests = len(http_records)
    non_200 = sum(1 for r in http_records if r["status"] != 200)
    error_rate = round(non_200 / total_requests * 100, 2) if total_requests else 0

    # 3.4 错误趋势（按时间窗口统计非 200 请求数）
    non_200_records = [r for r in http_records if r["status"] != 200]
    error_trend = time_bucket(non_200_records, window="auto")

    return {
        "error_top_n": error_top,
        "status_code_dist": status_dist,
        "panic_list": panic_list,
        "error_rate": error_rate,
        "error_trend": error_trend,
        "total_errors": len(error_records),
        "total_warns": len(warn_records),
        "total_requests": total_requests,
    }


def _compute_error_top_n(records, top_n):
    """按模板分组并标注来源层。"""
    # 分组
    groups = {}
    for r in records:
        tpl = r["template"]
        if tpl not in groups:
            groups[tpl] = {
                "template": tpl,
                "count": 0,
                "level": r["level"],
                "originals": [],
            }
        groups[tpl]["count"] += 1
        # 保留最多 5 个原始消息用于详细报告中提取 URL
        if len(groups[tpl]["originals"]) < 5:
            groups[tpl]["originals"].append(r["original"])

    total = len(records)
    result = []
    for g in sorted(groups.values(), key=lambda x: -x["count"]):
        source_layer = classify_source_layer(g["template"], g["originals"][0] if g["originals"] else "")
        result.append(
            {
                "template": g["template"],
                "count": g["count"],
                "pct": round(g["count"] / total * 100, 1) if total else 0,
                "source_layer": source_layer,
                "level": g["level"],
                "sample_originals": g["originals"],
            }
        )
        if len(result) >= top_n:
            break

    return result


def _grep_lines(log_file, pattern, tail=None):
    """用 grep 从日志文件提取匹配行。"""
    try:
        if tail:
            # 先 tail 再 grep
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
    """简单 shell 引号转义。"""
    return "'" + s.replace("'", "'\\''") + "'"


# ════════════════════════════════════════════════════════════════
# 报告格式化
# ════════════════════════════════════════════════════════════════


def format_errors_report(result):
    """将分析结果格式化为终端报告。

    Args:
        result: analyze_errors 返回的 dict

    Returns:
        str: 格式化后的报告文本
    """
    sections = []

    # 标题
    sections.append("## 错误分析")
    sections.append("")

    # 概览
    sections.append(
        f'  ERROR: {result["total_errors"]}  |  '
        f'WARN: {result["total_warns"]}  |  '
        f'请求总数: {result["total_requests"]}  |  '
        f'错误率: {result["error_rate"]}%'
    )
    sections.append("")

    # Panic
    if result["panic_list"]:
        sections.append(f'  ⚠ Panic 事件: {len(result["panic_list"])} 次')
        for p in result["panic_list"][:5]:
            sections.append(f'    [{p["ts"]}] {p["context"][:100]}')
        sections.append("")

    # 错误 Top N
    if result["error_top_n"]:
        sections.append("### ERROR/WARN Top 分类")
        sections.append("")
        bar_data = []
        for e in result["error_top_n"][:10]:
            label = e["template"][:50]
            bar_data.append(
                {
                    "label": label,
                    "value": e["pct"],
                    "count": e["count"],
                }
            )
        sections.append(render_bar(bar_data, show_count=True))
        sections.append("")

        # 来源层表格
        table_data = []
        for e in result["error_top_n"][:10]:
            table_data.append(
                {
                    "模板": e["template"][:60],
                    "数量": e["count"],
                    "占比": f'{e["pct"]}%',
                    "级别": e["level"],
                    "来源层": e["source_layer"],
                }
            )
        sections.append(
            render_table(table_data, columns=["模板", "数量", "占比", "级别", "来源层"], right_align={"数量", "占比"})
        )
        sections.append("")
        yaml_missing_count = sum(
            e["count"] for e in result["error_top_n"] if "Failed to read YAML file config/register.yaml" in e["template"]
        )
        if yaml_missing_count > 0:
            sections.append(
                f"  ℹ `Failed to read YAML file config/register.yaml` 出现 {yaml_missing_count} 次：若未启用该配置文件，可忽略。"
            )
            sections.append("")

    # 状态码分布
    if result["status_code_dist"]:
        sections.append("### HTTP 状态码分布")
        sections.append("")
        bar_data = []
        for s in result["status_code_dist"]:
            bar_data.append(
                {
                    "label": str(s["value"]),
                    "value": s["pct"],
                    "count": s["count"],
                }
            )
        sections.append(render_bar(bar_data, show_count=True))
        sections.append("")

    # 错误趋势
    if result["error_trend"] and len(result["error_trend"]) > 1:
        sections.append("### 非 200 请求趋势")
        sections.append("")
        sections.append(
            render_sparkline(
                result["error_trend"],
                value_field="count",
                title="Error Count",
                y_label="req",
            )
        )
        sections.append("")

    return "\n".join(sections)
