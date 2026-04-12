#!/usr/bin/env python3
"""
Router Log Parser — FastDeploy Go Router 日志解析原语

支持两种调用方式：
1. 作为模块导入：from log_parser import parse_cache_strategy_line, parse_stats_line
2. 作为 CLI 工具：grep 'pattern' logfile | python3 log_parser.py parse-cache-strategy

Python 3 stdlib only，零依赖。
"""

import argparse
import json
import re
import sys
from datetime import datetime

# ════════════════════════════════════════════════════════════════
# 通用解析原语
# ════════════════════════════════════════════════════════════════


def parse_go_map(s):
    """解析 Go fmt.Sprintf('%v', map) 输出：map[key1:val1 key2:val2 ...]

    处理 URL 中冒号与 Go map key-value 分隔符的冲突（从最后一个冒号分割）。
    空 map 'map[]' 返回空 dict。
    """
    inner_match = re.search(r"map\[(.*?)\]", s)
    if not inner_match:
        return {}
    inner = inner_match.group(1).strip()
    if not inner:
        return {}
    result = {}
    for token in inner.split():
        idx = token.rfind(":")
        if idx > 0:
            key = token[:idx]
            val_str = token[idx + 1 :]
            try:
                result[key] = int(val_str) if "." not in val_str else float(val_str)
            except ValueError:
                result[key] = val_str
    return result


# 时间戳：YYYY/MM/DD HH:MM:SS
TS_RE = re.compile(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})")

# ts_ms：2025-01-15 18:25:33.123
TS_MS_RE = re.compile(r"ts_ms=(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def extract_ts(line):
    """从日志行提取 YYYY/MM/DD HH:MM:SS 时间戳。"""
    m = TS_RE.search(line)
    return m.group(1) if m else None


def parse_ts(ts_str):
    """将 YYYY/MM/DD HH:MM:SS 时间戳解析为 datetime。"""
    return datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")


# ════════════════════════════════════════════════════════════════
# 时间范围过滤
# ════════════════════════════════════════════════════════════════

import os
import subprocess
import tempfile

_FULL_DT_RE = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_DATE_ONLY_RE = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$")
_SHORT_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_TIME_ONLY_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def _get_log_boundary_ts(log_file, which="first"):
    """从日志文件首行或末行提取时间戳。"""
    cmd = "head" if which == "first" else "tail"
    try:
        r = subprocess.run([cmd, "-1", log_file], capture_output=True, text=True, timeout=5)
        return extract_ts(r.stdout) if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def complete_time_arg(time_str, log_file, is_end=False):
    """解析灵活时间输入，补全缺失部分。

    支持格式：
        'YYYY/MM/DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS', 'YYYY/MM/DD',
        'MM/DD', 'MM/DD HH:MM', 'HH:MM:SS', 'HH:MM'

    补全规则：
        - 缺年份：从日志首行取
        - 缺日期：从日志末行取
        - 缺时间：start→00:00:00, end→23:59:59

    Returns: 'YYYY/MM/DD HH:MM:SS' 格式字符串
    """
    if time_str is None:
        return None
    time_str = time_str.strip()

    # Case 1: 完整日期时间
    m = _FULL_DT_RE.match(time_str)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        h, mi = m.group(4).zfill(2), m.group(5)
        s = (m.group(6) or "00").zfill(2)
        return f"{y}/{mo}/{d} {h}:{mi}:{s}"

    # Case 2: 仅日期 YYYY/MM/DD
    m = _DATE_ONLY_RE.match(time_str)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        t = "23:59:59" if is_end else "00:00:00"
        return f"{y}/{mo}/{d} {t}"

    # Case 3: 短日期 MM/DD 或 MM/DD HH:MM[:SS]
    m = _SHORT_DATE_RE.match(time_str)
    if m:
        mo, d = m.group(1).zfill(2), m.group(2).zfill(2)
        ts = _get_log_boundary_ts(log_file, "first")
        year = ts[:4] if ts else str(datetime.now().year)
        if m.group(3):  # 有时间部分
            h, mi = m.group(3).zfill(2), m.group(4)
            s = (m.group(5) or "00").zfill(2)
            return f"{year}/{mo}/{d} {h}:{mi}:{s}"
        t = "23:59:59" if is_end else "00:00:00"
        return f"{year}/{mo}/{d} {t}"

    # Case 4: 仅时间 HH:MM[:SS]
    m = _TIME_ONLY_RE.match(time_str)
    if m:
        h, mi = m.group(1).zfill(2), m.group(2)
        s = (m.group(3) or "00").zfill(2)
        ts = _get_log_boundary_ts(log_file, "last")
        date_part = ts[:10] if ts else f"{datetime.now().year}/01/01"
        return f"{date_part} {h}:{mi}:{s}"

    # Fallback: 原样返回
    return time_str


def filter_file_by_time_range(log_file, start_str=None, end_str=None):
    """用 awk 按时间范围预过滤日志文件。

    时间戳 YYYY/MM/DD HH:MM:SS 天然字典序可比，直接用 awk 字符串比较。
    无时间戳的行（如 panic 堆栈续行）保留。

    Args:
        log_file: 原日志文件路径
        start_str: 起始时间 'YYYY/MM/DD HH:MM:SS'（含），或 None
        end_str: 结束时间 'YYYY/MM/DD HH:MM:SS'（含），或 None

    Returns:
        tuple: (file_path, is_temp) — is_temp=True 时调用方负责删除
    """
    if not start_str and not end_str:
        return (log_file, False)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, prefix="router_filtered_")
    tmp.close()

    awk_script = r"""{
        ts = ""
        if (match($0, /[0-9]{4}\/[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/)) {
            ts = substr($0, RSTART, RLENGTH)
        }
        if (ts == "") { print; next }
        if ((start == "" || ts >= start) && (end == "" || ts <= end)) print
    }"""

    cmd = ["awk", "-v", f'start={start_str or ""}', "-v", f'end={end_str or ""}', awk_script, log_file]

    try:
        with open(tmp.name, "w") as outf:
            result = subprocess.run(cmd, stdout=outf, stderr=subprocess.PIPE, text=True, timeout=120)
        if result.returncode != 0:
            os.unlink(tmp.name)
            return (log_file, False)
    except (subprocess.TimeoutExpired, OSError):
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return (log_file, False)

    return (tmp.name, True)


# Context tag：[session_id:...], [request_id:...], [trace_id:...], [req_id:...]
TAG_RE = re.compile(r"\[(session_id|request_id|trace_id|req_id):([^\]]+)\]")


def extract_tags(line):
    """从日志行提取 context tag。"""
    return {m.group(1): m.group(2) for m in TAG_RE.finditer(line)}


# ════════════════════════════════════════════════════════════════
# Cache-Aware 策略行解析（类别 A）
# ════════════════════════════════════════════════════════════════

URL_RE = r"(?:https?://)?[A-Za-z0-9.-]+(?::\d+)?"
STRATEGY_RE = re.compile(r"final strategy:\s*(\w+)")
SELECTED_RE = re.compile(rf"selected=({URL_RE})(?:,|\s|$)")
REASON_RE = re.compile(r"reason:\s*(.+?)(?:,\s*loads=|\.?\s*ts_ms=|$)")


def parse_cache_strategy_line(line):
    """解析 cache-aware prefill 策略行。

    输入示例：
        [INFO] 2026/03/30 20:16:57 logger.go:79: ... cache-aware prefill: final strategy:
        cache_aware_scoring, selected=http://10.52.95.17:9263, loads=map[...], hitRatios=map[...]

    返回 dict 或 None（如果不是策略行）。
    """
    sm = STRATEGY_RE.search(line)
    if not sm:
        return None

    ts = extract_ts(line)
    strategy = sm.group(1)
    record = {"ts": ts or "", "strategy": strategy}

    # selected worker URL
    sel_m = SELECTED_RE.search(line)
    if sel_m:
        record["selected"] = sel_m.group(1)

    # reason（仅 process_tokens fallback）
    reason_m = REASON_RE.search(line)
    if reason_m and strategy == "process_tokens":
        record["reason"] = reason_m.group(1).strip()

    # hitRatios map
    hr_match = re.search(r"hitRatios=(map\[.*?\])", line)
    if hr_match:
        hit_ratios = parse_go_map(hr_match.group(1))
        record["hitRatios"] = hit_ratios
        if "selected" in record:
            record["selected_hitRatio"] = hit_ratios.get(record["selected"], 0)
    else:
        record["hitRatios"] = {}
        if "selected" in record:
            record["selected_hitRatio"] = 0

    # loads map
    loads_match = re.search(r"loads=(map\[.*?\])", line)
    if loads_match:
        record["loads"] = parse_go_map(loads_match.group(1))

    # ts_ms（精确到毫秒的调度时间戳）
    ts_ms_m = TS_MS_RE.search(line)
    if ts_ms_m:
        record["ts_ms"] = ts_ms_m.group(1)

    # context tags
    tags = extract_tags(line)
    if tags:
        record["tags"] = tags

    return record


# ════════════════════════════════════════════════════════════════
# Stats 行解析（类别 B）
# ════════════════════════════════════════════════════════════════

TOTAL_RUNNING_RE = re.compile(r"total_running=(\d+)")
WORKER_RUNNING_RE = re.compile(rf"({URL_RE}): running=(\d+)")
CACHE_HR_RE = re.compile(r"cache_hit_rate=([\d.]+)%\s*\(hits=(\d+)/total=(\d+)\)")


def parse_stats_line(line):
    """解析 [stats] 统计行。

    输入示例：
        [INFO] 2026/03/30 20:14:38 logger.go:79: [stats] total_running=14,
        workers: [...], cache_hit_rate=0.00% (hits=0/total=7)

    注意：hits 和 total 是 per-interval 的（每 5s 重置），累计值必须 sum 所有行。

    返回 dict 或 None（如果不是 stats 行）。
    """
    if "[stats]" not in line:
        return None

    ts = extract_ts(line)
    record = {"ts": ts or ""}

    # total_running
    tr_m = TOTAL_RUNNING_RE.search(line)
    if tr_m:
        record["total_running"] = int(tr_m.group(1))

    # per-worker running
    workers = {}
    for wm in WORKER_RUNNING_RE.finditer(line):
        workers[wm.group(1)] = int(wm.group(2))
    record["workers"] = workers

    # cache_hit_rate + hits/total
    chr_m = CACHE_HR_RE.search(line)
    if chr_m:
        record["cache_hit_rate"] = float(chr_m.group(1))
        record["hits"] = int(chr_m.group(2))
        record["total"] = int(chr_m.group(3))

    return record


# ════════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════════


def _cli_parse_stream(parse_fn):
    """通用 CLI 流式解析：从 stdin 读入日志行，输出 JSON Lines 到 stdout。"""
    parsed = 0
    skipped = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        record = parse_fn(line)
        if record:
            print(json.dumps(record, ensure_ascii=False))
            parsed += 1
        else:
            skipped += 1
    print(f"Parsed {parsed} lines, skipped {skipped}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="FastDeploy Go Router Log Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("parse-cache-strategy", help="解析 cache-aware 策略行 → JSON Lines")
    sub.add_parser("parse-stats", help="解析 [stats] 统计行 → JSON Lines")

    args = parser.parse_args()

    if args.command == "parse-cache-strategy":
        _cli_parse_stream(parse_cache_strategy_line)
    elif args.command == "parse-stats":
        _cli_parse_stream(parse_stats_line)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
