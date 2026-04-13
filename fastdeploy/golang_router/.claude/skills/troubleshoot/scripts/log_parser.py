#!/usr/bin/env python3
"""
Router Log Parser — FastDeploy Go Router 日志解析原语

支持两种调用方式：
1. 作为模块导入：from log_parser import parse_http_line, parse_cache_strategy_line, ...
2. 作为 CLI 工具：grep 'pattern' logfile | python3 log_parser.py parse-http

Python 3 stdlib only，零依赖。
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta

# ════════════════════════════════════════════════════════════════
# 通用解析原语
# ════════════════════════════════════════════════════════════════

# Go time.Duration.String() parser: handles 1h2m3.456s, 500µs, 150.5ms, etc.
DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(h|m(?!s)|s|ms|[µu]s|ns)")


def parse_go_duration_ms(s):
    """解析 Go time.Duration.String() 输出为毫秒。

    Examples: '1.5s' -> 1500.0, '500µs' -> 0.5, '1m30s' -> 90000.0
    """
    total = 0.0
    for m in DURATION_RE.finditer(s):
        val, unit = float(m.group(1)), m.group(2)
        if unit == "h":
            total += val * 3600000
        elif unit == "m":
            total += val * 60000
        elif unit == "s":
            total += val * 1000
        elif unit == "ms":
            total += val
        elif unit in ("µs", "us"):
            total += val / 1000
        elif unit == "ns":
            total += val / 1000000
    return total


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


def filter_file_by_recent_minutes(log_file, minutes):
    """按日志末时间戳向前过滤最近 N 分钟日志。

    Returns:
        tuple: (file_path, is_temp) — is_temp=True 时调用方负责删除
    """
    if minutes is None or minutes <= 0:
        return (log_file, False)

    last_ts = _get_log_boundary_ts(log_file, "last")
    if not last_ts:
        return (log_file, False)

    try:
        end_dt = parse_ts(last_ts)
    except ValueError:
        return (log_file, False)

    start_dt = end_dt - timedelta(minutes=minutes)
    start_str = start_dt.strftime("%Y/%m/%d %H:%M:%S")
    end_str = end_dt.strftime("%Y/%m/%d %H:%M:%S")
    return filter_file_by_time_range(log_file, start_str=start_str, end_str=end_str)


# Context tag：[session_id:...], [request_id:...], [trace_id:...], [req_id:...]
TAG_RE = re.compile(r"\[(session_id|request_id|trace_id|req_id):([^\]]+)\]")


def extract_tags(line):
    """从日志行提取 context tag。"""
    return {m.group(1): m.group(2) for m in TAG_RE.finditer(line)}


# Log level
LEVEL_RE = re.compile(r"\[(INFO|ERROR|WARN|DEBUG)\]")


def extract_level(line):
    """从日志行提取日志级别。"""
    m = LEVEL_RE.search(line)
    return m.group(1) if m else None


# ════════════════════════════════════════════════════════════════
# HTTP 请求行解析（类别 H1）
# ════════════════════════════════════════════════════════════════

# H1 pattern: [METHOD] /path HTTP/1.1 STATUS LATENCY CLIENT_IP
HTTP_RE = re.compile(
    r"\[(POST|GET|PUT|DELETE|PATCH|HEAD|OPTIONS)\]\s+"
    r"(/\S*)\s+HTTP/\d\.\d\s+"
    r"(\d{3})\s+"
    r"(\S+)\s+"  # latency (Go duration)
    r"(\d+\.\d+\.\d+\.\d+)"  # client IP
)


def parse_http_line(line, inference_only=False):
    """解析 H1 HTTP 请求行。

    输入示例：
        [INFO] 2025/01/15 18:25:33 logger.go:45: [POST] /v1/chat/completions HTTP/1.1 200 1.234567s 10.0.0.1

    Args:
        line: 日志行
        inference_only: True 则仅保留推理路径 (/v1/chat/completions, /v1/completions)

    返回 dict 或 None。
    """
    ts = extract_ts(line)
    m = HTTP_RE.search(line)
    if not m:
        return None

    method, path, status, latency_raw, client_ip = m.groups()
    latency_ms = parse_go_duration_ms(latency_raw)

    if inference_only and path not in ("/v1/chat/completions", "/v1/completions"):
        return None

    record = {
        "ts": ts or "",
        "method": method,
        "path": path,
        "status": int(status),
        "latency_ms": round(latency_ms, 3),
        "client_ip": client_ip,
    }

    tags = extract_tags(line)
    if tags:
        record["tags"] = tags

    return record


# ════════════════════════════════════════════════════════════════
# Cache-Aware 策略行解析（类别 H6）
# ════════════════════════════════════════════════════════════════

URL_RE = r"(?:https?://)?[A-Za-z0-9.-]+(?::\d+)?"
STRATEGY_RE = re.compile(r"final strategy:\s*(\w+)")
SELECTED_RE = re.compile(rf"selected=({URL_RE})(?:,|\s|$)")
REASON_RE = re.compile(r"reason:\s*(.+?)(?:,\s*loads=|\.?\s*ts_ms=|$)")


def parse_cache_strategy_line(line):
    """解析 cache-aware prefill 策略行。

    返回 dict 或 None（如果不是策略行）。
    """
    sm = STRATEGY_RE.search(line)
    if not sm:
        return None

    ts = extract_ts(line)
    strategy = sm.group(1)
    record = {"ts": ts or "", "strategy": strategy}

    sel_m = SELECTED_RE.search(line)
    if sel_m:
        record["selected"] = sel_m.group(1)

    reason_m = REASON_RE.search(line)
    if reason_m and strategy == "process_tokens":
        record["reason"] = reason_m.group(1).strip()

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

    loads_match = re.search(r"loads=(map\[.*?\])", line)
    if loads_match:
        record["loads"] = parse_go_map(loads_match.group(1))

    ts_ms_m = TS_MS_RE.search(line)
    if ts_ms_m:
        record["ts_ms"] = ts_ms_m.group(1)

    tags = extract_tags(line)
    if tags:
        record["tags"] = tags

    return record


# ════════════════════════════════════════════════════════════════
# Stats 行解析（类别 H7）
# ════════════════════════════════════════════════════════════════

TOTAL_RUNNING_RE = re.compile(r"total_running=(\d+)")
WORKER_RUNNING_RE = re.compile(rf"({URL_RE}): running=(\d+)")
CACHE_HR_RE = re.compile(r"cache_hit_rate=([\d.]+)%\s*\(hits=(\d+)/total=(\d+)\)")


def parse_stats_line(line):
    """解析 [stats] 统计行。

    注意：hits 和 total 是 per-interval 的（每 5s 重置），累计值必须 sum 所有行。

    返回 dict 或 None（如果不是 stats 行）。
    """
    if "[stats]" not in line:
        return None

    ts = extract_ts(line)
    record = {"ts": ts or ""}

    tr_m = TOTAL_RUNNING_RE.search(line)
    if tr_m:
        record["total_running"] = int(tr_m.group(1))

    workers = {}
    for wm in WORKER_RUNNING_RE.finditer(line):
        workers[wm.group(1)] = int(wm.group(2))
    record["workers"] = workers

    chr_m = CACHE_HR_RE.search(line)
    if chr_m:
        record["cache_hit_rate"] = float(chr_m.group(1))
        record["hits"] = int(chr_m.group(2))
        record["total"] = int(chr_m.group(3))

    return record


# ════════════════════════════════════════════════════════════════
# 错误消息模板归一化
# ════════════════════════════════════════════════════════════════

NORMALIZE_PATTERNS = [
    (re.compile(r"https?://[\w.:]+"), "{url}"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "{uuid}"),
    (re.compile(r"\d+\.\d+\.\d+\.\d+:\d+"), "{ip:port}"),
    (re.compile(r"\b\d+\b"), "{N}"),
]

# Message extraction: everything after "logger.go:NN: " (and optional context tags)
MSG_RE = re.compile(r"logger\.go:\d+:\s*(?:\[[^\]]*\]\s*)*(.+)")


def normalize_message(msg):
    """将错误消息中的变量替换为占位符模板。"""
    for pat, repl in NORMALIZE_PATTERNS:
        msg = pat.sub(repl, msg)
    return msg


def parse_error_line(line):
    """解析 ERROR/WARN 行并进行模板归一化。

    返回 dict: {ts, level, original, template, tags}
    """
    ts = extract_ts(line)
    level = extract_level(line)
    tags = extract_tags(line)

    mm = MSG_RE.search(line)
    original = mm.group(1).strip() if mm else line

    template = normalize_message(original)

    record = {
        "ts": ts or "",
        "level": level or "",
        "original": original,
        "template": template,
    }
    if tags:
        record["tags"] = tags

    return record


# ════════════════════════════════════════════════════════════════
# Select/Release 事件匹配
# ════════════════════════════════════════════════════════════════

SELECT_RE = re.compile(rf"select worker\s*(?:\((\w+)\))?:\s*({URL_RE})")
RELEASE_RE = re.compile(rf"release worker\s*(?:\((\w+)\))?:\s*({URL_RE})")
FAILED_SELECT_RE = re.compile(r"Failed to select")
SELECT_TOKENS_RE = re.compile(rf"select worker \((\w+)\):\s*({URL_RE}),\s*tokens:\s*(\d+)")
RELEASE_TOKENS_RE = re.compile(rf"release (?:([a-zA-Z_]+)\s+)?tokens:\s*({URL_RE}),\s*tokens:\s*(\d+)")


def _parse_ts_safe(ts):
    if not ts:
        return None
    try:
        return parse_ts(ts)
    except ValueError:
        return None


def _select_match_key(tags):
    """构建请求关联 key，优先 request_id，其次 req_id/trace_id/session_id。"""
    if not tags:
        return (None, None)
    rid = tags.get("request_id")
    if rid:
        return ("request_id", f"request_id:{rid}")
    for k in ("req_id", "trace_id", "session_id"):
        v = tags.get(k)
        if v:
            return ("alt_id", f"{k}:{v}")
    return (None, None)


def _normalize_worker_type(worker_type):
    """归一化 worker type。"""
    t = (worker_type or "unknown").lower()
    if t in ("prefill", "decode", "mixed"):
        return t
    return "unknown"


def _normalize_worker_url_key(url):
    if not url:
        return ""
    return re.sub(r"^https?://", "", str(url).strip().rstrip("/"))


def _infer_release_worker_type(release, selects, fallback_window_s=120):
    """为未显式标注 type 的 release 近似推断 worker type。

    优先级：
      1) 同 worker、时间上最近且不晚于 release 的 select type
      2) 若无可解析时间戳，则使用同 worker 的最后一个 select type
      3) 推断失败返回 unknown
    """
    worker = release.get("worker")
    if not worker:
        return "unknown"

    r_ts = _parse_ts_safe(release.get("ts"))
    candidates = [s for s in selects if s.get("worker") == worker]
    if not candidates:
        return "unknown"

    if r_ts:
        best = None
        best_delta = None
        for s in candidates:
            s_ts = _parse_ts_safe(s.get("ts"))
            if not s_ts:
                continue
            delta = (r_ts - s_ts).total_seconds()
            if delta < 0 or delta > fallback_window_s:
                continue
            if best_delta is None or delta < best_delta:
                best = s
                best_delta = delta
        if best is not None:
            return _normalize_worker_type(best.get("type"))

    # 回退：按出现顺序取同 worker 的最近 select
    return _normalize_worker_type(candidates[-1].get("type"))


def _infer_token_release_worker_type(release, selects, fallback_window_s=120):
    """为 token release 推断 worker type（prefill/mixed）。

    注意：日志文本通常固定为 `release prefill tokens`，即使 mixed 也可能走这条日志。
    因此 token release 的类型优先依据同 worker 的邻近 select 推断。
    """
    worker = release.get("worker")
    if not worker:
        return "unknown"

    r_ts = _parse_ts_safe(release.get("ts"))
    candidates = [s for s in selects if s.get("worker") == worker and _normalize_worker_type(s.get("type")) in ("prefill", "mixed")]
    if not candidates:
        return "unknown"

    if r_ts:
        best = None
        best_delta = None
        for s in candidates:
            s_ts = _parse_ts_safe(s.get("ts"))
            if not s_ts:
                continue
            delta = (r_ts - s_ts).total_seconds()
            if delta < 0 or delta > fallback_window_s:
                continue
            if best_delta is None or delta < best_delta:
                best = s
                best_delta = delta
        if best is not None:
            return _normalize_worker_type(best.get("type"))

    return _normalize_worker_type(candidates[-1].get("type"))


def match_select_release(lines, fallback_window_s=120):
    """匹配 select/release worker 事件对。

    Args:
        lines: 日志行列表（字符串）

    Returns:
        dict: {matched, unmatched_selects, failed_selects, per_worker}
    """
    selects = []
    releases = []
    failed_selects = []

    for line_no, line in enumerate(lines, 1):
        ts = extract_ts(line)
        tags = extract_tags(line)

        # Token-bearing select
        tm = SELECT_TOKENS_RE.search(line)
        if tm:
            selects.append(
                {
                    "ts": ts,
                    "worker": tm.group(2),
                    "worker_key": _normalize_worker_url_key(tm.group(2)),
                    "type": _normalize_worker_type(tm.group(1)),
                    "tags": tags,
                    "tokens": int(tm.group(3)),
                    "line": line_no,
                }
            )
            continue

        # Token-bearing release
        trm = RELEASE_TOKENS_RE.search(line)
        if trm:
            token_type = trm.group(1)
            releases.append(
                {
                    "ts": ts,
                    "worker": trm.group(2),
                    "worker_key": _normalize_worker_url_key(trm.group(2)),
                    # 文本默认按 prefill 记，再结合同 worker 邻近 select 做纠偏（mixed 场景）
                    "type": f'{_normalize_worker_type(token_type or "prefill")}_tokens',
                    "raw_token_type": token_type or "",
                    "tags": tags,
                    "tokens": int(trm.group(3)),
                    "line": line_no,
                }
            )
            continue

        sm = SELECT_RE.search(line)
        if sm:
            selects.append(
                {
                    "ts": ts,
                    "worker": sm.group(2),
                    "worker_key": _normalize_worker_url_key(sm.group(2)),
                    "type": _normalize_worker_type(sm.group(1)),
                    "tags": tags,
                    "tokens": None,
                    "line": line_no,
                }
            )
            continue

        rm = RELEASE_RE.search(line)
        if rm:
            releases.append(
                {
                    "ts": ts,
                    "worker": rm.group(2),
                    "worker_key": _normalize_worker_url_key(rm.group(2)),
                    "type": _normalize_worker_type(rm.group(1)),
                    "tags": tags,
                    "tokens": None,
                    "line": line_no,
                }
            )
            continue

        if FAILED_SELECT_RE.search(line):
            failed_selects.append({"ts": ts, "tags": tags, "line": line_no})

    # Match by worker FIFO（select -> 同 worker 下一条 release）
    matched = []
    unmatched_selects = []
    release_used = set()

    # 请求生命周期匹配只使用 request counter release（排除 token release）
    # 说明：request_id 只用于覆盖率观测，不参与 select/release 配对条件。
    counter_release_indexes = [i for i, r in enumerate(releases) if not str(r.get("type", "")).endswith("_tokens")]
    # 请求 ID 覆盖（按 select 事件近似请求数）
    total_req_est = len(selects)
    with_request_id = 0
    with_alt_id = 0
    without_any_id = 0

    untracked_selects = []
    pending_selects = []
    for s in selects:
        key_type, key = _select_match_key(s.get("tags", {}))
        if key_type == "request_id":
            with_request_id += 1
        elif key_type == "alt_id":
            with_alt_id += 1
        else:
            without_any_id += 1

        if not key:
            # 没有任何可用 ID 时，不做退化匹配（只统计可观测信息）
            untracked_selects.append(
                {
                    "worker": s["worker"],
                    "select_ts": s["ts"],
                    "type": s["type"],
                    "tags": s["tags"],
                    "note": "no correlatable id (request_id/req_id/trace_id/session_id)",
                }
            )
        pending_selects.append(s)

    # worker FIFO + ID 一致性联合校验：
    # 1) 主匹配仍按 worker FIFO，保证在缺失 request_id 场景可工作
    # 2) 对已匹配对追加 ID 一致性检查（request_id/req_id/trace_id/session_id）
    id_consistency = {
        "both_present_and_equal": 0,
        "both_present_but_mismatch": 0,
        "only_select_has_id": 0,
        "only_release_has_id": 0,
        "both_missing": 0,
    }
    id_mismatched_matches = []

    for s in pending_selects:
        sdt = _parse_ts_safe(s.get("ts"))
        best_idx = None
        best_ts = None
        for ri in counter_release_indexes:
            if ri in release_used:
                continue
            r = releases[ri]
            if r.get("worker_key") != s.get("worker_key"):
                continue
            rdt = _parse_ts_safe(r.get("ts"))
            # 优先选择时间不早于 select 的最早 release；解析失败则按出现顺序
            if sdt and rdt and rdt < sdt:
                continue
            if best_idx is None:
                best_idx = ri
                best_ts = rdt
            elif rdt and best_ts and rdt < best_ts:
                best_idx = ri
                best_ts = rdt

        if best_idx is not None:
            r = releases[best_idx]
            s_key_type, s_key = _select_match_key(s.get("tags", {}))
            r_key_type, r_key = _select_match_key(r.get("tags", {}))
            if s_key and r_key:
                if s_key == r_key:
                    id_check = "match"
                    id_consistency["both_present_and_equal"] += 1
                else:
                    id_check = "mismatch"
                    id_consistency["both_present_but_mismatch"] += 1
                    id_mismatched_matches.append(
                        {
                            "worker": s["worker"],
                            "select_ts": s["ts"],
                            "release_ts": r["ts"],
                            "select_id_key": s_key_type,
                            "select_id": s_key,
                            "release_id_key": r_key_type,
                            "release_id": r_key,
                            "note": "worker FIFO matched, but ID mismatched",
                        }
                    )
            elif s_key and not r_key:
                id_check = "select_only"
                id_consistency["only_select_has_id"] += 1
            elif (not s_key) and r_key:
                id_check = "release_only"
                id_consistency["only_release_has_id"] += 1
            else:
                id_check = "both_missing"
                id_consistency["both_missing"] += 1

            matched.append(
                {
                    "request_id": s["tags"].get("request_id", ""),
                    "worker": s["worker"],
                    "select_ts": s["ts"],
                    "release_ts": r["ts"],
                    "type": s["type"],
                    "match_method": "worker_fifo",
                    "id_check": id_check,
                }
            )
            release_used.add(best_idx)
        else:
            unmatched_selects.append(
                {
                    "worker": s["worker"],
                    "select_ts": s["ts"],
                    "type": s["type"],
                    "tags": s["tags"],
                    "note": "no matching release found (worker FIFO)",
                }
            )

    # Per-worker summary（按 worker type 统计，不依赖日志中的 tokens 字段）
    # 规则：prefill/mixed 的 select 均计入 token_selects。
    per_worker = defaultdict(lambda: {"selects": 0, "releases": 0, "token_selects": 0, "token_releases": 0})
    for s in selects:
        s_type = _normalize_worker_type(s.get("type"))
        wkey = s.get("worker_key") or _normalize_worker_url_key(s.get("worker"))
        per_worker[wkey]["selects"] += 1
        if s_type in ("prefill", "mixed"):
            per_worker[wkey]["token_selects"] += 1
    for r in releases:
        wkey = r.get("worker_key") or _normalize_worker_url_key(r.get("worker"))
        if str(r.get("type", "")).endswith("_tokens"):
            per_worker[wkey]["token_releases"] += 1
        else:
            per_worker[wkey]["releases"] += 1

    pw_result = {}
    for w, counts in per_worker.items():
        pw_result[w] = {
            "selects": counts["selects"],
            "releases": counts["releases"],
            "delta": counts["selects"] - counts["releases"],
            "token_selects": counts["token_selects"],
            "token_releases": counts["token_releases"],
        }

    # 基于 select 构建 worker URL -> dominant type 映射
    per_worker_type_counts = defaultdict(lambda: defaultdict(int))
    for s in selects:
        wkey = s.get("worker_key") or _normalize_worker_url_key(s.get("worker"))
        per_worker_type_counts[wkey][_normalize_worker_type(s.get("type"))] += 1
    worker_dominant_type = {}
    for w, counts in per_worker_type_counts.items():
        worker_dominant_type[w] = sorted(counts.items(), key=lambda kv: -kv[1])[0][0] if counts else "unknown"

    # 为未显式标注 type 的 release 推断 worker type（避免大量 unknown）
    inferred_release_types = {}
    for i, r in enumerate(releases):
        r_type_raw = str(r.get("type", ""))
        if r_type_raw.endswith("_tokens"):
            base_t = _normalize_worker_type(r_type_raw.replace("_tokens", ""))
            # token release 按 worker URL 对应的 select 类型映射，不做邻近时间纠偏
            mapped_t = worker_dominant_type.get(r.get("worker_key") or _normalize_worker_url_key(r.get("worker")), "unknown")
            if mapped_t in ("prefill", "decode", "mixed"):
                base_t = mapped_t
            inferred_release_types[i] = f"{base_t}_tokens"
            continue
        base_t = _normalize_worker_type(r_type_raw)
        if base_t != "unknown":
            inferred_release_types[i] = base_t
            continue
        inferred_release_types[i] = _infer_release_worker_type(r, selects, fallback_window_s=fallback_window_s)

    # 按 worker type 分类统计（prefill/decode/mixed，必要时保留 unknown）
    type_summary = defaultdict(
        lambda: {
            "counter_selects": 0,
            "counter_releases": 0,
            "token_selects": 0,
            "token_releases": 0,
        }
    )
    for s in selects:
        s_type = _normalize_worker_type(s.get("type"))
        type_summary[s_type]["counter_selects"] += 1
        if s_type in ("prefill", "mixed"):
            type_summary[s_type]["token_selects"] += 1
    for i, r in enumerate(releases):
        inferred = inferred_release_types.get(i, _normalize_worker_type(str(r.get("type", ""))))
        r_type = _normalize_worker_type(str(inferred).replace("_tokens", ""))
        if str(inferred).endswith("_tokens"):
            type_summary[r_type]["token_releases"] += 1
        else:
            type_summary[r_type]["counter_releases"] += 1

    # 每个 worker URL 的类型画像（基于 select）
    worker_type_profile = {}
    for w, counts in per_worker_type_counts.items():
        dominant = "unknown"
        if counts:
            dominant = sorted(counts.items(), key=lambda kv: -kv[1])[0][0]
        worker_type_profile[w] = {
            "dominant_type": dominant,
            "prefill": counts.get("prefill", 0),
            "decode": counts.get("decode", 0),
            "mixed": counts.get("mixed", 0),
            "unknown": counts.get("unknown", 0),
        }

    unmatched_releases = []
    for i, r in enumerate(releases):
        if str(r.get("type", "")).endswith("_tokens"):
            # token release: 近邻存在 prefill/mixed select 则视为可解释，不计入 unmatched
            inferred_token_type = _normalize_worker_type(str(inferred_release_types.get(i, "unknown_tokens")).replace("_tokens", ""))
            if inferred_token_type == "unknown":
                unmatched_releases.append(
                    {
                        "worker": r.get("worker", ""),
                        "release_ts": r.get("ts", ""),
                        "type": inferred_token_type,
                        "release_kind": "token_release",
                    }
                )
            continue
        if i not in release_used:
            unmatched_releases.append(
                {
                    "worker": r.get("worker", ""),
                    "release_ts": r.get("ts", ""),
                    "type": _normalize_worker_type(inferred_release_types.get(i, "unknown")),
                    "release_kind": "request_release",
                }
            )

    return {
        "matched": matched,
        "unmatched_selects": unmatched_selects,
        "unmatched_releases": unmatched_releases,
        "untracked_selects": untracked_selects,
        "failed_selects": failed_selects,
        "per_worker": pw_result,
        "id_coverage": {
            "total_requests_estimated": total_req_est,
            "with_request_id": with_request_id,
            "without_request_id": total_req_est - with_request_id,
            "with_alt_id": with_alt_id,
            "without_any_id": without_any_id,
        },
        "id_consistency": id_consistency,
        "id_mismatched_matches": id_mismatched_matches,
        "type_summary": dict(type_summary),
        "worker_type_profile": worker_type_profile,
    }


# ════════════════════════════════════════════════════════════════
# 不支持请求检测
# ════════════════════════════════════════════════════════════════

# Router 已知路由白名单 (method, path)
KNOWN_ROUTES = {
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/register"),
    ("GET", "/registered_number"),
    ("GET", "/registered"),
    ("GET", "/health_generate"),
    ("GET", "/metrics"),
}


def find_unsupported_requests(lines):
    """从 HTTP 日志行中筛选不匹配任何已知路由的请求。

    Returns:
        dict: {details: [...], summary: {total, unique_paths: {path: count}}}
    """
    details = []
    path_counts = defaultdict(int)

    for line in lines:
        record = parse_http_line(line)
        if not record:
            continue
        key = (record["method"], record["path"])
        if key not in KNOWN_ROUTES:
            details.append(
                {
                    "ts": record["ts"],
                    "method": record["method"],
                    "path": record["path"],
                    "status": record["status"],
                    "client_ip": record["client_ip"],
                }
            )
            path_counts[f"{record['method']} {record['path']}"] += 1

    return {
        "details": details,
        "summary": {
            "total": len(details),
            "unique_paths": dict(path_counts),
        },
    }


def _cli_unsupported_requests(args):
    """CLI: 检测不支持的请求。"""
    lines = [line.rstrip("\n") for line in sys.stdin]
    result = find_unsupported_requests(lines)

    if args.summary_only:
        print(json.dumps(result["summary"], ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))


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


def _cli_parse_http(args):
    """CLI: 解析 HTTP 请求行。"""
    parsed = 0
    skipped = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        record = parse_http_line(line, inference_only=args.inference_only)
        if record:
            print(json.dumps(record, ensure_ascii=False))
            parsed += 1
        else:
            skipped += 1
    print(f"Parsed {parsed} lines, skipped {skipped}", file=sys.stderr)


def _cli_normalize_errors(args):
    """CLI: 归一化错误消息。"""
    parsed = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        record = parse_error_line(line)
        print(json.dumps(record, ensure_ascii=False))
        parsed += 1
    print(f"Normalized {parsed} lines", file=sys.stderr)


def _cli_match_select_release(args):
    """CLI: 匹配 select/release 事件。"""
    lines = [line.rstrip("\n") for line in sys.stdin]
    result = match_select_release(lines)
    print(json.dumps(result, ensure_ascii=False))


def _cli_self_test(args):
    """运行内置测试。"""
    passed = 0
    failed = 0

    def check(name, got, expected):
        nonlocal passed, failed
        if got == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    expected: {expected}")
            print(f"    got:      {got}")
            failed += 1

    print("=== Testing parse_go_duration_ms ===")
    check("simple seconds", parse_go_duration_ms("1.5s"), 1500.0)
    check("milliseconds", parse_go_duration_ms("150ms"), 150.0)
    check("fractional ms", parse_go_duration_ms("150.5ms"), 150.5)
    check("microseconds µs", parse_go_duration_ms("500µs"), 0.5)
    check("microseconds us", parse_go_duration_ms("500us"), 0.5)
    check("nanoseconds", parse_go_duration_ms("500ns"), 0.0005)
    check("composite m+s", parse_go_duration_ms("1m30s"), 90000.0)
    check("composite h+m+s", parse_go_duration_ms("1h2m3s"), 3723000.0)
    check("composite h+m+fractional_s", parse_go_duration_ms("1h2m3.456s"), 3723456.0)
    check("pure minutes", parse_go_duration_ms("2m"), 120000.0)
    check("zero", parse_go_duration_ms("0s"), 0.0)
    check("sub-ms decimal", parse_go_duration_ms("2.798235ms"), 2.798235)

    print("\n=== Testing parse_go_map ===")
    check("single entry", parse_go_map("map[http://10.0.0.1:9263:100]"), {"http://10.0.0.1:9263": 100})
    check(
        "multi entry",
        parse_go_map("map[http://10.0.0.1:9263:100 http://10.0.0.2:9867:50]"),
        {"http://10.0.0.1:9263": 100, "http://10.0.0.2:9867": 50},
    )
    check("empty map", parse_go_map("map[]"), {})
    check("float values", parse_go_map("map[http://10.0.0.1:9263:0.85]"), {"http://10.0.0.1:9263": 0.85})

    print("\n=== Testing extract_ts ===")
    check("standard", extract_ts("[INFO] 2025/01/15 18:25:33 logger.go:45: msg"), "2025/01/15 18:25:33")
    check("no timestamp", extract_ts("no timestamp here"), None)

    print("\n=== Testing extract_tags ===")
    check(
        "session+request",
        extract_tags("[session_id:abc] [request_id:def]"),
        {"session_id": "abc", "request_id": "def"},
    )
    check(
        "all four",
        extract_tags("[trace_id:t1] [req_id:r1] [session_id:s1] [request_id:rq1]"),
        {"trace_id": "t1", "req_id": "r1", "session_id": "s1", "request_id": "rq1"},
    )
    check("no tags", extract_tags("no tags here"), {})

    print("\n=== Testing parse_http_line ===")
    http_line = "[INFO] 2025/01/15 18:25:33 logger.go:45: [POST] /v1/chat/completions HTTP/1.1 200 2.798235ms 10.0.0.1"
    r = parse_http_line(http_line)
    check("http method", r["method"], "POST")
    check("http path", r["path"], "/v1/chat/completions")
    check("http status", r["status"], 200)
    check("http latency", r["latency_ms"], 2.798)
    check("http client_ip", r["client_ip"], "10.0.0.1")

    r_infer = parse_http_line(
        "[INFO] 2025/01/15 18:25:33 logger.go:45: [GET] /health HTTP/1.1 200 1ms 10.0.0.1", inference_only=True
    )
    check("inference_only filters health", r_infer, None)

    print("\n=== Testing normalize_message ===")
    check("url", normalize_message("Failed to connect to http://10.0.0.1:9965"), "Failed to connect to {url}")
    check("uuid", normalize_message("request abc12345-1234-5678-9012-abcdef123456 failed"), "request {uuid} failed")
    check(
        "ip:port",
        normalize_message("dial tcp 10.0.0.1:9965: connection refused"),
        "dial tcp {ip:port}: connection refused",
    )

    print("\n=== Testing match_select_release (token release type inference) ===")
    sample_lines = [
        "[INFO] 2026/04/12 10:00:00 logger.go:1: [request_id:r1] select worker (mixed): http://10.0.0.1:9965, count: 1",
        "[INFO] 2026/04/12 10:00:01 logger.go:1: [request_id:r1] release prefill tokens: http://10.0.0.1:9965, tokens: 10",
        "[INFO] 2026/04/12 10:00:02 logger.go:1: [request_id:r1] release worker: http://10.0.0.1:9965, count: 0",
    ]
    msr = match_select_release(sample_lines)
    check("mixed token_releases inferred", msr["type_summary"].get("mixed", {}).get("token_releases", 0), 1)
    check("prefill token_releases remains 0", msr["type_summary"].get("prefill", {}).get("token_releases", 0), 0)
    check("id consistency exact match", msr["id_consistency"].get("both_present_and_equal", 0), 1)

    mismatch_lines = [
        "[INFO] 2026/04/12 10:01:00 logger.go:1: [request_id:r2] select worker (decode): http://10.0.0.2:9965, count: 1",
        "[INFO] 2026/04/12 10:01:01 logger.go:1: [request_id:r3] release worker: http://10.0.0.2:9965, count: 0",
    ]
    mm = match_select_release(mismatch_lines)
    check("id mismatch detected", mm["id_consistency"].get("both_present_but_mismatch", 0), 1)

    print(f'\n{"=" * 40}')
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="FastDeploy Go Router Log Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("parse-http", help="解析 HTTP 请求行 (H1) → JSON Lines")
    p.add_argument("--inference-only", action="store_true", help="仅保留推理路径")

    sub.add_parser("parse-cache-strategy", help="解析 cache-aware 策略行 (H6) → JSON Lines")
    sub.add_parser("parse-stats", help="解析 [stats] 统计行 (H7) → JSON Lines")
    sub.add_parser("normalize-errors", help="ERROR/WARN 行模板归一化 → JSON Lines")
    sub.add_parser("match-select-release", help="匹配 select/release worker 事件")
    p = sub.add_parser("unsupported-requests", help="检测不匹配已知路由的请求")
    p.add_argument("--summary-only", action="store_true", help="仅输出汇总（不含详细列表）")
    sub.add_parser("self-test", help="运行内置测试")

    args = parser.parse_args()

    if args.command == "parse-http":
        _cli_parse_http(args)
    elif args.command == "parse-cache-strategy":
        _cli_parse_stream(parse_cache_strategy_line)
    elif args.command == "parse-stats":
        _cli_parse_stream(parse_stats_line)
    elif args.command == "normalize-errors":
        _cli_normalize_errors(args)
    elif args.command == "match-select-release":
        _cli_match_select_release(args)
    elif args.command == "unsupported-requests":
        _cli_unsupported_requests(args)
    elif args.command == "self-test":
        _cli_self_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
