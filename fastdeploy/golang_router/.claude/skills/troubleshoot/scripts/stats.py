#!/usr/bin/env python3
"""
Stats — 通用统计计算工具

提供百分位数、分布、时间窗口聚合、分组计数等通用统计函数。
不含任何业务逻辑或日志格式依赖。

Python 3 stdlib only，零依赖。
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta

# ════════════════════════════════════════════════════════════════
# 百分位数与基础统计
# ════════════════════════════════════════════════════════════════


def percentile(sorted_vals, p):
    """从已排序列表计算第 p 百分位数（线性插值）。"""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_statistics(values, percentiles_list=None, distribution_spec=None):
    """计算一组数值的统计量。

    Args:
        values: 数值列表
        percentiles_list: 要计算的百分位数列表，默认 [50, 90, 95, 99]
        distribution_spec: 分布区间规格字符串，如 '0-20,20-40,40-60,60-80,80-100'

    Returns:
        dict: {count, min, max, mean, sum, stddev, p50, p90, ..., distribution}
    """
    if percentiles_list is None:
        percentiles_list = [50, 90, 95, 99]

    if not values:
        result = {"count": 0, "min": 0, "max": 0, "mean": 0, "sum": 0, "stddev": 0}
        for p in percentiles_list:
            result[f"p{p}"] = 0
        if distribution_spec is not None:
            result["distribution"] = []
        return result

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    stddev = math.sqrt(variance)

    result = {
        "count": n,
        "min": round(sorted_vals[0], 3),
        "max": round(sorted_vals[-1], 3),
        "mean": round(mean, 3),
        "sum": round(total, 3),
        "stddev": round(stddev, 3),
    }

    for p in percentiles_list:
        result[f"p{p}"] = round(percentile(sorted_vals, p), 3)

    if distribution_spec is not None:
        result["distribution"] = compute_distribution(sorted_vals, distribution_spec)

    return result


def compute_distribution(sorted_vals, spec_str):
    """根据区间规格计算分布直方图。

    spec_str 示例：'0-20,20-40,40-60,60-80,80-100'
    每个区间是左闭右开 [lo, hi)。
    """
    buckets = _parse_distribution_spec(spec_str)
    n = len(sorted_vals)
    result = []
    for b in buckets:
        if b[0] == "lt":
            count = sum(1 for v in sorted_vals if v < b[1])
            label = b[2]
        elif b[0] == "gt":
            count = sum(1 for v in sorted_vals if v > b[1])
            label = b[2]
        elif b[0] == "range":
            count = sum(1 for v in sorted_vals if b[1] <= v < b[2])
            label = b[3]
        else:
            continue
        result.append({"range": label, "count": count, "pct": round(count / n * 100, 1) if n else 0})
    return result


def _parse_distribution_spec(spec_str):
    """解析分布区间规格：'<100,100-500,>1000' → bucket 定义列表。"""
    buckets = []
    for part in spec_str.split(","):
        part = part.strip()
        if part.startswith("<"):
            buckets.append(("lt", float(part[1:]), part))
        elif part.startswith(">"):
            buckets.append(("gt", float(part[1:]), part))
        elif "-" in part:
            lo, hi = part.split("-", 1)
            buckets.append(("range", float(lo), float(hi), part))
    return buckets


# ════════════════════════════════════════════════════════════════
# 时间窗口聚合
# ════════════════════════════════════════════════════════════════


def time_bucket(records, window="auto", agg_specs=None, ts_field="ts"):
    """按时间窗口聚合记录。

    Args:
        records: dict 列表，每个 dict 必须有 ts_field 字段
        window: 窗口大小 '5s'/'1m'/'5m'/'auto'
        agg_specs: 聚合规格列表 [(field, func), ...]，如 [('selected_hitRatio', 'mean')]
                   func 支持：count, sum, mean, min, max, pNN
        ts_field: 时间戳字段名

    Returns:
        list[dict]: 每个窗口一条记录 {bucket, count, field_func, ...}
    """
    if agg_specs is None:
        agg_specs = [("_", "count")]

    if not records:
        return []

    window_td = _parse_window(window, records, ts_field)

    # 按窗口分组
    buckets = defaultdict(list)
    for r in records:
        ts_str = r.get(ts_field, "")
        if not ts_str:
            continue
        try:
            dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            continue
        bucket_dt = _align_to_bucket(dt, window_td)
        bucket_key = bucket_dt.strftime("%Y/%m/%d %H:%M:%S")
        buckets[bucket_key].append(r)

    # 按时间排序并聚合
    result = []
    for bucket_key in sorted(buckets.keys()):
        bucket_records = buckets[bucket_key]
        entry = {"bucket": bucket_key, "count": len(bucket_records)}

        for field, func in agg_specs:
            if field == "_":
                if func == "count":
                    entry["count"] = len(bucket_records)
                continue

            values = []
            for r in bucket_records:
                v = r.get(field)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass

            out_key = f"{field}_{func}"
            entry[out_key] = _aggregate_values(values, func)

        result.append(entry)

    return result


def _parse_window(window_str, records, ts_field):
    """解析窗口字符串为 timedelta。'auto' 根据数据跨度自动选择。"""
    if window_str == "auto":
        timestamps = []
        for r in records:
            ts_str = r.get(ts_field, "")
            if ts_str:
                try:
                    timestamps.append(datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S"))
                except ValueError:
                    pass
        if len(timestamps) < 2:
            return timedelta(minutes=1)
        span = max(timestamps) - min(timestamps)
        if span < timedelta(minutes=30):
            return timedelta(seconds=5)
        elif span < timedelta(hours=3):
            return timedelta(minutes=1)
        else:
            return timedelta(minutes=5)
    elif window_str.endswith("s"):
        return timedelta(seconds=int(window_str[:-1]))
    elif window_str.endswith("m"):
        return timedelta(minutes=int(window_str[:-1]))
    elif window_str.endswith("h"):
        return timedelta(hours=int(window_str[:-1]))
    return timedelta(minutes=1)


def _align_to_bucket(dt, window_td):
    """将 datetime 对齐到窗口边界。"""
    secs = max(1, int(window_td.total_seconds()))
    epoch = datetime(dt.year, dt.month, dt.day)
    offset = int((dt - epoch).total_seconds())
    aligned = (offset // secs) * secs
    return epoch + timedelta(seconds=aligned)


def _aggregate_values(values, func):
    """用指定函数聚合一组数值。"""
    if not values:
        return 0
    if func == "count":
        return len(values)
    elif func == "sum":
        return round(sum(values), 3)
    elif func == "mean":
        return round(sum(values) / len(values), 3)
    elif func == "min":
        return round(min(values), 3)
    elif func == "max":
        return round(max(values), 3)
    elif func.startswith("p"):
        p = int(func[1:])
        return round(percentile(sorted(values), p), 3)
    return 0


# ════════════════════════════════════════════════════════════════
# 分组计数
# ════════════════════════════════════════════════════════════════


def count_by(records, field, top_n=None):
    """按指定字段分组计数。

    Args:
        records: dict 列表
        field: 分组字段名
        top_n: 只返回前 N 个（按计数降序）

    Returns:
        list[dict]: [{value, count, pct}]，按计数降序排列
    """
    counts = defaultdict(int)
    total = 0
    for r in records:
        val = r.get(field)
        if val is not None:
            counts[str(val)] += 1
            total += 1

    result = []
    for val, count in sorted(counts.items(), key=lambda x: -x[1]):
        result.append({"value": val, "count": count, "pct": round(count / total * 100, 1) if total else 0})

    if top_n:
        result = result[:top_n]

    return result
