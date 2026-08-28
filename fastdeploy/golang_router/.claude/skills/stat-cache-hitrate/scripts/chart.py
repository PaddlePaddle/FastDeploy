#!/usr/bin/env python3
"""
Chart — 终端可视化渲染工具

提供 sparkline 折线图、Unicode 柱状图、Markdown 表格的渲染函数。
所有函数返回字符串（不直接打印），方便组装到报告中。

Python 3 stdlib only，零依赖。
"""


# ════════════════════════════════════════════════════════════════
# Sparkline 折线图
# ════════════════════════════════════════════════════════════════

BLOCK_CHARS = " ▁▂▃▄▅▆▇█"


def render_sparkline(
    records, value_field="value", bucket_field="bucket", title=None, y_label=None, y_range=None, width=60
):
    """渲染 8 级 Unicode sparkline 折线图。

    Args:
        records: dict 列表，每个 dict 包含 bucket_field 和 value_field
        value_field: 数值字段名
        bucket_field: 时间桶字段名
        title: 图表标题
        y_label: Y 轴标签（如 '%'）
        y_range: Y 轴范围 (min, max) 元组，None 则自动
        width: 图表宽度（字符数）

    Returns:
        str: 渲染后的图表文本
    """
    if not records:
        return "  (no data)"

    all_values = []
    for r in records:
        v = r.get(value_field)
        if v is not None:
            all_values.append(float(v))

    if not all_values:
        return "  (no numeric data)"

    # Y 轴范围
    if y_range:
        y_min, y_max = y_range
    else:
        y_min = min(all_values)
        y_max = max(all_values)
        if y_max == y_min:
            y_min = 0 if y_max > 0 else y_max - 1
            y_max = max(y_max, 1)

    y_span = y_max - y_min if y_max != y_min else 1

    # 降采样
    n = len(records)
    if n > width:
        step = n / width
        sampled = []
        for i in range(width):
            start_idx = int(i * step)
            end_idx = int((i + 1) * step)
            chunk = records[start_idx:end_idx]
            vals = [float(r.get(value_field, 0)) for r in chunk if r.get(value_field) is not None]
            avg_record = {
                bucket_field: chunk[0].get(bucket_field, ""),
                value_field: sum(vals) / len(vals) if vals else 0,
            }
            sampled.append(avg_record)
        records = sampled

    lines = []

    # 标题行
    def fmt_val(v):
        if abs(v) >= 1000:
            return f"{v:.0f}"
        elif abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    header_parts = []
    if title:
        header_parts.append(title)
    header_parts.append(f"min={fmt_val(min(all_values))}")
    header_parts.append(f"max={fmt_val(max(all_values))}")
    if y_label:
        header_parts.append(f"({y_label})")
    lines.append("  " + "  ".join(header_parts))

    # Sparkline 字符
    spark_chars = []
    for r in records:
        v = r.get(value_field)
        if v is None:
            spark_chars.append(" ")
            continue
        v = float(v)
        normalized = (v - y_min) / y_span
        level = max(0, min(8, round(normalized * 8)))
        spark_chars.append(BLOCK_CHARS[level])
    lines.append("  " + "".join(spark_chars))

    # X 轴标签
    data_width = len(records)
    if data_width > 0:

        def short_bucket(r):
            b = str(r.get(bucket_field, ""))
            if " " in b:
                b = b.split(" ")[-1]
            return b[:5] if len(b) >= 5 else b

        lbl_width = 6
        max_labels = max(1, data_width // lbl_width)
        n_records = len(records)

        if n_records <= 2:
            indices = list(range(n_records))
        elif n_records <= max_labels:
            indices = [0, n_records - 1]
        else:
            n_labels = min(5, max(2, max_labels))
            indices = [int(i * (n_records - 1) / (n_labels - 1)) for i in range(n_labels)]

        label_line = [" "] * (data_width + lbl_width + 2)
        last_end = -1
        for idx in indices:
            lbl = short_bucket(records[idx])
            pos = idx
            if pos < last_end:
                continue
            for ci, c in enumerate(lbl):
                p = pos + ci
                if p < len(label_line):
                    label_line[p] = c
            last_end = pos + len(lbl) + 1
        lines.append("  " + "".join(label_line).rstrip())

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Unicode 柱状图
# ════════════════════════════════════════════════════════════════


def render_bar(data, bar_width=20, show_count=False):
    """渲染 Unicode 柱状图。

    Args:
        data: dict 列表，每个 dict 包含 label, value（百分比 0-100）, 可选 count
        bar_width: 柱状图宽度（字符数）
        show_count: 是否显示绝对数量

    Returns:
        str: 渲染后的图表文本
    """
    if not data:
        return "  (no data)"

    max_label_len = max(len(str(d.get("label", ""))) for d in data)
    max_label_len = max(max_label_len, 4)

    lines = []
    for d in data:
        label = str(d.get("label", ""))
        value = float(d.get("value", 0))
        count = d.get("count")

        filled = round(value / 100 * bar_width) if value > 0 else 0
        filled = max(1, filled) if value > 0 else 0
        filled = min(bar_width, filled)
        empty = bar_width - filled
        bar = "█" * filled + "░" * empty

        line = f"  {label:<{max_label_len}}  {bar} {value:>5.1f}%"
        if show_count and count is not None:
            line += f"  (N={count})"
        lines.append(line)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Markdown 表格
# ════════════════════════════════════════════════════════════════


def render_table(data, columns=None, right_align=None):
    """渲染 Markdown 表格。

    Args:
        data: dict 列表
        columns: 列名列表，None 则用第一条记录的所有 key
        right_align: 右对齐的列名集合

    Returns:
        str: 渲染后的表格文本
    """
    if not data:
        return "  (no data)"

    if columns is None:
        columns = list(data[0].keys())
    if right_align is None:
        right_align = set()

    # 计算列宽
    col_widths = {}
    for col in columns:
        col_widths[col] = len(col)
        for row in data:
            val = str(row.get(col, ""))
            col_widths[col] = max(col_widths[col], len(val))

    # 表头
    header_parts = []
    sep_parts = []
    for col in columns:
        w = col_widths[col]
        if col in right_align:
            header_parts.append(f" {col:>{w}} ")
        else:
            header_parts.append(f" {col:<{w}} ")
        sep_parts.append("-" * (w + 2))

    lines = []
    lines.append("|" + "|".join(header_parts) + "|")
    lines.append("|" + "|".join(sep_parts) + "|")

    # 数据行
    for row in data:
        row_parts = []
        for col in columns:
            val = str(row.get(col, ""))
            w = col_widths[col]
            if col in right_align:
                row_parts.append(f" {val:>{w}} ")
            else:
                row_parts.append(f" {val:<{w}} ")
        lines.append("|" + "|".join(row_parts) + "|")

    return "\n".join(lines)
