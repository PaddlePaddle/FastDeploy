#!/usr/bin/env python3
"""
窗口明细压缩工具：合并连续空窗口，降低 per_window_data.md 噪声。
"""

RUNNING_COL = "Total Running (prefill≈stats/2)"


def _is_blank_window_row(row):
    """判断是否为空窗口（无 Prefix/Session 明细值）。"""
    return (
        row.get("Prefix HR") == "-"
        and row.get("Session HR") == "-"
        and row.get("Scoring") in {"0", 0}
        and row.get("Fallback") in {"0", 0}
    )


def merge_blank_window_rows(rows, min_merge_len=5):
    """合并连续空窗口，避免明细表被大量 '-' 行淹没。

    对于连续空窗口段（长度 >= min_merge_len），压缩成 3 行：
      1) 起始时间行
      2) 合并说明行（含窗口数量）
      3) 结束时间行
    """
    if not rows:
        return rows

    merged = []
    i = 0
    while i < len(rows):
        if not _is_blank_window_row(rows[i]):
            merged.append(rows[i])
            i += 1
            continue

        j = i
        while j < len(rows) and _is_blank_window_row(rows[j]):
            j += 1

        seg_len = j - i
        if seg_len < min_merge_len:
            merged.extend(rows[i:j])
            i = j
            continue

        start_t = rows[i]["Time"]
        end_t = rows[j - 1]["Time"]
        merged.append(
            {
                "Time": start_t,
                "Prefix HR": "-",
                "Session HR": "-",
                "Scoring": "0",
                "Fallback": "0",
                RUNNING_COL: rows[i].get(RUNNING_COL, "-"),
            }
        )
        merged.append(
            {
                "Time": "|",
                "Prefix HR": "-",
                "Session HR": f"merged {seg_len} windows",
                "Scoring": "0",
                "Fallback": "0",
                RUNNING_COL: "-",
            }
        )
        merged.append(
            {
                "Time": end_t,
                "Prefix HR": "-",
                "Session HR": "-",
                "Scoring": "0",
                "Fallback": "0",
                RUNNING_COL: rows[j - 1].get(RUNNING_COL, "-"),
            }
        )
        i = j

    return merged
