#!/usr/bin/env python3
"""
Cache Analyzer — Cache 调度诊断

分析 cache-aware 调度策略：session 粘性、非最优选择评分、驱逐影响、
fallback 原因、冷启动识别、交叉诊断。
注意：cache 命中率数值分析由 stat-cache-hitrate skill 负责，本模块做策略诊断。
"""

import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chart import render_bar, render_table
from log_parser import parse_cache_strategy_line, parse_ts
from stats import compute_statistics, count_by

# ════════════════════════════════════════════════════════════════
# Fallback 分类
# ════════════════════════════════════════════════════════════════

TOKENIZER_WARN_RE = re.compile(r"tokenizer failed, fallback to char tokens")


def _strip_scheme(url):
    return re.sub(r"^https?://", "", url)


def classify_fallback(record, tokenizer_degraded_ts=None):
    """对 process_tokens 策略行分类 fallback 原因。

    Returns: 'A-Tokenizer退化' | 'B-char tokenize失败' | 'C-负载不均衡' | 'D-其他'
    """
    reason = record.get("reason", "")
    if "load imbalanced" in reason:
        return "C-负载不均衡"
    if "tokenize failed" in reason:
        return "B-char tokenize失败"
    return "D-其他"


# ════════════════════════════════════════════════════════════════
# 主分析函数
# ════════════════════════════════════════════════════════════════


def analyze_cache(log_file, tail=None, eviction_duration_mins=30, hit_ratio_weight=1.0, load_balance_weight=1.0):
    """分析 cache-aware 调度策略。

    Args:
        log_file: 日志文件路径
        tail: 尾部行数限制
        eviction_duration_mins: 驱逐时间（分钟，默认 30）
        hit_ratio_weight: hitRatio 权重（默认 1.0）
        load_balance_weight: loadBalance 权重（默认 1.0）

    Returns:
        dict: {strategy_dist, fallback_reasons, session_stickiness, suboptimal_selections,
               eviction_impact, cold_starts, hitratio_stats, diagnoses, summary}
    """
    h6_lines = _grep_lines(log_file, r"cache-aware prefill: final strategy:", tail)
    tokenizer_warn_lines = _grep_lines(log_file, r"tokenizer failed, fallback to char tokens", tail)

    # 解析策略行
    strategy_records = [r for line in h6_lines for r in [parse_cache_strategy_line(line)] if r]

    if not strategy_records:
        return {
            "strategy_dist": [],
            "fallback_reasons": [],
            "session_stickiness": {},
            "suboptimal_selections": [],
            "eviction_impact": [],
            "cold_starts": 0,
            "hitratio_stats": {},
            "diagnoses": [],
            "summary": "未检测到 cache-aware 策略日志",
        }

    # Tokenizer 退化次数
    tokenizer_degraded_count = len(tokenizer_warn_lines)

    # 策略分布
    strategy_dist = count_by(strategy_records, "strategy")

    # Fallback 原因
    fallback_records = [r for r in strategy_records if r.get("strategy") == "process_tokens"]
    fallback_reasons = []
    if fallback_records:
        for r in fallback_records:
            r["fallback_type"] = classify_fallback(r)
        fallback_reasons = count_by(fallback_records, "fallback_type")

    # hitRatio 统计
    hr_vals = [r.get("selected_hitRatio", 0) for r in strategy_records if "selected_hitRatio" in r]
    hitratio_stats = compute_statistics(hr_vals) if hr_vals else {}

    # Session 粘性分析
    session_stickiness = _analyze_session_stickiness(strategy_records)

    # 非最优选择分析
    suboptimal = _analyze_suboptimal(strategy_records, hit_ratio_weight, load_balance_weight)

    # 驱逐影响
    eviction_impact = _analyze_eviction(strategy_records, eviction_duration_mins)

    # 冷启动
    cold_starts = sum(1 for r in strategy_records if r.get("hitRatios") == {})

    total = len(strategy_records)
    cache_aware_count = sum(1 for r in strategy_records if r["strategy"] == "cache_aware_scoring")
    fallback_count = len(fallback_records)

    diagnoses = _diagnose(
        strategy_dist,
        fallback_reasons,
        session_stickiness,
        suboptimal,
        eviction_impact,
        cold_starts,
        total,
        tokenizer_degraded_count,
        hitratio_stats,
    )

    return {
        "strategy_dist": strategy_dist,
        "fallback_reasons": fallback_reasons,
        "session_stickiness": session_stickiness,
        "suboptimal_selections": suboptimal,
        "eviction_impact": eviction_impact,
        "cold_starts": cold_starts,
        "hitratio_stats": hitratio_stats,
        "tokenizer_degraded_count": tokenizer_degraded_count,
        "diagnoses": diagnoses,
        "summary": f"{total} 策略决策, cache_aware {cache_aware_count}, fallback {fallback_count}, "
        f"冷启动 {cold_starts}",
    }


def _analyze_session_stickiness(records):
    """Session 粘性分析。"""
    sessions = defaultdict(list)
    for r in records:
        sid = (r.get("tags") or {}).get("session_id")
        if sid and "selected" in r:
            sessions[sid].append(r["selected"])

    result = {}
    for sid, workers in sessions.items():
        if len(workers) < 2:
            continue
        same_count = sum(1 for i in range(1, len(workers)) if workers[i] == workers[i - 1])
        stickiness = round(same_count / (len(workers) - 1) * 100, 1)
        switches = [(i, workers[i - 1], workers[i]) for i in range(1, len(workers)) if workers[i] != workers[i - 1]]
        result[sid] = {
            "total_requests": len(workers),
            "stickiness_pct": stickiness,
            "switches": len(switches),
        }

    return result


def _analyze_suboptimal(records, hr_weight, lb_weight):
    """非最优选择分析：selected 的 hitRatio 不是最高时，重新计算 score 对比。"""
    suboptimal = []
    for r in records:
        if r.get("strategy") != "cache_aware_scoring":
            continue
        hit_ratios = r.get("hitRatios", {})
        loads = r.get("loads", {})
        selected = r.get("selected")
        if not hit_ratios or not selected or selected not in hit_ratios:
            continue

        max_hr = max(hit_ratios.values()) if hit_ratios else 0
        sel_hr = hit_ratios.get(selected, 0)

        if sel_hr >= max_hr:
            continue

        # 计算 scores: score = (100-hitRatio)/100 * hrWeight + loadRatio * lbWeight
        # Go 源码使用 maxLoad 做归一化: loadRatio = load / maxLoad
        max_load = max(loads.values()) if loads else 1
        max_load = max(max_load, 1)
        scores = {}
        for w_url in hit_ratios:
            hr = hit_ratios.get(w_url, 0)
            load = loads.get(w_url, 0)
            load_ratio = load / max_load
            score = (100 - hr) / 100 * hr_weight + load_ratio * lb_weight
            scores[w_url] = round(score, 4)

        best_by_hr = min(hit_ratios, key=lambda w: -hit_ratios[w])
        sel_score = scores.get(selected, 0)
        best_hr_score = scores.get(best_by_hr, 0)

        # 分类原因
        load_diff = abs(loads.get(selected, 0) - loads.get(best_by_hr, 0))
        if load_diff > 5:
            reason = "负载主导"
        elif max_hr < 10:
            reason = "区分度不够"
        elif abs(sel_score - best_hr_score) < 0.05:
            reason = "正常竞争"
        else:
            reason = "综合权衡"

        suboptimal.append(
            {
                "ts": r.get("ts", ""),
                "selected": _strip_scheme(selected),
                "selected_hr": sel_hr,
                "best_hr_worker": _strip_scheme(best_by_hr),
                "best_hr": max_hr,
                "reason": reason,
            }
        )

    return suboptimal


def _analyze_eviction(records, eviction_mins):
    """驱逐影响分析：同 session 连续请求间隔 > eviction_duration。"""
    sessions = defaultdict(list)
    for r in records:
        sid = (r.get("tags") or {}).get("session_id")
        ts = r.get("ts")
        if sid and ts:
            sessions[sid].append(r)

    impacts = []
    for sid, reqs in sessions.items():
        reqs.sort(key=lambda x: x.get("ts", ""))
        for i in range(1, len(reqs)):
            try:
                prev_dt = parse_ts(reqs[i - 1]["ts"])
                curr_dt = parse_ts(reqs[i]["ts"])
                interval_mins = (curr_dt - prev_dt).total_seconds() / 60
                if interval_mins > eviction_mins:
                    curr_hr = reqs[i].get("selected_hitRatio", -1)
                    impacts.append(
                        {
                            "session_id": sid,
                            "interval_mins": round(interval_mins, 1),
                            "hitRatio_after": curr_hr,
                            "evicted": curr_hr == 0,
                        }
                    )
            except (ValueError, KeyError):
                pass

    return impacts


def _diagnose(
    strategy_dist,
    fallback_reasons,
    session_stickiness,
    suboptimal,
    eviction_impact,
    cold_starts,
    total,
    tokenizer_degraded_count,
    hitratio_stats,
):
    """生成 cache 调度诊断。"""
    diagnoses = []

    # Tokenizer 退化
    if tokenizer_degraded_count > 0:
        pct = round(tokenizer_degraded_count / max(total, 1) * 100, 1)
        sev = "HIGH" if pct > 10 else "MEDIUM"
        diagnoses.append(
            {
                "severity": sev,
                "message": f"Tokenizer 退化 {tokenizer_degraded_count} 次 ({pct}%)，精度降低",
                "source_layer": "Router",
            }
        )

    # Fallback 比例
    for s in strategy_dist:
        if s["value"] == "process_tokens" and s["pct"] > 20:
            diagnoses.append(
                {
                    "severity": "MEDIUM",
                    "message": f'Fallback 到 process_tokens {s["pct"]}%，cache-aware 策略未生效',
                    "source_layer": "Router",
                }
            )

    # 非最优选择
    if suboptimal and total > 0:
        pct = round(len(suboptimal) / total * 100, 1)
        if pct > 20:
            diagnoses.append(
                {
                    "severity": "MEDIUM",
                    "message": f"非最优选择 {pct}%（{len(suboptimal)}/{total}）",
                    "source_layer": "Router",
                }
            )

    # 冷启动
    if cold_starts > 0 and total > 0:
        pct = round(cold_starts / total * 100, 1)
        if pct > 10:
            diagnoses.append(
                {"severity": "LOW", "message": f"冷启动 {pct}%（hitRatios=map[]）", "source_layer": "Router"}
            )

    # 驱逐影响
    evicted = [e for e in eviction_impact if e["evicted"]]
    if evicted:
        diagnoses.append(
            {
                "severity": "MEDIUM",
                "message": f"{len(evicted)} 次驱逐后 hitRatio=0，考虑增大 eviction-duration-mins",
                "source_layer": "Router",
            }
        )

    # hitRatio 整体偏低
    if hitratio_stats.get("mean", 100) < 20:
        diagnoses.append(
            {
                "severity": "LOW",
                "message": f'平均 hitRatio {hitratio_stats["mean"]}%，缓存效果较差',
                "source_layer": "Router",
            }
        )

    return diagnoses


# ════════════════════════════════════════════════════════════════
# 报告格式化
# ════════════════════════════════════════════════════════════════


def format_cache_report(result):
    """将分析结果格式化为终端报告。"""
    sections = ["## Cache 调度诊断", ""]
    sections.append(f'  {result["summary"]}')
    sections.append("")

    if result["diagnoses"]:
        sections.append("### 诊断")
        sections.append("")
        for d in result["diagnoses"]:
            sections.append(f'  [{d["severity"]}] [{d["source_layer"]}] {d["message"]}')
        sections.append("")

    # 策略分布
    if result["strategy_dist"]:
        sections.append("### 策略分布")
        sections.append("")
        bar_data = [{"label": s["value"], "value": s["pct"], "count": s["count"]} for s in result["strategy_dist"]]
        sections.append(render_bar(bar_data, show_count=True))
        sections.append("")

    # hitRatio 统计
    hs = result.get("hitratio_stats", {})
    if hs:
        sections.append("### hitRatio 统计")
        sections.append("")
        sections.append(
            f'  mean={hs.get("mean",0)}%  p50={hs.get("p50",0)}%  p90={hs.get("p90",0)}%  '
            f'p99={hs.get("p99",0)}%  max={hs.get("max",0)}%'
        )
        sections.append("")

    # Fallback 原因
    if result["fallback_reasons"]:
        sections.append("### Fallback 原因分布")
        sections.append("")
        bar_data = [{"label": f["value"], "value": f["pct"], "count": f["count"]} for f in result["fallback_reasons"]]
        sections.append(render_bar(bar_data, show_count=True))
        sections.append("")

    # Tokenizer 退化
    if result.get("tokenizer_degraded_count", 0) > 0:
        sections.append(f'  Tokenizer 退化: {result["tokenizer_degraded_count"]} 次')
        sections.append("")

    # Session 粘性
    stickiness = result.get("session_stickiness", {})
    if stickiness:
        sections.append("### Session 粘性")
        sections.append("")
        table_data = [
            {
                "Session": sid[:16],
                "请求数": str(s["total_requests"]),
                "粘性率": f'{s["stickiness_pct"]}%',
                "切换次数": str(s["switches"]),
            }
            for sid, s in sorted(stickiness.items(), key=lambda x: x[1]["stickiness_pct"])
        ]
        sections.append(
            render_table(
                table_data[:10],
                columns=["Session", "请求数", "粘性率", "切换次数"],
                right_align={"请求数", "粘性率", "切换次数"},
            )
        )
        sections.append("")

    # 非最优选择
    if result.get("suboptimal_selections"):
        subs = result["suboptimal_selections"]
        sections.append(f"### 非最优选择 ({len(subs)} 次)")
        sections.append("")
        reason_counts = defaultdict(int)
        for s in subs:
            reason_counts[s["reason"]] += 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            sections.append(f"  {reason}: {count} 次")
        sections.append("")

    # 驱逐影响
    if result.get("eviction_impact"):
        evictions = result["eviction_impact"]
        evicted = [e for e in evictions if e["evicted"]]
        sections.append(f"### 驱逐影响 ({len(evictions)} 次超时, {len(evicted)} 次缓存失效)")
        sections.append("")

    # 冷启动
    if result.get("cold_starts", 0) > 0:
        sections.append(f'  冷启动: {result["cold_starts"]} 次（hitRatios=map[]）')
        sections.append("")

    return "\n".join(sections)


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
