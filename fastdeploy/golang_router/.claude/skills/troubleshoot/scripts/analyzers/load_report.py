#!/usr/bin/env python3
"""Load report formatter."""

from chart import render_bar, render_sparkline, render_table


def _strip_scheme(url):
    import re
    return re.sub(r"^https?://", "", url)


def format_load_report(result):
    """将分析结果格式化为终端报告。

    Returns:
        tuple: (summary_text, detail_text)
    """
    sections = ["## 负载与计数器分析", ""]
    sections.append(f'  {result["summary"]}')
    sections.append("")
    detail_sections = ["# 负载与计数器详情", ""]
    detail_sections.append(f'总结: {result["summary"]}')
    detail_sections.append("")

    if result["diagnoses"]:
        sections.append("### 诊断")
        sections.append("")
        max_diag_in_summary = 8
        for d in result["diagnoses"][:max_diag_in_summary]:
            sections.append(f'  [{d["severity"]}] [{d["source_layer"]}] {d["message"]}')
        if len(result["diagnoses"]) > max_diag_in_summary:
            sections.append(f'  ... 其余 {len(result["diagnoses"]) - max_diag_in_summary} 项见 detail 报告')
        sections.append("")
        detail_sections.append("## 诊断")
        detail_sections.append("")
        for d in result["diagnoses"]:
            detail_sections.append(f'[{d["severity"]}] [{d["source_layer"]}] {d["message"]}')
        detail_sections.append("")

    # 负载概览
    ls = result.get("load_stats", {})
    if ls:
        sections.append("### 负载概览 (total_running)")
        sections.append("")
        sections.append("  说明: stats 采样来自 `[stats]` 周期日志（通常每 5s 一条），用于观察当前并发与负载变化趋势。")
        sections.append(
            f'  mean={ls.get("mean",0)}  p50={ls.get("p50",0)}  p90={ls.get("p90",0)}  '
            f'p99={ls.get("p99",0)}  max={ls.get("max",0)}  stddev={ls.get("stddev",0)}'
        )
        sections.append("")

    # Per-Worker 负载
    if result["worker_load"]:
        sections.append("### Per-Worker 负载")
        sections.append("")
        bar_data = [
            {"label": w["worker"][:25], "value": min(100, w["avg_running"] * 5), "count": w["avg_running"]}
            for w in result["worker_load"]
        ]
        sections.append(render_bar(bar_data, show_count=True))
        sections.append("")

    # 负载趋势
    if result["load_trend"] and len(result["load_trend"]) > 1:
        sections.append("### 负载趋势")
        sections.append("")
        sections.append(
            render_sparkline(
                result["load_trend"], value_field="total_running_mean", title="Total Running", y_label="req"
            )
        )
        sections.append("")

    # Counter 异常
    if result["counter_anomalies"]:
        sections.append("### 计数器异常")
        sections.append("")
        for a in result["counter_anomalies"]:
            workers_str = ", ".join(f'{_strip_scheme(w)}({c})' for w, c in a["workers"].items())
            sections.append(f'  {a["type"]}: {a["total"]} 次 [{workers_str}]')
        sections.append("")
        detail_sections.append("## 计数器异常")
        detail_sections.append("")
        for a in result["counter_anomalies"]:
            workers_str = ", ".join(f'{_strip_scheme(w)}({c})' for w, c in a["workers"].items())
            detail_sections.append(f'- {a["type"]}: {a["total"]} 次 [{workers_str}]')
        detail_sections.append("")

    # 按 prefill / decode / mixed 分类统计
    type_summary = result.get("select_release", {}).get("type_summary", {})
    if type_summary:
        sections.append("### 按类型统计（prefill / decode / mixed）")
        sections.append("")
        type_rows = []
        for t in ("prefill", "decode", "mixed", "unknown"):
            s = type_summary.get(t)
            if not s:
                continue
            token_display = "-"
            if t == "prefill":
                token_display = f'{s.get("token_selects",0)}/{s.get("token_releases",0)}'
            elif t == "mixed" and (s.get("token_selects", 0) > 0 or s.get("token_releases", 0) > 0):
                token_display = f'{s.get("token_selects",0)}/{s.get("token_releases",0)}'
            type_rows.append(
                {
                    "type": t,
                    "counter(S/R)": f'{s.get("counter_selects",0)}/{s.get("counter_releases",0)}',
                    "token(S/R)": token_display,
                }
            )
        if type_rows:
            sections.append(render_table(type_rows, columns=["type", "counter(S/R)", "token(S/R)"]))
            sections.append("")
            sections.append("  说明: prefill/mixed 的 token-select 同时表示 request counter + token counter 增加；decode 仅 request counter。")
            sections.append("  说明: token-release 由同 worker 邻近 select 推断到 prefill/mixed，不直接依赖 `release prefill tokens` 文本。")
            if type_summary.get("unknown"):
                sections.append("  说明: unknown 表示日志里缺少 worker type，且无法从邻近 select/release 关系推断。")
            sections.append("")
            detail_sections.append("## 按类型统计")
            detail_sections.append("")
            detail_sections.append(render_table(type_rows, columns=["type", "counter(S/R)", "token(S/R)"]))
            detail_sections.append("")

    id_cov = result.get("select_release", {}).get("id_coverage", {})
    if id_cov:
        sections.append("### 请求标识覆盖（基于 select 近似请求数）")
        sections.append("")
        sections.append(
            "  total={total} | with_request_id={with_rid} | without_request_id={without_rid} | "
            "with_alt_id={with_alt} | without_any_id={without_any}".format(
                total=id_cov.get("total_requests_estimated", 0),
                with_rid=id_cov.get("with_request_id", 0),
                without_rid=id_cov.get("without_request_id", 0),
                with_alt=id_cov.get("with_alt_id", 0),
                without_any=id_cov.get("without_any_id", 0),
            )
        )
        if id_cov.get("without_any_id", 0) > 0:
            sections.append("  ℹ 无 request/session/trace/req_id 时，不做退化匹配，仅统计为 untracked。")
        sections.append("  字段说明: total=select 事件总数估算；with_request_id=含 request_id；without_request_id=不含 request_id；with_alt_id=含 req_id/trace_id/session_id；without_any_id=四类 ID 都缺失。")
        sections.append("")
        detail_sections.append("## 请求标识覆盖字段说明")
        detail_sections.append("")
        detail_sections.append(
            "- total: select 事件总数（近似请求数）\n"
            "- with_request_id: 携带 request_id 的 select 数\n"
            "- without_request_id: 未携带 request_id 的 select 数\n"
            "- with_alt_id: 无 request_id 但携带 req_id/trace_id/session_id 的 select 数\n"
            "- without_any_id: 四类 ID 都没有，无法做请求级关联"
        )
        detail_sections.append("")

    # Select/Release 匹配
    sr = result.get("select_release", {})
    if sr.get("per_worker"):
        sections.append("### Select/Release 匹配")
        sections.append("")
        id_cov = sr.get("id_coverage", {})
        no_correlatable_id = (id_cov.get("with_request_id", 0) + id_cov.get("with_alt_id", 0)) == 0
        table_data = []
        for w_url, pw in sorted(sr["per_worker"].items()):
            delta_display = "N/A" if no_correlatable_id else str(pw["delta"])
            table_data.append(
                {
                    "Worker": _strip_scheme(w_url),
                    "ReqSelect": str(pw["selects"]),
                    "ReqRelease": str(pw["releases"]),
                    "ReqDelta": delta_display,
                    "TokenSelect": str(pw.get("token_selects", 0)),
                    "TokenRelease": str(pw.get("token_releases", 0)),
                }
            )
        sections.append(
            render_table(
                table_data,
                columns=["Worker", "ReqSelect", "ReqRelease", "ReqDelta", "TokenSelect", "TokenRelease"],
                right_align={"ReqSelect", "ReqRelease", "ReqDelta", "TokenSelect", "TokenRelease"},
            )
        )
        sections.append("")
        if no_correlatable_id:
            sections.append("  ℹ 当前样本无可关联 ID，Delta 不用于请求泄漏结论。")
            sections.append("")
        sections.append("  说明: TokenSelect 按 worker type 统计（prefill + mixed 的 select 都计入），不依赖日志里是否出现 tokens 字段。")
        sections.append("")
        detail_sections.append("## Select/Release Per-Worker")
        detail_sections.append("")
        detail_sections.append(
            render_table(
                table_data,
                columns=["Worker", "ReqSelect", "ReqRelease", "ReqDelta", "TokenSelect", "TokenRelease"],
                right_align={"ReqSelect", "ReqRelease", "ReqDelta", "TokenSelect", "TokenRelease"},
            )
        )
        detail_sections.append("")

    if sr.get("unmatched_selects"):
        sections.append(f'  ⚠ {len(sr["unmatched_selects"])} 个未匹配 select（疑似请求卡住）')
        sections.append("  解释: 出现 request select，但在 request release 口径下找不到匹配。可能是请求卡住、日志缺失、或窗口外释放。")
        for u in sr["unmatched_selects"][:3]:
            sections.append(f'    [{u.get("select_ts","")}] {_strip_scheme(u["worker"])} ({u["type"]})')
        sections.append("  > 完整列表见: [details/load_select_release.md](details/load_select_release.md)")
        sections.append("")
        detail_sections.append("## 未匹配 select（完整）")
        detail_sections.append("")
        for u in sr["unmatched_selects"]:
            detail_sections.append(
                f'- [{u.get("select_ts","")}] worker={_strip_scheme(u["worker"])} type={u["type"]} note={u.get("note","")}'
            )
        detail_sections.append("")

    if sr.get("untracked_selects"):
        sections.append(f'  ℹ {len(sr["untracked_selects"])} 个 select 缺少可关联 ID，未参与卡住判定')
        for u in sr["untracked_selects"][:3]:
            sections.append(f'    [{u.get("select_ts","")}] {_strip_scheme(u["worker"])} ({u["type"]})')
        sections.append("  > 完整列表见: [details/load_select_release.md](details/load_select_release.md)")
        sections.append("")
        detail_sections.append("## Untracked selects（缺少可关联 ID）")
        detail_sections.append("")
        for u in sr["untracked_selects"]:
            detail_sections.append(
                f'- [{u.get("select_ts","")}] worker={_strip_scheme(u["worker"])} type={u["type"]} note={u.get("note","")}'
            )
        detail_sections.append("")

    if sr.get("failed_selects"):
        sections.append(f'  ⚠ Failed to select: {len(sr["failed_selects"])} 次')
        sections.append("  解释: 路由在该时刻未能选出可用 worker，通常意味着可用池不足或健康状态异常。")
        sections.append("")
        detail_sections.append("## Failed to select")
        detail_sections.append("")
        for f in sr["failed_selects"]:
            detail_sections.append(f'- [{f.get("ts","")}] line={f.get("line","")}')
        detail_sections.append("")

    # Token 统计
    if result.get("token_stats"):
        sections.append("### Token 计数器")
        sections.append("")
        sections.append(
            render_table(
                result["token_stats"],
                columns=["worker", "alloc_count", "alloc_avg", "release_count"],
                right_align={"alloc_count", "alloc_avg", "release_count"},
            )
        )
        sections.append("")

    return "\n".join(sections), "\n".join(detail_sections)
