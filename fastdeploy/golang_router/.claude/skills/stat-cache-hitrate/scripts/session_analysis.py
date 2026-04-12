#!/usr/bin/env python3
"""
Session 维度分析：聚合每个 session 的命中率、worker 切换与突降请求。
"""

from collections import defaultdict


def compute_session_details(strategies, strip_scheme):
    """按 session_id（优先）或 trace_id（兜底）统计命中详情。"""

    def _req_id_from_tags(tags, fallback):
        return tags.get("request_id") or tags.get("req_id") or tags.get("trace_id") or fallback

    session_records = defaultdict(list)
    for idx, rec in enumerate(strategies):
        if rec.get("strategy") != "cache_aware_scoring":
            continue
        tags = rec.get("tags", {}) or {}
        session_id = tags.get("session_id")
        trace_id = tags.get("trace_id")
        identity = session_id or trace_id
        if not identity:
            continue
        session_records[identity].append((idx, rec))

    rows = []
    for identity, items in session_records.items():
        items.sort(key=lambda x: (x[1].get("ts_ms", ""), x[1].get("ts", ""), x[0]))
        recs = [r for _, r in items]
        hits = [int(r.get("selected_hitRatio", 0)) for r in recs]
        if not hits:
            continue

        non_first = hits[1:]
        avg_excl_first = round(sum(non_first) / len(non_first), 1) if non_first else "-"
        workers = {r.get("selected", "") for r in recs if r.get("selected")}

        prefill_urls = []
        for r in recs:
            u = r.get("selected", "")
            if u and u not in prefill_urls:
                prefill_urls.append(u)

        switch_events = []
        sharp_drop_req_ids = []
        for i in range(1, len(recs)):
            prev_r = recs[i - 1]
            curr_r = recs[i]
            prev_url = prev_r.get("selected", "")
            curr_url = curr_r.get("selected", "")
            prev_tags = prev_r.get("tags", {}) or {}
            curr_tags = curr_r.get("tags", {}) or {}
            prev_req = _req_id_from_tags(prev_tags, f"idx#{i}")
            curr_req = _req_id_from_tags(curr_tags, f"idx#{i+1}")

            if prev_url and curr_url and prev_url != curr_url:
                switch_events.append(f"{prev_req}->{curr_req} ({strip_scheme(prev_url)}→{strip_scheme(curr_url)})")

            prev_hit = int(prev_r.get("selected_hitRatio", 0))
            curr_hit = int(curr_r.get("selected_hitRatio", 0))
            if curr_hit - prev_hit <= -30:
                sharp_drop_req_ids.append(f"{curr_req} ({prev_hit}%→{curr_hit}%)")

        rows.append(
            {
                "session": identity,
                "id_type": "session_id" if recs[0].get("tags", {}).get("session_id") else "trace_id",
                "req_count": len(hits),
                "first_hit": f"{hits[0]}%",
                "avg_hit(excl_first)": f"{avg_excl_first}%" if avg_excl_first != "-" else "-",
                "max_hit": f"{max(hits)}%",
                "min_hit": f"{min(hits)}%",
                "all_hits": ", ".join(f"{h}%" for h in hits),
                "sticky": "yes" if len(workers) <= 1 else "no",
                "unique_workers": len(workers),
                "prefill_urls": " | ".join(strip_scheme(u) for u in prefill_urls),
                "switch_req_pairs": " ; ".join(switch_events) if switch_events else "-",
                "sharp_drop_request_ids": " ; ".join(sharp_drop_req_ids) if sharp_drop_req_ids else "-",
            }
        )

    rows.sort(key=lambda r: (r["req_count"], r["session"]), reverse=True)
    return rows


def summarize_session_details(rows):
    """生成 session 级摘要指标。"""
    if not rows:
        return {
            "total_sessions": 0,
            "multi_req": 0,
            "single_req": 0,
            "sticky_multi": 0,
            "non_sticky_multi": 0,
            "non_first_avg": 0,
            "non_first_total": 0,
        }

    multi_req_rows = [r for r in rows if r["req_count"] > 1]
    sticky_multi = [r for r in multi_req_rows if r["sticky"] == "yes"]
    non_sticky_multi = [r for r in multi_req_rows if r["sticky"] == "no"]

    non_first_vals = []
    for r in rows:
        hit_tokens = [h.strip().rstrip("%") for h in r["all_hits"].split(",") if h.strip()]
        nums = [int(x) for x in hit_tokens if x.isdigit()]
        if len(nums) > 1:
            non_first_vals.extend(nums[1:])

    return {
        "total_sessions": len(rows),
        "multi_req": len(multi_req_rows),
        "single_req": len(rows) - len(multi_req_rows),
        "sticky_multi": len(sticky_multi),
        "non_sticky_multi": len(non_sticky_multi),
        "non_first_avg": round(sum(non_first_vals) / len(non_first_vals), 2) if non_first_vals else 0,
        "non_first_total": len(non_first_vals),
    }
