#!/usr/bin/env python3
"""generate_report.py — 从多场景 benchmark 结果生成多模式 HTML 报告

用法:
    python3 generate_report.py \
        --data-json all_metrics.json \
        --output benchmark_report.html \
        [--model-name "GLM-4.7-Flash"] \
        [--default-quant bf16] \
        [--default-bs 512]

或者从多个日志文件解析:
    python3 generate_report.py \
        --log-dir /path/to/logs \
        --model-name "GLM-4.7-Flash" \
        --output benchmark_report.html

数据 JSON 格式:
{
  "bf16_bs1": {"fd": {...metrics...}, "sg": {...metrics...}},
  "bf16_bs32": {"fd": {...}, "sg": {...}},
  ...
}
"""

import argparse
import json
import os
import re
import sys


def parse_benchmark_log(filepath):
    """解析 benchmark_serving.py 的日志文件，提取所有指标"""
    metrics = {}
    if not os.path.isfile(filepath):
        print(f"[WARN] 文件不存在: {filepath}", file=sys.stderr)
        return metrics

    with open(filepath, "r") as f:
        content = f.read()

    patterns = {
        "successful_requests": r"Successful requests:\s+([\d.]+)",
        "benchmark_duration": r"Benchmark duration \(s\):\s+([\d.]+)",
        "total_input_tokens": r"Total input tokens:\s+([\d.]+)",
        "total_generated_tokens": r"Total generated tokens:\s+([\d.]+)",
        "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "total_token_throughput": r"Total Token throughput \(tok/s\):\s+([\d.]+)",
        "mean_ttft": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "median_ttft": r"Median TTFT \(ms\):\s+([\d.]+)",
        "p80_ttft": r"P80 TTFT \(ms\):\s+([\d.]+)",
        "p95_ttft": r"P95 TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "median_tpot": r"Median TPOT \(ms\):\s+([\d.]+)",
        "p80_tpot": r"P80 TPOT \(ms\):\s+([\d.]+)",
        "p95_tpot": r"P95 TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot": r"P99 TPOT \(ms\):\s+([\d.]+)",
        "mean_itl": r"Mean ITL \(ms\):\s+([\d.]+)",
        "median_itl": r"Median ITL \(ms\):\s+([\d.]+)",
        "p80_itl": r"P80 ITL \(ms\):\s+([\d.]+)",
        "p95_itl": r"P95 ITL \(ms\):\s+([\d.]+)",
        "p99_itl": r"P99 ITL \(ms\):\s+([\d.]+)",
        "mean_e2el": r"Mean E2EL \(ms\):\s+([\d.]+)",
        "median_e2el": r"Median E2EL \(ms\):\s+([\d.]+)",
        "p80_e2el": r"P80 E2EL \(ms\):\s+([\d.]+)",
        "p95_e2el": r"P95 E2EL \(ms\):\s+([\d.]+)",
        "p99_e2el": r"P99 E2EL \(ms\):\s+([\d.]+)",
        "mean_decode": r"Mean Decode \(tok/s\):\s+([\d.]+)",
        "median_decode": r"Median Decode \(tok/s\):\s+([\d.]+)",
        "p80_decode": r"P80 Decode \(tok/s\):\s+([\d.]+)",
        "p95_decode": r"P95 Decode \(tok/s\):\s+([\d.]+)",
        "p99_decode": r"P99 Decode \(tok/s\):\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def scan_log_dir(log_dir):
    """扫描日志目录，自动识别场景并提取指标

    文件命名约定: *_bs<N>_[<quant>_]<fd|sg>.txt
    例如: GLM-4.7-Flash_long_bs32_fd.txt, GLM-4.7-Flash_long_bs512_fp8_sg.txt
    """
    data = {}
    if not os.path.isdir(log_dir):
        print(f"[ERROR] 日志目录不存在: {log_dir}", file=sys.stderr)
        return data

    for root, dirs, files in os.walk(log_dir):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            filepath = os.path.join(root, fname)

            # 尝试从文件名解析场景信息
            # 格式: *_bs<N>_[<quant>_]<fd|sg>.txt
            m = re.search(r"_bs(\d+)_(?:(fp8|bf16|wint4|wint8)_)?(fd|sg)\.txt$", fname, re.IGNORECASE)
            if not m:
                # 也尝试无 quant 的模式 (默认 bf16)
                m = re.search(r"_bs(\d+)_(fd|sg)\.txt$", fname, re.IGNORECASE)
                if m:
                    bs = m.group(1)
                    quant = "bf16"
                    framework = m.group(2).lower()
                else:
                    continue
            else:
                bs = m.group(1)
                quant = (m.group(2) or "bf16").lower()
                framework = m.group(3).lower()

            key = f"{quant}_bs{bs}"
            if key not in data:
                data[key] = {}

            metrics = parse_benchmark_log(filepath)
            if metrics:
                data[key][framework] = metrics
                print(f"[INFO] 解析成功: {fname} -> {key}/{framework} ({len(metrics)} metrics)")

    return data


def generate_html(benchmark_data, config):
    """生成完整的多模式 HTML 报告"""

    # 确定可用的量化方式和并发数
    quants = sorted(set(k.split("_bs")[0] for k in benchmark_data.keys()))
    bs_values = sorted(set(k.split("_bs")[1] for k in benchmark_data.keys()), key=int)

    model_name = config.get("model_name", "Unknown Model")
    default_quant = config.get("default_quant", quants[0] if quants else "bf16")
    default_bs = config.get("default_bs", bs_values[-1] if bs_values else "32")
    gpu_type = config.get("gpu_type", "H800")
    tp_size = config.get("tp_size", 1)
    dp_size = config.get("dp_size", 1)
    ep_size = config.get("ep_size", 0)
    fd_attention = config.get("fd_attention", "MLA_ATTN (FlashAttn v3)")
    sg_attention = config.get("sg_attention", "flashmla")
    sg_version = config.get("sg_version", "0.5.10.post1")
    fd_commit_date = config.get("fd_commit_date", "")
    fd_commit_short = config.get("fd_commit_short", "")
    fd_commit_full = config.get("fd_commit_full", "")
    max_model_len = config.get("max_model_len", 65536)
    dataset_url = config.get("dataset_url", "")
    dataset_desc = config.get("dataset_desc", "")
    test_date = config.get("test_date", "")
    model_type = config.get("model_type", "")
    model_size = config.get("model_size", "")
    model_experts = config.get("model_experts", "")
    model_layers_hidden = config.get("model_layers_hidden", "")

    # 生成量化选择器按钮
    def quant_btn_label(q):
        if q == "fp8":
            return "FP8 (Block-Wise)"
        return q.upper()

    quant_buttons = "\n".join(
        f'                <div class="seg-btn" data-val="{q}" onclick="setQuant(\'{q}\')" title="{"FD: block_wise_fp8 / SG: fp8" if q == "fp8" else ""}">{quant_btn_label(q)}</div>'
        for q in quants
    )

    # 生成并发选择器按钮
    bs_buttons = "\n".join(
        f'                <div class="seg-btn" data-val="{bs}" onclick="setBS(\'{bs}\')">{bs}</div>'
        for bs in bs_values
    )

    data_json = json.dumps(benchmark_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FastDeploy vs SGLang - {model_name} 性能对比报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --fd-primary: #6366f1;
            --fd-light: #818cf8;
            --fd-glow: rgba(99,102,241,0.3);
            --sg-primary: #f59e0b;
            --sg-light: #fbbf24;
            --sg-glow: rgba(245,158,11,0.3);
            --success: #10b981;
            --danger: #ef4444;
            --radius: 20px;
            --transition: 0.35s cubic-bezier(0.4,0,0.2,1);
        }}
        [data-theme="dark"] {{
            --bg-primary: #0a0a0f;
            --bg-card: rgba(15,15,25,0.85);
            --bg-card-hover: rgba(20,20,35,0.95);
            --bg-table-header: rgba(255,255,255,0.03);
            --border: rgba(255,255,255,0.06);
            --border-hover: rgba(255,255,255,0.14);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --chart-grid: rgba(255,255,255,0.04);
            --shadow-card: 0 8px 40px rgba(0,0,0,0.4);
            --gradient-bg: radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.08) 0%, transparent 60%),
                           radial-gradient(ellipse 60% 40% at 80% 90%, rgba(245,158,11,0.06) 0%, transparent 50%);
        }}
        [data-theme="light"] {{
            --bg-primary: #f8fafc;
            --bg-card: rgba(255,255,255,0.9);
            --bg-card-hover: rgba(255,255,255,1);
            --bg-table-header: rgba(0,0,0,0.02);
            --border: rgba(0,0,0,0.08);
            --border-hover: rgba(0,0,0,0.15);
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --chart-grid: rgba(0,0,0,0.06);
            --shadow-card: 0 4px 24px rgba(0,0,0,0.08);
            --gradient-bg: radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.04) 0%, transparent 60%),
                           radial-gradient(ellipse 60% 40% at 80% 90%, rgba(245,158,11,0.03) 0%, transparent 50%);
            --fd-light: #4f46e5;
            --sg-light: #d97706;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'PingFang SC', sans-serif;
            background: var(--bg-primary); color: var(--text-primary);
            min-height: 100vh; padding: 60px 24px; line-height: 1.6;
            transition: background var(--transition), color var(--transition);
        }}
        body::before {{
            content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: var(--gradient-bg); pointer-events: none; z-index: 0;
            transition: background var(--transition);
        }}
        .container {{ max-width: 1440px; margin: 0 auto; position: relative; z-index: 1; }}
        .theme-toggle {{
            position: fixed; top: 24px; right: 24px; z-index: 999;
            width: 52px; height: 28px; border-radius: 100px;
            background: var(--bg-card); border: 1px solid var(--border);
            cursor: pointer; display: flex; align-items: center; padding: 3px;
            transition: background var(--transition), border-color var(--transition);
            backdrop-filter: blur(12px);
        }}
        .theme-toggle:hover {{ border-color: var(--border-hover); }}
        .theme-toggle .toggle-knob {{
            width: 22px; height: 22px; border-radius: 50%;
            background: var(--fd-primary);
            transition: transform var(--transition), background var(--transition);
            display: flex; align-items: center; justify-content: center; font-size: 12px;
        }}
        [data-theme="dark"] .toggle-knob {{ transform: translateX(0); }}
        [data-theme="light"] .toggle-knob {{ transform: translateX(24px); background: var(--sg-primary); }}
        .toggle-knob::after {{ content: '\\1F319'; font-size: 11px; }}
        [data-theme="light"] .toggle-knob::after {{ content: '\\2600\\FE0F'; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .header h1 {{
            font-size: 3.2rem; font-weight: 800; letter-spacing: -0.03em;
            background: linear-gradient(135deg, var(--fd-light) 0%, #a78bfa 40%, var(--sg-light) 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px;
        }}
        .header .subtitle {{ font-size: 1.1rem; color: var(--text-secondary); font-weight: 400; }}
        .header .badge-row {{ display: flex; justify-content: center; gap: 12px; margin-top: 20px; flex-wrap: wrap; }}
        .badge {{
            padding: 6px 16px; border-radius: 100px; font-size: 0.78rem; font-weight: 600;
            letter-spacing: 0.3px; border: 1px solid var(--border); background: var(--bg-card);
            color: var(--text-secondary); transition: background var(--transition), border-color var(--transition), color var(--transition);
        }}
        .selector-bar {{
            display: flex; justify-content: center; align-items: center; gap: 24px;
            margin-bottom: 40px; flex-wrap: wrap;
        }}
        .selector-group {{ display: flex; align-items: center; gap: 10px; }}
        .selector-label {{ font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }}
        .seg-control {{
            display: flex; background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 12px; padding: 4px; gap: 2px;
            transition: background var(--transition), border-color var(--transition);
        }}
        .seg-btn {{
            padding: 8px 18px; border-radius: 9px; font-size: 0.82rem; font-weight: 600;
            cursor: pointer; border: none; background: transparent;
            color: var(--text-muted); transition: all 0.2s ease; user-select: none;
        }}
        .seg-btn:hover {{ color: var(--text-secondary); }}
        .seg-btn.active {{
            background: var(--fd-primary); color: #fff;
            box-shadow: 0 2px 8px rgba(99,102,241,0.3);
        }}
        .config-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 48px; }}
        @media (max-width: 1000px) {{ .config-grid {{ grid-template-columns: 1fr; }} }}
        .config-card {{
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 32px; backdrop-filter: blur(20px);
            transition: background var(--transition), border-color var(--transition), box-shadow var(--transition);
        }}
        .config-card:hover {{ border-color: var(--border-hover); box-shadow: var(--shadow-card); }}
        .config-card .card-tag {{
            display: inline-flex; align-items: center; gap: 8px;
            margin-bottom: 20px; padding: 6px 14px; border-radius: 10px;
            font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;
        }}
        .config-card.model .card-tag {{ background: rgba(168,85,247,0.12); color: #c084fc; }}
        .config-card.fd .card-tag {{ background: rgba(99,102,241,0.12); color: var(--fd-light); }}
        .config-card.sg .card-tag {{ background: rgba(245,158,11,0.12); color: var(--sg-light); }}
        .config-grid-inner {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        .config-item .label {{ font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }}
        .config-item .value {{ font-size: 0.95rem; color: var(--text-primary); font-weight: 600; }}
        .config-item.full {{ grid-column: 1 / -1; }}
        .params-bar {{
            display: flex; justify-content: center; flex-wrap: wrap; gap: 16px 32px;
            margin-bottom: 48px; padding: 20px 32px;
            background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px;
            transition: background var(--transition), border-color var(--transition);
        }}
        .param-item {{ display: flex; align-items: center; gap: 10px; }}
        .param-item .p-label {{ font-size: 0.72rem; color: var(--text-muted); letter-spacing: 0.5px; text-transform: uppercase; }}
        .param-item .p-value {{
            font-size: 0.88rem; color: var(--text-primary); font-weight: 600;
            padding: 4px 12px; background: var(--bg-table-header); border: 1px solid var(--border); border-radius: 8px;
        }}
        .param-item .p-value a {{ color: var(--text-primary); text-decoration: underline; text-underline-offset: 3px; }}
        .legend {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 40px; }}
        .legend-item {{ display: flex; align-items: center; gap: 10px; font-size: 0.9rem; font-weight: 500; color: var(--text-secondary); }}
        .legend-dot {{ width: 14px; height: 14px; border-radius: 4px; }}
        .legend-dot.fd {{ background: var(--fd-primary); box-shadow: 0 0 12px var(--fd-glow); }}
        .legend-dot.sg {{ background: var(--sg-primary); box-shadow: 0 0 12px var(--sg-glow); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 48px; }}
        .metric-card {{
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 28px; transition: transform 0.25s cubic-bezier(0.4,0,0.2,1), box-shadow 0.25s, border-color 0.25s, background var(--transition);
        }}
        .metric-card:hover {{ transform: translateY(-6px); border-color: var(--border-hover); box-shadow: var(--shadow-card); }}
        .metric-card .m-title {{ font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 20px; font-weight: 600; }}
        .metric-card .m-values {{ display: flex; justify-content: space-between; align-items: center; }}
        .metric-card .m-fw {{ text-align: center; }}
        .metric-card .m-fw .m-name {{ font-size: 0.68rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 500; }}
        .metric-card .m-fw .m-num {{ font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; }}
        .fd-c {{ color: var(--fd-light); }}
        .sg-c {{ color: var(--sg-light); }}
        .metric-card .m-diff {{ font-size: 0.75rem; padding: 5px 12px; border-radius: 100px; font-weight: 700; letter-spacing: 0.3px; }}
        .m-better {{ background: rgba(16,185,129,0.12); color: var(--success); border: 1px solid rgba(16,185,129,0.2); }}
        .m-worse {{ background: rgba(239,68,68,0.12); color: var(--danger); border: 1px solid rgba(239,68,68,0.2); }}
        .metric-card .m-hint {{ text-align: center; margin-top: 12px; font-size: 0.7rem; color: var(--text-muted); }}
        .charts-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 48px; }}
        @media (max-width: 1000px) {{ .charts-row {{ grid-template-columns: 1fr 1fr; }} }}
        @media (max-width: 700px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}
        .chart-card {{
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px;
            transition: background var(--transition), border-color var(--transition);
        }}
        .chart-card h3 {{ font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 20px; font-weight: 600; }}
        .table-wrap {{
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 36px; overflow-x: auto; margin-bottom: 40px;
            transition: background var(--transition), border-color var(--transition);
        }}
        .table-wrap h2 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 24px; color: var(--text-primary); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ padding: 14px 16px; text-align: center; font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; border-bottom: 1px solid var(--border); background: var(--bg-table-header); }}
        td {{ padding: 14px 16px; text-align: center; font-size: 0.92rem; border-bottom: 1px solid var(--border); color: var(--text-secondary); transition: background var(--transition); }}
        tr:hover td {{ background: var(--bg-table-header); }}
        td.winner-fd {{ color: var(--fd-light); font-weight: 700; }}
        td.winner-sg {{ color: var(--sg-light); font-weight: 700; }}
        td.diff-good {{ color: var(--success); font-weight: 600; }}
        td.diff-bad {{ color: var(--danger); font-weight: 600; }}
        td.win-label-fd {{ color: var(--fd-light); font-weight: 600; font-size: 0.8rem; }}
        td.win-label-sg {{ color: var(--sg-light); font-weight: 600; font-size: 0.8rem; }}
        .conclusion {{
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 40px; margin-bottom: 40px;
            transition: background var(--transition), border-color var(--transition);
        }}
        .conclusion h2 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 20px; }}
        .conclusion p {{ color: var(--text-secondary); margin-bottom: 14px; font-size: 0.95rem; }}
        .conclusion strong {{ color: var(--text-primary); }}
        .highlight-fd {{ color: var(--fd-light); font-weight: 600; }}
        .highlight-sg {{ color: var(--sg-light); font-weight: 600; }}
        .footer {{ text-align: center; padding: 24px; color: var(--text-muted); font-size: 0.8rem; border-top: 1px solid var(--border); }}
    </style>
</head>
<body>
<div class="theme-toggle" onclick="toggleTheme()" title="切换明暗模式">
    <div class="toggle-knob"></div>
</div>
<div class="container">
    <div class="header">
        <h1>FastDeploy vs SGLang</h1>
        <p class="subtitle">{model_name} 推理性能基准测试报告</p>
        <div class="badge-row">
            <span class="badge">{gpu_type} x{tp_size * dp_size}</span>
            <span class="badge">TP={tp_size}{f' DP={dp_size}' if dp_size > 1 else ''}{f' EP={ep_size}' if ep_size > 0 else ''}</span>
            <span class="badge" id="badge-quant">{default_quant.upper()}</span>
            <span class="badge" id="badge-bs">并发 {default_bs}</span>
            {f'<span class="badge">{test_date}</span>' if test_date else ''}
        </div>
    </div>

    <!-- Mode Selector -->
    <div class="selector-bar">
        <div class="selector-group">
            <span class="selector-label">量化</span>
            <div class="seg-control" id="quant-selector">
{quant_buttons}
            </div>
        </div>
        <div class="selector-group">
            <span class="selector-label">并发</span>
            <div class="seg-control" id="bs-selector">
{bs_buttons}
            </div>
        </div>
    </div>

    <!-- Config Cards -->
    <div class="config-grid">
        <div class="config-card model">
            <div class="card-tag">Model</div>
            <div class="config-grid-inner">
                <div class="config-item full"><div class="label">模型名称</div><div class="value">{model_name}</div></div>
                {f'<div class="config-item"><div class="label">架构</div><div class="value">{model_type}</div></div>' if model_type else ''}
                {f'<div class="config-item"><div class="label">权重大小</div><div class="value">{model_size}</div></div>' if model_size else ''}
                {f'<div class="config-item"><div class="label">Expert</div><div class="value">{model_experts}</div></div>' if model_experts else ''}
                {f'<div class="config-item"><div class="label">Layers / Hidden</div><div class="value">{model_layers_hidden}</div></div>' if model_layers_hidden else ''}
            </div>
        </div>
        <div class="config-card fd">
            <div class="card-tag">FastDeploy</div>
            <div class="config-grid-inner">
                <div class="config-item"><div class="label">GPU</div><div class="value">{gpu_type} x{tp_size * dp_size}</div></div>
                <div class="config-item"><div class="label">部署方式</div><div class="value">TP={tp_size}{f' DP={dp_size}' if dp_size > 1 else ''}{' EP' if ep_size > 0 else ''}</div></div>
                <div class="config-item"><div class="label">并发</div><div class="value" id="cfg-fd-bs">{default_bs}</div></div>
                <div class="config-item"><div class="label">Max Len</div><div class="value">{max_model_len}</div></div>
                <div class="config-item"><div class="label">Attention</div><div class="value">{fd_attention}</div></div>
                <div class="config-item"><div class="label">量化</div><div class="value" id="cfg-fd-quant">{default_quant.upper()}</div></div>
                {f'<div class="config-item"><div class="label">代码日期</div><div class="value">{fd_commit_date}</div></div>' if fd_commit_date else ''}
                {f'<div class="config-item"><div class="label">Commit</div><div class="value" title="{fd_commit_full}">{fd_commit_short}</div></div>' if fd_commit_short else ''}
            </div>
        </div>
        <div class="config-card sg">
            <div class="card-tag">SGLang</div>
            <div class="config-grid-inner">
                <div class="config-item"><div class="label">GPU</div><div class="value">{gpu_type} x{tp_size * dp_size}</div></div>
                <div class="config-item"><div class="label">部署方式</div><div class="value">TP={tp_size}{f' DP={dp_size}' if dp_size > 1 else ''}{f' EP={ep_size}' if ep_size > 0 else ''}</div></div>
                <div class="config-item"><div class="label">并发</div><div class="value" id="cfg-sg-bs">{default_bs}</div></div>
                <div class="config-item"><div class="label">Context Len</div><div class="value">{max_model_len}</div></div>
                <div class="config-item"><div class="label">Attention</div><div class="value">{sg_attention}</div></div>
                <div class="config-item"><div class="label">量化</div><div class="value" id="cfg-sg-quant">{default_quant.upper()}</div></div>
                {f'<div class="config-item"><div class="label">版本</div><div class="value">{sg_version}</div></div>' if sg_version else ''}
                <div class="config-item"></div>
            </div>
        </div>
    </div>

    <!-- Params Bar -->
    <div class="params-bar">
        <div class="param-item"><span class="p-label">请求数</span><span class="p-value" id="param-reqs">1024</span></div>
        {f'<div class="param-item"><span class="p-label">数据集</span><span class="p-value"><a href="{dataset_url}" target="_blank">{dataset_desc}（点击可下载）</a></span></div>' if dataset_url else ''}
    </div>

    <!-- Legend -->
    <div class="legend">
        <div class="legend-item"><div class="legend-dot fd"></div><span>FastDeploy</span></div>
        <div class="legend-item"><div class="legend-dot sg"></div><span>SGLang</span></div>
    </div>

    <!-- Metric Cards -->
    <div class="metrics-grid" id="metrics-grid"></div>

    <div class="charts-row">
        <div class="chart-card"><h3>吞吐量对比 (tok/s)</h3><canvas id="throughputChart"></canvas></div>
        <div class="chart-card"><h3>TTFT (ms)</h3><canvas id="ttftChart"></canvas></div>
        <div class="chart-card"><h3>TPOT & ITL (ms)</h3><canvas id="tpotItlChart"></canvas></div>
        <div class="chart-card"><h3>E2EL (ms)</h3><canvas id="e2elChart"></canvas></div>
    </div>

    <!-- Table -->
    <div class="table-wrap">
        <h2>详细指标对比</h2>
        <table>
            <thead><tr><th style="text-align:left">指标</th><th>FastDeploy</th><th>SGLang</th><th>FD 优势</th><th>胜出</th></tr></thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>

    <!-- Conclusion -->
    <div class="conclusion" id="conclusion-section"></div>

    <!-- Quantization Note (only shown when fp8 is available) -->
    {'<div class="config-card" style="margin-bottom:40px; padding:28px 36px;" id="quant-note"><div class="card-tag" style="background:rgba(245,158,11,0.12);color:var(--sg-light);margin-bottom:16px;">&#9888; 量化说明</div><div style="color:var(--text-secondary);font-size:0.9rem;line-height:1.8;"><p style="margin-bottom:10px;"><strong style="color:var(--text-primary);">本报告中 FP8 量化指的是 Block-Wise FP8（分块量化）</strong>，不同于 Per-Tensor FP8 或 Per-Channel FP8。</p><ul style="padding-left:20px;"><li><strong>FastDeploy</strong>：使用 <code style="background:var(--bg-table-header);padding:2px 6px;border-radius:4px;font-size:0.85em;">--quantization block_wise_fp8</code>，对权重按 block 粒度进行 FP8 量化</li><li><strong>SGLang</strong>：使用 <code style="background:var(--bg-table-header);padding:2px 6px;border-radius:4px;font-size:0.85em;">--quantization fp8</code>，默认为 per-tensor FP8 量化方式</li></ul><p style="margin-top:10px;color:var(--text-muted);font-size:0.82rem;">两者量化粒度不完全相同，FD 的 block-wise 粒度更细、精度损失更小，但计算开销略高。对比结果需结合量化方式差异理解。</p></div></div>' if 'fp8' in quants else ''}

    <div class="footer">Generated by Ducc Benchmark Skill &middot; {model_name} &middot; {gpu_type}</div>
</div>

<script>
const benchmarkData = {data_json};
const availableQuants = {json.dumps(quants)};
const availableBS = {json.dumps(bs_values)};
let currentQuant = localStorage.getItem('bench-quant') || '{default_quant}';
let currentBS = localStorage.getItem('bench-bs') || '{default_bs}';
if (!availableQuants.includes(currentQuant)) currentQuant = availableQuants[0];
if (!availableBS.includes(currentBS)) currentBS = availableBS[availableBS.length - 1];

function getKey() {{ return currentQuant + '_bs' + currentBS; }}
function getData() {{ return benchmarkData[getKey()] || null; }}

function setQuant(q) {{
    currentQuant = q;
    localStorage.setItem('bench-quant', q);
    updateAll();
}}
function setBS(bs) {{
    currentBS = bs;
    localStorage.setItem('bench-bs', bs);
    updateAll();
}}

function updateSelectors() {{
    document.querySelectorAll('#quant-selector .seg-btn').forEach(btn => {{
        btn.classList.remove('active');
        if (btn.dataset.val === currentQuant) btn.classList.add('active');
    }});
    document.querySelectorAll('#bs-selector .seg-btn').forEach(btn => {{
        btn.classList.remove('active');
        if (btn.dataset.val === currentBS) btn.classList.add('active');
    }});
    const quantLabel = currentQuant === 'bf16' ? 'BF16' : currentQuant.toUpperCase();
    document.getElementById('badge-quant').textContent = quantLabel;
    document.getElementById('badge-bs').textContent = '\\u5e76\\u53d1 ' + currentBS;
    document.getElementById('cfg-fd-bs').textContent = currentBS;
    document.getElementById('cfg-sg-bs').textContent = currentBS;
    document.getElementById('cfg-fd-quant').textContent = currentQuant === 'bf16' ? 'BF16' : 'Block-Wise FP8 (block_wise_fp8)';
    document.getElementById('cfg-sg-quant').textContent = currentQuant === 'bf16' ? 'BF16' : 'FP8 (per-tensor)';
    const d = getData();
    if (d && d.fd) {{
        document.getElementById('param-reqs').textContent = Math.round(d.fd.successful_requests || 1024);
    }}
}}

const metricDefs = [
    {{ key: 'total_token_throughput', title: 'Total Token Throughput', unit: 'tok/s', hint: '越高越好', higher: true }},
    {{ key: 'output_token_throughput', title: 'Output Token Throughput', unit: 'tok/s', hint: '越高越好', higher: true }},
    {{ key: 'mean_ttft', title: 'Mean TTFT (首 Token 延迟)', unit: 'ms', hint: '越低越好', higher: false }},
    {{ key: 'mean_tpot', title: 'Mean TPOT (Token 间延迟)', unit: 'ms', hint: '越低越好', higher: false }},
    {{ key: 'mean_itl', title: 'Mean ITL (Inter-Token Latency)', unit: 'ms', hint: '越低越好', higher: false }},
    {{ key: 'mean_e2el', title: 'Mean E2EL (端到端延迟)', unit: 'ms', hint: '越低越好', higher: false }},
    {{ key: 'mean_decode', title: 'Decode Speed', unit: 'tok/s', hint: '越高越好', higher: true }},
    {{ key: 'request_throughput', title: 'Request Throughput', unit: 'req/s', hint: '越高越好', higher: true }},
];

function fmtNum(v) {{
    if (v >= 10000) return Math.round(v).toLocaleString();
    if (v >= 100) return Math.round(v).toString();
    if (v >= 10) return v.toFixed(1);
    return v.toFixed(2);
}}

function updateMetricCards() {{
    const d = getData();
    if (!d) return;
    const grid = document.getElementById('metrics-grid');
    let html = '';
    metricDefs.forEach(def => {{
        const fdVal = d.fd[def.key] || 0;
        const sgVal = d.sg[def.key] || 0;
        let diffPct;
        if (def.higher) {{
            diffPct = sgVal !== 0 ? ((fdVal - sgVal) / sgVal * 100) : 0;
        }} else {{
            diffPct = fdVal !== 0 ? ((sgVal - fdVal) / fdVal * 100) : 0;
        }}
        const isBetter = diffPct > 0;
        const diffClass = isBetter ? 'm-better' : 'm-worse';
        const diffText = (isBetter ? '+' : '') + diffPct.toFixed(1) + '%';
        html += '<div class="metric-card">' +
            '<div class="m-title">' + def.title + '</div>' +
            '<div class="m-values">' +
            '<div class="m-fw"><div class="m-name">FastDeploy</div><div class="m-num fd-c">' + fmtNum(fdVal) + '</div></div>' +
            '<div class="m-diff ' + diffClass + '">' + diffText + '</div>' +
            '<div class="m-fw"><div class="m-name">SGLang</div><div class="m-num sg-c">' + fmtNum(sgVal) + '</div></div>' +
            '</div>' +
            '<div class="m-hint">' + def.unit + ' \\u00b7 ' + def.hint + '</div></div>';
    }});
    grid.innerHTML = html;
}}

const tableMetrics = [
    {{ key: 'successful_requests', label: '成功请求数', higher: true }},
    {{ key: 'benchmark_duration', label: '测试总时长 (s)', higher: false }},
    {{ key: 'total_token_throughput', label: '总 Token Throughput (tok/s)', higher: true }},
    {{ key: 'output_token_throughput', label: '输出 Token Throughput (tok/s)', higher: true }},
    {{ key: 'request_throughput', label: 'Request Throughput (req/s)', higher: true }},
    {{ key: 'mean_decode', label: 'Mean Decode Speed (tok/s)', higher: true }},
    {{ key: 'mean_ttft', label: 'Mean TTFT (ms)', higher: false }},
    {{ key: 'p80_ttft', label: 'P80 TTFT (ms)', higher: false }},
    {{ key: 'p95_ttft', label: 'P95 TTFT (ms)', higher: false }},
    {{ key: 'p99_ttft', label: 'P99 TTFT (ms)', higher: false }},
    {{ key: 'mean_tpot', label: 'Mean TPOT (ms)', higher: false }},
    {{ key: 'p80_tpot', label: 'P80 TPOT (ms)', higher: false }},
    {{ key: 'p95_tpot', label: 'P95 TPOT (ms)', higher: false }},
    {{ key: 'p99_tpot', label: 'P99 TPOT (ms)', higher: false }},
    {{ key: 'mean_itl', label: 'Mean ITL (ms)', higher: false }},
    {{ key: 'p80_itl', label: 'P80 ITL (ms)', higher: false }},
    {{ key: 'p95_itl', label: 'P95 ITL (ms)', higher: false }},
    {{ key: 'p99_itl', label: 'P99 ITL (ms)', higher: false }},
    {{ key: 'mean_e2el', label: 'Mean E2EL (ms)', higher: false }},
    {{ key: 'p80_e2el', label: 'P80 E2EL (ms)', higher: false }},
    {{ key: 'p95_e2el', label: 'P95 E2EL (ms)', higher: false }},
    {{ key: 'p99_e2el', label: 'P99 E2EL (ms)', higher: false }},
];

function updateTable() {{
    const d = getData();
    if (!d) return;
    const tbody = document.getElementById('table-body');
    let html = '';
    tableMetrics.forEach(m => {{
        const fdVal = d.fd[m.key];
        const sgVal = d.sg[m.key];
        if (fdVal == null || sgVal == null) return;
        let fdWins;
        if (m.higher) {{ fdWins = fdVal > sgVal; }}
        else {{ fdWins = fdVal < sgVal; }}
        const fdClass = fdWins ? 'winner-fd' : '';
        const sgClass = !fdWins ? 'winner-sg' : '';
        let diffPct;
        if (m.higher) {{
            diffPct = sgVal !== 0 ? ((fdVal - sgVal) / sgVal * 100) : 0;
        }} else {{
            diffPct = fdVal !== 0 ? ((sgVal - fdVal) / fdVal * 100) : 0;
        }}
        const diffClass = diffPct >= 0 ? 'diff-good' : 'diff-bad';
        const diffText = (diffPct >= 0 ? '+' : '') + diffPct.toFixed(1) + '%';
        const winner = fdWins ? 'FastDeploy' : 'SGLang';
        const winClass = fdWins ? 'win-label-fd' : 'win-label-sg';
        html += '<tr><td style="text-align:left">' + m.label + '</td>' +
            '<td class="' + fdClass + '">' + fmtNum(fdVal) + '</td>' +
            '<td class="' + sgClass + '">' + fmtNum(sgVal) + '</td>' +
            '<td class="' + diffClass + '">' + diffText + '</td>' +
            '<td class="' + winClass + '">' + winner + '</td></tr>';
    }});
    tbody.innerHTML = html;
}}

function updateConclusion() {{
    const d = getData();
    if (!d) return;
    const fdTPS = d.fd.total_token_throughput || 0;
    const sgTPS = d.sg.total_token_throughput || 0;
    const fdWinsTPS = fdTPS > sgTPS;
    const quantLabel = currentQuant === 'bf16' ? 'BF16 无量化' : 'Block-Wise FP8 在线量化 (FD: block_wise_fp8, SG: fp8)';
    const section = document.getElementById('conclusion-section');
    let wins = {{ fd: 0, sg: 0 }};
    tableMetrics.forEach(m => {{
        const fv = d.fd[m.key], sv = d.sg[m.key];
        if (fv == null || sv == null) return;
        if (m.higher) {{ fv > sv ? wins.fd++ : wins.sg++; }}
        else {{ fv < sv ? wins.fd++ : wins.sg++; }}
    }});
    const overallWinner = wins.fd > wins.sg ? 'FastDeploy' : 'SGLang';
    const winnerClass = wins.fd > wins.sg ? 'highlight-fd' : 'highlight-sg';
    section.innerHTML = '<h2>结论与分析</h2>' +
        '<p><strong>测试条件：</strong>{model_name}，{gpu_type} x{tp_size}，' + quantLabel + '，并发 ' + currentBS + '</p>' +
        '<p><strong>整体表现：</strong>在该配置下，<span class="' + winnerClass + '">' + overallWinner +
        ' 在 ' + Math.max(wins.fd, wins.sg) + '/' + (wins.fd + wins.sg) + ' 项指标上领先</span>。' +
        'Total Token Throughput: FD=' + fmtNum(fdTPS) + ' vs SG=' + fmtNum(sgTPS) + ' tok/s。</p>';
}}

let charts = {{}};
function getChartColors() {{
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {{
        fdColor: isDark ? 'rgba(129,140,248,0.85)' : 'rgba(79,70,229,0.8)',
        sgColor: isDark ? 'rgba(251,191,36,0.85)' : 'rgba(217,119,6,0.8)',
        fdBorder: isDark ? '#818cf8' : '#4f46e5',
        sgBorder: isDark ? '#fbbf24' : '#d97706',
        fdFill: isDark ? 'rgba(99,102,241,0.12)' : 'rgba(79,70,229,0.08)',
        sgFill: isDark ? 'rgba(245,158,11,0.12)' : 'rgba(217,119,6,0.08)',
        gridColor: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)',
        textColor: isDark ? '#94a3b8' : '#475569',
    }};
}}

function rebuildCharts() {{
    Object.values(charts).forEach(c => c.destroy());
    charts = {{}};
    buildCharts();
}}

function buildCharts() {{
    const d = getData();
    if (!d) return;
    const c = getChartColors();
    Chart.defaults.color = c.textColor;
    Chart.defaults.borderColor = c.gridColor;
    Chart.defaults.font.family = "'Inter', sans-serif";
    const barOpts = {{ responsive: true, plugins: {{ legend: {{ labels: {{ usePointStyle: true, padding: 20 }} }} }}, scales: {{ y: {{ beginAtZero: true, grid: {{ color: c.gridColor }} }}, x: {{ grid: {{ display: false }} }} }} }};

    const reqScale = d.fd.request_throughput < 50 ? 100 : 1;
    charts.throughput = new Chart(document.getElementById('throughputChart'), {{
        type: 'bar',
        data: {{
            labels: ['Total Token', 'Output Token', 'Request (x' + reqScale + ')'],
            datasets: [
                {{ label: 'FastDeploy', data: [d.fd.total_token_throughput, d.fd.output_token_throughput, d.fd.request_throughput * reqScale], backgroundColor: c.fdColor, borderColor: c.fdBorder, borderWidth: 2, borderRadius: 6 }},
                {{ label: 'SGLang', data: [d.sg.total_token_throughput, d.sg.output_token_throughput, d.sg.request_throughput * reqScale], backgroundColor: c.sgColor, borderColor: c.sgBorder, borderWidth: 2, borderRadius: 6 }}
            ]
        }},
        options: barOpts
    }});

    charts.ttft = new Chart(document.getElementById('ttftChart'), {{
        type: 'bar',
        data: {{
            labels: ['Mean TTFT'],
            datasets: [
                {{ label: 'FastDeploy', data: [d.fd.mean_ttft], backgroundColor: c.fdColor, borderColor: c.fdBorder, borderWidth: 2, borderRadius: 6 }},
                {{ label: 'SGLang', data: [d.sg.mean_ttft], backgroundColor: c.sgColor, borderColor: c.sgBorder, borderWidth: 2, borderRadius: 6 }}
            ]
        }},
        options: barOpts
    }});

    charts.tpotItl = new Chart(document.getElementById('tpotItlChart'), {{
        type: 'bar',
        data: {{
            labels: ['Mean TPOT', 'Mean ITL'],
            datasets: [
                {{ label: 'FastDeploy', data: [d.fd.mean_tpot, d.fd.mean_itl], backgroundColor: c.fdColor, borderColor: c.fdBorder, borderWidth: 2, borderRadius: 6 }},
                {{ label: 'SGLang', data: [d.sg.mean_tpot, d.sg.mean_itl], backgroundColor: c.sgColor, borderColor: c.sgBorder, borderWidth: 2, borderRadius: 6 }}
            ]
        }},
        options: barOpts
    }});

    charts.e2el = new Chart(document.getElementById('e2elChart'), {{
        type: 'bar',
        data: {{
            labels: ['Mean E2EL'],
            datasets: [
                {{ label: 'FastDeploy', data: [d.fd.mean_e2el], backgroundColor: c.fdColor, borderColor: c.fdBorder, borderWidth: 2, borderRadius: 6 }},
                {{ label: 'SGLang', data: [d.sg.mean_e2el], backgroundColor: c.sgColor, borderColor: c.sgBorder, borderWidth: 2, borderRadius: 6 }}
            ]
        }},
        options: barOpts
    }});
}}

function toggleTheme() {{
    const html = document.documentElement;
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('benchmark-theme', next);
    rebuildCharts();
}}
(function() {{
    const saved = localStorage.getItem('benchmark-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
}})();

function updateAll() {{
    updateSelectors();
    updateMetricCards();
    updateTable();
    updateConclusion();
    rebuildCharts();
}}

updateAll();
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="从 benchmark 数据生成多模式 HTML 报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从已有 JSON 生成
  python3 generate_report.py --data-json all_metrics.json --output report.html

  # 从日志目录扫描生成
  python3 generate_report.py --log-dir /path/to/logs --model-name GLM-4.7-Flash --output report.html

  # 指定完整配置
  python3 generate_report.py --data-json data.json --output report.html \\
    --model-name GLM-4.7-Flash --gpu-type H800 --tp 1 \\
    --default-quant bf16 --default-bs 512 \\
    --fd-attention "MLA_ATTN (FlashAttn v3)" --sg-attention flashmla \\
    --sg-version 0.5.10.post1
        """,
    )

    # 数据来源（二选一）
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-json", help="已整理好的多场景 JSON 文件路径")
    source.add_argument("--log-dir", help="日志目录路径（自动扫描识别场景）")

    # 输出
    parser.add_argument("--output", default="benchmark_report.html", help="输出 HTML 路径")

    # 模型信息
    parser.add_argument("--model-name", default="Unknown Model", help="模型名称")
    parser.add_argument("--model-type", default="", help="模型架构描述")
    parser.add_argument("--model-size", default="", help="模型大小")
    parser.add_argument("--model-experts", default="", help="Expert 信息")
    parser.add_argument("--model-layers-hidden", default="", help="Layers / Hidden")

    # 部署配置
    parser.add_argument("--gpu-type", default="H800", help="GPU 型号")
    parser.add_argument("--tp", type=int, default=1, help="TP 大小")
    parser.add_argument("--dp", type=int, default=1, help="DP 大小")
    parser.add_argument("--ep", type=int, default=0, help="EP 大小 (0=不启用)")
    parser.add_argument("--max-model-len", type=int, default=65536, help="最大模型长度")
    parser.add_argument("--fd-attention", default="MLA_ATTN (FlashAttn v3)", help="FD Attention Backend")
    parser.add_argument("--sg-attention", default="flashmla", help="SG Attention Backend")
    parser.add_argument("--sg-version", default="", help="SGLang 版本")
    parser.add_argument("--fd-commit-date", default="", help="FD commit 日期")
    parser.add_argument("--fd-commit-short", default="", help="FD commit 短 hash")
    parser.add_argument("--fd-commit-full", default="", help="FD commit 完整 hash")

    # 显示配置
    parser.add_argument("--default-quant", default="bf16", help="默认量化选择")
    parser.add_argument("--default-bs", default="512", help="默认并发选择")
    parser.add_argument("--test-date", default="", help="测试日期")
    parser.add_argument("--dataset-url", default="", help="数据集下载链接")
    parser.add_argument("--dataset-desc", default="", help="数据集描述")

    args = parser.parse_args()

    # 加载数据
    if args.data_json:
        print(f"[INFO] 从 JSON 文件加载数据: {args.data_json}")
        with open(args.data_json, "r") as f:
            benchmark_data = json.load(f)
    else:
        print(f"[INFO] 扫描日志目录: {args.log_dir}")
        benchmark_data = scan_log_dir(args.log_dir)

    if not benchmark_data:
        print("[ERROR] 未找到有效的 benchmark 数据", file=sys.stderr)
        sys.exit(1)

    # 过滤掉不完整的场景（缺少 fd 或 sg）
    valid_data = {}
    for key, val in benchmark_data.items():
        if "fd" in val and "sg" in val and val["fd"] and val["sg"]:
            valid_data[key] = val
        else:
            print(f"[WARN] 场景 {key} 数据不完整，跳过", file=sys.stderr)

    if not valid_data:
        print("[ERROR] 没有完整的对比场景数据", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 有效场景: {', '.join(sorted(valid_data.keys()))}")

    # 构建配置
    config = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "model_size": args.model_size,
        "model_experts": args.model_experts,
        "model_layers_hidden": args.model_layers_hidden,
        "gpu_type": args.gpu_type,
        "tp_size": args.tp,
        "dp_size": args.dp,
        "ep_size": args.ep,
        "max_model_len": args.max_model_len,
        "fd_attention": args.fd_attention,
        "sg_attention": args.sg_attention,
        "sg_version": args.sg_version,
        "fd_commit_date": args.fd_commit_date,
        "fd_commit_short": args.fd_commit_short,
        "fd_commit_full": args.fd_commit_full,
        "default_quant": args.default_quant,
        "default_bs": args.default_bs,
        "test_date": args.test_date,
        "dataset_url": args.dataset_url,
        "dataset_desc": args.dataset_desc,
    }

    # 生成 HTML
    html = generate_html(valid_data, config)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] HTML 报告已生成: {args.output}")
    print(f"[INFO] 包含 {len(valid_data)} 个场景: {', '.join(sorted(valid_data.keys()))}")


if __name__ == "__main__":
    main()
