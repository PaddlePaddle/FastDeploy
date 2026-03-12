import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests

# ========= 服务配置 =========
DEFAULT_GPU_URL = "http://10.174.137.88:8188/v1/chat/completions"
DEFAULT_XPU_URL = "http://0.0.0.0:8188/v1/chat/completions"

HEADERS = {"Content-Type": "application/json"}

DEFAULT_TIMEOUT = 30
DEFAULT_LOG_FILE = "gpu_xpu_format_test.log"

# ========= 默认推理参数（可覆盖） =========
DEFAULT_GEN_PARAMS = {"logprobs": True, "top_logprobs": 0, "max_tokens": 5}


# ========= HTTP =========
def send_request(url: str, payload: dict, timeout: int) -> dict:
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ========= JSON 结构抽取 =========
def extract_structure(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: extract_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [] if not obj else [extract_structure(obj[0])]
    else:
        return type(obj).__name__


# ========= 结构对比 =========
def compare_structure(gpu: Any, xpu: Any, path: str = "") -> List[str]:
    diffs = []

    if type(gpu) is not type(xpu):
        diffs.append(f"{path or '$'}: type mismatch ({type(gpu).__name__} vs {type(xpu).__name__})")
        return diffs

    if isinstance(gpu, dict):
        gpu_keys = set(gpu.keys())
        xpu_keys = set(xpu.keys())

        for k in gpu_keys - xpu_keys:
            diffs.append(f"{path}.{k}: missing in XPU")
        for k in xpu_keys - gpu_keys:
            diffs.append(f"{path}.{k}: extra in XPU")

        for k in gpu_keys & xpu_keys:
            diffs.extend(compare_structure(gpu[k], xpu[k], path=f"{path}.{k}" if path else k))

    elif isinstance(gpu, list):
        if gpu and xpu:
            diffs.extend(compare_structure(gpu[0], xpu[0], path=f"{path}[0]"))

    else:
        if gpu != xpu:
            diffs.append(f"{path}: value type mismatch ({gpu} vs {xpu})")

    return diffs


# ========= 日志 =========
def write_log(record: dict, log_file: str):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=2))
        f.write("\n" + "=" * 80 + "\n")


# ========= 单条测试 =========
def run_test(
    messages: List[Dict],
    xpu_url: str,
    timeout: int,
    log_file: str,
    mode: str,
    gpu_url: Optional[str] = None,
    gen_params: Optional[Dict] = None,
):
    # 合并推理参数
    params = DEFAULT_GEN_PARAMS.copy()
    if gen_params:
        params.update(gen_params)

    payload = {"messages": messages, **params}

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        xpu_resp = send_request(xpu_url, payload, timeout)
    except Exception as e:
        xpu_resp = {"__error__": str(e)}

    gpu_resp = None
    diffs = []
    format_match = "__error__" not in xpu_resp

    if mode == "gpu-xpu":
        if gpu_url is None:
            raise ValueError("gpu_url must be set when mode=gpu-xpu")
        try:
            gpu_resp = send_request(gpu_url, payload, timeout)
        except Exception as e:
            gpu_resp = {"__error__": str(e)}
        gpu_struct = extract_structure(gpu_resp)
        xpu_struct = extract_structure(xpu_resp)
        diffs = compare_structure(gpu_struct, xpu_struct)
        format_match = len(diffs) == 0

    record = {
        "timestamp": ts,
        "mode": mode,
        "request": payload,
        "xpu_response": xpu_resp,
        "format_match": format_match,
        "format_diffs": diffs,
    }
    if gpu_resp is not None:
        record["gpu_response"] = gpu_resp

    write_log(record, log_file)

    print(f"[{ts}] format_match={record['format_match']}")
    if diffs:
        for d in diffs:
            print("  -", d)


def parse_args():
    parser = argparse.ArgumentParser(description="Logprobs format tester")
    parser.add_argument(
        "--mode",
        choices=["xpu-only", "gpu-xpu"],
        default="xpu-only",
        help="xpu-only: only test XPU endpoint; gpu-xpu: compare GPU and XPU response structure.",
    )
    parser.add_argument("--xpu-url", default=DEFAULT_XPU_URL, help="XPU chat completions URL")
    parser.add_argument("--gpu-url", default=DEFAULT_GPU_URL, help="GPU chat completions URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="Path to output log file")
    return parser.parse_args()


# ========= 入口 =========
if __name__ == "__main__":
    args = parse_args()
    test_cases = [
        {"messages": [{"role": "user", "content": "你叫什么？"}]},
        {
            "messages": [
                {"role": "system", "content": "你是一个有帮助的助手"},
                {"role": "user", "content": "给我讲一个笑话"},
            ],
            "gen_params": {"max_tokens": 32},
        },
        {
            "messages": [
                {"role": "user", "content": "先自我介绍"},
                {"role": "assistant", "content": "我是一个AI"},
                {"role": "user", "content": "你能做什么？"},
            ],
            "gen_params": {"logprobs": False},
        },
        {
            "messages": [
                {"role": "user", "content": "先自我介绍"},
                {"role": "assistant", "content": "我是一个AI"},
                {"role": "user", "content": "你能做什么？"},
            ],
            "gen_params": {"top_logprobs": 1},
        },
        {
            "messages": [
                {"role": "user", "content": "先自我介绍"},
                {"role": "assistant", "content": "我是一个AI"},
                {"role": "user", "content": "你能做什么？"},
            ],
            "gen_params": {"top_logprobs": 2},
        },
        {
            "messages": [
                {"role": "user", "content": "先自我介绍"},
                {"role": "assistant", "content": "我是一个AI"},
                {"role": "user", "content": "你能做什么？"},
            ],
            "gen_params": {"top_logprobs": 20},
        },
        {
            "messages": [
                {"role": "user", "content": "先自我介绍"},
                {"role": "assistant", "content": "我是一个AI"},
                {"role": "user", "content": "你能做什么？"},
            ],
            "gen_params": {"top_logprobs": -1},
        },
        {
            "messages": [
                {"role": "user", "content": "先自我介绍"},
                {"role": "assistant", "content": "我是一个AI"},
                {"role": "user", "content": "你能做什么？"},
            ],
            "gen_params": {"top_logprobs": 21},
        },
    ]

    for case in test_cases:
        run_test(
            messages=case["messages"],
            xpu_url=args.xpu_url,
            gpu_url=args.gpu_url if args.mode == "gpu-xpu" else None,
            timeout=args.timeout,
            log_file=args.log_file,
            mode=args.mode,
            gen_params=case.get("gen_params"),
        )
