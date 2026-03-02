import json
import time
import requests

XPU_URL = "http://127.0.0.1:8188/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 30
LOG_FILE = "/home/paddle_test/works/fd/FastDeploy/benchmarks/gpu_xpu_format_test.log"


def write_log(record: dict):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=2))
        f.write("\n" + "=" * 80 + "\n")


def main():
    payload = {
        "messages": [{"role": "user", "content": "你叫什么？"}],
        "logprobs": True,
        "top_logprobs": 0,
        "max_tokens": 1,
    }
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        resp = requests.post(XPU_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        ok = True
    except Exception as e:
        data = {"__error__": str(e)}
        ok = False

    record = {
        "timestamp": ts,
        "mode": "xpu-min-case",
        "request": payload,
        "xpu_response": data,
        "ok": ok,
    }
    write_log(record)
    print(f"[{ts}] ok={ok}")


if __name__ == "__main__":
    main()
