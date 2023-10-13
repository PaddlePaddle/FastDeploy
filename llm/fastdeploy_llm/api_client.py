"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests


def post_http_request(text: str, api_url: str,
                      stream: bool=False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "text": text,
        "max_dec_len": 1024,
        "min_dec_len": 2,
        "penalty_score": 1.0,
        "temperature": 1.0,
        "topp": 0.0,
        "frequency_score": 0.0,
        "eos_token_id": 2,
        "presence_score": 0.0,
        "task_id": 1,
    }
    response = requests.post(
        api_url, headers=headers, json=pload, stream=False)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url)

    output = json.loads(response.content)
    print("result: ", output)
