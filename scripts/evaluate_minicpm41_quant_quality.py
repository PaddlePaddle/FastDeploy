# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the deterministic MiniCPM4.1 quantization quality smoke suite."""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

QUALITY_CASES = (
    {
        "name": "integer_multiplication",
        "prompt": "请只输出阿拉伯数字，不要解释：17乘以23等于多少？",
        "exact_answers": ("391",),
    },
    {
        "name": "world_capital",
        "prompt": "请只输出城市名，不要解释：法国的首都是什么？",
        "exact_answers": ("巴黎",),
    },
    {
        "name": "water_formula",
        "prompt": "请只输出化学式，不要解释：水的化学式是什么？",
        "exact_answers": ("H2O",),
    },
    {
        "name": "square_is_rectangle",
        "prompt": "请只回答“是”或“否”，不要解释：所有正方形都是矩形吗？",
        "exact_answers": ("是",),
    },
    {
        "name": "number_sequence",
        "prompt": "请只输出下一个数字，不要解释：数列2、4、8、16的下一项是什么？",
        "exact_answers": ("32",),
    },
    {
        "name": "rayleigh_scattering",
        "prompt": "请用一句话解释为什么天空是蓝色的。",
        "required_terms": ("蓝", "散射"),
    },
    {
        "name": "integer_addition",
        "prompt": "请只输出阿拉伯数字，不要解释：125加376等于多少？",
        "exact_answers": ("501",),
    },
    {
        "name": "square_root",
        "prompt": "请只输出阿拉伯数字，不要解释：81的算术平方根是多少？",
        "exact_answers": ("9",),
    },
    {
        "name": "largest_planet",
        "prompt": "请只输出行星名，不要解释：太阳系中体积最大的行星是什么？",
        "exact_answers": ("木星",),
    },
    {
        "name": "binary_conversion",
        "prompt": "请只输出阿拉伯数字，不要解释：二进制1010对应的十进制数是多少？",
        "exact_answers": ("10",),
    },
    {
        "name": "literature_author",
        "prompt": "请只输出人名，不要解释：《红楼梦》的作者通常认为是谁？",
        "exact_answers": ("曹雪芹",),
    },
    {
        "name": "basic_syllogism",
        "prompt": "请只回答“是”或“否”，不要解释：所有猫都是动物，小花是一只猫，因此小花是动物，对吗？",
        "exact_answers": ("是",),
    },
)


def generation_config(max_tokens: int) -> dict:
    return {
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Served model name or local checkpoint path.")
    parser.add_argument("--label", required=True, help="Stable report label, for example bf16 or wint4.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8180", help="FastDeploy API base URL.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the JSON report.")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-passed", type=int, default=10)
    return parser.parse_args()


def normalize_short_answer(output: str) -> str:
    normalized = unicodedata.normalize("NFKC", output).strip()
    return normalized.rstrip("。.!！").strip()


def score_output(case: dict, output: str) -> tuple[bool, str]:
    normalized = normalize_short_answer(output)
    exact_answers = case.get("exact_answers")
    if exact_answers is not None:
        expected = tuple(normalize_short_answer(answer) for answer in exact_answers)
        passed = normalized in expected
        return passed, f"expected one of {expected}, got {normalized!r}"

    required_terms = tuple(case["required_terms"])
    missing = [term for term in required_terms if term not in normalized]
    return not missing, "missing required terms: " + ", ".join(missing) if missing else "all required terms found"


def send_chat(args: argparse.Namespace, prompt: str) -> tuple[dict, float]:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        **generation_config(args.max_tokens),
    }
    request = Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"cannot reach {args.base_url}: {error.reason}") from error
    return result, time.perf_counter() - start


def extract_output(response: dict) -> str:
    try:
        output = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"unexpected chat completion response: {response}") from error
    if not isinstance(output, str):
        raise ValueError(f"chat completion content must be a string, got {type(output).__name__}")
    return output


def run_suite(args: argparse.Namespace) -> dict:
    if not 1 <= args.min_passed <= len(QUALITY_CASES):
        raise ValueError(f"--min-passed must be between 1 and {len(QUALITY_CASES)}")

    results = []
    for case in QUALITY_CASES:
        response, elapsed = send_chat(args, case["prompt"])
        output = extract_output(response)
        passed, detail = score_output(case, output)
        results.append(
            {
                "name": case["name"],
                "prompt": case["prompt"],
                "output": output,
                "passed": passed,
                "score_detail": detail,
                "latency_seconds": elapsed,
                "usage": response.get("usage", {}),
            }
        )

    passed = sum(result["passed"] for result in results)
    return {
        "label": args.label,
        "model": args.model,
        "generation_config": generation_config(args.max_tokens),
        "threshold": {"min_passed": args.min_passed, "total": len(QUALITY_CASES)},
        "summary": {
            "passed": passed,
            "total": len(QUALITY_CASES),
            "pass_rate": passed / len(QUALITY_CASES),
            "accepted": passed >= args.min_passed,
        },
        "cases": results,
    }


def main() -> int:
    args = parse_args()
    try:
        report = run_suite(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1

    summary = report["summary"]
    for case in report["cases"]:
        state = "PASS" if case["passed"] else "FAIL"
        print(f"[{state}] {case['name']}: {case['output']}")
    print(
        f"{args.label}: {summary['passed']}/{summary['total']} passed; "
        f"accepted={summary['accepted']}; report={args.output}"
    )
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
