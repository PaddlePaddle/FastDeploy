"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import argparse
import asyncio
import contextlib
import os
from typing import Union

from benchmark_dataset import EBChatDataset, EBDataset
from benchmark_serving import benchmark


def prepare_input_requests(num_prompts: int, dataset_name: str, dataset_path: str) -> Union[EBDataset, EBChatDataset]:
    dataset_mapping = {
        "EB": lambda: EBDataset(dataset_path=dataset_path).sample(num_requests=num_prompts),
        "EBChat": lambda: EBChatDataset(dataset_path=dataset_path).sample(num_requests=num_prompts),
    }

    try:
        input_requests = dataset_mapping[dataset_name]()
    except KeyError as err:
        raise ValueError(f"Unknown dataset: {dataset_name}") from err

    return input_requests


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def send_one_batch(base_url, max_concurrency, input_requests, disable_tqdm):
    selected_percentile_metrics = ["s_itl"]
    selected_percentiles = []
    # Run benchmark
    results = asyncio.run(
        benchmark(
            backend="openai-chat",
            api_url=f"{base_url}/v1/chat/completions",
            base_url=base_url,
            model_id="default",
            model_name="default",
            input_requests=input_requests,
            hyper_parameters={},
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=disable_tqdm,
            profile=False,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            ignore_eos=False,
            goodput_config_dict=None,
            max_concurrency=max_concurrency,
            lora_modules=None,
            extra_body=None,
        )
    )

    record = {
        "mean_s_itl_ms": results["mean_s_itl_ms"],
    }

    return record


def calculate_speedup(acceptance_rate, draft_token_step, t_ori, t_mtp):

    tmp = 0.0
    for i in range(draft_token_step):
        tmp += pow(acceptance_rate, i + 1)

    r_ac = tmp / (1 + tmp)

    return t_ori / ((1 - r_ac) * t_mtp)


def main(args):
    base_url = f"http://{args.host}:{args.port}"

    input_requests = prepare_input_requests(args.num_prompts, args.dataset_name, args.dataset_path)

    if len(args.max_concurrency) != len(args.s_itl_base_model):
        raise ValueError("--max_concurrency should be same length as --s_itl_base_model")

    for max_concurrency, s_itl in zip(args.max_concurrency, args.s_itl_base_model):
        # Warmup
        print("Starting warmup...")
        with open(os.devnull, "w") as f:
            with contextlib.redirect_stdout(f):
                send_one_batch(
                    base_url,
                    max_concurrency,
                    input_requests[0:max_concurrency],
                    True,
                )

        # Benchmark
        record = send_one_batch(base_url, max_concurrency, input_requests, False)

        metric_header = "Speed up"
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        for draft_token_step in args.draft_token_steps:
            speedup = calculate_speedup(
                args.acceptance_rate,
                draft_token_step,
                s_itl,
                record["mean_s_itl_ms"],
            )
            print("{:<40} {:<10.2f}".format(f"Speed up on {draft_token_step} steps draft", speedup))
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--acceptance-rate",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--draft-token-steps",
        type=int,
        nargs="+",
        default=(1, 2),
    )
    parser.add_argument(
        "--s_itl-base-model",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="EBChat",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
    )
    args = parser.parse_args()

    main(args)
