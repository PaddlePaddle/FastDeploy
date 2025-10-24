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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/latency.py

import argparse
import dataclasses
import json
import time

import numpy as np
from tqdm import tqdm

import fastdeploy.envs as envs
from fastdeploy.engine.args_utils import EngineArgs


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters", type=int, default=30, help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include " "detokenization time in the latency measurement)"),
    )

    parser = EngineArgs.add_cli_args(parser)
    # V1 enables prefix caching by default which skews the latency
    # numbers. We need to disable prefix caching by default.
    parser.set_defaults(enable_prefix_caching=False)


def main(args: argparse.Namespace):
    if args.profile and not envs.VLLM_TORCH_PROFILER_DIR:
        raise OSError(
            "The environment variable 'VLLM_TORCH_PROFILER_DIR' is not set. "
            "Please set it to a valid path to use torch profiler."
        )
    engine_args = EngineArgs.from_cli_args(args)

    # Lazy import to avoid importing LLM when the bench command is not selected.
    from fastdeploy import LLM, SamplingParams

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.cfg.model_config.max_model_len >= (args.input_len + args.output_len), (
        "Please ensure that max_model_len is greater than" " the sum of input_len and output_len."
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.output_len,
    )
    dummy_prompt_token_ids = np.random.randint(10000, size=(args.batch_size, args.input_len))
    dummy_prompts = [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]

    def llm_generate():
        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False, stream=True)

    def run_to_completion():
        start_time = time.perf_counter()
        llm_generate()
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion()

    if args.profile:
        print("Profiling...")
        run_to_completion()
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion())
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
