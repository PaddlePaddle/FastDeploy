#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
Real inference entry point for EngineService architecture comparison.

This script performs actual inference with real prompts and supports
switching between old and new architectures via environment variables.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine_service_factory import create_engine_service
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with EngineService for architecture comparison"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_PATH", "./models/Qwen/Qwen2.5-7B"),
        help="Path to model directory",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=[
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
            "Write a short poem about the ocean.",
        ],
        help="List of prompts to process",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model length",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Maximum number of batched tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Sampling top_p",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        help="GPU ID to use",
    )
    parser.add_argument(
        "--use-new-arch",
        action="store_true",
        help="Use new architecture (default: old architecture)",
    )
    parser.add_argument(
        "--splitwise-role",
        type=str,
        choices=["none", "prefill", "decode", "mixed"],
        default="mixed",
        help="Splitwise role",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file to save results (JSON format)",
    )
    return parser.parse_args()


def run_inference(args):
    """
    Run inference with specified architecture and prompts.

    Args:
        args: Parsed command line arguments
    """
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["FD_USE_NEW_ENGINE_ARCHITECTURE"] = "1" if args.use_new_arch else "0"

    arch_name = "NEW" if args.use_new_arch else "OLD"
    print(f"\n{'='*60}")
    print(f"Running inference with {arch_name} architecture")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(args.prompts)}")
    print(f"GPU: {args.gpu_id}")
    print(f"Splitwise role: {args.splitwise_role}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")

    # Create engine args
    engine_args = EngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        splitwise_role=args.splitwise_role,
        engine_worker_queue_port=6778,
        cache_queue_port=6779,
    )

    # Create LLM instance
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Run inference
    results = []
    start_time = time.time()

    for i, prompt in enumerate(args.prompts):
        print(f"\n--- Prompt {i+1}/{len(args.prompts)} ---")
        print(f"Input: {prompt}\n")

        prompt_start = time.time()
        outputs = llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        prompt_time = time.time() - prompt_start

        # Get output
        output = outputs[0]
        output_text = output.outputs.text if hasattr(output, "outputs") else ""
        output_tokens = output.outputs.token_ids if hasattr(output, "outputs") else []

        print(f"Output: {output_text}")
        print(f"Tokens generated: {len(output_tokens)}")
        print(f"Time: {prompt_time:.2f}s")

        results.append({
            "prompt": prompt,
            "output": output_text,
            "tokens": output_tokens,
            "num_tokens": len(output_tokens),
            "time": prompt_time,
        })

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(args.prompts):.2f}s")
    print(f"{'='*60}\n")

    # Save results to file if specified
    if args.output_file:
        import json
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_data = {
            "architecture": "new" if args.use_new_arch else "old",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "splitwise_role": args.splitwise_role,
            "total_time": total_time,
            "num_prompts": len(args.prompts),
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        sys.exit(1)

    try:
        results = run_inference(args)
        return 0
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
