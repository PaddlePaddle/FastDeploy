# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark offline inference throughput."""
import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from fastdeploy.benchmarks.datasets import (
    EBChatDataset,
    EBDataset,
    RandomDataset,
    SampleRequest,
)
from fastdeploy.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import RequestOutput


def run_fd(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
) -> tuple[float, Optional[list[RequestOutput]]]:
    from fastdeploy import LLM, SamplingParams

    llm = LLM(**dataclasses.asdict(engine_args))
    assert all(
        llm.llm_engine.cfg.max_model_len >= (request.prompt_len + request.expected_output_len) for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of"
        " prompt_len and expected_output_len for all requests."
    )
    # Add the requests to the engine.
    prompts = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        # 处理tokenized输入
        if "prompt_token_ids" in request.prompt:
            prompt = {
                "prompt_token_ids": request.prompt["prompt_token_ids"],
                "multi_modal_data": getattr(request, "multi_modal_data", None),
            }
        # 处理普通文本输入
        else:
            prompt = {"prompt": str(request.prompt), "multi_modal_data": getattr(request, "multi_modal_data", None)}
        prompts.append(prompt)

        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                max_tokens=request.expected_output_len,
            )
        )
    outputs = None
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start, outputs


def run_fd_chat(
    requests: list[SampleRequest], n: int, engine_args: EngineArgs, disable_detokenize: bool = False
) -> tuple[float, list[RequestOutput]]:
    """
    Run vLLM chat benchmark. This function is recommended ONLY for benchmarking
    multimodal models as it properly handles multimodal inputs and chat
    formatting. For non-multimodal models, use run_vllm() instead.
    """
    from fastdeploy import LLM, SamplingParams

    llm = LLM(**dataclasses.asdict(engine_args))

    assert all(
        llm.llm_engine.cfg.max_model_len >= (request.prompt_len + request.expected_output_len) for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of "
        "prompt_len and expected_output_len for all requests."
    )

    prompts = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                max_tokens=request.expected_output_len,
            )
        )
    start = time.perf_counter()
    outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start, outputs


def run_hf(
    requests: list[SampleRequest],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
    disable_detokenize: bool = False,
) -> float:
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: list[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt = requests[i].prompt
        prompt_len = requests[i].prompt_len
        output_len = requests[i].expected_output_len
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            next_prompt_len = requests[i + 1].prompt_len
            next_output_len = requests[i + 1].expected_output_len
            if (max(max_prompt_len, next_prompt_len) + max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        if not disable_detokenize:
            # Include the decoding time.
            tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def save_to_pytorch_benchmark_format(args: argparse.Namespace, results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={k: results[k] for k in ["elapsed_time", "num_requests", "total_num_tokens"]},
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        # "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        # "max_loras": args.max_loras,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }
    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["tokenizer"] = tokenizer
        dataset_cls = RandomDataset
    elif args.dataset_name == "EB":
        dataset_cls = EBDataset
    elif args.dataset_name == "EBChat":
        dataset_cls = EBChatDataset
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    return dataset_cls(**common_kwargs).sample(**sample_kwargs)


def validate_args(args):
    """
    Validate command-line arguments.
    """

    # === Deprecation and Defaulting ===
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next release. "
            "Please use '--dataset-name' and '--dataset-path' instead.",
            stacklevel=2,
        )
        args.dataset_path = args.dataset

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    # === Backend Validation ===
    valid_backends = {"fastdeploy", "hf", "fastdeploy-chat"}
    if args.backend not in valid_backends:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # === Dataset Configuration ===
    if not args.dataset and not args.dataset_path:
        print("When dataset path is not set, it will default to random dataset")
        args.dataset_name = "random"
        if args.input_len is None:
            raise ValueError("input_len must be provided for a random dataset")

    # === Dataset Name Specific Checks ===
    # --hf-subset and --hf-split: only used
    # when dataset_name is 'hf'
    if args.dataset_name != "hf" and (
        getattr(args, "hf_subset", None) is not None or getattr(args, "hf_split", None) is not None
    ):
        warnings.warn(
            "--hf-subset and --hf-split will be ignored \
                since --dataset-name is not 'hf'.",
            stacklevel=2,
        )
    # elif args.dataset_name == "hf":
    #     if args.dataset_path in (
    #             VisionArenaDataset.SUPPORTED_DATASET_PATHS.keys()
    #             | ConversationDataset.SUPPORTED_DATASET_PATHS):
    #         assert args.backend == "vllm-chat", f"{args.dataset_path} needs to use vllm-chat as the backend."  #noqa: E501
    #     elif args.dataset_path in (InstructCoderDataset.SUPPORTED_DATASET_PATHS
    #                                | AIMODataset.SUPPORTED_DATASET_PATHS):
    #         assert args.backend == "vllm", f"{args.dataset_path} needs to use vllm as the backend."  #noqa: E501
    #     else:
    #         raise ValueError(
    #             f"{args.dataset_path} is not supported by hf dataset.")

    # --random-range-ratio: only used when dataset_name is 'random'
    if args.dataset_name != "random" and args.random_range_ratio is not None:
        warnings.warn(
            "--random-range-ratio will be ignored since \
                --dataset-name is not 'random'.",
            stacklevel=2,
        )

    # --prefix-len: only used when dataset_name is 'random', 'sonnet', or not
    # set.
    if args.dataset_name not in {"random", "sonnet", None} and args.prefix_len is not None:
        warnings.warn(
            "--prefix-len will be ignored since --dataset-name\
                 is not 'random', 'sonnet', or not set.",
            stacklevel=2,
        )

    # === LoRA Settings ===
    if getattr(args, "enable_lora", False) and args.lora_path is None:
        raise ValueError("LoRA path must be provided when enable_lora is True")

    # === Backend-specific Validations ===
    if args.backend == "hf" and args.hf_max_batch_size is None:
        raise ValueError("HF max batch size is required for HF backend")
    if args.backend != "hf" and args.hf_max_batch_size is not None:
        raise ValueError("HF max batch size is only for HF backend.")

    if args.backend in {"hf", "mii"} and getattr(args, "quantization", None) is not None:
        raise ValueError("Quantization is only for vLLM backend.")


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--backend", type=str, choices=["fastdeploy", "hf", "fastdeploy-chat"], default="fastdeploy")
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["EBChat", "random", "EB"],
        help="Name of the dataset to benchmark on.",
        default="random",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in\
            the next release. The dataset is expected to "
        "be a json in form of list[dict[..., conversations: "
        "list[dict[..., value: <prompt_or_response>]]]]",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the " "output length from the dataset.",
    )
    parser.add_argument("--n", type=int, default=1, help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts to process.")
    parser.add_argument("--hf-max-batch-size", type=int, default=None, help="Maximum batch size for HF backend.")
    parser.add_argument(
        "--output-json", type=str, default=None, help="Path to save the throughput results in JSON format."
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        default=False,
        help="Disable decoupled async engine frontend.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize the response (i.e. do not include " "detokenization time in the measurement)"),
    )
    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the lora adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before the random " "context in a request (default: 0).",
    )
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for RandomDataset. Must be in the range [0, 1) to define "
        "a symmetric sampling range "
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )

    # hf dtaset
    parser.add_argument("--hf-subset", type=str, default=None, help="Subset of the HF dataset.")
    parser.add_argument("--hf-split", type=str, default=None, help="Split of the HF dataset.")

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)


def main(args: argparse.Namespace):
    if args.tokenizer is None:
        args.tokenizer = args.model
    validate_args(args)
    if args.seed is None:
        args.seed = 0
    random.seed(args.seed)
    # Sample the requests.
    if args.backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    else:
        tokenizer = None
    requests = get_requests(args, tokenizer)
    # is_multi_modal = any(request.multi_modal_data is not None
    #                      for request in requests)
    request_outputs: Optional[list[RequestOutput]] = None
    if args.backend == "fastdeploy":
        elapsed_time, request_outputs = run_fd(
            requests, args.n, EngineArgs.from_cli_args(args), args.disable_detokenize
        )
    elif args.backend == "hf":
        if not TORCH_AVAILABLE:
            raise Exception("PyTorch is not available.")
        else:
            assert args.tensor_parallel_size == 1
            elapsed_time = run_hf(
                requests,
                args.model,
                tokenizer,
                args.n,
                args.hf_max_batch_size,
                args.trust_remote_code,
                args.disable_detokenize,
            )
    elif args.backend == "fastdeploy-chat":
        elapsed_time, request_outputs = run_fd_chat(
            requests, args.n, EngineArgs.from_cli_args(args), args.disable_detokenize
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if request_outputs:
        # Note: with the vllm and vllm-chat backends,
        # we have request_outputs, which we use to count tokens.
        total_prompt_tokens = 0
        total_output_tokens = 0
        for ro in request_outputs:
            if not isinstance(ro, RequestOutput):
                continue
            total_prompt_tokens += len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
            if ro.outputs and hasattr(ro.outputs, "token_ids"):
                total_output_tokens += len(ro.outputs.token_ids)
        total_num_tokens = total_prompt_tokens + total_output_tokens
    else:
        total_num_tokens = sum(r.prompt_len + r.expected_output_len for r in requests)
        total_output_tokens = sum(r.expected_output_len for r in requests)
        total_prompt_tokens = total_num_tokens - total_output_tokens

    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)
