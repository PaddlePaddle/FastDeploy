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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_dataset.py
import argparse
import base64
import io
import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
from fontTools.feaLib import ast
from PIL import Image
from transformers import PreTrainedTokenizerBase

from fastdeploy.utils import FlexibleArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    no: int
    prompt: Union[str, Any]
    history_QA: Union[str, Any]
    json_data: Optional[dict]
    prompt_len: int
    expected_output_len: int


class BenchmarkDataset(ABC):
    """BenchmarkDataset"""

    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
        shuffle: bool = False,
        hyperparameter_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None
        self.shuffle = shuffle
        self.hyperparameter_path = hyperparameter_path
        self.hyperparameters = {}

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError("load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(self, num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(self, requests: list[SampleRequest], num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests, k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long or combined_too_long)


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

    if isinstance(image, str):
        image_url = image if image.startswith(("http://", "file://")) else f"file://{image}"
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image" " or str or dictionary with raw image bytes."
    )


class EBDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    temperature: float
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float
    top_p: float
    prompt_len: int

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = [json.loads(i.strip()) for i in f.readlines()]

        if self.shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.data)

    def sample(
        self,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        cnt = 1
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt = entry["text"]
            self.temperature = float(entry["temperature"])
            self.repetition_penalty = float(entry["penalty_score"])
            self.frequency_penalty = float(entry["frequency_score"])
            self.presence_penalty = float(entry["presence_score"])
            self.top_p = float(entry["topp"])
            self.prompt_len = int(entry["input_token_num"])
            new_output_len = int(entry["max_dec_len"])

            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(prompt, None)
            samples.append(
                SampleRequest(
                    no=cnt,
                    prompt=prompt,
                    prompt_len=self.prompt_len,
                    history_QA=[],
                    expected_output_len=new_output_len,
                )
            )
            cnt += 1

        self.maybe_oversample_requests(samples, num_requests)
        return samples


class EBChatDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    prompt_len: int

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = [json.loads(i.strip()) for i in f.readlines()]

        if self.shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.data)

    def sample(
        self,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        cnt = 1
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            json_data = entry
            prompt = entry["messages"][-1].get("content", "")
            history_QA = entry.get("messages", [])
            new_output_len = int(entry.get("max_tokens", 12288))

            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(prompt, None)
            samples.append(
                SampleRequest(
                    no=cnt,
                    json_data=json_data,
                    prompt=prompt,
                    prompt_len=0,
                    history_QA=history_QA,
                    expected_output_len=new_output_len,
                )
            )
            cnt += 1

        self.maybe_oversample_requests(samples, num_requests)
        return samples


class RandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        # Enforce range_ratio < 1
        assert range_ratio < 1.0, "random_range_ratio must be < 1.0 to ensure a valid sampling range"
        cnt = 1
        vocab_size = tokenizer.vocab_size
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        real_input_len = input_len - num_special_tokens

        prefix_token_ids = np.random.randint(0, vocab_size, size=prefix_len).tolist() if prefix_len > 0 else []

        # New sampling logic: [X * (1 - b), X * (1 + b)]
        input_low = int(real_input_len * (1 - range_ratio))
        input_high = int(real_input_len * (1 + range_ratio))
        output_low = int(output_len * (1 - range_ratio))
        output_high = int(output_len * (1 + range_ratio))

        # Add logging for debugging
        logger.info(
            "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
            input_low,
            input_high,
            output_low,
            output_high,
        )

        input_lens = np.random.randint(input_low, input_high + 1, size=num_requests)
        output_lens = np.random.randint(output_low, output_high + 1, size=num_requests)
        offsets = np.random.randint(0, vocab_size, size=num_requests)

        requests = []
        for i in range(num_requests):
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) % vocab_size).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            # After decoding the prompt we have to encode and decode it again.
            # This is done because in some cases N consecutive tokens
            # give a string tokenized into != N number of tokens.
            # For example for GPT2Tokenizer:
            # [6880, 6881] -> ['Ġcalls', 'here'] ->
            # [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
            # To avoid uncontrolled change of the prompt length,
            # the encoded sequence is truncated before being decode again.
            total_input_len = prefix_len + int(input_lens[i])
            re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[:total_input_len]
            prompt = tokenizer.decode(re_encoded_sequence)
            total_input_len = len(re_encoded_sequence)
            requests.append(
                SampleRequest(
                    no=cnt,
                    prompt=prompt,
                    prompt_len=total_input_len,
                    history_QA=[],
                    json_data=None,
                    expected_output_len=int(output_lens[i]),
                )
            )
            cnt += 1
        return requests


class _ValidateDatasetArgs(argparse.Action):
    """Argparse action to validate dataset name and path compatibility."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

        # Get current values of both dataset_name and dataset_path
        dataset_name = getattr(namespace, "dataset_name", "random")
        dataset_path = getattr(namespace, "dataset_path", None)

        # Validate the combination
        if dataset_name == "random" and dataset_path is not None:
            parser.error(
                "Cannot use 'random' dataset with --dataset-path. "
                "Please specify the appropriate --dataset-name (e.g., "
                "'sharegpt', 'custom', 'sonnet') for your dataset file: "
                f"{dataset_path}"
            )


def get_samples(args):
    """Get the sample requests from the specified dataset."""
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    # For datasets that follow a similar structure, use a mapping.
    dataset_mapping = {
        "EB": lambda: EBDataset(random_seed=args.seed, dataset_path=args.dataset_path, shuffle=args.shuffle).sample(
            num_requests=args.num_prompts,
            output_len=args.sharegpt_output_len,
        ),
        "EBChat": lambda: EBChatDataset(
            random_seed=args.seed, dataset_path=args.dataset_path, shuffle=args.shuffle
        ).sample(
            num_requests=args.num_prompts,
            output_len=args.sharegpt_output_len,
        ),
    }

    try:
        input_requests = dataset_mapping[args.dataset_name]()
    except KeyError as err:
        raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    return input_requests


def add_dataset_parser(parser: FlexibleArgumentParser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "burstgpt",
            "sonnet",
            "random",
            "hf",
            "EB",
            "EBChat",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        action=_ValidateDatasetArgs,
        help="Path to the sharegpt/sonnet dataset. " "Or the huggingface dataset ID if using HF dataset.",
    )
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Do not oversample if the dataset has " "fewer samples than num-prompts.",
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help="Number of output tokens per request, used only for custom dataset.",
    )
    custom_group.add_argument(
        "--custom-skip-chat-template",
        action="store_true",
        help="Skip applying chat template to prompt, used only for custom dataset.",
    )

    spec_bench_group = parser.add_argument_group("spec bench dataset options")
    spec_bench_group.add_argument(
        "--spec-bench-output-len",
        type=int,
        default=256,
        help="Num of output tokens per request, used only for spec bench dataset.",
    )
    spec_bench_group.add_argument(
        "--spec-bench-category",
        type=str,
        default=None,
        help="Category for spec bench dataset. If None, use all categories.",
    )

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length " "from the ShareGPT dataset.",
    )

    blazedit_group = parser.add_argument_group("blazedit dataset options")
    blazedit_group.add_argument(
        "--blazedit-min-distance",
        type=float,
        default=0.0,
        help="Minimum distance for blazedit dataset. Min: 0, Max: 1.0",
    )
    blazedit_group.add_argument(
        "--blazedit-max-distance",
        type=float,
        default=1.0,
        help="Maximum distance for blazedit dataset. Min: 0, Max: 1.0",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )
    random_group.add_argument(
        "--random-batch-size",
        type=int,
        default=1,
        help=("Batch size for random sampling. " "Only used for embeddings benchmark."),
    )

    def _parse_mm_bucket_config(v: object) -> dict[tuple[int, int, int], float]:
        # If already a dict (e.g., programmatic call), normalize keys
        def normalize(d: dict) -> dict[tuple[int, int, int], float]:
            out: dict[tuple[int, int, int], float] = {}
            for k, val in d.items():
                key = k
                if isinstance(key, str):
                    with suppress(Exception):
                        key = ast.literal_eval(key)
                if not (isinstance(key, tuple) and len(key) == 3 and all(isinstance(x, int) for x in key)):
                    raise ValueError(f"Invalid bucket key {k!r}. Expected tuple (H, W, T).")
                out[(int(key[0]), int(key[1]), int(key[2]))] = float(val)
            return out

        if isinstance(v, dict):
            return normalize(v)
        if isinstance(v, str):
            # Python literal (supports tuple keys)
            parsed = ast.literal_eval(v)
            if not isinstance(parsed, dict):
                raise ValueError("Bucket config must parse to a dict.")
            return normalize(parsed)
        raise ValueError("Unsupported value for --random-mm-bucket-config.")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset", type=str, default=None, help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split", type=str, default=None, help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-name",
        type=str,
        default=None,
        help=(
            "Name of the dataset on HuggingFace "
            "(e.g., 'lmarena-ai/VisionArena-Chat'). "
            "Specify this if your dataset-path is a local path."
        ),
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths " "from the sampled HF dataset.",
    )

    prefix_repetition_group = parser.add_argument_group("prefix repetition dataset options")
    prefix_repetition_group.add_argument(
        "--prefix-repetition-prefix-len",
        type=int,
        default=256,
        help="Number of prefix tokens per request, used only for prefix " "repetition dataset.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-suffix-len",
        type=int,
        default=256,
        help="Number of suffix tokens per request, used only for prefix "
        "repetition dataset. Total input length is prefix_len + suffix_len.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-num-prefixes",
        type=int,
        default=10,
        help="Number of prefixes to generate, used only for prefix repetition "
        "dataset. Prompts per prefix is num_requests // num_prefixes.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for prefix " "repetition dataset.",
    )
