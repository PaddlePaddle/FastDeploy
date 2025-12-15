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


import base64
import io
import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional, Union

from PIL import Image

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
    response_format: Optional[dict] = None
    random_flag: bool = False


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
            response_format = entry.get("response_format")
            new_output_len = int(entry.get("max_tokens", output_len if output_len else 12288))

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
                    response_format=response_format,
                )
            )
            cnt += 1

        self.maybe_oversample_requests(samples, num_requests)
        return samples


class RandomTextDataset(BenchmarkDataset):
    """
    Generates random English words for pure text benchmarking.
    """

    # Common English words vocabulary
    COMMON_WORDS = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "is",
        "are",
        "was",
        "were",
        "been",
        "has",
        "had",
        "did",
        "done",
        "said",
        "told",
        "asked",
        "thought",
        "went",
        "saw",
        "looked",
        "found",
        "took",
        "gave",
        "made",
        "put",
        "set",
        "got",
        "ran",
        "came",
        "walked",
        "stood",
        "sat",
        "lay",
        "felt",
        "heard",
        "saw",
        "knew",
        "thought",
        "understood",
        "believed",
        "wanted",
        "needed",
        "liked",
        "loved",
        "hated",
        "feared",
        "hoped",
        "expected",
        "planned",
        "decided",
        "agreed",
        "disagreed",
        "argued",
        "discussed",
        "explained",
        "described",
        "reported",
        "announced",
        "declared",
        "stated",
        "claimed",
        "suggested",
        "proposed",
        "recommended",
        "advised",
        "warned",
        "threatened",
        "promised",
        "offered",
        "refused",
        "denied",
        "admitted",
        "confessed",
        "apologized",
        "forgave",
        "thanked",
        "congratulated",
        "celebrated",
        "welcomed",
        "greeted",
        "introduced",
        "presented",
        "showed",
        "demonstrated",
        "proved",
        "tested",
        "examined",
        "studied",
        "learned",
        "taught",
        "trained",
        "practiced",
        "performed",
        "played",
        "worked",
        "built",
        "created",
        "designed",
        "developed",
        "improved",
        "changed",
        "fixed",
        "solved",
        "completed",
        "finished",
        "started",
        "began",
        "continued",
        "stopped",
        "ended",
        "left",
        "arrived",
        "departed",
        "traveled",
        "moved",
        "stayed",
        "waited",
        "rested",
        "slept",
        "woke",
        "ate",
        "drank",
        "cooked",
        "cleaned",
        "washed",
        "dressed",
        "undressed",
        "showered",
        "bathed",
        "brushed",
        "combed",
        "shaved",
        "cut",
        "trimmed",
        "painted",
        "drew",
        "wrote",
        "read",
        "spoke",
        "listened",
        "heard",
        "saw",
        "watched",
        "looked",
        "observed",
        "noticed",
        "recognized",
        "remembered",
        "forgot",
        "learned",
        "understood",
        "knew",
        "believed",
        "doubted",
        "wondered",
        "thought",
        "considered",
        "decided",
        "chose",
        "selected",
        "preferred",
        "liked",
        "loved",
        "hated",
        "feared",
        "worried",
        "hoped",
        "expected",
        "planned",
        "prepared",
        "organized",
        "arranged",
        "scheduled",
        "timed",
        "measured",
        "counted",
        "calculated",
        "estimated",
        "valued",
        "priced",
        "cost",
        "paid",
        "bought",
        "sold",
        "traded",
        "exchanged",
        "shared",
        "divided",
        "combined",
        "joined",
        "connected",
        "attached",
        "separated",
        "divided",
        "cut",
        "broke",
        "fixed",
        "repaired",
        "built",
        "created",
        "made",
        "produced",
        "manufactured",
        "assembled",
        "constructed",
        "designed",
        "planned",
        "developed",
        "improved",
        "enhanced",
        "changed",
        "modified",
        "adjusted",
        "adapted",
        "converted",
        "transformed",
        "turned",
        "became",
        "grew",
        "developed",
        "evolved",
        "progressed",
        "advanced",
        "moved",
        "went",
        "came",
        "arrived",
        "departed",
        "left",
        "returned",
        "went back",
        "came back",
        "arrived back",
        "departed again",
        "left again",
        "returned again",
        "went away",
        "came close",
        "moved away",
        "approached",
        "reached",
        "arrived at",
        "departed from",
        "left from",
        "returned to",
        "went to",
        "came from",
        "traveled to",
        "traveled from",
        "moved to",
        "moved from",
        "stayed at",
        "remained at",
        "waited for",
        "rested at",
        "slept at",
        "woke up at",
        "ate at",
        "drank at",
        "cooked at",
        "cleaned at",
        "washed at",
        "dressed at",
        "undressed at",
        "showered at",
        "bathed at",
        "brushed at",
        "combed at",
        "shaved at",
        "cut at",
        "trimmed at",
        "painted at",
        "drew at",
        "wrote at",
        "read at",
        "spoke at",
        "listened at",
        "heard at",
        "saw at",
        "watched at",
        "looked at",
        "observed at",
        "noticed at",
        "recognized at",
        "remembered at",
        "forgot at",
        "learned at",
        "understood at",
        "knew at",
        "believed at",
        "doubted at",
        "wondered at",
        "thought at",
        "considered at",
        "decided at",
        "chose at",
        "selected at",
        "preferred at",
        "liked at",
        "loved at",
        "hated at",
        "feared at",
        "worried at",
        "hoped at",
        "expected at",
        "planned at",
        "prepared at",
        "organized at",
        "arranged at",
        "scheduled at",
        "timed at",
        "measured at",
        "counted at",
        "calculated at",
        "estimated at",
        "valued at",
        "priced at",
        "cost at",
        "paid at",
        "bought at",
        "sold at",
        "traded at",
        "exchanged at",
        "shared at",
        "divided at",
        "combined at",
        "joined at",
        "connected at",
        "attached at",
        "separated at",
        "divided at",
        "cut at",
        "broke at",
        "fixed at",
        "repaired at",
        "built at",
        "created at",
        "made at",
        "produced at",
        "manufactured at",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample(
        self,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        random_input_len: Optional[int] = None,
        random_output_len: Optional[int] = None,
        random_range_ratio: Optional[float] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples = []

        def sample_len(base_len: int, ratio: float) -> int:
            if base_len is None:
                return None
            if ratio is None or ratio <= 0:
                return base_len
            lo = max(1, int(base_len * (1 - ratio)))
            hi = int(base_len * (1 + ratio))
            return random.randint(lo, hi)

        for i in range(1, num_requests + 1):
            # [length * (1 - range_ratio), length * (1 + range_ratio)]
            sampled_input_len = sample_len(random_input_len, random_range_ratio)
            sampled_output_len = sample_len(random_output_len, random_range_ratio)

            words = [random.choice(self.COMMON_WORDS) for _ in range(sampled_input_len)]
            prompt_text = " ".join(words)

            data = {
                "messages": [{"role": "user", "content": prompt_text}],
            }

            samples.append(
                SampleRequest(
                    no=i,
                    json_data=data,
                    prompt=prompt_text,
                    prompt_len=sampled_input_len,
                    history_QA=data["messages"],
                    expected_output_len=sampled_output_len,
                    random_flag=True,
                )
            )
        return samples
