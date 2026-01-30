"""
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

import argparse
import json
import os
import re
from typing import Dict

import numpy as np
import paddle
from paddleformers.transformers.model_utils import shard_checkpoint
from paddleformers.utils.env import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from paddleformers.utils.log import logger
from safetensors import safe_open
from safetensors.numpy import save_file as safe_save_file


def parse_args():
    """"""
    parser = argparse.ArgumentParser(description="Extract and save MTP weights from safetensors.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input safetensors model directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for saving processed weights.",
    )
    return parser.parse_args()


def dtype_byte_size(dtype) -> int:
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    NOTE: This returns an integer number of bytes for determinism in metadata.

    Example:

    ```py
    >>> dtype_byte_size(paddle.float32)
    4
    ```
    """
    s = str(dtype)

    # bool is stored as 1 byte in most tensor formats; for metadata determinism, use 1.
    if s in {"paddle.bool", "bool"}:
        return 1

    # Paddle float8 types
    if s in {"paddle.float8_e4m3fn", "paddle.float8_e5m2", "float8_e4m3fn", "float8_e5m2"}:
        return 1

    bit_search = re.search(r"[^\d](\d+)$", s)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    if bit_size % 8 != 0:
        raise ValueError(f"Unsupported dtype with non-byte-aligned bits: {dtype}.")
    return bit_size // 8


def _sorted_required_files(weight_map: Dict[str, str]) -> list[str]:
    """
    Collect shard file names that contain 'mtp' weights and return a stable sorted list.
    """
    # Use set to unique then sort to make iteration deterministic across runs.
    required_files = sorted({v for k, v in weight_map.items() if "mtp" in k})
    return required_files


def extract_mtp_weights(input_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all MTP-related weights from safetensors files in input_dir.

    Determinism:
    - iterate shards in sorted order
    - iterate tensor keys in sorted order
    """
    index_path = os.path.join(input_dir, SAFE_WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    required_files = _sorted_required_files(weight_map)
    logger.info(f"Found {len(required_files)} shards with MTP weights.")

    state_dict: Dict[str, np.ndarray] = {}
    for file_name in required_files:
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            logger.warning(f"Shard not found: {file_path}")
            continue

        logger.info(f"Loading shard: {file_path}")
        with safe_open(file_path, framework="np", device="cpu") as f:
            # Sort keys for determinism
            for k in sorted(f.keys()):
                if "mtp" in k:
                    state_dict[k] = f.get_tensor(k)

    # Final sort of state_dict by key to make sharding deterministic.
    state_dict = dict(sorted(state_dict.items(), key=lambda kv: kv[0]))

    logger.info(f"Loaded {len(state_dict)} MTP weights.")
    return state_dict


def save_safetensors(state_dict: Dict[str, object], output_dir: str):
    """
    Save state_dict as safetensors shards into output_dir.

    Determinism:
    - ensure state_dict is ordered by key before sharding
    - when generating single-shard index, sort keys for stable weight_map ordering
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Converting tensors to numpy arrays.")
    for k in list(state_dict.keys()):
        if isinstance(state_dict[k], paddle.Tensor):
            tensor = state_dict.pop(k)
            array = tensor.cpu().numpy()
            state_dict[k] = array

    # Ensure deterministic order before sharding
    state_dict = dict(sorted(state_dict.items(), key=lambda kv: kv[0]))

    logger.info("Sharding and saving safetensors.")
    shards, index = shard_checkpoint(
        state_dict,
        max_shard_size="5GB",
        weights_name=SAFE_WEIGHTS_NAME,
        shard_format="naive",
    )

    # Save shards in stable order (by filename)
    for shard_file in sorted(shards.keys()):
        shard = shards[shard_file]
        save_path = os.path.join(output_dir, shard_file)
        logger.info(f"Saving shard: {save_path}")
        safe_save_file(shard, save_path, metadata={"format": "np"})

    # If only one shard is returned, SAFE_WEIGHTS_INDEX_NAME will be null
    if len(shards) == 1:
        logger.info("Generate index file for single shard")

        # Be robust: infer the only shard file name
        only_shard_file = next(iter(shards.keys()))
        only_shard = shards[only_shard_file]

        weight_size = 0
        for key in sorted(only_shard.keys()):
            weight = only_shard[key]
            weight_size += int(np.prod(weight.shape)) * int(dtype_byte_size(weight.dtype))

        index = {
            "metadata": {"total_size": int(weight_size)},
            "weight_map": {k: only_shard_file for k in sorted(only_shard.keys())},
        }

    index_path = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Saved index file: {index_path}")


def main():
    """"""
    args = parse_args()
    logger.info(f"Input dir: {args.input_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    state_dict = extract_mtp_weights(args.input_dir)
    save_safetensors(state_dict, args.output_dir)
    logger.info("MTP weights extracted and saved successfully.")


if __name__ == "__main__":
    main()
