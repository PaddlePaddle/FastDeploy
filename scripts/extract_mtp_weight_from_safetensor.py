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


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(paddle.float32)
    4
    ```
    """
    if str(dtype) in {"paddle.bool", "bool"}:
        return 1 / 8
    if str(dtype) in {"paddle.float8_e4m3fn", "paddle.float8_e5m2", "float8_e4m3fn", "float8_e5m2"}:
        return 1
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def extract_mtp_weights(input_dir: str) -> dict:
    """
    Load all MTP-related weights from safetensors files in input_dir.
    """
    index_path = os.path.join(input_dir, SAFE_WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    required_files = {v for k, v in weight_map.items() if "mtp" in k}
    logger.info(f"Found {len(required_files)} shards with MTP weights.")

    state_dict = {}
    for file_name in required_files:
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            logger.warning(f"Shard not found: {file_path}")
            continue
        logger.info(f"Loading shard: {file_path}")
        with safe_open(file_path, framework="np", device="cpu") as f:
            for k in f.keys():
                if "mtp" in k:
                    state_dict[k] = f.get_tensor(k)

    logger.info(f"Loaded {len(state_dict)} MTP weights.")
    return state_dict


def save_safetensors(state_dict: dict, output_dir: str):
    """
    Save state_dict as safetensors shards into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Converting tensors to numpy arrays.")
    for k in list(state_dict.keys()):
        if isinstance(state_dict[k], paddle.Tensor):
            tensor = state_dict.pop(k)
            array = tensor.cpu().numpy()
            state_dict[k] = array

    logger.info("Sharding and saving safetensors.")
    shards, index = shard_checkpoint(
        state_dict,
        max_shard_size="5GB",
        weights_name=SAFE_WEIGHTS_NAME,
        shard_format="naive",
    )

    for shard_file, shard in shards.items():
        save_path = os.path.join(output_dir, shard_file)
        logger.info(f"Saving shard: {save_path}")
        safe_save_file(shard, save_path, metadata={"format": "np"})

    # If only one shard is returned, SAFE_WEIGHTS_INDEX_NAME will be null
    if len(shards) == 1:
        logger.info("Generate index file for single shard")
        weight_size = 0
        for key, weight in shards["model.safetensors"].items():
            weight_size += np.prod(weight.shape) * dtype_byte_size(weight.dtype)

        index = {
            "metadata": {"total_size": int(weight_size)},
            "weight_map": {k: "model.safetensors" for k in shards["model.safetensors"].keys()},
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
