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

import argparse
import json
import logging
import math
import os
from typing import List, Tuple

from fastdeploy.model_executor.ops.gpu.deep_gemm.jit_kernels.gemm import get_smem_config

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
logger.setLevel(os.getenv("PRE_COMPILE_LOG_LEVEL", "INFO"))


def generate_kn_pairs(args, model_cfg: dict) -> Tuple[List, List, List]:
    hidden_size = model_cfg["hidden_size"]
    intermediate_size = model_cfg["intermediate_size"]
    moe_intermediate_size = model_cfg["moe_intermediate_size"]
    num_attention_heads = model_cfg["num_attention_heads"]
    num_key_value_heads = model_cfg["num_key_value_heads"]
    head_dim = int(hidden_size / num_attention_heads)
    tp_size = args.tensor_parallel_size
    ep_size = args.expert_parallel_size
    has_shared_experts = args.has_shared_experts.lower() == "true"

    gemm_kn_pairs = []
    grouped_gemm_contiguous_kn_pairs = []
    grouped_gemm_masked_kn_pairs = []
    logger.debug("Generating kn pairs for tensor parallel.")
    # Dense normal gemm
    gemm_kn_pairs.extend(
        [
            [int(intermediate_size / tp_size), hidden_size],
            [hidden_size, int(head_dim * (num_attention_heads + num_key_value_heads * 2) / tp_size)],
            [hidden_size, int(intermediate_size * 2 / tp_size)],
            [int(hidden_size / tp_size), hidden_size],
        ]
    )

    # Moe grouped gemm contiguous
    grouped_gemm_contiguous_kn_pairs.extend(
        [
            [int(moe_intermediate_size / tp_size), hidden_size],
            [hidden_size, int(moe_intermediate_size * 2 / tp_size)],
        ]
    )

    if ep_size > 1:
        # Moe grouped gemm masked
        grouped_gemm_masked_kn_pairs.extend(
            [
                [moe_intermediate_size, hidden_size],
                [hidden_size, int(moe_intermediate_size * 2)],
            ]
        )
    if has_shared_experts:
        logger.debug("Generating kn pairs for models with shared experts.")
        gemm_kn_pairs.extend(
            [
                [hidden_size, int(moe_intermediate_size * 4 / tp_size)],
                [int(moe_intermediate_size * 2 / tp_size), hidden_size],
            ]
        )

    return (
        gemm_kn_pairs,
        grouped_gemm_contiguous_kn_pairs,
        grouped_gemm_masked_kn_pairs,
    )


def generate_json(
    kn_pairs: list,
    moe_num_experts: int,
    output_path: str,
    is_grouped_contiguous: bool = False,
    is_grouped_masked: bool = False,
):
    if not is_grouped_contiguous:
        BLOCK_MS = [64, 128, 256]
    else:
        BLOCK_MS = [128]
    BLOCK_NS = list(range(16, 129, 8)) + [144, 160]
    TMA_MULTICAST_CONFIGS = [(1, True), (1, False), (2, True), (2, False)]
    counter = 0
    with open(output_path, "a+", encoding="utf-8") as f:
        for block_m in BLOCK_MS:
            # NOTES: the block sizes can not be too large, so at least one dim less than 128
            for block_n in filter(lambda bn: block_m <= 128 or bn <= 128, BLOCK_NS):
                if 128 % block_n != 0 and 128 // math.gcd(128, block_n) <= 4:
                    NUM_STAGES = [4, 3]
                else:
                    NUM_STAGES = [8, 7, 6, 5, 4, 3]
                for num_stages in NUM_STAGES:
                    for kn_pair in kn_pairs:
                        smem_config = get_smem_config(num_stages, kn_pair[0], block_m, block_n)
                        for tma_multicast_config in TMA_MULTICAST_CONFIGS:
                            cfg = {
                                "N": kn_pair[1],
                                "K": kn_pair[0],
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "SWIZZLE_D_MODE": smem_config[1],
                                "BLOCK_N_PADDING": smem_config[2],
                                "NUM_STAGES": num_stages,
                                "NUM_TMA_MULTICAST": tma_multicast_config[0],
                                "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
                                "IS_GROUPED_CONTIGUOUS": is_grouped_contiguous,
                                "IS_GROUPED_MASKED": is_grouped_masked,
                                "MOE_NUM_EXPERTS": moe_num_experts,
                            }
                            f.write(json.dumps(cfg) + "\n")
                            counter += 1

    return counter


def main(args):
    with open(os.path.join(args.model, "config.json"), "r") as f:
        model_cfg = json.load(f)
    logger.debug(
        f"TP Size: {args.tensor_parallel_size}, "
        f"EP Size: {args.expert_parallel_size}, "
        f"has shared experts: {args.has_shared_experts}"
    )
    logger.info(f"Configurations generated and saved to {args.output}")
    (
        gemm_kn_pairs,
        grouped_gemm_contiguous_kn_pairs,
        grouped_gemm_masked_kn_pairs,
    ) = generate_kn_pairs(args, model_cfg)
    logger.debug(f"GEMM KN pairs: {gemm_kn_pairs}")
    logger.debug(f"Grouped GEMM Contiguous KN pairs: {grouped_gemm_contiguous_kn_pairs}")
    logger.debug(f"Grouped GEMM Masked KN pairs: {grouped_gemm_masked_kn_pairs}")
    if len(gemm_kn_pairs) > 0:
        num_gemm = generate_json(
            gemm_kn_pairs,
            model_cfg["moe_num_experts"],
            args.output,
        )
        logger.info(f"Generated {num_gemm} gemm configuration.")
    if len(grouped_gemm_contiguous_kn_pairs) > 0:
        num_grouped_contiguous = generate_json(
            grouped_gemm_contiguous_kn_pairs,
            model_cfg["moe_num_experts"],
            args.output,
            is_grouped_contiguous=True,
        )
        logger.info(f"Generated {num_grouped_contiguous} grouped_gemm_contiguous configuration.")
    if len(grouped_gemm_masked_kn_pairs) > 0:
        num_grouped_masked = generate_json(
            grouped_gemm_masked_kn_pairs,
            model_cfg["moe_num_experts"],
            args.output,
            is_grouped_masked=True,
        )
        logger.info(f"Generated {num_grouped_masked} grouped_gemm_masked configuration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--expert-parallel-size",
        "--ep",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--has-shared-experts",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./deep_gemm_pre_compile_config.jsonl",
    )
    args = parser.parse_args()
    main(args)
