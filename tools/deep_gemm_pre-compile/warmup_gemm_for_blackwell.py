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
import os

import paddle

from fastdeploy.model_executor.layers.quantization.fp8_utils import (
    deep_gemm,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
logger.setLevel(os.getenv("PRE_COMPILE_LOG_LEVEL", "INFO"))


def get_sm():
    """
    get_sm
    """
    prop = paddle.device.cuda.get_device_properties()
    return prop.major * 10 + prop.minor


def generate_m_grouped_fp8_gemm_nt_contiguous(cfg, use_ue8m0: bool = True, align: int = 128):
    max_m = cfg["max_m"]
    n = cfg["n"]
    k = cfg["k"]
    num_groups = cfg["num_groups"]

    a = paddle.randn([max_m, k], dtype=paddle.bfloat16)

    a_fp8, a_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        a,
        using_pow2_scale=use_ue8m0,
        output_scale_transpose=use_ue8m0,
        using_ue8m0_scale=use_ue8m0,
    )
    a_scale = a_scale.T[: a_fp8.shape[0]]

    bs = []
    b_scales = []
    for _ in range(num_groups):
        b = paddle.randn([n, k], dtype=paddle.bfloat16)
        b_fp8, b_scale = quant_weight_ue8m0(b, [align, align])
        bs.append(b_fp8)
        b_scales.append(b_scale)

    b_fp8 = paddle.concat([tmp.unsqueeze(0) for tmp in bs], axis=0)
    b_scale = paddle.concat([tmp.unsqueeze(0) for tmp in b_scales], axis=0)

    y = paddle.empty([max_m, n], dtype=paddle.bfloat16)
    m_indices = paddle.arange(num_groups, dtype=paddle.int32)
    npart = (max_m + num_groups - 1) // num_groups
    m_indices = paddle.concat([m_indices] * npart, axis=0)

    for i in range(0, max_m + align, align):
        if i == 0:
            continue

        config = {
            "m": i,
            "m_group": num_groups,
            "n": n,
            "k": k,
            "use_ue8m0": use_ue8m0,
        }
        print("contiguous:", json.dumps(config))

        a_scale_tmp = a_scale[:i].T.contiguous().T
        # breakpoint()
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_fp8[:i], a_scale_tmp),
            (b_fp8, b_scale),
            y[:i],
            m_indices=m_indices[:i],
        )


def generate_m_grouped_fp8_gemm_nt_masked(cfg, use_ue8m0: bool = True, align: int = 128):
    max_m = cfg["max_m"]
    n = cfg["n"]
    k = cfg["k"]
    num_groups = cfg["num_groups"]

    d = paddle.empty([num_groups, max_m, n], dtype=paddle.bfloat16)

    mask_d = paddle.randint(0, max_m, [num_groups, 1], dtype=paddle.int32)

    a_s = []
    a_scales = []
    for _ in range(num_groups):
        a = paddle.randn([max_m, k], dtype=paddle.bfloat16)
        a_fp8, a_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            a,
            using_pow2_scale=use_ue8m0,
            output_scale_transpose=use_ue8m0,
            using_ue8m0_scale=use_ue8m0,
        )
        a_scale = a_scale.T
        a_s.append(a_fp8)
        a_scales.append(a_scale)
    a_fp8 = paddle.concat([tmp.unsqueeze(0) for tmp in a_s], axis=0)
    a_scale = paddle.stack(a_scales, axis=0).transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])

    bs = []
    b_scales = []
    for _ in range(num_groups):
        b = paddle.randn([n, k], dtype=paddle.bfloat16)
        b_fp8, b_scale = quant_weight_ue8m0(b, [align, align])
        bs.append(b_fp8)
        b_scales.append(b_scale)

    b_fp8 = paddle.concat([tmp.unsqueeze(0) for tmp in bs], axis=0)
    b_scale = paddle.concat([tmp.unsqueeze(0) for tmp in b_scales], axis=0)
    config = {
        "max_m": max_m,
        "m_group": num_groups,
        "n": n,
        "k": k,
        "use_ue8m0": use_ue8m0,
    }
    print("masked:", json.dumps(config))
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (a_fp8, a_scale), (b_fp8, b_scale), d, mask_d, max_m, disable_ue8m0_cast=not use_ue8m0
    )


def generate_fp8_gemm_nt(cfg, use_ue8m0: bool = True, align: int = 128):
    max_m = cfg["max_m"]
    n = cfg["n"]
    k = cfg["k"]

    a = paddle.randn([max_m, k], dtype=paddle.bfloat16)
    d = paddle.empty([max_m, n], dtype=paddle.bfloat16)

    a_fp8, a_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        a,
        using_pow2_scale=use_ue8m0,
        output_scale_transpose=True,
        using_ue8m0_scale=use_ue8m0,
    )
    a_scale = a_scale.T[: a_fp8.shape[0], ...]
    b = paddle.randn((n, k), dtype=paddle.bfloat16)

    b_fp8, b_scale = quant_weight_ue8m0(b, [align, align])
    if use_ue8m0:
        b_scale = transform_scale_ue8m0(
            b_scale,
            mn=n,
            weight_block_size=[align, align],
        )

    for m in range(0, max_m + align, align):
        a_scale_tmp = a_scale[:m].T.contiguous().T
        print(
            "gemm:",
            json.dumps(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "a:": a_fp8[:m].shape,
                    "a_scale:": a_scale_tmp[:m].shape,
                    "b_fp8:": b_fp8.shape,
                    "b_scale:": b_scale.shape,
                }
            ),
        )
        deep_gemm.fp8_gemm_nt(
            (a_fp8[:m], a_scale_tmp),
            (b_fp8, b_scale),
            d[:m],
        )


def generate_kernel(config_list):
    for cfg in config_list:
        if cfg["kernel_name"] == "m_grouped_fp8_gemm_nt_contiguous":
            generate_m_grouped_fp8_gemm_nt_contiguous(cfg)
        elif cfg["kernel_name"] == "m_grouped_fp8_gemm_nt_masked":
            generate_m_grouped_fp8_gemm_nt_masked(cfg)
        elif cfg["kernel_name"] == "fp8_gemm_nt":
            generate_fp8_gemm_nt(cfg)
        else:
            print(f"invaild kernel:{cfg['kernel_name']}")


def main(args):
    with open(os.path.join(args.model, "config.json"), "r") as f:
        model_cfg = json.load(f)
    chunk_size = args.chunk_size
    hidden_size = model_cfg["hidden_size"]
    intermediate_size = model_cfg["intermediate_size"]
    moe_intermediate_size = model_cfg["moe_intermediate_size"]
    num_attention_heads = model_cfg["num_attention_heads"]
    num_key_value_heads = model_cfg["num_key_value_heads"]
    head_dim = int(hidden_size / num_attention_heads)
    tp_size = args.tensor_parallel_size
    ep_size = args.expert_parallel_size
    has_shared_experts = args.has_shared_experts.lower() == "true"

    local_num_experts = int(model_cfg["moe_num_experts"] / ep_size)
    max_m_in_moe = int(args.chunk_size * ep_size / tp_size * local_num_experts)
    max_m_per_expert = int(args.chunk_size * ep_size / tp_size)

    config_list = [
        {"kernel_name": "fp8_gemm_nt", "max_m": chunk_size, "n": hidden_size, "k": int(intermediate_size / tp_size)},
        {
            "kernel_name": "fp8_gemm_nt",
            "max_m": chunk_size,
            "n": int(head_dim * (num_attention_heads + num_key_value_heads * 2) / tp_size),
            "k": hidden_size,
        },
        {
            "kernel_name": "fp8_gemm_nt",
            "max_m": chunk_size,
            "n": int(intermediate_size * 2 / tp_size),
            "k": hidden_size,
        },
        # {"kernel_name": "fp8_gemm_nt", "max_m": chunk_size, "n": hidden_size, "k": int(hidden_size / tp_size)},
        {
            "kernel_name": "m_grouped_fp8_gemm_nt_contiguous",
            "max_m": max_m_in_moe,
            "n": hidden_size,
            "k": int(moe_intermediate_size),
            "num_groups": local_num_experts,
        },
        {
            "kernel_name": "m_grouped_fp8_gemm_nt_contiguous",
            "max_m": max_m_in_moe,
            "n": int(moe_intermediate_size * 2),
            "k": hidden_size,
            "num_groups": local_num_experts,
        },
        {
            "kernel_name": "m_grouped_fp8_gemm_nt_masked",
            "max_m": max_m_per_expert,
            "n": hidden_size,
            "k": int(moe_intermediate_size),
            "num_groups": local_num_experts,
        },
        {
            "kernel_name": "m_grouped_fp8_gemm_nt_masked",
            "max_m": max_m_per_expert,
            "n": int(moe_intermediate_size * 2),
            "k": hidden_size,
            "num_groups": local_num_experts,
        },
    ]
    if has_shared_experts:
        config_list.append(
            {"kernel_name": "fp8_gemm_nt", "max_m": chunk_size, "n": hidden_size, "k": int(moe_intermediate_size * 2)}
        )
        config_list.append(
            {"kernel_name": "fp8_gemm_nt", "max_m": chunk_size, "n": int(moe_intermediate_size * 4), "k": hidden_size}
        )
    generate_kernel(config_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=131072,
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
    args = parser.parse_args()
    main(args)
