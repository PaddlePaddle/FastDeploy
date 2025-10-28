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
import threading
from queue import Queue
from time import time

import paddle
from tqdm import tqdm

from fastdeploy.model_executor.ops.gpu.deep_gemm.jit.compiler import build
from fastdeploy.model_executor.ops.gpu.deep_gemm.jit.template import (
    cpp_format,
    generate,
)
from fastdeploy.model_executor.ops.gpu.deep_gemm.jit_kernels.gemm import (
    includes as gemm_includes,
)
from fastdeploy.model_executor.ops.gpu.deep_gemm.jit_kernels.gemm import (
    template as gemm_template,
)
from fastdeploy.model_executor.ops.gpu.deep_gemm.jit_kernels.m_grouped_gemm import (
    includes as m_grouped_includes,
)
from fastdeploy.model_executor.ops.gpu.deep_gemm.jit_kernels.m_grouped_gemm import (
    template as m_grouped_template,
)

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
logger.setLevel(os.getenv("PRE_COMPILE_LOG_LEVEL", "INFO"))


class CompileWorker(threading.Thread):
    def __init__(self, queue, pbar):
        super().__init__()
        self.queue = queue
        self.pbar = pbar

    def run(self):
        while True:
            cfg = self.queue.get()
            if cfg is None:
                break

            try:
                logger.debug(f"Compiling for config: {cfg}")
                keys = {
                    "N": cfg["N"],
                    "K": cfg["K"],
                    "BLOCK_M": cfg["BLOCK_M"],
                    "BLOCK_N": cfg["BLOCK_N"],
                    "SWIZZLE_D_MODE": cfg["SWIZZLE_D_MODE"],
                    "BLOCK_N_PADDING": cfg["BLOCK_N_PADDING"],
                    "NUM_STAGES": cfg["NUM_STAGES"],
                    "NUM_TMA_MULTICAST": cfg["NUM_TMA_MULTICAST"],
                    "IS_TMA_MULTICAST_ON_A": cfg["IS_TMA_MULTICAST_ON_A"],
                }
                arg_defs = (
                    ("lhs", paddle.float8_e4m3fn),
                    ("lhs_scales", paddle.float32),
                    ("rhs", paddle.float8_e4m3fn),
                    ("rhs_scales", paddle.float32),
                    ("out", paddle.bfloat16),
                    ("m", int),
                    ("stream", paddle.device.cuda.Stream),
                    ("num_sms", int),
                    ("smem_size", int),
                )
                name = "gemm_fp8_fp8_bf16_nt"
                includes = gemm_includes
                template = gemm_template
                if cfg["IS_GROUPED_CONTIGUOUS"]:
                    keys["GEMM_TYPE"] = "GroupedContiguous"
                    arg_defs = (
                        ("lhs", paddle.float8_e4m3fn),
                        ("lhs_scales", paddle.float32),
                        ("rhs", paddle.float8_e4m3fn),
                        ("rhs_scales", paddle.float32),
                        ("out", paddle.bfloat16),
                        ("grouped_layout", paddle.int32),
                        ("m", int),
                        ("num_groups", int),
                        ("stream", paddle.device.cuda.Stream),
                        ("num_sms", int),
                        ("smem_size", int),
                    )
                if cfg["IS_GROUPED_MASKED"]:
                    keys["GEMM_TYPE"] = "GroupedMasked"
                    arg_defs = (
                        ("lhs", paddle.float8_e4m3fn),
                        ("lhs_scales", paddle.float32),
                        ("rhs", paddle.float8_e4m3fn),
                        ("rhs_scales", paddle.float32),
                        ("out", paddle.bfloat16),
                        ("grouped_layout", paddle.int32),
                        ("m", int),
                        ("stream", paddle.device.cuda.Stream),
                        ("num_sms", int),
                        ("smem_size", int),
                    )
                if cfg["IS_GROUPED_CONTIGUOUS"] or cfg["IS_GROUPED_MASKED"]:
                    keys["NUM_GROUPS"] = int(cfg["MOE_NUM_EXPERTS"] / cfg["EXPERT_PARALLEL"])
                    includes = m_grouped_includes
                    template = m_grouped_template
                    name = "m_grouped_gemm_fp8_fp8_bf16_nt"

                code = generate(includes, arg_defs, cpp_format(template, keys))
                build(name, arg_defs, code)
            except Exception as e:
                logger.error(f"Failed to compile config {cfg}: {e!s}")
                raise RuntimeError(e)
            finally:
                self.pbar.update(1)
                self.queue.task_done()


def pre_compile_from_config(config_file: str, num_threads: int, expert_parallel: int):
    with open(config_file, "r") as f:
        start_time = time()
        lines = f.readlines()

        queue = Queue()
        pbar = tqdm(total=len(lines), desc="Compiling")
        workers = []
        for _ in range(num_threads):
            worker = CompileWorker(queue, pbar)
            worker.start()
            workers.append(worker)

        for line in lines:
            cfg = json.loads(line)
            cfg["EXPERT_PARALLEL"] = expert_parallel
            queue.put(cfg)

        queue.join()

        for _ in range(num_threads):
            queue.put(None)
        for worker in workers:
            worker.join()

        pbar.close()

        logger.info(f"Total compilation time: {time() - start_time:.2f} seconds")


def main(args):
    pre_compile_from_config(args.config_file, args.num_threads, args.expert_parallel_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        default="./deep_gemm_pre_compile_config.jsonl",
    )
    parser.add_argument(
        "--expert-parallel-size",
        "--ep",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
    )
    args = parser.parse_args()
    main(args)
