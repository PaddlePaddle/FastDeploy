# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""tune_cublaslt_gemm"""
import paddle

from fastdeploy.utils import llm_logger as logger


def tune_cublaslt_int8_gemm(
    ns: list,
    ks: list,
    m_min: int = 32,
    m_max: int = 32768,
    dtype="int8",
    is_test=True,
    is_read_from_file=False,
    path="./cublaslt_gemm_search.csv",
):
    """
    tune cublaslt int8 gemm performance
    """
    K_tensor = paddle.to_tensor(ks)
    N_tensor = paddle.to_tensor(ns)

    try:
        from fastdeploy.model_executor.ops.gpu import tune_cublaslt_gemm
    except ImportError:
        logger.warning("From fastdeploy.model_executor.ops.gpu import tune_cublaslt_gemm Failed!")
        return

    tune_cublaslt_gemm(
        K_tensor,
        N_tensor,
        m_min,
        m_max,
        dtype,
        is_test,
        is_read_from_file,
        path,
    )
