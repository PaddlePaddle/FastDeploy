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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import fused_rms_norm_xpu

# from paddle.incubate.nn.functional import fused_rms_norm


def find_max_diff(arr1, arr2):
    """找出两个数组元素差值的最大值及其索引
    返回:
        max_diff (float): 最大绝对值差
        index (tuple): 最大值的位置索引
        actual_diff (float): 实际差值（带符号）
    """
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    flat_idx = np.argmax(abs_diff)
    idx = np.unravel_index(flat_idx, arr1.shape)
    return abs_diff[idx], idx, diff[idx], arr1[idx], arr2[idx]


def naive_rmsnorm(
    x,
    gamma,
    beta=None,
    epsilon=1e-6,
    begin_norm_axis=1,
    bias=None,
    residual=None,
):
    residual_out = None
    if bias is not None:
        x = x + bias
    if residual is not None:
        x = x + residual
        residual_out = x
    variance = (x * x).mean(axis=-1)
    out = np.expand_dims(1.0 / np.sqrt(variance + epsilon), axis=-1) * x
    out = out * gamma
    if beta is not None:
        out = out + beta
    return out, residual_out


def run_and_compare(x_in, residual, bias, norm_weight):
    x_in_pd = paddle.to_tensor(x_in).astype(data_type)
    residual_pd = None
    if residual is not None:
        residual_pd = paddle.to_tensor(residual).astype(data_type)
    bias_pd = paddle.to_tensor(bias).astype(data_type)
    norm_weight_pd = paddle.to_tensor(norm_weight).astype(data_type)
    # norm_bias_pd = paddle.to_tensor(norm_bias).astype(data_type)

    out_np, residual_out_np = naive_rmsnorm(x_in, norm_weight, None, epsilon, begin_norm_axis, bias, residual)
    out_pd, residual_out_pd = fused_rms_norm_xpu(
        x_in_pd,
        bias_pd,
        residual_pd,
        norm_weight_pd,
        None,  # norm_bias_pd,
        epsilon,
        begin_norm_axis,
        -1,
        0,
        0,
        0,
    )
    """
    out_pd1, residual_out_pd1 = fused_rms_norm(
        x_in_pd,
        norm_weight=norm_weight_pd,
        norm_bias=norm_bias_pd,
        epsilon=epsilon,
        begin_norm_axis=1,
        bias=bias_pd,
        residual=residual_pd,
        quant_scale=-1,
        quant_round_type=0,
        quant_max_bound=0,
        quant_min_bound=0,
    )
    """
    abs_diff, idx, diff, val1, val2 = find_max_diff(out_np, out_pd.astype("float32").numpy())
    print(f"out compare: abs_diff={abs_diff}, index={idx}, diff={diff}, {val1} vs {val2}")
    assert np.allclose(out_np, out_pd.astype("float32").numpy(), rtol=1e-5, atol=1e-5)

    if residual is not None:
        abs_diff, idx, diff, val1, val2 = find_max_diff(residual_out_np, residual_out_pd.astype("float32").numpy())
        print(f"residual_out compare: abs_diff={abs_diff}, index={idx}, diff={diff}, {val1} vs {val2}")
        assert np.allclose(
            residual_out_np,
            residual_out_pd.astype("float32").numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    seed = np.random.randint(0, 1e8)
    print(f"numpy random seed is {seed}")
    np.random.seed(seed)

    m = 7
    n = 8192
    epsilon = 1e-5
    begin_norm_axis = 1
    data_type = "float32"

    x_in = (np.random.random([m, n]) - 0.5).astype("float32")
    residual = (np.random.random([m, n]) - 0.5).astype("float32")
    bias = (np.random.random([n]) - 0.5).astype("float32")
    norm_weight = (np.random.random([n]) - 0.5).astype("float32")
    # norm_bias = np.zeros([n]).astype("float32")
    # norm_bias = (np.random.random([n]) - 0.5).astype("float32")
    x_in_pd = paddle.to_tensor(x_in).astype(data_type)
    residual_pd = paddle.to_tensor(residual).astype(data_type)
    bias_pd = paddle.to_tensor(bias).astype(data_type)
    norm_weight_pd = paddle.to_tensor(norm_weight).astype(data_type)
    # norm_bias_pd = paddle.to_tensor(norm_bias).astype(data_type)

    run_and_compare(x_in, residual, bias, norm_weight)
    run_and_compare(x_in, None, bias, norm_weight)
