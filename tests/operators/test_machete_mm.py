# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re
import struct
import unittest

import numpy as np
import paddle
import paddle.nn.quant as Q
from paddle import base
from paddle.base import core
from paddle.framework import set_default_dtype

from fastdeploy.model_executor.ops.gpu import machete_mm, machete_prepack_B

np.random.seed(123)
paddle.seed(123)


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r"release (\S+),"
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split(".")
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack("<f", struct.pack("<I", np.uint32(x) << np.uint32(16)))[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


from typing import Optional

import numpy as np
import paddle


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def pack_rows(
    q_w: paddle.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == [size_k, size_n]

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.place
    q_w_np = q_w.numpy().astype(np.uint32)

    q_res = np.zeros((size_k // pack_factor, size_n), dtype=np.uint32)

    for i in range(pack_factor):
        q_res |= q_w_np[i::pack_factor, :] << num_bits * i

    q_res = paddle.to_tensor(q_res.astype(np.int32), place=orig_device)
    return q_res


def quantize_weights(
    w: paddle.Tensor,
    group_size: Optional[int],
    quant_type: str = "uint4b8",
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    """
    Quantize weights in PaddlePaddle, similar to PyTorch implementation.

    Args:
        w: Input weight tensor (must be float type).
        quant_type: Target quantization type (e.g., `uint4`, `uint4b8`).
        group_size: Group size for quantization. If `-1`, use channel-wise quantization.
        zero_points: Whether to compute zero points (only for unsigned quant types).
        ref_zero_points_after_scales: If True, apply zero points after scales in dequantization.

    Returns:
        w_ref: Dequantized reference weights.
        w_q: Quantized weights.
        w_s: Scales (None if `group_size` is None).
        maybe_w_zp: Zero points (None if `zero_points=False`).
    """
    assert paddle.is_floating_point(w), "w must be float type"
    assert quant_type in ["uint4", "uint4b8"], "only support quant_type = uint4, uint4b8"

    if zero_points:
        assert group_size is not None, "group_size must be provided for zero_points"

    orig_device = w.place
    # orig_type = w.dtype
    size_k, size_n = w.shape

    if group_size == -1:
        group_size = size_k

    # Reshape to [group_size, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape([-1, group_size, size_n])
        w = w.transpose([1, 0, 2])
        w = w.reshape([group_size, -1])

    # Compute scale for each group
    max_val = paddle.max(w, axis=0, keepdim=True)
    min_val = paddle.min(w, axis=0, keepdim=True)

    max_q_val = float(7.0)
    min_q_val = float(-8.0)

    w_s = paddle.ones([1], dtype=paddle.float32)  # unscaled case
    maybe_w_zp = None

    if group_size is not None:
        if zero_points:
            w_s = (max_val - min_val).clip(min=1e-5) / max_q_val
            maybe_w_zp = paddle.round(paddle.abs(min_val / w_s)).clip(min_q_val, max_q_val).astype(paddle.int32)
        else:
            # Avoid division by zero
            max_scale = paddle.maximum(
                paddle.abs(max_val / (max_q_val if max_q_val != 0 else float("inf"))),
                paddle.abs(min_val / (min_q_val if min_q_val != 0 else float("inf"))),
            )
            w_s = max_scale

    # Quantize
    w_q = paddle.round(w / w_s).astype(paddle.int32) + (maybe_w_zp if zero_points else 0)
    w_q = paddle.clip(w_q, min_q_val, max_q_val)
    # w_q = paddle.clip(w_q, min_q_val, max_q_val).astype(quant_type)

    # Compute ref (dequantized)
    # if ref_zero_points_after_scales and maybe_w_zp is not None:
    #     w_ref = w_q.astype(orig_type) * w_s - maybe_w_zp.astype(orig_type) * w_s
    # else:
    #     w_ref = (w_q.astype(orig_type) - (maybe_w_zp.astype(orig_type) if zero_points else 0)) * w_s

    # if hasattr(quant_type, 'bias'):  # Custom quantization bias (if applicable)
    # w_q += quant_type.bias
    if quant_type == "uint4b8":
        w_q += 8

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w_tensor):
            w_tensor = w_tensor.reshape([group_size, -1, size_n])
            w_tensor = w_tensor.transpose([1, 0, 2])
            w_tensor = w_tensor.reshape([size_k, size_n])
            return w_tensor

        w_q = reshape_w(w_q)
        # w_ref = reshape_w(w_ref)
        w_s = w_s.reshape([-1, size_n])

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape([-1, size_n])
        maybe_w_zp = maybe_w_zp.cpu() if orig_device.is_cpu_place() else maybe_w_zp.cuda()

    # Move tensors back to original device
    # w_ref = w_ref.to(orig_device)
    w_q = w_q.to(orig_device)
    if w_s is not None:
        w_s = w_s.to(orig_device)

    return w_q, w_s, maybe_w_zp


def maybe_convert_zeropoints(zps: Optional[paddle.Tensor], s: paddle.Tensor):
    return zps if zps is None else -1 * s * (zps.astype(s.dtype))


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase(unittest.TestCase):
    def config(self):
        self.dtype = "float16"
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = False
        self.batch = 1
        self.token = 512
        self.in_features = 7168
        self.out_features = 256
        self.weight_dtype = "int4"
        self.static = False
        self.group_size = -1

    def weightQuantizeCPUGPUConsistenceCheck(self, weight_float):
        for arch in [70, 75, 80, 86]:
            weight_gpu, weight_scale_gpu = Q.weight_quantize(
                (weight_float.cuda() if self.weight_dtype == "int8" else self.weight.cpu()),
                algo=("weight_only_int8" if self.weight_dtype == "int8" else "weight_only_int4"),
                arch=arch,
                group_size=self.group_size,
            )
            weight_cpu, weight_scale_cpu = Q.weight_quantize(
                weight_float.cpu(),
                algo=("weight_only_int8" if self.weight_dtype == "int8" else "weight_only_int4"),
                arch=arch,
                group_size=self.group_size,
            )
            np.testing.assert_allclose(
                weight_gpu.numpy(),
                weight_cpu.numpy(),
                atol=1.5,
                rtol=2,
            )
            np.testing.assert_allclose(
                weight_scale_gpu.numpy(),
                weight_scale_cpu.numpy(),
                atol=1e-5,
                rtol=1e-3,
            )
            pass
        pass

    def setUp(self):
        self.config()
        if self.dtype == "bfloat16" or self.weight_dtype == "int4":
            self.atol = 1.3e-1
        x = np.random.random((self.token, self.in_features))
        self.x = paddle.to_tensor(x, dtype=self.dtype)
        if self.bias:
            bias_attr = base.ParamAttr(
                trainable=False,
                regularizer=None,
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
        else:
            bias_attr = None
        set_default_dtype(self.dtype)
        self.linear = paddle.nn.Linear(self.in_features, self.out_features, bias_attr=bias_attr)

        self.bias = self.linear.bias
        self.weight = self.linear.weight
        self.float_weight = self.linear.weight
        self.weight_scale = None
        # check weight quantize
        self.weightQuantizeCPUGPUConsistenceCheck(self.float_weight)

        self.weight, self.weight_scale = Q.weight_quantize(
            (
                self.float_weight.cuda()
                # if self.weight_dtype == "int8"
                # else self.weight.cpu()
            ),
            algo=("weight_only_int8" if self.weight_dtype == "int8" else "weight_only_int4"),
            group_size=self.group_size,
        )

    def get_linear_out(self):
        out = self.linear(self.x)
        return out.numpy()

    def get_weight_only_linear_out(self):
        for i in range(10):
            out = Q.weight_only_linear(
                self.x,
                self.weight,
                bias=self.bias,
                weight_scale=self.weight_scale,
                weight_dtype=self.weight_dtype,
                group_size=self.group_size,
            )
        # print(out)
        return out.numpy()

    def get_machete_weight_only_linear_out(self):
        w_q, w_s, w_zp = quantize_weights(
            self.float_weight.cuda(),
            # self.w,
            # wtype,
            group_size=-1,
            zero_points=False,
        )

        # print(w_q)
        # print(w_s)

        # w_q = self.weight
        # w_s = self.weight_scale.reshape([-1, self.out_features])
        # print(w_q)
        # print(w_s)

        w_q = pack_rows(w_q, 4, *w_q.shape)
        w_q_col = w_q.transpose([1, 0]).contiguous()  # convert to col major

        # print(w_q)

        w_prepack = machete_prepack_B(
            w_q_col,
            self.dtype,
            "uint4b8",
            "",
        )

        for i in range(10):
            out = machete_mm(
                self.x,
                w_prepack,
                w_s,  # group scales
                None,  # group zeros
                None,  # per-channel scale
                None,  # per-token scale
                "uint4b8",  # weight_dtype
                "",  # out_dtype
                -1,  # group_size
                "",  # scheduler
            )
        # print(out)
        return out[0].numpy()

    def test_weight_only_linear(self):
        # out_expect = self.get_linear_out()
        out_real = self.get_weight_only_linear_out()
        out_machete = self.get_machete_weight_only_linear_out()
        # print(out_expect)
        # print(out_real)
        # print(out_machete)

        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            # out_expect = convert_uint16_to_float(out_expect)
            out_machete = convert_uint16_to_float(out_machete)
        np.testing.assert_allclose(out_real, out_machete, rtol=self.rtol, atol=self.atol)


M = [32, 64, 128, 256, 512, 1024, 2048]
# M = [1024]
# K_N = [[7168, 1536], [2048, 5120], [4096, 2048]]
K_N = [[7168, 256]]


def make_case(m, k, n):
    class Case(WeightOnlyLinearTestCase):
        def config(self, _m=m, _k=k, _n=n):
            super().config()
            self.token = m
            self.in_features = k
            self.out_features = n

    Case.name = f"WeightOnlyLinearTestCase{m}{k}{n}"
    return Case


for k, n in K_N:
    for m in M:
        print(m, n, k)
        cls = make_case(m, k, n)
        globals()[cls.name] = cls

if __name__ == "__main__":
    unittest.main()
