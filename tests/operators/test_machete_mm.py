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

from fastdeploy.model_executor.layers.quantization.ops import (
    machete_quantize_and_pack,
    machete_wint_mm,
)

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


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack("<f", struct.pack("<I", np.uint32(x) << np.uint32(16)))[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_sm_version() < 90,
    "machete only support sm90.",
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
        self.out_features = 1024
        self.weight_dtype = "int4"
        self.static = False
        self.group_size = -1

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

        self.weight, self.weight_scale = Q.weight_quantize(
            (self.float_weight.cuda() if self.weight_dtype == "int8" else self.weight.cpu()),
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
        return out.numpy()

    def get_machete_weight_only_linear_out(self):
        w_q, w_s = machete_quantize_and_pack(
            w=self.float_weight.cuda(),
            atype=self.dtype,
            quant_type="uint4b8",
        )

        out = machete_wint_mm(
            self.x,
            w_prepack=w_q,
            w_g_s=w_s,  # group scales
            weight_dtype="uint4b8",  # weight_dtype
        )
        return out.numpy()

    def test_weight_only_linear(self):
        # out_expect = self.get_linear_out()
        out_paddle = self.get_weight_only_linear_out()
        out_machete = self.get_machete_weight_only_linear_out()

        if self.dtype == "bfloat16":
            out_paddle = convert_uint16_to_float(out_paddle)
            # out_expect = convert_uint16_to_float(out_expect)
            out_machete = convert_uint16_to_float(out_machete)
        np.testing.assert_allclose(out_paddle, out_machete, rtol=self.rtol, atol=self.atol)


M = [32, 128]
K_N = [[2048, 4096]]


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
        cls = make_case(m, k, n)
        globals()[cls.name] = cls

if __name__ == "__main__":
    unittest.main()
