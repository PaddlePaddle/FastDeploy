# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


import sys
import types
import unittest
from unittest import mock

import paddle

from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
    BlockWiseFP8MoEMethod,
)
from fastdeploy.model_executor.layers.quantization.block_wise_fp8 import (
    BlockWiseFP8Config,
    BlockWiseFP8LinearMethod,
)


class DummyLinearLayer(paddle.nn.Layer):
    def __init__(self, fd_config, weight_shape, with_bias=False):
        super().__init__()
        self.weight_shape = weight_shape
        self.with_bias = with_bias
        self.bias = None
        self.fd_config = fd_config
        self.weight = paddle.randn(self.weight_shape, paddle.bfloat16)


class DummyFusedMoELayer(paddle.nn.Layer):
    def __init__(self, fd_config, num_local_experts, moe_intermediate_size, hidden_size):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_size = hidden_size
        self.gate_correction_bias = paddle.zeros([1], dtype=paddle.float32)
        self.top_k = 1
        self.ep_size = 1
        self.ep_rank = 0
        self.tp_size = 1
        self.tp_rank = 0
        self.fd_config = fd_config

        self.up_gate_proj_weight = paddle.randn(
            [num_local_experts, hidden_size, moe_intermediate_size * 2], dtype="bfloat16"
        )
        self.down_proj_weight = paddle.randn([num_local_experts, moe_intermediate_size, hidden_size], dtype="bfloat16")


class TestFP8LinearWithUe8m0Scale(unittest.TestCase):
    def setUp(self):
        self.quant_config = BlockWiseFP8Config(weight_block_size=[128, 128], is_checkpoint_bf16=True)
        self.quant_config.deepgemm_scale_ue8m0 = True  # set deepgemm_scale_ue8m0 to True

    def test_create_layer_with_ue8m0_scale(self):
        def fake_per_block_cast_to_fp8(x, use_ue8m0=True):
            out_w = x.astype(paddle.float8_e4m3fn)
            out_s = paddle.ones([(x.shape[0] // 128), (x.shape[1] // 128)], dtype=paddle.float32)
            return out_w, out_s

        fd_config = mock.MagicMock()
        fd_config.load_config.load_choices.return_value = "default_v1"
        layer = DummyLinearLayer(fd_config=fd_config, weight_shape=[128, 1024])
        method = BlockWiseFP8LinearMethod(quant_config=self.quant_config)

        if "fastdeploy.model_executor.ops.gpu.deep_gemm.utils" in sys.modules:
            # This is for sm90, which DeepGEMM does not support ue8m0 scale
            fake = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm")
            fake2 = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm.utils.math")
            fake2.per_block_cast_to_fp8 = fake_per_block_cast_to_fp8
            fake3 = types.ModuleType("deep_gemm")
            fake4 = types.ModuleType("deep_gemm.utils")
            fake4.align = lambda x, y: (x + y - 1) // y * y
            fake4.get_tma_aligned_size = lambda x, y: (x + 16 // y - 1) // (16 // y) * (16 // y)
            sys.modules["deep_gemm"] = fake3
            sys.modules["deep_gemm.utils"] = fake4

            deep_gemm_utils = sys.modules["fastdeploy.model_executor.ops.gpu.deep_gemm.utils"]
            fake.utils = deep_gemm_utils
            deep_gemm_utils.math = fake2
            fake3.utils = fake4

        method.model_format = "torch"
        method.process_weights_after_loading(layer)
        self.assertTrue(layer.weight_scale_inv.dtype == paddle.int32)
        self.assertEqual(layer.weight_scale_inv.shape, [128, 2])  # 1024 / 128 / 4


class TestFP8FusedMoeWithUe8m0Scale(unittest.TestCase):
    def setUp(self):
        self.quant_config = BlockWiseFP8Config(weight_block_size=[128, 128], is_checkpoint_bf16=True)
        self.quant_config.deepgemm_scale_ue8m0 = True  # set deepgemm_scale_ue8m0 to True

    def test_create_layer_with_ue8m0_scale(self):
        def fake_per_block_cast_to_fp8(x, use_ue8m0=True):
            out_w = x.astype(paddle.float8_e4m3fn)
            out_s = paddle.ones([(x.shape[0] // 128), (x.shape[1] // 128)], dtype=paddle.float32)
            return out_w, out_s

        fd_config = mock.MagicMock()
        fd_config.load_config.load_choices.return_value = "default_v1"
        layer = DummyFusedMoELayer(
            fd_config=fd_config, num_local_experts=1, moe_intermediate_size=256, hidden_size=256
        )
        method = BlockWiseFP8MoEMethod(quant_config=self.quant_config)

        method.up_gate_proj_weight_shape = [1, 512, 256]
        method.down_proj_weight_shape = [1, 256, 256]
        method.up_gate_proj_scale_shape = [1, 512, 1]
        method.down_proj_scale_shape = [1, 256, 1]

        if "fastdeploy.model_executor.ops.gpu.deep_gemm.utils" in sys.modules:
            # This is for sm90, which DeepGEMM does not support ue8m0 scale
            fake = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm")
            fake2 = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm.utils.math")
            fake2.per_block_cast_to_fp8 = fake_per_block_cast_to_fp8
            fake3 = types.ModuleType("deep_gemm")
            fake4 = types.ModuleType("deep_gemm.utils")
            fake4.align = lambda x, y: (x + y - 1) // y * y
            fake4.get_tma_aligned_size = lambda x, y: (x + 16 // y - 1) // (16 // y) * (16 // y)
            sys.modules["deep_gemm"] = fake3
            sys.modules["deep_gemm.utils"] = fake4

            deep_gemm_utils = sys.modules["fastdeploy.model_executor.ops.gpu.deep_gemm.utils"]
            fake.utils = deep_gemm_utils
            deep_gemm_utils.math = fake2
            fake3.utils = fake4

        method.model_format = "torch"
        method.process_weights_after_loading(layer)

        self.assertTrue(layer.down_proj_weight_scale_inv.dtype == paddle.int32)
        self.assertEqual(layer.down_proj_weight_scale_inv.shape, method.down_proj_scale_shape)


if __name__ == "__main__":
    unittest.main()
