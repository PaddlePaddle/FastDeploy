"""
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
"""

from abc import abstractmethod

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantMethodBase


class FusedMoEMethodBase(QuantMethodBase):
    """
    All MoE Method should inherit this class.
    and must implement following methods!

    """

    @abstractmethod
    def create_weights(self,
                       layer: nn.Layer,
                       moe_compute_params,
                       ffn1_tensor,
                       ffn2_tensor,
                       ffn1_bias=None,
                       ffn2_bias=None):
        """
        How to create weights, you must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Layer,
        moe_compute_params,
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Compute methods, you must implement this method.
        """

        raise NotImplementedError
