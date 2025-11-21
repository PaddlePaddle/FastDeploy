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

import math
from collections import OrderedDict

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn


class NewGELUActivation(nn.Layer):
    """Google BERT style GELU."""

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5 * input * (1.0 + paddle.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * paddle.pow(input, 3.0))))
        )


class GELUActivation(nn.Layer):
    """Original GELU implementation."""

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        self.act = self._gelu_python if use_gelu_python else nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + paddle.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class FastGELUActivation(nn.Layer):
    """Fast GELU approximation."""

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + paddle.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Layer):
    """Quick GELU approximation."""

    def forward(self, input: Tensor) -> Tensor:
        return input * F.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Layer):
    """Clipped GELU used by some quantized models."""

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return paddle.clip(gelu(x), self.min, self.max)


class SiLUActivation(nn.Layer):
    """SiLU / Swish activation."""

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input)


class MishActivation(nn.Layer):
    """Mish activation."""

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input)


class LinearActivation(nn.Layer):
    """Identity activation."""

    def forward(self, input: Tensor) -> Tensor:
        return input


class ClassInstantier(OrderedDict):
    """Instantiates layers lazily when accessed."""

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_tanh": (nn.GELU, {"approximate": "tanh"}),
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation_fn(hidden_act: str):
    if hidden_act == "gelu_pytorch_tanh":
        return ACT2FN["gelu_tanh"]
    if hidden_act in ACT2FN:
        return ACT2FN[hidden_act]
    raise KeyError(f"function {hidden_act} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation_fn("gelu_python")
gelu_new = get_activation_fn("gelu_new")
gelu = get_activation_fn("gelu")
gelu_fast = get_activation_fn("gelu_fast")
quick_gelu = get_activation_fn("quick_gelu")
silu = get_activation_fn("silu")
mish = get_activation_fn("mish")
linear_act = get_activation_fn("linear")
