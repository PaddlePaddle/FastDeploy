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

from paddleformers.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "DFNRopeVisionTransformerConfig",
]


# qwen2_5 视觉参数
"""
  "vision_config": {
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "window_size": 112,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "tokens_per_second": 2,
    "temporal_patch_size": 2
  },
"""


# qwen:
#   hidden_size -> embed_dim
#   out_hidden_size -> hidden_size
#   intermediate_size -> qwen_vision_block 中 mlp/mlp_hidden_dim
#   fullatt_block_indexes 区分vit部分不同attention的layer_index
#   spatial_patch_size 和 tokens_per_second 在vllm中没用到
class DFNRopeVisionTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ErnieModel`]. It is used to instantiate an Ernie
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "DFNRope_vision_transformer"

    def __init__(
        self,
        depth=32,
        hidden_size=1280,
        out_hidden_size=3584,
        intermediate_size=3420,
        hidden_act="silu",
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
        fullatt_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.temporal_patch_size = temporal_patch_size
