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

import copy

from fastdeploy.config import ModelConfig

from .dfnrope.modeling import DFNRopeVisionTransformerConfig

__all__ = [
    "Ernie4_5_VLMoeConfig",
]


class Ernie4_5_VLMoeConfig(ModelConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ErnieModel`]. It is used to instantiate an Ernie
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Ernie model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ErnieModel`] or [`~TFErnieModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from paddleformers.transformer import ErnieModel, ErnieConfig

    >>> # Initializing a Ernie ernie-7b style configuration
    >>> configuration = ErnieConfig()

    >>> # Initializing a model from the ernie-7b style configuration
    >>> model = ErnieModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "erniemoevl"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }

    def __init__(
        self,
        vision_config=None,
        im_patch_id=None,
        pixel_hidden_size=None,  # None for fuyu
        modality_detach=False,
        temporal_conv_size=2,
        spatial_conv_size=2,
        mm_vocab_size=0,  # vocab for mm specialtokens
        max_text_id=None,
        use_temporal_conv=True,
        moe_use_size_all2all=False,
        moe_num_attn_experts=False,
        moe_dense_experts_token_type_id: int = 3,
        moe_use_hard_gate: bool = True,
        moe_fuse_experts: bool = False,
        moe_use_token_type_bias: bool = False,
        disable_ffn_model_parallel=False,
        fuse_attn_ffn=True,
        rope_3d=True,
        freq_allocation=20,
        using_precision_check=False,
        use_recompute_resampler=False,
        resampler_fuse_rms_norm=False,
        moe_layer_feed_fake_token=False,
        moe_num_experts=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = DFNRopeVisionTransformerConfig(
            **vision_config) if vision_config else None
        self.im_patch_id = im_patch_id
        self.pixel_hidden_size = pixel_hidden_size
        self.modality_detach = modality_detach
        self.temporal_conv_size = temporal_conv_size
        self.spatial_conv_size = spatial_conv_size
        self.mm_vocab_size = mm_vocab_size
        self.max_text_id = max_text_id
        self.use_temporal_conv = use_temporal_conv

        self.moe_use_size_all2all = moe_use_size_all2all
        self.moe_num_attn_experts = moe_num_attn_experts
        self.moe_dense_experts_token_type_id = moe_dense_experts_token_type_id
        self.moe_use_hard_gate = moe_use_hard_gate
        self.moe_fuse_experts = moe_fuse_experts
        self.moe_use_token_type_bias = moe_use_token_type_bias
        self.disable_ffn_model_parallel = disable_ffn_model_parallel

        self.fuse_attn_ffn = fuse_attn_ffn
        self.rope_3d = rope_3d
        self.freq_allocation = freq_allocation
        self.using_precision_check = using_precision_check
        self.use_recompute_resampler = use_recompute_resampler
        self.resampler_fuse_rms_norm = resampler_fuse_rms_norm
        self.moe_layer_feed_fake_token = moe_layer_feed_fake_token
        self.moe_num_experts = moe_num_experts

    @property
    def multimodel_experts(self) -> bool:
        """是否有多种类型的experts."""
        return isinstance(self.moe_num_experts,
                          (tuple, list)) and len(self.moe_num_experts) > 1

    @property
    def use_moe(self) -> bool:
        """
        Check if model is using MoE architecture.

        Returns:
            bool: True if moe_num_experts > 0, False otherwise
        """
        return sum(
            self.moe_num_experts
        ) > 0 if self.multimodel_experts else self.moe_num_experts > 0

    def to_dict(self, saving_file=False):
        """to_dict"""
        output = copy.deepcopy(self.__dict__)
        if self.vision_config:
            output["vision_config"] = (
                self.vision_config.to_diff_dict() if isinstance(
                    self.vision_config,
                    (DFNRopeVisionTransformerConfig)) else self.vision_config)

        output["model_type"] = self.__class__.model_type
        return output
