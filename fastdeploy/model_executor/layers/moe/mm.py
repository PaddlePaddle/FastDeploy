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

import os
import paddle
from paddle import nn
from fastdeploy.model_executor.layers.moe.moe import MoELayer
from fastdeploy.model_executor.layers.utils import get_tensor


class TextMoELayer(MoELayer):
    """
    MoELayer is a layer that performs MoE (Mixture of Experts) computation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
            初始化函数，用于设置类的属性和方法。
        参数：
            - args (tuple, optional): 可变长度的位置参数列表，默认为空元组。
            - kwargs (dict, optional): 关键字参数字典，默认为空字典。
        返回值：
            无返回值，直接修改类的属性和方法。
        """
        kwargs["moe_tag"] = "Text"
        super().__init__(*args, **kwargs)

    def load_gate_state_dict(self, state_dict):
        """
            加载门状态字典，用于初始化网络参数。
        将从给定的状态字典中弹出的参数赋值给网络的门参数。

        Args:
            state_dict (OrderedDict): 包含网络门参数的字典。

        Returns:
            tuple (list, list): 返回两个列表，分别代表上阶网关投影和下阶投影的参数。
                每个元素都是一个列表，长度为网络的专家数量。
        """
        up_gate_proj_weight = []
        up_gate_proj_weight_scale = []
        down_proj_weight = []
        down_proj_weight_scale = []
        for j in range(0, self.num_experts):
            up_gate_proj_weight.append(
                get_tensor(state_dict.pop(self.ffn1_expert_weight_key.format(j)))
            )
            down_proj_weight.append(
                get_tensor(state_dict.pop(self.ffn2_expert_weight_key.format(j)))
            )
        return (
            up_gate_proj_weight,
            down_proj_weight,
            up_gate_proj_weight_scale,
            down_proj_weight_scale,
        )

    def load_gate_correction_bias(self, state_dict):
        """
            加载网关校正偏置。如果使用了网关校正偏置，则从state_dict中获取相应的张量并设置到网关校正偏置上。
        参数：
            state_dict (OrderedDict): 包含模型参数和状态的字典。
        返回值：
            无返回值，直接修改了网关校正偏置的值。
        """
        if self.moe_config.moe_use_gate_correction_bias:
            gate_correction_bias_tensor = get_tensor(
                state_dict[self.gate_correction_bias_key]
            )
            self.gate_correction_bias.set_value(
                gate_correction_bias_tensor[0].unsqueeze(0)
            )


class ImageMoELayer(MoELayer):
    """
    MoELayer is a layer that performs MoE (Mixture of Experts) computation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
            初始化函数，用于设置类的属性和方法。
        参数：
            - args (tuple, optional): 可变长度的位置参数列表，默认为空元组。
            - kwargs (dict, optional): 关键字参数字典，默认为空字典。
        返回值：
            无返回值，直接修改类的属性和方法。
        """
        moe_quant_type = os.getenv("ELLM_MM_IMAGE_QUANT_TYPE", None)
        if moe_quant_type is not None:
            kwargs["moe_quant_type"] = moe_quant_type
        kwargs["moe_tag"] = "Image"
        super().__init__(*args, **kwargs)

    def load_gate_state_dict(self, state_dict):
        """
            加载门状态字典。
        从给定的状态字典中提取并返回两个专家的上下关门投影权重，以及两个专家的下降投影权重。
        参数：
            state_dict (OrderedDict): 包含网络参数的有序字典。
        返回值：
            tuple (list, list)，分别是两个专家的上下关门投影权重和两个专家的下降投影权重，都是列表类型。
        """
        up_gate_proj_weight = []
        up_gate_proj_weight_scale = []
        down_proj_weight = []
        down_proj_weight_scale = []
        for j in range(self.num_experts, self.num_experts + self.num_experts):
            up_gate_proj_weight.append(
                get_tensor(state_dict.pop(self.ffn1_expert_weight_key.format(j)))
            )
            down_proj_weight.append(
                get_tensor(state_dict.pop(self.ffn2_expert_weight_key.format(j)))
            )
        return (
            up_gate_proj_weight,
            down_proj_weight,
            up_gate_proj_weight_scale,
            down_proj_weight_scale,
        )

    def load_gate_correction_bias(self, state_dict):
        """
            加载门级别校正偏置参数，如果使用门级别校正偏置则从state_dict中获取并设置到gate_correction_bias中。
        参数：
            state_dict (OrderedDict): 模型的状态字典，包含所有需要被加载的参数。
        返回值：
            无返回值，直接修改了gate_correction_bias的值。
        """
        if self.moe_config.moe_use_gate_correction_bias:
            gate_correction_bias_tensor = get_tensor(
                state_dict[self.gate_correction_bias_key]
            )
            self.gate_correction_bias.set_value(
                gate_correction_bias_tensor[1].unsqueeze(0)
            )


class MultimodalityMoeLayer(nn.Layer):
    """
    Multimodality MOE Layer
    """

    def __init__(
        self,
        inference_args,
        layer_name,
        layer_idx,
    ):
        """
            初始化一个 MoELayer。

        Args:
            inference_args (InferenceArgs): 推理参数类，包含了所有必要的配置信息。
            layer_name (str): 当前 MoE Layer 的名称。
            layer_idx (int): 当前 MoE Layer 在模型中的索引。

        Returns:
            None, 无返回值。
        """
        super().__init__()

        self.text_moe_layer = TextMoELayer(
            inference_args=inference_args,
            moe_config=inference_args.moe_config,
            layer_name=layer_name + ".text",
            gate_weight_key=f"ernie.layers.{layer_idx}.mlp.gate.weight",
            ffn1_expert_weight_key=f"ernie.layers.{layer_idx}.mlp.experts"
            + ".{}.up_gate_proj.weight",
            ffn2_expert_weight_key=f"ernie.layers.{layer_idx}.mlp.experts"
            + ".{}.down_proj.weight",
            gate_correction_bias_key=f"ernie.layers.{layer_idx}.mlp.moe_statics.e_score_correction_bias",
            ffn1_bias_key=None,
            ffn2_bias_key=None,
            ffn1_shared_weight_key=None,
            ffn1_shared_bias_key=None,
            ffn2_shared_weight_key=None,
            ffn2_shared_bias_key=None,
            layer_idx=layer_idx,
        )

        self.image_moe_layer = ImageMoELayer(
            inference_args=inference_args,
            moe_config=inference_args.moe_config_1,
            layer_name=layer_name + ".image",
            gate_weight_key=f"ernie.layers.{layer_idx}.mlp.gate.weight_1",
            ffn1_expert_weight_key=f"ernie.layers.{layer_idx}.mlp.experts"
            + ".{}.up_gate_proj.weight",
            ffn2_expert_weight_key=f"ernie.layers.{layer_idx}.mlp.experts"
            + ".{}.down_proj.weight",
            gate_correction_bias_key=f"ernie.layers.{layer_idx}.mlp.moe_statics.e_score_correction_bias",
            ffn1_bias_key=None,
            ffn2_bias_key=None,
            ffn1_shared_weight_key=None,
            ffn1_shared_bias_key=None,
            ffn2_shared_weight_key=None,
            ffn2_shared_bias_key=None,
            layer_idx=layer_idx,
        )

    def load_state_dict(self, state_dict):
        """
            加载模型参数。
        将给定的字典中的参数覆盖到当前模型上，并返回一个新的字典，其中包含未被覆盖的键值对。

        Args:
            state_dict (dict): 包含了要加载的模型参数的字典。

        Returns:
            dict: 包含未被覆盖的键值对的字典。
        """
        self.text_moe_layer.load_state_dict(state_dict)
        self.image_moe_layer.load_state_dict(state_dict)
        state_dict.pop(self.text_moe_layer.gate_correction_bias_key)

    def forward(self, x, **kwargs):
        """
            前向计算函数，将输入的张量进行处理并返回结果。
        该函数接受以下键值对参数：
            - token_type_ids (Optional, Tensor, default=None): 一个bool型Tensor，用于指定每个元素是否为文本类型（值为0）或图像类型（值为1）。
                如果未提供此参数，则会引发AssertionError。
        返回值是一个Tensor，形状与输入相同，表示处理后的结果。

        Args:
            x (Tensor): 输入张量，形状为[token_num, hidden_size]，其中token_num是序列长度，hidden_size是隐藏状态维度。
            kwargs (dict, optional): 可选参数字典，默认为None，包含以下键值对：
                - token_type_ids (Tensor, optional): 一个bool型Tensor，用于指定每个元素是否为文本类型（值为0）或图像类型（值为1），默认为None。

        Returns:
            Tensor: 一个Tensor，形状与输入相同，表示处理后的结果。

        Raises:
            AssertionError: 当未提供token_type_ids参数时会引发此错误。
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        assert token_type_ids is not None

        # x.shape is [token_num, hidden_size]
        fused_moe_out = paddle.zeros_like(x)

        text_mask = token_type_ids == 0  # [token_num]
        image_mask = token_type_ids == 1

        if text_mask.any():
            text_out = self.text_moe_layer(x[text_mask])
            fused_moe_out[text_mask] = text_out

        if image_mask.any():
            image_out = self.image_moe_layer(x[image_mask])
            fused_moe_out[image_mask] = image_out

        return fused_moe_out
