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

from typing import Dict

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantMethodBase
from fastdeploy.model_executor.layers.quantization.weight_only import \
    WeightOnlyConfig
from fastdeploy.model_executor.ops.xpu import weight_quantize_xpu

from .fused_moe_backend_base import MoEMethodBase

from ..utils import create_and_set_parameter, get_tensor

class XPUMoEMethod(MoEMethodBase):
    """
    XPU MOE
    """

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        # bf16
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
        for weights in [up_gate_proj_weights, down_proj_weights]:
            for idx, weight in enumerate(weights):
                weights[idx] = weight.transpose([1, 0])
        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)
        for idx, weight_tensor in enumerate(
            [stacked_up_gate_proj_weights, stacked_down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            setattr(
                layer, weight_name,
                layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=weight_tensor.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ))
            getattr(layer, weight_name).set_value(weight_tensor)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        fused_moe_out = xpu_moe_layer(
            x,
            layer.gate_weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # up_gate_proj bias
            None,  # down_proj bias
            None,  # up_gate_proj scale
            None,  # down_proj scale
            None, # up_gate_proj_in_scale
            "", # moe_quant_type
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError

class XPUW4A8MoEMethod(XPUMoEMethod):
    """
    XPU w4a8 MoE Method
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.moe_quant_type = "w4a8"
        self.pack_num = 2

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        print(f"----------为layer{layer.full_name}加载权重------------")
        # for k, v in state_dict.items():
        #     print(f"key is : {k}")
        #     print(f"value.type is {v.dtype}")
        #     print(f"value.shape is {v.shape}")

        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
        self.check(layer, up_gate_proj_weights, down_proj_weights)
        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            weight_list = []
            for i in range(layer.num_local_experts):
                # print(f"self.moe_quant_type : {self.moe_quant_type}")
                # print(f"weight_tensor[i].shape: {weight_tensor[i].shape}")
                # print(f"weight_tensor[i].dtpye: {weight_tensor[i].dtype}")
                # print(f"self.moe_quant_type : {self.moe_quant_type}")
                quant_weight, scale = weight_quantize_xpu(weight_tensor[i],
                                                      self.moe_quant_type,
                                                      -1,-1)
                weight_list.append(quant_weight)
            quanted_weight = paddle.stack(weight_list, axis=0)
            create_and_set_parameter(layer, weight_name, quanted_weight)

        self.create_w4a8_scale_weights(layer, layer.weight_key_map, state_dict)
    
    
    def create_w4a8_scale_weights(self, layer: nn.Layer, weight_key_map: dict,
                                  state_dict: dict):
        """
        Get w4a8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
            state_dict (dict): The state dict.
        """

        def _extract_scale_tensor(state_dict, key_template, expert_idx):
            return get_tensor(state_dict.pop(key_template.format(expert_idx)))

        def _process_in_scale(name: str, in_scales: list[paddle.Tensor]):
            processed_in_scale = 1 / paddle.concat(in_scales)
            create_and_set_parameter(layer, name, processed_in_scale)
            return processed_in_scale

        def _process_weight_scale(name: str,
                                  weight_scales: list[paddle.Tensor],
                                  processed_in_scale: paddle.Tensor):
            processed_weight_scale = (paddle.stack(weight_scales, axis=0) /
                                      (127 * 112) /
                                      processed_in_scale[:, None]).cast(
                                          dtype="float32")

            # print(f"paddle.get_default_dtype() ： {paddle.get_default_dtype()}")
            processed_weight_scale = (paddle.stack(weight_scales, axis=0) /
                                      (127 * 112) /
                                      processed_in_scale[:, None]).cast(
                                          paddle.get_default_dtype())
            
            create_and_set_parameter(layer, name, processed_weight_scale)

        # 1. Init scale containers and maps
        up_gate_proj_weight_scales = []
        down_proj_weight_scales = []
        up_gate_proj_in_scales = []
        down_proj_in_scales = []

        scale_weight_map = {
            "up_gate_proj_weight_scale": up_gate_proj_weight_scales,
            "down_proj_weight_scale": down_proj_weight_scales,
            "up_gate_proj_in_scale": up_gate_proj_in_scales,
            "down_proj_in_scale": down_proj_in_scales,
        }
        scale_key_map = {
            "up_gate_proj_weight_scale":
            weight_key_map.get("up_gate_proj_expert_weight_scale_key", None),
            "down_proj_weight_scale":
            weight_key_map.get("down_proj_expert_weight_scale_key", None),
            "up_gate_proj_in_scale":
            weight_key_map.get("up_gate_proj_expert_in_scale_key", None),
            "down_proj_in_scale":
            weight_key_map.get("down_proj_expert_in_scale_key", None),
        }
        for name, value in scale_key_map.items():
            if value is None:
                raise ValueError(
                    f"scale {name} should not be none in w4a8 mode.")

        # 2. Extract scale tensor from state dict
        for local_expert_idx in range(layer.num_local_experts):
            expert_idx = local_expert_idx + layer.expert_id_offset * layer.num_local_experts
            for name, scale_key_template in scale_key_map.items():
                scale_tensor = _extract_scale_tensor(state_dict,
                                                     scale_key_template,
                                                     expert_idx)
                # print(f"scale_tensor.dtype : {scale_tensor.dtype}")
                scale_weight_map[name].append(scale_tensor)

        # 3. Process scale tensor and set to layer
        in_scales = []
        for in_scale_name in ["up_gate_proj_in_scale", "down_proj_in_scale"]:
            in_scales.append(
                _process_in_scale(in_scale_name,
                                  scale_weight_map[in_scale_name]))

        for i, weight_scale_name in enumerate(
            ["up_gate_proj_weight_scale", "down_proj_weight_scale"]):
            _process_weight_scale(weight_scale_name,
                                  scale_weight_map[weight_scale_name],
                                  in_scales[i])


    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        XPU compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        print(f"layer.up_gate_proj_weight.shape : {layer.up_gate_proj_weight.shape}")
        print(f"layer.down_proj_weight.shappe : {layer.down_proj_weight.shape}")

        print(f"layer.up_gate_proj_weight.transpose([2, 1]).shape : {layer.up_gate_proj_weight.transpose([0, 2, 1]).shape}")
        print(f"layer.down_proj_weight.transpose([2, 1]).shape {layer.down_proj_weight.transpose([2, 1]).shape}")
        print(f"layer.up_gate_proj_weight_scale.dtype : {layer.up_gate_proj_weight_scale.dtype}")
        print(f"layer.down_proj_weight_scale.dtype : {layer.down_proj_weight_scale.dtype}")

        # if layer.up_gate_proj_weight_scale is not None:
        #     # layer.up_gate_proj_weight_scale.set_value(paddle.cast(layer.up_gate_proj_weight_scale, dtype="float32"))
        #     layer.up_gate_proj_weight_scale.astype("float32")
            

        # if layer.down_proj_weight_scale is not None:
        #     # layer.down_proj_weight_scale.set_value(paddle.cast(layer.down_proj_weight_scale, dtype="float32"))
        #     layer.down_proj_weight_scale.astype("float32")

        if layer.up_gate_proj_weight_scale is not None:
            # 先转换数据类型
            casted_tensor = paddle.cast(layer.up_gate_proj_weight_scale, dtype="float32")
            
            # 先删除旧参数
            if hasattr(layer, 'up_gate_proj_weight_scale'):
                del layer.up_gate_proj_weight_scale
            
            # 创建新参数，会自动生成唯一名称
            layer.up_gate_proj_weight_scale = paddle.create_parameter(
                shape=casted_tensor.shape,
                dtype=casted_tensor.dtype,
                default_initializer=paddle.nn.initializer.Assign(casted_tensor)
                # 不指定name参数，让框架自动生成
            )

        if layer.down_proj_weight_scale is not None:
            casted_tensor = paddle.cast(layer.down_proj_weight_scale, dtype="float32")
            
            if hasattr(layer, 'down_proj_weight_scale'):
                del layer.down_proj_weight_scale
            
            layer.down_proj_weight_scale = paddle.create_parameter(
                shape=casted_tensor.shape,
                dtype=casted_tensor.dtype,
                default_initializer=paddle.nn.initializer.Assign(casted_tensor)
            )
        
        # print(f"x.dtype : {x.dtype}")
        # print(f"layer.gate_weight: {layer.gate_weight.dtype}")
        # print(f"layer.gate_correction_bias: {layer.gate_correction_bias.dtype}")
        # print(f"layer.up_gate_proj_weight.dtype: {layer.up_gate_proj_weight.dtype}")
        # print(f"layer.down_proj_weight.dtype: {layer.down_proj_weight.dtype}")
        # print(f"up_gate_proj_weight_scale: {layer.up_gate_proj_weight_scale.dtype}") if hasattr(layer, "up_gate_proj_weight_scale") else None
        # print(f"down_proj_weight_scale: {layer.down_proj_weight_scale.dtype}") if hasattr(layer, "down_proj_weight_scale") else None

        fused_moe_out = xpu_moe_layer(
            x,
            layer.gate_weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.up_gate_proj_weight.transpose([0, 2, 1]),
            layer.down_proj_weight.transpose([0, 2, 1]),
            None,  # up_gate_proj bias
            None,  # down_proj bias
            (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
            (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
            # (layer.down_proj_in_scale
            #  if hasattr(layer, "down_proj_in_scale") else None),
            None,
            self.moe_quant_type,
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out


class XPUWeightOnlyMoEMethod(QuantMethodBase):
    """
    XPU Fused MoE Method.
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.moe_quant_type = self.quant_config.algo

    def create_weights(self, layer: nn.Layer, state_dict: Dict[str,
                                                               paddle.Tensor]):
        """
        Paddle cutlass create weight process.
        """
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)

        for i in range(len(up_gate_proj_weights)):
            print(f"up_gate_proj_weights[i].shape : {up_gate_proj_weights[i].shape}")
            print(f"down_proj_weights[i].shape : {down_proj_weights[i].shape}")
        print(f"layer.hidden_size : {layer.hidden_size}")
        print(f"layer.moe_intermediate_size : {layer.moe_intermediate_size}")
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        # 一个专家的shape
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size, layer.moe_intermediate_size * 2
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size, layer.hidden_size
        ]

        added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        added_scale_attrs = ["up_gate_proj_weight_scale", "down_proj_weight_scale"]

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = added_weight_attrs[idx]
            scale_name = added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize_xpu(
                    weight_tensor[i], self.moe_quant_type, -1,
                    -1)  # weight is [k,n]
                weight_list.append(quant_weight.transpose(
                    [1, 0]))  # transpose weight to [n,k]
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            setattr(
                layer, weight_name,
                layer.create_parameter(
                    shape=quanted_weight.shape,
                    dtype=quanted_weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ))
            getattr(layer, weight_name).set_value(quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            setattr(
                layer, scale_name,
                layer.create_parameter(
                    shape=quanted_weight_scale.shape,
                    dtype=quanted_weight_scale.dtype,
                ))
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        XPU compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        fused_moe_out = xpu_moe_layer(
            x,
            layer.gate_weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # up_gate_proj bias
            None,  # down_proj bias
            (layer.up_gate_proj_weight_scale
             if hasattr(layer, "up_gate_proj_weight_scale") else None),
            (layer.down_proj_weight_scale
             if hasattr(layer, "down_proj_weight_scale") else None),
            (layer.down_proj_in_scale
             if hasattr(layer, "down_proj_in_scale") else None),
            self.moe_quant_type,
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out
