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

import copy
import json
import os
import re
from enum import Enum

import numpy as np
import paddle
from paddlenlp.utils.log import logger


class GenerationPhase(Enum):
    """
    The generation phase of the model.
    """

    PREFILL = 1
    DECODER = 2


class InferenceArgs:
    """
    The parameters used for inference, including model parameters and quantization information.
    """

    def __init__(
        self,
        quant_type,
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        hidden_size,
        ffn_hidden_size,
        mp_rank,
        mp_size,
        model_path="",
        use_fake_parameter=False,
        fp8_type="e4m3fn",
        quant_round_type=0,
        quant_max_bound=0,
        quant_min_bound=0,
        has_zero_point=False,
        is_channel_wise=False,
        gqa_use_tensorcore=False,
        use_dynamic_cachekv_quant=False,
        max_position_embeddings=512,
        speculate_method=None,
        speculate_max_draft_token_num=1,
        use_moe=False,
        moe_num_experts=None,
        moe_intermediate_size=None,
        moe_use_gate_correction_bias=False,
        moe_every2=False,
        moe_topk=8,
        moe_num_shared_experts=0,
        moe_layer_start_index=0,
        moe_use_ffn_shared_weight_and_bias=False,
        moe_group=False,
        moe_quant_type="default",
        use_ep=False,
        generation_phase=GenerationPhase.PREFILL,
        use_micro_batch=False,
        weight_block_size=[-1, -1],
        start_layer_index=0,
        scale_dir=None,
        enable_redundant_experts: bool = False,
        redundant_experts_num: int = 0,
        use_offline_quant=False,
        max_batch_size: int = 128,
    ):
        """
        Initialization function for quantization of the Transformer model

        Args:
        quant_type (str): Type of quantization. Options include 'abs_max', 'moving_average_abs_max',
            'range_abs_max', 'default'.
        num_layers (int): Number of layers in the Transformer model.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (int): Number of key-value heads. If less than or equal to 0,
            it is equal to num_attention_heads.
        hidden_size (int): Size of the hidden layer.
        ffn_hidden_size (int): Size of the hidden layer in the feedforward neural network.
        mp_rank (int): Rank of the current process in model parallelism.
        mp_size (int): Size of model parallelism.
        model_path (str, optional): Path to the model. Default is an empty string.
        use_fake_parameter (bool, optional): Whether to use fake parameters. Default is False.
        fp8_type (str, optional): Type of fp8. Options include 'e4m3fn', 'e5m2'. Default is 'e4m3fn'.
        quant_round_type (int, optional): Rounding type for quantization. Default is 0.
        quant_max_bound (float, optional): Maximum bound for quantization. Default is 0.
        quant_min_bound (float, optional): Minimum bound for quantization. Default is 0.
        use_dynamic_cachekv_quant (bool, optional): Whether to use dynamic caching for kv quantization.
            Default is False.
        max_position_embeddings (int, optional): Maximum position embeddings. Default is 512.
        Returns:
        None
        """
        self.quant_type = quant_type.lower()
        self.scale_dir = scale_dir

        if self.quant_type == "default":
            self.quant_type = ""

        self.moe_quant_type = moe_quant_type.lower()
        if self.moe_quant_type == "default":
            self.moe_quant_type = ""

        self.weight_block_size = weight_block_size
        # self.weight_block_size = [-1, -1]
        self.use_offline_quant = use_offline_quant
        self.ffn_hidden_size = ffn_hidden_size
        self.mp_rank = mp_rank
        if use_ep:
            self.mp_size = 1
            self.nranks = mp_size
        else:
            self.mp_size = mp_size
            self.nranks = mp_size
        self.use_ep = use_ep
        self.generation_phase = generation_phase
        self.use_micro_batch = use_micro_batch

        self.num_layers = num_layers
        self.start_layer_index = start_layer_index
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (num_key_value_heads if num_key_value_heads
                                    >= 0 else self.num_attention_heads)
        self.qkv_hidden_size = (self.num_attention_heads +
                                2 * self.num_key_value_heads) * self.head_dim
        self.dim_feedforward = ffn_hidden_size

        self.max_position_embeddings = max_position_embeddings
        self.model_path = model_path
        self.use_fake_parameter = use_fake_parameter
        self.fp8_type = fp8_type
        self.default_type = paddle.get_default_dtype()
        (
            self.weight_dtype,
            self.act_dtype,
            self.cachekv_dtype,
        ) = self.parser_quant_type(self.quant_type)
        logger.info(
            f"quant_type: weight[{self.weight_dtype}], act[{self.act_dtype}], cachekv[{self.cachekv_dtype}]"
        )
        self.enable_redundant_experts = enable_redundant_experts
        self.redundant_experts_num = redundant_experts_num

        self.max_batch_size = max_batch_size

        class MoEConfig:
            """
            Initialization moe config

            Args:
            use_moe (bool): whether your model have moe layer.
            num_experts (int): num of experts in moe layer.
            top_k (int): top_k in moe layer.
            moe_intermediate_size (int): the 2th linear's input dim.
            activation (str): the activation in your moe layer.
            """

            use_moe: bool = False
            num_experts: int = -1
            top_k: int = -1
            moe_intermediate_size: int = -1
            num_experts_per_rank: int = -1
            num_experts_start_offset: int = -1
            activation = "swiglu"

            moe_use_gate_correction_bias = False
            moe_every2 = (False, )
            moe_topk = (8, )
            moe_num_shared_experts = (0, )
            moe_layer_start_index = 0
            moe_use_ffn_shared_weight_and_bias = (False, )
            moe_group = (False, )
            moe_quant_type = self.moe_quant_type
            num_max_dispatch_tokens_per_rank = 256

            has_multimodality: bool = False
            im_patch_id = (
                100295  # multimodality, TODO(liuyuanle): read from config.json
            )

        self.moe_config = MoEConfig()
        self.moe_config.use_moe = use_moe
        if use_moe:
            if isinstance(moe_num_experts, list):
                self.moe_config.has_multimodality = True
                self.moe_config.num_experts = moe_num_experts[0]
            else:
                self.moe_config.num_experts = moe_num_experts
            self.moe_config.num_experts_per_rank = (
                self.moe_config.num_experts + redundant_experts_num
            ) // self.nranks
            self.moe_config.num_experts_start_offset = (
                self.moe_config.num_experts_per_rank * self.mp_rank)
            if isinstance(moe_intermediate_size, list):
                self.moe_config.moe_intermediate_size = moe_intermediate_size[
                    0]
            else:
                self.moe_config.moe_intermediate_size = moe_intermediate_size
            self.moe_config.moe_every2 = moe_every2
            self.moe_config.moe_num_shared_experts = moe_num_shared_experts
            self.moe_config.moe_layer_start_index = moe_layer_start_index
            self.moe_config.moe_use_ffn_shared_weight_and_bias = (
                moe_use_ffn_shared_weight_and_bias)
            self.moe_config.moe_group = moe_group
            self.moe_config.top_k = moe_topk
            self.moe_config.moe_use_gate_correction_bias = moe_use_gate_correction_bias

        if isinstance(moe_num_experts, list):
            # multimodality
            self.moe_config_1 = copy.deepcopy(self.moe_config)
            self.moe_config_1.num_experts = moe_num_experts[1]
            self.moe_config_1.moe_intermediate_size = moe_intermediate_size[1]

        self.use_weight_only = True if self.weight_dtype != self.act_dtype else False
        # arch (int): The compute arch for target device. For example, A100 is 80, v100 is 70,
        # if you do not assign arch, we will get arch from your device, default: None.
        self.weight_only_linear_arch = os.getenv(
            "FLAGS_weight_only_linear_arch")
        if self.weight_only_linear_arch is not None:
            self.weight_only_linear_arch = int(self.weight_only_linear_arch)

        self.use_append_attn = os.getenv("FLAGS_use_append_attn")
        if self.use_append_attn is not None:
            self.use_append_attn = int(self.use_append_attn) == 1
        else:
            self.use_append_attn = False

        self.has_zero_point = has_zero_point
        self.is_channel_wise = is_channel_wise
        self.gqa_use_tensorcore = gqa_use_tensorcore
        if self.gqa_use_tensorcore:
            logger.warning("TensorCore Attention is not supported yet.")

        if self.cachekv_dtype == "int8":
            self.cache_quant_type = "cache_int8"
            if self.has_zero_point:
                self.cache_quant_type += "_zp"
            self.cache_quant_max_bound = 127.0
            self.cache_quant_min_bound = -127.0
        elif self.cachekv_dtype == "float8_e4m3fn":
            self.cache_quant_type = "cache_fp8"
            self.cache_quant_max_bound = 448.0
            self.cache_quant_min_bound = -448.0
        elif self.cachekv_dtype == "int4":
            self.cache_quant_type = "cache_int4"
            self.cache_quant_max_bound = 7.0
            self.cache_quant_min_bound = -7.0
            if self.has_zero_point:
                self.cache_quant_type += "_zp"
        elif self.cachekv_dtype in ["bfloat16", "float16"]:
            self.cache_quant_type = "none"
        else:
            raise ValueError(f"Unsupported cachekv dtype {self.cachekv_dtype}")

        self.quant_round_type = quant_round_type
        self.quant_max_bound = quant_max_bound
        self.quant_min_bound = quant_min_bound
        self.use_dynamic_cachekv_quant = use_dynamic_cachekv_quant

        self.speculate_method = speculate_method
        self.speculate_max_draft_token_num = speculate_max_draft_token_num

        # set_scales
        if (self.act_dtype == "float8_e4m3fn"
            ):  # 4 exponent bits, 3 mantissa bits, and supports finite numbers
            self.quant_max_bound = 448.0
            self.quant_min_bound = -448.0
            self.quant_round_type = 1
        elif self.act_dtype == "int8":
            self.quant_max_bound = 127.0
            self.quant_min_bound = -127.0
            self.quant_round_type = 0
        elif self.act_dtype == "int4":
            self.quant_max_bound = 7.0
            self.quant_min_bound = -7.0
            self.quant_round_type = 0

        self.weight_scale_dict = {}
        self.act_scale_dict = {}
        self.cachekv_scale_dict = {}
        if not self.use_fake_parameter:
            self.set_scales()

    # TODO(tangbinhan):Add a unit test for this function.
    def parser_quant_type(self, quant_type):
        """
        Parse the quantization type string and return the corresponding quantization types for weights,
        activations, and custom.

        Args:
            quant_type (str): The quantization type string. It can be one of the following formats:
                - "weight_only_int8" or "wint8": Only weights are quantized to int8.
                - "weight_only_int4" or "wint4": Only weights are quantized to int4.
                - A custom string in the format of "wxaybzcfp8", where 'x', 'y', 'z' are the quantization bitwidths
                for weights, activations, and custom respectively,
                  and 'a', 'b', 'c' are the prefixes indicating the quantization types
                  (e.g., 'fp8' for floating-point 8-bit).
                  If a prefix is missing, the default quantization type will be used.

        Returns:
            tuple: A tuple of three strings representing the quantization types for weights, activations,
                   and custom respectively.
                   If the input is "weight_only_int8" or "wint8", returns ("int8", default_type, default_type).
                   If the input is "weight_only_int4" or "wint4", returns ("int4", default_type, default_type).
                   For custom strings, returns the parsed quantization types based on the input format.

        Raises:
            AssertionError: If the custom quantization type string format is incorrect.
        """
        cache_type = self.default_type
        if "c8" in quant_type:
            cache_type = "int8"
        elif "cfp8" in quant_type:
            cache_type = "fp8"
        elif "c4" in quant_type:
            cache_type = "int4"
        if "weight_only_int8" in quant_type or "wint8" in quant_type:
            return "int8", self.default_type, cache_type
        elif "weight_only_int4" in quant_type or "wint4" in quant_type:
            return "int4", self.default_type, cache_type
        else:
            # split quant type, eg. w4afp8c8 -> ['w', '4', 'a', 'fp8', 'c', '8']
            pattern = f"({'|'.join(map(re.escape, ['w', 'a', 'c']))})"
            splited_type = re.split(pattern, quant_type)
            splited_type = [tmp_type for tmp_type in splited_type if tmp_type]
            assert (len(splited_type) % 2 == 0 and len(splited_type)
                    <= 6), f"Quant type[{quant_type}] format error."

            quant_type_list = []
            if "w" in splited_type:
                w_idx = splited_type.index("w")
                quant_type_list.append(
                    self.get_quant_dtype(splited_type[w_idx + 1]))
            else:
                quant_type_list.append(self.default_type)
            if "a" in splited_type:
                a_idx = splited_type.index("a")
                quant_type_list.append(
                    self.get_quant_dtype(splited_type[a_idx + 1]))
            else:
                quant_type_list.append(self.default_type)
            if "c" in splited_type:
                c_idx = splited_type.index("c")
                quant_type_list.append(
                    self.get_quant_dtype(splited_type[c_idx + 1]))
            else:
                quant_type_list.append(self.default_type)

            return quant_type_list[0], quant_type_list[1], quant_type_list[2]

    def get_quant_dtype(self, quant_bit):
        """
        Get the quantized data type based on the specified bit width.

        Args:
            quant_bit (str): The bit width for quantization.
                Supported values include "8" for int8, "4" for int4, "fp8" for float8
                    (with additional type specified by self.fp8_type),
                "fp16" for float16, "bf16" for bfloat16, and "fp32" for float32.

        Returns:
            str: The corresponding quantized data type.

        Raises:
            ValueError: If the specified quant_bit is not supported.
        """

        if quant_bit == "8":
            return "int8"
        elif quant_bit == "4":
            return "int4"
        elif quant_bit == "16":
            return self.default_type
        elif quant_bit == "fp8":
            return "float8_" + self.fp8_type
        elif quant_bit == "fp16":
            return "float16"
        elif quant_bit == "bf16":
            return "bfloat16"
        elif quant_bit == "fp32":
            return "float32"
        else:
            raise ValueError(
                "only support [int8, int4, float8_e4m3fn, float8_e5m2, fp16/bf16/fp32]"
            )

    def set_cache_scales(self):
        """
        Set scales for weight, activation, and cache key-value.

        This method loads scales from JSON files located in the model path. It supports
        loading scales for weights, activations, and cache key-value parameters.
        Scales for unsupported parameters are ignored.

        Raises:
            NotImplementedError: If fake parameters are enabled (self.use_fake_parameter is True).
        """
        if not self.use_fake_parameter:
            # cachekv_scale
            if self.cachekv_dtype in ["bfloat16", "float16", "float32"]:
                return
            from glob import glob

            scale_dir = self.scale_dir
            scale_paths = glob(os.path.join(scale_dir, "*.json*"))
            cachekv_scale_dict_all = []
            self.cachekv_scale_dict = {}
            for possible_cache_scales_file_name in scale_paths:
                fi = open(possible_cache_scales_file_name)
                cachekv_scale_dict_all.append(json.load(fi))
            for cache_scale_dict in cachekv_scale_dict_all:
                for k, v in cache_scale_dict.items():
                    if k not in self.cachekv_scale_dict.keys():
                        self.cachekv_scale_dict[k] = []
                        self.cachekv_scale_dict[k].extend(v)
                    else:
                        self.cachekv_scale_dict[k].extend(v)
            print("self.cachekv_scale_dict: ", self.cachekv_scale_dict)

            num_heads = self.num_attention_heads // self.mp_size
            kv_num_heads = self.num_key_value_heads // self.mp_size
            col_dim = (kv_num_heads *
                       self.head_dim if self.is_channel_wise else kv_num_heads)

            for k, v in self.cachekv_scale_dict.items():
                # cache_kv_scale
                if k.endswith(".activation_quanter"):
                    if self.is_channel_wise:
                        v_array = (np.array(v).reshape(
                            -1, self.head_dim).astype(np.float32))
                    else:
                        v_array = np.array(v).reshape(-1).astype(np.float32)
                    if v_array.size > col_dim:
                        cache_scale = [
                            v_array[i].tolist()
                            for i in range(0, num_heads, num_heads //
                                           kv_num_heads)
                        ]
                    else:
                        cache_scale = [
                            v_array[i].tolist()
                            for i in range(0, kv_num_heads)
                        ]

                    if (self.has_zero_point
                            and self.cachekv_dtype == "int4"):  # cache_int4_zp
                        self.cachekv_scale_dict[k] = 1.0 / np.array(
                            cache_scale).flatten().astype(np.float32)
                    else:
                        self.cachekv_scale_dict[k] = (
                            self.cache_quant_max_bound /
                            np.array(cache_scale).flatten().astype(np.float32))
                # cache_kv_zp
                elif k.endswith(".zero_point"):
                    if self.is_channel_wise:
                        v_array = (np.array(v).reshape(
                            -1, self.head_dim).astype(np.float32))
                    else:
                        v_array = np.array(v).reshape(-1).astype(np.float32)
                    if v_array.size > col_dim:
                        cache_zp = [
                            v_array[i].tolist()
                            for i in range(0, num_heads, num_heads //
                                           kv_num_heads)
                        ]
                    else:
                        cache_zp = [
                            v_array[i].tolist()
                            for i in range(0, kv_num_heads)
                        ]
                    self.cachekv_scale_dict[k] = (
                        np.array(cache_zp).flatten().astype(np.float32))
                else:
                    continue
        else:
            raise NotImplementedError("fake parameter not support now")

    def set_scales(self):
        """
        Set scales for weight, activation, and cache key-value.

        This method loads scales from JSON files located in the model path. It supports
        loading scales for weights, activations, and cache key-value parameters.
        Scales for unsupported parameters are ignored.

        Raises:
            NotImplementedError: If fake parameters are enabled (self.use_fake_parameter is True).
        """
        if not self.use_fake_parameter:
            # weight_scale
            if self.use_ep:
                weight_scale_json_path = os.path.join(self.model_path,
                                                      "weight_scales.json")
            else:
                weight_scale_json_path = os.path.join(
                    self.model_path, f"weight_scales_{self.mp_rank}.json")
            if os.path.exists(weight_scale_json_path):
                with open(weight_scale_json_path) as json_file:
                    self.weight_scale_dict = json.load(json_file)
            for k, v in self.weight_scale_dict.items():
                if not k.endswith(".weight_quanter"):
                    continue
                self.weight_scale_dict[k] = np.array(v).astype(np.float32)

            # act_scale
            if self.use_ep:
                act_scale_json_path = os.path.join(self.model_path,
                                                   "act_scales.json")
            else:
                act_scale_json_path = os.path.join(
                    self.model_path, f"act_scales_{self.mp_rank}.json")
            if os.path.exists(act_scale_json_path):
                with open(act_scale_json_path) as json_file:
                    self.act_scale_dict = json.load(json_file)
            for k, v in self.act_scale_dict.items():
                if not k.endswith(".activation_quanter"):
                    continue
                self.act_scale_dict[k] = 1.0 / np.array(v).astype(np.float32)

            # cachekv_scale
            if self.cachekv_dtype in ["bfloat16", "float16", "float32"]:
                return
            if self.use_ep:
                from glob import glob

                scale_dir = self.scale_dir
                scale_paths = glob(os.path.join(scale_dir, "cachekv_scale*"))
                cachekv_scale_dict_all = []
                self.cachekv_scale_dict = {}
                for possible_cache_scales_file_name in scale_paths:
                    fi = open(possible_cache_scales_file_name)
                    cachekv_scale_dict_all.append(json.load(fi))
                for cache_scale_dict in cachekv_scale_dict_all:
                    for k, v in cache_scale_dict.items():
                        if k not in self.cachekv_scale_dict.keys():
                            self.cachekv_scale_dict[k] = []
                            self.cachekv_scale_dict[k].extend(v)
                        else:
                            self.cachekv_scale_dict[k].extend(v)
            else:
                for possible_cache_scales_file_name in [
                        f"cachekv_scales_{self.mp_rank}.json",
                        f"cachekv_act_scales_{self.mp_rank}.json",
                ]:
                    cache_scale_json_path = os.path.join(
                        self.model_path, possible_cache_scales_file_name)
                    if os.path.exists(cache_scale_json_path):
                        with open(cache_scale_json_path) as json_file:
                            self.cachekv_scale_dict = json.load(json_file)
                        break
            num_heads = self.num_attention_heads // self.mp_size
            kv_num_heads = self.num_key_value_heads // self.mp_size
            col_dim = (kv_num_heads *
                       self.head_dim if self.is_channel_wise else kv_num_heads)

            for k, v in self.cachekv_scale_dict.items():
                # cache_kv_scale
                if k.endswith(".activation_quanter"):
                    if self.is_channel_wise:
                        v_array = (np.array(v).reshape(
                            -1, self.head_dim).astype(np.float32))
                    else:
                        v_array = np.array(v).reshape(-1).astype(np.float32)
                    if v_array.size > col_dim:
                        cache_scale = [
                            v_array[i].tolist()
                            for i in range(0, num_heads, num_heads //
                                           kv_num_heads)
                        ]
                    else:
                        cache_scale = [
                            v_array[i].tolist()
                            for i in range(0, kv_num_heads)
                        ]

                    if (self.has_zero_point
                            and self.cachekv_dtype == "int4"):  # cache_int4_zp
                        self.cachekv_scale_dict[k] = 1.0 / np.array(
                            cache_scale).flatten().astype(np.float32)
                    else:
                        self.cachekv_scale_dict[k] = (
                            self.cache_quant_max_bound /
                            np.array(cache_scale).flatten().astype(np.float32))
                # cache_kv_zp
                elif k.endswith(".zero_point"):
                    if self.is_channel_wise:
                        v_array = (np.array(v).reshape(
                            -1, self.head_dim).astype(np.float32))
                    else:
                        v_array = np.array(v).reshape(-1).astype(np.float32)
                    if v_array.size > col_dim:
                        cache_zp = [
                            v_array[i].tolist()
                            for i in range(0, num_heads, num_heads //
                                           kv_num_heads)
                        ]
                    else:
                        cache_zp = [
                            v_array[i].tolist()
                            for i in range(0, kv_num_heads)
                        ]
                    self.cachekv_scale_dict[k] = (
                        np.array(cache_zp).flatten().astype(np.float32))
                else:
                    continue
        else:
            raise NotImplementedError("fake parameter not support now")
