"""
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
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import struct
from functools import partial

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.common_ops_import import convert_dtype
from paddle.distributed import fleet
from paddleformers.transformers.model_utils import (_add_variant,
                                                    load_tp_checkpoint)
from paddleformers.transformers.utils import paddleformers_load
from paddleformers.utils.env import (PADDLE_WEIGHTS_INDEX_NAME,
                                     SAFE_MASTER_WEIGHTS_INDEX_NAME,
                                     SAFE_PEFT_WEIGHTS_INDEX_NAME,
                                     SAFE_WEIGHTS_INDEX_NAME)
from paddleformers.utils.log import logger
from safetensors import safe_open
from tqdm import tqdm

from fastdeploy.config import ModelConfig

MAX_BSZ = 512
MAX_DRAFT_TOKENS = 6


class UniqueIDGenerator:
    """
    The generator for the export model id
    """

    def __init__(self):
        pass

    def generate_unique_id(self, state_dict):
        """
        Generate the model id from the timestamp
        """
        keys = state_dict.keys()
        sorted_keys = sorted(keys)
        first_key = sorted_keys[0]
        first_parameter = state_dict[first_key].cast("float32")
        # 假设模型参数是唯一的，通过第一个key来获取md5sum
        model_md5 = hashlib.md5(str(
            first_parameter.sum()).encode("utf-8")).hexdigest()
        unique_id = f"{model_md5}-{random.randint(10000, 99999)}"
        return unique_id


def load_sharded_checkpoint(folder, variant=None, return_numpy=False):
    """

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.

    """
    # Load the index
    pdparams_file = os.path.join(folder,
                                 _add_variant("model_state.pdparams", variant))
    lora_pdparams_file = os.path.join(
        folder, _add_variant("lora_model_state.pdparams", variant))
    safetensors_file = os.path.join(folder,
                                    _add_variant("model.safetensors", variant))
    if os.path.isfile(pdparams_file):
        return paddle.load(pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(lora_pdparams_file):
        return paddle.load(lora_pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(safetensors_file):
        try:
            from paddleformers.utils.safetensors import \
                fast_load_file as safe_load_file
        except ImportError:
            from safetensors.numpy import load_file as safe_load_file

        state_dict = safe_load_file(safetensors_file)
        if not return_numpy:
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = paddle.Tensor(state_dict.pop(key),
                                                    zero_copy=True)
        return state_dict

    index_file = os.path.join(folder,
                              _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(
        folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
    safe_master_file = os.path.join(
        folder, _add_variant(SAFE_MASTER_WEIGHTS_INDEX_NAME, variant))
    safe_peft_file = os.path.join(
        folder, _add_variant(SAFE_PEFT_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    safe_master_present = os.path.isfile(safe_master_file)
    safe_peft_present = os.path.isfile(safe_peft_file)

    load_safe = False
    load_index = None
    if safe_index_present:
        load_safe = True  # load safe due to preference
        load_index = safe_index_file
    elif safe_master_present:
        load_safe = True
        load_index = safe_master_file
    elif index_present:
        load_index = index_file
    elif safe_peft_present:
        load_safe = True
        load_index = safe_peft_file
    else:
        raise ValueError(
            f"Could not find {index_file} or {safe_index_file} or {safe_peft_file}"
        )

    if load_safe:
        try:
            from paddleformers.utils.safetensors import \
                fast_load_file as safe_load_file
        except ImportError:
            from safetensors.numpy import load_file as safe_load_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loader = (safe_load_file if load_safe else partial(
        paddleformers_load, map_location="np" if return_numpy else "cpu"))

    ret = {}
    for shard_file in tqdm(shard_files):
        state_dict = loader(os.path.join(folder, shard_file))
        ret.update(state_dict)

    if not return_numpy:
        for key in list(ret.keys()):
            if isinstance(ret[key], np.ndarray):
                ret[key] = paddle.Tensor(ret.pop(key), zero_copy=True)

    return ret


def convert_ndarray_dtype(np_array: np.ndarray,
                          target_dtype: str) -> np.ndarray:
    """convert ndarray

    Args:
        np_array (np.ndarray): numpy ndarray instance
        target_dtype (str): the target dtype

    Returns:
        np.ndarray: converted numpy ndarray instance
    """
    source_dtype = convert_dtype(np_array.dtype)
    if source_dtype == "uint16" or target_dtype == "bfloat16":
        if paddle.is_compiled_with_xpu():
            # xpu not support bf16.
            tensor = paddle.to_tensor(np_array, place=paddle.CPUPlace())
        else:
            tensor = paddle.to_tensor(np_array)
        tensor = paddle.cast(tensor, target_dtype)
        return tensor.numpy()

        # TODO(wj-Mcat): device_guard will slow the converting
        # with device_guard("cpu"):
        #     tensor = paddle.to_tensor(np_array)
        #     tensor = paddle.cast(tensor, target_dtype)
        # return tensor.numpy()

    if target_dtype == "bfloat16":
        target_dtype = "uint16"

    return np_array.astype(target_dtype)


def set_seed(seed: int):
    """
    set random seed for all random modules
    """
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def pad_batch_data(insts, pad_id=0, return_seq_len=False, pad_style="right"):
    """Pad the instances to the max sequence length in batch."""
    # pad to max input len i bsz
    max_len = max(map(len, insts))
    # pad to max input len
    # max_len = args.max_len
    if pad_style == "left":
        inst_data = np.array([[pad_id] * (max_len - len(inst)) + list(inst)
                              for inst in insts])
    else:
        inst_data = np.array(
            [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])
    if return_seq_len:
        seq_len = np.array([len(inst) for inst in insts])
        return inst_data.astype("int64").reshape([-1, max_len]), seq_len
    else:
        return inst_data.astype("int64").reshape([-1, max_len])


def load_prefix_weights(
    prefix_path: str,
    inference: bool = False,
    batch_size: int = 1,
    dtype: str = "bfloat16",
) -> np.ndarray | list[paddle.Tensor]:
    """load prefix weight by path

    Args:
        prefix_path (str): the path of prefix weight
    """
    past_key_values = paddle.to_tensor(
        np.load(f"{prefix_path}/pre_caches.npy")).unsqueeze(2)

    if batch_size > 1:
        past_key_values = paddle.concat([past_key_values] * batch_size, axis=2)

    # .chatglm static model require one tensor, otherwise list of tensor
    past_key_values = past_key_values.astype(dtype)
    if inference:
        return past_key_values.numpy()
    return past_key_values


def init_distributed_env() -> tuple[int, int]:
    """init distributed envs, and only support mp in ErnieBotModel

    Returns:
        tuple[int, int]: tensor_parallel_degree, tensor_parallel_rank
    """
    tensor_parallel_degree = dist.get_world_size()
    tensor_parallel_rank = 0

    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        tensor_parallel_rank = hcg.get_model_parallel_rank()

    return tensor_parallel_degree, tensor_parallel_rank


def w4a8_weight_convert(state_dict):
    """W4A8 权重转换函数
    Args:
        state_dict (dict): state_dict of model
    """

    def w4_weight_squash(value, name, w4a8_weight_bites_name_map):
        weight_dq = value
        # W8表象下的W4权重的absmax值为112，使用正负112进行权重类型判断
        if weight_dq.max() == 112 or weight_dq.min() == -112:
            weight_dq = weight_dq.cast("int8")
            np_weight_dq = np.array(weight_dq, dtype="int8").view("uint8")
            np_weight_dq_left_div_16 = (np_weight_dq / 16).astype("int8")
            # weight_q = (weight_dq/16).cast('int8')
            weight_q = paddle.to_tensor(np_weight_dq_left_div_16, dtype="int8")
            logger.debug(f"int4 weight:{name}")
            w4a8_weight_bites_name_map[name] = 4
            return weight_q.cast("int8")
        elif weight_dq.max() == 127 or weight_dq.min() == -128:
            logger.debug(f"int8 weight:{name}")
            w4a8_weight_bites_name_map[name] = 8
            return weight_dq.cast("int8")
        else:
            logger.debug(f"fp16/bf16/float weight:{name}")
            return weight_dq

    w4a8_weight_bites_name_map = {}
    for name, value in state_dict.items():
        if value.dtype == "uint16":
            weight_q = w4_weight_squash(
                paddle.to_tensor(value).cast("float32"),
                name,
                w4a8_weight_bites_name_map,
            )
            state_dict[name] = weight_q.numpy(
            ) if weight_q is not None else value
            del weight_q
    w4a8_weight_bites_layers_map = {}
    w4a8_weight_bites_layers_map["qkv_gemm_bits_map"] = []
    w4a8_weight_bites_layers_map["out_gemm_bits_map"] = []
    w4a8_weight_bites_layers_map["ffn1_gemm_bits_map"] = []
    w4a8_weight_bites_layers_map["ffn2_gemm_bits_map"] = []
    for name_keys, gemm_bits in w4a8_weight_bites_name_map.items():
        if "qkv_proj" in name_keys:
            w4a8_weight_bites_layers_map["qkv_gemm_bits_map"].append(gemm_bits)
        elif "out_proj" in name_keys:
            w4a8_weight_bites_layers_map["out_gemm_bits_map"].append(gemm_bits)
        elif "linear1" in name_keys:
            w4a8_weight_bites_layers_map["ffn1_gemm_bits_map"].append(
                gemm_bits)
        elif "linear2" in name_keys:
            w4a8_weight_bites_layers_map["ffn2_gemm_bits_map"].append(
                gemm_bits)
    logger.debug(
        f"w4a8_weight_bites_layers_map:{w4a8_weight_bites_layers_map}")
    return state_dict, w4a8_weight_bites_layers_map


def _vocab_size_with_padding(vocab_size, div_unit, mp_degree):
    padded_size = vocab_size
    multiple = div_unit * mp_degree
    while (padded_size % multiple) != 0:
        padded_size += 1
    # logger.warning(
    #     " > padded vocab (size: {}) with {} dummy tokens "
    #     "(new size: {})".format(vocab_size, padded_size - vocab_size, padded_size)
    # )
    return padded_size


def save_test_case(cases: list[list[dict]], file: str):
    """save test to result file

    Args:
        cases (list[list[dict]]): the content of case
        file (str): the path of saved file
    """
    with open(file, "w+", encoding="utf-8") as f:
        for case in cases:
            raw = json.dumps(case, ensure_ascii=False)
            f.write(raw + "\n")


def infer_save_test_case(cases: list[list[dict]], file: str):
    """save test to result file

    Args:
        cases (list[list[dict]]): the content of case
        file (str): the path of saved file
    """
    with open(file, "a+", encoding="utf-8") as f:
        for case in cases:
            raw = json.dumps(case, ensure_ascii=False)
            f.write(raw + "\n")


def deserialize_from_file(fp):
    """
    deserialize a binary file into an array
    """

    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b"0":
        data = fp.read(4)
        data_out = struct.unpack("f", data)[0]
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b"1":
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b"2":
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


def calculate_effective_tokens(training_args, train_dataset, max_seq_len):
    """
    Calculate the effective tokens during training.
    """
    total_effective_tokens = 0
    try:
        data_parallel_degree = training_args.data_parallel_degree
    except Exception:
        data_parallel_degree = 1
    if training_args.sharding_parallel_degree > 1:
        sharding_parallel_degree = training_args.sharding_parallel_degree
    else:
        sharding_parallel_degree = 1

    total_batch = (training_args.max_steps *
                   training_args.per_device_train_batch_size *
                   training_args.gradient_accumulation_steps *
                   sharding_parallel_degree * data_parallel_degree)
    for i, data in enumerate(train_dataset):
        if i == total_batch:
            break
        for dd in data:
            total_effective_tokens += len(dd.token_ids)
    total_tokens = total_batch * max_seq_len

    return total_effective_tokens, total_tokens


def load_ep_checkpoint(model_path: str,
                       config: ModelConfig,
                       return_numpy: bool = False,
                       return_key_name: bool = True):
    """
    load ep checkpoint
    """
    # return_numpy=True cpu
    # return_numpy=False gpu
    with open(os.path.join(model_path, "model.safetensors.index.json"),
              "r") as f:
        weight_list = json.load(f)["weight_map"]
    filtered_map = {k: v for k, v in weight_list.items() if "experts" not in k}
    num_local_ffn_keys = []

    for i in range(config.moe_layer_start_index, config.num_layers):
        for j in range(
                config.num_experts_start_offset,
                config.num_experts_start_offset + config.num_experts_per_rank,
        ):
            ffn1_key = f"ernie.layers.{i}.mlp.experts.{j}.up_gate_proj.weight"
            ffn2_key = (
                f"ernie.layers.{i}.mlp.experts.{j}.down_proj.weight")
            
            ffn1_quant_key = f"ernie.layers.{i}.mlp.experts.{j}.up_gate_proj.quant_weight"
            ffn2_quant_key = (
                f"ernie.layers.{i}.mlp.experts.{j}.down_proj.quant_weight")

            ffn1_scale_key = f"ernie.layers.{i}.mlp.experts.{j}.up_gate_proj.weight_scale"
            ffn2_scale_key = (
                f"ernie.layers.{i}.mlp.experts.{j}.down_proj.weight_scale")
            num_local_ffn_keys.append(ffn1_key)
            num_local_ffn_keys.append(ffn2_key)
            num_local_ffn_keys.append(ffn1_quant_key)
            num_local_ffn_keys.append(ffn2_quant_key)
            num_local_ffn_keys.append(ffn1_scale_key)
            num_local_ffn_keys.append(ffn2_scale_key)

    for k in num_local_ffn_keys:
        if k in weight_list:
            filtered_map[k] = weight_list[k]

    state_dict = {}
    # Get all safetensor file paths that need to be opened
    safetensor_paths = set(filtered_map.values())

    # Open each safetensor file sequentially with progress bar
    for safetensor_path in tqdm(safetensor_paths,
                                desc="Loading safetensor files",
                                unit="file"):
        with safe_open(os.path.join(model_path, safetensor_path),
                       framework="np",
                       device="cpu") as f:
            # Check if this file contains keys from filtered_map
            for k in filtered_map:
                if filtered_map[k] == safetensor_path and k in f.keys():
                    weight = f.get_tensor(k)
                    if not return_numpy:
                        weight = paddle.Tensor(weight, zero_copy=True)
                        weight = weight._copy_to(
                            paddle.framework._current_expected_place(), False)
                    state_dict[k] = weight
    return state_dict


def get_safetensor_file(model_path):
    """
    get_safetensor_file
    """
    with open(os.path.join(model_path, "model.safetensors.index.json"),
              "r") as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(
            os.path.join(model_path, weight_map[weight_name]))
    key_name_list = list(set(weight_map.keys()))
    safetensor_list = list(weight_files_in_index)
    safetensor_list.sort()
    return key_name_list, safetensor_list


def safetensors_weights_iterator(safe_tensor_list: list[str], ):
    """
    safetensors_weights_iterator
    """
    for st_file in tqdm(
            safe_tensor_list,
            desc="Loading safetensors checkpoint shards",
    ):
        with safe_open(st_file, framework="np") as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                yield name, param


def fastsafetensors_weights_iterator(safetensor_list: list[str]):
    """
    fastsafetensors_weights_iterator
    """
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
    world_size = dist.get_world_size()
    if world_size > 1:
        dist.init_parallel_env()
        pg = dist.get_group()
        device = f"gpu:{pg.rank}" if paddle.is_compiled_with_cuda() else "cpu"
    else:
        pg = SingleGroup()
        device = f"gpu:{pg.rank()}" if paddle.is_compiled_with_cuda(
        ) else "cpu"

    safetensor_files_sub_lists = [
        safetensor_list[i:i + world_size]
        for i in range(0, len(safetensor_list), world_size)
    ]
    for st_file in tqdm(
            safetensor_files_sub_lists,
            desc="Loading fastsafetensors checkpoint shards",
    ):
        loader = SafeTensorsFileLoader(pg,
                                       device,
                                       nogds=True,
                                       debug_log=False,
                                       framework="paddle")
        rank_file_map = {i: [f] for i, f in enumerate(st_file)}
        loader.add_filenames(rank_file_map)
        try:
            fb = loader.copy_files_to_device()
            try:
                keys = list(fb.key_to_rank_lidx.keys())
                for k in keys:
                    t = fb.get_tensor(k)
                    yield k, t
            finally:
                fb.close()
        finally:
            loader.close()


def get_state_dict(model_path, config, use_fastsafetensor=False):
    """
    get_state_dict
    """
    state_dict = {}
    _, safetensor_list = get_safetensor_file(
        os.path.join(model_path, f"rank{config.tensor_parallel_rank}"))
    if use_fastsafetensor:
        weights_iterator = fastsafetensors_weights_iterator(safetensor_list)
    else:
        weights_iterator = safetensors_weights_iterator(safetensor_list)

    for name, weight in weights_iterator:
        state_dict[name] = weight
    return state_dict


def apply_quant(name_action_quant_mappings, key, tensor, state_dict):
    """
    apply_quant
    """
    if key in name_action_quant_mappings:
        action = name_action_quant_mappings.pop(key)
        quant_weight_tensor, weight_scale_tensor = action(tensor)
        if quant_weight_tensor is not None and weight_scale_tensor is not None:
            state_dict[key + ".quant_weight"] = quant_weight_tensor
            state_dict[key + ".weight_scale"] = weight_scale_tensor
        else:
            state_dict[key] = quant_weight_tensor
    else:
        state_dict[key] = tensor


def load_checkpoint(model_path, cls, config, return_numpy=True, load_gpu=True):
    """
    load checkpoint
    """
    if getattr(config, "parallel_config", None) is not None:
        use_ep = getattr(config.parallel_config, "use_ep", False)
        tensor_parallel_degree = config.parallel_config.tensor_parallel_degree
    else:
        use_ep = getattr(config, "use_ep", False)
        tensor_parallel_degree = config.tensor_parallel_degree

    if getattr(config, "model_config", None) is not None:
        model_config = config.model_config
    else:
        model_config = config

    if use_ep:
        state_dict = load_ep_checkpoint(model_path,
                                        config,
                                        return_numpy=True,
                                        return_key_name=True)
    else:
        rank_dirs = [
            f for f in os.listdir(model_path) if f.startswith("rank")
            and os.path.isdir(os.path.join(model_path, f))
        ]
        if len(rank_dirs) > 1:
            if tensor_parallel_degree != len(rank_dirs):
                raise ValueError(
                    f"Your model only supports loading with tp{len(rank_dirs)}"
                )
            state_dict = get_state_dict(model_path, model_config)
        else:
            state_dict = load_tp_checkpoint(model_path,
                                            cls,
                                            model_config,
                                            return_numpy=return_numpy)
            import re
            for k, v in state_dict.items():
                match = re.search(r'layers\.(\d+)', k)
                if match and int(match.group(1)) > 0:
                    continue
    return state_dict


def parser_quant_type(quant_type):
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
    default_type = paddle.get_default_dtype()
    if quant_type == "default" or quant_type is None:
        return default_type, default_type, default_type
    conver_dict = {
        "8": "int8",
        "4": "int4",
        "16": paddle.get_default_dtype(),
        "fp8": "float8_e4m3fn",
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32"
    }
    cache_type = default_type
    if "c8" in quant_type:
        cache_type = "int8"
    elif "cfp8" in quant_type:
        cache_type = "fp8"
    elif "c4" in quant_type:
        cache_type = "int4"

    if "weight_only_int8" in quant_type or "wint8" in quant_type:
        return "int8", default_type, cache_type
    elif "weight_only_int4" in quant_type or "wint4" in quant_type:
        return "int4", default_type, cache_type
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
            quant_type_list.append(conver_dict[splited_type[w_idx + 1]])
        else:
            quant_type_list.append(default_type)
        if "a" in splited_type:
            a_idx = splited_type.index("a")
            quant_type_list.append(conver_dict[splited_type[a_idx + 1]])
        else:
            quant_type_list.append(default_type)
        if "c" in splited_type:
            c_idx = splited_type.index("c")
            quant_type_list.append(conver_dict[splited_type[c_idx + 1]])
        else:
            quant_type_list.append(default_type)

        return quant_type_list[0], quant_type_list[1], quant_type_list[2]
