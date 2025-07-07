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

import concurrent
import concurrent.futures
import contextlib
import json
import os
import re
from typing import Union

import numpy as np
import paddle
import paddle.distributed as dist
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
from paddleformers.transformers import PretrainedModel
from safetensors import safe_open
from tqdm import tqdm

from fastdeploy.config import FDConfig, ModelConfig
from fastdeploy.model_executor.models.tp_utils import \
    check_tensor_parallel_prerequisites
from fastdeploy.platforms import current_platform


def load_ep_checkpoint(model_path: str,
                       config: ModelConfig,
                       return_numpy: bool = False):
    """
    load ep checkpoint
    """
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
            ffn2_key = (f"ernie.layers.{i}.mlp.experts.{j}.down_proj.weight")

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
                       framework="pp",
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


def safetensors_weights_iterator(safe_tensor_list: list[str] ):
    """
    safetensors_weights_iterator
    """
    for st_file in tqdm(
            safe_tensor_list,
            desc="Loading safetensors checkpoint shards",
    ):
        with safe_open(st_file, framework="pp") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def fastsafetensors_weights_iterator(safetensor_list: list[str], ):
    """
    Return an iterator over tensors on GPU from a given safetensor_list.
    """
    world_size = dist.get_world_size()
    if world_size > 1:
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


def load_pre_sharded_checkpoint(model_path: str,
                                local_rank: int,
                                use_fastsafetensor: bool = False):
    """
    load_pre_sharded_checkpoint
    """
    state_dict = {}
    _, safetensor_files = get_all_safetensors(
        os.path.join(model_path, f"rank{local_rank}"))
    weights_iterator = safetensors_weights_iterator(safetensor_files)
    for name, weight in weights_iterator:
        state_dict[name] = weight
    return state_dict


def get_all_safetensors(model_path: str):
    """
    get_all_safetensors
    """
    safe_model_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safe_model_path):
        safetensor_list = [safe_model_path]
        with safe_open(safe_model_path, framework="pp", device="cpu") as f:
            key_name_list = f.keys()
        return key_name_list, safetensor_list
    else:
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



def _add_variant(weights_name: str, variant=None) -> str:
    if variant is not None and len(variant) > 0:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name

@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)

def _split_keys_evenly(keys: list, n: int) -> list:

    total_len = len(keys)
    base_size = total_len // n
    extra = total_len % n

    result = []
    index = 0
    for _ in range(n):
        part_size = base_size + 1 if extra > 0 else base_size
        extra -= 1
        result.append(keys[index : index + part_size])
        index += part_size

    return result

def load_sharded_checkpoint_as_one(folder, variant=None, return_numpy=False):
    pdparams_file = os.path.join(folder, _add_variant("model_state.pdparams", variant))
    lora_pdparams_file = os.path.join(folder, _add_variant("lora_model_state.pdparams", variant))
    safetensors_file = os.path.join(folder, _add_variant("model.safetensors", variant))
    if os.path.isfile(pdparams_file):
        return paddle.load(pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(lora_pdparams_file):
        return paddle.load(lora_pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(safetensors_file):
        state_dict = {}
        with safe_open(safetensors_file, framework="pp") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor()
        if not return_numpy:
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = paddle.Tensor.__call__(state_dict.pop(key), zero_copy=True)
        return state_dict

    PADDLE_WEIGHTS_INDEX_NAME = "model_state.pdparams.index.json"
    SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
    SAFE_MASTER_WEIGHTS_INDEX_NAME = "master_weights.safetensors.index.json"
    SAFE_PEFT_WEIGHTS_INDEX_NAME = "peft_model.safetensors.index.json"

    index_file = os.path.join(folder, _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
    safe_master_file = os.path.join(folder, _add_variant(SAFE_MASTER_WEIGHTS_INDEX_NAME, variant))
    safe_peft_file = os.path.join(folder, _add_variant(SAFE_PEFT_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    safe_master_present = os.path.isfile(safe_master_file)
    safe_peft_present = os.path.isfile(safe_peft_file)

    load_index = None
    if safe_index_present:
        load_index = safe_index_file
    elif safe_master_present:
        load_index = safe_master_file
    elif index_present:
        load_index = index_file
    elif safe_peft_present:
        load_index = safe_peft_file
    else:
        raise ValueError(f"Could not find {index_file} or {safe_index_file} or {safe_peft_file}")

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    ret = {}
    for shard_file in tqdm(shard_files):
        with safe_open(os.path.join(folder, shard_file), framework="pp") as f:
            for key in f.keys():
                ret[key] = f.get_tensor(key)
    if not return_numpy:
        for key in list(ret.keys()):
            if isinstance(ret[key], np.ndarray):
                ret[key] = paddle.Tensor.__call__(ret.pop(key), zero_copy=True)
    return ret

def _load_part_state_dict(
    keys,
    checkpoint_file: Union[str, os.PathLike],
    tensor_parallel_split_mapping,
    fliter_dict_keys,
    return_numpy=False,
):
    part_state_dict = {}
    with safe_open(checkpoint_file, framework="pp") as f:
        for key in keys:
            py_safe_slice_ = f.get_tensor(key)
            if key in tensor_parallel_split_mapping:
                weight = tensor_parallel_split_mapping[key](py_safe_slice_)
            else:
                weight = py_safe_slice_
            if not return_numpy:
                with device_guard():
                    weight = paddle.Tensor.__call__(weight, zero_copy=True)
                weight = weight._copy_to(paddle.framework._current_expected_place(), False)
            part_state_dict[key] = weight
    return part_state_dict

def load_tp_state_dict(checkpoint_file: Union[str, os.PathLike],
    tensor_parallel_split_mapping=None,
    fliter_dict_keys=None,
    device="cpu",
    return_numpy=False):

    if tensor_parallel_split_mapping is None:
        tensor_parallel_split_mapping = {}

    if (
        checkpoint_file.endswith(".safetensors") or re.search(r"\.safetensors_shard_\d{4}$", checkpoint_file)
    ):
        thread_num = int(os.environ.get("LOAD_STATE_DICT_THREAD_NUM", "1"))
        state_dict = {}
        if thread_num <= 1:
            with safe_open(checkpoint_file, framework="pp") as f:
                state_dict = _load_part_state_dict(
                    list(f.keys()),
                    checkpoint_file,
                    tensor_parallel_split_mapping,
                    fliter_dict_keys,
                    return_numpy,
                )
        else:
            # Load state dict in multi-thread to speed up loading
            with safe_open(checkpoint_file, framework="pp") as f:
                keys_groups = _split_keys_evenly(list(f.keys()), thread_num)
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
                future_to_key = {
                    executor.submit(
                        _load_part_state_dict,
                        keys,
                        checkpoint_file,
                        tensor_parallel_split_mapping,
                        fliter_dict_keys,
                        return_numpy,
                    ): keys
                    for keys in keys_groups
                }
                for future in concurrent.futures.as_completed(future_to_key):
                    res_state_dict = future.result()
                    state_dict.update(res_state_dict)

        if not return_numpy:
            if device == "cpu":
                with device_guard():
                    for k in list(state_dict.keys()):
                        state_dict[k] = paddle.Tensor.__call__(state_dict.pop(k), zero_copy=True)
            elif device == "pin_memory":
                for k in list(state_dict.keys()):
                    state_dict[k] = paddle.to_tensor(state_dict.pop(k), place=paddle.CUDAPinnedPlace())

        return state_dict


def load_tp_checkpoint(
    folder: str,
    cls: PretrainedModel,
    config: ModelConfig,
    return_numpy: bool = True,
):
    if config.tensor_parallel_degree == 1 or config.tensor_parallel_degree == -1:
        return load_sharded_checkpoint_as_one(folder, return_numpy=return_numpy)
    rank_model_path = os.path.join(folder, f"model_state.tp0{config.tensor_parallel_rank}.pdparams")
    model_path = os.path.join(folder, "model_state.pdparams")
    safe_model_path = os.path.join(folder, "model.safetensors")
    if os.path.exists(rank_model_path):
        return paddle.load(rank_model_path, return_numpy=return_numpy)
    elif os.path.exists(model_path):
        state_dict = cls.convert_tensor_parallel(model_path, config)
    elif os.path.exists(safe_model_path):
        with safe_open(safe_model_path, framework="pp", device="cpu") as f:
            loaded_keys = f.keys()
        tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_keys)
        state_dict = load_tp_state_dict(safe_model_path, tp_actions, return_numpy=return_numpy)
    else:  # shard files safetensors
        resolved_archive_file, resolved_sharded_files, sharded_metadata, is_sharded = cls._resolve_model_file_path(
            pretrained_model_name_or_path=folder,
            use_safetensors=True,
        )
        if len(resolved_sharded_files) > 1:
            resolved_sharded_files = tqdm(resolved_sharded_files, desc="Loading checkpoint shards")
        loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_state_dict_keys, ignore_error=True)
        state_dict = {}
        for shard_file in resolved_sharded_files:
            shard_state_dict = load_tp_state_dict( # todo: for this function
                shard_file,
                tp_actions,
                loaded_state_dict_keys,
                return_numpy=return_numpy,
            )
            state_dict.update(shard_state_dict)
    return state_dict

def load_tp_checkpoint_v1(
    model_path: str,
    cls: PretrainedModel,
    fd_config: FDConfig,
    use_fastsafetensor: bool = True,
):
    """
    load_tp_checkpoint_v1
    """

    safetensor_keys, safetensor_files = get_all_safetensors(model_path)

    if use_fastsafetensor:
        weights_iterator = fastsafetensors_weights_iterator(safetensor_files)
    else:
        weights_iterator = safetensors_weights_iterator(safetensor_files)

    tensor_parallel_filtered_map = {}
    check_tensor_parallel_prerequisites(
        fd_config,
        cls,
        tensor_parallel_filtered_map,
        safetensor_keys,
    )
    need_tp = True if tensor_parallel_filtered_map else False
    state_dict = {}
    for key, weight in weights_iterator:
        paddle.device.synchronize()
        if need_tp and key in tensor_parallel_filtered_map:
            action = tensor_parallel_filtered_map.pop(key)
            tensor = action(weight).clone()
        else:
            tensor = weight.clone()
        state_dict[key] = tensor
        weight.value().get_tensor()._clear()
    return state_dict


def deal_state_dict(state_dict):
    """deal_state_dict"""
    device = paddle.CUDAPinnedPlace()
    for name, src in state_dict.items():
        if src._is_initialized() and not isinstance(src.place,
                                                    paddle.CUDAPinnedPlace):
            dst = src._copy_to(device, True)
            dst_tensor = dst.value().get_tensor()
            src_tensor = src.value().get_tensor()
            src_tensor._clear()
            src_tensor._share_data_with(dst_tensor)


def load_composite_checkpoint(
    model_path: str,
    cls: PretrainedModel,
    fd_config: FDConfig,
    return_numpy=True,
):
    """
    # This method supports loading model weights under three parallelism strategies:
    # 1. Expert Parallel (EP)
    # 2. Tensor Parallel (TP)
    # 3. Pre-sharded (pre-split)
    """
    if fd_config.parallel_config.use_ep:
        state_dict = load_ep_checkpoint(model_path,
                                        fd_config.model_config,
                                        return_numpy=True)
    else:
        rank_dirs = [
            f for f in os.listdir(model_path) if f.startswith("rank")
            and os.path.isdir(os.path.join(model_path, f))
        ]
        if len(rank_dirs) > 1:
            if fd_config.parallel_config.tensor_parallel_degree != len(
                    rank_dirs):
                raise ValueError(
                    f"Your model only supports loading with tp{len(rank_dirs)}"
                )
            state_dict = load_pre_sharded_checkpoint(
                model_path,
                fd_config.parallel_config.tensor_parallel_rank,
                use_fastsafetensor=False,
            )
        else:
            if fd_config.load_config.use_fastsafetensor and (
                    current_platform.available()
                    and current_platform.is_cuda()):
                state_dict = load_tp_checkpoint_v1(model_path,
                                                   cls,
                                                   fd_config,
                                                   use_fastsafetensor=True)
                deal_state_dict(state_dict)
            else:
                state_dict = load_tp_checkpoint(model_path,
                                                cls,
                                                fd_config.model_config,
                                                return_numpy=return_numpy)
    if not state_dict:
        raise ValueError("weight not found in state_dict !")
    return state_dict
