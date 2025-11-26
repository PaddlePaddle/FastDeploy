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

import contextlib
import copy
import hashlib
import inspect
import json
import os
import pickle
import time
from functools import wraps
from pathlib import Path

import paddle
import paddle.distributed as dist
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.model_utils import load_tp_checkpoint
from paddleformers.utils.log import logger
from paddleformers.utils.safetensors import fast_safe_open
from safetensors import safe_open
from tqdm import tqdm

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.linear import KVBatchLinear
from fastdeploy.model_executor.models.tp_utils import (
    check_tensor_parallel_prerequisites,
)
from fastdeploy.model_executor.utils import multi_switch_config_context
from fastdeploy.platforms import current_platform


def pdparams_weight_iterator(paddle_file_list: list[str]):
    for pdparams_file in tqdm(
        paddle_file_list,
        desc="Loading pdparams checkpoint shards",
    ):
        state_dict = paddle.load(pdparams_file)
        yield from state_dict.items()
        del state_dict


def load_weights_from_cache(model, weights_iterator):
    params_dict = dict(model.named_parameters())
    for loaded_weight_name, loaded_weight in weights_iterator:
        if loaded_weight_name not in params_dict:
            logger.info(f"{loaded_weight_name} is not in model parameters.")
            continue
        param = params_dict[loaded_weight_name]
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch between loaded weight {loaded_weight_name}: {loaded_weight.shape}, expected shape: {param.shape}"
            )
        param.copy_(loaded_weight, False)
        if "embeddings" in loaded_weight_name and getattr(model, "tie_word_embeddings", False):
            model.lm_head.linear.weight.set_value(loaded_weight.transpose([1, 0]))
        for _, model_sublayer in model.named_sublayers():
            if isinstance(model_sublayer, KVBatchLinear):
                model_sublayer.process_weights_after_loading()


def get_weight_iterator(model_path: str):
    _, files_list, use_safetensors = get_all_weights_file(model_path)
    if use_safetensors:
        weights_iterator = safetensors_weights_iterator(files_list)
    else:
        weights_iterator = pdparams_weight_iterator(files_list)
    return weights_iterator


def is_weight_cache_enabled(fd_config, weight_cache_path=".cache"):

    weight_cache_context = contextlib.nullcontext()
    weight_cache_dir = None
    enable_cache = False
    if envs.FD_ENABLE_MODEL_LOAD_CACHE and fd_config.quant_config is not None:
        model_weight_cache_path = os.path.join(fd_config.model_config.model, weight_cache_path)
        # model_type + quantization + tp_size + ep_size
        weight_cache_key = "_".join(
            [
                fd_config.model_config.model_type,
                fd_config.quant_config.name(),
                str(fd_config.parallel_config.tensor_parallel_size),
                str(fd_config.parallel_config.expert_parallel_size),
            ]
        )
        # only support tp now
        hash_key = hashlib.md5(pickle.dumps(weight_cache_key)).hexdigest()
        weight_cache_dir = os.path.join(model_weight_cache_path, hash_key)
        if os.path.exists(weight_cache_dir):
            logger.info(
                f"Loading will prioritize cached models. Users are responsible for ensuring the saved model is correct. If any error occurs, deleting the cache at {weight_cache_dir} may resolve it."
            )
            enable_cache = True

            weight_cache_context = multi_switch_config_context(
                (fd_config.quant_config, "is_checkpoint_bf16", False),
            )

    return enable_cache, weight_cache_dir, weight_cache_context


def save_model(model_arg_name="model", config_arg_name="fd_config"):
    @measure_time("Model saving")
    def _save_model(model_dict, weight_cache_dir):
        # Note: ProcessGroupNCCL do not support deepcopy protocol, we made modifications here.
        paddle.distributed.communication.group.Group.__deepcopy__ = lambda self, _: self
        paddle.distributed.communication.group.Group.to_json = lambda self: repr(self)
        paddle.save(model_dict, weight_cache_dir)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            fd_config = bound_args.arguments.get(config_arg_name, None)
            model = bound_args.arguments.get(model_arg_name, None)
            enable_cache, weight_cache_dir, _ = is_weight_cache_enabled(fd_config)
            assert fd_config is not None, "fd_config cannot be None"
            assert model is not None, "model cannot be None"
            if enable_cache:
                tp_weight_cache_dir = os.path.join(
                    weight_cache_dir, f"rank{str(fd_config.parallel_config.tensor_parallel_rank)}"
                )
                context = multi_switch_config_context((fd_config.model_config, "model", tp_weight_cache_dir))
            else:
                context = contextlib.nullcontext()

            with context:
                result = func(*args, **kwargs)

            if envs.FD_ENABLE_MODEL_LOAD_CACHE:
                if not (
                    fd_config.quant_config is not None and getattr(fd_config.quant_config, "is_checkpoint_bf16", False)
                ):
                    # Save cache only for dynamic quantization
                    return result
                if weight_cache_dir is None:
                    return result
                tp_weight_cache_dir = os.path.join(
                    weight_cache_dir, f"rank{str(fd_config.parallel_config.tensor_parallel_rank)}"
                )
                if not os.path.exists(tp_weight_cache_dir):
                    logger.info(f"Saving model to {tp_weight_cache_dir}")
                    os.makedirs(
                        tp_weight_cache_dir,
                        exist_ok=True,
                    )
                    _save_model(model.state_dict(), os.path.join(tp_weight_cache_dir, "cache.pdparams"))
                else:
                    reason = "weights already cached" if envs.FD_ENABLE_MODEL_LOAD_CACHE else "cache disabled"
                    logger.info(f"Skip saving ,{reason}")
            return result

        return wrapper

    return decorator


def measure_time(prefix: str = "Model loading"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time_before = time.time()
            result = func(*args, **kwargs)
            time_after = time.time()
            logger.info(f"{prefix} took {time_after - time_before:.3f} seconds")
            return result

        return wrapper

    return decorator


def load_reordered_experts(model_path: str, key_name: str):
    from safetensors import safe_open

    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        weight_list = json.load(f)["weight_map"]
    safetensor_path = os.path.join(model_path, weight_list[key_name])
    with safe_open(safetensor_path, framework="np", device="cpu") as f:
        if key_name in f.keys():
            weight = f.get_tensor(key_name)
            weight = paddle.Tensor(weight, zero_copy=True)
            weight = weight._copy_to(paddle.framework._current_expected_place(), False)
            return weight


def load_ep_checkpoint(cls: PretrainedModel, model_path: str, fd_config: FDConfig, return_numpy: bool = False):
    """
    load ep checkpoint
    """
    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        weight_list = json.load(f)["weight_map"]
    filtered_map = {k: v for k, v in weight_list.items() if ".experts." not in k}
    num_local_ffn_keys = []

    from itertools import chain

    def get_expert_ranges(fd_config):
        """
        Generate expert index ranges based on configuration parameters

        This function is primarily used in Mixture-of-Experts (MoE) models to generate
        expert index ranges according to configuration parameters. When moe_num_experts
        is a list in the fd_config, it returns a chained combination of two ranges, otherwise
        returns a single range.

        Args:
            fd_config: FastDeploy Configuration object

        Returns:
            If moe_num_experts is a list:
                Returns a chained combination (chain object) of two ranges:
                    1. Base range: [num_experts_start_offset, num_experts_start_offset + num_experts_per_rank)
                    2. Offset range: [base_range.start + moe_num_experts[0], base_range.stop + moe_num_experts[0])
            Else:
                Returns single range: [num_experts_start_offset, num_experts_start_offset + num_experts_per_rank)
        """
        base_range = range(
            fd_config.parallel_config.num_experts_start_offset,
            fd_config.parallel_config.num_experts_start_offset + fd_config.parallel_config.num_experts_per_rank,
        )
        if isinstance(fd_config.model_config.moe_num_experts, list):
            return chain(
                base_range,
                range(
                    base_range.start + fd_config.model_config.moe_num_experts[0],
                    base_range.stop + fd_config.model_config.moe_num_experts[0],
                ),
            )
        return base_range

    prefix_layer_name = (
        "mtp_block" if getattr(fd_config.speculative_config, "model_type", "main") == "mtp" else "layers"
    )

    for i in range(fd_config.model_config.moe_layer_start_index, fd_config.model_config.num_hidden_layers):
        for j in get_expert_ranges(fd_config):
            up_gate_proj_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.up_gate_proj.weight"
            down_proj_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.down_proj.weight"

            up_gate_proj_quant_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.up_gate_proj.quant_weight"
            down_proj_quant_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.down_proj.quant_weight"

            up_gate_proj_scale_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.up_gate_proj.weight_scale"
            down_proj_scale_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.down_proj.weight_scale"

            down_proj_in_scale_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.down_proj.activation_scale"
            num_local_ffn_keys.append(up_gate_proj_key)
            num_local_ffn_keys.append(down_proj_key)
            num_local_ffn_keys.append(up_gate_proj_quant_key)
            num_local_ffn_keys.append(down_proj_quant_key)
            num_local_ffn_keys.append(up_gate_proj_scale_key)
            num_local_ffn_keys.append(down_proj_scale_key)
            num_local_ffn_keys.append(down_proj_in_scale_key)

        # for EP w4a8, we need all expert's activation_scale for up_gate_proj
        num_experts = fd_config.model_config.moe_num_experts
        if isinstance(num_experts, list):
            num_experts = num_experts[0]

        for j in range(num_experts):
            up_gate_proj_in_scale_key = f"ernie.{prefix_layer_name}.{i}.mlp.experts.{j}.up_gate_proj.activation_scale"
            num_local_ffn_keys.append(up_gate_proj_in_scale_key)

    for k in num_local_ffn_keys:
        if k in weight_list:
            filtered_map[k] = weight_list[k]

    if fd_config.parallel_config.tensor_parallel_size > 1:
        no_tp_action_keys = copy.deepcopy(num_local_ffn_keys)
        if fd_config.parallel_config.use_sequence_parallel_moe:
            for i in range(fd_config.model_config.moe_layer_start_index, fd_config.model_config.num_hidden_layers):
                k = f"ernie.{prefix_layer_name}.{i}.self_attn.o_proj.weight"
                if k in weight_list:
                    no_tp_action_keys.append(k)
        tp_actions = cls._get_tensor_parallel_mappings(fd_config.model_config.pretrained_config)
        new_actions = {k: v for k, v in tp_actions.items() if k not in no_tp_action_keys}

    state_dict = {}
    # Get all safetensor file paths that need to be opened
    safetensor_paths = set(filtered_map.values())

    # Open each safetensor file sequentially with progress bar
    for safetensor_path in tqdm(safetensor_paths, desc="Loading safetensor files", unit="file"):
        with safe_open(
            os.path.join(model_path, safetensor_path),
            framework="np",
            device="cpu",
        ) as f:
            # Check if this file contains keys from filtered_map
            for k in filtered_map:
                if filtered_map[k] == safetensor_path and k in f.keys():
                    weight = f.get_tensor(k)
                    if fd_config.parallel_config.tensor_parallel_size > 1:
                        if k in new_actions:
                            weight = new_actions[k](weight)
                    if not return_numpy:
                        weight = paddle.Tensor(weight, zero_copy=True)
                        weight = weight._copy_to(paddle.framework._current_expected_place(), False)
                    state_dict[k] = weight
    return state_dict


def safetensors_weights_iterator(safe_tensor_list: list[str]):
    """
    safetensors_weights_iterator
    """
    for st_file in tqdm(
        safe_tensor_list,
        desc="Loading safetensors checkpoint shards",
    ):
        with safe_open(st_file, framework="paddle", device="cpu") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def fast_weights_iterator(safe_tensor_list: list[str]):
    """
    paddleformers' iterator for safetensors
    """
    for st_file in tqdm(
        safe_tensor_list,
        desc="Loading safetensors checkpoint shards",
    ):
        with fast_safe_open(st_file, framework="np") as f:
            for name in f.keys():
                param_slice = f.get_slice(name)
                yield name, param_slice


def fastsafetensors_weights_iterator(
    safetensor_list: list[str],
):
    """
    Return an iterator over tensors on GPU from a given safetensor_list.
    """
    world_size = dist.get_world_size()
    if world_size > 1:
        pg = dist.get_group()
        device = f"gpu:{pg.rank}" if paddle.is_compiled_with_cuda() else "cpu"
    else:
        pg = SingleGroup()
        device = f"gpu:{pg.rank()}" if paddle.is_compiled_with_cuda() else "cpu"

    safetensor_files_sub_lists = [
        safetensor_list[i : i + world_size] for i in range(0, len(safetensor_list), world_size)
    ]

    for st_file in tqdm(
        safetensor_files_sub_lists,
        desc="Loading fastsafetensors checkpoint shards",
    ):
        loader = SafeTensorsFileLoader(pg, device, nogds=True, debug_log=False, framework="paddle")
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


def load_pre_sharded_checkpoint(model_path: str, local_rank: int, use_fastsafetensor: bool = False):
    """
    load_pre_sharded_checkpoint
    """

    state_dict = {}
    _, safetensor_files, _ = get_all_weights_file(os.path.join(model_path, f"rank{local_rank}"))
    weights_iterator = safetensors_weights_iterator(safetensor_files)
    for name, weight in weights_iterator:
        state_dict[name] = weight.clone()
    return state_dict


def get_all_weights_file(model_path: str):
    """
    get_all_safetensors
    """
    model_path = Path(model_path)
    use_safetensors = True
    files_list = [str(file) for file in model_path.glob("*.pdparams") if file.name != "scheduler.pdparams"]
    if len(files_list) > 0:
        key_name_list = []
        use_safetensors = False
    else:
        safe_model_path = model_path / "model.safetensors"
        if safe_model_path.exists():
            files_list = [str(safe_model_path)]
            with safe_open(safe_model_path, framework="np", device="cpu") as f:
                key_name_list = f.keys()
            return key_name_list, files_list, use_safetensors
        else:
            index_file = model_path / "model.safetensors.index.json"
            with index_file.open("r") as f:
                weight_map = json.load(f)["weight_map"]
            weight_files_in_index = {str(model_path / weight_map[name]) for name in weight_map}
            key_name_list = list(weight_map.keys())
            files_list = sorted(weight_files_in_index)
    return key_name_list, files_list, use_safetensors


def load_tp_checkpoint_v1(
    model_path: str,
    cls: PretrainedModel,
    fd_config: FDConfig,
    use_fastsafetensor: bool = True,
):
    """
    load_tp_checkpoint_v1
    """

    safetensor_keys, safetensor_files, _ = get_all_weights_file(model_path)

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
        if src._is_initialized() and not isinstance(src.place, paddle.CUDAPinnedPlace):
            dst = src._copy_to(device, True)
            dst_tensor = dst.value().get_tensor()
            src_tensor = src.value().get_tensor()
            src_tensor._clear()
            src_tensor._share_data_with(dst_tensor)


def load_cache_scale(fd_config, state_dict):
    file_path = fd_config.model_config.kv_cache_quant_scale_path
    prefix_layer_name = fd_config.model_config.prefix_layer_name
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            for i in range(fd_config.model_config.num_hidden_layers):

                k_scale_name = f"ernie.{prefix_layer_name}.{i}.self_attn.cachek_matmul.activation_scale"
                v_scale_name = f"ernie.{prefix_layer_name}.{i}.self_attn.cachev_matmul.activation_scale"

                k_scale = data[k_scale_name]
                k_scale_tensor = paddle.to_tensor(k_scale, dtype=paddle.get_default_dtype())
                state_dict[k_scale_name] = k_scale_tensor * 448.0

                v_scale = data[v_scale_name]
                v_scale_tensor = paddle.to_tensor(v_scale, dtype=paddle.get_default_dtype())
                state_dict[v_scale_name] = v_scale_tensor * 448.0

                logger.info(f"Loaded kv cache scales for layer {i}.")
    else:
        logger.warning(f"No kv_cache_scale.json found at {file_path}, skipping...")


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
        state_dict = load_ep_checkpoint(cls, model_path, fd_config, return_numpy=True)
    else:
        rank_dirs = [
            f for f in os.listdir(model_path) if f.startswith("rank") and os.path.isdir(os.path.join(model_path, f))
        ]
        if len(rank_dirs) > 1:
            if fd_config.parallel_config.tensor_parallel_size != len(rank_dirs):
                raise ValueError(f"Your model only supports loading with tp{len(rank_dirs)}")
            state_dict = load_pre_sharded_checkpoint(
                model_path,
                fd_config.parallel_config.tensor_parallel_rank,
                use_fastsafetensor=False,
            )
        else:
            if fd_config.load_config.use_fastsafetensor and (
                current_platform.available() and current_platform.is_cuda()
            ):
                state_dict = load_tp_checkpoint_v1(model_path, cls, fd_config, use_fastsafetensor=True)
                deal_state_dict(state_dict)
            else:
                fd_config.model_config.pretrained_config.use_sequence_parallel_moe = (
                    fd_config.parallel_config.use_sequence_parallel_moe
                )
                # NOTE: for very big model, cpu will be out of memory
                state_dict = load_tp_checkpoint(
                    model_path,
                    cls,
                    fd_config.model_config.pretrained_config,
                    return_numpy=return_numpy,
                )
    if not state_dict:
        raise ValueError("weight not found in state_dict !")

    if hasattr(fd_config.quant_config, "kv_cache_quant_type"):
        kv_cache_quant_type = fd_config.quant_config.kv_cache_quant_type
        if kv_cache_quant_type == "float8_e4m3fn":
            load_cache_scale(fd_config, state_dict)

    return state_dict
