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
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.platforms import current_platform


class BitMaskTracker:
    def __init__(self, length: int):
        """
        Track filling status along a single dimension using a bitmask.

        Args:
            length (int): Number of positions to track (e.g., columns or rows)
        """
        self.length = length
        self.mask = 0

    def mark(self, start: int, end: int):
        """
        Mark the range [start, end) as filled.

        Args:
            start (int): Start index (inclusive)
            end (int): End index (exclusive)
        """
        if start < 0 or end > self.length or start >= end:
            raise ValueError("Invalid mark range")
        block = ((1 << (end - start)) - 1) << start
        self.mask |= block

    def is_full(self) -> bool:
        """Return True if all positions are filled."""
        return self.mask == (1 << self.length) - 1


class TensorTracker:
    def __init__(self, shape: tuple, output_dim: int):
        """
        Unified tracker for 2D or 3D tensors.

        Args:
            shape (tuple): Tensor shape
            output_dim (bool):
                - 2D: True = track columns (dim=1), False = track rows (dim=0)
                - 3D: True = track columns (dim=2), False = track rows (dim=1)
        """
        self.shape = shape
        self.output_dim = output_dim

        if len(shape) == 2:
            self.track_dim = 1 if output_dim else 0
            self.trackers = [BitMaskTracker(shape[self.track_dim])]
        elif len(shape) == 3:
            batch = shape[0]
            self.track_dim = 2 if output_dim else 1
            self.trackers = [BitMaskTracker(shape[self.track_dim]) for _ in range(batch)]
        else:
            raise ValueError("Only 2D or 3D tensors supported")

    def mark(self, start: int = 0, end: int = None, batch_id: int = None):
        """
        Mark a slice of the tensor as filled.

        Args:
            batch_id (int, optional): Batch index for 3D tensors
            start (int): Start index along tracked dimension
            end (int): End index along tracked dimension
        """
        if end is None:
            end = self.shape[self.track_dim]

        if len(self.shape) == 2:
            self.trackers[0].mark(start, end)
        else:
            if batch_id is None:
                raise ValueError("batch_id must be provided for 3D tensor")
            self.trackers[batch_id].mark(start, end)

    def is_fully_copied(self) -> bool:
        """Return True if the tensor is fully filled along tracked dimension(s)."""
        return all(tr.is_full() for tr in self.trackers)


def set_weight_attrs(param, param_attr_map: Optional[dict[str, Any]]):
    if param_attr_map is None:
        return
    for key, value in param_attr_map.items():
        setattr(param, key, value)


def slice_fn(weight_or_paramter, output_dim, start, end, step=1):
    if hasattr(weight_or_paramter, "get_shape"):
        shape = weight_or_paramter.get_shape()
    else:
        shape = weight_or_paramter.shape
    if len(shape) == 1:
        weight_or_paramter = weight_or_paramter[start:end]
    elif output_dim:
        weight_or_paramter = weight_or_paramter[..., start:end]
    else:
        weight_or_paramter = weight_or_paramter[start:end, ...]
    return weight_or_paramter


def process_weight_transpose(layer, weight_name):
    weight = getattr(layer, weight_name)
    if len(weight.shape) == 2:
        weight_shape = weight.shape[::-1]
    elif len(weight.shape) == 3:
        weight_shape = [weight.shape[0]] + list(weight.shape[1:][::-1])
    weight_tmp = layer.create_parameter(
        shape=weight_shape,
        dtype=weight.dtype,
        default_initializer=paddle.nn.initializer.Constant(0),
        is_bias=False,
    )
    if layer.fd_config.load_config.dynamic_load_weight or layer.fd_config.model_config.enable_cache:
        free_tensor(weight)
        setattr(layer, weight_name, weight_tmp)
        return

    if len(weight.shape) == 2:
        weight_transpose = weight.transpose([1, 0])
    elif len(weight.shape) == 3:
        weight_transpose = weight.transpose([0, 2, 1])
    weight_tmp.copy_(weight_transpose, False)
    free_tensor(weight)
    setattr(layer, weight_name, weight_tmp)


def process_weights_after_loading(sublayers_dict: dict, fd_config: FDConfig):
    """
    process_weights_after_loading:
    """

    def fn(model_sublayer_name: str, param=None):
        from fastdeploy.model_executor.layers.linear import (
            KVBatchLinear,
            UnquantizedLinearMethod,
        )
        from fastdeploy.model_executor.layers.moe.moe import get_moe_method

        if model_sublayer_name not in sublayers_dict:
            return
        model_sublayer = sublayers_dict[model_sublayer_name]
        if isinstance(model_sublayer, KVBatchLinear):
            model_sublayer.process_weights_after_loading()
        if fd_config.quant_config and not fd_config.quant_config.is_checkpoint_bf16:
            # skip for offline quantization
            return
        if hasattr(model_sublayer, "quant_method"):
            quant_method = getattr(model_sublayer, "quant_method", None)
            unquant_moe_layer = get_moe_method()
            if unquant_moe_layer is None:
                unquant_moe_cls = object
            else:
                unquant_moe_cls = type(unquant_moe_layer)
            if type(quant_method) is UnquantizedLinearMethod or type(quant_method) is unquant_moe_cls:
                # skip unquantized linear
                return
            if not hasattr(quant_method, "process_weights_after_loading"):
                return
            if param is not None and hasattr(param, "tensor_track") and param.tensor_track is None:
                return
            if param is not None and hasattr(param, "tensor_track") and not param.tensor_track.is_fully_copied():
                return
            quant_method.process_weights_after_loading(model_sublayer)

    return fn


@dataclass
class WeightsMapper:
    orig_to_new_prefix: Mapping[str, Optional[str]] = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                key = key.replace(prefix, new_key, 1)
        return key

    def apply(self, weight_name):
        return self._map_name(weight_name)


def process_weights_before_loading(
    *, skip_prefixes: Optional[List[str]] = None, mapper: Optional[WeightsMapper] = None
):
    def _can_skip(weight_name):
        return any(weight_name.startswith(p) for p in (skip_prefixes or []))

    def fn(weight_name):
        if mapper is not None:
            weight_name = mapper.apply(weight_name)
        if _can_skip(weight_name):
            weight_name = None
        return weight_name

    return fn


def weight_fully_copied(weight):
    return (
        hasattr(weight, "tensor_track") and weight.tensor_track is not None and weight.tensor_track.is_fully_copied()
    )


def process_final_after_loading(model, fd_config: FDConfig):
    # process_final_after_loading handles the post-loading process for cases other than dynamic quantization.
    from fastdeploy.model_executor.layers.linear import (
        KVBatchLinear,
        UnquantizedLinearMethod,
    )
    from fastdeploy.model_executor.layers.moe.moe import get_moe_method

    for name, sublayer in model.named_sublayers():
        if isinstance(sublayer, KVBatchLinear):
            continue
        quant_method = getattr(sublayer, "quant_method", None)
        if quant_method is not None:
            unquant_moe_layer = get_moe_method()
            if unquant_moe_layer is None:
                unquant_moe_cls = object
            else:
                unquant_moe_cls = type(unquant_moe_layer)
            is_unquant_cls = type(quant_method) is UnquantizedLinearMethod or type(quant_method) is unquant_moe_cls
            is_offline_quantized_ckpt = not (fd_config.quant_config and fd_config.quant_config.is_checkpoint_bf16)
            if is_unquant_cls or is_offline_quantized_ckpt:
                if hasattr(quant_method, "process_weights_after_loading"):
                    quant_method.process_weights_after_loading(sublayer)
                continue
        if not hasattr(sublayer, "process_weights_after_loading"):
            continue
        sublayer.process_weights_after_loading()


def free_tensor(tensor):
    if hasattr(tensor, "tensor_track"):
        tensor.tensor_track = None
    tensor.value().get_tensor()._clear()
    del tensor


def fd_cast(weight, param):
    if weight.dtype != param.dtype:
        if weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
            weight = weight.view(param.dtype)
        else:
            weight = weight.cast(param.dtype)
    return weight


def default_weight_loader(fd_config: FDConfig = None) -> None:
    """Default weight loader"""

    def fn(param, loaded_weight, shard_id: Optional[Union[int, str]] = None):
        """fn"""

        output_dim = getattr(param, "output_dim", None)
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if weight_need_transpose:
            loaded_weight = loaded_weight.transpose([1, 0])
        # Tensor parallelism splits the weight along the output_dim
        if output_dim is not None and fd_config is not None and fd_config.parallel_config.tensor_parallel_size > 1:
            dim = -1 if output_dim else 0
            if isinstance(loaded_weight, paddle.Tensor):
                size = loaded_weight.shape[dim]
            else:
                size = loaded_weight.get_shape()[dim]
            block_size = size // fd_config.parallel_config.tensor_parallel_size
            shard_offset = fd_config.parallel_config.tensor_parallel_rank * block_size
            shard_size = (fd_config.parallel_config.tensor_parallel_rank + 1) * block_size
            loaded_weight = slice_fn(loaded_weight, output_dim, shard_offset, shard_size)

        tp_row_bias = getattr(param, "tp_row_bias", None)
        if tp_row_bias:
            loaded_weight = loaded_weight / fd_config.parallel_config.tensor_parallel_size

        # mlp.gate.weight is precision-sensitive, so we cast it to float32 for computation
        loaded_weight = fd_cast(loaded_weight, param)
        if param.shape != loaded_weight.shape:
            # for e_score_correction_bias
            loaded_weight = loaded_weight.reshape(param.shape)
        assert param.shape == loaded_weight.shape, (
            f" Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({param.shape})"
        )
        h2d_copy(dst=param, src=loaded_weight)

    return fn


def is_pre_sliced_weight(model_path):
    rank_dirs = [
        f for f in os.listdir(model_path) if f.startswith("rank") and os.path.isdir(os.path.join(model_path, f))
    ]
    return len(rank_dirs) > 1


def is_paddle_support_v1_loader():
    src_shape = [32, 32]
    tgt_shape = [1, 32, 64]
    src_tensor = paddle.ones(src_shape, dtype="float32")
    tgt_tensor = paddle.zeros(tgt_shape, dtype="float32")
    for exp_id in range(tgt_shape[0]):
        # gate
        gate_tgt = tgt_tensor[exp_id][..., : tgt_shape[2] // 2]
        gate_tgt.copy_(src_tensor, False)
        # up
        up_tgt = tgt_tensor[exp_id][..., tgt_shape[2] // 2 :]
        up_tgt.copy_(src_tensor, False)
    is_same = bool(paddle.all(tgt_tensor == 1))
    return is_same


_support_new_h2d = None


def is_paddle_support_new_h2d():
    import subprocess
    import sys

    global _support_new_h2d
    if _support_new_h2d is not None:
        return _support_new_h2d

    code = """
import paddle
try:
    dst = paddle.zeros([2, 4], dtype='bfloat16')
    src = paddle.ones([2, 2], dtype='bfloat16', device='cpu')
    dst = dst[..., :2]
    dst.copy_(src)
    print(1)
except:
    print(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    _support_new_h2d = result.stdout.strip() == b"1"
    return _support_new_h2d


def h2d_copy(dst, src, blocking=True):
    if not current_platform.is_cuda() or not is_paddle_support_new_h2d():
        # For non-GPU devices, data is transferred to device (H2D) in advance.
        src = get_tensor(src)
    if not dst._is_initialized():
        dst.initialize()
    dst.copy_(src, blocking)


def v1_loader_support(fd_config):
    _v1_no_support_archs = ["Qwen2VLForConditionalGeneration"]

    def _get_unsupported_quant():
        if current_platform.is_cuda():
            return {"w4a8", "w4afp8", "wint2"}
        elif current_platform.is_xpu():
            return {"w4a8", "w8a8"}
        return set()

    def _err_msg(msg: str) -> str:
        logger.info(msg + "; fallback to the v0 loader for model loading.")

    if not (
        current_platform.is_cuda()
        or current_platform.is_xpu()
        or current_platform.is_iluvatar()
        or current_platform.is_maca()
    ):
        _err_msg("v1loader currently only support backends gpu, xpu, iluvatar and maca")
        return False

    if is_pre_sliced_weight(fd_config.model_config.model):
        _err_msg("v1 loader currently does not support pre-sliced weights")
        return False

    if envs.FD_MOE_BACKEND.lower() == "marlin":
        _err_msg("v1 loader currently does not support marlin backend")
        return False

    if fd_config.quant_config is not None:
        if fd_config.quant_config.name() == "mix_quant":
            moe_quant_type = fd_config.quant_config.moe_quant_type
            dense_quant_type = fd_config.quant_config.dense_quant_type
        else:
            moe_quant_type = fd_config.quant_config.name()
            dense_quant_type = fd_config.quant_config.name()
        unsupported_quant = _get_unsupported_quant()

        if unsupported_quant & {moe_quant_type, dense_quant_type}:
            _err_msg("v1 loader currently does not support w4a8/w4afp8/win2 quantization")
            return False
    if fd_config.model_config.architectures[0] in _v1_no_support_archs:
        _err_msg(f"v1 loader currently does not support {fd_config.model_config.architectures[0]}")
        return False

    if not is_paddle_support_v1_loader():
        _err_msg("The installed Paddle does not support v1 loader")
        return False
    return True


@contextmanager
def temporary_dtype(dtype: str):
    """Temporarily set Paddle default dtype"""
    orig_dtype = paddle.get_default_dtype()
    try:
        if dtype is not None and dtype == "float32":
            paddle.set_default_dtype(dtype)
        yield
    finally:
        paddle.set_default_dtype(orig_dtype)


@contextmanager
def multi_switch_config_context(*changes):
    """
    changes: (obj, attr, new_value)
    """
    originals = []
    try:
        for obj, attr, new_value in changes:
            old_value = getattr(obj, attr)
            originals.append((obj, attr, old_value))
            setattr(obj, attr, new_value)
        yield
    finally:
        for obj, attr, old_value in originals:
            setattr(obj, attr, old_value)


def rename_offline_ckpt_suffix_to_fd_suffix(
    fd_config, ckpt_weight_suffix: str = "quant_weight", ckpt_scale_suffix="weight_scale"
):
    """
    Create a function to rename checkpoint key suffixes for FastDeploy.

    Replaces the original suffix (default "weight_scale") with the FD target
    suffix (default "quant_weight"). Only the suffix is changed.

    Args:
        fd_config: FastDeploy configuration.
        ckpt_weight_suffix: Original checkpoint key suffix.
        ckpt_scale_suffix: Target FastDeploy key suffix.

    Returns:
        Callable: Function that renames checkpoint keys.
    """
    fd_suffix_map = {}  # noqa: F841
    fp8_suffix_map = {
        ckpt_weight_suffix: "weight",
        ckpt_scale_suffix: "weight_scale_inv",
    }
    moe_quant_type = ""
    dense_quant_type = ""
    if fd_config.quant_config is not None:
        if fd_config.quant_config.name() == "mix_quant":
            moe_quant_type = fd_config.quant_config.moe_quant_type
            dense_quant_type = fd_config.quant_config.dense_quant_type
        else:
            moe_quant_type = fd_config.quant_config.name()
            dense_quant_type = fd_config.quant_config.name()

    def fn(loaded_weight_name, is_moe):
        if fd_config.quant_config is None or fd_config.quant_config.is_checkpoint_bf16:
            return loaded_weight_name
        # Can be extended to other offline quantization suffixes if needed.
        if (is_moe and moe_quant_type == "block_wise_fp8") or (not is_moe and dense_quant_type == "block_wise_fp8"):
            fd_suffix_map = fp8_suffix_map
        for ckpt_suffix, fd_suffix in fd_suffix_map.items():
            if re.search(rf"{ckpt_suffix}$", loaded_weight_name):
                loaded_weight_name = loaded_weight_name.replace(ckpt_suffix, fd_suffix)
                return loaded_weight_name
        return loaded_weight_name

    return fn
