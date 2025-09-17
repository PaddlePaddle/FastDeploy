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

import itertools
import re
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import paddle
import paddle.nn as nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.utils import get_logger

logger = get_logger("utils", "utils.log")


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


def process_weights_after_loading(sublayers_dict: dict):
    """
    process_weights_after_loading: e.g., handle extracted weights (quantization, reshaping, etc.)
    """

    def fn(model_sublayer_name: str, param=None):
        from fastdeploy.model_executor.layers.linear import KVBatchLinear

        if model_sublayer_name not in sublayers_dict:
            return
        model_sublayer = sublayers_dict[model_sublayer_name]
        if isinstance(model_sublayer, KVBatchLinear):
            model_sublayer.process_weights_after_loading()
        if hasattr(model_sublayer, "quant_method"):
            quant_method = getattr(model_sublayer, "quant_method", None)
            if not hasattr(quant_method, "process_weights_after_loading"):
                return
            if param is not None and hasattr(param, "tensor_track") and not param.tensor_track.is_fully_copied():
                return
            quant_method.process_weights_after_loading(model_sublayer)

    return fn


def free_tensor(tensor):
    if hasattr(tensor, "tensor_track"):
        tensor.tensor_track = None
    tensor.value().get_tensor()._clear()
    del tensor


def default_weight_loader(fd_config: FDConfig) -> None:
    """Default weight loader"""

    def fn(param, loaded_weight, shard_id: Optional[Union[int, str]] = None):
        """fn"""
        output_dim = getattr(param, "output_dim", None)
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if weight_need_transpose:
            loaded_weight = get_tensor(loaded_weight)
            loaded_weight = loaded_weight.transpose([1, 0])
        # Tensor parallelism splits the weight along the output_dim
        if output_dim is not None and fd_config.parallel_config.tensor_parallel_size > 1:
            dim = -1 if output_dim else 0
            if isinstance(loaded_weight, paddle.Tensor):
                size = loaded_weight.shape[dim]
            else:
                size = loaded_weight.get_shape()[dim]
            block_size = size // fd_config.parallel_config.tensor_parallel_size
            shard_offset = fd_config.parallel_config.tensor_parallel_rank * block_size
            shard_size = (fd_config.parallel_config.tensor_parallel_rank + 1) * block_size
            loaded_weight = slice_fn(loaded_weight, output_dim, shard_offset, shard_size)

        loaded_weight = get_tensor(loaded_weight)
        # mlp.gate.weight is precision-sensitive, so we cast it to float32 for computation
        if param.dtype != loaded_weight.dtype:
            if loaded_weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
                loaded_weight = loaded_weight.view(param.dtype)
            else:
                loaded_weight = loaded_weight.cast(param.dtype)
        if param.shape != loaded_weight.shape:
            # for e_score_correction_bias
            loaded_weight = loaded_weight.reshape(param.shape)
        assert param.shape == loaded_weight.shape, (
            f" Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({param.shape})"
        )
        param.copy_(loaded_weight, False)

    return fn


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
def switch_config_context(config_obj, config_attr_name, value):
    """switch_config_context"""
    origin_value = getattr(config_obj, config_attr_name)
    setattr(config_obj, config_attr_name, value)
    try:
        yield
    finally:
        setattr(config_obj, config_attr_name, origin_value)


def is_pin_memory_available() -> bool:
    pass


WeightsMapping = Mapping[str, Optional[str]]


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = key.replace(suffix, new_key, 1)

        return key

    def apply(self, weights: Iterable[tuple[str, paddle.Tensor]]) -> Iterable[tuple[str, paddle.Tensor]]:
        return ((out_name, data) for name, data in weights if (out_name := self._map_name(name)) is not None)

    def apply_list(self, values: list[str]) -> list[str]:
        return [out_name for name in values if (out_name := self._map_name(name)) is not None]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {out_name: value for name, value in values.items() if (out_name := self._map_name(name)) is not None}


class AutoWeightsLoader:
    """
    Helper class to load weights into a [`paddle.nn.Layer`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a ``load_weights`` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a ``weight_loader`` method.

    """

    # Models trained using early version ColossalAI
    # may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        layers: nn.Layer,
        *,
        skip_prefixes: Optional[list[str]] = None,
        skip_substrs: Optional[list[str]] = None,
        ignore_unexpected_prefixes: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        self.layers = layers
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, paddle.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, paddle.Tensor]]]]:
        weights_by_parts = ((weight_name.split(".", 1), weight_data) for weight_name, weight_data in weights)

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (("" if len(parts) == 1 else parts[1], weights_data) for parts, weights_data in group),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.ignore_unexpected_prefixes)

    def _load_param(
        self,
        base_prefix: str,
        param: paddle.Tensor,
        weights: Iterable[tuple[str, paddle.Tensor]],
        fd_config: Optional[FDConfig] = None,
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight '{weight_qualname}' " f"into a single parameter '{base_prefix}'"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

            yield weight_qualname

    def _add_loadable_non_param_tensors(self, layer: nn.Layer, child_params: dict[str, paddle.Tensor]):
        """
        Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats.
        """
        if isinstance(
            layer,
            (
                nn.BatchNorm1D,
                nn.BatchNorm2D,
                nn.BatchNorm3D,
                nn.SyncBatchNorm,
            ),
        ):
            layer_state_dict = layer.state_dict()
            for stat_name in ("_mean", "_variance"):
                if stat_name in layer_state_dict:
                    child_params[stat_name.lstrip("_")] = layer_state_dict[stat_name]

    def _load_layer(
        self,
        base_prefix: str,
        layer: nn.Layer,
        weights: Iterable[tuple[str, paddle.Tensor]],
        fd_config: Optional[FDConfig] = None,
    ) -> Iterable[str]:

        if layer != self.layers:
            layer_load_weights = getattr(layer, "load_weights", None)
            if callable(layer_load_weights):
                loaded_params = layer_load_weights(weights)
                if loaded_params is None:
                    logger.warning("Unable to collect loaded parameters " "for layer %s", layer)
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )
                    return

        child_layers = dict(layer.named_sublayers(include_self=False))
        child_params = {}

        for name, param in layer.named_parameters(include_sublayers=False):
            child_params[name] = param

        self._add_loadable_non_param_tensors(layer, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_layers:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping layer %s", prefix)
                    continue

                yield from self._load_layer(prefix, child_layers[child_prefix], child_weights, fd_config)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)
                    continue

                yield from self._load_param(prefix, child_params[child_prefix], child_weights, fd_config)
            else:
                can_skip_layer = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_layer or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)
                    continue

                can_ignore_layer = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_layer or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)
                    continue

                msg = f"There is no module or parameter named '{prefix}' " f"in {type(self.layers).__name__}"
                raise ValueError(msg)

    def load_weights(
        self,
        weights: Iterable[tuple[str, paddle.Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
        fd_config: Optional[FDConfig] = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)
        # Filter out weights with first-prefix/substr to skip in name
        weights = ((name, weight) for name, weight in weights if not self._can_skip(name))

        autoloaded_weights = set(self._load_layer("", self.layers, weights, fd_config))
        return autoloaded_weights


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
    if fd_config.quant_config is None:
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
