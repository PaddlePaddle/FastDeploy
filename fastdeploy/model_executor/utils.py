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

from contextlib import contextmanager
from typing import Any, Optional

import paddle


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


@contextmanager
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


@contextmanager
def switch_config_context(config_obj, config_attr_name, value):
    """switch_config_context"""
    origin_value = getattr(config_obj, config_attr_name)
    setattr(config_obj, config_attr_name, value)
    try:
        yield
    finally:
        setattr(config_obj, config_attr_name, origin_value)
