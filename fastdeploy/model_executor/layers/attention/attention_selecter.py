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

from functools import cache

from fastdeploy import envs
from fastdeploy.platforms import _Backend, current_platform
from fastdeploy.utils import resolve_obj_from_strname


def backend_name_to_enum(backend_name: str) -> _Backend:
    """backend_name_to_enum"""
    assert backend_name is not None
    return _Backend.__members__.get(backend_name)


@cache
def _get_attn_backend(selected_backend: str) -> object:
    """_get_attn_backend"""
    if isinstance(selected_backend, str):
        selected_backend = backend_name_to_enum(selected_backend)
    attention_cls = current_platform.get_attention_backend_cls(selected_backend)

    if not attention_cls:
        raise ValueError(f"Invalid attention backend for {current_platform.device_name}")
    return resolve_obj_from_strname(attention_cls)


def get_attention_backend() -> object:
    """Selects which attention backend."""
    attention_backend = envs.FD_ATTENTION_BACKEND
    return _get_attn_backend(attention_backend)
