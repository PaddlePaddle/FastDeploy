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

from fastdeploy.config import FDConfig


def init_rank_and_device_id(fd_config: FDConfig):
    """ """
    rank = (
        fd_config.parallel_config.expert_parallel_rank * fd_config.parallel_config.tensor_parallel_size
        + fd_config.parallel_config.tensor_parallel_rank
    )

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)

    if cuda_visible_devices is None:
        device_id = rank
    else:
        cuda_visible_devices = cuda_visible_devices.split(",")
        rank_index = rank % len(cuda_visible_devices)
        device_id = cuda_visible_devices[rank_index]

    return rank, device_id


import functools
import inspect
import re

try:
    import torch
except ImportError:
    torch = None

try:
    import paddle
except ImportError:
    paddle = None


class HookManager:
    def __init__(self):
        self.hooked_funcs = {}

    def _set_print_options(self):
        if paddle is not None:
            paddle.set_printoptions(precision=4, threshold=80, edgeitems=10, sci_mode=False, linewidth=120)
        # if torch is not None:
        #     torch.set_printoptions(precision=4, threshold=80, edgeitems=10, sci_mode=False, linewidth=120)

    def _extract_call_argnames(self, func_name):
        """尝试在上层调用源代码中提取函数调用参数字符串"""
        try:
            frame = inspect.currentframe().f_back.f_back  # 跳过 wrapper
            code_context = inspect.getframeinfo(frame).code_context
            if not code_context:
                return []
            src_line = "".join(code_context).strip()

            # 匹配函数名后括号内的内容
            m = re.search(rf"{func_name}\s*\((.*)\)", src_line)
            if not m:
                return []
            arg_str = m.group(1)

            # 简单拆分逗号（不处理嵌套括号复杂表达式）
            arg_names = [s.strip() for s in re.split(r",(?![^()]*\))", arg_str)]
            return arg_names
        except Exception:
            return []

    def _print_value(self, name, value, indent=2, level=0):
        prefix = " " * (indent * level)
        # pad = " " * (indent * (level + 1))

        # torch tensor
        # if torch is not None and isinstance(value, torch.Tensor):
        #     if value.numel() == 1:
        #         print(f"{prefix}[torch] {name}: scalar, dtype={value.dtype}, value={value.item()}")
        #     else:
        #         print(f"{prefix}[torch] {name}: tensor, dtype={value.dtype}, shape={tuple(value.shape)}")
        #         print(f"{pad}{value}")
        #     return

        # paddle tensor
        if paddle is not None and isinstance(value, paddle.Tensor):
            if value.numel() == 1:
                print(f"{prefix}[paddle] {name}: scalar, dtype={value.dtype}, value={value.item()}")
            else:
                print(f"{prefix}[paddle] {name}: tensor, dtype={value.dtype}, shape={tuple(value.shape)}")
                # print(f"{pad}{value}")
            return

        # 容器类型递归
        if isinstance(value, dict):
            print(f"{prefix}{name}: dict (len={len(value)})")
            for k, v in value.items():
                self._print_value(f"[{repr(k)}]", v, indent, level + 1)
        elif isinstance(value, (list, tuple, set)):
            typename = type(value).__name__
            print(f"{prefix}{name}: {typename} (len={len(value)})")
            for i, v in enumerate(value):
                self._print_value(f"[{i}]", v, indent, level + 1)
        else:
            print(f"{prefix}{name}: {type(value).__name__}, value={repr(value)}")

    def _create_wrapper(self, func):
        name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print("\n==============start==============")

            self._set_print_options()

            arg_exprs = self._extract_call_argnames(func.__name__)
            print(f"[HOOK] Calling {name}(" + ", ".join(arg_exprs or ["..."]) + ")")
            for i, arg in enumerate(args):
                arg_name = arg_exprs[i] if i < len(arg_exprs) else f"arg[{i}]"
                self._print_value(arg_name, arg)
            for k, v in kwargs.items():
                self._print_value(k, v)

            result = func(*args, **kwargs)

            print(f"[HOOK] {name} returned:")
            self._print_value("result", result)
            print("===============end===============\n")
            return result

        return wrapper

    def register(self, func):
        if func in self.hooked_funcs:
            return
        wrapped = self._create_wrapper(func)
        self.hooked_funcs[func] = wrapped
        module = inspect.getmodule(func)
        if module:
            setattr(module, func.__name__, wrapped)
            print(f"[HOOK] 已hook函数: {module.__name__}.{func.__name__}")
