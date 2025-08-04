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

import importlib
import inspect
import os
import re
import sys

import paddle
import triton

from .triton_utils import (
    SubstituteTemplate,
    build_package,
    compile_file,
    extract_triton_kernel,
    find_so_path,
    get_pointer_hint,
    link_file,
    multi_process_do,
    python_path,
    rename_c_to_cu,
)


def get_value_hint(x):
    """
    Get the value hint from input list.
    """
    hint = ""
    for ele in x:
        if isinstance(ele, int):
            hint += "i64,"
            continue
            if ele % 16 == 0 and ele > 0:
                hint += "i64:16,"
            elif ele == 1:
                hint += "i64:1,"
            else:
                hint += "i64,"
        if isinstance(ele, float):
            hint += "fp32,"
    return hint


common_template = """
#include "${op_name}_kernel.h"
#include "paddle/extension.h"

void ${op_name}_func(${tensor_and_attr}) {
    auto run_stream = a_ptr->stream();
    auto res_flag = ${op_name}_kernel(run_stream, ${triton_kernel_args}, 0);
    if (res_flag == CUDA_ERROR_INVALID_VALUE) {
        PD_THROW("${op_name}_kernel failed");
    }
}

PYBIND11_MODULE(${op_name}_package, m) {

  m.def("${op_name}_func", ${op_name}_func, "get expert token num");
}

"""


class KernelInterface:
    """
    triton kernel interface.
    """

    def __init__(
        self,
        func,
        other_config,
        key_args=["1"],
    ):
        """
        triton kernel interface.
        """
        self.func = func
        self.key_args = key_args

        signature = inspect.signature(func)
        self.arg_names = [v.name for v in signature.parameters.values()]
        for ele in self.arg_names:
            assert self.arg_names.count(ele) == 1
        # arg_defaults = [v.default for v in signature.parameters.values()]

        # self.annotations = {
        #     name: ty for name, ty in func.__annotations__.items()
        # }
        self.annotations = dict(func.__annotations__)

        self.constexprs = [
            self.arg_names.index(name)
            for name in self.arg_names
            if self.annotations.get(name) == triton.language.core.constexpr
        ]

        self.arg_exclude_constexpr = [
            self.arg_names[i] for i in range(len(self.arg_names)) if i not in self.constexprs
        ]

        import textwrap

        py_script = textwrap.dedent(inspect.getsource(func))

        pat = r"def\s" + func.__name__
        func_begin = re.findall(pat, py_script)
        assert len(func_begin) == 1
        func_begin = func_begin[0]
        py_script = py_script[py_script.find(func_begin) :]

        self.func_map = {}

        def decorator(*args, **kwargs):
            """
            decorator for triton kernels.
            Args:
                *args: positional arguments
                **kwargs: keyword arguments
            """
            op_name = f'haha_N{str(kwargs["N"])}_K{str(kwargs["K"])}'
            if op_name in self.func_map.keys():
                return self.func_map[op_name](*args)

            all_input = []

            for i in range(len(args)):
                all_input.append(args[i])

            position_arguments_num = len(all_input)
            for i in range(position_arguments_num, len(self.arg_names)):
                if self.arg_names[i] in kwargs.keys():
                    all_input.append(kwargs[self.arg_names[i]])
                else:
                    # means this input is not specified, it muse be a tl.constexpr.
                    assert i in self.constexprs
                    all_input.append(None)

            dtypes = []
            x_list = []
            const_args = [self.arg_names[i] for i in self.constexprs]

            decalare_arg_exclude_constexpr = list(self.arg_exclude_constexpr)
            passed_arg_exclude_constexpr = list(self.arg_exclude_constexpr)

            const_hint_dict = {}
            for i in range(len(all_input)):
                ele = all_input[i]

                if type(ele) in [
                    paddle.Tensor,
                    paddle.base.framework.EagerParamBase,
                    paddle.base.framework.Parameter,
                    paddle.base.framework.Variable,
                    paddle.base.libpaddle.pir.Value,
                    type(None),
                ]:
                    if ele is not None:
                        dtypes.append(ele.dtype)
                        passed_arg_exclude_constexpr[i] = f"(CUdeviceptr)({passed_arg_exclude_constexpr[i]}->data())"
                    else:
                        dtypes.append(paddle.int8)
                        passed_arg_exclude_constexpr[i] = "(CUdeviceptr)(nullptr)"
                    decalare_arg_exclude_constexpr[i] = (
                        "const paddle::optional<paddle::Tensor>&" + decalare_arg_exclude_constexpr[i]
                    )
                elif i in self.constexprs:
                    if isinstance(ele, bool):
                        const_hint_dict[self.arg_names[i]] = (int)(ele)
                    elif isinstance(ele, int):
                        if ele < 0:
                            const_hint_dict[self.arg_names[i]] = 0
                        else:
                            const_hint_dict[self.arg_names[i]] = ele
                    else:
                        assert False
                else:
                    x_list.append(ele)
                    if isinstance(ele, int):
                        decalare_arg_exclude_constexpr[i] = "const int64_t " + decalare_arg_exclude_constexpr[i]
                    elif isinstance(ele, float):
                        decalare_arg_exclude_constexpr[i] = "const float " + decalare_arg_exclude_constexpr[i]
                    else:
                        assert False

            python_package_name = f"{op_name}_package"
            tp_rank = paddle.distributed.get_rank()

            generated_dir = os.getenv("TRITON_KERNEL_CACHE_DIR", f"/tmp/triton_cache/rank{tp_rank}")
            print("the kernel cache dir is:", generated_dir)
            generated_dir = f"{generated_dir}/{op_name}"
            os.makedirs(generated_dir, exist_ok=True)

            py_script_file = f"{generated_dir}/triton_kernels.py"
            extract_triton_kernel(func, py_script_file)

            address_hint = get_pointer_hint(dtypes)
            value_hint = get_value_hint(x_list)
            const_args = [f"{{{ele}}}" for ele in const_args]
            const_args = ",".join(const_args)

            lanuch_grid = list(self.grid)
            for i in range(len(lanuch_grid)):
                ele = lanuch_grid[i]
                if isinstance(ele, str):
                    keys = list(const_hint_dict.keys())
                    keys.sort(key=len, reverse=True)
                    for key in keys:
                        if key in ele:
                            ele = ele.replace(key, f"{const_hint_dict[key]}")
                else:
                    ele = str(ele)
                lanuch_grid[i] = ele

            if len(lanuch_grid) < 3:
                lanuch_grid += ["1"] * (3 - len(lanuch_grid))
            lanuch_grid = ",".join(lanuch_grid)

            op_dict = {"op_name": op_name}
            op_dict["triton_kernel_args"] = ",".join(passed_arg_exclude_constexpr)
            op_dict["tensor_and_attr"] = ",".join(decalare_arg_exclude_constexpr)

            paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
            so_path = find_so_path(generated_dir, python_package_name)

            if so_path is None:
                print("== we do not find so_path, we need to compile it")
                with open(paddle_custom_op_file_path, "w") as f:
                    f.write(
                        SubstituteTemplate(
                            common_template,
                            op_dict,
                        )
                    )
                    f.close()

                # ahead of time compile command.
                aot_template = (
                    f"""{python_path}  {compile_file} {py_script_file}  """
                    + f""" -n {func.__name__} -o {generated_dir}/{op_name}_kernel """
                    + f"""--out-name {op_name}_kernel  """
                    + """ -w {num_warps} -ns {num_stages} """
                    + f""" -s"{address_hint} {value_hint} {const_args}" """
                    + f"""  -g "{lanuch_grid}" """
                )

                all_tune_config = [const_hint_dict]
                # reset const_hint_dict as empty.
                const_hint_dict = {}
                codegen_commands = []
                for config in all_tune_config:
                    for key in const_hint_dict.keys():
                        if const_hint_dict[key] is not None:
                            if key not in config.keys():
                                config[key] = const_hint_dict[key]
                            else:
                                if config[key] == const_hint_dict[key]:
                                    pass
                                else:
                                    message = (
                                        f"you specify {key} both in arguments and config, "
                                        "and they are not same, this is wrong."
                                    )
                                    raise ValueError(message)
                        else:
                            assert key in config.keys(), f"you must specify {key} in your config."
                    if "num_warps" not in config.keys():
                        config["num_warps"] = 4
                    if "num_stages" not in config.keys():
                        config["num_stages"] = 4

                    for key in config:
                        assert config[key] is not None, f"{key} must be specified."
                    codegen_command = aot_template.format(
                        **config,
                    )
                    print(codegen_command)
                    codegen_commands.append(codegen_command)
                multi_process_do(codegen_commands)

                link_command = (
                    f"{python_path}  {link_file} " f"{generated_dir}/*.h -o {generated_dir}/{op_name}_kernel"
                )
                re = os.system(link_command)
                assert re == 0

                # rename the .c file to .cu
                rename_c_to_cu(generated_dir)
                # build the package to so, not install
                build_package(generated_dir, python_package_name)

            # so_path have be found!
            so_path = find_so_path(generated_dir, python_package_name)
            print("== we find so_path: ", so_path)
            assert so_path is not None
            dir_path = os.path.dirname(so_path)
            sys.path.append(dir_path)
            lib = importlib.import_module(python_package_name)
            pybind_func = getattr(lib, f"{op_name}_func")
            self.func_map[op_name] = pybind_func

            # run this op!
            self.func_map[op_name](*args)

        self.decorator = decorator

    def __getitem__(self, op_name_and_grid):
        """
        override the operator [], which will call the decorator function.
        Args:
            op_name_and_grid: the name of the operator and the grid size.
        Returns:
            the decorator function.
        """
        self.grid = (
            (
                "((max_possible_num_post_padded + BLOCK_SIZE_M -1)/ BLOCK_SIZE_M) * ((N + BLOCK_SIZE_N-1) / BLOCK_SIZE_N)"
            ),
        )

        return self.decorator


def paddle_use_triton_v2(other_config={}, key=[]):
    """
    The decorator function that wraps the original function.
    Args:
        func: the original function.
    Returns:
        the wrapped function.
    """

    def decorator(func):
        """
        The decorator function that wraps the original function.
        Args:
            func: the original function.
        Returns:
            the wrapped function.
        """
        return KernelInterface(func, other_config, key)

    return decorator
