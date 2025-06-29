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
"""import ops"""
import functools
import importlib
import inspect
import os

import paddle

from fastdeploy.utils import llm_logger as logger


def import_custom_ops(package, module_name, global_ns):
    """
    Imports custom operations from a specified module within a package and adds them to a global namespace.

    Args:
        package (str): The name of the package containing the module.
        module_name (str): The name of the module within the package.
        global_ns (dict): The global namespace to add the imported functions to.
    """
    try:
        module = importlib.import_module(module_name, package=package)
        functions = inspect.getmembers(module)
        for func_name, func in functions:
            if func_name.startswith("__") or func_name == "_C_ops":
                continue
            logger.debug(f"Import {func_name} from {package}")
            try:
                global_ns[func_name] = func
            except Exception as e:
                logger.warning(f"Failed to import op {func_name}: {e}")

    except Exception:
        logger.warning(
            f"Ops of {package} import failed, it may be not compiled.")

    preprocess_static_op(global_ns)


def rename_imported_op(old_name, new_name, global_ns):
    """
    Renames an imported operation in the global namespace.

    Args:
        old_name (str): The original name of the operation in the global namespace.
        new_name (str): The new name to be given to the operation.
        global_ns (dict): The global namespace where the operation is stored.
    """
    if old_name not in global_ns:
        return
    global_ns[new_name] = global_ns[old_name]
    del global_ns[old_name]


def wrap_unified_op(original_cpp_ext_op, original_custom_op):
    """
    Wrap a static operator into a unified operator with runtime dispatching.
    Args:
        original_cpp_ext_op: Original C++ extension operator function.
        original_custom_op: Original custom operator function.
    """

    @paddle.jit.marker.unified
    @functools.wraps(original_custom_op)
    def unified_op(*args, **kwargs):
        if paddle.in_dynamic_mode():
            return original_cpp_ext_op(*args, **kwargs)
        return original_custom_op(*args, **kwargs)

    return unified_op


def preprocess_static_op(global_ns):
    """
    Transforms operator/function references in the global namespace based on the presence of 'static_op_' prefixes.

    Args:
        global_ns (dict): The global namespace (typically globals()) to modify.
        flag (bool): Determines transformation behavior.
    """
    static_op_prefix = "static_op_"
    static_op_names = [k for k in global_ns if k.startswith(static_op_prefix)]
    enforce_eager = int(os.getenv("FD_ENFORCE_EAGER", "0")) == 1

    for static_op in static_op_names:
        op_name = static_op[len(static_op_prefix):]
        has_dynamic_op = op_name in global_ns

        if has_dynamic_op:
            if not enforce_eager:
                original_cpp_ext_op = global_ns[op_name]
                original_custom_op = global_ns[static_op]
                global_ns[op_name] = wrap_unified_op(original_cpp_ext_op,
                                                     original_custom_op)
        else:
            global_ns[op_name] = global_ns[static_op]
