#!/usr/bin/env python3

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
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved.

Build and setup XPU custom ops for ERNIE Bot.
"""
import os
import shutil
import subprocess
from pathlib import Path

import paddle
from paddle.utils.cpp_extension import CppExtension, setup

current_file = Path(__file__).resolve()
base_dir = current_file.parent


def build_plugin(CLANG_PATH, XRE_INC_DIR, XRE_LIB_DIR, XDNN_INC_DIR,
                 XDNN_LIB_DIR):
    """
    build xpu plugin
    """
    current_working_directory = base_dir
    print(f"Current working directory: {current_working_directory}")

    # 设置环境变量
    os.environ["XRE_INC_DIR"] = XRE_INC_DIR
    os.environ["XRE_LIB_DIR"] = XRE_LIB_DIR
    os.environ["XDNN_INC_DIR"] = XDNN_INC_DIR
    os.environ["XDNN_LIB_DIR"] = XDNN_LIB_DIR

    # 设置 Clang 路径
    os.environ["CLANG_PATH"] = CLANG_PATH

    # 删除指定目录
    dirs_to_remove = [
        "dist", "fastdeploy_ops.egg-info", "build", "plugin/build"
    ]
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed directory: {dir_name}")

    # 在 plugin 目录中执行构建脚本
    plugin_dir = "plugin"
    build_script = os.path.join(current_working_directory, plugin_dir,
                                "build.sh")

    print("build_script: ", build_script)

    if not os.path.isfile(build_script):
        print(f"Error: Build script not found at {build_script}")
        return

    # 赋予执行权限 (如果尚未设置)
    if not os.access(build_script, os.X_OK):
        os.chmod(build_script, 0o755)

    # 执行构建脚本
    try:
        print("Running build script...")
        subprocess.run([build_script],
                       check=True,
                       cwd=os.path.join(current_working_directory, plugin_dir))
        print("Build completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def xpu_setup_ops():
    """
    setup xpu ops
    """
    PADDLE_PATH = os.path.dirname(paddle.__file__)
    PADDLE_INCLUDE_PATH = os.path.join(PADDLE_PATH, "include")
    PADDLE_LIB_PATH = os.path.join(PADDLE_PATH, "libs")

    BKCL_PATH = os.getenv("BKCL_PATH")
    if BKCL_PATH is None:
        BKCL_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xpu")
        BKCL_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libbkcl.so")
    else:
        BKCL_INC_PATH = os.path.join(BKCL_PATH, "include")
        BKCL_LIB_PATH = os.path.join(BKCL_PATH, "so", "libbkcl.so")

    CLANG_PATH = os.getenv("CLANG_PATH")
    assert CLANG_PATH is not None, "CLANG_PATH is not set."

    XRE_PATH = os.getenv("XRE_PATH")
    if XRE_PATH is None:
        XRE_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xre")
        XRE_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxpucuda.so")
        XRE_LIB_DIR = os.path.join(PADDLE_LIB_PATH)
    else:
        XRE_INC_PATH = os.path.join(XRE_PATH, "include")
        XRE_LIB_PATH = os.path.join(XRE_PATH, "so", "libxpucuda.so")
        XRE_LIB_DIR = os.path.join(XRE_PATH, "so")

    XDNN_PATH = os.getenv("XDNN_PATH")
    if XDNN_PATH is None:
        XDNN_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH)
        XDNN_LIB_DIR = os.path.join(PADDLE_LIB_PATH)
    else:
        XDNN_INC_PATH = os.path.join(XDNN_PATH, "include")
        XDNN_LIB_DIR = os.path.join(XDNN_PATH, "so")

    XVLLM_PATH = os.getenv("XVLLM_PATH")
    assert XVLLM_PATH is not None, "XVLLM_PATH is not set."
    XVLLM_KERNEL_INC_PATH = os.path.join(XVLLM_PATH, "infer_ops", "include")
    XVLLM_KERNEL_LIB_PATH = os.path.join(XVLLM_PATH, "infer_ops", "so",
                                         "libapiinfer.so")
    XVLLM_KERNEL_LIB_DIR = os.path.join(XVLLM_PATH, "infer_ops", "so")
    XVLLM_OP_INC_PATH = os.path.join(XVLLM_PATH, "xft_blocks", "include")
    XVLLM_OP_LIB_PATH = os.path.join(XVLLM_PATH, "xft_blocks", "so",
                                     "libxft_blocks.so")
    XVLLM_OP_LIB_DIR = os.path.join(XVLLM_PATH, "xft_blocks", "so")

    # build plugin
    build_plugin(CLANG_PATH, XRE_INC_PATH, XRE_LIB_DIR, XDNN_INC_PATH,
                 XDNN_LIB_DIR)

    ops = [
        # custom ops
        "./ops/save_with_output_msg.cc",
        "./ops/stop_generation_multi_ends.cc",
        "./ops/set_value_by_flags_and_idx.cc",
        "./ops/get_token_penalty_multi_scores.cc",
        "./ops/get_padding_offset.cc",
        "./ops/update_inputs.cc",
        "./ops/get_output.cc",
        "./ops/step.cc",
        "./ops/get_infer_param.cc",
        "./ops/adjust_batch.cc",
        "./ops/gather_next_token.cc",
        "./ops/block_attn.cc",
        "./ops/moe_layer.cc",
        "./ops/weight_quantize_xpu.cc",

        # device manage ops
        "./ops/device/get_context_gm_max_mem_demand.cc",
        "./ops/device/get_free_global_memory.cc",
        "./ops/device/get_total_global_memory.cc",
        "./ops/device/get_used_global_memory.cc",
    ]
    ops = [os.path.join(base_dir, op) for op in ops]

    include_dirs = [
        os.path.join(base_dir, "./"),
        os.path.join(base_dir, "./plugin/include"),
        BKCL_INC_PATH,
        XRE_INC_PATH,
        XVLLM_KERNEL_INC_PATH,
        XVLLM_OP_INC_PATH,
    ]
    extra_objects = [
        os.path.join(base_dir, "./plugin/build/libxpuplugin.a"),
        BKCL_LIB_PATH,
        XRE_LIB_PATH,
        XVLLM_KERNEL_LIB_PATH,
        XVLLM_OP_LIB_PATH,
    ]

    setup(
        name="fastdeploy_ops",
        ext_modules=CppExtension(
            sources=ops,
            include_dirs=include_dirs,
            extra_objects=extra_objects,
            extra_compile_args={
                "cxx": [
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    "-DPADDLE_WITH_XPU",
                    "-DBUILD_MULTI_XPU",
                ]
            },
            runtime_library_dirs=[XVLLM_KERNEL_LIB_DIR, XVLLM_OP_LIB_DIR],
        ),
    )


if __name__ == "__main__":
    xpu_setup_ops()
