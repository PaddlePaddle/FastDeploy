# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

""" setup for FASTDEPLOY custom cpu ops """
import os
import subprocess
from paddle.utils.cpp_extension import setup, CppExtension
from setuptools import find_namespace_packages
import glob
import tarfile

BUILDING_ARCS = []
use_bf16 = os.getenv("CPU_USE_BF16", "False") == "True"


def download_and_extract(url, destination_directory):
    """
    Download a .tar.gz file using wget to the destination directory
    and extract its contents without renaming the downloaded file.

    :param url: The URL of the .tar.gz file to download.
    :param destination_directory: The directory where the file should be downloaded and extracted.
    """
    os.makedirs(destination_directory, exist_ok=True)

    filename = os.path.basename(url)
    file_path = os.path.join(destination_directory, filename)

    try:
        subprocess.run(
            ["wget", "-O", file_path, url],
            check=True,
        )
        print(f"Downloaded: {file_path}")

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=destination_directory)
            print(f"Extracted: {file_path} to {destination_directory}")
        os.remove(file_path)
        print(f"Deleted downloaded file: {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
    except Exception as e:
        print(f"Error extracting file: {e}")


x86_simd_sort_dir = "third_party/x86-simd-sort"
if not os.path.exists(x86_simd_sort_dir) or not os.listdir(x86_simd_sort_dir):
    x86_simd_sort_url = (
        "https://paddlepaddle-inference-banchmark.bj.bcebos.com/x86-simd-sort.tar.gz"
    )
    download_and_extract(x86_simd_sort_url, "third_party")
xft_dir = "third_party/xFasterTransformer"
if not os.path.exists(xft_dir) or not os.listdir(xft_dir):
    if use_bf16:
        xft_url = "https://paddlepaddle-inference-banchmark.bj.bcebos.com/xft.tar.gz"
    else:
        xft_url = (
            "https://paddlepaddle-inference-banchmark.bj.bcebos.com/xft_no_bf16.tar.gz"
        )
    download_and_extract(xft_url, "third_party")

libs = [
    "xfastertransformer",
    "xft_comm_helper",
    "x86simdsortcpp",
]
xft_dir = "third_party/xFasterTransformer"
x86_simd_sort_dir = "third_party/x86-simd-sort"
paddle_custom_kernel_include = [
    os.path.join(xft_dir, "include"),
    os.path.join(xft_dir, "src/common"),  # src
    os.path.join(xft_dir, "src/kernels"),  # src
    os.path.join(xft_dir, "src/layers"),  # src
    os.path.join(xft_dir, "src/models"),  # src
    os.path.join(xft_dir, "src/utils"),  # src
    os.path.join(xft_dir, "3rdparty/onednn/include"),  # src
    os.path.join(xft_dir, "3rdparty/onednn/build/include"),  # src
    os.path.join(xft_dir, "3rdparty/xdnn"),  # src
    os.path.join(xft_dir, "3rdparty"),
    os.path.join(xft_dir, "3rdparty/mkl/include"),
    os.path.join(x86_simd_sort_dir, "src"),  # src
]

# cc flags
paddle_extra_compile_args = [
    "-std=c++17",
    "-shared",
    "-fPIC",
    "-Wno-parentheses",
    "-DPADDLE_WITH_CUSTOM_KERNEL",
    "-mavx512f",
    "-mavx512vl",
    "-fopenmp",
    "-mavx512bw",
    "-mno-mmx",
    "-Wall",
    "-march=skylake-avx512",
    "-O3",
    "-g",
    "-lstdc++fs",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
]
if use_bf16:
    # avx512-bf16 flags
    paddle_extra_compile_args += [
        "-DAVX512_BF16_WEIGHT_ONLY_BF16=true",
        "-DAVX512_FP16_WEIGHT_ONLY_INT8=true",
        "-DAVX512_FP16_WEIGHT_ONLY_FP16=true",
    ]
else:
    # no avx512-bf16 flags
    paddle_extra_compile_args += [
        "-DAVX512_FP32_WEIGHT_ONLY_INT8=true",
        "-DAVX512_FP32_WEIGHT_ONLY_FP16=true",
    ]
paddle_custom_kernel_library_dir = [
    "third_party/xFasterTransformer/build/",
    "third_party/x86-simd-sort/builddir",
]

include_files = []
for include_dir in paddle_custom_kernel_include:
    include_files.extend(glob.glob(os.path.join(include_dir, "*.h")))
so_files = []
for library_dir in paddle_custom_kernel_library_dir:
    if os.path.isdir(library_dir):
        for lib in libs:
            lib_file = os.path.join(library_dir, f"lib{lib}.so")
            if os.path.isfile(lib_file):
                so_files.append(lib_file)
setup(
    name="fastdeploy_cpu_ops",
    ext_modules=CppExtension(
        sources=[
            "cpu_ops/simd_sort.cc",
            "cpu_ops/set_value_by_flags.cc",
            "cpu_ops/token_penalty_multi_scores.cc",
            "cpu_ops/stop_generation_multi_ends.cc",
            "cpu_ops/update_inputs.cc",
            "cpu_ops/get_padding_offset.cc",
            "cpu_ops/xft_all_layer.cc",
            "cpu_ops/xft_greedy_search.cc",
            "cpu_ops/avx_weight_only.cc",
        ],
        extra_link_args=[
            "-Wl,-rpath,$ORIGIN/x86-simd-sort/builddir",
            "-Wl,-rpath,$ORIGIN/xFasterTransformer/build",
        ],
        include_dirs=paddle_custom_kernel_include,
        library_dirs=paddle_custom_kernel_library_dir,
        libraries=libs,
        extra_compile_args=paddle_extra_compile_args,
    ),
    packages=find_namespace_packages(where="third_party"),
    package_dir={"": "third_party"},
    package_data={"fastdeploy_cpu_ops": include_files + so_files},
    include_package_data=True,
)
