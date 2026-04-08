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
"""setup for FASTDEPLOY custom cpu ops"""
import os
import subprocess
import tarfile
from pathlib import Path

from paddle.utils.cpp_extension import CppExtension, setup
from setuptools import find_namespace_packages

ROOT_DIR = Path(__file__).parent.parent

# cannot import envs directly because it depends on fastdeploy,
#  which is not installed yet
from .setup_ops import load_module_from_path

envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "fastdeploy", "envs.py"))

BUILDING_ARCS = []
use_bf16 = envs.FD_CPU_USE_BF16 == "True"


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


# cc flags
paddle_extra_compile_args = [
    "-std=c++17",
    "-shared",
    "-fPIC",
    "-Wno-parentheses",
    "-DPADDLE_WITH_CUSTOM_KERNEL",
    "-Wall",
    "-O3",
    "-g",
    "-lstdc++fs",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
]

setup(
    name="fastdeploy_cpu_ops",
    ext_modules=CppExtension(
        sources=[
            "cpu_ops/set_value_by_flags.cc",
            "cpu_ops/token_penalty_multi_scores.cc",
            "cpu_ops/stop_generation_multi_ends.cc",
            "cpu_ops/update_inputs.cc",
            "cpu_ops/get_padding_offset.cc",
            "cpu_ops/rebuild_padding.cc",
        ],
        extra_compile_args=paddle_extra_compile_args,
    ),
    packages=find_namespace_packages(where="third_party"),
    package_dir={"": "third_party"},
    include_package_data=True,
)
