"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import glob
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import paddle
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel

long_description = "FastDeploy: Large Language Model Serving.\n\n"
long_description += "GitHub: https://github.com/PaddlePaddle/FastDeploy\n"
long_description += "Email: dltp@baidu.com"

# Platform to CMake mapping
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CustomBdistWheel(bdist_wheel):
    """Custom wheel builder for pure Python packages."""

    def finalize_options(self):
        """Configure wheel as pure Python and platform-independent."""
        super().finalize_options()
        self.root_is_pure = True
        self.python_tag = "py3"
        self.abi_tag = "none"
        self.plat_name_supplied = True
        self.plat_name = "any"


class CMakeExtension(Extension):
    """A setuptools Extension for CMake-based builds."""

    def __init__(self, name: str, sourcedir: str = "", version: str = None) -> None:
        """
        Initialize CMake extension.

        Args:
            name (str): Name of the extension.
            sourcedir (str): Source directory path.
            version (str): Optional version string (set to None to disable version info)
        """
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.version = version


class CMakeBuild(build_ext):
    """Custom build_ext command using CMake."""

    def get_ext_filename(self, ext_name):
        """Remove Python version tag from extension filename"""
        return ext_name.split(".")[0] + ".so"

    def build_extension(self, ext: CMakeExtension) -> None:
        """
        Build the CMake extension.

        Args:
            ext (CMakeExtension): The extension to build.
        """
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        cfg = "Debug" if int(os.environ.get("DEBUG", 0)) else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DVERSION_INFO=",
            "-DPYBIND11_PYTHON_VERSION=",
            "-DPYTHON_VERSION=",
            f"-DPYTHON_INCLUDE_DIR={sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}",
            f"-DPYTHON_LIBRARY={sys.prefix}/lib/libpython{sys.version_info.major}.{sys.version_info.minor}.so",
        ]
        build_args = []

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass
        else:
            if "NMake" not in cmake_generator and "Ninja" not in cmake_generator:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            if "NMake" not in cmake_generator and "Ninja" not in cmake_generator:
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ and hasattr(self, "parallel") and self.parallel:
            build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True)


class PostInstallCommand(install):
    """在标准安装完成后执行自定义命令"""

    def run(self):
        # 先执行标准安装步骤
        install.run(self)
        # 执行自定义命令
        subprocess.check_call(["opentelemetry-bootstrap", "-a", "install"])


def load_requirements():
    """Load dependencies from requirements.txt"""
    requirements_file_name = "requirements.txt"
    if paddle.is_compiled_with_custom_device("iluvatar_gpu"):
        requirements_file_name = "requirements_iluvatar.txt"
    elif paddle.is_compiled_with_rocm():
        requirements_file_name = "requirements_dcu.txt"
    elif paddle.device.is_compiled_with_custom_device("metax_gpu"):
        requirements_file_name = "requirements_metaxgpu.txt"
    requirements_path = os.path.join(os.path.dirname(__file__), requirements_file_name)
    with open(requirements_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_device_type():
    """Get the device type (rocm/gpu/xpu/npu/cpu/metax-gpu) that paddle is compiled with."""
    if paddle.is_compiled_with_rocm():
        return "rocm"
    elif paddle.is_compiled_with_cuda():
        return "gpu"
    elif paddle.is_compiled_with_xpu():
        return "xpu"
    elif paddle.is_compiled_with_custom_device("npu"):
        return "npu"
    elif paddle.is_compiled_with_custom_device("iluvatar_gpu"):
        return "iluvatar-gpu"
    elif paddle.is_compiled_with_custom_device("gcu"):
        return "gcu"
    elif paddle.device.is_compiled_with_custom_device("metax_gpu"):
        return "metax-gpu"
    elif paddle.is_compiled_with_custom_device("intel_hpu"):
        return "intel-hpu"
    else:
        return "cpu"


def check_header(header_path):
    return os.path.exists(header_path)


def check_library(lib_name):
    # search /usr/lib /usr/lib64 /lib /lib64 .etc
    paths = [
        "/usr/lib",
        "/usr/lib32",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/lib",
        "/lib32",
        "/lib64",
        "/usr/local/lib",
        "/usr/local/lib64",
    ]
    for p in paths:
        if glob.glob(os.path.join(p, lib_name)):
            return True
    return False


def check_rdma_packages():
    results = {}

    # libibverbs-dev
    results["libibverbs header"] = check_header("/usr/include/infiniband/verbs.h")
    results["libibverbs library"] = check_library("libibverbs.so*") or check_library("libibverbs.so")

    # librdmacm-dev
    results["librdmacm header"] = check_header("/usr/include/rdma/rdma_cma.h")
    results["librdmacm library"] = check_library("librdmacm.so*") or check_library("librdmacm.so")

    print("===== RDMA Library Check Results =====")
    for k, v in results.items():
        status = "FOUND" if v else "NOT FOUND"
        print(f"{k:25}: {status}")

    print("\n== Summary ==")
    if all(results.values()):
        print("All required RDMA libraries are installed.")
        return True
    else:
        print("Some RDMA libraries are missing. Suggested commands:")
        print("\nUbuntu/Debian:")
        print("    sudo apt-get install -y libibverbs-dev librdmacm-dev")
        print("\nCentOS/RHEL:")
        print("    sudo yum install -y libibverbs-devel librdmacm-devel")
        return False


@lru_cache(maxsize=1)
def rdma_comm_supported():
    supported = (
        get_device_type() in ["gpu", "xpu"]
        and check_rdma_packages()
        and os.getenv("FD_ENABLE_RDMA_COMPILE", "1") == "1"
    )
    return supported


def get_name():
    """get package name"""
    return "fastdeploy-" + get_device_type()


cmdclass_dict = {"bdist_wheel": CustomBdistWheel}
cmdclass_dict["build_ext"] = CMakeBuild
FASTDEPLOY_VERSION = os.environ.get("FASTDEPLOY_VERSION", "2.3.0-dev")
cmdclass_dict["build_optl"] = PostInstallCommand


def write_version_to_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    version_file_path = os.path.join(current_dir, "fastdeploy/version.txt")
    with open(version_file_path, "a") as f:
        f.write(f"fastdeploy version: {FASTDEPLOY_VERSION}\n")


write_version_to_file()

setup(
    name=get_name(),
    version=FASTDEPLOY_VERSION,
    author="PaddlePaddle",
    author_email="dltp@baidu.com",
    description="FastDeploy: Large Language Model Serving.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/FastDeploy",
    packages=find_packages(),
    package_dir={"fastdeploy": "fastdeploy/"},
    # For deprecated method (egg-based installation), `.so` files are placed in the `model_executor/ops/XXX` directory.
    # For modern method (PEP 517/518-based installation), `.so` files are placed in the `model_executor/ops/XXX/fastdeploy_ops` directory.
    # Therefore, the `fastdeploy_ops` directory should be included for modern Python packaging.
    package_data={
        "fastdeploy": [
            "model_executor/ops/gpu/*",
            "model_executor/ops/gpu/fastdeploy_ops/*",
            "model_executor/ops/gpu/deep_gemm/include/**/*",
            "model_executor/ops/cpu/*",
            "model_executor/ops/cpu/fastdeploy_cpu_ops/*",
            "model_executor/ops/xpu/*",
            "model_executor/ops/xpu/fastdeploy_ops/*",
            "model_executor/ops/xpu/libs/*",
            "model_executor/ops/xpu/fastdeploy_ops/libs/*",
            "model_executor/ops/npu/*",
            "model_executor/ops/npu/fastdeploy_ops/*",
            "model_executor/ops/base/*",
            "model_executor/ops/base/fastdeploy_ops/*",
            "model_executor/ops/iluvatar/*",
            "model_executor/ops/iluvatar/fastdeploy_ops/*",
            "model_executor/models/*",
            "model_executor/layers/*",
            "input/ernie4_5_vl_processor/utils/*",
            "model_executor/ops/gcu/*",
            "model_executor/ops/gcu/fastdeploy_ops/*",
            "version.txt",
        ]
    },
    install_requires=load_requirements(),
    ext_modules=(
        [
            CMakeExtension(
                "rdma_comm",
                sourcedir="fastdeploy/cache_manager/transfer_factory/kvcache_transfer",
                version=None,
            )
        ]
        if rdma_comm_supported()
        else []
    ),
    cmdclass=cmdclass_dict if rdma_comm_supported() else {},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires=">=3.7",
    extras_require={
        "test": ["pytest>=6.0"],
        "eval": ["lm-eval==0.4.9.1"],
    },
    entry_points={
        "console_scripts": ["fastdeploy=fastdeploy.entrypoints.cli.main:main"],
    },
)
