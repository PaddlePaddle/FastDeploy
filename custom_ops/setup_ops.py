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
""" setup for FastDeploy custom ops """
import glob
import json
import os
import shutil
import subprocess
import tarfile

import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup
from setuptools import find_namespace_packages, find_packages

archs = json.loads(os.getenv("BUILDING_ARCS", "[]"))
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


def clone_git_repo(version, repo_url, destination_path):
    """
    Clone git repo to destination path.
    """
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "-b",
                version,
                "--single-branch",
                repo_url,
                destination_path,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_sm_version(archs):
    """
    Get sm version of paddle.
    """
    arch_set = set(archs)
    try:
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        arch_set.add(cc)
    except ValueError:
        pass
    return list(arch_set)


def get_gencode_flags(archs):
    """
    Get gencode flags for current device or input.
    """
    cc_s = get_sm_version(archs)
    flags = []
    for cc in cc_s:
        if cc == 90:
            cc = f"{cc}a"
            flags += ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
        else:
            flags += ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    return flags


def find_end_files(directory, end_str):
    """
    Find files with end str in directory.
    """
    gen_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(end_str):
                gen_files.append(os.path.join(root, file))
    return gen_files


if paddle.is_compiled_with_rocm():
    # NOTE(@duanyanhui): paddle.is_compiled_with_cuda() returns True when paddle compiled with rocm.
    # so we need to check if paddle compiled with rocm at first.
    setup(
        name="fastdeploy_ops",
        ext_modules=CUDAExtension(
            sources=[
                "gpu_ops/save_with_output.cc",
                "gpu_ops/set_mask_value.cu",
                "gpu_ops/set_value_by_flags.cu",
                "gpu_ops/ngram_mask.cu",
                "gpu_ops/gather_idx.cu",
                "gpu_ops/token_penalty_multi_scores.cu",
                "gpu_ops/token_penalty_only_once.cu",
                "gpu_ops/stop_generation.cu",
                "gpu_ops/stop_generation_multi_ends.cu",
                "gpu_ops/stop_generation_multi_stop_seqs.cu",
                "gpu_ops/set_flags.cu",
                "gpu_ops/fused_get_rope.cu",
                "gpu_ops/transfer_output.cc",
                "gpu_ops/get_padding_offset.cu",
                "gpu_ops/update_inputs.cu",
                "gpu_ops/update_inputs_beam.cu",
                "gpu_ops/beam_search_softmax.cu",
                "gpu_ops/rebuild_padding.cu",
                "gpu_ops/save_with_output_msg.cc",
                "gpu_ops/get_output.cc",
                "gpu_ops/get_output_msg_with_topk.cc",
                "gpu_ops/reset_need_stop_value.cc",
                "gpu_ops/step.cu",
                "gpu_ops/step_reschedule.cu",
                "gpu_ops/set_data_ipc.cu",
                "gpu_ops/read_data_ipc.cu",
                "gpu_ops/dequant_int8.cu",
                "gpu_ops/enforce_generation.cu",
                "gpu_ops/tune_cublaslt_gemm.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "hipcc": [
                    "-O3",
                    "--gpu-max-threads-per-block=1024",
                    "-U__HIP_NO_HALF_OPERATORS__",
                    "-U__HIP_NO_HALF_CONVERSIONS__",
                    "-U__HIP_NO_BFLOAT16_OPERATORS__",
                    "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
                    "-U__HIP_NO_BFLOAT162_OPERATORS__",
                    "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
                ],
            },
        ),
    )
elif paddle.is_compiled_with_cuda():
    sources = [
        "gpu_ops/set_mask_value.cu", "gpu_ops/set_value_by_flags.cu",
        "gpu_ops/ngram_mask.cu", "gpu_ops/gather_idx.cu",
        "gpu_ops/get_output_ep.cc", "gpu_ops/get_mm_split_fuse.cc",
        "gpu_ops/token_penalty_multi_scores.cu",
        "gpu_ops/token_penalty_only_once.cu", "gpu_ops/stop_generation.cu",
        "gpu_ops/stop_generation_multi_ends.cu",
        "gpu_ops/stop_generation_multi_stop_seqs.cu", "gpu_ops/set_flags.cu",
        "gpu_ops/step.cu", "gpu_ops/step_reschedule.cu",
        "gpu_ops/fused_get_rope.cu", "gpu_ops/get_padding_offset.cu",
        "gpu_ops/update_inputs.cu", "gpu_ops/update_inputs_beam.cu",
        "gpu_ops/beam_search_softmax.cu", "gpu_ops/rebuild_padding.cu",
        "gpu_ops/set_data_ipc.cu", "gpu_ops/read_data_ipc.cu",
        "gpu_ops/enforce_generation.cu", "gpu_ops/dequant_int8.cu",
        "gpu_ops/tune_cublaslt_gemm.cu", "gpu_ops/swap_cache_batch.cu",
        "gpu_ops/swap_cache.cu", "gpu_ops/step_system_cache.cu",
        "gpu_ops/cpp_extensions.cu", "gpu_ops/share_external_data.cu",
        "gpu_ops/per_token_quant_fp8.cu",
        "gpu_ops/extract_text_token_output.cu",
        "gpu_ops/update_split_fuse_input.cu"
    ]

    # pd_disaggregation
    sources += [
        "gpu_ops/remote_cache_kv_ipc.cc",
        "gpu_ops/open_shm_and_get_meta_signal.cc",
        "gpu_ops/init_signal_layerwise.cc",
    ]

    cutlass_dir = "third_party/cutlass"
    if not os.path.exists(cutlass_dir) or not os.listdir(cutlass_dir):
        if not os.path.exists(cutlass_dir):
            os.makedirs(cutlass_dir)
        clone_git_repo("v3.8.0", "https://github.com/NVIDIA/cutlass.git",
                       cutlass_dir)
        if not os.listdir(cutlass_dir):
            raise ValueError("Git clone cutlass failed!")

    # deep gemm
    dg_third_party_include_dirs = (
        "third_party/cutlass/include/cute",
        "third_party/cutlass/include/cutlass",
    )

    dg_include_dir = "gpu_ops/fp8_deep_gemm/deep_gemm/include"
    os.makedirs(dg_include_dir, exist_ok=True)

    for d in dg_third_party_include_dirs:
        dirname = d.split("/")[-1]
        src_dir = d
        dst_dir = os.path.join(dg_include_dir, dirname)

        # Remove existing directory if it exists
        if os.path.exists(dst_dir):
            if os.path.islink(dst_dir):
                os.unlink(dst_dir)
            else:
                shutil.rmtree(dst_dir)
        print(f"Copying {src_dir} to {dst_dir}")

        # Copy the directory
        try:
            shutil.copytree(src_dir, dst_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to copy from {src_dir} to {dst_dir}: {e}")

    json_dir = "third_party/nlohmann_json"
    if not os.path.exists(json_dir) or not os.listdir(json_dir):
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        clone_git_repo("v3.11.3", "https://github.com/nlohmann/json.git",
                       json_dir)
        if not os.listdir(json_dir):
            raise ValueError("Git clone nlohmann_json failed!")

    nvcc_compile_args = get_gencode_flags(archs)
    nvcc_compile_args += ["-DPADDLE_DEV"]
    nvcc_compile_args += [
        "-Igpu_ops/cutlass_kernels",
        "-Ithird_party/cutlass/include",
        "-Igpu_ops/fp8_gemm_with_cutlass",
        "-Igpu_ops",
        "-Ithird_party/nlohmann_json/include",
    ]
    cc = max(get_sm_version(archs))
    print(f"cc = {cc}")
    if cc >= 80:
        # append_attention
        sources += ["gpu_ops/append_attention.cu"]
        sources += find_end_files("gpu_ops/append_attn", ".cu")
        # gemm_dequant
        sources += ["gpu_ops/int8_gemm_with_cutlass/gemm_dequant.cu"]
        # speculate_decoding
        sources += find_end_files("gpu_ops/speculate_decoding", ".cu")
        sources += find_end_files("gpu_ops/speculate_decoding", ".cc")
        nvcc_compile_args += ["-DENABLE_BF16"]
        # moe
        sources += find_end_files("gpu_ops/cutlass_kernels/moe_gemm/", ".cu")
        sources += find_end_files("gpu_ops/cutlass_kernels/w4a8_moe/", ".cu")
        sources += find_end_files("gpu_ops/moe/", ".cu")
        nvcc_compile_args += ["-Igpu_ops/moe"]

    if cc >= 89:
        # Running generate fp8 gemm codes.
        nvcc_compile_args += ["-DENABLE_FP8"]
        os.system("python auto_gen_fp8_fp8_gemm_fused_kernels.py")
        os.system("python auto_gen_fp8_fp8_dual_gemm_fused_kernels.py")
        os.system("python auto_gen_visitor_fp8_gemm_fused_kernels.py")

        nvcc_compile_args += [
            "-Igpu_ops/cutlass_kernels/fp8_gemm_fused/autogen"
        ]

        sources += [
            "gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
            "gpu_ops/cutlass_kernels/fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.cu",
            "gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
            "gpu_ops/cutlass_kernels/fp8_gemm_fused/fp8_fp8_dual_gemm_scale_bias_act.cu",
            "gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_half_cuda_core_gemm.cu",
            "gpu_ops/fp8_gemm_with_cutlass/per_channel_fp8_fp8_half_gemm.cu",
            "gpu_ops/cutlass_kernels/fp8_gemm_fused/visitor_fp8_gemm_fused.cu",
            "gpu_ops/scaled_gemm_f8_i4_f16_gemm.cu",
            "gpu_ops/scaled_gemm_f8_i4_f16_weight_quantize.cu",
            "gpu_ops/cutlass_kernels/cutlass_heuristic.cu",
            "gpu_ops/cutlass_kernels/cutlass_preprocessors.cu",
            "gpu_ops/air_topp_sampling.cu",
        ]
    if cc >= 90:
        nvcc_compile_args += [
            "-gencode",
            "arch=compute_90a,code=compute_90a",
            "-O3",
            "-DNDEBUG",
        ]
        os.system("python auto_gen_fp8_fp8_block_gemm_fused_kernels_sm90.py")
        sources += ["gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_half_block_gemm.cu"]

    # for fp8 autogen *.cu
    if cc >= 89:
        sources += find_end_files(
            "gpu_ops/cutlass_kernels/fp8_gemm_fused/autogen", ".cu")

    setup(
        name="fastdeploy_ops",
        ext_modules=CUDAExtension(
            sources=sources,
            extra_compile_args={"nvcc": nvcc_compile_args},
            libraries=["cublasLt"],
        ),
        packages=find_packages(where="gpu_ops/fp8_deep_gemm"),
        package_dir={"": "gpu_ops/fp8_deep_gemm"},
        package_data={
            "deep_gemm": [
                "include/deep_gemm/**/*",
                "include/cute/**/*",
                "include/cutlass/**/*",
            ]
        },
        include_package_data=True,
    )
elif paddle.is_compiled_with_xpu():
    # TODO zhangsishuai@baidu.com to add xpu ops
    setup(
        name="fastdeploy_ops",
        ext_modules=CUDAExtension(sources=[
            "xpu_ops/set_mask_value.cu",
            "xpu_ops/set_value_by_flags.cu",
            "xpu_ops/ngram_mask.cu",
            "xpu_ops/gather_idx.cu",
            "xpu_ops/token_penalty_multi_scores.cu",
            "xpu_ops/token_penalty_only_once.cu",
        ]),
    )
else:
    use_bf16 = os.getenv("CPU_USE_BF16", "False") == "True"
    x86_simd_sort_dir = "third_party/x86-simd-sort"
    if not os.path.exists(x86_simd_sort_dir) or not os.listdir(
            x86_simd_sort_dir):
        x86_simd_sort_url = "https://paddlepaddle-inference-banchmark.bj.bcebos.com/x86-simd-sort.tar.gz"
        download_and_extract(x86_simd_sort_url, "third_party")
    xft_dir = "third_party/xFasterTransformer"
    if not os.path.exists(xft_dir) or not os.listdir(xft_dir):
        if use_bf16:
            xft_url = (
                "https://paddlepaddle-inference-banchmark.bj.bcebos.com/xft.tar.gz"
            )
        else:
            xft_url = "https://paddlepaddle-inference-banchmark.bj.bcebos.com/xft_no_bf16.tar.gz"
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
