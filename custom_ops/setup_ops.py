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
"""setup for FastDeploy custom ops"""
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup
from setuptools import find_namespace_packages, find_packages


def load_module_from_path(module_name, path):
    """
    load python module from path
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def update_git_repo():
    try:
        print("update third party repo...", flush=True)
        original_dir = os.getcwd()
        submodule_dir = os.path.dirname(os.path.abspath(__file__))
        third_party_path = os.path.join(submodule_dir, "third_party")
        root_path = Path(third_party_path)

        # check if third_party is empty
        update_third_party = False
        for dirpath in root_path.iterdir():
            if dirpath.is_dir():
                has_content = any(dirpath.iterdir())
                if not has_content:
                    update_third_party = True

        if update_third_party:
            os.chdir(submodule_dir)
            subprocess.run(
                "git submodule sync --recursive && git submodule update --init --recursive",
                shell=True,
                check=True,
                text=True,
            )
        else:
            print(
                "\033[33m[===WARNING===]third_party directory already exists, skip clone and update.\033[0m",
                flush=True,
            )

        # apply deep gemm patch
        deep_gemm_dir = "third_party/DeepGEMM"
        dst_path = os.path.join(submodule_dir, deep_gemm_dir)
        patch = "0001-DeepGEMM-95e81b3.patch"
        patch_source = os.path.join(submodule_dir, patch)
        patch_destination = os.path.join(dst_path, patch)
        if not os.path.exists(patch_destination):
            shutil.copy(patch_source, patch_destination)
            apply_cmd = ["git", "apply", patch]
            os.chdir(dst_path)
            subprocess.run(apply_cmd, check=True)
        os.chdir(original_dir)
    except subprocess.CalledProcessError:
        raise Exception("Git submodule update and apply patch failed. Maybe network connection is poor.")


ROOT_DIR = Path(__file__).parent.parent

# cannot import envs directly because it depends on fastdeploy,
#  which is not installed yet
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "fastdeploy", "envs.py"))

archs = json.loads(envs.FD_BUILDING_ARCS)
use_bf16 = envs.FD_CPU_USE_BF16 == "True"

update_git_repo()


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


def get_sm_version(archs):
    """
    Get sm version of paddle.
    """
    arch_set = set(archs)
    if len(arch_set) == 0:
        try:
            prop = paddle.device.cuda.get_device_properties()
            cc = prop.major * 10 + prop.minor
            arch_set.add(cc)
        except ValueError:
            pass
    return list(arch_set)


def get_nvcc_version():
    """
    Get cuda version of nvcc.
    """
    nvcc_output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = float(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_gencode_flags(archs):
    """
    Get gencode flags for current device or input.
    """
    cc_s = get_sm_version(archs)
    flags = []
    for cc_val in cc_s:
        if cc_val == 90:
            arch_code = "90a"
            flags += [
                "-gencode",
                f"arch=compute_{arch_code},code=sm_{arch_code}",
            ]
        elif cc_val == 100:  # Assuming 100 is the code for Blackwell SM10.x
            # Per NVIDIA dev blog, for CUTLASS and architecture-specific features on CC 10.0, use '100a'
            # https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
            # "The CUTLASS build instructions specify using the a flag when building for devices of CC 9.0 and 10.0"
            arch_code = "100a"
            flags += [
                "-gencode",
                f"arch=compute_{arch_code},code=sm_{arch_code}",
            ]
        else:
            flags += ["-gencode", f"arch=compute_{cc_val},code=sm_{cc_val}"]
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
    sources = [
        "gpu_ops/save_with_output_msg.cc",
        "gpu_ops/get_output.cc",
        "gpu_ops/get_output_msg_with_topk.cc",
        "gpu_ops/save_output_msg_with_topk.cc",
        "gpu_ops/transfer_output.cc",
        "gpu_ops/set_value_by_flags_and_idx.cu",
        "gpu_ops/token_penalty_multi_scores.cu",
        "gpu_ops/stop_generation.cu",
        "gpu_ops/stop_generation_multi_ends.cu",
        "gpu_ops/get_padding_offset.cu",
        "gpu_ops/update_inputs.cu",
        "gpu_ops/rebuild_padding.cu",
        "gpu_ops/step.cu",
        "gpu_ops/set_data_ipc.cu",
        "gpu_ops/unset_data_ipc.cu",
        "gpu_ops/moe/tritonmoe_preprocess.cu",
        "gpu_ops/step_system_cache.cu",
        "gpu_ops/get_output_ep.cc",
        "gpu_ops/speculate_decoding/speculate_get_padding_offset.cu",
        "gpu_ops/speculate_decoding/speculate_get_output.cc",
        "gpu_ops/share_external_data.cu",
        "gpu_ops/speculate_decoding/speculate_clear_accept_nums.cu",
        "gpu_ops/speculate_decoding/speculate_get_output_padding_offset.cu",
        "gpu_ops/speculate_decoding/speculate_get_seq_lens_output.cu",
        "gpu_ops/speculate_decoding/speculate_save_output.cc",
        "gpu_ops/speculate_decoding/speculate_set_value_by_flags_and_idx.cu",
        "gpu_ops/speculate_decoding/speculate_step.cu",
        "gpu_ops/speculate_decoding/speculate_step_system_cache.cu",
        "gpu_ops/speculate_decoding/speculate_update_v3.cu",
        "gpu_ops/get_position_ids_and_mask_encoder_batch.cu",
        "gpu_ops/fused_rotary_position_encoding.cu",
        "gpu_ops/step_reschedule.cu",
    ]
    setup(
        name="fastdeploy_ops",
        ext_modules=CUDAExtension(
            sources=sources,
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
                    "-DPADDLE_DEV",
                    "-Ithird_party/nlohmann_json/include",
                    "-Igpu_ops",
                ],
            },
        ),
    )
elif paddle.is_compiled_with_cuda():
    sources = [
        "gpu_ops/helper.cu",
        "gpu_ops/save_with_output_msg.cc",
        "gpu_ops/get_output.cc",
        "gpu_ops/get_output_msg_with_topk.cc",
        "gpu_ops/save_output_msg_with_topk.cc",
        "gpu_ops/transfer_output.cc",
        "gpu_ops/set_mask_value.cu",
        "gpu_ops/set_value_by_flags_and_idx.cu",
        "gpu_ops/ngram_mask.cu",
        "gpu_ops/gather_idx.cu",
        "gpu_ops/get_output_ep.cc",
        "gpu_ops/get_mm_split_fuse.cc",
        "gpu_ops/get_img_boundaries.cc",
        "gpu_ops/token_penalty_multi_scores.cu",
        "gpu_ops/token_penalty_only_once.cu",
        "gpu_ops/stop_generation.cu",
        "gpu_ops/stop_generation_multi_ends.cu",
        "gpu_ops/set_flags.cu",
        "gpu_ops/update_inputs_v1.cu",
        "gpu_ops/recover_decode_task.cu",
        "gpu_ops/step.cu",
        "gpu_ops/step_reschedule.cu",
        "gpu_ops/fused_get_rotary_embedding.cu",
        "gpu_ops/get_padding_offset.cu",
        "gpu_ops/update_inputs.cu",
        "gpu_ops/update_inputs_beam.cu",
        "gpu_ops/beam_search_softmax.cu",
        "gpu_ops/rebuild_padding.cu",
        "gpu_ops/set_data_ipc.cu",
        "gpu_ops/unset_data_ipc.cu",
        "gpu_ops/read_data_ipc.cu",
        "gpu_ops/enforce_generation.cu",
        "gpu_ops/dequant_int8.cu",
        "gpu_ops/tune_cublaslt_gemm.cu",
        "gpu_ops/swap_cache_batch.cu",
        "gpu_ops/swap_cache.cu",
        "gpu_ops/step_system_cache.cu",
        "gpu_ops/cpp_extensions.cc",
        "gpu_ops/share_external_data.cu",
        "gpu_ops/per_token_quant_fp8.cu",
        "gpu_ops/update_split_fuse_input.cu",
        "gpu_ops/text_image_index_out.cu",
        "gpu_ops/text_image_gather_scatter.cu",
        "gpu_ops/sample_kernels/rejection_top_p_sampling.cu",
        "gpu_ops/sample_kernels/top_k_renorm_probs.cu",
        "gpu_ops/sample_kernels/min_p_sampling_from_probs.cu",
        "gpu_ops/get_position_ids_and_mask_encoder_batch.cu",
        "gpu_ops/fused_rotary_position_encoding.cu",
        "gpu_ops/noaux_tc.cu",
        "gpu_ops/noaux_tc_redundant.cu",
        "gpu_ops/custom_all_reduce/all_reduce.cu",
        "gpu_ops/merge_prefill_decode_output.cu",
        "gpu_ops/limit_thinking_content_length_v1.cu",
        "gpu_ops/limit_thinking_content_length_v2.cu",
        "gpu_ops/update_attn_mask_offsets.cu",
        "gpu_ops/fused_neox_rope_embedding.cu",
        "gpu_ops/gelu_tanh.cu",
    ]

    # pd_disaggregation
    sources += [
        "gpu_ops/remote_cache_kv_ipc.cc",
        "gpu_ops/open_shm_and_get_meta_signal.cc",
        "gpu_ops/init_signal_layerwise.cc",
        "gpu_ops/get_data_ptr_ipc.cu",
        "gpu_ops/ipc_sent_key_value_cache_by_remote_ptr.cu",
    ]

    dg_third_party_include_dirs = (
        "third_party/cutlass/include/cute",
        "third_party/cutlass/include/cutlass",
    )

    dg_include_dir = "third_party/DeepGEMM/deep_gemm/include"
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
            raise RuntimeError(f"Failed to copy from {src_dir} to {dst_dir}: {e}")

    cc_compile_args = []
    nvcc_compile_args = get_gencode_flags(archs)
    nvcc_compile_args += ["-DPADDLE_DEV"]
    nvcc_compile_args += ["-DPADDLE_ON_INFERENCE"]
    nvcc_compile_args += ["-DPy_LIMITED_API=0x03090000"]
    nvcc_compile_args += [
        "-Igpu_ops/cutlass_kernels",
        "-Ithird_party/cutlass/include",
        "-Ithird_party/cutlass/tools/util/include",
        "-Igpu_ops/fp8_gemm_with_cutlass",
        "-Igpu_ops",
        "-Ithird_party/nlohmann_json/include",
    ]
    worker_threads = os.cpu_count()
    nvcc_compile_args += ["-t", str(worker_threads)]

    nvcc_version = get_nvcc_version()
    print(f"nvcc_version = {nvcc_version}")
    if nvcc_version >= 12.0:
        sources += ["gpu_ops/sample_kernels/air_top_p_sampling.cu"]
    cc = max(get_sm_version(archs))
    print(f"cc = {cc}")
    fp8_auto_gen_directory = "gpu_ops/cutlass_kernels/fp8_gemm_fused/autogen"
    if os.path.isdir(fp8_auto_gen_directory):
        shutil.rmtree(fp8_auto_gen_directory)

    if cc >= 75:
        nvcc_compile_args += [
            "-DENABLE_SCALED_MM_C2X=1",
            "-Igpu_ops/cutlass_kernels/w8a8",
        ]
        sources += [
            "gpu_ops/cutlass_kernels/w8a8/scaled_mm_entry.cu",
            "gpu_ops/cutlass_kernels/w8a8/scaled_mm_c2x.cu",
            "gpu_ops/quantization/common.cu",
        ]

    if cc >= 80:
        # append_attention
        os.system(
            "python utils/auto_gen_template_instantiation.py --config gpu_ops/append_attn/template_config.json --output gpu_ops/append_attn/template_instantiation/autogen"
        )
        sources += ["gpu_ops/append_attention.cu"]
        sources += find_end_files("gpu_ops/append_attn", ".cu")
        # mla
        sources += ["gpu_ops/multi_head_latent_attention.cu"]
        # gemm_dequant
        sources += ["gpu_ops/int8_gemm_with_cutlass/gemm_dequant.cu"]
        # speculate_decoding
        sources += find_end_files("gpu_ops/speculate_decoding", ".cu")
        sources += find_end_files("gpu_ops/speculate_decoding", ".cc")
        nvcc_compile_args += ["-DENABLE_BF16"]
        # moe
        os.system("python gpu_ops/moe/moe_wna16_marlin_utils/generate_kernels.py")
        os.system(
            "python utils/auto_gen_template_instantiation.py --config gpu_ops/moe/template_config.json --output gpu_ops/moe/template_instantiation/autogen"
        )
        sources += find_end_files("gpu_ops/cutlass_kernels/moe_gemm/", ".cu")
        sources += find_end_files("gpu_ops/cutlass_kernels/w4a8_moe/", ".cu")
        sources += find_end_files("gpu_ops/moe/", ".cu")
        nvcc_compile_args += ["-Igpu_ops/moe"]

    if cc >= 89:
        # Running generate fp8 gemm codes.
        # Common for SM89, SM90, SM100 (Blackwell)
        nvcc_compile_args += ["-DENABLE_FP8"]
        nvcc_compile_args += ["-Igpu_ops/cutlass_kernels/fp8_gemm_fused/autogen"]
        # This script seems general enough for different SM versions, specific templates are chosen by CUTLASS.
        os.system("python utils/auto_gen_visitor_fp8_gemm_fused_kernels.py")

        if cc >= 90:  # Hopper and newer
            # SM90 (Hopper) specific auto-generation and flags
            if cc == 90:  # Only for SM90
                nvcc_compile_args += [
                    # The gencode for 90a is added in get_gencode_flags now
                    # "-gencode",
                    # "arch=compute_90a,code=compute_90a",
                    "-O3",
                    "-DNDEBUG",  # NDEBUG is common, consider moving if not specific to 90a
                ]
                print("SM90: Running SM90-specific FP8 kernel auto-generation.")
                os.system("python utils/auto_gen_fp8_fp8_gemm_fused_kernels_sm90.py")
                os.system("python utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels_sm90.py")
                os.system("python utils/auto_gen_fp8_fp8_block_gemm_fused_kernels_sm90.py")

                nvcc_compile_args += [
                    "-DENABLE_SCALED_MM_SM90=1",
                ]
                sources += [
                    "gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_half_block_gemm.cu",
                    "gpu_ops/cutlass_kernels/w8a8/scaled_mm_c3x_sm90.cu",
                    "gpu_ops/cutlass_kernels/w8a8/c3x/scaled_mm_sm90_fp8.cu",
                    "gpu_ops/cutlass_kernels/w8a8/c3x/scaled_mm_sm90_int8.cu",
                    "gpu_ops/cutlass_kernels/w8a8/c3x/scaled_mm_azp_sm90_int8.cu",
                ]
            elif cc == 100 and nvcc_version >= 12.9:  # Blackwell SM100 specifics
                print("SM100 (Blackwell): Applying SM100 configurations.")
                nvcc_compile_args += [
                    # The gencode for 100a is added in get_gencode_flags
                    # "-gencode",
                    # "arch=compute_100a,code=compute_100a",
                    "-O3",  # Common optimization flag
                    "-DNDEBUG",  # Common debug flag
                    # Potentially add -DENABLE_SM100_FEATURES if specific macros are identified
                ]
                # Placeholder for SM100-specific kernel auto-generation scripts
                # These might be needed if Blackwell has new FP8 hardware features
                # not covered by existing generic CUTLASS templates or SM90 scripts.
                # print("SM100: Running SM100-specific FP8 kernel auto-generation (if any).")
                # os.system("python utils/auto_gen_fp8_fp8_gemm_fused_kernels_sm100.py") # Example
                # os.system("python utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels_sm100.py") # Example

                # Add SM100 specific sources if any, e.g., for new hardware intrinsics
                # sources += ["gpu_ops/cutlass_kernels/w8a8/c4x_sm100.cu"] # Example
                pass  # No SM100 specific sources identified yet beyond what CUTLASS handles
            else:  # For cc >= 89 but not 90 or 100 (e.g. SM89)
                print(f"SM{cc}: Running generic FP8 kernel auto-generation.")
                os.system("python utils/auto_gen_fp8_fp8_gemm_fused_kernels.py")
                os.system("python utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels.py")

        else:  # For cc == 89 (Ada)
            print("SM89: Running generic FP8 kernel auto-generation.")
            os.system("python utils/auto_gen_fp8_fp8_gemm_fused_kernels.py")
            os.system("python utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels.py")

        # Common FP8 sources for SM89+
        sources += [
            "gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
            "gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
            "gpu_ops/fp8_gemm_with_cutlass/fp8_fp8_half_cuda_core_gemm.cu",
            "gpu_ops/fp8_gemm_with_cutlass/per_channel_fp8_fp8_half_gemm.cu",
            "gpu_ops/cutlass_kernels/fp8_gemm_fused/visitor_fp8_gemm_fused.cu",
            "gpu_ops/scaled_gemm_f8_i4_f16_gemm.cu",
            "gpu_ops/scaled_gemm_f8_i4_f16_weight_quantize.cu",
            "gpu_ops/cutlass_kernels/cutlass_heuristic.cu",
            "gpu_ops/cutlass_kernels/cutlass_preprocessors.cu",
            "gpu_ops/fused_hadamard_quant_fp8.cu",
        ]

        sources += find_end_files(fp8_auto_gen_directory, ".cu")

    if cc >= 90 and nvcc_version >= 12.0:
        # Hopper optimized mla
        sources += find_end_files("gpu_ops/mla_attn", ".cu")
        sources += ["gpu_ops/flash_mask_attn/flash_mask_attn.cu"]
        sources += find_end_files("gpu_ops/moba_attn/moba_decoder_attn/", ".cu")
        sources += find_end_files("gpu_ops/moba_attn/moba_encoder_attn/", ".cu")
        sources += find_end_files("gpu_ops/moba_attn/moba_process/", ".cu")
        sources += ["gpu_ops/moba_attn/moba_attn.cu"]
        os.system("python utils/auto_gen_w4afp8_gemm_kernel.py")
        sources += find_end_files("gpu_ops/w4afp8_gemm", ".cu")
        os.system("python utils/auto_gen_wfp8afp8_sparse_gemm_kernel.py")
        sources += find_end_files("gpu_ops/wfp8afp8_sparse_gemm", ".cu")
        os.system("python gpu_ops/machete/generate.py")
        sources += find_end_files("gpu_ops/machete", ".cu")
        cc_compile_args += ["-DENABLE_MACHETE"]

    setup(
        name="fastdeploy_ops",
        ext_modules=CUDAExtension(
            sources=sources,
            extra_compile_args={"cxx": cc_compile_args, "nvcc": nvcc_compile_args},
            libraries=["cublasLt"],
            extra_link_args=["-lcuda", "-lnvidia-ml"],
        ),
        packages=find_packages(where="third_party/DeepGEMM"),
        package_dir={"": "third_party/DeepGEMM"},
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
    assert False, "For XPU, please use setup_ops.py in the xpu_ops directory to compile custom ops."
elif paddle.is_compiled_with_custom_device("iluvatar_gpu"):
    setup(
        name="fastdeploy_ops",
        ext_modules=CUDAExtension(
            extra_compile_args={
                "nvcc": [
                    "-DPADDLE_DEV",
                    "-DPADDLE_WITH_CUSTOM_DEVICE",
                ]
            },
            sources=[
                "gpu_ops/save_with_output_msg.cc",
                "gpu_ops/get_output.cc",
                "gpu_ops/get_output_msg_with_topk.cc",
                "gpu_ops/save_output_msg_with_topk.cc",
                "gpu_ops/transfer_output.cc",
                "gpu_ops/get_padding_offset.cu",
                "gpu_ops/set_value_by_flags_and_idx.cu",
                "gpu_ops/rebuild_padding.cu",
                "gpu_ops/update_inputs.cu",
                "gpu_ops/stop_generation_multi_ends.cu",
                "gpu_ops/step.cu",
                "gpu_ops/token_penalty_multi_scores.cu",
                "gpu_ops/sample_kernels/rejection_top_p_sampling.cu",
                "gpu_ops/sample_kernels/top_k_renorm_probs.cu",
                "gpu_ops/text_image_index_out.cu",
                "gpu_ops/text_image_gather_scatter.cu",
                "gpu_ops/set_data_ipc.cu",
                "gpu_ops/limit_thinking_content_length_v1.cu",
                "gpu_ops/limit_thinking_content_length_v2.cu",
                "iluvatar_ops/moe_dispatch.cu",
                "iluvatar_ops/moe_reduce.cu",
                "iluvatar_ops/paged_attn.cu",
                "iluvatar_ops/prefill_fused_attn.cu",
                "iluvatar_ops/mixed_fused_attn.cu",
                "iluvatar_ops/w8a16_group_gemm.cu",
                "iluvatar_ops/runtime/iluvatar_context.cc",
            ],
            include_dirs=["iluvatar_ops/runtime", "gpu_ops"],
            extra_link_args=[
                "-lcuinfer",
            ],
        ),
    )
elif paddle.is_compiled_with_custom_device("gcu"):
    setup(
        name="fastdeploy_ops",
        ext_modules=CppExtension(
            sources=[
                "gpu_ops/save_with_output_msg.cc",
                "gpu_ops/get_output.cc",
                "gpu_ops/get_output_msg_with_topk.cc",
            ]
        ),
    )
elif paddle.device.is_compiled_with_custom_device("metax_gpu"):
    maca_path = os.getenv("MACA_PATH", "/opt/maca")
    sources = [
        "gpu_ops/update_inputs_v1.cu",
        "gpu_ops/save_with_output_msg.cc",
        "gpu_ops/get_output.cc",
        "gpu_ops/get_output_msg_with_topk.cc",
        "gpu_ops/save_output_msg_with_topk.cc",
        "gpu_ops/transfer_output.cc",
        "gpu_ops/save_with_output.cc",
        "gpu_ops/set_mask_value.cu",
        "gpu_ops/set_value_by_flags_and_idx.cu",
        "gpu_ops/ngram_mask.cu",
        "gpu_ops/gather_idx.cu",
        "gpu_ops/get_output_ep.cc",
        "gpu_ops/token_penalty_multi_scores.cu",
        "gpu_ops/token_penalty_only_once.cu",
        "gpu_ops/stop_generation.cu",
        "gpu_ops/stop_generation_multi_ends.cu",
        "gpu_ops/set_flags.cu",
        "gpu_ops/fused_get_rotary_embedding.cu",
        "gpu_ops/get_padding_offset.cu",
        "gpu_ops/update_inputs.cu",
        "gpu_ops/update_inputs_beam.cu",
        "gpu_ops/beam_search_softmax.cu",
        "gpu_ops/rebuild_padding.cu",
        "gpu_ops/step.cu",
        "gpu_ops/step_reschedule.cu",
        "gpu_ops/step_system_cache.cu",
        "gpu_ops/set_data_ipc.cu",
        "gpu_ops/read_data_ipc.cu",
        "gpu_ops/dequant_int8.cu",
        "gpu_ops/share_external_data.cu",
        "gpu_ops/recover_decode_task.cu",
        "gpu_ops/noaux_tc.cu",
        "gpu_ops/noaux_tc_redundant.cu",
        "gpu_ops/fused_rotary_position_encoding.cu",
        "gpu_ops/text_image_gather_scatter.cu",
        "gpu_ops/text_image_index_out.cu",
        "gpu_ops/get_position_ids_and_mask_encoder_batch.cu",
        "gpu_ops/limit_thinking_content_length_v1.cu",
        "gpu_ops/limit_thinking_content_length_v2.cu",
        "gpu_ops/update_attn_mask_offsets.cu",
        "gpu_ops/append_attn/mla_cache_kernel.cu",
        "gpu_ops/append_attn/get_block_shape_and_split_kv_block.cu",
        "gpu_ops/moe/tritonmoe_preprocess.cu",
        "gpu_ops/moe/moe_topk_select.cu",
        "metax_ops/moe_dispatch.cu",
        "metax_ops/moe_ffn.cu",
        "metax_ops/moe_reduce.cu",
        "metax_ops/fused_moe.cu",
        "metax_ops/apply_rope.cu",
    ]

    sources += find_end_files("gpu_ops/speculate_decoding", ".cu")
    sources += find_end_files("gpu_ops/speculate_decoding", ".cc")

    setup(
        name="fastdeploy_ops",
        ext_modules=CUDAExtension(
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-Ithird_party/nlohmann_json/include",
                    "-Igpu_ops",
                    "-DPADDLE_DEV",
                    "-DPADDLE_WITH_CUSTOM_DEVICE_METAX_GPU",
                ],
            },
            library_dirs=[os.path.join(maca_path, "lib")],
            extra_link_args=["-lruntime_cu", "-lmctlassEx"],
            include_dirs=[
                os.path.join(maca_path, "include"),
                os.path.join(maca_path, "include/mcr"),
                os.path.join(maca_path, "include/common"),
            ],
        ),
    )
elif paddle.is_compiled_with_custom_device("intel_hpu"):
    pass
else:
    use_bf16 = envs.FD_CPU_USE_BF16 == "True"

    # cc flags
    paddle_extra_compile_args = [
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-Wno-parentheses",
        "-DPADDLE_WITH_CUSTOM_KERNEL",
        "-DPADDLE_ON_INFERENCE",
        "-Wall",
        "-O3",
        "-g",
        "-lstdc++fs",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        "-DPy_LIMITED_API=0x03090000",
    ]

    setup(
        name="fastdeploy_cpu_ops",
        ext_modules=CppExtension(
            sources=[
                "gpu_ops/save_with_output_msg.cc",
                "gpu_ops/get_output.cc",
                "gpu_ops/get_output_msg_with_topk.cc",
                "gpu_ops/save_output_msg_with_topk.cc",
                "gpu_ops/transfer_output.cc",
                "cpu_ops/rebuild_padding.cc",
                "cpu_ops/simd_sort.cc",
                "cpu_ops/set_value_by_flags.cc",
                "cpu_ops/token_penalty_multi_scores.cc",
                "cpu_ops/stop_generation_multi_ends.cc",
                "cpu_ops/update_inputs.cc",
                "cpu_ops/get_padding_offset.cc",
            ],
            extra_link_args=[
                "-Wl,-rpath,$ORIGIN/x86-simd-sort/builddir",
                "-Wl,-rpath,$ORIGIN/xFasterTransformer/build",
            ],
            extra_compile_args=paddle_extra_compile_args,
        ),
        packages=find_namespace_packages(where="third_party"),
        package_dir={"": "third_party"},
        include_package_data=True,
    )
