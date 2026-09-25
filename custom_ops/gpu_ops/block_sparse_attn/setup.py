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
Standalone build for Block-Sparse-Attention (BSA) Paddle custom op.

BSA bundles its own CUTLASS 3.3 in
  /root/paddlejob/share-storage/gpfs/system-public/tzx/SongGuo/Block-Sparse-Attention/csrc/cutlass/include
which has API conflicts with FastDeploy's newer CUTLASS in
  FastDeploy/custom_ops/third_party/cutlass
Since nvcc -I flags are global per compilation, we build BSA in its own
extension (independent .so) so that it sees ONLY BSA's CUTLASS headers.

Supported GPUs (auto-detected from CUDA toolkit version, see
``_build_gencode_flags`` below):
    - sm_80  : Ampere   (A100, A800)              -- always emitted
    - sm_90  : Hopper   (H100, H800)              -- CUDA >= 11.8
    - sm_100 : Blackwell (B100, B200, GB200)      -- CUDA >= 12.8
The BSA kernels themselves are sm_80-native (m16n8k16 mma, no wgmma /
tcgen05), so adding sm_90 / sm_100 only requires the gencode flags here;
no kernel rewrite is needed when migrating from H800 -> B200.

Usage:
    cd FastDeploy/custom_ops/gpu_ops/block_sparse_attn
    python setup.py build_ext --inplace        # local build for dev
    # or
    python setup.py install                    # system install

Override architectures explicitly:
    BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90;100" python setup.py install

Output: block_sparse_attn_ops*.so containing the `block_sparse_attn_fwd` op.
"""
import glob
import os
import subprocess
from pathlib import Path

from packaging.version import Version, parse

from paddle.utils.cpp_extension import CUDAExtension, setup


def _nvcc_version() -> Version:
    """Return CUDA toolkit version reported by nvcc -V (e.g. 12.8)."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    try:
        out = subprocess.check_output([f"{cuda_home}/bin/nvcc", "-V"], universal_newlines=True)
        tok = out.split()
        idx = tok.index("release") + 1
        return parse(tok[idx].split(",")[0])
    except Exception:
        # If nvcc unavailable, assume oldest supported toolchain (12.0).
        return parse("12.0")


def _build_gencode_flags() -> list[str]:
    """Mirror the upstream BSA repo's add_cuda_gencodes:

    - sm_80 always (Ampere baseline; kernels are sm80-native).
    - sm_90 (Hopper, e.g. H100/H800) when CUDA >= 11.8.
    - sm_100 (Blackwell, e.g. B100/B200) when CUDA >= 12.8.
        * On CUDA >= 12.9 use the family-specific arch=compute_100f.
    - PTX for the newest target arch for forward compatibility.

    BSA kernel sources are written with sm_80 baseline instructions
    (m16n8k16 mma, no wgmma / tcgen05), so they compile cleanly for
    sm_90 and sm_100 with these gencodes — no kernel rewrite needed.
    """
    archs = os.environ.get("BLOCK_SPARSE_ATTN_CUDA_ARCHS", "80;90;100").split(";")
    archs = {a.strip() for a in archs if a.strip()}

    cuda_ver = _nvcc_version()
    flags: list[str] = []

    if "80" in archs:
        flags += ["-gencode", "arch=compute_80,code=sm_80"]
    if "90" in archs and cuda_ver >= Version("11.8"):
        flags += ["-gencode", "arch=compute_90,code=sm_90"]
    if "100" in archs and cuda_ver >= Version("12.8"):
        if cuda_ver >= Version("12.9"):
            # Blackwell family-specific (introduced in CUDA 12.9).
            flags += ["-gencode", "arch=compute_100f,code=sm_100"]
        else:
            flags += ["-gencode", "arch=compute_100,code=sm_100"]
    # Embed PTX of the newest selected arch so future GPUs JIT-compile.
    numeric = sorted((a for a in archs if a.isdigit()), key=int)
    if numeric:
        if numeric[-1] == "100" and cuda_ver < Version("12.8"):
            # CUDA toolkit too old for Blackwell PTX -> fall back to sm_90 PTX.
            newest = "90" if "90" in archs else "80"
        else:
            newest = numeric[-1]
        flags += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]
    return flags

THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# BSA upstream repo provides its own CUTLASS 3.3 (incompatible with newer
# CUTLASS used elsewhere in FastDeploy). Resolve relative to this file via
# the existing `src` symlink target, so the build keeps working if the BSA
# repo lives at any path on disk.
BSA_REPO_ROOT = (THIS_DIR / "src").resolve().parents[1]   # .../Block-Sparse-Attention/csrc
BSA_CUTLASS_INCLUDE = BSA_REPO_ROOT / "cutlass" / "include"
assert BSA_CUTLASS_INCLUDE.exists(), (
    f"BSA bundled CUTLASS not found at {BSA_CUTLASS_INCLUDE}; "
    f"expected the upstream Block-Sparse-Attention checkout to include csrc/cutlass/include."
)

# --- Sources: wrapper + 12 forward kernels (skip backward) -------------------
sources = [str(THIS_DIR / "block_sparse_attn_fwd.cu")]
sources += [
    s for s in sorted(glob.glob(str(THIS_DIR / "src" / "*.cu")))
    if "flash_bwd_" not in os.path.basename(s)
]

# --- Compile flags -----------------------------------------------------------
nvcc_flags = [
    "-O3",
    "-std=c++17",
    # Enable __half/__bfloat16 native arithmetic ops used by BSA kernels.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    # Paddle's compat layer provides C10_CUDA_CHECK / CompatException but
    # does NOT define C10_CUDA_KERNEL_LAUNCH_CHECK. Force-define it as a
    # no-op so BSA's flash_fwd_launch_template.h compiles.
    "-DC10_CUDA_KERNEL_LAUNCH_CHECK()=",
    "-DENABLE_BF16",
    # GPU arch gencodes. sm_80 (A100), sm_90 (H100/H800), sm_100 (B100/B200)
    # are emitted automatically based on the installed CUDA toolkit. Override
    # with `BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90;100"` if needed.
    *_build_gencode_flags(),
]

cxx_flags = ["-O3", "-std=c++17"]

include_dirs = [
    str(THIS_DIR),                     # for at_shim/, src/, headers
    str(THIS_DIR / "src"),
    str(THIS_DIR / "at_shim"),         # torch -> paddle stubs
    str(BSA_CUTLASS_INCLUDE),          # BSA bundled CUTLASS 3.3
]

setup(
    name="block_sparse_attn_ops",
    ext_modules=CUDAExtension(
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
)
