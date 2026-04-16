"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import subprocess
import sys


def test_run_distributed():
    """Launch multi-GPU distributed test via paddle.distributed.launch as subprocess"""
    # clearn flashinfer cache directory
    flashinfer_cache_dir = os.path.join(os.sep, "root", ".cache", "flashinfer")
    if os.path.exists(flashinfer_cache_dir):
        print(f"=== Clearing flashinfer cache directory: {flashinfer_cache_dir} ===")
        subprocess.run(["rm", "-rf", flashinfer_cache_dir], check=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_script = os.path.join(current_dir, "trtllm_allreduce_rms_fusion.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    command = [
        sys.executable,
        "-m",
        "paddle.distributed.launch",
        "--gpus",
        "0,1",
        run_script,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        stdout, stderr = process.communicate(timeout=400)
        return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return_code = -1
    print(f"=== Distributed test stdout ===\n{stdout}")
    print(f"=== Distributed test stderr ===\n{stderr}")
    assert return_code in (0, 250), f"Process exited with code {return_code}"


test_run_distributed()
