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

import os

os.environ.setdefault("DG_NVCC_OVERRIDE_CPP_STANDARD", "20")

import subprocess
import sys


def test_fused_moe_launch():
    """
    test_fused_moe
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    py_script = os.path.join(current_dir, "../layers/test_fusedmoe.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["USE_FUSEDMOE_TP"] = "1"

    command = [
        sys.executable,
        "-m",
        "paddle.distributed.launch",
        "--gpus",
        "0,1",
        "--master",
        "127.0.0.1:6175",
        "--nnodes",
        "1",
        "--rank",
        "0",
        py_script,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        stdout, stderr = process.communicate(timeout=400)
        return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return_code = -1
    print(f"std_out: {stdout}")
    assert return_code in (0, 250), f"Process exited with code {return_code}, stdout: {stdout}, stderr: {stderr}"


test_fused_moe_launch()
