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

import sys
import time

import paddle

from fastdeploy.model_executor.ops.xpu import set_data_ipc, share_external_data

shape = [8, 128]
dtype = "bfloat16"
shm_name = "xpu_shm_tensor"

paddle.set_device("xpu:0")

if sys.argv[1] == "0":
    print("set data ipc")
    input_tensor = paddle.cast(paddle.rand(shape), dtype)
    set_data_ipc(input_tensor, shm_name)
    print(input_tensor)
    time.sleep(120)
elif sys.argv[1] == "1":
    print("test share_external_data")
    tmp_input = paddle.empty([], dtype=dtype)
    output = share_external_data(tmp_input, shm_name, shape, use_ipc=True)
    print(output.shape)
    print(output.cpu())  # use xpu_memcpy
else:
    print("test share_external_data")
    tmp_input = paddle.empty([], dtype=dtype)
    output = share_external_data(tmp_input, shm_name, shape, use_ipc=False)
    temp_output = output * 1  # avoid xpu_memcpy
    print(temp_output)
