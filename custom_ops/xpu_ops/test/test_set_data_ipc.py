# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import time

import paddle

from fastdeploy.model_executor.ops.xpu import set_data_ipc

x = paddle.full(shape=[512, 8, 64, 128], fill_value=2, dtype="float32")
set_data_ipc(x, "test_set_data_ipc")
print("set_data_ipc done")

time.sleep(60)
