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
""" UT for topp_sampling """
import paddle
import numpy as np
from fastdeploy.model_executor.ops.gpu import topp_sampling

paddle.seed(2022)

x = paddle.randn([4, 100000], dtype="float16")
x = paddle.nn.functional.softmax(x)
top_ps = paddle.to_tensor(
    np.array(
        [
            0.9,
        ]
        * 4
    ).astype(np.float16)
)
print(x)
print(top_ps)
out = topp_sampling(x, top_ps)
print(out)
