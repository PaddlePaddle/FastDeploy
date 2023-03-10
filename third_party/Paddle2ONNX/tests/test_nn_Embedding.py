# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from onnxbase import APIOnnx
from onnxbase import randtool
import numpy as np


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()
        self._embedding = paddle.nn.Embedding(
            num_embeddings=10,
            embedding_dim=3,
            padding_idx=None,
            sparse=True,
            weight_attr=None,
            name=None)

    def forward(self, inputs):
        """
        forward
        """
        x = self._embedding(inputs)
        return x


def test_Embedding_base():
    """
    api: paddle.Embedding
    op version: 9, 10, 11, 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Embedding', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(np.arange(3, 6).reshape((3, 1)).astype(np.int64)))
    obj.run()
