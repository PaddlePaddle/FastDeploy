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


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, overwrite=True):
        super(Net, self).__init__()
        self.overwrite = overwrite

    def forward(self, inputs, _index, _updates):
        """
        forward
        """
        x = paddle.scatter(inputs, _index, _updates, overwrite=self.overwrite)
        return x


def test_scatter_11():
    """
    api: paddle.scatter
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'scatter', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([[1, 1], [2, 2], [3, 3]]).astype('float32'),
        paddle.to_tensor([2, 1, 0]).astype('int64'),
        paddle.to_tensor([[1, 1], [2, 2], [3, 3]]).astype('float32'))
    obj.run()


def test_scatter_12():
    """
    api: paddle.scatter
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'scatter', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([[1, 1], [2, 2], [3, 3]]).astype('float32'),
        paddle.to_tensor([2, 1, 0]).astype('int64'),
        paddle.to_tensor([[1, 1], [2, 2], [3, 3]]).astype('float32'))
    obj.run()
