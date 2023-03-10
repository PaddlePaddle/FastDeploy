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

    def __init__(self):
        super(Net, self).__init__()
        self.shuffle = paddle.nn.PixelShuffle(
            upscale_factor=3, data_format='NCHW', name=None)

    def forward(self, inputs):
        """
        forward
        """
        x = self.shuffle(inputs)
        return x


def test_PixelShuffle_base():
    """
    api: paddle.nn.PixelShuffle
    op version: 9, 10, 11, 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-10, rtol=1e-11
    obj = APIOnnx(op, 'nn_PixelShuffle', [11, 12])
    obj.set_input_data("input_data", paddle.rand((2, 9, 4, 4)))
    obj.run()
