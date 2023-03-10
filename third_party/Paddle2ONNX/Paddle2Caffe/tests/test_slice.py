# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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
from caffebase import APICaffe
from caffebase import randtool


class Net0(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net0, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.slice(inputs, axes=(2, 3), starts=(1, 1), ends=(1000, 1000))
        return x


class Net1(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net1, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.slice(inputs, axes=(2, 3), starts=(0, 0), ends=(-1, -1))
        return x


def test_slice_right_bottom():
    """
    api: paddle.slice
    """
    op = Net0()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'slice')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 3, 11, 11]).astype('float32')))
    obj.run()


def test_slice_left_top():
    """
    api: paddle.slice
    """
    op = Net1()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'slice')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 3, 11, 11]).astype('float32')))
    obj.run()


if __name__ == '__main__':
    # test_slice_right_bottom()
    test_slice_left_top()
