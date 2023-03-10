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


class Net1(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net1, self).__init__()

    def forward(self, input_x, input_y):
        """
        forward
        """
        x = paddle.add(input_x, input_y)
        return x


class Net2(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, input_x):
        """
        forward
        """
        x = paddle.add(input_x, paddle.to_tensor([2, 3, 4], 'float64'))
        return x


def test_elementwise_add_input():
    """
    api: paddle.nn.add
    """
    op = Net1()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'elementwise_add')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 5, 5]).astype('float32')),
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 5, 5]).astype('float32')))
    obj.run()


def test_elementwise_add_params():
    """
    api: paddle.nn.add
    """
    op = Net2()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'elementwise_add')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3]).astype('float64')))
    obj.run()


if __name__ == "__main__":
    test_elementwise_add_input()
    # test_elementwise_add_params()  # TODO
