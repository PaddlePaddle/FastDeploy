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

    def __init__(self, axis=0):
        super(Net, self).__init__()
        self.axis = axis

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.unsqueeze(inputs, axis=self.axis)
        return x


def test_unsqueeze_9():
    """
    api: paddle.unsqueeze
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_10():
    """
    api: paddle.unsqueeze
    op version: 10
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_11():
    """
    api: paddle.unsqueeze
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_12():
    """
    api: paddle.unsqueeze
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_axis_13():
    """
    api: paddle.unsqueeze
    op version: 13
    """
    op = Net(axis=paddle.to_tensor(1))
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_13_two_tensor_axis():
    """
    api: paddle.unsqueeze
    op version: 13
    """
    op = Net(axis=paddle.to_tensor([0, -1]))
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_9_two_axis():
    """
    api: paddle.unsqueeze
    op version: 9
    """
    op = Net(axis=[0, -1])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_9_multil_axis():
    """
    api: paddle.unsqueeze
    op version: 9
    """
    op = Net(axis=[1, 2, 3, 4])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [5, 10]).astype('float32')))
    obj.run()


def test_unsqueeze_9_multil_negative_axis():
    """
    api: paddle.unsqueeze
    op version: 9
    """
    op = Net(axis=[1, 2, 3, -1])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unsqueeze', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [5, 10]).astype('float32')))
    obj.run()


# if __name__ == '__main__':
#     test_unsqueeze_9()
#     test_unsqueeze_10()
#     test_unsqueeze_11()
#     test_unsqueeze_12()
#     test_unsqueeze_axis_12()
#     test_unsqueeze_9_two_tensor_axis()
#     test_unsqueeze_9_two_axis()
#     test_unsqueeze_9_multil_axis()
#     test_unsqueeze_9_multil_negative_axis()
