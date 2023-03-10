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
        x = paddle.squeeze(inputs, axis=self.axis)
        return x


def test_squeeze_9():
    """
    api: paddle.squeeze
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 10]).astype('float32')))
    obj.run()


def test_squeeze_10():
    """
    api: paddle.squeeze
    op version: 10
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 10]).astype('float32')))
    obj.run()


def test_squeeze_11():
    """
    api: paddle.squeeze
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 10]).astype('float32')))
    obj.run()


def test_squeeze_12():
    """
    api: paddle.squeeze
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 10]).astype('float32')))
    obj.run()


def test_squeeze_9_None():
    """
    api: paddle.squeeze
    op version: 12
    """
    op = Net(axis=None)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 10, 1]).astype('float32')))
    obj.run()


# def test_squeeze_9_None_no_one():
#     """
#     api: paddle.squeeze
#     op version: 12
#     """
#     op = Net(axis=None)
#     op.eval()
#     # net, name, ver_list, delta=1e-6, rtol=1e-5
#     obj = APIOnnx(op, 'squeeze', [13])
#     obj.set_input_data(
#         "input_data",
#         paddle.to_tensor(
#             randtool("float", -1, 1, [3, 10]).astype('float32')))
#     obj.run()


def test_squeeze_9_None_has_one_negtive():
    """
    api: paddle.squeeze
    op version: 12
    """
    op = Net(axis=[0, -2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 1, 10]).astype('float32')))
    obj.run()


def test_squeeze_9_None_has_two_negtive1():
    """
    api: paddle.squeeze
    op version: 12
    """
    op = Net(axis=[3, 5])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 1, 1, 10, 1]).astype('float32')))
    obj.run()


def test_squeeze_9_None_has_two_negtive2():
    """
    api: paddle.squeeze
    op version: 12
    """
    op = Net(axis=[5, 3])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 1, 1, 10, 1]).astype('float32')))
    obj.run()


def test_squeeze_9_None_has_two_negtive():
    """
    api: paddle.squeeze
    op version: 12
    """
    op = Net(axis=[-1, 2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'squeeze', [13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 3, 1, 1, 10, 1]).astype('float32')))
    obj.run()


# if __name__ == '__main__':
#     test_squeeze_9()
#     test_squeeze_10()
#     test_squeeze_11()
#     test_squeeze_12()
#     test_squeeze_9_None()
#     # test_squeeze_9_None_no_one()
#     test_squeeze_9_None_has_one_negtive()
#     test_squeeze_9_None_has_two_negtive1()
#     test_squeeze_9_None_has_two_negtive2()
