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

    def __init__(self,
                 in_channels=1,
                 out_channels=2,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW'):
        super(Net, self).__init__()
        self.conv2d = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, inputs):
        """
        forward
        """
        x = self.conv2d(inputs)
        return x


def test_Conv2D_9():
    """
    api: paddle.Conv2D
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2D', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_10():
    """
    api: paddle.nn.Conv2D
    op version: 10
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2D', [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_11():
    """
    api: paddle.nn.Conv2D
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2D', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_12():
    """
    api: paddle.nn.Conv2D
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2D', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_padding_0_9():
    """
    api: paddle.nn.Conv2D
    op version: 9
    """
    op = Net(padding=[1, 2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_padding_1_9():
    """
    api: paddle.nn.Conv3D
    op version: 9
    """
    op = Net(padding=[1, 2, 3, 4])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_padding_2_9():
    """
    api: paddle.nn.Conv2D
    op version: 9
    """
    op = Net(padding=[[0, 0], [0, 0], [1, 2], [2, 3]])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_groups_1_9():
    """
    api: paddle.nn.Conv2D
    op version: 9
    """
    op = Net(in_channels=16, out_channels=16, groups=4)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 16, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_groups_2_9():
    """
    api: paddle.nn.Conv2D
    op version: 9
    """
    op = Net(in_channels=16, out_channels=16, groups=16)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 16, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2D_dilation_2_9():
    """
    api: paddle.nn.Conv2D
    op version: 9
    """
    op = Net(in_channels=16, out_channels=16, dilation=3)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 16, 10, 10]).astype('float32')))
    obj.run()
