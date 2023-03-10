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
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(Net, self).__init__()
        self._conv2d_t = paddle.nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, inputs):
        """
        forward
        """
        x = self._conv2d_t(inputs)
        return x


def test_Conv2DTranspose_9_10_11_12():
    """
    api: paddle.Conv2DTranspose
    op version: 9, 10, 11, 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2DTranspose', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2DTranspose_9_output_padding():
    """
    api: paddle.Conv2DTranspose
    op version: 9
    """
    op = Net(output_padding=1, stride=[3, 2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2DTranspose', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2DTranspose_9_output_padding_1():
    """
    api: paddle.Conv2DTranspose
    op version: 9
    """
    op = Net(output_padding=1, stride=[3, 2], padding=[1, 2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2DTranspose', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2DTranspose_9_output_padding_2():
    """
    api: paddle.Conv2DTranspose
    op version: 9
    """
    op = Net(output_padding=1, stride=[3, 2], padding=[1, 2, 3, 4])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2DTranspose', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2DTranspose_9_output_padding_3():
    """
    api: paddle.Conv2DTranspose
    op version: 9
    """
    op = Net(output_padding=1,
             stride=[3, 2],
             padding=[[0, 0], [0, 0], [1, 2], [2, 3]])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2DTranspose', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Conv2DTranspose_9_10_11_12_group():
    """
    api: paddle.Conv2DTranspose
    op version: 9
    """
    op = Net(in_channels=16,
             out_channels=16,
             groups=16,
             padding=[[0, 0], [0, 0], [1, 2], [2, 3]])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv2DTranspose', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [16, 16, 10, 10]).astype('float32')))
    obj.run()


# if __name__ == '__main__':
#     test_Conv2DTranspose_9_10_11_12_group()
#     test_Conv2DTranspose_9_output_padding_3()
#     test_Conv2DTranspose_9_output_padding_2()
#     test_Conv2DTranspose_9_output_padding_1()
#     test_Conv2DTranspose_9_output_padding()
#     test_Conv2DTranspose_9_10_11_12()