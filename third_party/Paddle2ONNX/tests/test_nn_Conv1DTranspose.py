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
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 dilation=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCL"):
        super(Net, self).__init__()
        self._conv2d_t = paddle.nn.Conv1DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, inputs):
        """
        forward
        """
        x = self._conv2d_t(inputs)
        return x


def test_Conv1DTranspose_9():
    """
    api: paddle.Conv1DTranspose
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


def test_Conv1DTranspose_10():
    """
    api: paddle.nn.Conv1DTranspose
    op version: 10
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


def test_Conv1DTranspose_11():
    """
    api: paddle.nn.Conv1DTranspose
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


def test_Conv1DTranspose_12_VALID():
    """
    api: paddle.nn.Conv1DTranspose
    op version: 12
    """
    op = Net(padding='VALID')
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


def test_Conv1DTranspose_12_SAME():
    """
    api: paddle.nn.Conv1DTranspose
    op version: 12
    """
    op = Net(padding='SAME')
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


def test_Conv1DTranspose_12_Padding_list1():
    """
    api: paddle.nn.Conv1DTranspose
    op version: 12
    """
    op = Net(padding=[1, 2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


def test_Conv1DTranspose_12_Padding_list2():
    """
    api: paddle.nn.Conv1DTranspose
    op version: 12
    """
    op = Net(padding=[[0, 0], [0, 0], [2]])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


def test_Conv1DTranspose_12_Padding_tuple1():
    """
    api: paddle.nn.Conv1DTranspose
    op version: 12
    """
    op = Net(padding=(1, 2))
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Conv1DTranspose', [9, 10, 11, 12, 13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10]).astype('float32')))
    obj.run()


# if __name__ == '__main__':
#     test_Conv1DTranspose_9()
#     test_Conv1DTranspose_10()
#     test_Conv1DTranspose_11()
#     test_Conv1DTranspose_12_VALID()
#     test_Conv1DTranspose_12_SAME()
#     test_Conv1DTranspose_12_Padding_list1()
#     test_Conv1DTranspose_12_Padding_list2()
#     test_Conv1DTranspose_12_Padding_tuple1()
