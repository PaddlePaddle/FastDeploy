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

    def __init__(self, axis=None):
        super(Net, self).__init__()
        self.axis = axis

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.unique(inputs, axis=self.axis)

        return x


def test_unique_11():
    """
    api: paddle.unique
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unique_12():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unique_axis():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net(axis=1)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


class Net_mult_2(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self,
                 return_index=False,
                 return_inverse=False,
                 return_counts=False,
                 axis=None):
        super(Net_mult_2, self).__init__()
        self.return_index = return_index
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.axis = axis

    def forward(self, inputs):
        """
        forward
        """
        x, y = paddle.unique(
            inputs,
            axis=self.axis,
            return_index=self.return_index,
            return_inverse=self.return_inverse,
            return_counts=self.return_counts)

        return x + y


def test_unique_return_index():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net_mult_2(return_index=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unique_return_inverse():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net_mult_2(return_inverse=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unique_return_counts():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net_mult_2(return_counts=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


class Net_mult_3(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self,
                 return_index=False,
                 return_inverse=False,
                 return_counts=False,
                 axis=None):
        super(Net_mult_3, self).__init__()
        self.return_index = return_index
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.axis = axis

    def forward(self, inputs):
        """
        forward
        """
        x, y, z = paddle.unique(
            inputs,
            axis=self.axis,
            return_index=self.return_index,
            return_inverse=self.return_inverse,
            return_counts=self.return_counts)

        return x + y + z


def test_unique_return_index_inverse():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net_mult_3(return_index=True, return_inverse=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unique_return_index_counts():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net_mult_3(return_index=True, return_counts=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_unique_return_inverse_counts():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net_mult_3(return_inverse=True, return_counts=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


class Net_mult_all(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, axis=None):
        super(Net_mult_all, self).__init__()
        self.axis = axis

    def forward(self, inputs):
        """
        forward
        """
        x, y, z, w = paddle.unique(
            inputs,
            axis=self.axis,
            return_index=True,
            return_inverse=True,
            return_counts=True)

        return x + y + z + w


def test_unique_return_all():
    """
    api: paddle.unique
    op version: 12
    """
    op = Net_mult_all()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unique', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()
