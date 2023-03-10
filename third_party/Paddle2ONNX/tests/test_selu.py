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
                 alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946):
        super(Net, self).__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.nn.functional.selu(
            inputs, alpha=self.alpha, scale=self.scale)
        return x


def test_nn_functional_selu_10():
    """
    api: paddle.nn.functional.selu
    op version: 10
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn.functional.selu', [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 3, 3]).astype('float32')))
    obj.run()


def test_nn_functional_selu_11():
    """
    api: paddle.nn.functional.selu
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn.functional.selu', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 3, 3]).astype('float32')))
    obj.run()


def test_nn_functional_selu_12():
    """
    api: paddle.nn.functional.selu
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn.functional.selu', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 3, 3]).astype('float32')))
    obj.run()


def test_nn_functional_selu_scale():
    """
    api: paddle.nn.functional.selu
    op version: 12
    """
    op = Net(scale=2)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn.functional.selu', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 3, 3]).astype('float32')))
    obj.run()


def test_nn_functional_selu_alpha():
    """
    api: paddle.nn.functional.selu
    op version: 12
    """
    op = Net(alpha=2)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn.functional.selu', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 3, 3]).astype('float32')))
    obj.run()
