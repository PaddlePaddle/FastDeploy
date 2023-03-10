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
import paddle.nn as nn
from onnxbase import APIOnnx
from onnxbase import randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=False,
                 align_mode=0,
                 data_format='NCHW'):
        super(Net, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format

    def forward(self, inputs):
        """
        forward
        """
        x = nn.functional.interpolate(
            x=inputs,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)
        return x


def test_nn_functional_interpolate_nearest_scale_factor_float():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(scale_factor=1.5)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [2, 3, 6, 10]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_nearest_scale_factor_list():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(scale_factor=[1, 2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [2, 3, 6, 10]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_nearest_scale_factor_tuple():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(scale_factor=(1, 2))
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [2, 3, 6, 10]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_nearest_size():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(size=[4, 11])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [2, 2, 2, 10]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_nearest_date_format():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(size=[4, 12], data_format='NCHW')
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [2, 2, 2, 10]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_bicubic_scale_factor_float():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(mode='bicubic', scale_factor=1.5)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 2, 2, 5]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_bicubic_scale_factor_list():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(mode='bicubic', scale_factor=[1.5, 1.5])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 2, 2, 5]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_bicubic_scale_factor_tuple():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(mode='bicubic', scale_factor=(1.5, 1.5))
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 2, 2, 5]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_bicubic_size():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(mode='bicubic', size=[5, 5])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 2, 2, 5]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_bicubic_align_corners():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(mode='bicubic', scale_factor=1.5, align_corners=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 2, 2, 5]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_linear_scale_factor_float():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(mode='linear',
             scale_factor=1.5,
             align_corners=False,
             align_mode=1,
             data_format="NCW")
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 2, 2]).astype('float32')))
    obj.run()


def test_nn_functional_interpolate_linear_scale_factor_tensor():
    """
    api: paddle.nn.functional.interpolate
    op version: 11
    """
    op = Net(mode='linear',
             size=paddle.to_tensor(
                 12, dtype='int32'),
             align_corners=False,
             align_mode=1,
             data_format="NCW")
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_functional_interpolate', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 2, 2]).astype('float32')))
    obj.run()
