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
        x = paddle.nn.functional.upsample(
            x=inputs,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)
        return x


def test_Unsample_size():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(size=[12, 12], align_mode=1)

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


# has a bug
# def test_Unsample_size_tensor():
#     """
#     api: paddle.Upsample
#     op version: 11, 12
#     """
#     op = Net(scale_factor=(paddle.to_tensor(2), paddle.to_tensor(2)),
#              align_mode=1)
#
#     op.eval()
#     # net, name, ver_list, delta=1e-6, rtol=1e-5
#     obj = APIOnnx(op, 'nn_Unsample', [11, 12])
#     obj.set_input_data(
#         "input_data",
#         paddle.to_tensor(
#             randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
#     obj.run()


def test_Unsample_scale_factor():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(scale_factor=[2, 3])

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_size_linear_tensor():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(size=paddle.to_tensor(
        12, dtype='int32'),
             mode='linear',
             data_format='NCW',
             align_mode=1)

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 1, 10]).astype('float32')))
    obj.run()


def test_Unsample_size_linear():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(size=[12],
             mode='linear',
             align_corners=False,
             align_mode=1,
             data_format="NCW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 1, 10]).astype('float32')))
    obj.run()


def test_Unsample_scale_factor_linear():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(scale_factor=[1.5],
             mode='linear',
             align_corners=False,
             align_mode=1,
             data_format="NCW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 1, 10]).astype('float32')))
    obj.run()


def test_Unsample_size_bilinear():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(size=[12, 15],
             mode='bilinear',
             align_corners=False,
             align_mode=1,
             data_format="NCHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_scale_factor_bilinear():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(scale_factor=[2, 3],
             mode='bilinear',
             align_corners=False,
             align_mode=1,
             data_format="NCHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_size_nearest():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(size=[12, 15],
             mode='nearest',
             align_corners=False,
             align_mode=1,
             data_format="NCHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_scale_factor_nearest():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(scale_factor=[2, 3],
             mode='nearest',
             align_corners=False,
             align_mode=1,
             data_format="NCHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_size_bicubic():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(size=[12, 15],
             mode='bicubic',
             align_corners=False,
             align_mode=1,
             data_format="NCHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_scale_factor_bicubic():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(scale_factor=[2, 3],
             mode='bicubic',
             align_corners=False,
             align_mode=1,
             data_format="NCHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_size_trilinear():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(size=[12, 15, 20],
             mode='trilinear',
             align_corners=False,
             align_mode=1,
             data_format="NCDHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10, 10]).astype('float32')))
    obj.run()


def test_Unsample_scale_factor_trilinear():
    """
    api: paddle.Upsample
    op version: 11, 12
    """
    op = Net(scale_factor=[2, 3, 4],
             mode='trilinear',
             align_corners=False,
             align_mode=1,
             data_format="NCDHW")

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Unsample', [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10, 10]).astype('float32')))
    obj.run()


# if __name__ == '__main__':
#     test_Unsample_size()
#     test_Unsample_scale_factor()
#     test_Unsample_size_linear_tensor()
#     test_Unsample_size_linear()
#     test_Unsample_scale_factor_linear()
#     test_Unsample_size_bilinear()
#     test_Unsample_scale_factor_bilinear()
#     test_Unsample_size_nearest()
#     test_Unsample_scale_factor_nearest()
#     test_Unsample_size_bicubic()
#     test_Unsample_scale_factor_bicubic()
#     test_Unsample_size_trilinear()
#     test_Unsample_scale_factor_trilinear()
