# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import

import numpy as np
import paddle
from paddle.fluid import layers
from paddle2onnx.legacy.op_mapper import CustomPaddleOp, register_custom_paddle_op
from paddle2onnx import utils
from paddle2onnx.legacy.constant import dtypes
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.legacy.op_mapper import mapper_helper


class DeformConv2d(CustomPaddleOp):
    def check_attribute(self, node):
        utils.compare_attr_between_dims(
            node.attr('strides'), (0, 1), 'strides', 'equal')
        utils.compare_attr_between_dims(
            node.attr('paddings'), (0, 1), 'paddings', 'equal')
        utils.compare_attr_between_dims(
            node.input_shape('Offset', 0), (2, 3), 'Offset', 'equal')
        utils.compare_attr(
            node.attr('deformable_groups'), 1, 'deformable_groups', 'equal')

    def __init__(self, node, **kw):
        super(DeformConv2d, self).__init__(node)
        self.check_attribute(node)
        self.in_channel = node.input_shape('Input', 0)[1]
        self.offset_channel = node.input_shape('Offset', 0)[1]
        self.stride = node.attr('strides')[0]
        self.padding = node.attr('paddings')
        if len(self.padding) == 2:
            self.padding += self.padding
        self.groups = node.attr('groups')
        self.dilation = node.attr('dilations')[0]
        self.padded_x_h = node.input_shape('Input', 0)[2]
        self.padded_x_w = node.input_shape('Input', 0)[3]
        if self.padded_x_h > 0:
            self.padded_x_h = self.padded_x_h + self.padding[0] + self.padding[1]
        if self.padded_x_w > 0:
            self.padded_x_w = self.padded_x_w + self.padding[2] + self.padding[3]

        self.kernel_size = node.input_shape('Filter', 0)[2]
        self.N = self.kernel_size**2
        self.num_filters = node.input_shape('Filter', 0)[0]

    def forward(self):
        input = self.input('Input', 0)
        weight = self.input('Filter', 0)
        mask = self.input('Mask', 0)
        offset = self.input('Offset', 0)

        input = layers.pad2d(input, self.padding)
        input_shape = paddle.shape(input)
        if self.padded_x_h < 0 or self.padded_x_w < 0:
            self.padded_x_h = input_shape[2]
            self.padded_x_w = input_shape[3]

        offset_x = paddle.strided_slice(
            offset,
            axes=[1],
            starts=[0],
            ends=[self.offset_channel],
            strides=[2])
        offset_y = paddle.strided_slice(
            offset,
            axes=[1],
            starts=[1],
            ends=[self.offset_channel],
            strides=[2])
        offset = paddle.concat([offset_x, offset_y], axis=1)
        offset_shape = paddle.shape(offset)
        offset_h = offset_shape[2]
        offset_w = offset_shape[3]

        coordinate = self.get_offset_coordinate(offset, 'float32', offset_shape)

        coordinate = coordinate.transpose((0, 2, 3, 1))
        coord_lt, coord_rb, coord_lb, coord_rt = self.get_bilinear_corner_coordinate(
            coordinate, self.padded_x_h, self.padded_x_w)

        # clip coordinate
        coordinate = paddle.concat(
            [
                paddle.clip(coordinate[:, :, :, :self.N], 0,
                            self.padded_x_h - 1),
                paddle.clip(coordinate[:, :, :, self.N:], 0,
                            self.padded_x_w - 1)
            ],
            axis=-1)

        cof_lt, cof_rb, cof_lb, cof_rt = self.get_bilinear_coefficient(
            coord_lt, coord_rb, coord_lb, coord_rt, coordinate)

        feature_lt = self.get_feature_by_coordinate(input, coord_lt, offset_h,
                                                    offset_w, self.padded_x_w)
        feature_rb = self.get_feature_by_coordinate(input, coord_rb, offset_h,
                                                    offset_w, self.padded_x_w)
        feature_lb = self.get_feature_by_coordinate(input, coord_lb, offset_h,
                                                    offset_w, self.padded_x_w)
        feature_rt = self.get_feature_by_coordinate(input, coord_rt, offset_h,
                                                    offset_w, self.padded_x_w)

        feature_after_deformation = paddle.unsqueeze(cof_lt, 1) * feature_lt + \
                   paddle.unsqueeze(cof_rb, 1) * feature_rb + \
                   paddle.unsqueeze(cof_lb, 1) * feature_lb + \
                   paddle.unsqueeze(cof_rt, 1) * feature_rt

        # modulation
        if mask is not None:
            mask = paddle.transpose(mask, (0, 2, 3, 1))
            mask = paddle.unsqueeze(mask, 1)
            mask = paddle.tile(mask, [1, self.in_channel, 1, 1, 1])
            feature_after_deformation *= mask

        feature_after_deformation = self.reshape_feature(
            feature_after_deformation, offset_h, offset_w)

        out = paddle.nn.functional.conv2d(
            feature_after_deformation,
            weight,
            stride=self.kernel_size,
            groups=self.groups)

        return {'Output': [out]}

    def get_offset_coordinate(self, offset, dtype, offset_shape):
        kernel_grid_origin_x = paddle.arange(
            0,
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1),
            step=self.dilation,
            dtype=dtype)
        kernel_grid_origin_x = kernel_grid_origin_x.unsqueeze(1)
        kernel_grid_origin_x = paddle.tile(kernel_grid_origin_x,
                                           [1, self.kernel_size])
        kernel_grid_origin_y = paddle.arange(
            0,
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1),
            step=self.dilation,
            dtype=dtype)
        kernel_grid_origin_y = kernel_grid_origin_y.unsqueeze(0)
        kernel_grid_origin_y = paddle.tile(kernel_grid_origin_y,
                                           [self.kernel_size, 1])
        kernel_grid_origin_x = paddle.reshape(kernel_grid_origin_x, [-1])
        kernel_grid_origin_y = paddle.reshape(kernel_grid_origin_y, [-1])
        kernel_grid_origin = paddle.concat(
            [kernel_grid_origin_x, kernel_grid_origin_y], -1)
        kernel_grid_origin = paddle.reshape(kernel_grid_origin,
                                            (1, 2 * self.N, 1, 1))

        kernel_offset_x = paddle.arange(
            0, offset_shape[2] * self.stride, step=self.stride, dtype=dtype)
        kernel_offset_x = kernel_offset_x.unsqueeze(1)
        kernel_offset_x = paddle.expand(kernel_offset_x, offset_shape[2:])
        kernel_offset_y = paddle.arange(
            0, offset_shape[3] * self.stride, step=self.stride, dtype=dtype)
        kernel_offset_y = kernel_offset_y.unsqueeze(0)
        kernel_offset_y = paddle.expand(kernel_offset_y, offset_shape[2:])
        kernel_offset_x = kernel_offset_x.unsqueeze([0, 1])
        kernel_offset_x = paddle.tile(kernel_offset_x, (1, self.N, 1, 1))
        kernel_offset_y = kernel_offset_y.unsqueeze([0, 1])
        kernel_offset_y = paddle.tile(kernel_offset_y, (1, self.N, 1, 1))

        kernel_offset = paddle.concat([kernel_offset_x, kernel_offset_y], 1)
        offset = offset + paddle.cast(kernel_offset, 'float32') + paddle.cast(
            kernel_grid_origin, 'float32')

        return offset

    def get_bilinear_corner_coordinate(self, coord, padded_h, padded_w):
        coord_lt = coord.floor()
        coord_rb = coord_lt + 1
        coord_lt = paddle.cast(
            paddle.concat(
                [
                    paddle.clip(coord_lt[:, :, :, :self.N], 0, padded_h - 1),
                    paddle.clip(coord_lt[:, :, :, self.N:], 0, padded_w - 1)
                ],
                axis=-1),
            dtype='int64')
        coord_rb = paddle.cast(
            paddle.concat(
                [
                    paddle.clip(coord_rb[:, :, :, :self.N], 0, padded_h - 1),
                    paddle.clip(coord_rb[:, :, :, self.N:], 0, padded_w - 1)
                ],
                axis=-1),
            dtype='int64')
        coord_lb = paddle.concat(
            [coord_lt[:, :, :, :self.N], coord_rb[:, :, :, self.N:]], axis=-1)
        coord_rt = paddle.concat(
            [coord_rb[:, :, :, :self.N], coord_lt[:, :, :, self.N:]], axis=-1)

        return coord_lt, coord_rb, coord_lb, coord_rt

    def get_bilinear_coefficient(self, coord_lt, coord_rb, coord_lb, coord_rt,
                                 p):
        cof_lt = (1 + (paddle.cast(
            coord_lt[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
                  ) * (1 + paddle.cast(
                      coord_lt[:, :, :, self.N:], dtype='float32') -
                       p[:, :, :, self.N:])
        cof_rb = (1 - (paddle.cast(
            coord_rb[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
                  ) * (1 - (paddle.cast(
                      coord_rb[:, :, :, self.N:], dtype='float32') -
                            p[:, :, :, self.N:]))
        cof_lb = (1 + (paddle.cast(
            coord_lb[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
                  ) * (1 - (paddle.cast(
                      coord_lb[:, :, :, self.N:], dtype='float32') -
                            p[:, :, :, self.N:]))
        cof_rt = (1 - (paddle.cast(
            coord_rt[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
                  ) * (1 + paddle.cast(
                      coord_rt[:, :, :, self.N:], dtype='float32') -
                       p[:, :, :, self.N:])

        return cof_lt, cof_rb, cof_lb, cof_rt

    def get_feature_by_coordinate(self, x, coord, offset_h, offset_w,
                                  padded_x_w):
        x = paddle.reshape(x, [0, 0, -1])
        index = paddle.cast(
            coord[:, :, :, :self.N] * padded_x_w,
            dtype='int64') + coord[:, :, :, self.N:]  # offset_x*w + offset_y
        index = paddle.unsqueeze(index, 1)
        index = paddle.tile(index, [1, self.in_channel, 1, 1, 1])
        index = paddle.reshape(index, (0, 0, -1))
        x_range = list(range(3))
        dim = 2
        x_range[0] = dim
        x_range[dim] = 0
        x_swaped = paddle.transpose(x, perm=x_range)
        index_range = list(range(3))
        index_range[0] = dim
        index_range[dim] = 0
        index_swaped = paddle.transpose(index, perm=index_range)
        x_shape = layers.shape(x_swaped)
        index_shape = layers.shape(index_swaped)
        prod = paddle.prod(x_shape[1:], keepdim=True)
        x_swaped_flattend = paddle.reshape(x_swaped, [-1])
        index_swaped_flattend = paddle.reshape(index_swaped, [-1])
        index_swaped_flattend *= prod
        bias = paddle.arange(start=0, end=prod, step=1, dtype='float32')
        bias = paddle.tile(bias, index_shape[0])
        index_swaped_flattend += bias
        gathered = paddle.gather(x_swaped_flattend, index_swaped_flattend)
        gathered = paddle.reshape(gathered, layers.shape(index_swaped))
        x_offset = paddle.transpose(gathered, perm=x_range)
        x_offset = paddle.reshape(
            x_offset, (-1, self.in_channel, offset_h, offset_w, self.N))
        return x_offset

    def reshape_feature(self, x_offset, offset_h, offset_w):
        x_offset = paddle.concat(
            [
                paddle.reshape(x_offset[:, :, :, :, s:s + self.kernel_size], (
                    -1, self.in_channel, offset_h, offset_w * self.kernel_size))
                for s in range(0, self.N, self.kernel_size)
            ],
            axis=-1)
        x_offset = paddle.reshape(x_offset, (-1, self.in_channel,
                                             offset_h * self.kernel_size,
                                             offset_w * self.kernel_size))
        return x_offset

@op_mapper('deformable_conv')
class Deformconv2d:
    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'deformable_conv',
            inputs=node.input('Input')+node.input('Filter')+node.input('Mask')+node.input('Offset'),
            outputs=node.output('Output'),
            stride = node.attr('strides'),
            padding = node.attr('paddings'),
            groups = node.attr('groups'),
            dilation = node.attr('dilations'),
            deformable_groups = node.attr('deformable_groups'),
            domain = 'custom')
            
register_custom_paddle_op('deformable_conv', DeformConv2d)
