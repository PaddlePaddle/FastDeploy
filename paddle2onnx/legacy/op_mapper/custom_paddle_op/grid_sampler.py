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
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.legacy.op_mapper import mapper_helper


class GridSampler(CustomPaddleOp):
    def __init__(self, node, **kw):
        super(GridSampler, self).__init__(node)
        self.padding_mode = node.attr('padding_mode')
        self.mode = node.attr('mode')
        self.align_corners = node.attr('align_corners')

    def paddle_bilinear_grid_sample(self, im, grid, align_corners=False):
        # this code reference: https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/point_sample.html
        im_shape = paddle.shape(im)
        n, c, h, w = paddle.split(im_shape, num_or_sections=4)
        grid_shape = paddle.shape(grid)
        gn, gh, gw, _ = paddle.split(grid_shape, num_or_sections=4)

        # n, c, h, w = im.shape
        # gn, gh, gw, _ = grid.shape
        # assert n == gn

        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]

        if align_corners:
            x = ((x + 1) / 2) * (w - 1)
            y = ((y + 1) / 2) * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2

        x = paddle.reshape(x, [n, -1])
        y = paddle.reshape(y, [n, -1])

        x0 = paddle.floor(x).astype('int64')
        y0 = paddle.floor(y).astype('int64')
        x1 = x0 + 1
        y1 = y0 + 1

        x1_cast = x1.astype(grid.dtype)
        x0_cast = x0.astype(grid.dtype)
        y1_cast = y1.astype(grid.dtype)
        y0_cast = y0.astype(grid.dtype)
        wa = paddle.unsqueeze(((x1_cast - x) * (y1_cast - y)), 1)
        wb = paddle.unsqueeze(((x1_cast - x) * (y - y0_cast)), 1)
        wc = paddle.unsqueeze(((x - x0_cast) * (y1_cast - y)), 1)
        wd = paddle.unsqueeze(((x - x0_cast) * (y - y0_cast)), 1)

        # Apply default for grid_sample function zero padding
        im_padded = paddle.nn.functional.pad(im,
                                             pad=[1, 1, 1, 1],
                                             mode='constant',
                                             value=0)
        if im_padded.dtype != im.dtype:
            im_padded = paddle.cast(im_padded, im.dtype)
        padded_h = h + 2
        padded_w = w + 2
        # save points positions after padding
        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

        # Clip coordinates to padded image size
        tensor_zero = paddle.full(shape=[1], dtype='int64', fill_value=0.0)
        tensor_padded_w = paddle.full(
            shape=[1], dtype='int64', fill_value=padded_w - 1)
        tensor_padded_h = paddle.full(
            shape=[1], dtype='int64', fill_value=padded_h - 1)
        x0 = paddle.where(x0 < 0, tensor_zero, x0)
        x0 = paddle.where(x0 > padded_w - 1, tensor_padded_w, x0)
        x1 = paddle.where(x1 < 0, tensor_zero, x1)
        x1 = paddle.where(x1 > padded_w - 1, tensor_padded_w, x1)
        y0 = paddle.where(y0 < 0, tensor_zero, y0)
        y0 = paddle.where(y0 > padded_h - 1, tensor_padded_h, y0)
        y1 = paddle.where(y1 < 0, tensor_zero, y1)
        y1 = paddle.where(y1 > padded_h - 1, tensor_padded_h, y1)
        im_padded = paddle.reshape(im_padded, [n, c, -1])

        x0_y0 = paddle.expand(
            paddle.unsqueeze((x0 + y0 * padded_w), 1), [-1, c, -1])
        x0_y1 = paddle.expand(
            paddle.unsqueeze((x0 + y1 * padded_w), 1), [-1, c, -1])
        x1_y0 = paddle.expand(
            paddle.unsqueeze((x1 + y0 * padded_w), 1), [-1, c, -1])
        x1_y1 = paddle.expand(
            paddle.unsqueeze((x1 + y1 * padded_w), 1), [-1, c, -1])

        Ia = self.paddle_gather(im_padded, 2, x0_y0)
        Ib = self.paddle_gather(im_padded, 2, x0_y1)
        Ic = self.paddle_gather(im_padded, 2, x1_y0)
        Id = self.paddle_gather(im_padded, 2, x1_y1)

        return paddle.reshape((Ia * wa + Ib * wb + Ic * wc + Id * wd),
                              [n, c, gh, gw])

    def paddle_gather(self, x, dim, index):
        # index_shape = index.shape
        index_shape = paddle.shape(index)
        x_shape = paddle.shape(x)
        index_flatten = index.flatten()
        if dim < 0:
            dim = len(x.shape) + dim
        nd_index = []
        for k in range(len(x.shape)):
            if k == dim:
                nd_index.append(index_flatten)
            else:
                reshape_shape = [1] * len(x.shape)
                x_shape_k = x_shape[k]
                # x_shape_k = x.shape[k]
                reshape_shape[k] = x_shape_k
                x_arange = paddle.arange(x_shape_k, dtype=index.dtype)
                x_arange = x_arange.reshape(reshape_shape)
                dim_index = paddle.expand(x_arange, index_shape).flatten()
                nd_index.append(dim_index)
        ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
        paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
        return paddle_out

    def forward(self):
        input = self.input('X', 0)
        grid = self.input('Grid', 0)
        if self.mode != 'bilinear' or self.padding_mode != 'zeros':
            raise Exception(
                "grid_sample only is supported with mode should be 'bilinear' and padding_mode should be 'zeros'"
            )
        res = self.paddle_bilinear_grid_sample(
            input, grid, align_corners=self.align_corners)
        return {'Output': [res]}


register_custom_paddle_op('grid_sampler', GridSampler)
