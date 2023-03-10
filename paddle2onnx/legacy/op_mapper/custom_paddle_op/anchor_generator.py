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

class AnchorGenerator(CustomPaddleOp):
    def __init__(self, node, **kw):
        super(AnchorGenerator, self).__init__(node)
        #self.x_shape = node.input_shape('Input', 0)
        self.anchor_sizes = node.attr('anchor_sizes')
        self.aspect_ratios = node.attr('aspect_ratios')
        self.offset = node.attr('offset')
        self.strides = node.attr('stride')
        self.variances = node.attr('variances')
        self.shapes = self.compute_shapes()

    def compute_shapes(self):
        shapes = list()
        for r in range(len(self.aspect_ratios)):
            ar = self.aspect_ratios[r]
            for s in range(len(self.anchor_sizes)):
                anchor_size = self.anchor_sizes[s]
                area = self.strides[0] * self.strides[1]
                area_ratios = area / ar
                base_w = np.floor(np.sqrt(area_ratios) + 0.5)
                base_h = np.floor(base_w * ar + 0.5)
                scale_w = anchor_size / self.strides[0]
                scale_h = anchor_size / self.strides[1]
                w = scale_w * base_w
                h = scale_h * base_h
                shapes.append([
                    -0.5 * (w - 1), -0.5 * (h - 1), 0.5 * (w - 1), 0.5 * (h - 1)
                ])
        return shapes

    def forward(self):
        input_feature = self.input('Input', 0)
        input_shape = paddle.shape(input_feature)
        n, c, h, w = paddle.tensor.split(input_shape, num_or_sections=4)
        x_ctr = paddle.arange(start=0, end=w, step=1, dtype=input_feature.dtype)
        y_ctr = paddle.arange(start=0, end=h, step=1, dtype=input_feature.dtype)
        x_ctr = x_ctr * self.strides[0] + self.offset * (self.strides[0] - 1)
        y_ctr = y_ctr * self.strides[1] + self.offset * (self.strides[1] - 1)
        tensor_one = paddle.ones(shape=[1], dtype='int64')
        tensor_len_shape = paddle.full(
            shape=[1], fill_value=len(self.shapes), dtype='int64')
        x_ctr = paddle.reshape(x_ctr, shape=(1, -1))
        y_ctr = paddle.reshape(y_ctr, shape=(1, -1))
        x_ctr = paddle.tile(x_ctr, repeat_times=(h, tensor_one))
        y_ctr = paddle.tile(y_ctr, repeat_times=(w, tensor_one))
        y_ctr = paddle.transpose(y_ctr, perm=[1, 0])
        centers = paddle.stack([x_ctr, y_ctr], axis=-1)
        centers = paddle.tensor.unsqueeze(centers, axis=[2])
        centers = paddle.tile(centers, repeat_times=(1, 1, len(self.shapes), 2))
        shape_tensor = paddle.assign(np.array(self.shapes).astype('float32'))
        anchors = centers + shape_tensor
        variance_tensor = paddle.assign(
            np.asarray(self.variances).astype('float32'))
        vars = paddle.reshape(variance_tensor, shape=[1, 1, 1, -1])
        vars = paddle.tile(
            vars, repeat_times=(h, w, tensor_len_shape, tensor_one))
        return {'Anchors': [anchors], 'Variances': [vars]}

@op_mapper('anchor_generator')
class Anchors_generator:
    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'anchor_generator',
            inputs=node.input('Input'),
            outputs=node.output('Anchors') + node.output('Variances'),
            anchor_sizes = node.attr('anchor_sizes'),
            aspect_ratios = node.attr('aspect_ratios'),
            offset = node.attr('offset'),
            strides = node.attr('stride'),
            variances = node.attr('variances'),
            domain = 'custom')

register_custom_paddle_op('anchor_generator', AnchorGenerator)
