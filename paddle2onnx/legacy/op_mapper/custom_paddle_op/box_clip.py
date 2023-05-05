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

class BoxClip(CustomPaddleOp):
    def __init__(self, node, **kw):
        super(BoxClip, self).__init__(node)

    def forward(self):
        input = self.input('Input', 0)
        im_info = self.input('ImInfo', 0)
        im_info = paddle.reshape(im_info, shape=[3])
        h, w, s = paddle.tensor.split(im_info, axis=0, num_or_sections=3)
        tensor_one = paddle.full(shape=[1], dtype='float32', fill_value=1.0)
        tensor_zero = paddle.full(shape=[1], dtype='float32', fill_value=0.0)
        h = paddle.subtract(h, tensor_one)
        w = paddle.subtract(w, tensor_one)
        xmin, ymin, xmax, ymax = paddle.tensor.split(
            input, axis=-1, num_or_sections=4)
        xmin = paddle.maximum(paddle.minimum(xmin, w), tensor_zero)
        ymin = paddle.maximum(paddle.minimum(ymin, h), tensor_zero)
        xmax = paddle.maximum(paddle.minimum(xmax, w), tensor_zero)
        ymax = paddle.maximum(paddle.minimum(ymax, h), tensor_zero)
        cliped_box = paddle.concat([xmin, ymin, xmax, ymax], axis=-1)

        return {'Output': [cliped_box]}

@op_mapper('box_clip')
class Boxclip:
    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'box_clip',
            inputs=node.input('Input')+node.input('ImInfo'),
            outputs=node.output('Output'),
            domain = 'custom')
register_custom_paddle_op('box_clip', BoxClip)
