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
from paddle2onnx.legacy.constant import dtypes
from paddle2onnx.utils import logging
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper


@op_mapper('im2sequence')
class Im2Sequence():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        n, c, h, w = node.input_shape('X', 0)
        assert h > 0 and w > 0, "Only supported fixed input shape for im2sequence operator."
        stride_h, stride_w = node.attr('strides')
        paddings = node.attr('paddings')
        assert node.attr(
            'out_stride'
        ) != 1, "Only out_stride==1 is supported for im2sequence operator."
        h = h + paddings[0] + paddings[1]
        w = w + paddings[1] + paddings[2]
        kernel_h, kernel_w = node.attr('kernels')
        out_h = 1 + (h - kernel_h + stride_h - 1) // stride_h
        out_w = 1 + (w - kernel_w + stride_w - 1) // stride_w
        h_steps = list()
        for i in range(out_h):
            h_steps.append([i * stride_h, i * stride_h + kernel_h])
        w_steps = list()
        for i in range(out_w):
            w_steps.append([i * stride_w, i * stride_w + kernel_w])

        slice_node_blocks = list()
        for i in range(out_h):
            for j in range(out_w):
                starts_node = graph.make_node(
                    'Constant',
                    dtype=dtypes.ONNX.INT64,
                    dims=[4],
                    value=[0, 0, h_steps[i][0], w_steps[j][0]])
                ends_node = graph.make_node(
                    'Constant',
                    dtype=dtypes.ONNX.INT64,
                    dims=[4],
                    value=[999999, 999999, h_steps[i][1], w_steps[j][1]])
                nodes.extend([starts_node, ends_node])

                slice_block_node = graph.make_node(
                    'Slice',
                    inputs=[node.input('X', 0), starts_node, ends_node])
                flatten_block_node = graph.make_node(
                    "Flatten", inputs=[slice_block_node], axis=0)
                nodes.extend([slice_block_node, flatten_block_node])
        concat_block_node = graph.make_node(
            "Concat",
            inputs=slice_node_blocks,
            outputs=node.output('Out'),
            axis=0)
        logging.info("==========Importance Notice===========")
        logging.info(
            "Since im2sequence operator is used in your paddlepaddle model, the translated onnx model only support input data with batch_size=1."
        )
        logging.info("======================================")
