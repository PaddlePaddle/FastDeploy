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

import sys
import numpy as np
from paddle2onnx.legacy.constant import dtypes
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.legacy.op_mapper import mapper_helper
from onnx import TensorProto
import paddle

MAX_FLOAT32 = 3.402823466E+38


@op_mapper('yolo_box')
class YOLOBox():
    support_opset_verison_range = (9, 12)

    node_pred_box_x1_decode = None
    node_pred_box_y1_decode = None
    node_pred_box_x2_decode = None
    node_pred_box_y2_decode = None
    node_pred_box_x2_sub_w = None
    node_pred_box_y2_sub_h = None

    @classmethod
    def opset_9(cls, graph, node, **kw):
        model_name = node.output('Boxes', 0)
        input_shape = node.input_shape('X', 0)
        mapper_helper.is_static_shape(input_shape)
        image_size = node.input('ImgSize')
        input_height = input_shape[2]
        input_width = input_shape[3]
        class_num = node.attr('class_num')
        anchors = node.attr('anchors')
        num_anchors = int(len(anchors)) // 2
        scale_x_y = node.attr('scale_x_y')
        downsample_ratio = node.attr('downsample_ratio')
        input_size = input_height * downsample_ratio
        conf_thresh = node.attr('conf_thresh')
        conf_thresh_mat = [conf_thresh
                           ] * num_anchors * input_height * input_width

        cls.score_shape = [
            1, input_height * input_width * int(num_anchors), class_num
        ]

        im_outputs = []

        x_shape = [1, num_anchors, 5 + class_num, input_height, input_width]
        node_x_shape = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': x_shape})

        node_x_reshape = graph.make_node(
            'Reshape', inputs=[node.input('X')[0], node_x_shape])
        node_x_transpose = graph.make_node(
            'Transpose', inputs=[node_x_reshape], perm=[0, 1, 3, 4, 2])

        range_x = []
        range_y = []
        for i in range(0, input_width):
            range_x.append(i)
        for j in range(0, input_height):
            range_y.append(j)

        node_range_x = graph.make_node(
            'Constant',
            attrs={
                'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
                'value': range_x,
            })

        node_range_y = graph.make_node(
            'Constant',
            inputs=[],
            attrs={
                'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
                'value': range_y,
            })

        range_x_new_shape = [1, input_width]
        range_y_new_shape = [input_height, 1]

        node_range_x_new_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=range_x_new_shape)
        node_range_y_new_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=range_y_new_shape)

        node_range_x_reshape = graph.make_node(
            'Reshape', inputs=[node_range_x, node_range_x_new_shape])
        node_range_y_reshape = graph.make_node(
            'Reshape', inputs=[node_range_y, node_range_y_new_shape])

        node_grid_x = graph.make_node(
            "Tile", inputs=[node_range_x_reshape, node_range_y_new_shape])

        node_grid_y = graph.make_node(
            "Tile", inputs=[node_range_y_reshape, node_range_x_new_shape])

        node_box_x = model_name + "@box_x"
        node_box_y = model_name + "@box_y"
        node_box_w = model_name + "@box_w"
        node_box_h = model_name + "@box_h"
        node_conf = model_name + "@conf"
        node_prob = model_name + "@prob"
        output = [
            node_box_x, node_box_y, node_box_w, node_box_h, node_conf, node_prob
        ]
        node_split_input = mapper_helper.split_helper(
            graph, [node_x_transpose],
            output,
            -1, [1, 1, 1, 1, 1, class_num],
            dtype=node.input_dtype('X', 0))

        node_box_x_sigmoid = graph.make_node("Sigmoid", inputs=[node_box_x])

        node_box_y_sigmoid = graph.make_node("Sigmoid", inputs=[node_box_y])

        if scale_x_y is not None:
            bias_x_y = -0.5 * (scale_x_y - 1.0)
            scale_x_y_node = graph.make_node(
                'Constant',
                attrs={
                    'dtype':
                    dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
                    'value': scale_x_y
                })

            bias_x_y_node = graph.make_node(
                'Constant',
                attrs={
                    'dtype':
                    dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
                    'value': bias_x_y
                })
            node_box_x_sigmoid = graph.make_node(
                "Mul", inputs=[node_box_x_sigmoid, scale_x_y_node])
            node_box_x_sigmoid = graph.make_node(
                "Add", inputs=[node_box_x_sigmoid, bias_x_y_node])
            node_box_y_sigmoid = graph.make_node(
                "Mul", inputs=[node_box_y_sigmoid, scale_x_y_node])
            node_box_y_sigmoid = graph.make_node(
                "Add", inputs=[node_box_y_sigmoid, bias_x_y_node])
        node_box_x_squeeze = mapper_helper.squeeze_helper(
            graph, node_box_x_sigmoid, [4])

        node_box_y_squeeze = mapper_helper.squeeze_helper(
            graph, node_box_y_sigmoid, [4])

        node_box_x_add_grid = graph.make_node(
            "Add", inputs=[node_grid_x, node_box_x_squeeze])

        node_box_y_add_grid = graph.make_node(
            "Add", inputs=[node_grid_y, node_box_y_squeeze])

        node_input_h = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
            value=[input_height])

        node_input_w = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
            value=[input_width])

        node_box_x_encode = graph.make_node(
            'Div', inputs=[node_box_x_add_grid, node_input_w])

        node_box_y_encode = graph.make_node(
            'Div', inputs=[node_box_y_add_grid, node_input_h])

        node_anchor_tensor = graph.make_node(
            "Constant",
            inputs=[],
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
            value=anchors)

        anchor_shape = [int(num_anchors), 2]
        node_anchor_shape = graph.make_node(
            "Constant", inputs=[], dtype=dtypes.ONNX.INT64, value=anchor_shape)

        node_anchor_tensor_reshape = graph.make_node(
            "Reshape", inputs=[node_anchor_tensor, node_anchor_shape])

        node_input_size = graph.make_node(
            "Constant",
            inputs=[],
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
            value=[input_size])

        node_anchors_div_input_size = graph.make_node(
            "Div", inputs=[node_anchor_tensor_reshape, node_input_size])

        node_anchor_w = model_name + "@anchor_w"
        node_anchor_h = model_name + "@anchor_h"

        node_anchor_split = mapper_helper.split_helper(
            graph,
            inputs=node_anchors_div_input_size,
            axis=1,
            split=[1, 1],
            outputs=[node_anchor_w, node_anchor_h],
            dtype=node.input_dtype('X', 0))

        new_anchor_shape = [1, int(num_anchors), 1, 1]
        node_new_anchor_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=new_anchor_shape)

        node_anchor_w_reshape = graph.make_node(
            'Reshape', inputs=[node_anchor_w, node_new_anchor_shape])

        node_anchor_h_reshape = graph.make_node(
            'Reshape', inputs=[node_anchor_h, node_new_anchor_shape])

        node_box_w_squeeze = mapper_helper.squeeze_helper(graph, node_box_w,
                                                          [4])
        node_box_h_squeeze = mapper_helper.squeeze_helper(graph, node_box_h,
                                                          [4])

        node_box_w_exp = graph.make_node("Exp", inputs=[node_box_w_squeeze])
        node_box_h_exp = graph.make_node("Exp", inputs=[node_box_h_squeeze])

        node_box_w_encode = graph.make_node(
            'Mul', inputs=[node_box_w_exp, node_anchor_w_reshape])

        node_box_h_encode = graph.make_node(
            'Mul', inputs=[node_box_h_exp, node_anchor_h_reshape])

        node_conf_sigmoid = graph.make_node('Sigmoid', inputs=[node_conf])

        node_conf_thresh = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
            value=conf_thresh_mat)

        conf_shape = [1, int(num_anchors), input_height, input_width, 1]
        node_conf_shape = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.INT64, value=conf_shape)

        node_conf_thresh_reshape = graph.make_node(
            'Reshape', inputs=[node_conf_thresh, node_conf_shape])

        node_conf_sub = graph.make_node(
            'Sub', inputs=[node_conf_sigmoid, node_conf_thresh_reshape])

        node_conf_clip = mapper_helper.clip_helper(graph, node, node_conf_sub,
                                                   float(MAX_FLOAT32), 0.0)

        node_zeros = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
            value=[0])

        node_conf_clip_bool = graph.make_node(
            'Greater', inputs=[node_conf_clip, node_zeros])

        node_conf_clip_cast = graph.make_node(
            'Cast',
            inputs=[node_conf_clip_bool],
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])

        node_conf_set_zero = graph.make_node(
            'Mul', inputs=[node_conf_sigmoid, node_conf_clip_cast])

        node_prob_sigmoid = graph.make_node('Sigmoid', inputs=[node_prob])

        new_shape = [1, int(num_anchors), input_height, input_width, 1]
        node_new_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            dims=[len(new_shape)],
            value=new_shape)

        node_conf_new_shape = graph.make_node(
            'Reshape', inputs=[node_conf_set_zero, node_new_shape])

        cls.node_score = graph.make_node(
            'Mul', inputs=[node_prob_sigmoid, node_conf_new_shape])

        node_conf_bool = graph.make_node(
            'Greater', inputs=[node_conf_new_shape, node_zeros])

        node_box_x_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_x_encode, node_new_shape])

        node_box_y_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_y_encode, node_new_shape])

        node_box_w_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_w_encode, node_new_shape])

        node_box_h_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_h_encode, node_new_shape])

        node_pred_box = graph.make_node(
            'Concat',
            inputs=[node_box_x_new_shape, node_box_y_new_shape, \
                   node_box_w_new_shape, node_box_h_new_shape],
            axis=4)

        node_conf_cast = graph.make_node(
            'Cast',
            inputs=[node_conf_bool],
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])

        node_pred_box_mul_conf = graph.make_node(
            'Mul', inputs=[node_pred_box, node_conf_cast])

        box_shape = [1, int(num_anchors) * input_height * input_width, 4]
        node_box_shape = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.INT64, value=box_shape)

        node_pred_box_new_shape = graph.make_node(
            'Reshape', inputs=[node_pred_box_mul_conf, node_box_shape])

        node_pred_box_x = model_name + "@_pred_box_x"
        node_pred_box_y = model_name + "@_pred_box_y"
        node_pred_box_w = model_name + "@_pred_box_w"
        node_pred_box_h = model_name + "@_pred_box_h"
        if node.input_dtype('X', 0) == paddle.float64:
            node_pred_box_new_shape = graph.make_node(
                'Cast', inputs=[node_pred_box_new_shape], to=TensorProto.FLOAT)
        node_pred_box_split = mapper_helper.split_helper(
            graph,
            inputs=node_pred_box_new_shape,
            axis=2,
            split=[1, 1, 1, 1],
            outputs=[
                node_pred_box_x, node_pred_box_y, node_pred_box_w,
                node_pred_box_h
            ],
            dtype=node.input_dtype('X', 0))

        if node.input_dtype('X', 0) == paddle.float64:
            node_pred_box_x = graph.make_node(
                'Cast',
                inputs=[node_pred_box_x],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])
            node_pred_box_y = graph.make_node(
                'Cast',
                inputs=[node_pred_box_y],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])
            node_pred_box_w = graph.make_node(
                'Cast',
                inputs=[node_pred_box_w],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])
            node_pred_box_h = graph.make_node(
                'Cast',
                inputs=[node_pred_box_h],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])
        node_number_two = graph.make_node(
            "Constant",
            inputs=[],
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
            value=[2])

        node_half_w = graph.make_node(
            "Div", inputs=[node_pred_box_w, node_number_two])

        node_half_h = graph.make_node(
            "Div", inputs=[node_pred_box_h, node_number_two])

        node_pred_box_x1 = graph.make_node(
            'Sub', inputs=[node_pred_box_x, node_half_w])

        node_pred_box_y1 = graph.make_node(
            'Sub', inputs=[node_pred_box_y, node_half_h])

        node_pred_box_x2 = graph.make_node(
            'Add', inputs=[node_pred_box_x, node_half_w])

        node_pred_box_y2 = graph.make_node(
            'Add', inputs=[node_pred_box_y, node_half_h])

        node_sqeeze_image_size = mapper_helper.squeeze_helper(
            graph, image_size[0], [0])

        node_img_height = model_name + "@img_height"
        node_img_width = model_name + "@img_width"
        node_image_size_split = mapper_helper.split_helper(
            graph, [node_sqeeze_image_size], [node_img_height, node_img_width],
            -1, [1, 1],
            dtype=node.input_dtype('X', 0))

        node_img_width_cast = graph.make_node(
            'Cast',
            inputs=[node_img_width],
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])

        node_img_height_cast = graph.make_node(
            'Cast',
            inputs=[node_img_height],
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)])

        cls.node_pred_box_x1_decode = graph.make_node(
            'Mul',
            inputs=[node_pred_box_x1, node_img_width_cast])  #boxes[box_idx]

        cls.node_pred_box_y1_decode = graph.make_node(
            'Mul', inputs=[node_pred_box_y1,
                           node_img_height_cast])  #boxes[box_idx + 1]

        cls.node_pred_box_x2_decode = graph.make_node(
            'Mul',
            inputs=[node_pred_box_x2, node_img_width_cast])  #boxes[box_idx + 2]

        cls.node_pred_box_y2_decode = graph.make_node(
            'Mul', inputs=[node_pred_box_y2,
                           node_img_height_cast])  #boxes[box_idx + 3]

        if node.attr('clip_bbox'):
            node_number_one = graph.make_node(
                'Constant',
                inputs=[],
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
                value=[1])

            node_new_img_height = graph.make_node(
                'Sub', inputs=[node_img_height_cast, node_number_one])

            node_new_img_width = graph.make_node(
                'Sub', inputs=[node_img_width_cast, node_number_one])

            cls.node_pred_box_x2_sub_w = graph.make_node(
                'Sub',
                inputs=[cls.node_pred_box_x2_decode, node_new_img_width])

            cls.node_pred_box_y2_sub_h = graph.make_node(
                'Sub',
                inputs=[cls.node_pred_box_y2_decode, node_new_img_height])

            node_pred_box_x1_clip = mapper_helper.clip_helper(
                graph, node, cls.node_pred_box_x1_decode,
                float(MAX_FLOAT32), 0.0)
            node_pred_box_y1_clip = mapper_helper.clip_helper(
                graph, node, cls.node_pred_box_y1_decode,
                float(MAX_FLOAT32), 0.0)
            node_pred_box_x2_clip = mapper_helper.clip_helper(
                graph, node, cls.node_pred_box_x2_sub_w,
                float(MAX_FLOAT32), 0.0)
            node_pred_box_y2_clip = mapper_helper.clip_helper(
                graph, node, cls.node_pred_box_y2_sub_h,
                float(MAX_FLOAT32), 0.0)
            node_pred_box_x2_res = graph.make_node(
                'Sub',
                inputs=[cls.node_pred_box_x2_decode, node_pred_box_x2_clip])

            node_pred_box_y2_res = graph.make_node(
                'Sub',
                inputs=[cls.node_pred_box_y2_decode, node_pred_box_y2_clip])

            node_pred_box_result = graph.make_node(
                'Concat',
                inputs=[
                    node_pred_box_x1_clip, node_pred_box_y1_clip,
                    node_pred_box_x2_res, node_pred_box_y2_res
                ],
                outputs=node.output('Boxes'),
                axis=-1)
        else:
            node_pred_box_result = graph.make_node(
                'Concat',
                inputs=[
                    cls.node_pred_box_x1_decode, cls.node_pred_box_y1_decode,
                    cls.node_pred_box_x2_decode, cls.node_pred_box_y2_decode
                ],
                outputs=node.output('Boxes'),
                axis=-1)
        node_score_shape = graph.make_node(
            "Constant",
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=cls.score_shape)

        node_score_new_shape = graph.make_node(
            'Reshape',
            inputs=[cls.node_score, node_score_shape],
            outputs=node.output('Scores'))
