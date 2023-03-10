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

import math
import numpy as np
from paddle2onnx.legacy.constant import dtypes
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.utils import require_fixed_shape


@op_mapper('density_prior_box')
class DensityPriorBox():
    """
    In this function, use the attribute to get the prior box, because we do not use
    the image data and feature map, wo could the python code to create the varaible,
    and to create the onnx tensor as output.
    """
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        clip = bool(node.attr('clip'))
        densities = node.attr('densities')
        fixed_ratios = node.attr('fixed_ratios')
        fixed_sizes = node.attr('fixed_sizes')
        flatten_to_2d = bool(node.attr('flatten_to_2d'))
        offset = node.attr('offset')
        step_h = node.attr('step_h')
        step_w = node.attr('step_w')
        variances = node.attr('variances')

        input_shape = node.input_shape('Input', 0)
        image_shape = node.input_shape('Image', 0)

        img_width = image_shape[3]
        img_height = image_shape[2]
        feature_width = input_shape[3]
        feature_height = input_shape[2]

        assert img_width > 0 and img_height > 0, require_fixed_shape(
            cls.__name__)

        if step_w == 0.0 or step_h == 0.0:
            step_w = float(img_width / feature_width)
            step_h = float(img_height / feature_height)

        num_priors = 0
        if len(fixed_sizes) > 0 and len(densities) > 0:
            for density in densities:
                if len(fixed_ratios) > 0:
                    num_priors += len(fixed_ratios) * (pow(density, 2))

        out_dim = (feature_height, feature_width, num_priors, 4)
        out_boxes = np.zeros(out_dim).astype('float32')
        out_var = np.zeros(out_dim).astype('float32')
        step_average = int((step_w + step_h) * 0.5)

        for h in range(feature_height):
            for w in range(feature_width):
                c_x = (w + offset) * step_w
                c_y = (h + offset) * step_h
                idx = 0

                for density, fixed_size in zip(densities, fixed_sizes):
                    if (len(fixed_ratios) > 0):
                        for ar in fixed_ratios:
                            shift = int(step_average / density)
                            box_width_ratio = fixed_size * math.sqrt(ar)
                            box_height_ratio = fixed_size / math.sqrt(ar)
                            for di in range(density):
                                for dj in range(density):
                                    c_x_temp = c_x - step_average / 2.0 + shift / 2.0 + dj * shift
                                    c_y_temp = c_y - step_average / 2.0 + shift / 2.0 + di * shift
                                    out_boxes[h, w, idx, :] = [
                                        max((c_x_temp - box_width_ratio / 2.0) /
                                            img_width, 0),
                                        max((c_y_temp - box_height_ratio / 2.0)
                                            / img_height, 0),
                                        min((c_x_temp + box_width_ratio / 2.0) /
                                            img_width, 1),
                                        min((c_y_temp + box_height_ratio / 2.0)
                                            / img_height, 1)
                                    ]
                                    idx += 1

        if clip:
            out_boxes = np.clip(out_boxes, 0.0, 1.0)
        # set the variance.
        out_var = np.tile(variances,
                          (feature_height, feature_width, num_priors, 1))

        if flatten_to_2d:
            out_boxes = out_boxes.reshape((-1, 4))
            out_var = out_var.reshape((-1, 4))

        #make node that

        node_boxes = graph.make_node(
            'Constant',
            inputs=[],
            outputs=node.output('Boxes'),
            dtype=dtypes.ONNX.FLOAT,
            dims=out_boxes.shape,
            value=out_boxes.flatten().tolist())

        node_vars = graph.make_node(
            'Constant',
            inputs=[],
            outputs=node.output('Variances'),
            dtype=dtypes.ONNX.FLOAT,
            dims=out_var.shape,
            value=out_var.flatten().tolist())
