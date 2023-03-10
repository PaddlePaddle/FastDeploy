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


def expand_aspect_rations(input_aspect_ratior, flip):
    expsilon = 1e-6
    output_ratios = [1.0]
    for input_ratio in input_aspect_ratior:
        already_exis = False
        for output_ratio in output_ratios:
            if abs(input_ratio - output_ratio) < expsilon:
                already_exis = True
                break
        if already_exis == False:
            output_ratios.append(input_ratio)
            if flip:
                output_ratios.append(1.0 / input_ratio)
    return output_ratios


@op_mapper('prior_box')
class PriorBox():
    """
    In this function, use the attribute to get the prior box, because we do not use
    the image data and feature map, wo could the python code to create the varaible,
    and to create the onnx tensor as output.
    """
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        flip = bool(node.attr('flip'))
        clip = bool(node.attr('clip'))
        min_max_aspect_ratios_order = bool(
            node.attr('min_max_aspect_ratios_order'))
        min_sizes = [float(size) for size in node.attr('min_sizes')]
        max_sizes = [float(size) for size in node.attr('max_sizes')]
        if isinstance(node.attr('aspect_ratios'), list):
            aspect_ratios = [
                float(ratio) for ratio in node.attr('aspect_ratios')
            ]
        else:
            aspect_ratios = [float(node.attr('aspect_ratios'))]
        variances = [float(var) for var in node.attr('variances')]
        # set min_max_aspect_ratios_order = false
        output_ratios = expand_aspect_rations(aspect_ratios, flip)

        step_w = float(node.attr('step_w'))
        step_h = float(node.attr('step_h'))
        offset = float(node.attr('offset'))

        input_shape = node.input_shape('Input', 0)
        image_shape = node.input_shape('Image', 0)

        img_width = image_shape[3]
        img_height = image_shape[2]
        feature_width = input_shape[3]
        feature_height = input_shape[2]
        assert img_width > 0 and img_height > 0, require_fixed_shape(
            cls.__name__)

        step_width = 1.0
        step_height = 1.0

        if step_w == 0.0 or step_h == 0.0:
            step_w = float(img_width / feature_width)
            step_h = float(img_height / feature_height)

        num_priors = len(output_ratios) * len(min_sizes)
        if len(max_sizes) > 0:
            num_priors += len(max_sizes)
        out_dim = (feature_height, feature_width, num_priors, 4)
        out_boxes = np.zeros(out_dim).astype('float32')
        out_var = np.zeros(out_dim).astype('float32')

        idx = 0
        for h in range(feature_height):
            for w in range(feature_width):
                c_x = (w + offset) * step_w
                c_y = (h + offset) * step_h
                idx = 0
                for s in range(len(min_sizes)):
                    min_size = min_sizes[s]
                    if not min_max_aspect_ratios_order:
                        # rest of priors
                        for r in range(len(output_ratios)):
                            ar = output_ratios[r]
                            c_w = min_size * math.sqrt(ar) / 2
                            c_h = (min_size / math.sqrt(ar)) / 2
                            out_boxes[h, w, idx, :] = [(c_x - c_w) / img_width,
                                                       (c_y - c_h) / img_height,
                                                       (c_x + c_w) / img_width,
                                                       (c_y + c_h) / img_height]
                            idx += 1

                        if len(max_sizes) > 0:
                            max_size = max_sizes[s]
                            # second prior: aspect_ratio = 1,
                            c_w = c_h = math.sqrt(min_size * max_size) / 2
                            out_boxes[h, w, idx, :] = [(c_x - c_w) / img_width,
                                                       (c_y - c_h) / img_height,
                                                       (c_x + c_w) / img_width,
                                                       (c_y + c_h) / img_height]
                            idx += 1
                    else:
                        c_w = c_h = min_size / 2.
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) / img_height,
                            (c_x + c_w) / img_width, (c_y + c_h) / img_height
                        ]
                        idx += 1
                        if len(max_sizes) > 0:
                            max_size = max_sizes[s]
                            # second prior: aspect_ratio = 1,
                            c_w = c_h = math.sqrt(min_size * max_size) / 2
                            out_boxes[h, w, idx, :] = [(c_x - c_w) / img_width,
                                                       (c_y - c_h) / img_height,
                                                       (c_x + c_w) / img_width,
                                                       (c_y + c_h) / img_height]
                            idx += 1

                        # rest of priors
                        for r in range(len(output_ratios)):
                            ar = output_ratios[r]
                            if abs(ar - 1.) < 1e-6:
                                continue
                            c_w = min_size * math.sqrt(ar) / 2
                            c_h = (min_size / math.sqrt(ar)) / 2
                            out_boxes[h, w, idx, :] = [(c_x - c_w) / img_width,
                                                       (c_y - c_h) / img_height,
                                                       (c_x + c_w) / img_width,
                                                       (c_y + c_h) / img_height]
                            idx += 1

        if clip:
            out_boxes = np.clip(out_boxes, 0.0, 1.0)
        # set the variance.
        out_var = np.tile(variances,
                          (feature_height, feature_width, num_priors, 1))

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
