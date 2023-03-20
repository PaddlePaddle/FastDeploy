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


class DistributeFpnProposals(CustomPaddleOp):
    def __init__(self, node, **kw):
        super(DistributeFpnProposals, self).__init__(node)
        self.max_level = node.attr('max_level')
        self.min_level = node.attr('min_level')
        self.refer_level = node.attr('refer_level')
        self.refer_scale = node.attr('refer_scale')
        self.pixel_offset = node.attr('pixel_offset')

    def bbox_area(self, boxes):
        offset = 1 if self.pixel_offset else 0
        xmin, ymin, xmax, ymax = paddle.tensor.split(
            boxes, axis=1, num_or_sections=4)
        width = xmax - xmin + offset
        height = ymax - ymin + offset
        areas = width * height
        return areas

    def forward(self):
        fpn_rois = self.input('FpnRois', 0)
        areas = self.bbox_area(fpn_rois)
        scale = paddle.sqrt(areas)
        num_level = self.max_level - self.min_level + 1
        target_level = paddle.log(scale / self.refer_scale + 1e-06) / np.log(2)
        target_level = paddle.floor(self.refer_level + target_level)
        target_level = paddle.clip(
            target_level, min=self.min_level, max=self.max_level)

        rois = list()
        rois_idx_order = list()
        rois_num_per_level = list()

        for level in range(self.min_level, self.max_level + 1):
            level_tensor = paddle.full_like(target_level, fill_value=level)
            res = paddle.equal(target_level, level_tensor)
            res = paddle.squeeze(res, axis=1)
            res = paddle.cast(res, dtype='int32')
            index = paddle.nonzero(res)
            roi = paddle.gather(fpn_rois, index, axis=0)
            rois.append(roi)
            rois_idx_order.append(index)
            rois_num_per_level.append(paddle.shape(roi)[0])
        rois_idx_order = paddle.concat(rois_idx_order, axis=0)
        size = paddle.shape(rois_idx_order)[0]
        _, rois_idx_restore = paddle.topk(
            rois_idx_order, axis=0, sorted=True, largest=False, k=size)

        rois_idx_restore = paddle.cast(rois_idx_restore, dtype='int32')
        if len(self.input('RoisNum')) > 0:
            # trick: to keep rois num
            rois_num_per_level[0] += self.input('RoisNum', 0) * 0
            return {
                'MultiFpnRois': rois,
                'RestoreIndex': [rois_idx_restore],
                'MultiLevelRoIsNum': rois_num_per_level
            }
        else:
            return {'MultiFpnRois': rois, 'RestoreIndex': [rois_idx_restore]}


@op_mapper('distribute_fpn_proposals')
class Distributefpnproposals:
    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'distribute_fpn_proposals',
            inputs=node.input('FpnRois'),
            outputs=node.output('MultiFpnRois') + node.output('RestoreIndex'),
            max_level=node.attr('max_level'),
            min_level=node.attr('min_level'),
            refer_level=node.attr('refer_level'),
            refer_scale=node.attr('refer_scale'),
            domain='custom')


register_custom_paddle_op('distribute_fpn_proposals', DistributeFpnProposals)
