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


class CollectFpnProposals(CustomPaddleOp):
    def __init__(self, node, **kw):
        super(CollectFpnProposals, self).__init__(node)
        self.post_nms_top_n = node.attr('post_nms_topN')

    def forward(self):
        multi_level_rois = self.input('MultiLevelRois')
        multi_level_scores = self.input('MultiLevelScores')
        multi_level_rois = paddle.concat(multi_level_rois, axis=0)
        multi_level_scores = paddle.concat(multi_level_scores, axis=0)
        proposal_num = paddle.shape(multi_level_scores)[0]
        post_nms_top_n_tensor = paddle.assign(
            np.array([self.post_nms_top_n]).astype('int32'))
        k_candidate = paddle.concat([proposal_num, post_nms_top_n_tensor])
        k = paddle.min(k_candidate)
        scores, index = paddle.topk(multi_level_scores, k=k, axis=0)
        rois = paddle.gather(multi_level_rois, index, axis=0)
        return {"FpnRois": [rois]}

@op_mapper('collect_fpn_proposals')
class Collectfpnproposals:
    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'collect_fpn_proposals',
            inputs=node.input('MultiLevelRois')+ node.input('MultiLevelScores'),
            outputs=node.output('FpnRois'),
            post_nms_top_n = node.attr('post_nms_topN'),
            domain = 'custom')
            
register_custom_paddle_op('collect_fpn_proposals', CollectFpnProposals)
