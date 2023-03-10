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
import math
from paddle.fluid import layers
from paddle2onnx.legacy.op_mapper import CustomPaddleOp, register_custom_paddle_op
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.legacy.op_mapper import mapper_helper

BBOX_CLIP_DEFAULT = math.log(1000.0 / 16.0)


class GenerateProposals(CustomPaddleOp):
    def __init__(self, node, **kw):
        paddle.enable_static()
        super(GenerateProposals, self).__init__(node)
        self.eta = node.attr('eta')
        self.min_size = node.attr('min_size')
        self.nms_thresh = node.attr('nms_thresh')
        self.post_nms_topN = node.attr('post_nms_topN')
        self.pre_nms_topN = node.attr('pre_nms_topN')
        self.type = node.type
        if self.type == 'generate_proposals_v2':
            self.pixel_offset = node.attr('pixel_offset')
        else:
            self.pixel_offset = True

    def filter_boxes(self, boxes, im_w, im_h, im_s, min_size):
        min_size = max(min_size, 1.0)
        xmin, ymin, xmax, ymax = paddle.tensor.split(
            boxes, axis=1, num_or_sections=4)
        x_ctr = (xmax + xmin) / 2 + 0.5
        y_ctr = (ymax + ymin) / 2 + 0.5
        ws = (xmax - xmin) / im_s + 1
        hs = (ymax - ymin) / im_s + 1

        min_size = np.asarray([min_size], dtype='float32')
        min_size = paddle.assign(min_size)
        valid_flag_ws = paddle.greater_equal(ws, min_size)
        valid_flag_hs = paddle.greater_equal(hs, min_size)
        valid_flag_x = paddle.less_equal(x_ctr, im_w)
        valid_flag_y = paddle.less_equal(y_ctr, im_h)
        valid_flag = paddle.logical_and(valid_flag_ws, valid_flag_hs)
        valid_flag = paddle.logical_and(valid_flag, valid_flag_x)
        valid_flag = paddle.logical_and(valid_flag, valid_flag_y)
        valid_flag = paddle.squeeze(valid_flag, axis=1)
        valid_inds = paddle.nonzero(valid_flag)

        return valid_inds

    def filter_boxes_v2(self, boxes, im_w, im_h, min_size, pixel_offset=True):
        min_size = max(min_size, 1.0)
        xmin, ymin, xmax, ymax = paddle.tensor.split(
            boxes, axis=1, num_or_sections=4)

        offset = 1 if pixel_offset else 0
        ws = (xmax - xmin) + offset
        hs = (ymax - ymin) + offset

        min_size = np.asarray([min_size], dtype='float32')
        min_size = paddle.assign(min_size)
        valid_flag_ws = paddle.greater_equal(ws, min_size)
        valid_flag_hs = paddle.greater_equal(hs, min_size)
        valid_flag = paddle.logical_and(valid_flag_ws, valid_flag_hs)
        if pixel_offset:
            x_ctr = xmin + ws / 2
            y_ctr = ymin + hs / 2
            valid_flag_x = paddle.less_equal(x_ctr, im_w)
            valid_flag_y = paddle.less_equal(y_ctr, im_h)
            valid_flag = paddle.logical_and(valid_flag, valid_flag_x)
            valid_flag = paddle.logical_and(valid_flag, valid_flag_y)

        valid_flag = paddle.squeeze(valid_flag, axis=1)
        valid_inds = paddle.nonzero(valid_flag)
        return valid_inds

    def clip_tiled_boxes(self, im_w, im_h, input_boxes, pixel_offset=True):
        offset = 1 if pixel_offset else 0
        xmin, ymin, xmax, ymax = paddle.tensor.split(
            input_boxes, axis=1, num_or_sections=4)
        xmin = paddle.clip(xmin, max=im_w - offset, min=0)
        ymin = paddle.clip(ymin, max=im_h - offset, min=0)
        xmax = paddle.clip(xmax, max=im_w - offset, min=0)
        ymax = paddle.clip(ymax, max=im_h - offset, min=0)
        input_boxes = paddle.concat([xmin, ymin, xmax, ymax], axis=1)
        return input_boxes

    def box_encode(self, anchors, bbox_deltas, variances, pixel_offset=True):
        offset = 1 if pixel_offset else 0
        anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax = paddle.tensor.split(
            anchors, axis=1, num_or_sections=4)
        anchor_width = anchor_xmax - anchor_xmin + offset
        anchor_height = anchor_ymax - anchor_ymin + offset
        anchor_center_x = anchor_xmin + 0.5 * anchor_width
        anchor_center_y = anchor_ymin + 0.5 * anchor_height
        var_center_x, var_center_y, var_width, var_height = paddle.tensor.split(
            variances, axis=1, num_or_sections=4)
        delta_center_x, delta_center_y, delta_width, delta_height = paddle.tensor.split(
            bbox_deltas, axis=1, num_or_sections=4)

        bbox_center_x = var_center_x * delta_center_x * anchor_width + anchor_center_x
        bbox_center_y = var_center_y * delta_center_y * anchor_height + anchor_center_y
        bbox_width = paddle.exp(
            paddle.clip(
                var_width * delta_width, max=BBOX_CLIP_DEFAULT)) * anchor_width
        bbox_height = paddle.exp(
            paddle.clip(
                var_height * delta_height,
                max=BBOX_CLIP_DEFAULT)) * anchor_height

        proposal_xmin = bbox_center_x - bbox_width / 2
        proposal_ymin = bbox_center_y - bbox_height / 2
        proposal_xmax = bbox_center_x + bbox_width / 2 - offset
        proposal_ymax = bbox_center_y + bbox_height / 2 - offset
        proposal = paddle.concat(
            [proposal_xmin, proposal_ymin, proposal_xmax, proposal_ymax],
            axis=1)
        return proposal

    def proposal_for_single_sample(self, anchors, bbox_deltas, im_info, scores,
                                   variances):
        proposal_num = paddle.shape(scores)[0]
        pre_nms_top_n_tensor = paddle.assign(
            np.asarray(
                [self.pre_nms_topN], dtype='int32'))
        k_candidate = paddle.concat([proposal_num, pre_nms_top_n_tensor])
        k = paddle.min(k_candidate)
        scores, index = paddle.topk(scores, k=k, axis=0)
        bbox_deltas = paddle.gather(bbox_deltas, index, axis=0)
        anchors = paddle.gather(anchors, index, axis=0)
        variances = paddle.gather(variances, index, axis=0)

        proposal = self.box_encode(anchors, bbox_deltas, variances,
                                   self.pixel_offset)
        if self.type == "generate_proposals_v2":
            im_h, im_w = paddle.tensor.split(im_info, axis=1, num_or_sections=2)
        else:
            im_h, im_w, im_s = paddle.tensor.split(
                im_info, axis=1, num_or_sections=3)
        proposal = self.clip_tiled_boxes(im_w, im_h, proposal,
                                         self.pixel_offset)

        if self.type == "generate_proposals_v2":
            keep = self.filter_boxes_v2(proposal, im_w, im_h, self.min_size,
                                        self.pixel_offset)
        else:
            keep = self.filter_boxes(proposal, im_w, im_h, im_s, self.min_size)

        tail_proposal = paddle.zeros(shape=[1, 4], dtype=proposal.dtype)
        proposal_num = paddle.shape(proposal)[0]
        tail_keep = paddle.reshape(proposal_num, shape=[1, 1])
        tail_keep = paddle.cast(tail_keep, dtype=keep.dtype)
        tail_scores = paddle.zeros(shape=[1, 1], dtype=scores.dtype)
        # proposal = paddle.concat([proposal, tail_proposal])
        # keep = paddle.concat([keep, tail_keep])
        # scores = paddle.concat([scores, tail_scores])

        bbox_sel = paddle.gather(proposal, keep, axis=0)
        scores_sel = paddle.gather(scores, keep, axis=0)
        proposal = paddle.unsqueeze(bbox_sel, axis=0)
        scores = paddle.transpose(scores_sel, perm=[1, 0])
        scores = paddle.unsqueeze(scores, axis=0)
        out = layers.multiclass_nms(
            proposal,
            scores,
            background_label=-1,
            nms_top_k=self.pre_nms_topN,
            score_threshold=-10000.,
            keep_top_k=self.post_nms_topN,
            nms_threshold=self.nms_thresh,
            normalized=False if self.pixel_offset else True,
            nms_eta=self.eta)
        label, scores, proposal = paddle.tensor.split(
            out, axis=1, num_or_sections=[1, 1, 4])
        return scores, proposal

    def forward(self):
        anchors = self.input('Anchors', 0)
        bboxdeltas = self.input('BboxDeltas', 0)
        if self.type == 'generate_proposals_v2':
            iminfo = self.input('ImShape', 0)
        else:
            iminfo = self.input('ImInfo', 0)
        scores = self.input('Scores', 0)
        variances = self.input('Variances', 0)

        bboxdeltas = paddle.transpose(bboxdeltas, perm=[0, 2, 3, 1])
        bboxdeltas = paddle.reshape(bboxdeltas, [-1, 4])
        scores = paddle.transpose(scores, perm=[0, 2, 3, 1])
        scores = paddle.reshape(scores, [-1, 1])
        anchors = paddle.reshape(anchors, [-1, 4])
        variances = paddle.reshape(variances, [-1, 4])

        new_scores, proposals = self.proposal_for_single_sample(
            anchors, bboxdeltas, iminfo, scores, variances)
        if len(self.node.outputs) == 3:
            rois_num = paddle.shape(new_scores)[0]
            return {
                'RpnRoiProbs': [new_scores],
                'RpnRois': [proposals],
                'RpnRoisNum': [rois_num]
            }
        else:
            return {'RpnRoiProbs': [new_scores], 'RpnRois': [proposals]}


register_custom_paddle_op('generate_proposals', GenerateProposals)
register_custom_paddle_op('generate_proposals_v2', GenerateProposals)
