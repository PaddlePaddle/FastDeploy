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

import numpy as np
from paddle2onnx.utils import logging
from paddle2onnx.legacy.constant import dtypes
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.legacy.op_mapper import mapper_helper


@op_mapper(
    ['multiclass_nms', 'multiclass_nms2', 'matrix_nms', 'multiclass_nms3'])
class MultiClassNMS():
    support_opset_verision_range = (10, 16)
    """
    Convert the paddle multiclass_nms to onnx op.
    This op is get the select boxes from origin boxes.
    """

    @classmethod
    def opset_10(cls, graph, node, **kw):
        if node.input_shape("BBoxes", 0)[0] != 1:
            logging.warning(
                "Due to the operator:{}, the converted ONNX model will only supports input[batch_size] == 1.".
                format(node.type))
        scores = node.input('Scores', 0)
        bboxes = node.input('BBoxes', 0)
        num_class = node.input_shape('Scores', 0)[1]
        if len(node.input_shape('Scores', 0)) == 2:
            # inputs: scores & bboxes is lod tensor
            scores = graph.make_node('Transpose', inputs=[scores], perm=[1, 0])
            scores = mapper_helper.unsqueeze_helper(graph, scores, [0])
            if graph.opset_version < 13:
                scores_list = graph.make_node(
                    'Split',
                    inputs=scores,
                    outputs=num_class,
                    axis=1,
                    split=[1] * num_class)
            else:
                split_const = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[1] * num_class)
                scores_list = graph.make_node(
                    "Split",
                    inputs=[scores] + [split_const],
                    outputs=num_class,
                    axis=1)

            bboxes = graph.make_node('Transpose', inputs=bboxes, perm=[1, 0, 2])
            if graph.opset_version < 13:
                bboxes_list = graph.make_node(
                    'Split',
                    inputs=bboxes,
                    outputs=num_class,
                    axis=0,
                    split=[1] * num_class)
            else:
                split_const = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[1] * num_class)
                bboxes_list = graph.make_node(
                    "Split",
                    inputs=[bboxes] + [split_const],
                    outputs=num_class,
                    axis=0)
            bbox_ids = []
            if not isinstance(scores_list, list):
                scores_list = [scores_list]
            if not isinstance(bboxes_list, list):
                bboxes_list = [bboxes_list]
            for i in range(num_class):
                bbox_id = cls.nms(graph,
                                  node,
                                  scores_list[i],
                                  bboxes_list[i],
                                  class_id=i)
                bbox_ids.append(bbox_id)
            bbox_ids = graph.make_node('Concat', inputs=bbox_ids, axis=0)
            const_shape = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[1, -1, 4])
            bboxes = graph.make_node('Reshape', inputs=[bboxes, const_shape])
            cls.keep_top_k(
                graph, node, bbox_ids, scores, bboxes, is_lod_input=True)
        else:
            bbox_ids = cls.nms(graph, node, scores, bboxes)
            cls.keep_top_k(graph, node, bbox_ids, scores, bboxes)

    @classmethod
    def nms(cls, graph, node, scores, bboxes, class_id=None):
        normalized = node.attr('normalized')
        nms_top_k = node.attr('nms_top_k')
        if node.type == 'matrix_nms':
            iou_threshold = 0.5
            logging.warning(
                "Operator:{} is not supported completely, so we use traditional"
                " NMS (nms_theshold={}) to instead it, which introduce some difference.".
                format(node.type, str(iou_threshold)))
        else:
            iou_threshold = node.attr('nms_threshold')
        if nms_top_k == -1:
            nms_top_k = 100000

        #convert the paddle attribute to onnx tensor
        score_threshold = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(node.attr('score_threshold'))])
        iou_threshold = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[float(iou_threshold)])
        nms_top_k = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[np.int64(nms_top_k)])

        # the paddle data format is x1,y1,x2,y2
        kwargs = {'center_point_box': 0}

        if normalized:
            select_bbox_indices = graph.make_node(
                'NonMaxSuppression',
                inputs=[
                    bboxes, scores, nms_top_k, iou_threshold, score_threshold
                ])
        elif not normalized:
            value_one = graph.make_node(
                'Constant', dims=[1], dtype=dtypes.ONNX.FLOAT, value=1.0)
            if graph.opset_version < 13:
                new_bboxes = graph.make_node(
                    'Split',
                    inputs=[bboxes],
                    outputs=4,
                    axis=2,
                    split=[1, 1, 1, 1])
            else:
                split_const = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[1, 1, 1, 1])
                new_bboxes = graph.make_node(
                    "Split", inputs=[bboxes] + [split_const], outputs=4, axis=2)
            new_xmax = graph.make_node('Add', inputs=[new_bboxes[2], value_one])
            new_ymax = graph.make_node('Add', inputs=[new_bboxes[3], value_one])
            new_bboxes = graph.make_node(
                'Concat',
                inputs=[new_bboxes[0], new_bboxes[1], new_xmax, new_ymax],
                axis=2)
            select_bbox_indices = graph.make_node(
                'NonMaxSuppression',
                inputs=[
                    new_bboxes, scores, nms_top_k, iou_threshold,
                    score_threshold
                ])

        if class_id is not None and class_id != 0:
            class_id = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[0, class_id, 0])
            class_id = mapper_helper.unsqueeze_helper(graph, class_id, [0])
            select_bbox_indices = graph.make_node(
                'Add', inputs=[select_bbox_indices, class_id])

        return select_bbox_indices

    @classmethod
    def keep_top_k(cls,
                   graph,
                   node,
                   select_bbox_indices,
                   scores,
                   bboxes,
                   is_lod_input=False):
        # step 1 nodes select the nms class
        # create some const value to use
        background = node.attr('background_label')
        const_values = []
        for value in [0, 1, 2, -1]:
            const_value = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[value])
            const_values.append(const_value)

        # In this code block, we will deocde the raw score data, reshape N * C * M to 1 * N*C*M
        # and the same time, decode the select indices to 1 * D, gather the select_indices
        class_id = graph.make_node(
            'Gather', inputs=[select_bbox_indices, const_values[1]], axis=1)

        squeezed_class_id = mapper_helper.squeeze_helper(graph, class_id, [1])

        bbox_id = graph.make_node(
            'Gather', inputs=[select_bbox_indices, const_values[2]], axis=1)

        if background == 0:
            nonzero = graph.make_node('NonZero', inputs=[squeezed_class_id])
        else:
            filter_cls_id = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT32, value=[background])
            cast = graph.make_node(
                'Cast', inputs=[squeezed_class_id], to=dtypes.ONNX.INT32)
            filter_index = graph.make_node('Sub', inputs=[cast, filter_cls_id])
            nonzero = graph.make_node('NonZero', inputs=[filter_index])

        class_id = graph.make_node('Gather', inputs=[class_id, nonzero], axis=0)
        class_id = graph.make_node(
            'Cast', inputs=[class_id], to=dtypes.ONNX.INT64)

        bbox_id = graph.make_node('Gather', inputs=[bbox_id, nonzero], axis=0)
        bbox_id = graph.make_node(
            'Cast', inputs=[bbox_id], to=dtypes.ONNX.INT64)

        # get the shape of scores
        shape_scores = graph.make_node('Shape', inputs=scores)

        # gather the index: 2 shape of scores
        class_num = graph.make_node(
            'Gather', inputs=[shape_scores, const_values[2]], axis=0)

        # reshape scores N * C * M to (N*C*M) * 1
        scores = graph.make_node('Reshape', inputs=[scores, const_values[-1]])

        # mul class * M
        mul_classnum_boxnum = graph.make_node(
            'Mul', inputs=[class_id, class_num])

        # add class * M * index
        add_class_indices = graph.make_node(
            'Add', inputs=[mul_classnum_boxnum, bbox_id])

        # Squeeze the indices to 1 dim
        score_indices = mapper_helper.squeeze_helper(graph, add_class_indices,
                                                     [0, 2])

        # gather the data from flatten scores
        scores = graph.make_node(
            'Gather', inputs=[scores, score_indices], axis=0)

        keep_top_k = node.attr('keep_top_k')
        keep_top_k = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT64,
            dims=[1, 1],
            value=[node.attr('keep_top_k')])

        # get min(topK, num_select)
        shape_select_num = graph.make_node('Shape', inputs=[scores])
        const_zero = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[0])
        gather_select_num = graph.make_node(
            'Gather', inputs=[shape_select_num, const_zero], axis=0)
        unsqueeze_select_num = mapper_helper.unsqueeze_helper(
            graph, gather_select_num, [0])

        concat_topK_select_num = graph.make_node(
            'Concat', inputs=[unsqueeze_select_num, keep_top_k], axis=0)
        cast_concat_topK_select_num = graph.make_node(
            'Cast', inputs=[concat_topK_select_num], to=6)
        keep_top_k = graph.make_node(
            'ReduceMin', inputs=[cast_concat_topK_select_num], keepdims=0)
        # unsqueeze the indices to 1D tensor
        keep_top_k = mapper_helper.unsqueeze_helper(graph, keep_top_k, [0])

        # cast the indices to INT64
        keep_top_k = graph.make_node('Cast', inputs=[keep_top_k], to=7)

        # select topk scores  indices
        keep_topk_scores, keep_topk_indices = graph.make_node(
            'TopK', inputs=[scores, keep_top_k], outputs=2)

        # gather topk label, scores, boxes
        gather_topk_scores = graph.make_node(
            'Gather', inputs=[scores, keep_topk_indices], axis=0)

        gather_topk_class = graph.make_node(
            'Gather', inputs=[class_id, keep_topk_indices], axis=1)

        # gather the boxes need to gather the boxes id, then get boxes
        if is_lod_input:
            gather_topk_boxes_id = graph.make_node(
                'Gather', [add_class_indices, keep_topk_indices], axis=1)
        else:
            gather_topk_boxes_id = graph.make_node(
                'Gather', [bbox_id, keep_topk_indices], axis=1)

        # squeeze the gather_topk_boxes_id to 1 dim
        squeeze_topk_boxes_id = mapper_helper.squeeze_helper(
            graph, gather_topk_boxes_id, [0, 2])

        gather_select_boxes = graph.make_node(
            'Gather', inputs=[bboxes, squeeze_topk_boxes_id], axis=1)

        # concat the final result
        # before concat need to cast the class to float
        cast_topk_class = graph.make_node(
            'Cast', inputs=[gather_topk_class], to=1)

        unsqueeze_topk_scores = mapper_helper.unsqueeze_helper(
            graph, gather_topk_scores, [0, 2])

        inputs_concat_final_results = [
            cast_topk_class, unsqueeze_topk_scores, gather_select_boxes
        ]

        sort_by_socre_results = graph.make_node(
            'Concat', inputs=inputs_concat_final_results, axis=2)

        # sort by class_id
        squeeze_cast_topk_class = mapper_helper.squeeze_helper(
            graph, cast_topk_class, [0, 2])

        neg_squeeze_cast_topk_class = graph.make_node(
            'Neg', inputs=[squeeze_cast_topk_class])

        data, indices = graph.make_node(
            'TopK', inputs=[neg_squeeze_cast_topk_class, keep_top_k], outputs=2)

        concat_final_results = graph.make_node(
            'Gather', inputs=[sort_by_socre_results, indices], axis=1)

        concat_final_results = mapper_helper.squeeze_helper(
            graph, concat_final_results, [0], node.output('Out'))

        if node.type in ['multiclass_nms2', 'matrix_nms', 'multiclass_nms3']:
            final_indices = mapper_helper.squeeze_helper(graph, bbox_id, [0],
                                                         node.output('Index'))
            if node.type in ['matrix_nms', 'multiclass_nms3']:
                select_bboxes_shape = graph.make_node('Shape', inputs=[indices])
                select_bboxes_shape1 = graph.make_node(
                    'Cast', inputs=[select_bboxes_shape], to=dtypes.ONNX.INT32)
                indices = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[0])
                rois_num = None
                if 'NmsRoisNum' in node.outputs:
                    rois_num = node.output('NmsRoisNum')
                elif 'RoisNum' in node.outputs:
                    rois_num = node.output('RoisNum')
                if rois_num is not None:
                    graph.make_node(
                        "Gather",
                        inputs=[select_bboxes_shape1, indices],
                        outputs=rois_num)
