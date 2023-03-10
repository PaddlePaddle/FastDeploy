import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype


@paddle.jit.not_to_static
def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k,
                   keep_top_k,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Tensor): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image.
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.
    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    Examples:
        .. code-block:: python
            import paddle
            from ppdet.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = ops.multiclass_nms(bboxes=boxes,
                                            scores=scores,
                                            background_label=0,
                                            score_threshold=0.5,
                                            nms_top_k=400,
                                            nms_threshold=0.3,
                                            keep_top_k=200,
                                            normalized=False,
                                            return_index=True)
    """
    helper = LayerHelper('multiclass_nms3', **locals())

    if in_dygraph_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                 nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                 'normalized', normalized)
        output, index, nms_rois_num = core.ops.multiclass_nms3(bboxes, scores,
                                                               rois_num, *attrs)
        if not return_index:
            index = None
        return output, nms_rois_num, index

    else:
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {'BBoxes': bboxes, 'Scores': scores}
        outputs = {'Out': output, 'Index': index}

        if rois_num is not None:
            inputs['RoisNum'] = rois_num

        if return_rois_num:
            nms_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            outputs['NmsRoisNum'] = nms_rois_num

        helper.append_op(
            type="multiclass_nms3",
            inputs=inputs,
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'nms_top_k': nms_top_k,
                'nms_threshold': nms_threshold,
                'keep_top_k': keep_top_k,
                'nms_eta': nms_eta,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True
        index.stop_gradient = True
        if not return_index:
            index = None
        if not return_rois_num:
            nms_rois_num = None

        return output, nms_rois_num,
